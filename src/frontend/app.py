"""Streamlit 主入口：streamlit run src/frontend/app.py
用户在页面上做的每件事（提问、点批准、传文件），
都不直接执行—— 只写进队列，然后让页面重跑一次，由下一次运行在锁内执行 。这是理解全文件的总钥匙

页面结构：
- 侧边栏：业务模块选择（会话固定）、HITL 开关、附件上传、新建会话；
- 主区：聊天历史（助手消息内联溯源面板）、HITL 复核面板、输入框。

并发模型（重要）：
用户提交（提问 / HITL 决策）只写入 pending_submission 队列后 rerun，
由后续脚本运行在 per-thread 锁内消费。消费遵循"抢锁即认领"：拿到锁后
先把提交移出队列再执行阻塞 invoke——否则旧运行在 invoke 期间被重连
stop 时，锁会正常释放而队列未清，新运行会把同一提交串行重跑一遍
（实测 LangGraph 对中断中的 thread 再发普通 inputs 会从 START 整条
重放）。配合 per-thread 锁，长任务期间重复点击或 WebSocket 重连产生的
多次脚本运行，同一 thread_id 的 invoke/resume 既不会并发也不会重复
（MemorySaver 不支持同 thread 并发，否则 checkpoint 会交错写入）。
拿不到锁的运行渲染等待态并稍后自动重跑；每次空闲运行还会以 graph
checkpoint 为权威状态校准 UI，兜底恢复断连运行。
"""

import time

import streamlit as st
from langgraph.types import Command

from src.core.logger import get_logger
from src.frontend.components.attachment import render_attachment_uploader
from src.frontend.components.chat import render_chat
from src.frontend.components.hitl import render_hitl_panel
from src.frontend.components.mode_selector import render_mode_selector
from src.frontend.session import (
    MODE_LABELS,
    add_message,
    get_app,
    get_graph_config,
    get_graph_lock,
    get_messages,
    get_thread_id,
    init_session_state,
    new_session,
)
from src.graph.nodes import CHAT_ANSWER, GENERATE_DRAFT_ANSWER

logger = get_logger(__name__)

st.set_page_config(page_title="刑事案件智能分析", page_icon="⚖️", layout="wide")

# 允许流式渲染 token 的节点：只有面向用户产出正文的两个回答节点。
# 意图识别 / 工具循环 / 业务分析 / 引用校验等节点的 LLM token 不外泄到界面。
_STREAM_NODES = frozenset({CHAT_ANSWER, GENERATE_DRAFT_ANSWER})


def _sink_result(values: dict, interrupt_value: dict | None) -> None:
    """
    收口一次图执行（invoke / resume / 流式）的最终结果

    有中断：暂存 pending_interrupt，等待人工决策 resume；
    无中断：写入助手消息与 final_result，聊天历史完成一轮。

    values 为 checkpoint 当前 state values；
    interrupt_value 为中断载荷，None 表示已到 END。
    """
    if interrupt_value is not None:
        st.session_state.pending_interrupt = interrupt_value
        return

    st.session_state.pending_interrupt = None
    result = values.get("final_result") or {}
    add_message("assistant", values.get("final_answer", "（无回答）"), result)
    st.session_state.last_result = result


def _stream_graph(
    payload: dict | Command, running_label: str, done_label: str
) -> tuple[dict, dict | None]:
    """
    以 messages 流模式跑图，边收回答 token 边渲染，跑完后读 checkpoint 收口

    LangGraph 的 messages 模式会给节点内的 LLM 调用挂流式回调，
    ChatOpenAI 检测到该回调即走流式 API，因此节点里的 .invoke() 无需改造。
    思考模型的推理 token 走 additional_kwargs.reasoning_content，
    chunk.content 为空，天然不会被渲染。

    payload 首次调用传 inputs dict，resume 传 Command(resume=...)；
    running_label / done_label 分别为执行中与完成状态文案。
    返回 (最终 state values, 中断载荷 | None)。
    """
    cfg = get_graph_config()
    with st.status(running_label, expanded=True) as status:
        box = st.empty()
        # 按节点各自累积，理论上一轮只会有一个回答节点产出
        buffers: dict[str, list[str]] = {}
        current_node: str | None = None
        for chunk, metadata in get_app().stream(
            payload, cfg, stream_mode="messages"
        ):
            node = metadata.get("langgraph_node")
            if node not in _STREAM_NODES:
                continue
            if node != current_node:
                current_node = node
                buffers[node] = []
            # 多模态 content 可能是 list，本场景只渲染纯文本
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if piece:
                buffers[node].append(piece)
                box.markdown("".join(buffers[node]) + " ▌")
        status.update(label=done_label, state="complete", expanded=False)

    # 流结束即最终权威状态：中断中取中断载荷，否则到 END 取 values
    snapshot = get_app().get_state(cfg)
    interrupt_value = next(
        (task.interrupts[0].value for task in snapshot.tasks if task.interrupts),
        None,
    )
    return snapshot.values or {}, interrupt_value


def _reconcile_from_checkpoint() -> None:
    """
    以 graph checkpoint 为权威来源校准 UI 状态（断连恢复兜底）

    长任务期间浏览器可能断开旧脚本运行、重连后发起新运行：旧运行对
    session_state 的写入不一定可见。这里空闲时（无待消费提交）直接读
    checkpoint：中断中就恢复复核面板，已到 END 就按内容去重补一条
    助手消息，避免界面永远停在加载态。
    """
    snapshot = get_app().get_state(get_graph_config())

    # 图仍在中断中：恢复中断载荷
    pending = None
    for task in snapshot.tasks:
        if task.interrupts:
            pending = task.interrupts[0].value
            break
    if pending is not None:
        st.session_state.pending_interrupt = pending
        return

    # 图未到 END 且无中断（理论上 busy 锁会挡住，竞态下忽略，等下轮校准）
    if snapshot.next:
        return

    # 已到 END：补全最终回复（按正文去重，避免与正常路径重复追加）
    st.session_state.pending_interrupt = None
    values = snapshot.values or {}
    answer = values.get("final_answer")
    if not answer:
        return
    messages = get_messages()
    if messages and messages[-1]["role"] == "assistant" \
            and messages[-1]["content"] == answer:
        return
    add_message("assistant", answer, values.get("final_result"))
    st.session_state.last_result = values.get("final_result")


def _consume_submission(lock) -> bool:
    """
    在锁内消费 pending_submission

    认领时机至关重要：抢锁成功后必须【先】把提交移出队列、【再】执行
    invoke。invoke 是 2~3 分钟的阻塞长任务，期间旧脚本运行随时可能被
    WebSocket 重连 stop；若 invoke 返回后才清队列，被 stop 的旧运行会
    在 finally 里正常释放锁，而队列仍是同一条提交——新运行拿到锁后会
    对同一 thread_id 再 invoke 一遍。LangGraph 对停在 interrupt 的
    thread 收到普通 inputs 时不报错，而是从 START 整条重跑（实测节点
    全部二次执行、付费调用翻倍、初稿被覆盖）。先认领则新运行只会读到
    空队列，由 checkpoint 校准兜底，图最多执行一次。

    lock 为当前会话的图调用锁；返回 True 表示本轮应 rerun，
    False 表示未执行（应渲染等待态后重跑）。
    """
    if not lock.acquire(timeout=0.5):
        # 另一个脚本运行（通常是断连的旧运行）仍在执行同一图
        return False

    # 抢锁即认领：立即移出队列，此后本运行在任何时刻被 stop 都不会重复执行
    submission = st.session_state.pending_submission
    st.session_state.pending_submission = None
    if not submission:
        lock.release()
        return False

    try:
        if submission["type"] == "invoke":
            # 流式跑图：回答 token 逐块渲染，跑完从 checkpoint 收口
            values, interrupt_value = _stream_graph(
                submission["state"], "正在分析案情…", "分析完成"
            )
        elif submission["type"] == "resume":
            # 中断指纹校验：若 checkpoint 当前中断与提交时不一致，
            # 说明该中断已被另一个运行消费（断连恢复竞态），丢弃本次
            # resume，改由 checkpoint 权威校准，绝不重复决策。
            snapshot = get_app().get_state(get_graph_config())
            current = next(
                (
                    task.interrupts[0].value
                    for task in snapshot.tasks
                    if task.interrupts
                ),
                None,
            )
            if current is None or current.get("draft") != submission.get(
                "interrupt_draft"
            ):
                logger.warning("[frontend] 中断已被处理，丢弃重复的 resume")
                _reconcile_from_checkpoint()
                return True

            # 批准走引用校验（无流式正文）；驳回重写会逐块产出新报告
            values, interrupt_value = _stream_graph(
                Command(resume=submission["decision"]),
                "正在按您的决策继续处理…",
                "处理完成",
            )
        else:
            logger.warning(f"[frontend] 未知提交类型，已丢弃：{submission!r}")
            return True

        _sink_result(values, interrupt_value)
    except Exception as e:  # noqa: BLE001
        logger.exception("[frontend] 图调用失败")
        st.session_state.pending_interrupt = None
        st.error(f"执行失败：{type(e).__name__}: {e}，请点「新建会话」后重试。")
    finally:
        lock.release()
    return True


def main() -> None:
    init_session_state()
    get_app()  # 预热编译，让配置错误在首条消息前暴露
    lock = get_graph_lock(get_thread_id())

    # busy 以锁的实际持有状态为准：长任务期间发生断连重跑时，会话内
    # 标志位可能残留，锁不会。
    busy = lock.locked()

    # ========= 1) 侧边栏：模块 / HITL / 附件 =========
    attach_files: list[dict] = []
    with st.sidebar:
        st.title("⚖️ 刑事案件智能分析")
        render_mode_selector()
        st.toggle("人工复核（HITL）", key="hitl_toggle", disabled=busy)
        attach_files = render_attachment_uploader()
        if st.button("新建会话", use_container_width=True, disabled=busy):
            new_session(st.session_state.input_mode)
            st.rerun()
        st.caption(
            f"会话 {get_thread_id()}｜"
            f"模块「{MODE_LABELS[st.session_state.input_mode]}」"
        )

    # ========= 2) 空闲：以 checkpoint 权威状态校准 UI（断连恢复兜底） =========
    # 必须在 render_chat 之前：校准可能补一条助手消息。
    if st.session_state.pending_submission is None:
        _reconcile_from_checkpoint()

    # ========= 3) 主区：聊天历史 =========
    # 先渲染历史再消费提交：用户刚发送的问题上一轮已入 session_state，
    # 这里立即上屏；后续 st.status / 流式气泡自然排在历史最下方，
    # 而不是悬在页面顶部。
    st.title("刑事案件智能分析助手")
    render_chat(get_messages())

    # ========= 4) 待消费提交（仍在 HITL return 分支之前，避免队列饿死） =========
    # 上一轮运行把用户提交放入 pending_submission；本轮在锁内真正调用图。
    # 必须排在 HITL 面板之前——该面板会提前 return，放后面队列永远得不到
    # 处理（死锁）。整个消费过程放进助手气泡，加载态与流式正文落在对话底部。
    if st.session_state.pending_submission is not None:
        with st.chat_message("assistant"):
            consumed = _consume_submission(lock)
        if consumed:
            st.rerun()
        # 锁被另一运行持有（断连旧运行仍在执行）：只渲染等待态，稍后重跑
        with st.spinner("上一轮分析仍在执行，等待其完成（请不要重复提交）…"):
            time.sleep(3)
        st.rerun()

    # ========= 5) HITL：有中断时渲染复核面板（提交仅入队） =========
    if st.session_state.pending_interrupt:
        decision = render_hitl_panel(
            st.session_state.pending_interrupt, disabled=busy
        )
        if decision:
            # 带上中断时的初稿指纹，消费前校验中断是否仍是同一个
            st.session_state.pending_submission = {
                "type": "resume",
                "decision": decision,
                "interrupt_draft": st.session_state.pending_interrupt.get(
                    "draft", ""
                ),
            }
            st.rerun()
        return

    # ========= 6) 正常输入（提交仅入队，下一轮在步骤 4 消费） =========
    query = st.chat_input(
        "输入案情或法律问题（支持多轮追问）", disabled=busy
    )
    if query:
        add_message("user", query)
        hitl_on = bool(st.session_state.hitl_toggle)
        logger.info(
            f"[frontend] 入队 invoke：mode={st.session_state.input_mode} "
            f"HITL={hitl_on} 附件 {len(attach_files)} 个"
        )
        st.session_state.pending_submission = {
            "type": "invoke",
            "state": {
                "fixed_input_mode": st.session_state.input_mode,
                "query": query,
                "attach_files": attach_files,
                "human_review_enable": hitl_on,
            },
        }
        st.rerun()


if __name__ == "__main__":
    main()
