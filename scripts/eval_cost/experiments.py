"""两个 token 节省做法的实验逻辑：E1/E2 均为实测跑图。

全程 patch 掉元典工具（未配置 token 时本就返回空列表，这里双保险），
实验只产生 DashScope LLM 费用。

实验设计与省 token 机制见 scripts/eval_cost/run.py 顶部设计说明。
"""

import contextlib
import json
import sys
import uuid
from pathlib import Path

# 允许 python -m scripts.eval_cost.run 与直接运行两种方式
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval_cost.token_handler import TokenUsageHandler

# ============ 固定实验样本 ============
# simple 类：刑事相关但不应触发报告链路，不含 FORCED_REPORT_KEYWORDS
SIMPLE_QUERY: str = "正当防卫的成立条件是什么？"


def _load_report_case() -> tuple[str, str]:
    """
    从 data_test.json 取首条盗窃案，构造 report 实验输入。

    返回 (发给图的 query, mock 意图节点的 fact_extract)。
    """
    path = _PROJECT_ROOT / "data" / "cases" / "data_test.json"
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    fact = first["fact"]
    fact_extract = "、".join(first["meta"]["accusation"]) + "案，退赔获谅解"
    return f"请针对以下案情出具法律评估：\n{fact}", fact_extract


# ============ monkeypatch 上下文 ============
@contextlib.contextmanager
def _patches(
    *,
    force_report: bool = False,
    fact_extract: str | None = None,
    force_thinking: bool = False,
    trace_handler: TokenUsageHandler | None = None,
    block_yuandian: bool = True,
):
    """
    临时改写 nodes 模块的全局绑定，退出时全部还原。

    force_report：mock 意图识别节点，强制 report + 固定 fact_extract；
    force_thinking：把节点内所有 get_llm 调用强制 thinking=True（E2 对照）；
    trace_handler：节点外直调时经 model_copy 注入的 handler（E2）；
    block_yuandian：屏蔽元典工具，保证零积分、变量可控。
    """
    import src.graph.nodes as nodes

    saved: dict[str, object] = {}

    def save(name: str, value: object) -> None:
        saved[name] = getattr(nodes, name)
        setattr(nodes, name, value)

    if block_yuandian:
        save("build_yuandian_tools", lambda: [])

    if force_report:
        decision = nodes._IntentDecision(
            intent="criminal",
            task_type="report",
            fact_extract=fact_extract or "案情摘要",
        )
        save("_recognize_with_llm", lambda q: decision)

    if force_thinking or trace_handler is not None:
        original = nodes.get_llm

        def traced_get_llm(thinking: bool = False, max_tokens: int | None = None):
            # force_thinking 时把 thinking=False 的节点也开思考；否则保留原值
            use_thinking = True if force_thinking else thinking
            llm = original(use_thinking, max_tokens=max_tokens)
            # 节点外直调没有 config callbacks，拷贝实例挂 handler；
            # 跑图时 config 已挂 handler，trace_handler 传 None 即不注入
            if trace_handler is not None:
                llm = llm.model_copy(update={"callbacks": [trace_handler]})
            return llm

        save("get_llm", traced_get_llm)

    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(nodes, name, value)


# ============ 跑图封装 ============
def run_graph(
    query: str,
    handler: TokenUsageHandler,
    *,
    mode: str = "consult",
    force_report: bool = False,
    fact_extract: str | None = None,
) -> dict:
    """
    用全新编译的图（独立 MemorySaver）跑一轮，HITL 关闭；返回最终 state

    每次实验独立编译图，避免不同组之间 checkpoint / 节点状态互串。
    """
    from langgraph.types import Command

    from src.graph.builder import build_graph

    app = build_graph()
    config = {
        "configurable": {"thread_id": f"cost-{uuid.uuid4().hex[:8]}"},
        "callbacks": [handler],
    }
    initial_state = {
        "fixed_input_mode": mode,
        "query": query,
        "attach_files": [],
        "human_review_enable": False,
    }

    with _patches(force_report=force_report, fact_extract=fact_extract):
        state = app.invoke(initial_state, config)
        # 双保险：若仍进入中断（配置默认开 HITL），自动批准收尾
        while "__interrupt__" in state:
            state = app.invoke(
                Command(resume={"action": "approve", "comment": "", "edited_content": ""}),
                config,
            )
    return state


# ============ E1 链路分流 ============
def experiment_e1(handler: TokenUsageHandler) -> dict:
    """
    A：simple query 自然路由（预期走 simple_chat 轻量链）
    B：同一规模的案情输入强制 report（走完整报告链）
    两组的 token 差即"两级分流"省下的量。
    """
    # ---- A 自然路由 ----
    handler.reset()
    state_a = run_graph(SIMPLE_QUERY, handler)
    snap_a = handler.snapshot()

    # ---- B 强制 report ----
    report_query, fact = _load_report_case()
    handler.reset()
    state_b = run_graph(
        report_query, handler, force_report=True, fact_extract=fact
    )
    snap_b = handler.snapshot()

    return {
        "simple": {
            "query": SIMPLE_QUERY,
            "routed_task_type": state_a.get("task_type"),
            "usage": snap_a,
            "final_state": state_a,
        },
        "report": {
            "query": report_query,
            "routed_task_type": state_b.get("task_type"),
            "usage": snap_b,
            "final_state": state_b,
        },
    }


# ============ E2 thinking 精准配置 ============
def experiment_e2(handler: TokenUsageHandler, report_state: dict) -> dict:
    """
    对两个 thinking=False 的节点做单节点 A/B 重放（不跑全图）：
      - intent_recognize（_recognize_with_llm 直调）
      - citation_check（用 E1 report 组跑完的 state 重放）
    A=现状关思考，B=强制开思考，output 差即白付的推理费。
    """
    import src.graph.nodes as nodes

    def _replay_intent() -> dict:
        nodes._recognize_with_llm(SIMPLE_QUERY)
        return handler.snapshot()

    def _replay_citation(state: dict) -> dict:
        nodes.citation_check_node(state)
        return handler.snapshot()

    # A 现状（thinking=False），handler 经 model_copy 注入
    with _patches(trace_handler=handler):
        handler.reset("intent-thinking=False")
        intent_off = _replay_intent()
        handler.reset("citation-thinking=False")
        citation_off = _replay_citation(report_state)

    # B 强制 thinking=True
    with _patches(force_thinking=True, trace_handler=handler):
        handler.reset("intent-thinking=True")
        intent_on = _replay_intent()
        handler.reset("citation-thinking=True")
        citation_on = _replay_citation(report_state)

    return {
        "intent": {"thinking_off": intent_off, "thinking_on": intent_on},
        "citation": {
            "thinking_off": citation_off,
            "thinking_on": citation_on,
        },
    }
