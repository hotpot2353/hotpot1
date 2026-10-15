"""Streamlit 会话状态管理：thread_id、业务模块、聊天历史、HITL 暂停态。

LangGraph 图状态由 MemorySaver（进程内图单例）持久化，前端负责：
1. 生成并稳定持有 thread_id，同一会话的所有 invoke / resume 复用同一 config；
2. 维护展示用聊天历史与附件解析缓存等 UI 状态；
3. 持有 HITL 中断载荷，跨多次脚本重跑完成 resume 循环。
"""

import threading
import uuid

import streamlit as st

from src.graph.builder import get_graph

# per-thread 图调用锁：Streamlit 长任务期间可能因重复点击 / WebSocket
# 重连产生并发脚本运行，对同一 thread_id 并发 invoke/resume 会使
# MemorySaver 的 checkpoint 交错写入（节点重复执行、状态错乱），
# 必须在进程内按会话串行化所有图调用。

# thread_id → 该会话专属锁；_GRAPH_LOCKS_GUARD 串行化字典本身的创建与读取
_GRAPH_LOCKS: dict[str, threading.Lock] = {}
_GRAPH_LOCKS_GUARD = threading.Lock()


def get_graph_lock(thread_id: str | None = None) -> threading.Lock:
    """按 thread_id 获取会话专属的图调用锁，不存在则创建（进程内全局复用）。"""
    thread_id = thread_id or get_thread_id()
    with _GRAPH_LOCKS_GUARD:
        lock = _GRAPH_LOCKS.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _GRAPH_LOCKS[thread_id] = lock
        return lock

# 业务模块中文标签（与 state.InputMode 一一对应）
MODE_LABELS: dict[str, str] = {
    "accuse_analysis": "刑事控告",
    "defend_analysis": "刑事辩护",
    "doc_review": "文书审查",
    "consult": "综合咨询",
}

DEFAULT_MODE = "consult"

# 聊天消息结构：{"role": user/assistant, "content": str,
#               "result": final_result dict | None}
_KEY_MESSAGES = "messages"


def init_session_state() -> None:
    """
    初始化 / 恢复 session_state（幂等，每次脚本重跑都调用）

    首次运行创建默认会话；后续重跑只补缺失键，不覆盖已有值。
    """
    defaults = {
        "thread_id": f"ui-{uuid.uuid4().hex[:8]}",
        "input_mode": DEFAULT_MODE,
        "hitl_toggle": True,  # widget key，下轮 invoke 生效
        _KEY_MESSAGES: [],
        "pending_interrupt": None,  # HITL 中断载荷（dict | None）
        "last_result": None,  # 最近一轮 final_result（dict | None）
        "attach_cache": {},  # {(filename, size): content} 附件解析缓存
        # 待执行的图调用提交（{type: invoke/resume, ...} | None）：
        # 用户提交只入队，由后续运行在锁内消费，防止重复触发并发执行
        "pending_submission": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_session(mode: str) -> None:
    """
    新建会话：换 thread_id 并清空轮次内状态

    fixed_input_mode 会话内不可切换，切换模块必须走这里。
    """
    st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    st.session_state.input_mode = mode
    st.session_state[_KEY_MESSAGES] = []
    st.session_state.pending_interrupt = None
    st.session_state.last_result = None
    st.session_state.attach_cache = {}
    st.session_state.pending_submission = None


def get_thread_id() -> str:
    """获取当前会话 thread_id（init_session_state 已保证存在）。"""
    return st.session_state.thread_id


def get_graph_config() -> dict:
    """构造 invoke / resume 共用的 LangGraph config。"""
    return {"configurable": {"thread_id": get_thread_id()}}


def add_message(role: str, content: str, result: dict | None = None) -> None:
    """追加一条聊天历史。"""
    st.session_state[_KEY_MESSAGES].append(
        {"role": role, "content": content, "result": result}
    )


def get_messages() -> list[dict]:
    """读取全部聊天历史。"""
    return st.session_state[_KEY_MESSAGES]


@st.cache_resource
def get_app():
    """
    编译图并做进程级缓存（跨脚本重跑共享同一 MemorySaver）

    必须缓存：Streamlit 每次交互都重跑脚本，若每次重新编译，
    MemorySaver 里的中断与历史会全部丢失，HITL resume 将失效。
    """
    return get_graph()
