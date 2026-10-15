"""条件边路由函数。"""

from src.core.config import get_settings
from src.core.logger import get_logger
from src.graph.nodes import (
    ACCUSE_ANALYSIS,
    CITATION_CHECK,
    CONSULT,
    DEFEND_ANALYSIS,
    DOC_REVIEW,
    HITL_REVIEW,
)
from src.graph.state import AgentState, InputMode

logger = get_logger(__name__)

# fixed_input_mode → report 业务节点名
_BUSINESS_NODE_BY_MODE: dict[InputMode, str] = {
    "accuse_analysis": ACCUSE_ANALYSIS,
    "defend_analysis": DEFEND_ANALYSIS,
    "doc_review": DOC_REVIEW,
    "consult": CONSULT,
}

# 会话模式缺失或非法时的兜底去向（综合咨询最中立，不预设立场）
_FALLBACK_NODE = CONSULT


def route_after_full_retrieve(state: AgentState) -> str:
    """
    全量检索后的业务分发

    读取会话固定字段 ``fixed_input_mode``（节点链路不改写它），分发到
    控告 / 辩护 / 文书审查 / 综合咨询四个业务节点之一。

    该字段理论上在新建会话时必写入；缺失或为非法值时兜底到综合咨询，
    保证图不会因为状态异常而中断。
    """
    mode = state.get("fixed_input_mode")
    target = _BUSINESS_NODE_BY_MODE.get(mode)  # type: ignore[arg-type]

    if target is None:
        logger.warning(
            f"[route_after_full_retrieve] fixed_input_mode={mode!r} 缺失或非法，"
            f"兜底路由到 {_FALLBACK_NODE}"
        )
        return _FALLBACK_NODE

    logger.info(f"[route_after_full_retrieve] mode={mode} → {target}")
    return target


def route_after_report_draft(state: AgentState) -> str:
    """
    报告初稿后的 HITL 开关分流

    优先读会话 state 中的 ``human_review_enable``（前端建会话时可显式指定）；
    字段缺失时回退 configs/config.yaml 的全局 ``hitl.enabled``。
    """
    enabled = state.get("human_review_enable")
    if enabled is None:
        enabled = get_settings().hitl.enabled

    target = HITL_REVIEW if enabled else CITATION_CHECK
    logger.info(f"[route_after_report_draft] HITL enabled={enabled} → {target}")
    return target
