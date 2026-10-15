"""StateGraph 构建：注册全部节点与边，编译带 Checkpointer 的图。

入口三分支沿用节点内 Command(goto)（V3.2 设计既定）；report 链路内部的
分发与 HITL 开关用 builder 层条件边表达，保证主流程在图结构上可见。
"""

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.core.logger import get_logger
from src.graph.edges import route_after_full_retrieve, route_after_report_draft
from src.graph.nodes import (
    ACCUSE_ANALYSIS,
    APPLY_HUMAN_DECISION,
    CHAT_ANSWER,
    CITATION_CHECK,
    CONSULT,
    DEFEND_ANALYSIS,
    DOC_REVIEW,
    FULL_RETRIEVE,
    GENERATE_DRAFT_ANSWER,
    HITL_REVIEW,
    INTENT_RECOGNIZE,
    LIGHT_RETRIEVE,
    REJECT_NON_CRIMINAL,
    RESULT_COMBINE,
    accuse_analysis_node,
    apply_human_decision_node,
    chat_answer_node,
    citation_check_node,
    consult_node,
    defend_analysis_node,
    doc_review_node,
    full_retrieve_node,
    generate_draft_answer_node,
    hitl_review_node,
    intent_recognize_node,
    light_retrieve_node,
    reject_non_criminal_node,
    result_combine_node,
)
from src.graph.state import AgentState

logger = get_logger(__name__)

# report 全量检索后可分发到的四个业务节点（条件边 path_map）
_BUSINESS_NODES: tuple[str, ...] = (
    ACCUSE_ANALYSIS,
    DEFEND_ANALYSIS,
    DOC_REVIEW,
    CONSULT,
)


def build_state_graph() -> StateGraph:
    """
    构造未编译的 StateGraph（注册全部节点与边）

    单独拆出便于对节点表、静态边、条件边做结构断言；编译见 build_graph。
    注意：入口三分支是节点内 Command(goto) 动态跳转，编译后 get_graph()
    的静态遍历看不到下游边，这是可视化限制，不影响运行时路由。
    """
    graph = StateGraph(AgentState)

    # ========= 注册节点（14 个） =========
    graph.add_node(INTENT_RECOGNIZE, intent_recognize_node)
    graph.add_node(REJECT_NON_CRIMINAL, reject_non_criminal_node)
    graph.add_node(LIGHT_RETRIEVE, light_retrieve_node)
    graph.add_node(CHAT_ANSWER, chat_answer_node)
    graph.add_node(FULL_RETRIEVE, full_retrieve_node)
    graph.add_node(ACCUSE_ANALYSIS, accuse_analysis_node)
    graph.add_node(DEFEND_ANALYSIS, defend_analysis_node)
    graph.add_node(DOC_REVIEW, doc_review_node)
    graph.add_node(CONSULT, consult_node)
    graph.add_node(GENERATE_DRAFT_ANSWER, generate_draft_answer_node)
    graph.add_node(HITL_REVIEW, hitl_review_node)
    graph.add_node(APPLY_HUMAN_DECISION, apply_human_decision_node)
    graph.add_node(CITATION_CHECK, citation_check_node)
    graph.add_node(RESULT_COMBINE, result_combine_node)

    # ========= 入口 =========
    # 入口节点没有静态出边：三种去向全部由节点内 Command(goto) 决定
    graph.add_edge(START, INTENT_RECOGNIZE)

    # ========= 分支 A：非刑事 =========
    graph.add_edge(REJECT_NON_CRIMINAL, END)

    # ========= 分支 B：simple_chat 轻量链路 =========
    graph.add_edge(LIGHT_RETRIEVE, CHAT_ANSWER)
    graph.add_edge(CHAT_ANSWER, CITATION_CHECK)

    # ========= 分支 C：report 报告链路 =========
    graph.add_conditional_edges(
        FULL_RETRIEVE,
        route_after_full_retrieve,
        {node_name: node_name for node_name in _BUSINESS_NODES},
    )
    for business_node in _BUSINESS_NODES:
        graph.add_edge(business_node, GENERATE_DRAFT_ANSWER)

    graph.add_conditional_edges(
        GENERATE_DRAFT_ANSWER,
        route_after_report_draft,
        {
            HITL_REVIEW: HITL_REVIEW,
            CITATION_CHECK: CITATION_CHECK,
        },
    )
    graph.add_edge(HITL_REVIEW, APPLY_HUMAN_DECISION)
    # apply 节点无静态出边：批准/修改/驳回三个动作全部由节点返回
    # Command(goto=...) 互斥路由。若加静态边到 citation_check，驳回时会与
    # Command(goto=draft) 同时生效产生扇出：一边回炉重写、一边提前走校验收尾

    # ========= 共同出口：引用校验 → 结果组合 → END =========
    graph.add_edge(CITATION_CHECK, RESULT_COMBINE)
    graph.add_edge(RESULT_COMBINE, END)

    return graph


def build_graph(checkpointer=None):
    """编译刑事案件 Agent 工作流。

    不传 checkpointer 时使用内存 MemorySaver（演示项目足够；前端 Streamlit
    需以模块级单例持有，见 get_graph）。
    """
    app = build_state_graph().compile(checkpointer=checkpointer or MemorySaver())
    logger.info("[build_graph] 编译完成，Checkpointer 已启用")
    return app


@lru_cache(maxsize=1)
def get_graph():
    """
    获取进程级单例图（前端 / 后端服务复用同一个 Checkpointer）

    用模块级单例而不是在请求里现编译：MemorySaver 的会话状态保存在
    编译产物内部，每次重新编译都会丢失全部中断与历史。
    """
    return build_graph()


if __name__ == "__main__":
    app = get_graph()
    print("节点：", sorted(app.get_graph().nodes))
    print("可选入口：", app.get_graph().config_specs)
