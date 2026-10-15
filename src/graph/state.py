"""AgentState：LangGraph 共享状态定义（V3.2）。

字段按生命周期分四组：
1. 会话固定字段：新建会话时写入，由 Checkpointer 持久保存，节点只读不写；
2. 本轮输入：每轮 invoke 时传入；
3. 本轮中间产物：各节点写入、下一跳读取，每轮随 invoke 刷新；
4. 输出字段：result_combine_node 组合后供前端展示。
"""
from typing import Literal, TypedDict

# 会话固定业务模块，四选一
InputMode = Literal[
    "accuse_analysis",  # 刑事控告
    "defend_analysis",  # 刑事辩护
    "doc_review",  # 文书审查
    "consult",  # 综合咨询
]

TaskType = Literal["simple_chat", "report"]

IntentType = Literal["criminal", "non_criminal"]

# HITL 人工动作 [通过初稿，修改后通过，打回重做]
HumanAction = Literal["approve", "edit", "reject"]


class AttachFile(TypedDict, total=False):
    """用户上传的案卷附件（前端解析后传入）。"""

    filename: str
    content: str


class AgentState(TypedDict, total=False):
    """
    刑事案件智能分析 Agent 的完整状态。

    每轮 Graph 从入口 intent_recognize_node 重新执行；
    会话级字段由 Checkpointer 持久化、跨轮恢复。
    """

    # ========= 会话固定字段（只读） =========
    # 前端新建会话点选模块时写入；切换模块必须新建会话。
    fixed_input_mode: InputMode

    # ========= 用户本轮输入 =========
    query: str
    attach_files: list[AttachFile]

    # ========= 意图识别输出（每轮刷新） =========
    intent: IntentType
    task_type: TaskType
    fact_extract: str  # 简要案情摘要

    # ========= 检索输出 =========
    retrieve_laws: list[dict]  # 本地检索到的法条（结构化）
    retrieve_cases: list[dict]  # 本地检索到的类案（结构化）
    # 全部检索工具返回内容的汇编（本地 + 元典，带来源标记），拼成一个大字符串，供后续生成节点引用
    retrieval_material: str
    constitutive_elements: str
    source_tag: str  # 来源标记，如 local_knowledge+yuandian_mcp

    # ========= 元典 MCP 熔断 =========
    yuandian_available: bool
    mcp_failure_count: int

    # ========= 报告定制框架（仅 report 链路） =========
    # 由四个业务节点分别写入，generate_draft_answer_node 读取渲染。
    report_chapter_titles: list[str] | None

    # ========= HITL 人工复核（仅 report 链路生效） =========
    human_review_enable: bool  # HITL 总开关
    draft_legal_output: str  # 报告初稿
    interrupt_payload: dict | None  # 暂停时发给前端的载荷
    human_decision: dict | None  # {"action": approve/edit/reject, "comment": ...}

    # ========= 输出 =========
    final_answer: str  # 最终回答 / 报告正文
    citation_check_result: str  # 引用幻觉校验结果
    final_result: dict | None  # 组合后的最终结果，供前端展示
