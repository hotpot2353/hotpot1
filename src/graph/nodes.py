"""LangGraph 业务节点。

- intent_recognize_node：每轮入口，识别 intent / task_type 并 Command(goto) 分流。
- light_retrieve_node / full_retrieve_node：工具调用检索（本地 Function-Calling
  + 元典 MCP），附件按链路差异化使用，回写 MCP 熔断状态。
- accuse / defend / doc_review / consult 四个业务节点：按 fixed_input_mode
  产出该业务的分析素材与 7 个专属报告章节标题。
- generate_draft_answer_node：按章节标题生成正式报告，支持 HITL 驳回重写。
- hitl_review_node / apply_human_decision_node：interrupt 暂停 + 三动作处理。
- chat_answer / reject_non_criminal / citation_check / result_combine：收尾节点。

节点名常量集中在此定义，builder 与节点内 Command(goto=...) 统一引用，
避免字符串散落两处导致图注册名与跳转名不一致。
"""

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.logger import get_logger
from src.core.prompt_loader import load_prompt_sections
from src.graph.state import AgentState
from src.llm.client import get_llm
from src.rag.retriever import hybrid_search
from src.tools.knowledge_tool import _format_cases, _format_laws, search_knowledge_base
from src.tools.mcp_yuandian import (
    YuandianMCPError,
    build_yuandian_tools,
    get_yuandian_client,
)

logger = get_logger(__name__)

# ========= 节点名（与 graph.png 流程图一致） =========
INTENT_RECOGNIZE = "intent_recognize_node"
REJECT_NON_CRIMINAL = "reject_non_criminal_node"
LIGHT_RETRIEVE = "light_retrieve_node"
CHAT_ANSWER = "chat_answer_node"
FULL_RETRIEVE = "full_retrieve_node"
ACCUSE_ANALYSIS = "accuse_analysis_node"
DEFEND_ANALYSIS = "defend_analysis_node"
DOC_REVIEW = "doc_review_node"
CONSULT = "consult_node"
GENERATE_DRAFT_ANSWER = "generate_draft_answer_node"
HITL_REVIEW = "hitl_review_node"
APPLY_HUMAN_DECISION = "apply_human_decision_node"
CITATION_CHECK = "citation_check_node"
RESULT_COMBINE = "result_combine_node"

# 兜底：报告意图强制关键词：命中即判 report，即便输入很短或 LLM 判成 simple_chat。
FORCED_REPORT_KEYWORDS: tuple[str, ...] = (
    "生成报告",
    "出具报告",
    "完整案件分析",
    "完整分析",
    "全面分析",
    "出具法律评估",
    "法律评估",
    "法律意见书",
    "梳理全案风险",
)


class _IntentDecision(BaseModel):
    """意图识别结构化输出（function-calling 约束）。"""
    intent: Literal["criminal", "non_criminal"] = Field(description="是否刑事法律相关")
    task_type: Literal["simple_chat", "report"] = Field(description="本轮任务类型")
    fact_extract: str = Field(description="一句话案情/诉求摘要，不超过50字")


def _wants_report(query: str) -> bool:
    """关键词兜底：用户显式索要正式报告成果。防止 LLM 漏判 report"""
    return any(keyword in query for keyword in FORCED_REPORT_KEYWORDS)


def _recognize_with_llm(query: str) -> _IntentDecision:
    """调用 LLM 做两轴意图识别：是否刑事法律相关、任务类型。"""
    sections = load_prompt_sections("intent_recognize")
    messages = [
        SystemMessage(content=sections.get("system", "")),
        HumanMessage(content=sections.get("user", "{query}").replace(
            "{query}", query
        )),
    ]
    structured_llm = get_llm(thinking=False).with_structured_output(_IntentDecision)
    decision = structured_llm.invoke(messages)

    # 防御性归一，避免上游异常值进 state
    if decision.intent not in ("criminal", "non_criminal"):
        decision.intent = "criminal"
    if decision.task_type not in ("simple_chat", "report"):
        decision.task_type = "simple_chat"
    if not decision.fact_extract:
        decision.fact_extract = query[:50]

    return decision


# ========= 检索节点参数 =========
# 轻量链路：少量召回，附件只取片段，只调本地工具，不消耗元典积分
LIGHT_LAWS_TOP_K: int = 3
LIGHT_CASES_TOP_K: int = 2
LIGHT_ATTACH_CHARS: int = 500  # 轻量链路附件片段长度上限
LIGHT_MAX_TOOL_ROUNDS: int = 2  # 轻量链路最多工具调用轮数

# 全量链路：充分召回，附件给全文，本地 + 元典双源，最多 3 轮工具调用
FULL_LAWS_TOP_K: int = 5
FULL_CASES_TOP_K: int = 5
FULL_MAX_TOOL_ROUNDS: int = 3

# 单次工具结果进入检索素材/回喂模型的长度上限，防止超长文书撑爆上下文
MATERIAL_CAP_PER_CALL: int = 12000

# 轻量链路检索提示词
LIGHT_RETRIEVAL_INSTRUCTION: str = (
    "本轮为简单问答：只做少量精准检索，本地知识库通常调用一次即可，"
    "不要展开正式分析。"
)
FULL_RETRIEVAL_INSTRUCTION: str = (
    "本轮为报告生成：请充分检索，覆盖罪名构成要件、量刑标准、司法解释与"
    "权威/典型案例；本地结果不足时用元典工具补充，附件中的案情事实应作为"
    "构造检索词的依据。"
)


class _LocalSearchInput(BaseModel):
    """本地知识库检索入参。"""
    query: str = Field(description="自然语言检索问题")


def _serialize_local(result) -> dict:
    """把本地 RetrievalResult 序列化为可入 state 的 dict。"""
    chunk = result.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "name": chunk.metadata.get("name"),
        "content": chunk.content,
        "score": result.score,
        "meta": chunk.metadata.get("meta"),
        "source": "local_qdrant",
    }


def _build_local_search_tool(
    laws_top_k: int,
    cases_top_k: int, # 类案召回条数
    laws_bucket: list[dict], # 收集法条结构化结果的列表（按 chunk_id 去重）
    cases_bucket: list[dict],# 收集类案结构化结果的列表（按 chunk_id 去重）
) -> StructuredTool:
    """
    构造本节点专用的本地知识库工具

    与注册表里的 search_knowledge_base 同名同描述（保证 LLM 选参行为一致），
    但 top_k 按 light/full 链路区分；进程内执行，顺手把结构化结果收进桶里，
    供节点回写 state（MCP 工具无法回写 state，本地工具可以）。
    """
    seen_laws: set[str] = set()
    seen_cases: set[str] = set()

    def run(query: str) -> str:
        laws = hybrid_search(query, corpus="laws", top_k=laws_top_k)
        cases = hybrid_search(query, corpus="cases", top_k=cases_top_k)

        for result in laws:
            if result.chunk.chunk_id not in seen_laws:
                seen_laws.add(result.chunk.chunk_id)
                laws_bucket.append(_serialize_local(result))
        for result in cases:
            if result.chunk.chunk_id not in seen_cases:
                seen_cases.add(result.chunk.chunk_id)
                cases_bucket.append(_serialize_local(result))

        logger.info(
            f"[retrieve] 本地检索 '{query}' → 法条 {len(laws)}、类案 {len(cases)}"
        )
        return (
            f"【法条检索】\n{_format_laws(laws)}"
            f"\n\n【类案检索】\n{_format_cases(cases)}"
        )

    return StructuredTool.from_function(
        run,
        name="search_knowledge_base",
        description=search_knowledge_base.description,
        args_schema=_LocalSearchInput,
    )


def _attachment_context(attach_files: list[dict] | None, *, full: bool) -> str:
    """
    组装附件文本。simple_chat 链路每个附件只取前 LIGHT_ATTACH_CHARS 字（片段
    问答）；report 链路给全文（完整案情、证据要素抽取）；无附件返回空串。
    """
    if not attach_files:
        return ""

    parts: list[str] = []
    for item in attach_files:
        name = item.get("filename", "未命名附件")
        content = item.get("content", "") or ""
        if not full and len(content) > LIGHT_ATTACH_CHARS:
            content = content[:LIGHT_ATTACH_CHARS] + "……（简单问答仅截取附件片段）"
        parts.append(f"### {name}\n{content}")

    return "\n\n".join(parts)


def _cap(text: str) -> str:
    """截断超长工具结果，保留头部并显式标注。"""
    if len(text) <= MATERIAL_CAP_PER_CALL:
        return text
    return text[:MATERIAL_CAP_PER_CALL] + "\n……（工具结果过长，已截断）"


def _mcp_status() -> tuple[bool, int]:
    """读取元典熔断状态快照；未配置 token 视为不可用。"""
    client = get_yuandian_client()
    if client is None:
        return False, 0
    status = client.status()
    return bool(status["yuandian_available"]), int(status["mcp_failure_count"])


def _run_retrieval_loop(
    state: AgentState,
    tools: list,
    max_rounds: int,  # 最多工具调用轮数
    instruction: str,  # 本轮检索要求，light/full 不同
    *,
    full: bool,  # 是否全量链路，影响附件上下文组装
) -> tuple[str, bool]:
    """
    驱动 LLM 工具调用循环完成检索。

    每轮把模型的 tool_calls 全部执行并把结果以 ToolMessage 回喂；
    模型不再调用工具即结束。元典工具异常不抛出循环外，转成错误提示让
    模型换用本地结果（熔断降级在工具层与提示层同时生效）。
    """
    sections = load_prompt_sections("tool_agent")
    user_content = (
        sections.get("user", "")
        .replace("{query}", state.get("query", ""))
        .replace("{fact_extract}", state.get("fact_extract") or "（无）")
        .replace(
            "{attach_context}",
            _attachment_context(state.get("attach_files"), full=full)
            or "（无附件）",
        )
        .replace("{retrieval_instruction}", instruction)
    )

    messages = [
        SystemMessage(content=sections.get("system", "")),
        HumanMessage(content=user_content),
    ]
    by_name = {tool.name: tool for tool in tools}
    materials: list[str] = []
    yuandian_used = False
    bound_llm = get_llm(thinking=False).bind_tools(tools)

    for round_no in range(1, max_rounds + 1):
        ai_message = bound_llm.invoke(messages)
        messages.append(ai_message)
        if not ai_message.tool_calls:
            logger.info(f"[retrieve] 第 {round_no} 轮模型停止调用工具，检索结束")
            break
        for call in ai_message.tool_calls:
            tool = by_name.get(call["name"])
            is_yuandian = call["name"].startswith("yuandian_")

            if tool is None:
                output = f"工具 {call['name']} 不存在，请改用其他工具。"
            else:
                try:
                    output = tool.invoke(call["args"])
                    if is_yuandian:
                        yuandian_used = True
                except YuandianMCPError as e:
                    output = (
                        f"外部元典检索暂时不可用（{e}）。"
                        "请仅依据本地知识库结果继续，不要再次调用该工具。"
                    )
                    logger.warning(f"[retrieve] 元典工具 {call['name']} 降级：{e}")
                except Exception as e:
                    output = f"工具执行出错：{type(e).__name__}: {e}，请换用其他工具。"
                    logger.warning(f"[retrieve] 工具 {call['name']} 异常：{e}")
            source_label = "元典智库" if is_yuandian else "本地知识库"
            output = _cap(output)
            materials.append(
                f"【来源：{source_label}｜{call['name']}】\n{output}"
            )
            # tool_call_id 必须与模型发起的工具调用 ID 配对
            messages.append(
                ToolMessage(content=output, tool_call_id=call["id"])
            )

    return "\n\n".join(materials), yuandian_used


def _do_retrieve(state: AgentState, *, full: bool) -> dict:
    """
    检索节点公共逻辑。full=False 为 light_retrieve（本地、片段、少量召回），
    True 为 full_retrieve（双源、全文、充分召回）。
    """
    laws_bucket: list[dict] = []
    cases_bucket: list[dict] = []
    local_tool = _build_local_search_tool(
        laws_top_k=FULL_LAWS_TOP_K if full else LIGHT_LAWS_TOP_K,
        cases_top_k=FULL_CASES_TOP_K if full else LIGHT_CASES_TOP_K,
        laws_bucket=laws_bucket,
        cases_bucket=cases_bucket,
    )

    tools = [local_tool]
    if full:
        # 轻量链路只挂本地工具：不消耗元典积分，延迟也更低
        tools.extend(build_yuandian_tools())
    instruction = FULL_RETRIEVAL_INSTRUCTION if full else LIGHT_RETRIEVAL_INSTRUCTION
    material, yuandian_used = _run_retrieval_loop(
        state, tools, FULL_MAX_TOOL_ROUNDS if full else LIGHT_MAX_TOOL_ROUNDS,
        instruction, full=full,
    )
    yuandian_available, failure_count = _mcp_status()
    source_tag = "local_knowledge"
    if full and yuandian_used:
        source_tag = "local_knowledge+yuandian_mcp"

    logger.info(
        f"[{'full' if full else 'light'}_retrieve] 法条 {len(laws_bucket)}、"
        f"类案 {len(cases_bucket)}、素材 {len(material)} 字、来源 {source_tag}、"
        f"元典可用={yuandian_available}（失败 {failure_count} 次）"
    )

    return {
        "retrieve_laws": laws_bucket,
        "retrieve_cases": cases_bucket,
        "retrieval_material": material,
        "source_tag": source_tag,
        "yuandian_available": yuandian_available,
        "mcp_failure_count": failure_count,
    }


def light_retrieve_node(state: AgentState) -> dict:
    """
    simple_chat 轻量检索节点。

    本地知识库少量召回（法条 3、类案 2），附件只取前 500 字片段，
    不调用元典、不生成报告、不触发 HITL。
    """
    return _do_retrieve(state, full=False)


def full_retrieve_node(state: AgentState) -> dict:
    """
    report 全量检索节点。

    本地充分召回（法条 5、类案 5）+ 元典 MCP 补充，附件读取全文，
    元典熔断时工具层自动报错、模型改走本地，链路不中断。
    """
    return _do_retrieve(state, full=True)


# ========= 四个业务节点 =========
# 每个业务固定 7 个报告章节标题，由业务节点写入 state，
# generate_draft_answer_node 读取后渲染，报告标题不再写死在生成节点里。
# 风险研判章节（第 5 章）的业务侧重由各业务提示词控制。
BUSINESS_CHAPTER_TITLES: dict[str, list[str]] = {
    "accuse_analysis": [
        "案件事实梳理",
        "涉嫌犯罪构成要件拆解",
        "法律依据",
        "类案立案裁判要点",
        "控告风险与立案可行性研判",
        "取证与控告行动建议",
        "引用溯源附录",
    ],
    "defend_analysis": [
        "案件事实梳理",
        "被控罪名构成要件分析",
        "法律依据",
        "类案辩护裁判参考",
        "辩护风险与辩护切入点研判",
        "辩护策略行动建议",
        "引用溯源附录",
    ],
    "doc_review": [
        "文书事实提取梳理",
        "文书涉及法律要点",
        "法律依据",
        "同类文书裁判参考",
        "文书法律风险与漏洞研判",
        "文书修改优化建议",
        "引用溯源附录",
    ],
    "consult": [
        "案件事实梳理",
        "涉案法律要件分析",
        "法律依据",
        "类案裁判参考要点",
        "控辩双向风险综合研判",
        "综合行动建议总结",
        "引用溯源附录",
    ],
}

# fixed_input_mode → 业务提示词文件名（prompts/<name>.md）
_BUSINESS_PROMPTS: dict[str, str] = {
    "accuse_analysis": "accuse_analysis",
    "defend_analysis": "defend_analysis",
    "doc_review": "doc_review",
    "consult": "consult",
}


def _business_analysis(state: AgentState, mode: str) -> dict:
    """
    四个业务节点的公共逻辑

    各节点的差异全部数据化：章节标题取 ``BUSINESS_CHAPTER_TITLES``，
    视角与风险侧重由各自的提示词承载，代码路径只有一条。

    业务分析需要推理，使用 thinking=True 的 LLM；模型异常时降级为
    "仅透传检索素材"——章节标题照常写入，报告节点仍可成文，链路不中断。
    """
    titles = BUSINESS_CHAPTER_TITLES[mode]
    sections = load_prompt_sections(mode)
    titles_text = "\n".join(
        f"{index}. {title}" for index, title in enumerate(titles, 1)
    )
    user_content = (
        sections.get("user", "")
        .replace("{query}", state.get("query", ""))
        .replace("{fact_extract}", state.get("fact_extract") or "（无）")
        .replace(
            "{attach_context}",
            _attachment_context(state.get("attach_files"), full=True) or "（无附件）",
        )
        .replace("{retrieval_material}", state.get("retrieval_material") or "（无）")
        .replace("{chapter_titles}", titles_text)
    )

    try:
        response = get_llm(thinking=True).invoke(
            [
                SystemMessage(content=sections.get("system", "")),
                HumanMessage(content=user_content),
            ]
        )
        analysis = response.content.strip()
        if not analysis:
            raise RuntimeError("业务分析返回为空")
    except Exception as e:
        logger.warning(
            f"[{mode}] 业务分析 LLM 失败（{type(e).__name__}: {e}），透传检索素材兜底"
        )
        analysis = (
            "（业务分析模型暂时不可用，以下为检索节点获取的原始素材，"
            "请报告生成节点基于素材直接成文）\n\n"
            + (state.get("retrieval_material") or "（无检索素材）")
        )

    logger.info(f"[{mode}] 业务分析完成，素材 {len(analysis)} 字，章节 {len(titles)} 个")

    return {
        "constitutive_elements": analysis,
        "report_chapter_titles": titles,
    }


def accuse_analysis_node(state: AgentState) -> dict:
    """刑事控告：输出控告分析素材，风险研判侧重立案与取证。"""
    return _business_analysis(state, "accuse_analysis")


def defend_analysis_node(state: AgentState) -> dict:
    """刑事辩护：输出辩护分析素材，风险研判侧重无罪罪轻切入点。"""
    return _business_analysis(state, "defend_analysis")


def doc_review_node(state: AgentState) -> dict:
    """文书审查：输出文书漏洞与修改素材，风险研判侧重漏洞与修改建议。"""
    return _business_analysis(state, "doc_review")


def consult_node(state: AgentState) -> dict:
    """综合咨询：控辩双向客观分析素材。"""
    return _business_analysis(state, "consult")


# ========= 报告生成节点 =========
# 7 章节报告是长文，全局 max_tokens=4096 容易截断，报告节点单独放大
DRAFT_MAX_TOKENS: int = 8192

# fixed_input_mode → 报告中的业务类型中文名
BUSINESS_LABELS: dict[str, str] = {
    "accuse_analysis": "刑事控告",
    "defend_analysis": "刑事辩护",
    "doc_review": "文书审查",
    "consult": "综合咨询",
}


def _chapter_titles_of(state: AgentState) -> list[str]:
    """
    取本次报告章节标题。

    正常路径由业务节点写入；异常缺失时兜底综合咨询的 7 章，
    保证报告节点永远有可渲染的章节框架。
    """
    titles = state.get("report_chapter_titles")
    if titles:
        return list(titles)
    logger.warning("[generate_draft] report_chapter_titles 缺失，兜底综合咨询章节")
    return list(BUSINESS_CHAPTER_TITLES["consult"])


def _check_chapters(draft: str, titles: list[str]) -> list[str]:
    """校验报告是否覆盖全部章节标题，返回缺失的标题列表（为空表示齐全）。"""
    return [title for title in titles if title not in draft]


def _skeleton_draft(state: AgentState, titles: list[str], note: str) -> str:
    """
    LLM 不可用时的骨架兜底报告。

    章节标题 + 分析素材原文拼装，保证 HITL / 引用校验 / 前端链路不断。
    """
    analysis = state.get("constitutive_elements") or "（无分析素材）"
    blocks = [
        f"## {titles[0]}",
        f"> {note}",
        analysis,
    ]
    blocks.extend(f"## {title}\n（本章内容待补充）" for title in titles[1:])
    return "\n\n".join(blocks)


def generate_draft_answer_node(state: AgentState) -> dict:
    """
    生成（或按人工驳回意见重写）7 章节正式报告

    两种进入方式：
    - 首次成文：业务节点之后进入，读 report_chapter_titles +
      constitutive_elements，使用 generate_draft 提示词；
    - 驳回重写：HITL 中人工选择 reject 后由 apply_human_decision_node
      跳回，使用 apply_human_decision 提示词，带上一版报告与复核意见。

    报告标题必须与 state 中的章节标题逐字一致；模型漏章时只告警不自动补写
    （兜底骨架除外），交由引用校验与人工复核发现。
    消费完 reject 决策后清空 human_decision，避免下一轮报告误入重写路径。
    """
    titles = _chapter_titles_of(state)
    titles_text = "\n".join(
        f"{index}. {title}" for index, title in enumerate(titles, 1)
    )
    mode = state.get("fixed_input_mode") or "consult"
    business_type = BUSINESS_LABELS.get(mode, "综合咨询")
    human_decision = state.get("human_decision") or {}
    is_reject_rewrite = human_decision.get("action") == "reject"
    if is_reject_rewrite:
        prompt_name = "apply_human_decision"
        human_comment = human_decision.get("comment") or "（未填写具体意见）"
        replacements = {
            "{business_type}": business_type,
            "{human_comment}": human_comment,
            "{draft}": state.get("draft_legal_output") or "（无上一版报告）",
            "{analysis}": state.get("constitutive_elements") or "（无分析素材）",
            "{chapter_count}": str(len(titles)),
            "{chapter_titles}": titles_text,
        }
        logger.info("[generate_draft] 按人工驳回意见重写报告")
    else:
        prompt_name = "generate_draft"
        replacements = {
            "{business_type}": business_type,
            "{query}": state.get("query", ""),
            "{fact_extract}": state.get("fact_extract") or "（无）",
            "{analysis}": state.get("constitutive_elements") or "（无分析素材）",
            "{context}": state.get("retrieval_material") or "（无检索素材）",
            "{chapter_count}": str(len(titles)),
            "{chapter_titles}": titles_text,
        }
    sections = load_prompt_sections(prompt_name)
    user_content = sections.get("user", "")
    for placeholder, value in replacements.items():
        user_content = user_content.replace(placeholder, value)

    try:
        response = get_llm(
            thinking=True, max_tokens=DRAFT_MAX_TOKENS
        ).invoke(
            [
                SystemMessage(content=sections.get("system", "")),
                HumanMessage(content=user_content),
            ]
        )
        draft = response.content.strip()
        if not draft:
            raise RuntimeError("报告生成返回为空")
    except Exception as e:
        logger.warning(
            f"[generate_draft] LLM 失败（{type(e).__name__}: {e}），输出骨架兜底报告"
        )
        draft = _skeleton_draft(state, titles, f"报告生成模型暂时不可用（{type(e).__name__}），以下为分析素材原文，请人工补全。")
    missing = _check_chapters(draft, titles)
    if missing:
        logger.warning(f"[generate_draft] 报告缺少章节：{missing}")
    else:
        logger.info(f"[generate_draft] 报告 {len(draft)} 字，{len(titles)} 个章节齐全")

    update: dict = {"draft_legal_output": draft}
    if is_reject_rewrite:
        # 决策已消费，必须清空：否则同一会话下一轮新报告会误入重写路径
        update["human_decision"] = None
    return update


# ========= HITL 人工复核（仅 report 链路可达） =========
# 非刑事案件拒绝话术（固定，不调 LLM）
NON_CRIMINAL_REPLY: str = (
    "抱歉，我是刑事法律智能助手，只能解答刑事法律相关问题。"
    "若你遇到的是民事、行政等方面的纠纷，建议咨询对应领域的专业律师。"
)

# 人工动作 → 中文标签，写入校验/组合结果便于前端展示
HUMAN_ACTION_LABELS: dict[str, str] = {
    "approve": "批准",
    "edit": "修改",
    "reject": "驳回",
}


def _normalize_decision(decision: Any) -> dict:
    """
    规范化前端 resume 回来的人工决策。

    允许 {"action": approve/edit/reject, "comment": ..., "edited_content": ...}；
    结构缺失或动作非法时兜底批准并告警，保证图能继续，不让异常输入卡死流程。
    """
    if not isinstance(decision, dict):
        logger.warning(f"[hitl] 人工决策格式非法（{type(decision).__name__}），兜底批准")
        return {"action": "approve", "comment": "", "edited_content": ""}

    action = decision.get("action")
    if action not in HUMAN_ACTION_LABELS:
        logger.warning(f"[hitl] 未知人工动作 {action!r}，兜底批准")
        action = "approve"

    return {
        "action": action,
        "comment": str(decision.get("comment") or ""),
        "edited_content": str(decision.get("edited_content") or ""),
    }


def hitl_review_node(state: AgentState) -> dict:
    """
    人工复核中断节点。

    用 LangGraph 原生 ``interrupt`` 暂停图执行，把报告初稿与可执行动作
    通过中断事件发给前端；前端经 ``Command(resume=决策)`` 恢复后，
    ``interrupt`` 返回人工决策，节点将其规范化写入 state。

    resume 时节点函数会从 checkpoint 恢复并从头重放：首次执行在
    interrupt 处暂停，此前的 return 不会发生，因此暂停期间前端需要的
    初稿、章节标题与动作集合只能经 interrupt() 的参数送出。
    """
    payload = {
        "type": "human_review",
        "draft": state.get("draft_legal_output", ""),
        "chapter_titles": state.get("report_chapter_titles") or [],
        "available_actions": [
            {"action": action, "label": label}
            for action, label in HUMAN_ACTION_LABELS.items()
        ],
        "instruction": (
            "请审核报告：批准则进入引用校验；修改请提交修改后全文；"
            "驳回请填写意见，系统将据此重新生成。"
        ),
    }
    raw_decision = interrupt(payload)
    decision = _normalize_decision(raw_decision)

    logger.info(f"[hitl] 收到人工决策：{HUMAN_ACTION_LABELS[decision['action']]}")
    return {"human_decision": decision, "interrupt_payload": None}


def apply_human_decision_node(state: AgentState) -> Command[str]:
    """
    处理人工复核结果。

    - approve：初稿不动，进入引用校验；
    - edit：用人工修改后的全文替换 draft_legal_output，再进引用校验；
      若前端选了 edit 但没给文本，视为批准并告警；
    - reject：``Command(goto)`` 回 generate_draft_answer_node，
      该节点检测到 human_decision.action == reject 后按意见重写。

    三个动作互斥路由，返回 Command(goto=citation_check) 或
    Command(goto=generate_draft)。
    """
    decision = state.get("human_decision") or {}
    action = decision.get("action", "approve")

    if action == "reject":
        # human_decision 保留在 state 中：generate_draft_answer_node 识别
        # reject 后带上一版报告与意见重写，重写完成会清空该决策
        logger.info("[apply_human_decision] 驳回，返回报告生成节点重写")
        return Command(goto=GENERATE_DRAFT_ANSWER)

    if action == "edit":
        edited_content = (decision.get("edited_content") or "").strip()
        if not edited_content:
            logger.warning("[apply_human_decision] 修改动作未提交文本，降级为批准")
        else:
            logger.info("[apply_human_decision] 采纳人工修改全文，进入引用校验")
            return Command(
                goto=CITATION_CHECK,
                update={"draft_legal_output": edited_content},
            )

    logger.info("[apply_human_decision] 批准，进入引用校验")
    return Command(goto=CITATION_CHECK)


# ========= simple_chat 与非刑事收尾 =========
def chat_answer_node(state: AgentState) -> dict:
    """
    simple_chat 简短回答节点。

    基于轻量检索素材生成 3~6 句话回答，不生成报告、不走业务节点。
    LLM 异常时退化为直接透传检索素材，保证有内容可校验、可返回。
    """
    sections = load_prompt_sections("chat_answer")
    user_content = (
        sections.get("user", "")
        .replace("{query}", state.get("query", ""))
        .replace("{fact_extract}", state.get("fact_extract") or "（无）")
        .replace("{retrieval_material}", state.get("retrieval_material") or "（无检索素材）")
    )

    try:
        response = get_llm(thinking=False).invoke(
            [
                SystemMessage(content=sections.get("system", "")),
                HumanMessage(content=user_content),
            ]
        )
        answer = response.content.strip()
        if not answer:
            raise RuntimeError("简短回答返回为空")
    except Exception as e:
        logger.warning(
            f"[chat_answer] LLM 失败（{type(e).__name__}: {e}），透传检索素材"
        )
        answer = state.get("retrieval_material") or "暂时无法生成回答，请稍后再试。"

    logger.info(f"[chat_answer] 简短回答 {len(answer)} 字")
    return {"final_answer": answer}


def reject_non_criminal_node(state: AgentState) -> dict:
    """
    非刑事案件拒绝回答节点。

    固定话术，不调 LLM、不做检索；按流程图直接到 END，
    但同样写齐 final_answer / final_result，保证前端数据结构统一。
    """
    logger.info("[reject_non_criminal] 非刑事输入，拒绝回答")
    return {
        "final_answer": NON_CRIMINAL_REPLY,
        "final_result": {
            "task_type": "rejected",
            "answer": NON_CRIMINAL_REPLY,
            "citation_check": None,
            "sources": {"laws": [], "cases": []},
            "source_tag": "none",
        },
    }


# ========= 引用校验与结果组合 =========
class _CitationVerdict(BaseModel):
    """引用幻觉校验结构化输出。"""
    status: Literal["supported", "partially_supported", "unsupported"] = Field(
        description="全部有依据 / 存在部分或无依据引用 / 基本无依据"
    )
    checks: list[str] = Field(description="逐条校验：引用内容 → 判定 → 素材依据")
    summary: str = Field(description="一句话总体结论")


_CITATION_STATUS_LABELS: dict[str, str] = {
    "supported": "全部引用有依据",
    "partially_supported": "存在部分/无依据引用",
    "unsupported": "引用基本无依据",
}


def citation_check_node(state: AgentState) -> dict:
    """
    引用幻觉校验节点（simple_chat 与 report 两条链路都经过）。
    LLM 异常时不阻断链路，写入"未完成校验，请人工核对"标记。
    """
    # report 链路校验 draft_legal_output，simple_chat 校验 final_answer
    answer = state.get("draft_legal_output") or state.get("final_answer") or ""
    # 唯一允许的校验依据是 retrieval_material
    material = state.get("retrieval_material") or "（无检索素材）"
    sections = load_prompt_sections("citation_check")
    user_content = (
        sections.get("user", "")
        .replace("{answer}", answer)
        .replace("{context}", material)
    )

    try:
        verdict = get_llm(thinking=False).with_structured_output( 
            _CitationVerdict
        ).invoke(
            [
                SystemMessage(content=sections.get("system", "")),
                HumanMessage(content=user_content),
            ]
        )
        checks_text = "\n".join(f"- {item}" for item in verdict.checks) or "- （未给出逐条结果）"
        result = (
            f"结论：{_CITATION_STATUS_LABELS.get(verdict.status, verdict.status)}\n"
            f"总体意见：{verdict.summary}\n逐条校验：\n{checks_text}"
        )
        logger.info(f"[citation_check] 校验完成：{verdict.status}")
    except Exception as e:
        logger.warning(f"[citation_check] 校验失败（{type(e).__name__}: {e}）")
        result = "引用校验未完成（校验模型暂时不可用），报告内容的法条与案号请人工核对。"

    return {"citation_check_result": result}


def result_combine_node(state: AgentState) -> dict:
    """
    最终结果组合节点（两条链路的共同出口）。

    把回答正文、引用校验结论、法条/类案来源、来源标记组装为前端统一
    消费的 final_result；report 链路此时才把 draft_legal_output
    提升为 final_answer。
    """
    is_report = state.get("task_type") == "report"
    answer = (
        state.get("draft_legal_output")
        if is_report
        else state.get("final_answer")
    ) or state.get("final_answer", "")

    result = {
        "task_type": "report" if is_report else "simple_chat",
        "fixed_input_mode": state.get("fixed_input_mode"),
        "answer": answer,
        "citation_check": state.get("citation_check_result"),
        "sources": {
            "laws": state.get("retrieve_laws", []),
            "cases": state.get("retrieve_cases", []),
        },
        "source_tag": state.get("source_tag", "local_knowledge"),
        "yuandian_available": state.get("yuandian_available", True),
        "human_action": None,
    }
    human_decision = state.get("human_decision")
    if human_decision:
        result["human_action"] = HUMAN_ACTION_LABELS.get(
            human_decision.get("action", ""), human_decision.get("action", "")
        )

    logger.info(
        f"[result_combine] task={result['task_type']} 正文 {len(answer)} 字、"
        f"法条 {len(result['sources']['laws'])}、类案 {len(result['sources']['cases'])}"
    )

    update: dict = {"final_result": result}
    if is_report and not state.get("final_answer"):
        update["final_answer"] = answer
    return update


def intent_recognize_node(state: AgentState) -> Command[str]:
    """
    每轮入口节点：识别 intent + task_type，提取案情摘要，并顶层分流

    只读 fixed_input_mode（本节点当前不消费，report 链路的路由节点才使用），
    通过 ``Command(goto=...)`` 跳到三个去向之一：

    - non_criminal → reject_non_criminal_node
    - simple_chat   → light_retrieve_node
    - report        → full_retrieve_node

    LLM 异常时降级为 criminal + simple_chat：轻量链路成本低、不生成报告、
    不触发 HITL，是最安全的兜底去向，保证入口不因模型抖动而中断。
    """
    query = (state.get("query") or "").strip()

    try:
        decision = _recognize_with_llm(query)
    except Exception as e:
        logger.warning(
            f"[intent_recognize] LLM 识别失败（{type(e).__name__}: {e}），"
            "降级 criminal + simple_chat"
        )
        decision = _IntentDecision(
            intent="criminal",
            task_type="simple_chat",
            fact_extract=query[:50],
        )

    # 兜底规则：显式索要报告的表述强制 report（仅对刑事输入生效）
    if decision.intent == "criminal" and _wants_report(query):
        if decision.task_type != "report":
            logger.info("[intent_recognize] 命中报告关键词，task_type 强制为 report")
        decision.task_type = "report"

    if decision.intent == "non_criminal":
        target = REJECT_NON_CRIMINAL
    elif decision.task_type == "report":
        target = FULL_RETRIEVE
    else:
        target = LIGHT_RETRIEVE

    logger.info(
        f"[intent_recognize] intent={decision.intent} "
        f"task_type={decision.task_type} → goto {target}"
    )

    return Command(
        goto=target,
        update={
            "intent": decision.intent,
            "task_type": decision.task_type,
            "fact_extract": decision.fact_extract,
        },
    )
