"""@tool 装饰器封装知识库检索。"""

from langchain_core.tools import tool

from src.core.logger import get_logger
from src.rag.retriever import hybrid_search

logger = get_logger(__name__)

# 两类语料各取几条：Agent 一次会话可能多次调用本工具，
# 这里比检索层的 top_k 更保守，避免上下文被检索结果占满
TOP_K_LAWS: int = 3
TOP_K_CASES: int = 3


def _format_laws(results: list) -> str:
    """法条结果格式化为文本。"""
    lines = []

    for order, result in enumerate(results, start=1):
        name = result.chunk.metadata.get("name", "未知法条")
        lines.append(f"[法条{order}] {name}（相关度 {result.score:.3f}）")
        lines.append(result.chunk.content.strip())
    return "\n".join(lines) if lines else "未检索到相关法条。"


def _format_cases(results: list) -> str:
    """类案结果格式化为文本。

    裁判文书块正文是案件事实，罪名、法条与刑期在 meta 里，
    单独列一行，便于按罪名和量刑档次类比。
    """
    lines = []

    for order, result in enumerate(results, start=1):
        meta = result.chunk.metadata.get("meta") or {}
        labels = []
        accusations = meta.get("accusation") or []
        if accusations:
            labels.append(f"罪名：{'、'.join(accusations)}")

        articles = meta.get("relevant_articles") or []
        if articles:
            labels.append(f"相关法条：{'、'.join(f'第{article}条' for article in articles)}")
        term = meta.get("term_of_imprisonment") or {}
        if term.get("death_penalty"):
            labels.append("刑罚：死刑")
        elif term.get("life_imprisonment"):
            labels.append("刑罚：无期徒刑")
        elif term.get("imprisonment"):
            labels.append(f"刑罚：有期徒刑 {term['imprisonment']} 个月")

        lines.append(f"[类案{order}] {' ｜ '.join(labels)}（相关度 {result.score:.3f}）")
        lines.append(result.chunk.content.strip())

    return "\n".join(lines) if lines else "未检索到相关类案。"


@tool
def search_knowledge_base(query: str) -> str:
    """检索本地刑事法律知识库，返回相关法条与相似裁判文书。

    涉及罪名构成要件、量刑标准、司法解释或需要参考类案裁判时调用。

    Args:
        query: 自然语言查询，如"入户盗窃的量刑标准"
    """
    laws = hybrid_search(query, corpus="laws", top_k=TOP_K_LAWS)
    cases = hybrid_search(query, corpus="cases", top_k=TOP_K_CASES)
    logger.info(f"[search_knowledge_base]{query} 命中法条 {len(laws)} 条、类案 {len(cases)} 条")

    return f"【法条检索】\n{_format_laws(laws)}\n\n【类案检索】\n{_format_cases(cases)}"


if __name__ == "__main__":
    print(search_knowledge_base.invoke("入户盗窃被发现后逃离现场如何定性"))