"""检索层数据结构：Chunk、RetrievalResult。

原始文档与切分流程统一使用 langchain 的 Document（page_content + metadata），
此处不再自定义同名类，避免概念混淆。
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    切分后的文本块，向量化与写入 Qdrant 前的载体。

    chunk_id 需为稳定值（如 uuid5(命名空间, doc_id + 序号)）：
    重跑入库时以相同 id upsert 覆盖，而非追加重复点。
    """

    chunk_id: str
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """
    混合检索（向量 + BM25 + 重排）的返回类型。

    score 为多路融合后的最终得分，不是单一检索器的原始分。
    前端溯源面板据此展示来源与相关度。
    """

    chunk: Chunk
    score: float