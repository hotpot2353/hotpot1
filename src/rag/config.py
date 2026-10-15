"""RAG 专属配置：读取 configs/rag.yaml。"""

from dataclasses import dataclass
from functools import lru_cache

from src.core.config import get_env_var, load_yaml


@dataclass(frozen=True)
class CollectionConfig:
    """集合名。"""

    laws: str
    cases: str


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 模型参数。"""

    model: str
    dimension: int


@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant 连接参数。"""

    url: str
    api_key: str | None


@dataclass(frozen=True)
class RetrievalConfig:
    """RAG 检索参数。"""

    top_k: int
    vector_weight: float
    bm25_weight: float
    rerank: bool
    rerank_model: str


@dataclass(frozen=True)
class ChunkingConfig:
    """切分参数。"""

    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class RagSettings:
    """RAG 专属配置。"""

    collection: CollectionConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig

    def collection_of(self, corpus: str) -> str:
        """按语料类型取 Qdrant collection 名（"laws" 或 "cases"）。"""
        if corpus not in ("laws", "cases"):
            raise ValueError(f"未知语料类型：{corpus}，可选 ['laws', 'cases']")

        return getattr(self.collection, corpus)


@lru_cache(maxsize=1)
def get_rag_settings() -> RagSettings:
    """加载并缓存 RAG 配置。"""
    raw = load_yaml("rag.yaml")

    collection_raw = raw.get("collection", {})
    embedding_raw = raw.get("embedding", {})
    retrieval_raw = raw.get("retrieval", {})
    chunking_raw = raw.get("chunking", {})

    return RagSettings(
        collection=CollectionConfig(
            laws=collection_raw.get("laws", "laws"),
            cases=collection_raw.get("cases", "cases"),
        ),
        embedding=EmbeddingConfig(
            # 环境变量优先于 yaml，便于部署时切换模型
            model=get_env_var("EMBEDDING_MODEL") or embedding_raw.get("model", ""),
            dimension=int(embedding_raw.get("dimension", 1024)),
        ),
        qdrant=QdrantConfig(
            url=get_env_var("QDRANT_URL", "http://localhost:6333"),
            # 本地 Docker 默认不校验认证，.env 留空即可。
            # 空字符串要转成 None：否则 qdrant-client 会带上空的 api-key 头，
            # 并警告 "Api key is used with an insecure connection"。
            api_key=get_env_var("QDRANT_API_KEY") or None,
        ),
        retrieval=RetrievalConfig(
            top_k=int(retrieval_raw.get("top_k", 5)),
            vector_weight=float(retrieval_raw.get("vector_weight", 0.7)),
            bm25_weight=float(retrieval_raw.get("bm25_weight", 0.3)),
            rerank=bool(retrieval_raw.get("rerank", True)),
            rerank_model=retrieval_raw.get("rerank_model", "gte-rerank-v2"),
        ),
        chunking=ChunkingConfig(
            chunk_size=int(chunking_raw.get("chunk_size", 500)),
            chunk_overlap=int(chunking_raw.get("chunk_overlap", 50)),
        ),
    )


if __name__ == "__main__":
    print(get_rag_settings())