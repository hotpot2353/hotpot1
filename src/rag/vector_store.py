"""Qdrant 客户端 + Embedding + 向量读写。"""

import hashlib
import json
import time
from functools import lru_cache
from uuid import UUID

from langchain_core.documents import Document
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.config import get_settings
from src.core.logger import get_logger
from src.rag.config import get_rag_settings

logger = get_logger(__name__)

# text-embedding-v4 单次请求最多 10 条（实测 11 条返回 400 batch size is invalid）
EMBED_BATCH_SIZE: int = 10

# 单次 upsert 的点数：embedding 也按此粒度分批，避免同时持有全部向量
UPSERT_BATCH_SIZE: int = 1000

# 两次 embedding 请求之间的间隔，规避限流
EMBED_SLEEP_SECONDS: float = 0.2

# 不参与内容哈希的易变字段
# source / index 由 ingestion 加载时注入；id 是数据集的导出行号，重新导出后会变。
# 排除它们，同一内容在不同文件、不同批次中才能算出同一个 id。
_VOLATILE_FIELDS: frozenset[str] = frozenset({"source", "index", "id"})


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """创建 Qdrant 客户端。"""
    qdrant = get_rag_settings().qdrant
    logger.info(f"[get_qdrant_client] 连接 {qdrant.url}")

    return QdrantClient(url=qdrant.url, api_key=qdrant.api_key)


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    """创建 DashScope OpenAI 兼容客户端，用于 embedding。"""
    llm = get_settings().llm

    return OpenAI(api_key=llm.api_key, base_url=llm.base_url)


def content_hash(document: Document) -> str:
    """
    计算 Document 的内容哈希（blake2b，32 字符十六进制摘要）。

    同时用作批内去重键与 Qdrant point id：同一内容在不同文件、不同批次中
    算出同一个 id，重跑入库时以 upsert 覆盖而非追加重复点。

    只哈希正文与稳定字段，排除 _VOLATILE_FIELDS 中由加载/导出产生的易变字段。
    """
    stable_meta = {
        key: value
        for key, value in document.metadata.items()
        if key not in _VOLATILE_FIELDS
    }
    blob = json.dumps(
        {"content": document.page_content, "meta": stable_meta},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    # 归一化空白：同一法条在不同数据源中尾随换行不一致
    normalized = "".join(blob.split())

    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def point_id(document: Document) -> str:
    """
    由内容哈希生成 Qdrant point id（标准 UUID 字符串）。

    Qdrant 的 id 只接受无符号整数或 UUID，所以把 32 位十六进制摘要
    转成标准 UUID 字符串。转换是确定的，同一内容始终得到同一 id。
    """
    return str(UUID(hex=content_hash(document)))


def ensure_collection(name: str) -> None:
    """按 embedding 维度创建 collection；已存在则跳过。"""
    client = get_qdrant_client()
    dimension = get_rag_settings().embedding.dimension

    if client.collection_exists(name):
        logger.info(f"[ensure_collection]{name} 已存在，跳过创建")
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    logger.info(f"[ensure_collection]{name} 创建完成，维度 {dimension}")


def _embed_batch(client: OpenAI, model: str, batch: list[str], attempts: int = 3) -> list[list[float]]:
    """
    单批 embedding 请求，失败重试，返回与本批顺序一致的向量。

    全量入库要发上万次请求，单次网络抖动就会中断整个任务，因此做有限重试。
    """
    for attempt in range(1, attempts + 1):
        try:
            response = client.embeddings.create(model=model, input=batch)
            # 接口按 index 返回，排序后再取，确保与入参顺序一致
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except Exception as e:
            if attempt == attempts:
                logger.error(f"[_embed_batch] 重试 {attempts} 次仍失败：{type(e).__name__} {e}")
                raise
            wait = attempt * 2
            logger.warning(
                f"[_embed_batch] 第 {attempt} 次失败（{type(e).__name__}），{wait}s 后重试"
            )
            time.sleep(wait)

    return []


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量生成向量，返回与 texts 等长、顺序一致的向量列表。

    text-embedding-v4 单次请求最多 10 条，按 EMBED_BATCH_SIZE 分批，
    批间 sleep 规避限流。
    """
    client = get_embedding_client()
    model = get_rag_settings().embedding.model
    vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch(client, model, batch))
        time.sleep(EMBED_SLEEP_SECONDS)

    return vectors


def upsert_chunks(collection: str, documents: list[Document]) -> int:
    """
    向量化并写入 Qdrant，返回实际写入的点数。

    逐批处理：embedding 一批、upsert 一批，不同时持有全部向量。
    同内容重跑会覆盖同一个 point，因此入库是幂等的。
    """
    client = get_qdrant_client()
    total = len(documents)
    written = 0

    for start in range(0, total, UPSERT_BATCH_SIZE):
        batch = documents[start : start + UPSERT_BATCH_SIZE]
        vectors = embed_texts([document.page_content for document in batch])
        points = [
            PointStruct(
                id=point_id(document),
                vector=vector,
                # content 放在最后，确保正文不被同名 metadata 覆盖
                payload={**document.metadata, "content": document.page_content},
            )
            for document, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        written += len(points)
        logger.info(f"[upsert_chunks]{collection} 已写入 {written}/{total}")

    return written


if __name__ == "__main__":
    settings = get_rag_settings()
    ensure_collection(settings.collection.laws)
    ensure_collection(settings.collection.cases)
    print(get_qdrant_client().get_collections().collections)