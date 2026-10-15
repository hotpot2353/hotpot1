"""检索策略：向量 + BM25 + 混合 + 重排。"""

import re

import httpx
from rank_bm25 import BM25Okapi

from src.core.config import get_settings
from src.core.logger import get_logger
from src.rag.config import get_rag_settings
from src.rag.schemas import Chunk, RetrievalResult
from src.rag.vector_store import embed_texts, get_qdrant_client

logger = get_logger(__name__)

# 向量召回的候选池：混合与重排都需要比 top_k 更大的候选集
CANDIDATE_MULTIPLIER: int = 10
MIN_CANDIDATE_POOL: int = 50

# RRF 融合的平滑常数：k 越大越弱化头部名次之间的差距，60 是常用取值
RRF_K: int = 60

# 送进重排的候选数：重排按 (query, doc) 逐条打分，条数直接决定调用成本
# （约 250 token/条），取融合结果的前 N 条即可，不必把整个候选池都送进去。
# 实测取 10 / 15 / 20 / 30 / 50 时 14 条标注 query 的 recall@5 基本一致
# （11/14，取 10 时为 12/14），故取 15，重排成本降到送满池的约 30%
RERANK_CANDIDATES: int = 15

# 重排走 DashScope 原生 REST（OpenAI 兼容模式只有 embeddings / chat）
RERANK_URL: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_TIMEOUT: float = 30.0

# 中文不引分词依赖：单字 + 相邻二元组，英文数字按词
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    """
    BM25 切词：单字 + 相邻二元组，英文数字按词。

    单字保证召回，相邻二元组补上"故意杀人"这类短语的区分度；
    不引入 jieba，避免为检索多加一个依赖。
    """
    units = _TOKEN_PATTERN.findall(text)
    return units + [first + second for first, second in zip(units, units[1:])]


def _lexical_text(result: RetrievalResult) -> str:
    """
    BM25 打分用的文本：法律名/条号拼在正文前。

    法条块正文是裸条文（"故意杀人的，处死刑……"），法律名与条号在 metadata 里。
    检索时拼上 name，查"故意杀人罪的量刑标准"才能命中第二百三十二条。
    """
    name = result.chunk.metadata.get("name")

    return f"{name} {result.chunk.content}" if name else result.chunk.content


def _to_result(point) -> RetrievalResult:
    """把 Qdrant 返回点转成 RetrievalResult。"""
    payload = dict(point.payload or {})
    content = payload.pop("content", "")

    return RetrievalResult(
        chunk=Chunk(
            chunk_id=str(point.id),
            doc_id=f"{payload.get('source', '')}#{payload.get('index', '')}",
            content=content,
            metadata=payload,
        ),
        score=float(point.score),
    )


def vector_search(query: str, corpus: str = "laws", top_k: int = 5) -> list[RetrievalResult]:
    """向量检索，按余弦相似度降序返回。"""
    collection = get_rag_settings().collection_of(corpus)
    vector = embed_texts([query])[0]
    response = get_qdrant_client().query_points(
        collection_name=collection, query=vector, limit=top_k
    )
    results = [_to_result(point) for point in response.points]
    logger.debug(f"[vector_search]{collection} 命中 {len(results)} 条：{query}")

    return results


def bm25_search(
    query: str, candidates: list[RetrievalResult], top_k: int = 5
) -> list[RetrievalResult]:
    """
    对候选文本块做 BM25 关键词排序，score 为原始 BM25 分。

    只在候选池内打分，不建全量索引：rank-bm25 需要把整个语料的词频常驻内存，
    22 万条文书块会占数 GB，而单次查询的候选池（默认 50 条）代价可忽略。
    代价是 IDF 只在池内统计，但条号、罪名、法律名这类精确命中依然是有效信号。
    """
    if not candidates:
        return []

    corpus = [_tokenize(_lexical_text(result)) for result in candidates]
    scores = BM25Okapi(corpus).get_scores(_tokenize(query))

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

    return [
        RetrievalResult(chunk=result.chunk, score=float(score))
        for result, score in ranked[:top_k]
    ]


def _rerank(
    query: str, candidates: list[RetrievalResult], top_k: int
) -> list[RetrievalResult]:
    """
    交叉编码器重排，score 为重排相关度。

    重排是增益项而非必需项：接口异常时退回融合结果的前 top_k，
    不让整个检索失败。
    """
    model = get_rag_settings().retrieval.rerank_model
    payload = {
        "model": model,
        "input": {
            "query": query,
            "documents": [_lexical_text(result) for result in candidates],
        },
        "parameters": {"return_documents": False},
    }
    headers = {
        "Authorization": f"Bearer {get_settings().llm.api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            RERANK_URL, headers=headers, json=payload, timeout=RERANK_TIMEOUT
        )
        response.raise_for_status()
        items = response.json()["output"]["results"]
    except Exception as e:
        # 异常文本可能回显含 key 的头部/URL，入日志前脱敏（空 key 不可 replace）
        api_key = get_settings().llm.api_key
        redacted = str(e).replace(api_key, "***") if api_key else str(e)
        logger.warning(
            f"[_rerank]{model} 调用失败（{type(e).__name__}: {redacted}），退回融合结果"
        )
        return candidates[:top_k]

    reranked = [
        RetrievalResult(chunk=candidates[item["index"]].chunk, score=float(item["relevance_score"]))
        for item in items
    ]
    reranked.sort(key=lambda result: result.score, reverse=True)
    logger.debug(f"[_rerank]{model} 重排 {len(reranked)} 条")

    return reranked[:top_k]


def _rrf_fuse(
    candidates: list[RetrievalResult],
    lexical: list[RetrievalResult],
    vector_weight: float,
    bm25_weight: float,
    k: int = RRF_K,
) -> list[RetrievalResult]:
    """
    RRF（Reciprocal Rank Fusion）加权融合。

    只用两路的名次、不用原始分数：BM25 的 IDF 在本候选池内统计，分数尺度不可靠
    （排序仍可用）；余弦分经 min-max 归一化也容易被单个高分候选拉伸。名次没有
    这两个问题。两路权重沿用 configs/rag.yaml 的 vector_weight / bm25_weight。
    """
    vector_rank = {item.chunk.chunk_id: rank for rank, item in enumerate(candidates, 1)}
    bm25_rank = {item.chunk.chunk_id: rank for rank, item in enumerate(lexical, 1)}

    scores = {
        item.chunk.chunk_id: vector_weight / (k + vector_rank[item.chunk.chunk_id])
        + bm25_weight / (k + bm25_rank[item.chunk.chunk_id])
        for item in candidates
    }

    fused = [
        RetrievalResult(chunk=item.chunk, score=scores[item.chunk.chunk_id])
        for item in candidates
    ]
    fused.sort(key=lambda result: result.score, reverse=True)

    return fused


def hybrid_search(
    query: str, corpus: str = "laws", top_k: int | None = None
) -> list[RetrievalResult]:
    """
    混合检索：向量召回候选池 → RRF 融合 → 重排。

    top_k 为 None 时取 configs/rag.yaml 的 retrieval.top_k。
    """
    retrieval = get_rag_settings().retrieval
    top_k = top_k or retrieval.top_k
    pool_size = max(top_k * CANDIDATE_MULTIPLIER, MIN_CANDIDATE_POOL)

    candidates = vector_search(query, corpus, pool_size)
    if not candidates:
        logger.warning(f"[hybrid_search]{corpus} 未召回到任何候选：{query}")
        return []

    # BM25 只在候选池内算，取它的名次参与融合，不用它的原始分数
    lexical = bm25_search(query, candidates, len(candidates))
    fused = _rrf_fuse(
        candidates, lexical, retrieval.vector_weight, retrieval.bm25_weight
    )

    if retrieval.rerank:
        # 只重排前 RERANK_CANDIDATES 条：重排分与输入顺序无关，送全池只是多花钱
        return _rerank(query, fused[:RERANK_CANDIDATES], top_k)

    return fused[:top_k]


if __name__ == "__main__":
    for item in hybrid_search("故意杀人罪的量刑标准", corpus="laws"):
        print(round(item.score, 4), item.chunk.metadata.get("name"))
        print("   ", item.chunk.content[:80].replace("\n", ""))