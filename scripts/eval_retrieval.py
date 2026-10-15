"""检索评测：用标注 query 跑 hybrid_search，输出检索层三项指标 + MRR。

三项主指标（均在 top_k 条结果上计算，条文级口径）：
  Recall@k         标注的应命中条文有多少出现在 top_k（item 级，宏平均）
  Precision@k      top_k 个位置里有多少是标注条文（未标注位置按不相关计）
  Context Prec.@k  排序感知精确率：相关条文排得越靠前越高，惩罚"夹在噪声后面"
另有 Hit Rate@k（query 级，命中任一条即算 1）与 MRR@k 作辅助信号。

标注只判"应命中的条文是否出现在 top_k 里"，不判答案质量，所以条目少也能
说明问题。指标掉了就说明检索链路（召回 / 融合 / 重排）有退化。

一条 query 可标注多个可接受条文：问"量刑"时罪名的定义条与处罚条都算命中
（受贿 385 定义 + 386 处罚，贪污 382 定义 + 383 处罚），只标定义条会把
模型给出的正确处罚条判成未命中。

口径细节：
- 同一 条文被切成多个 chunk 时只算一次（按完整条文名去重，重复位置视为不相关）；
- top_k 未返回满 k 条时，缺失位置按不相关计（少返回等同漏检）；
- Context Precision 用标注相关性直接计算，不调用 LLM 做相关性评判：
  CP@k = mean over queries of Σ(P@i · rel(i)) / |relevant retrieved|，
  只有一个相关条文时 CP@k = 1/rank，与 MRR 同源；多标注条文时比 MRR 信息更多。

用法：
    python -m scripts.eval_retrieval              # 线上配置（带重排）
    python -m scripts.eval_retrieval --no-rerank  # 只看召回 + RRF 融合
    python -m scripts.eval_retrieval -k 3
    python -m scripts.eval_retrieval --rerank-model qwen3-rerank  # A/B 重排模型
"""

import argparse
import dataclasses
import re
import sys
from pathlib import Path

# 允许直接 python scripts/eval_retrieval.py 运行，不必加 -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag import retriever  # noqa: E402

# 评测门槛：低于它说明检索有明显退化
PASS_RECALL: float = 0.8

# 标注集：(query, 法律名关键词, 可接受的条文)
# 量刑类问题把罪名的定义条与处罚条都列上，命中任一条即算命中。
# 交通肇事、诈骗此前排在 5 名之外，留作回归信号。
EVAL_SET: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("故意杀人罪的量刑标准", "刑法", ("第二百三十二条",)),
    ("盗窃罪的量刑标准", "刑法", ("第二百六十四条",)),
    ("抢劫罪的量刑标准", "刑法", ("第二百六十三条",)),
    ("正当防卫如何认定", "刑法", ("第二十条",)),
    ("交通肇事罪的构成要件", "刑法", ("第一百三十三条",)),
    ("强奸罪的量刑", "刑法", ("第二百三十六条",)),
    ("诈骗罪的量刑标准", "刑法", ("第二百六十六条",)),
    # 386 是受贿的处罚条，不标它会把它判成未命中
    ("受贿罪的量刑标准", "刑法", ("第三百八十五条", "第三百八十六条")),
    ("非法拘禁罪", "刑法", ("第二百三十八条",)),
    ("故意伤害罪的量刑", "刑法", ("第二百三十四条",)),
    ("绑架罪的量刑", "刑法", ("第二百三十九条",)),
    ("职务侵占罪的量刑", "刑法", ("第二百七十一条",)),
    ("寻衅滋事罪的认定", "刑法", ("第二百九十三条",)),
    ("贪污罪的量刑", "刑法", ("第三百八十二条", "第三百八十三条")),
)

# 条文号：只关心"第X条"，条文名里的"之一"等后缀不影响比对
_ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百零]+条")


def _article_of(name: str) -> str:
    """从法条 name 中取出条文号，取不到返回空串。"""
    match = _ARTICLE_PATTERN.search(name)

    return match.group() if match else ""


def _disable_rerank() -> None:
    """关闭重排：替换 retriever 读配置的函数，避免改全局配置。"""
    original = retriever.get_rag_settings

    def without_rerank():
        settings = original()

        return dataclasses.replace(
            settings,
            retrieval=dataclasses.replace(settings.retrieval, rerank=False),
        )

    retriever.get_rag_settings = without_rerank


def _use_rerank_model(model: str) -> None:
    """换用指定重排模型：同样替换读配置的函数，避免改全局配置。"""
    original = retriever.get_rag_settings

    def with_model():
        settings = original()

        return dataclasses.replace(
            settings,
            retrieval=dataclasses.replace(settings.retrieval, rerank_model=model),
        )

    retriever.get_rag_settings = with_model


@dataclasses.dataclass
class QueryScore:
    """单条 query 的检索评分明细。"""

    query: str
    law: str
    relevant: tuple[str, ...]
    # 命中标注条文 → 首次出现名次（名次从 1 开始）
    hit_ranks: dict[str, int]
    # top_k 每个位置的相关性 0/1；同一条文的重复 chunk 只在首次出现记 1
    relevance: list[int]
    returned: int  # 实际返回条数（可能不足 k）

    @property
    def hit(self) -> bool:
        return bool(self.hit_ranks)

    @property
    def best_rank(self) -> int | None:
        return min(self.hit_ranks.values()) if self.hit_ranks else None

    @property
    def recall(self) -> float:
        """item 级 recall：命中的标注条文 / 全部标注条文。"""
        return len(self.hit_ranks) / len(self.relevant)

    @property
    def precision(self) -> float:
        """Precision@k：相关位置数 / k（缺失位置按不相关，分母恒为 k）。"""
        k = len(self.relevance)

        return sum(self.relevance) / k if k else 0.0

    @property
    def context_precision(self) -> float:
        """
        排序感知精确率（RAGAS 口径，标注相关性替代 LLM 评判）

        CP@k = Σ_{i=1..k} P@i·rel(i) / 相关位置数；无相关位置时记 0。
        """
        total_rel = sum(self.relevance)
        if total_rel == 0:
            return 0.0

        acc = 0.0
        running_rel = 0
        for i, rel in enumerate(self.relevance, 1):
            if rel:
                running_rel += 1
                acc += running_rel / i  # 该位置的 P@i

        return acc / total_rel

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.best_rank if self.best_rank else 0.0


def _score_query(
    query: str, law: str, articles: tuple[str, ...], top_k: int
) -> QueryScore:
    """
    跑一次 hybrid_search 并按标注条文打分。

    law 为法律名关键词（如 "刑法"）；articles 为可接受的条文号，命中任一条即算命中；
    top_k 为取前多少条。返回该 query 的评分明细。
    """
    results = retriever.hybrid_search(query, corpus="laws", top_k=top_k)

    hit_ranks: dict[str, int] = {}
    relevance: list[int] = []

    for rank, result in enumerate(results, 1):
        name = str(result.chunk.metadata.get("name", ""))
        article = _article_of(name)
        # 多部法律条文号会重复，必须同时限定法律名；
        # 同一条文的多个 chunk 只在首次出现时记相关
        if law in name and article in articles and article not in hit_ranks:
            hit_ranks[article] = rank
            relevance.append(1)
        else:
            relevance.append(0)

    # 不足 top_k 的缺失位置按不相关补齐
    relevance.extend([0] * (top_k - len(relevance)))

    return QueryScore(
        query=query,
        law=law,
        relevant=articles,
        hit_ranks=hit_ranks,
        relevance=relevance,
        returned=len(results),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检索评测：Recall@k / Precision@k / Context Precision + MRR"
    )
    parser.add_argument("-k", type=int, default=5, help="评测的 top_k，默认 5")
    parser.add_argument("--no-rerank", action="store_true", help="关闭交叉编码器重排")
    parser.add_argument("--rerank-model", help="覆盖重排模型，用于 A/B 对比")
    args = parser.parse_args()

    if args.no_rerank:
        _disable_rerank()
    elif args.rerank_model:
        _use_rerank_model(args.rerank_model)

    scores: list[QueryScore] = []
    short_returns: list[str] = []

    for query, law, articles in EVAL_SET:
        score = _score_query(query, law, articles, args.k)
        scores.append(score)
        if score.returned < args.k:
            short_returns.append(f"{query}({score.returned}/{args.k})")

        if score.hit:
            hit_articles = " / ".join(
                article for article, _ in sorted(
                    score.hit_ranks.items(), key=lambda item: item[1]
                )
            )
            extra = f"  P@k={score.precision:.2f} CP@k={score.context_precision:.2f}"
            print(
                f"[命中] rank={score.best_rank}  {query} → 《{law}》{hit_articles}{extra}"
            )
        else:
            print(f"[未中] rank=-  {query} → 《{law}》{' / '.join(articles)}")

    total = len(scores)
    k = args.k

    # ---- 汇总 ----
    hits = sum(1 for s in scores if s.hit)                       # Hit Rate 分子
    hit_articles = sum(len(s.hit_ranks) for s in scores)         # item 级 recall 分子
    all_articles = sum(len(s.relevant) for s in scores)          # item 级 recall 分母
    relevant_positions = sum(sum(s.relevance) for s in scores)   # precision 分子

    hit_rate = hits / total
    recall = hit_articles / all_articles
    precision = relevant_positions / (total * k)
    context_precision = sum(s.context_precision for s in scores) / total
    mrr = sum(s.reciprocal_rank for s in scores) / total
    passed = hit_rate >= PASS_RECALL

    print("-" * 72)
    rerank_label = "关闭" if args.no_rerank else retriever.get_rag_settings().retrieval.rerank_model
    print(f"重排：{rerank_label} | top_k={k} | 样本 {total}（标注条文 {all_articles} 条）")
    print(f"Hit Rate@{k}        = {hit_rate:.2f} ({hits}/{total})  "
          f"门槛 {PASS_RECALL:.2f} → {'通过' if passed else '未通过'}")
    print(f"Recall@{k}（条文级）= {recall:.2f} ({hit_articles}/{all_articles})")
    print(f"Precision@{k}       = {precision:.2f} ({relevant_positions}/{total * k})")
    print(f"Context Prec.@{k}   = {context_precision:.3f}")
    print(f"MRR@{k}              = {mrr:.3f}")

    missed = [s.query for s in scores if not s.hit]
    if missed:
        print("未命中：" + "、".join(missed))
    if short_returns:
        print("返回不足 top_k：" + "、".join(short_returns))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
