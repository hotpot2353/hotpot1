"""Token 成本评测入口与报表。

两个本项目特有的 token 节省做法：
  E1（实测）两级链路分流：simple_chat 跳过检索增强/业务分析/长报告/引用校验
  E2（实测）thinking 精准配置：仅业务分析与 draft 开思考，其余节点关闭

用法：
    python -m scripts.eval_cost.run                # 跑全部
    python -m scripts.eval_cost.run --only e2

前置：本地 Qdrant 已启动且 laws/cases 集合已入库；.env 配好 DashScope key。
全程不调用元典工具。
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval_cost.experiments import experiment_e1, experiment_e2
from scripts.eval_cost.token_handler import TokenUsageHandler

# DashScope 单价（元 / 百万 token）。占位取 qwen 主流档位，
# 出正式报告前请按控制台你所用模型（configs/config.yaml: qwen3.7-plus）的实际价目修改。
PRICE_PER_MTOK: dict[str, float] = {
    "llm_input": 0.8,
    "llm_output": 3.0,
}


def _fmt(n: int) -> str:
    return f"{n:,}"


def _print_node_table(title: str, snap: dict) -> None:
    """打印一组实验的按节点 token 明细表。"""
    print(f"\n  [{title}]")
    print(f"    {'节点':<32}{'调用':>4}{'input':>10}{'output':>10}{'推理':>9}")
    by_node = snap["by_node"]
    for node in sorted(by_node):
        row = by_node[node]
        reasoning = f"{row['reasoning']:,}" if row["has_reasoning"] else "-"
        print(
            f"    {node:<32}{row['calls']:>4}{row['input']:>10,}"
            f"{row['output']:>10,}{reasoning:>9}"
        )
    i, o = snap["totals"]
    print(f"    {'合计':<32}{snap['calls']:>4}{i:>10,}{o:>10,}")


def _print_e1(result: dict) -> None:
    print("=" * 72)
    print("E1 两级链路分流（实测）")
    print("=" * 72)
    simple, report = result["simple"], result["report"]
    print(
        f"\n  路由验证：simple query 实际路由 → {simple['routed_task_type']}"
        f"（预期 simple_chat）"
    )
    _print_node_table("A 简单问答链路", simple["usage"])
    _print_node_table("B 强制报告链路", report["usage"])

    si, so = simple["usage"]["totals"]
    ri, ro = report["usage"]["totals"]
    saved = (ri + ro) - (si + so)
    pct = saved / (ri + ro) * 100 if (ri + ro) else 0
    print(f"\n  单条 query 节省：{_fmt(saved)} token（{pct:.1f}%）")


def _print_e2(result: dict) -> tuple[int, int]:
    print("\n" + "=" * 72)
    print("E2 thinking 精准配置（单节点实测重放）")
    print("=" * 72)
    print(f"\n    {'节点/配置':<28}{'input':>9}{'output':>9}{'推理token':>10}")
    spent = [0, 0]
    for stage_key, stage_name in (("intent", "意图识别"), ("citation", "引用校验")):
        for flag_key, label in (
            ("thinking_off", f"{stage_name} thinking=关(现状)"),
            ("thinking_on", f"{stage_name} thinking=开(对照)"),
        ):
            snap = result[stage_key][flag_key]
            i, o = snap["totals"]
            rows = snap["by_node"]
            reasoning = sum(r["reasoning"] for r in rows.values())
            print(f"    {label:<28}{i:>9,}{o:>9,}{reasoning:>10,}")
            spent[0] += i
            spent[1] += o
    ii = result["intent"]["thinking_off"]["totals"]
    io_ = result["intent"]["thinking_on"]["totals"]
    co = result["citation"]["thinking_off"]["totals"]
    co2 = result["citation"]["thinking_on"]["totals"]
    waste = (io_[1] - ii[1]) + (co2[1] - co[1])
    print(f"\n  两节点若全开 thinking，每次执行多付 output 约：{_fmt(max(waste, 0))} token")
    return spent[0], spent[1]


def _print_cost_summary(llm_input: int, llm_output: int) -> None:
    print("\n" + "=" * 72)
    print("本次评测实际花费")
    print("=" * 72)
    llm_cost = (
        llm_input / 1_000_000 * PRICE_PER_MTOK["llm_input"]
        + llm_output / 1_000_000 * PRICE_PER_MTOK["llm_output"]
    )
    print(f"  LLM     input {_fmt(llm_input)} + output {_fmt(llm_output)}"
          f" → ¥{llm_cost:.4f}")
    print(f"  合计：¥{llm_cost:.4f}")
    print(f"  单价口径：input ¥{PRICE_PER_MTOK['llm_input']}、"
          f"output ¥{PRICE_PER_MTOK['llm_output']}"
          f" / 百万 token（占位，请按实际套餐改）")


def main() -> int:
    parser = argparse.ArgumentParser(description="Token 成本评测")
    parser.add_argument(
        "--only",
        default="e1,e2",
        help="逗号分隔的实验编号，默认全部：e1,e2",
    )
    args = parser.parse_args()
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}

    handler = TokenUsageHandler()
    e1_data = None

    total_llm_input = 0
    total_llm_output = 0

    # E2 重放依赖 E1 report 组跑完的 state，选了 E2 时自动补跑 E1
    if "e2" in wanted:
        wanted.add("e1")

    if "e1" in wanted:
        e1_data = experiment_e1(handler)
        _print_e1(e1_data)
        for key in ("simple", "report"):
            i, o = e1_data[key]["usage"]["totals"]
            total_llm_input += i
            total_llm_output += o

    if "e2" in wanted:
        i, o = _print_e2(experiment_e2(handler, e1_data["report"]["final_state"]))
        total_llm_input += i
        total_llm_output += o

    _print_cost_summary(total_llm_input, total_llm_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
