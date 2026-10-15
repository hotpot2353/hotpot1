"""命令行 Demo 入口：不启动前端也能跑完整 Agent 链路（含 HITL 交互）。

用法：
    python -m src.main "怎么去自首"
    python -m src.main "给我一份完整的刑事辩护分析报告" --mode defend_analysis
    python -m src.main "审查这份文书" --mode doc_review --attach ./案情.docx --no-hitl
    python -m src.main "出具法律评估：正当防卫如何认定" --auto-approve
"""

import argparse
import sys
import uuid
from pathlib import Path

from src.core.file_loader import load_file
from src.core.logger import get_logger
from src.graph.builder import get_graph
from langgraph.types import Command

logger = get_logger(__name__)

VALID_MODES = ("accuse_analysis", "defend_analysis", "doc_review", "consult")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="刑事法律 Agent 命令行 Demo")
    parser.add_argument("query", help="用户问题")
    parser.add_argument(
        "--mode",
        default="consult",
        choices=VALID_MODES,
        help="会话固定业务模块（同一会话不可切换），默认综合咨询",
    )
    parser.add_argument(
        "--attach",
        action="append",
        default=[],
        metavar="PATH",
        help="附件路径，可重复传入（支持 txt/pdf/docx）",
    )
    parser.add_argument(
        "--hitl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否开启人工复核；不传则用 configs/config.yaml 的全局开关",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="HITL 暂停时自动批准（不进入交互，便于脚本/演示）",
    )
    parser.add_argument("--thread", default=None, help="自定义 thread_id")
    return parser.parse_args()


def _load_attachments(paths: list[str]) -> list[dict]:
    """把附件路径解析为 state 需要的 [{filename, content}]。"""
    files = []
    for raw in paths:
        path = Path(raw)
        try:
            content = load_file(path)
        except Exception as e:
            print(f"[警告] 附件 {path} 解析失败，已跳过：{e}")
            continue
        files.append({"filename": path.name, "content": content})
        print(f"[附件] 已加载 {path.name}（{len(content)} 字）")
    return files


def _ask_human_decision() -> dict:
    """终端交互收集人工决策：批准 / 驳回（驳回需填意见）。"""
    while True:
        action = input("人工复核 [a]批准  [r]驳回重写（默认 a）：").strip().lower()
        if action in ("", "a", "approve"):
            return {"action": "approve", "comment": "", "edited_content": ""}
        if action in ("r", "reject"):
            comment = input("请输入驳回意见（将据此重新生成报告）：").strip()
            return {"action": "reject", "comment": comment or "请按要求重新完善报告",
                    "edited_content": ""}
        print("无效输入，请重新选择。")


def _print_result(state: dict) -> None:
    """打印最终结果。"""
    result = state.get("final_result") or {}
    print("\n" + "=" * 60)
    print(state.get("final_answer", "（无回答）"))
    print("=" * 60)

    sources = result.get("sources", {})
    print(f"\n[溯源] 来源={result.get('source_tag', '-')} "
          f"法条 {len(sources.get('laws', []))} 条、"
          f"类案 {len(sources.get('cases', []))} 条")
    if result.get("human_action"):
        print(f"[人工复核] {result['human_action']}")
    citation = result.get("citation_check")
    if citation:
        first_line = citation.splitlines()[0] if citation else ""
        print(f"[引用校验] {first_line}")


def main() -> int:
    args = _parse_args()

    app = get_graph()
    config = {"configurable": {"thread_id": args.thread or f"cli-{uuid.uuid4().hex[:8]}"}}
    initial_state: dict = {
        "fixed_input_mode": args.mode,
        "query": args.query,
        "attach_files": _load_attachments(args.attach),
    }
    if args.hitl is not None:
        initial_state["human_review_enable"] = args.hitl

    state = app.invoke(initial_state, config)

    # HITL 循环：暂停 → 交互/自动批准 → resume，直到图到 END
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print("\n" + "-" * 60)
        print(f"[人工复核暂停] 报告初稿 {len(payload.get('draft', ''))} 字")
        if args.auto_approve:
            decision = {"action": "approve", "comment": "", "edited_content": ""}
            print("自动批准（--auto-approve）")
        else:
            decision = _ask_human_decision()
        state = app.invoke(Command(resume=decision), config)

    _print_result(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
