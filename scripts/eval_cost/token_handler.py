"""TokenUsageHandler：旁路统计每次 LLM 调用的 token 用量，按 LangGraph 节点分桶。

零侵入生产代码：跑图时经 ``config={"callbacks": [handler]}`` 挂载；
节点外直接调用节点函数（E2）时，通过给模型实例 ``model_copy(callbacks=...)``
注入同一个 handler。

用量读取双通路：langchain-openai 在 ``AIMessage.usage_metadata`` 做了标准化，
DashScope OpenAI 兼容模式也在 ``LLMResult.llm_output.token_usage`` 返回原生
OpenAI 字段，两处都取以兼容模型/SDK 版本差异。thinking 模型的推理 token
包含在 output_tokens 内，单独抽一列展示。
"""

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# 中英混合法律文本的经验折算率：1 token ≈ 1.5 字符。
# 仅用于 E3/E4 估算项；实测项全部读接口返回的真实 token 数，不做折算。
CHARS_PER_TOKEN: float = 1.5


@dataclass
class UsageRecord:
    """单次 LLM 调用的用量明细。"""

    node: str
    model: str
    input_tokens: int
    output_tokens: int
    # thinking 模型的推理 token（已含在 output_tokens 内）；接口未返回则 None
    reasoning_tokens: int | None


class TokenUsageHandler(BaseCallbackHandler):
    """收集 LLM 调用流水，提供按节点聚合与总量查询。"""

    def __init__(self, label: str | None = None) -> None:
        # 节点外直调时的手动标签；跑图期间有 langgraph_node tag，标签让位
        self.label = label
        self._node_of_run: dict[str, str] = {}
        self._model_of_run: dict[str, str] = {}
        self.records: list[UsageRecord] = []

    def reset(self, label: str | None = None) -> None:
        """清空上一组实验的流水，开始下一组统计。"""
        self.label = label
        self._node_of_run.clear()
        self._model_of_run.clear()
        self.records.clear()

    def _grab(
        self,
        run_id: Any,
        serialized: dict | None,
        tags: list[str] | None,
        metadata: dict | None,
    ) -> None:
        # 优先级：LangGraph 注入的节点 tag > metadata > 手动 label
        node: str | None = None
        for tag in tags or []:
            if isinstance(tag, str) and tag.startswith("langgraph_node:"):
                node = tag.split(":", 1)[1]
                break
        if not node and metadata:
            node = metadata.get("langgraph_node")
        if not node:
            node = self.label or "(unknown)"

        kwargs = (serialized or {}).get("kwargs") or {}
        model = kwargs.get("model") or kwargs.get("model_name") or "unknown"

        rid = str(run_id)
        self._node_of_run[rid] = node
        self._model_of_run[rid] = model

    def on_chat_model_start(
        self, serialized, messages, *, run_id, tags=None, metadata=None, **kwargs
    ) -> None:
        self._grab(run_id, serialized, tags, metadata)

    def on_llm_start(
        self, serialized, prompts, *, run_id, tags=None, metadata=None, **kwargs
    ) -> None:
        self._grab(run_id, serialized, tags, metadata)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        rid = str(run_id)
        node = self._node_of_run.get(rid, "(unknown)")
        model = self._model_of_run.get(rid, "unknown")

        input_tokens = 0
        output_tokens = 0
        reasoning: int | None = None

        # 通路一：langchain 标准化 usage_metadata
        message = None
        try:
            message = response.generations[0][0].message
        except (IndexError, AttributeError):
            message = None
        um = getattr(message, "usage_metadata", None) or {}
        if um:
            input_tokens = int(um.get("input_tokens") or 0)
            output_tokens = int(um.get("output_tokens") or 0)
            details = um.get("output_token_details") or {}
            if details.get("reasoning") is not None:
                reasoning = int(details["reasoning"])

        # 通路二：OpenAI 原生 token_usage（标准化字段缺失时兜底）
        llm_output = getattr(response, "llm_output", None) or {}
        tu = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if not input_tokens:
            input_tokens = int(tu.get("prompt_tokens") or 0)
        if not output_tokens:
            output_tokens = int(
                tu.get("completion_tokens") or tu.get("total_tokens") or 0
            )
        if reasoning is None:
            details = tu.get("completion_tokens_details") or {}
            if details.get("reasoning_tokens") is not None:
                reasoning = int(details["reasoning_tokens"])
        if model == "unknown":
            model = llm_output.get("model_name") or model

        self.records.append(
            UsageRecord(node, model, input_tokens, output_tokens, reasoning)
        )

    def by_node(self) -> dict[str, dict]:
        """按节点聚合，返回 {节点名: {model, calls, input, output, reasoning, has_reasoning}}。"""
        agg: dict[str, dict] = {}
        for r in self.records:
            row = agg.setdefault(
                r.node,
                {
                    "model": r.model,
                    "calls": 0,
                    "input": 0,
                    "output": 0,
                    "reasoning": 0,
                    "has_reasoning": False,
                },
            )
            row["calls"] += 1
            row["input"] += r.input_tokens
            row["output"] += r.output_tokens
            if r.reasoning_tokens:
                row["reasoning"] += r.reasoning_tokens
                row["has_reasoning"] = True
        return agg

    def totals(self) -> tuple[int, int]:
        """全部调用的 (input_tokens, output_tokens)。"""
        return (
            sum(r.input_tokens for r in self.records),
            sum(r.output_tokens for r in self.records),
        )

    def snapshot(self) -> dict:
        """深拷贝当前流水，供 reset 开始下一组前留档。"""
        return {
            "by_node": {k: dict(v) for k, v in self.by_node().items()},
            "totals": self.totals(),
            "calls": len(self.records),
        }
