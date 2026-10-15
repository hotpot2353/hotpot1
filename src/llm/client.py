"""通义千问 LLM 客户端。"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_llm(thinking: bool = False, max_tokens: int | None = None) -> ChatOpenAI:
    """构建通义千问客户端（DashScope OpenAI 兼容模式），按 (thinking, max_tokens) 缓存实例。

    thinking 默认显式关闭以降本提速（qwen3.5+ 服务端默认开启），业务分析、
    报告生成等需推理的节点传 True；max_tokens 为 None 时取 config.yaml 的
    llm.max_tokens，长文场景可显式传入更大上限。
    """
    settings = get_settings()
    token_limit = max_tokens or settings.llm.max_tokens
    logger.info(
        f"初始化 LLM：{settings.llm.model}（thinking={thinking}，"
        f"max_tokens={token_limit}）"
    )

    return ChatOpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        max_tokens=token_limit,
        # DashScope 私有参数，经 extra_body 透传
        extra_body={"enable_thinking": thinking},
    )


if __name__ == "__main__":
    llm = get_llm()
    print(llm.invoke("只回复两个字：收到").content)