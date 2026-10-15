"""加载 yaml/.env，暴露 get_settings()。"""

import os
from dataclasses import dataclass
from functools import lru_cache

import yaml
from dotenv import load_dotenv

from src.core.logger import get_logger
from src.core.paths import CONFIGS_DIR, PROJECT_ROOT

logger = get_logger(__name__)

# 加载工程根目录下的 .env；override=False，系统环境变量优先于 .env
load_dotenv(PROJECT_ROOT / ".env")


def load_yaml(name: str) -> dict:
    """读取 configs/ 下的 yaml 配置，空文件返回 {}。"""
    path = CONFIGS_DIR / name

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.error(f"配置文件不存在：{path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析失败：{path}，{str(e)}")
        raise


def get_env_var(key: str, default: str = "") -> str:
    """从环境变量获取配置值。"""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """读取整数型环境变量，缺失或非法时回退默认值。"""
    raw = get_env_var(key).strip()
    if not raw:
        return default

    try:
        return int(raw)
    except ValueError:
        logger.warning(f"环境变量 {key}={raw} 不是整数，回退默认值 {default}")
        return default

# 全局配置冻结为不可变对象，避免运行期被意外修改
@dataclass(frozen=True)
class HitlConfig:
    """人工复核开关。"""

    enabled: bool


@dataclass(frozen=True)
class RetrievalConfig:
    """全局检索参数。"""

    top_k: int


@dataclass(frozen=True)
class LLMConfig:
    """LLM 连接与生成参数。"""

    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class MCPConfig:
    """元典 MCP 熔断参数。"""

    failure_threshold: int
    recovery_timeout: int


@dataclass(frozen=True)
class Settings:
    """全局配置。"""

    hitl: HitlConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    mcp: MCPConfig

# lru_cache 实现单例：配置仅在首次调用时加载，之后始终返回同一实例
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存全局配置。"""
    raw = load_yaml("config.yaml")

    hitl_raw = raw.get("hitl", {})
    retrieval_raw = raw.get("retrieval", {})
    llm_raw = raw.get("llm", {})
    mcp_raw = raw.get("mcp", {})

    return Settings(
        hitl=HitlConfig(enabled=bool(hitl_raw.get("enabled", True))),
        retrieval=RetrievalConfig(top_k=int(retrieval_raw.get("top_k", 5))),
        llm=LLMConfig(
            # 密钥只从环境变量取，不落 yaml
            api_key=get_env_var("DASHSCOPE_API_KEY"),
            base_url=get_env_var("DASHSCOPE_BASE_URL"),
            # 环境变量优先于 yaml，便于部署时覆盖
            model=get_env_var("LLM_MODEL") or llm_raw.get("model", ""),
            temperature=float(llm_raw.get("temperature", 0.2)),
            max_tokens=int(llm_raw.get("max_tokens", 4096)),
        ),
        mcp=MCPConfig(
            failure_threshold=_env_int(
                "MCP_FAILURE_THRESHOLD", int(mcp_raw.get("failure_threshold", 3))
            ),
            recovery_timeout=_env_int(
                "MCP_RECOVERY_TIMEOUT", int(mcp_raw.get("recovery_timeout", 60))
            ),
        ),
    )


if __name__ == "__main__":
    print(get_settings())