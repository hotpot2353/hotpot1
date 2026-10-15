"""日志初始化与获取。"""

import logging
import logging.config

import yaml

from src.core.paths import CONFIGS_DIR, LOGS_DIR, get_abs_path

_LOGGING_CONFIG = CONFIGS_DIR / "logging.yaml"
_initialized = False


def setup_logging() -> None:
    """依据 configs/logging.yaml 初始化日志（幂等，重复调用直接返回）。"""
    global _initialized
    if _initialized:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOGGING_CONFIG, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 配置里写的是相对路径，统一还原成工程内绝对路径，避免受启动目录影响
    log_file = get_abs_path(config["handlers"]["file"]["filename"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    config["handlers"]["file"]["filename"] = str(log_file)
    logging.config.dictConfig(config)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger。"""
    setup_logging()
    return logging.getLogger(name)