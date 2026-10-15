"""统一绝对路径：PROJECT_ROOT、DATA_DIR、PROMPTS_DIR 等。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT / "data"
# 法条原始数据：data/raws/law_library.jsonl
RAWS_DIR = DATA_DIR / "raws"
# 裁判文书原始数据：data/cases/data_{train,valid,test}.json
CASES_DIR = DATA_DIR / "cases"
PROCESSED_DIR = DATA_DIR / "processed"
QDRANT_DIR = DATA_DIR / "qdrant"
LOGS_DIR = PROJECT_ROOT / "logs"


def get_project_root() -> Path:
    """返回工程根目录绝对路径。"""
    return PROJECT_ROOT


def get_abs_path(relative_path: str) -> Path:
    """将相对工程根目录的路径（如 "configs/config.yaml"）转为绝对路径。"""
    return PROJECT_ROOT / relative_path


if __name__ == "__main__":
    print(get_abs_path("configs/config.yaml"))