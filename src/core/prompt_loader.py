"""读取 prompts/ 下的 md 提示词，返回原始文本，带 mtime 缓存。"""

import os
import re

from src.core.logger import get_logger
from src.core.paths import PROMPTS_DIR

logger = get_logger(__name__)

# 按 mtime 失效而非常驻：提示词文件改动后无需重启进程即可生效
_PROMPT_CACHE: dict[str, tuple[float, str]] = {}

# 匹配 "## system" / "## user" 这样的二级标题分段
_SECTION_PATTERN = re.compile(r"^##\s+(\w+)\s*$", re.MULTILINE)


def load_prompt(name: str) -> str:
    """按名称加载 prompts/<name>.md 提示词（不含扩展名），按 mtime 缓存。"""
    prompt_path = PROMPTS_DIR / f"{name}.md"

    try:
        mtime = os.path.getmtime(prompt_path)
        cached = _PROMPT_CACHE.get(str(prompt_path))
        if cached and cached[0] == mtime:
            return cached[1]

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        _PROMPT_CACHE[str(prompt_path)] = (mtime, content)
        return content
    except FileNotFoundError:
        logger.error(f"[{name}]提示词文件不存在：{prompt_path}")
        raise
    except Exception as e:
        logger.error(f"[{name}]读取提示词出错，{str(e)}")
        raise


def load_prompt_sections(name: str) -> dict[str, str]:
    """按 ``## 段名`` 切分提示词，返回 ``{段名: 段内容}``。

    约定 prompts/*.md 的结构：``# 标题`` 下用 ``## system``、``## user``
    分段；段内容剥离首尾空白。各节点拿 system/user 两段直接组装消息，
    缺失的段为空串。
    """
    content = load_prompt(name)
    matches = list(_SECTION_PATTERN.finditer(content))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        section_name = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        sections[section_name] = content[start:end].strip()

    return sections


if __name__ == "__main__":
    print(load_prompt("chat_answer"))
    print(load_prompt_sections("intent_recognize").keys())