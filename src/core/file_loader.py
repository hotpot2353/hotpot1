"""附件解析：PDF / DOCX / TXT 统一入口，返回纯文本。

具体解析器复用 src/rag/ingestion.py（LangChain loader），这里只做
"路径 → 纯文本"的薄封装，供前端上传附件后构造 state["attach_files"]。
"""

from pathlib import Path

from src.rag.ingestion import load_docx, load_pdf, load_txt


def load_file(path: str | Path) -> str:
    """按扩展名解析附件为纯文本，支持 .txt/.pdf/.docx，多页按换行拼接。"""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        documents = load_txt(path)
    elif suffix == ".pdf":
        documents = load_pdf(path)
    elif suffix == ".docx":
        documents = load_docx(path)
    elif suffix == ".doc":
        raise ValueError(f"不支持旧版 .doc：{path.name}，请先另存为 .docx")
    else:
        raise ValueError(f"不支持的文件类型：{suffix}（{path.name}）")

    return "\n".join(document.page_content for document in documents).strip()
