"""多文件上传：txt / pdf / docx 解析为 state.attach_files。"""

import tempfile
from pathlib import Path

import streamlit as st

from src.core.file_loader import load_file

SUPPORTED_SUFFIXES = (".txt", ".pdf", ".docx")


def render_attachment_uploader() -> list[dict]:
    """
    渲染多文件上传控件，返回 state 需要的 [{filename, content}]

    解析结果按 (文件名, 大小) 缓存在 session_state，避免 Streamlit 每次
    交互重跑脚本时重复解析 PDF / DOCX；解析失败的文件只提示不中断。
    """
    uploads = st.file_uploader(
        "上传案卷附件（可选，支持 txt / pdf / docx）",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx"],
    )
    if not uploads:
        return []
    attach_files: list[dict] = []
    cache = st.session_state.attach_cache
    for upload in uploads:
        key = (upload.name, upload.size)
        if key in cache:
            attach_files.append({"filename": upload.name, "content": cache[key]})
            continue
        suffix = Path(upload.name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            st.warning(f"已跳过不支持的附件：{upload.name}（仅支持 txt / pdf / docx）")
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(upload.getvalue())
            tmp_path = tmp.name
            content = load_file(tmp_path)
        except Exception as e:  # noqa: BLE001
            st.warning(f"附件 {upload.name} 解析失败，已跳过：{e}")
            continue
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
        cache[key] = content
        attach_files.append({"filename": upload.name, "content": content})

    if attach_files:
        st.caption("已解析：" + "、".join(f["filename"] for f in attach_files))
    return attach_files
