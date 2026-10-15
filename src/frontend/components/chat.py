"""聊天对话气泡：助手消息内联附带溯源面板。"""

import streamlit as st

from src.frontend.components.source_panel import render_source_panel


def render_chat(messages: list[dict]) -> None:
    """渲染全部聊天历史；消息含 result（final_result dict）时内联溯源面板。"""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("result"):
                render_source_panel(message["result"])
