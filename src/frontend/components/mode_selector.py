"""新建会话业务模块选择卡片。"""

import streamlit as st

from src.frontend.session import MODE_LABELS, new_session


def render_mode_selector() -> str:
    """
    渲染业务模块选择卡片，返回生效中的业务模块。

    fixed_input_mode 会话内固定：选择与当前不同的模块时自动新建会话
    （旧历史不迁移）并重跑页面。可选模块为刑事控告、刑事辩护、
    文书审查、综合咨询。
    """
    current = st.session_state.input_mode

    st.caption("业务模块（会话内固定，切换将新建会话）")
    selected = st.radio(
        "选择业务模块",
        list(MODE_LABELS),
        format_func=lambda mode: MODE_LABELS[mode],
        index=list(MODE_LABELS).index(current),
        label_visibility="collapsed",
        key="mode_radio",
    )

    if selected != current:
        new_session(selected)
        st.toast(f"已切换到「{MODE_LABELS[selected]}」，新会话已创建")
        st.rerun()

    return selected
