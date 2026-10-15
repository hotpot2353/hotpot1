"""HITL 人工复核界面：展示初稿，收集 批准 / 修改 / 驳回 决策。"""

import streamlit as st

from src.graph.nodes import HUMAN_ACTION_LABELS


def render_hitl_panel(payload: dict, *, disabled: bool = False) -> dict | None:
    """
    渲染人工复核面板并收集决策。

    payload 为图 interrupt() 抛出的载荷，含 draft / chapter_titles /
    available_actions / instruction。三动作语义：
    approve=批准（进入引用校验）、edit=修改（以 edited_content 替换初稿）、
    reject=驳回（带 comment 意见重新生成）。
    返回 {"action", "comment", "edited_content"}，可直接作为
    Command(resume=...) 的值恢复图执行；未提交返回 None。
    disabled 在上一轮图调用仍在执行时禁用全部输入，防止重复提交。
    """
    draft = payload.get("draft", "")

    with st.chat_message("assistant"):
        st.warning("报告初稿待人工复核")
        st.markdown(payload.get("instruction", ""))
        st.markdown(draft)
    action = st.radio(
        "复核决策",
        list(HUMAN_ACTION_LABELS),
        format_func=lambda a: HUMAN_ACTION_LABELS[a],
        horizontal=True,
        key="hitl_action",
        disabled=disabled,
    )

    comment = ""
    edited_content = ""
    if action == "reject":
        comment = st.text_area(
            "驳回意见（将据此重新生成报告）",
            key="hitl_comment",
            placeholder="例如：量刑情节部分遗漏自首从宽，请补充。",
            disabled=disabled,
        )
    elif action == "edit":
        edited_content = st.text_area(
            "修改后的报告全文（提交后替换初稿）",
            value=draft,
            key="hitl_edited",
            height=320,
            disabled=disabled,
        )

    if st.button(
        "提交复核决策",
        type="primary",
        key="hitl_submit",
        disabled=disabled,
    ):
        return {"action": action, "comment": comment, "edited_content": edited_content}
    return None
