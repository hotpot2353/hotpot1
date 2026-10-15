"""法条 / 类案溯源折叠面板：展示 final_result 中的来源与校验信息。"""

import streamlit as st

# 单个法条/类案片段在面板中的正文截断长度，保持折叠区紧凑
_CONTENT_CAP = 800


def render_source_panel(result: dict) -> None:
    """
    渲染单轮结果的溯源面板（折叠区，不喧宾夺主）。

    result 为 result_combine_node 输出的 final_result。
    """
    sources = result.get("sources") or {}
    laws = sources.get("laws", [])
    cases = sources.get("cases", [])
    # 内部任务类型映射为中文标签，未知类型原样展示
    task_label = {
        "report": "报告",
        "simple_chat": "问答",
        "rejected": "非刑事拒绝",
    }.get(result.get("task_type"), result.get("task_type"))
    badges = [
        f"类型：{task_label}",
        f"来源：{result.get('source_tag', '-')}",
        f"法条 {len(laws)} 条、类案 {len(cases)} 条",
    ]
    if result.get("human_action"):
        badges.append(f"人工复核：{result['human_action']}")
    st.caption("｜".join(badges))
    citation = result.get("citation_check")
    if citation:
        with st.expander("引用幻觉校验"):
            st.markdown(citation)

    with st.expander(f"引用法条（{len(laws)}）"):
        _render_chunks(laws, empty_hint="本轮未引用法条")

    with st.expander(f"参考类案（{len(cases)}）"):
        _render_chunks(cases, empty_hint="本轮未参考类案")


def _render_chunks(chunks: list[dict], *, empty_hint: str) -> None:
    """平铺渲染片段列表（Streamlit expander 不可嵌套，用分隔线分条）。"""
    if not chunks:
        st.caption(empty_hint)
        return
    for index, chunk in enumerate(chunks, 1):
        name = chunk.get("name") or chunk.get("chunk_id", "未命名片段")
        score = chunk.get("score")
        header = f"**{index}. {name}**"
        if score is not None:
            header += f"（相关度 {score:.3f}）"
        st.markdown(header)
        content = chunk.get("content", "")
        if len(content) > _CONTENT_CAP:
            content = content[:_CONTENT_CAP] + "……"
        st.markdown(content)
        if chunk.get("source"):
            st.caption(f"来源：{chunk['source']}")
        if index < len(chunks):
            st.divider()
