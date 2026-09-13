import os
import re
import sys
import time
from datetime import datetime

# 确保用户站点包在路径中
user_site = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python312", "site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import streamlit as st
from agent.react_agent import TravelAgent
from fpdf import FPDF
from utils.path_tool import get_abs_path


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("SimHei", "", "C:/Windows/Fonts/simhei.ttf")

    def header(self):
        self.set_font("SimHei", size=10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "智游通旅行规划助手 - 报告", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("SimHei", size=8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")


def extract_final_report(content: str) -> str:
    """从完整回复中提取最终报告部分（去掉思考过程）"""
    # 最终报告通常以 # 开头的标题行开始，如 "# 【吉林】3天2晚深度游攻略"
    lines = content.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") and ("【" in stripped or "[" in stripped):
            return "\n".join(lines[i:])
    return content


def save_report_pdf(title, question, content, filepath):
    pdf = ReportPDF()
    pdf.add_page()

    # 标题
    pdf.set_font("SimHei", size=18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, title, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 元信息
    pdf.set_font("SimHei", size=10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"用户提问: {question}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # 正文
    pdf.set_font("SimHei", size=11)
    pdf.set_text_color(30, 30, 30)

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("# ").strip()
            size = [16, 14, 13, 12, 11, 11][min(level - 1, 5)]
            pdf.set_font("SimHei", size=size)
            pdf.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("SimHei", size=11)
            pdf.cell(5)
            pdf.multi_cell(0, 7, f"  {line[2:]}", new_x="LMARGIN", new_y="NEXT")
        elif re.match(r"^\d+\.\s", line):
            pdf.set_font("SimHei", size=11)
            pdf.cell(5)
            pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("SimHei", size=11)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            text = re.sub(r"\\~", "~", text)
            pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")

    pdf.output(filepath)

# 标题
st.title("智游通旅行规划助手")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = TravelAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input("请输入您的旅行需求，如：帮我规划北京3日游")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 构建完整对话历史传给 agent
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["message"]]

    response_messages = []
    with st.spinner("旅行助手规划中..."):
        res_stream = st.session_state["agent"].execute_stream(messages)

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        full_response = "".join(response_messages)
        st.session_state["message"].append({"role": "assistant", "content": full_response})

        # 保存报告到 PDF 文件（只保存最终报告，去掉思考过程）
        report_dir = get_abs_path("data/reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_content = full_response.replace("\\~", "~")
        final_report = extract_final_report(clean_content)

        pdf_file = os.path.join(report_dir, f"report_{timestamp}.pdf")
        save_report_pdf("旅行规划报告", prompt, final_report, pdf_file)

        st.toast(f"报告已保存为 PDF", icon="✅")

        st.rerun()