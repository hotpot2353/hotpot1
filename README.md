# 🗺️ Hotpot1 - AI 智能旅行规划助手

基于 LangChain ReAct Agent + RAG 的智能旅行规划系统，通过 Streamlit 提供 Web 界面，能够根据用户需求自动生成完整的旅行方案并导出 PDF 报告。

##  功能特性

- **智能对话规划** — 通过多轮对话理解用户需求，动态调用工具获取实时信息
- **RAG 知识库检索** — 基于 ChromaDB 向量检索，提供交通、景点、美食等攻略参考
- **实时天气查询** — 接入高德天气 API，获取目的地天气预报
- **POI 与路线规划** — 高德地图 POI 搜索、周边推荐、驾车路线规划
- **个性化推荐** — 基于用户历史旅行记录，提供定制化旅行建议
- **PDF 报告生成** — 自动生成结构化的旅行方案报告，支持中文排版

## 🏗️ 项目结构

```
hotpot1/
├── agent/              # Agent 核心逻辑
│   ├── react_agent.py  # ReAct Agent 定义与执行
│   └── tools/          # 工具集与中间件
│       ├── agent_tools.py  # 7 个核心工具
│       └── middleware.py     # Agent 中间件
├── app.py              # Streamlit Web 应用入口
├── config/             # 配置文件
│   ├── agent.yml       # Agent 配置
│   ├── rag.yml         # RAG 模型配置
│   └── chroma.yml      # 向量库配置
├── data/               # 数据目录
│   ├── external/       # 外部数据（旅行记录 CSV）
│   ├── raw/            # 原始文档（攻略 TXT）
│   └── reports/        # 生成的 PDF 报告
├── model/              # 模型工厂（通义千问 + 嵌入模型）
├── prompts/            # 系统提示词模板
── rag/                # RAG 检索服务
│   ├── rag_service.py  # RAG 摘要服务
│   ── vector_store.py # ChromaDB 向量库
├── utils/              # 工具函数（配置加载等）
├── requirements.txt    # Python 依赖
── .env                # 环境变量（API 密钥，不提交）
```

##  快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/<your-username>/hotpot1.git
cd hotpot1

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

在项目根目录创建 `.env` 文件：

```env
AMAP_API_KEY=你的高德地图API密钥
DASHSCOPE_API_KEY=你的通义千问API密钥
```

> 高德 API 密钥申请：https://console.amap.com  
> 通义千问 API 密钥申请：https://dashscope.console.aliyun.com

### 3. 准备知识库数据

将旅游攻略文档（`.txt` 格式）放入 `data/raw/` 目录，系统会自动加载并构建向量索引。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

## 🛠️ 核心工具

| 工具 | 说明 |
|------|------|
| `search_travel_guide` | RAG 检索旅游攻略（交通/景点/美食） |
| `get_weather` | 高德天气 API 查询目的地天气 |
| `search_poi` | 高德 POI 搜索（景点、酒店、餐厅等） |
| `get_route_info` | 高德驾车路线规划 |
| `search_nearby` | 周边地点搜索 |
| `get_user_travel_prefs` | 查询用户历史旅行偏好 |
| `fill_context_for_report` | 触发 PDF 报告生成 |

## 📐 技术架构

```
用户输入 → Streamlit UI → ReAct Agent (LangChain)
                              ├── LLM: 通义千问 (qwen3-max)
                              ├── Tools: 高德 API + RAG + CSV
                              └── RAG: ChromaDB + text-embedding-v4
                                    ↓
                              PDF 报告生成 (fpdf2 + 黑体)
```

## 📦 主要依赖

- **Streamlit** — Web 界面
- **LangChain / LangGraph** — Agent 框架
- **ChromaDB** — 向量数据库
- **DashScope** — 通义千问 LLM + 嵌入模型
- **高德地图 API** — 天气、POI、路线
- **fpdf2** — PDF 报告生成

## 📝 注意事项

- `.env` 文件包含 API 密钥，已加入 `.gitignore`，请勿提交到仓库
- 首次运行时 RAG 模块会自动构建向量索引，可能需要一些时间
- PDF 报告使用 Windows 系统字体 `simhei.ttf`（黑体），确保中文正常显示
- 旅行记录数据存储在 `data/external/travel_records.csv`（GBK 编码）

