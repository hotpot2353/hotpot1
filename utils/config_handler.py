"""
yaml
k: v
"""
import os
import yaml
from dotenv import load_dotenv
from utils.path_tool import get_abs_path

# 加载 .env 文件（项目根目录）
load_dotenv(get_abs_path(".env"))

#用于加载 RAG（检索增强生成）相关的 YAML 配置文件
#yaml.load 把这堆文本，转换成 Python 认识的字典
def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#用于加载 Chroma 向量数据库的配置文件
def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#用于加载提示词（prompts）配置文件
def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#用于加载 Agent（智能体）配置文件
def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

#调用 load_rag_config() 函数（使用默认参数），加载 RAG 配置，并将返回的字典赋值给全局变量 rag_conf。该行在模块导入时即执行
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


def get_env_var(key: str, default: str = "") -> str:
    """从环境变量获取配置值"""
    return os.getenv(key, default)


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
