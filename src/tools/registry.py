"""收集所有工具：本地知识库 Function-Calling + 元典 MCP 外部检索。

未配置 YUANDIAN_MCP_TOKEN 或元典熔断/不可用时，``get_tools`` 只返回本地工具，
Agent 链路不中断，自动降级为纯本地知识库。
"""

from langchain_core.tools import BaseTool

from src.core.logger import get_logger
from src.tools.knowledge_tool import search_knowledge_base
from src.tools.mcp_yuandian import build_yuandian_tools

logger = get_logger(__name__)

# 本地进程内工具始终可用
local_tools: list[BaseTool] = [search_knowledge_base]


def get_tools(include_yuandian: bool = True) -> list[BaseTool]:
    """组装 Agent 可用工具列表。

    include_yuandian 为 False 时不注册元典 MCP 工具，只返回本地工具，用于降级或测试。
    """

    tools = list(local_tools)

    if include_yuandian:
        try:
            tools.extend(build_yuandian_tools())
        except Exception as e:
            # 注册表层面兜底：MCP 侧任何意外都不影响本地工具装配
            logger.warning(f"[registry] 元典工具装配失败，降级纯本地：{e}")

    logger.info(f"[registry] 共注册 {len(tools)} 个工具（本地 {len(local_tools)}）")
    return tools
