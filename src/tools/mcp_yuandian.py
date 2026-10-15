"""元典 MCP 原生客户端（mcp 2.2.0 streamable HTTP）+ 熔断降级 + LangChain 工具封装。

不使用 langchain-mcp-adapters：其 0.3.2 要求 mcp<2.0，与项目锁定的 mcp 2.2.0 冲突。
本模块只用 mcp 原生 SDK，做三件事：

1. ``YuandianMCPClient``：在独立线程的事件循环里持有一条长连接（initialize 只做
   一次），对同步调用方暴露 ``call_tool``；
2. 熔断：连续失败达到阈值后打开熔断，``recovery_timeout`` 秒内直接快速失败；
   冷却结束后放一个探针请求，成功则闭合、失败则重新计时；
3. ``build_yuandian_tools``：把元典侧工具按 allowlist 动态包装成 LangChain
   ``StructuredTool``，供 Agent 以 Function-Calling 方式调用。

MCP 工具不能直接读写 AgentState，熔断计数由本模块单例维护，
业务节点可通过 ``client.status()`` 读出后回写 state。
"""

import asyncio
import threading
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from src.core.config import get_env_var, get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

# 元典 streamable HTTP MCP 端点
DEFAULT_MCP_URL: str = "https://open.chineselaw.com/mcp"

# 单次工具调用超时：元典语义检索可能较慢，留足 60 秒
CALL_TIMEOUT_SECONDS: float = 60.0

# 暴露给 Agent 的刑事业务工具 allowlist（元典共 46 个工具，其余为证券合规 / 企业征信 / 合同审查 CLI 等，与刑事业务无关，不暴露以免干扰选参）。
# yuandian_hall_detect 是幻觉校验工具，留给 citation_check 节点直接用 client 调用，不注册为 Agent 工具，避免模型自行触发。
ALLOWED_MCP_TOOLS: frozenset[str] = frozenset(
    {
        "yuandian_semantic_search_legal_articles",  # 法条语义检索
        "yuandian_search_legal_articles",  # 法条关键词检索
        "yuandian_semantic_search_cases",  # 案例语义检索
        "yuandian_search_judicial_cases",  # 普通裁判案例检索
        "yuandian_search_authoritative_cases",  # 权威 / 典型案例检索
        "yuandian_get_legal_article_detail",  # 法条 / 法规详情
        "yuandian_get_case_detail",  # 案例详情
    }
)

# JSON Schema 标量类型 → Python 类型
_SCALAR_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


class YuandianMCPError(RuntimeError):
    """元典 MCP 调用异常（网络、鉴权、限流、协议错误等）。"""


class YuandianCircuitOpenError(YuandianMCPError):
    """熔断打开中：未到冷却时间，调用被直接拒绝。"""


def _model_from_schema(model_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """
    把 MCP 工具的 JSON Schema 入参定义递归转换成 pydantic 模型。

    所有非 required 字段统一给默认值 None：LangChain 绑参时未传即为 None，
    调用前会被剔除，不会把 null 发给服务端。
    """
    def resolve(field_schema: dict[str, Any], nested_name: str) -> Any:
        json_type = field_schema.get("type")

        if json_type in _SCALAR_TYPES:
            return _SCALAR_TYPES[json_type]

        if json_type == "array":
            item_schema = field_schema.get("items", {})
            if item_schema.get("type") == "object":
                item_type = _model_from_schema(f"{nested_name}_item", item_schema)
            else:
                item_type = _SCALAR_TYPES.get(item_schema.get("type"), object)
            return list[item_type]

        if json_type == "object":
            return _model_from_schema(nested_name, field_schema)

        # anyOf / 无类型声明等情况，退化为任意值
        return object

    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for field_name, field_schema in properties.items():
        python_type = resolve(field_schema, f"{model_name}_{field_name}")
        description = field_schema.get("description")
        if field_name in required:
            fields[field_name] = (python_type, Field(description=description))
        else:
            fields[field_name] = (
                Optional[python_type],
                Field(default=None, description=description),
            )

    return create_model(model_name, **fields)


class YuandianMCPClient:
    """
    元典 MCP 单连接客户端

    MCP 的 streamable HTTP 会话绑定事件循环，而 LangChain 工具是被同步调用的，
    因此在一条守护线程里跑专用事件循环，所有协程经 run_coroutine_threadsafe
    提交，initialize 握手全程只做一次。
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        failure_threshold: int,  # 连续失败多少次后打开熔断
        recovery_timeout: int,  # 熔断冷却秒数，过后放行探针请求
        call_timeout: float = CALL_TIMEOUT_SECONDS,  # 单次调用超时秒数
    ) -> None:

        self._url = url
        self._api_key = api_key
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._call_timeout = call_timeout

        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: Any = None
        self._http_client: Any = None
        self._stream_cm: Any = None
        self._tools_cache: dict[str, Any] = {}

        # 熔断状态
        self._failure_count = 0
        self._available = True
        self._opened_at = None

    # ---------- 事件循环与连接 ----------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """惰性启动守护线程事件循环，全程只启动一次。"""
        if self._loop is not None:
            return self._loop

        # 等待子线程完成事件循环创建与赋值后再返回，避免拿到 None
        ready = threading.Event()

        def run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=run_loop, name="yuandian-mcp-loop", daemon=True
        )
        thread.start()
        ready.wait(timeout=5)
        return self._loop  # type: ignore[return-value]

    def _submit(self, coro: Any, timeout: float | None = None) -> Any:
        """把协程提交到专用事件循环并同步等待结果。"""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout or self._call_timeout)

    async def _connect(self) -> None:
        """在事件循环线程内建立 HTTP 传输并完成 MCP 握手（只执行一次）。"""
        if self._session is not None:
            return

        # 延迟导入：未配置 key / 不使用元典时，import mcp 失败也不影响本地链路
        from mcp import ClientSession
        from mcp.client.streamable_http import (
            create_mcp_http_client,
            streamable_http_client,
        )

        headers = {
            # streamable HTTP 同时支持 JSON 响应与 SSE 流
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
        }
        self._http_client = create_mcp_http_client(headers=headers)
        self._stream_cm = streamable_http_client(
            self._url, http_client=self._http_client
        )
        # mcp 2.2 的传输上下文管理器解包为 (read_stream, write_stream) 二元组
        read_stream, write_stream = await self._stream_cm.__aenter__()
        self._session = ClientSession(read_stream, write_stream)
        # 手动进入会话上下文，关闭由 _invalidate_connection 统一处理
        await self._session.__aenter__()
        # MCP 握手：协商协议版本与双方能力
        result = await self._session.initialize()
        logger.info(
            f"[yuandian] MCP 连接已建立：{result.server_info.name} "
            f"v{result.server_info.version}"
        )

    # ---------- 熔断 ----------

    def _invalidate_connection(self) -> None:
        """
        失败后作废当前连接（传输可能已断）

        下次调用会重新握手；否则 session 对象虽在，底层流已关闭，
        熔断冷却后的探针也只会一直拿到 "Connection closed"。
        """
        if self._loop is None:
            return

        session, stream_cm, http_client = (
            self._session,
            self._stream_cm,
            self._http_client,
        )
        self._session = self._stream_cm = self._http_client = None
        self._tools_cache.clear()
        if session is None:
            return

        async def _shutdown() -> None:
            for context in (session, stream_cm, http_client):
                if context is None:
                    continue
                try:
                    await context.__aexit__(None, None, None)
                except Exception:
                    pass

        try:
            self._submit(_shutdown(), timeout=3)
        except Exception:
            pass

    def _record_success(self) -> None:
        """调用成功：重置计数，闭合熔断。"""
        if not self._available or self._failure_count:
            logger.info("[yuandian] 调用成功，熔断器闭合")
        self._failure_count = 0
        self._available = True
        self._opened_at = None

    def _redact(self, text: Any) -> str:
        """脱敏：底层异常文本可能回显含 key 的头部/URL，入日志或对外异常前必须抹除。"""
        rendered = str(text)
        return rendered.replace(self._api_key, "***") if self._api_key else rendered

    def _record_failure(self, error: Exception) -> None:
        """调用失败：作废连接、累加计数，达到阈值打开熔断。"""
        self._failure_count += 1
        logger.warning(
            f"[yuandian] 调用失败（{type(error).__name__}: {self._redact(error)}），"
            f"计数 {self._failure_count}/{self._failure_threshold}"
        )
        self._invalidate_connection()
        if self._available:
            if self._failure_count >= self._failure_threshold:
                self._available = False
                self._opened_at = _now()
                logger.warning(
                    f"[yuandian] 熔断打开，{self._recovery_timeout}s 内拒绝调用，降级本地知识库"
                )
        elif self._opened_at is not None:
            # 半开探针失败：重新计算冷却时间
            self._opened_at = _now()
            logger.warning(
                f"[yuandian] 探针失败，熔断重新计时 {self._recovery_timeout}s"
            )

    def _gate(self) -> None:
        """调用前熔断闸门：打开中且未冷却则快速失败；冷却结束放探针。"""
        if self._available:
            return

        elapsed = _now() - (self._opened_at or 0.0)
        if elapsed < self._recovery_timeout:
            raise YuandianCircuitOpenError(
                f"元典 MCP 熔断中，还需等待 {self._recovery_timeout - elapsed:.0f}s"
            )

        logger.info("[yuandian] 冷却结束，放行探针请求（half-open）")

    def status(self) -> dict[str, Any]:
        """熔断状态快照，供业务节点回写 AgentState。"""
        with self._lock:
            return {
                "yuandian_available": self._available,
                "mcp_failure_count": self._failure_count,
            }

    # ---------- 对外调用 ----------
    async def _call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """事件循环内执行：确保连接 → 调工具 → 拼文本结果。"""
        await self._connect()
        result = await self._session.call_tool(tool_name, arguments=arguments)
        if getattr(result, "is_error", False):
            raise YuandianMCPError(
                f"{tool_name} 返回工具级错误：{_extract_text(result)}"
            )
        return _extract_text(result)

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """
        同步调用一个 MCP 工具，异常统一包装并脱敏。

        加锁是为了让 _gate 读取 _available/_opened_at 与 _record_failure
        的写入互斥，避免熔断决策读到中间状态。

        Raises:
            YuandianCircuitOpenError: 熔断打开中
            YuandianMCPError: 调用失败
        """
        with self._lock:
            self._gate()
            try:
                text = self._submit(self._call(tool_name, arguments or {}))
            except YuandianCircuitOpenError:
                raise
            except Exception as e:
                self._record_failure(e)
                # 统一异常类型并脱敏，防止 key 随底层错误外泄
                raise YuandianMCPError(
                    f"调用 {tool_name} 失败：{type(e).__name__}: {self._redact(e)}"
                ) from e
            self._record_success()
            return text

    def list_mcp_tools(self, refresh: bool = False) -> dict[str, Any]:
        """列出服务端工具（带缓存），返回 ``{工具名: mcp.Tool}``。"""
        with self._lock:
            if not self._tools_cache or refresh:
                result = self._submit(self._list_tools())
                self._tools_cache = {tool.name: tool for tool in result.tools}
                self._record_success()
            return dict(self._tools_cache)

    async def _list_tools(self) -> Any:
        await self._connect()
        return await self._session.list_tools()

    def close(self) -> None:
        """尽力关闭连接（演示项目进程退出即可，守护线程不会阻止退出）。"""
        with self._lock:
            self._invalidate_connection()


def _now() -> float:
    """单调时钟，不受系统时间回拨影响。"""
    import time

    return time.monotonic()


def _extract_text(result: Any) -> str:
    """把 MCP CallToolResult 的 content 各文本块拼成字符串。"""
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "\n".join(parts)


_client_singleton: YuandianMCPClient | None = None
_singleton_lock = threading.Lock()


def get_yuandian_client() -> YuandianMCPClient | None:
    """
    获取元典客户端单例

    未配置 YUANDIAN_MCP_TOKEN 时返回 None，调用方据此走纯本地检索，
    而不是在启动期直接报错。
    """
    global _client_singleton

    api_key = get_env_var("YUANDIAN_MCP_TOKEN").strip()
    if not api_key:
        logger.warning("[yuandian] 未配置 YUANDIAN_MCP_TOKEN，外部检索不可用")
        return None

    with _singleton_lock:
        if _client_singleton is None:
            settings = get_settings().mcp
            _client_singleton = YuandianMCPClient(
                url=get_env_var("YUANDIAN_MCP_URL").strip() or DEFAULT_MCP_URL,
                api_key=api_key,
                failure_threshold=settings.failure_threshold,
                recovery_timeout=settings.recovery_timeout,
            )
        return _client_singleton


def _make_tool_runner(client: YuandianMCPClient, tool_name: str):
    """生成绑定工具名的同步调用闭包，剔除未传（None）参数。"""

    def run(**kwargs: Any) -> str:
        arguments = {key: value for key, value in kwargs.items() if value is not None}
        logger.info(f"[yuandian] Function-Calling 调用 {tool_name}：{arguments}")
        return client.call_tool(tool_name, arguments)

    run.__name__ = tool_name
    return run


def build_yuandian_tools() -> list[BaseTool]:
    """
    把 allowlist 内的元典 MCP 工具包装成 LangChain StructuredTool

    入参模型由各工具的 ``input_schema`` 动态生成，无需手写参数类；
    未配置 key 或列工具失败（含熔断）时返回空列表，由本地知识库兜底。
    """
    client = get_yuandian_client()
    if client is None:
        return []

    try:
        remote_tools = client.list_mcp_tools()
    except Exception as e:
        logger.warning(f"[yuandian] 获取工具清单失败，本次不注册外部工具：{e}")
        return []
    tools: list[BaseTool] = []
    for tool_name in sorted(ALLOWED_MCP_TOOLS):
        remote = remote_tools.get(tool_name)
        if remote is None:
            logger.warning(f"[yuandian] 服务端未提供 allowlist 工具：{tool_name}")
            continue
        args_schema = _model_from_schema(f"{tool_name}_args", remote.input_schema)
        tool = StructuredTool.from_function(
            _make_tool_runner(client, tool_name),
            name=remote.name,
            description=remote.description or tool_name,
            args_schema=args_schema,
        )
        tools.append(tool)

    logger.info(f"[yuandian] 已注册 {len(tools)} 个外部 MCP 工具")
    return tools


if __name__ == "__main__":
    # 最小烟囱测试：列工具 + 查积分
    c = get_yuandian_client()
    if c is None:
        print("未配置 YUANDIAN_MCP_TOKEN")
    else:
        print(c.call_tool("yuandian_get_user_balance", {}))
        c.close()
