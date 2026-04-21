import asyncio
import logging
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from src.llms.base import (
    AgentStep,
    LLMCaller,
    Msg,
    ToolInvocation,
    build_chat_model,
    final_assistant_text,
    to_langchain_messages,
)
from src.llms.types import MCPServerConfig

_logger = logging.getLogger(__name__)


def _servers_config(mcp_servers: list[MCPServerConfig]) -> dict[str, dict]:
    cfg: dict[str, dict] = {}
    for server in mcp_servers:
        name = server.get("server_name", f"server_{len(cfg)}")
        cfg[name] = {
            "transport": "sse",
            "url": server["url"],
            **({"headers": server["headers"]} if "headers" in server else {}),
        }
    return cfg


class AgentCaller(LLMCaller):
    """LangGraph ReAct-style (function-calling) agent with MCP tools.

    Parallel tool calls are handled natively by ToolNode when the model
    emits multiple tool_calls per turn.
    """

    def __init__(
        self,
        llm_config: dict[str, Any],
        mcp_servers: list[MCPServerConfig],
        *,
        max_steps: int = 10,
        timeout_s: float | None = 120.0,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        super().__init__(llm_config, callbacks=callbacks)
        self.mcp_servers = list(mcp_servers)
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self._mcp_client: MultiServerMCPClient | None = None
        self._agent = None

    async def _setup(self) -> None:
        self._mcp_client = MultiServerMCPClient(_servers_config(self.mcp_servers))
        tools = await self._mcp_client.get_tools()
        _logger.debug("Loaded %s MCP tools", len(tools))
        model = build_chat_model(self.llm_config)
        # create_react_agent binds tools to the model internally and handles
        # the tool-call loop, including parallel tool calls via ToolNode.
        self._agent = create_react_agent(model, tools)

    async def _close(self) -> None:
        if self._mcp_client is not None:
            close = getattr(self._mcp_client, "aclose", None) or getattr(self._mcp_client, "close", None)
            if close is not None:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            self._mcp_client = None
        self._agent = None

    async def achat(self, messages: list[Msg]) -> str:
        text, _ = await self.achat_with_trajectory(messages)
        return text

    async def achat_with_trajectory(
        self,
        messages: list[Msg],
    ) -> tuple[str, list[AgentStep] | None]:
        if self._agent is None:
            raise RuntimeError("AgentCaller not set up. Use 'async with AgentCaller(...)'.")

        coro = self._agent.ainvoke(
            {"messages": to_langchain_messages(messages)},
            config={
                "callbacks": self.callbacks,
                "recursion_limit": self.max_steps * 2,
            },
        )
        if self.timeout_s is not None:
            coro = asyncio.wait_for(coro, timeout=self.timeout_s)
        result = await coro

        msgs: list[BaseMessage] = result["messages"]
        return final_assistant_text(msgs), _extract_trajectory(msgs)


def _extract_trajectory(messages: list[BaseMessage]) -> list[AgentStep]:
    """Walk LangGraph's message list and group into per-turn AgentSteps.

    A turn = one AIMessage + (zero or more) following ToolMessages whose
    tool_call_id matches the AIMessage's tool_calls.
    """
    tool_outputs: dict[str, str] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_outputs[m.tool_call_id] = str(m.content or "")

    steps: list[AgentStep] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        reasoning = (m.additional_kwargs or {}).get("reasoning_content")
        raw_calls = getattr(m, "tool_calls", None) or []
        invocations = [
            ToolInvocation(
                name=c.get("name", ""),
                args=dict(c.get("args") or {}),
                output=tool_outputs.get(c.get("id", ""), ""),
            )
            for c in raw_calls
        ]
        steps.append(AgentStep(
            thought=str(m.content or ""),
            reasoning=str(reasoning) if reasoning else None,
            tool_calls=invocations,
        ))
    return steps
