from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_litellm import ChatLiteLLM


Msg = dict[str, str]


@dataclass
class ToolInvocation:
    name: str
    args: dict[str, Any]
    output: str


@dataclass
class AgentStep:
    """One turn of an agent: the model's reasoning/thought, plus the tool
    calls it emitted (which may be empty for the final-answer turn)."""
    thought: str                            # AIMessage.content for this turn
    reasoning: str | None = None            # reasoning_content (o1/r1/qwen3, optional)
    tool_calls: list[ToolInvocation] = field(default_factory=list)


class LLMCaller(ABC):
    """Base for both LLM modes. Async context manager so agent-flavoured
    subclasses can set up/tear down MCP connections; simple subclasses get
    no-op enter/exit for free."""

    def __init__(self, llm_config: dict[str, Any], callbacks: list[BaseCallbackHandler] | None = None):
        self.llm_config = dict(llm_config)
        self.callbacks = list(callbacks or [])

    async def __aenter__(self) -> "LLMCaller":
        await self._setup()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._close()

    async def _setup(self) -> None:
        """Subclasses that need async init (e.g. MCP discovery) override this."""

    async def _close(self) -> None:
        """Subclasses that own resources (e.g. MCP sessions) override this."""

    @abstractmethod
    async def achat(self, messages: list[Msg]) -> str:
        """Run a chat completion and return the final assistant text."""
        raise NotImplementedError

    async def achat_with_trajectory(
        self,
        messages: list[Msg],
    ) -> tuple[str, list[AgentStep] | None]:
        """Return (final_text, trajectory). Default implementation delegates
        to `achat` and returns no trajectory — agent subclasses override."""
        text = await self.achat(messages)
        return text, None


def build_chat_model(llm_config: dict[str, Any]) -> ChatLiteLLM:
    """Build a ChatLiteLLM from a LiteLLM config dict.

    ChatLiteLLM only accepts its declared fields; anything else (reasoning_effort,
    thinking, parallel_tool_calls, seed, ...) is forwarded to litellm via
    `model_kwargs` at call time.
    """
    known = set(ChatLiteLLM.model_fields)
    base = {k: v for k, v in llm_config.items() if k in known}
    extras = {k: v for k, v in llm_config.items() if k not in known and v is not None}
    if extras:
        base["model_kwargs"] = {**(base.get("model_kwargs") or {}), **extras}
    return ChatLiteLLM(**base)


_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def to_langchain_messages(messages: list[Msg]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
            continue
        cls = _ROLE_TO_MESSAGE.get(role, HumanMessage)
        out.append(cls(content=content))
    return out


def final_assistant_text(messages: list[BaseMessage]) -> str:
    """Return the last non-empty assistant message content, or '' if none."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""
