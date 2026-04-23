from typing import Any

from src.llms.base import AgentStep, LLMCaller, Msg, ToolInvocation


class ReplayCaller(LLMCaller):
    """Return pre-recorded outputs and trajectories instead of calling an LLM."""

    def __init__(
        self,
        llm_config: dict[str, Any],
        lookup: dict[str, tuple[str, list[dict[str, Any]]]],
        callbacks=None,
    ):
        super().__init__(llm_config, callbacks=callbacks)
        self.lookup = dict(lookup)

    async def achat(self, messages: list[Msg]) -> str:
        output, _ = await self.achat_with_trajectory(messages)
        return output

    async def achat_with_trajectory(
        self,
        messages: list[Msg],
    ) -> tuple[str, list[AgentStep] | None]:
        user_text = _last_user_message(messages)
        if user_text not in self.lookup:
            raise ValueError(f"No replay entry found for rendered input: {user_text[:120]!r}")

        output, trajectory = self.lookup[user_text]
        return output, [_deserialize_agent_step(step) for step in trajectory]


def _last_user_message(messages: list[Msg]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    raise ValueError("ReplayCaller expected at least one user message")


def _deserialize_agent_step(step: dict[str, Any]) -> AgentStep:
    tool_calls = [
        ToolInvocation(
            name=str(call.get("name") or ""),
            args=dict(call.get("args") or {}),
            output=str(call.get("output") or ""),
        )
        for call in step.get("tool_calls", [])
    ]
    reasoning = step.get("reasoning")
    return AgentStep(
        thought=str(step.get("thought") or ""),
        reasoning=str(reasoning) if reasoning is not None else None,
        tool_calls=tool_calls,
    )
