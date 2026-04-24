from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src.llms.base import AgentStep, LLMCaller, Msg


class ReplayCaller(LLMCaller):
    """Return pre-recorded outputs instead of calling an LLM.

    The lookup is keyed by the rendered user input and yields the
    pre-recorded `(output, trajectory)` to return for that input.
    """

    def __init__(
        self,
        llm_config: dict[str, Any],
        lookup: dict[str, tuple[str, list[dict[str, Any]]]],
        callbacks: list[BaseCallbackHandler] | None = None,
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

        output, _trajectory = self.lookup[user_text]
        # Benchmark trajectories are baked into `output` as formatted text, so
        # we do not surface a structured trajectory here.
        return output, None


def _last_user_message(messages: list[Msg]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    raise ValueError("ReplayCaller expected at least one user message")
