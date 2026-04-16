import asyncio

from src.llms.base import LLMCaller, Msg, build_chat_model, to_langchain_messages


class SimpleCaller(LLMCaller):
    """Single model call. No tools, no graph, no lifecycle.

    Wraps ChatLiteLLM and forwards Langfuse/LangChain callbacks so traces
    land in the same pipeline as the agent mode.
    """

    def __init__(self, *args, timeout_s: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout_s = timeout_s
        self._model = build_chat_model(self.llm_config)

    async def achat(self, messages: list[Msg]) -> str:
        coro = self._model.ainvoke(
            to_langchain_messages(messages),
            config={"callbacks": self.callbacks},
        )
        if self.timeout_s is not None:
            coro = asyncio.wait_for(coro, timeout=self.timeout_s)
        response = await coro
        return str(response.content or "")
