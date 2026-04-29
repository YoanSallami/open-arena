import json
import logging
import re
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel

_RETRY_EXCEPTIONS = (OutputParserException,)

from dataclasses import asdict

from src.evaluation.base import PointwiseEvaluator
from src.evaluation.types import JudgeResponse
from src.execution import ExecutionResult
from src.llms import AgentStep
from src.llms.base import build_chat_model, to_langchain_messages

_logger = logging.getLogger(__name__)


class LLMAsJudgeEvaluator(PointwiseEvaluator):
    """Score each output by asking a judge LLM. Two modes:

    - With `expected_output`: judges similarity to ground truth.
    - Without: judges output quality from input alone.

    Uses `ChatLiteLLM.with_structured_output(JudgeResponse, method="json_schema",
    strict=True)` so the judge returns a validated Pydantic instance via the
    provider's constrained-decoding / strict JSON schema path when available.
    """

    def __init__(
        self,
        results: list[ExecutionResult],
        llm_config: dict[str, Any],
        system_prompt: str,
        system_prompt_no_reference: str,
        score_name: str = "evaluation_score",
        max_concurrency: int = 10,
        max_retries: int = 3,
        callbacks: list[BaseCallbackHandler] | None = None,
        timeout_s: float | None = None,
    ):
        super().__init__(
            results=results,
            score_name=score_name,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
        )
        self.system_prompt = system_prompt
        self.system_prompt_no_reference = system_prompt_no_reference
        self._callbacks = list(callbacks or [])
        judge_model = build_chat_model(llm_config)
        model_name = str(llm_config.get("model") or "")
        api_base = str(llm_config.get("api_base") or "")
        api_key = str(llm_config.get("api_key") or "")
        if (
            model_name.startswith("ollama/")
            or model_name.startswith("ollama_chat/")
            or "localhost:11434" in api_base
            or api_key == "ollama"
        ):
            self._judge = _JSONJudgeAdapter(
                model=judge_model,
                max_retries=max_retries,
            )
        else:
            self._judge = judge_model.with_structured_output(
                JudgeResponse, method="json_schema", strict=True
            ).with_retry(retry_if_exception_type=_RETRY_EXCEPTIONS, stop_after_attempt=max_retries)

    async def _score(
        self,
        input: str,
        output: str,
        expected_output: str | None = None,
        trajectory: list[AgentStep] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float | None, str | None, str | None]:
        payload_obj: dict = {"input": input, "output": output}
        if expected_output is None:
            system = self.system_prompt_no_reference
        else:
            system = self.system_prompt
            payload_obj["expected_output"] = expected_output
        if trajectory:
            payload_obj["trajectory"] = [asdict(step) for step in trajectory]
        payload = json.dumps(payload_obj, indent=2, default=str)

        messages = to_langchain_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ])

        try:
            parsed: JudgeResponse = await self._judge.ainvoke(
                messages,
                config={"callbacks": self._callbacks},
            )
        except Exception as e:
            return None, None, str(e)

        return (parsed.score - 1) / 4.0, parsed.thinking, None


class _JSONJudgeAdapter:
    """Fallback adapter for local providers that struggle with strict JSON schema.

    Some local Ollama models stall on LangChain's strict structured-output path.
    For those models we use a plain chat completion and parse the first JSON
    object from the returned text ourselves.
    """

    def __init__(self, model: BaseChatModel, max_retries: int):
        self._model = model
        self._max_retries = max(1, max_retries)

    async def ainvoke(self, messages, config=None) -> JudgeResponse:
        last_error: Exception | None = None
        for _ in range(self._max_retries):
            try:
                response = await self._model.ainvoke(messages, config=config)
                content = getattr(response, "content", response)
                text = content if isinstance(content, str) else json.dumps(content)
                return _parse_judge_response(text)
            except _RETRY_EXCEPTIONS as exc:
                last_error = exc
        raise last_error or OutputParserException("Judge response could not be parsed")


def _parse_judge_response(text: str) -> JudgeResponse:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return JudgeResponse.model_validate_json(stripped)

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise OutputParserException(f"Expected JSON object from judge, got: {stripped[:200]!r}")
    return JudgeResponse.model_validate_json(match.group(0))
