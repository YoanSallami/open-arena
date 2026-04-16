"""LLM-as-a-Verifier evaluator).

Pointwise scoring via token-level `top_logprobs`. Scales over score-token
granularity (G) and repeated verification (K).

Reference: Kwok et al. 2026 — LLM-as-a-Verifier.
"""

import json
from dataclasses import asdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from src import default_prompts
from src.evaluation.base import PointwiseEvaluator
from src.evaluation.evaluators._verifier_reward import (
    build_score_tokens,
    verifier_reward,
)
from src.execution import ExecutionResult
from src.llms import AgentStep


class LLMAsVerifierEvaluator(PointwiseEvaluator):
    """Verifier that produces a continuous score per item by taking the
    expectation of scalar scores under the model's `top_logprobs`
    distribution at a single `<score>` position. Averages over K repeats.
    """

    def __init__(
        self,
        results: list[ExecutionResult],
        llm_config: dict[str, Any],
        system_prompt: str = default_prompts["evaluation"]["llm_as_verifier"],
        system_prompt_no_reference: str = default_prompts["evaluation"]["llm_as_verifier_no_reference"],
        granularity: int = 8,
        repeats: int = 1,
        score_name: str = "verifier_score",
        max_concurrency: int = 10,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        super().__init__(results=results, score_name=score_name, max_concurrency=max_concurrency)
        self.llm_config = llm_config
        self.granularity = granularity
        self.repeats = max(1, repeats)
        self._callbacks = list(callbacks or [])
        self._score_tokens = build_score_tokens(granularity)
        self.system_prompt = _render(system_prompt, self._score_tokens)
        self.system_prompt_no_reference = _render(system_prompt_no_reference, self._score_tokens)

    async def _score(
        self,
        input: str,
        output: str,
        expected_output: str | None = None,
        trajectory: list[AgentStep] | None = None,
    ) -> tuple[float | None, str | None, str | None]:
        system = self.system_prompt if expected_output is not None else self.system_prompt_no_reference

        payload_obj: dict[str, Any] = {"input": input, "output": output}
        if expected_output is not None:
            payload_obj["expected_output"] = expected_output
        if trajectory:
            payload_obj["trajectory"] = [asdict(step) for step in trajectory]
        payload = json.dumps(payload_obj, indent=2, default=str)

        reward, error = await verifier_reward(
            llm_config=self.llm_config,
            system_prompt=system,
            user_payload=payload,
            granularity=self.granularity,
            repeats=self.repeats,
        )
        if error is not None and reward is None:
            return None, None, error
        explanation = f"R(t,τ)={reward:.4f} (G={self.granularity}, K={self.repeats})"
        return reward, explanation, None


def _render(prompt: str, score_tokens: list[str]) -> str:
    """Substitute the known placeholders without requiring full str.format
    semantics (user overrides may contain stray braces)."""
    return (
        prompt.replace("{granularity}", str(len(score_tokens)))
        .replace("{score_letters}", ", ".join(score_tokens))
        .replace("{best_letter}", score_tokens[0])
        .replace("{worst_letter}", score_tokens[-1])
    )
