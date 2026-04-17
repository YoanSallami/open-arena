import asyncio
import itertools
import json
import logging
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException

_RETRY_EXCEPTIONS = (OutputParserException,)
from pydantic import BaseModel, Field

from dataclasses import asdict

from src import default_prompts
from src.evaluation.base import GroupEvaluator
from src.execution import ExecutionResult
from src.llms import AgentStep
from src.llms.base import build_chat_model, to_langchain_messages

_logger = logging.getLogger(__name__)


class PairwiseVerdict(BaseModel):
    thinking: str = Field(..., description="Step-by-step reasoning before choosing a winner.")
    winner: Literal["A", "B", "tie"] = Field(..., description="Which output wins, or 'tie'.")


class LLMPairwiseJudgeEvaluator(GroupEvaluator):
    """Round-robin tournament judge.

    For each group of N model outputs, runs all `N*(N-1)/2` pairs and each
    pair twice with positions swapped (position-bias mitigation). Per-model
    score = win rate in [0, 1] (ties count as 0.5 win).
    """

    def __init__(
        self,
        groups: list[dict[str, ExecutionResult]],
        llm_config: dict[str, Any],
        system_prompt: str = default_prompts["evaluation"]["pairwise_judge"],
        system_prompt_no_reference: str = default_prompts["evaluation"]["pairwise_judge_no_reference"],
        score_name: str = "pairwise_score",
        max_concurrency: int = 10,
        max_retries: int = 3,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        super().__init__(groups=groups, score_name=score_name, max_concurrency=max_concurrency)
        self.system_prompt = system_prompt
        self.system_prompt_no_reference = system_prompt_no_reference
        self._callbacks = list(callbacks or [])
        self._judge = build_chat_model(llm_config).with_structured_output(
            PairwiseVerdict, method="json_schema", strict=True
        ).with_retry(retry_if_exception_type=_RETRY_EXCEPTIONS, stop_after_attempt=max_retries)

    async def _score_group(
        self,
        input: str,
        outputs: dict[str, str],
        expected_output: str | None = None,
        trajectories: dict[str, list[AgentStep]] | None = None,
    ) -> tuple[dict[str, float] | None, str | None, str | None]:
        models = list(outputs)
        if len(models) < 2:
            return None, None, "Pairwise tournament requires at least 2 models"

        pairs = list(itertools.combinations(models, 2))
        match_specs = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]

        def traj_for(m: str) -> list[AgentStep] | None:
            return trajectories.get(m) if trajectories else None

        try:
            verdicts = await asyncio.gather(*[
                self._judge_pair(input, a, outputs[a], b, outputs[b], expected_output,
                                 traj_for(a), traj_for(b))
                for a, b in match_specs
            ])
        except Exception as e:
            return None, None, str(e)

        wins: dict[str, float] = {m: 0.0 for m in models}
        reasons: list[str] = []
        failures = 0
        for (a, b), verdict in zip(match_specs, verdicts):
            if verdict is None:
                failures += 1
                continue
            if verdict.winner == "A":
                wins[a] += 1.0
            elif verdict.winner == "B":
                wins[b] += 1.0
            else:
                wins[a] += 0.5
                wins[b] += 0.5
            reasons.append(f"{a} vs {b}: {verdict.winner} — {verdict.thinking}")

        total_per_model = 2 * (len(models) - 1)
        scores = {m: wins[m] / total_per_model for m in models}
        explanation = "\n".join(reasons)
        error = f"{failures} pair judgements failed" if failures else None
        return scores, explanation, error

    async def _judge_pair(
        self,
        input: str,
        name_a: str,
        output_a: str,
        name_b: str,
        output_b: str,
        expected_output: str | None,
        trajectory_a: list[AgentStep] | None = None,
        trajectory_b: list[AgentStep] | None = None,
    ) -> PairwiseVerdict | None:
        payload_obj: dict = {"input": input, "output_A": output_a, "output_B": output_b}
        if expected_output is None:
            system = self.system_prompt_no_reference
        else:
            system = self.system_prompt
            payload_obj["expected_output"] = expected_output
        if trajectory_a:
            payload_obj["trajectory_A"] = [asdict(step) for step in trajectory_a]
        if trajectory_b:
            payload_obj["trajectory_B"] = [asdict(step) for step in trajectory_b]
        payload = json.dumps(payload_obj, indent=2, default=str)

        messages = to_langchain_messages([
            {"role": "system", "content": system},
            {"role": "user", "content": payload},
        ])

        try:
            async with self._judge_semaphore:
                return await self._judge.ainvoke(messages, config={"callbacks": self._callbacks})
        except Exception as e:
            _logger.error(f"Pair judgement failed ({name_a} vs {name_b}): {e}")
            return None
