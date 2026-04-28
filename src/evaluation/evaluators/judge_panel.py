"""Judge-panel evaluator: multiple panelist LLMs score each response
against a rubric-driven scenario; if they agree (per-metric spread within
`1 - metric.threshold`) we return the consensus, otherwise we escalate
the disagreeing metric to a smarter model.

Agreement rule:
    spread = max(panelist_scores) - min(panelist_scores)
    agreed = (len(successes) >= 2) and (spread <= 1 - metric.threshold)

At least two panelists must produce a score for a consensus; a lone
survivor is not a consensus and routes to the smart judge. The smart
model is mandatory — every metric has a guaranteed escalation path.

Per-metric sources: "panel" (consensus over all panelists), "panel_partial"
(consensus over surviving panelists after some failed), "smart_judge"
(escalation because the panel disagreed or too few panelists survived).

Aggregation: the row score is the mean of per-metric scores. Metrics that
errored contribute 0.0 to the mean; their errors are surfaced on the row's
`error` field and in the per-metric explanation.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException

from src.config.types import ScenarioConfig, ScenarioMetricConfig
from src.evaluation.base import PointwiseEvaluator
from src.evaluation.types import PanelJudgeResponse
from src.execution import ExecutionResult
from src.llms import AgentStep
from src.llms.base import build_chat_model, to_langchain_messages

_RETRY_EXCEPTIONS = (OutputParserException,)


@dataclass(frozen=True, slots=True)
class _PanelistResult:
    panelist_index: int
    score: float | None  # normalized to [0, 1]; None on failure
    reasoning: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _MetricOutcome:
    name: str
    score: float | None  # final normalized score, None if metric errored
    threshold: float
    passed: bool | None
    source: str  # "panel" | "panel_partial" | "smart_judge" | "error"
    reason: str
    error: str | None


class JudgePanelEvaluator(PointwiseEvaluator):
    """Run a panel of panelist LLMs against a scenario's rubric metrics."""

    def __init__(
        self,
        results: list[ExecutionResult],
        scenario: ScenarioConfig,
        panelist_llm_configs: list[dict[str, Any]],
        smart_llm_config: dict[str, Any],
        score_name: str = "judge_panel_score",
        max_concurrency: int = 10,
        max_retries: int = 3,
        callbacks: list[BaseCallbackHandler] | None = None,
        timeout_s: float | None = None,
    ):
        if len(panelist_llm_configs) < 2:
            raise ValueError(
                "JudgePanelEvaluator requires at least 2 panelists; got "
                f"{len(panelist_llm_configs)}."
            )
        super().__init__(
            results=results,
            score_name=score_name,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
        )
        self.scenario = scenario
        self.panelist_llm_configs = list(panelist_llm_configs)
        self.smart_llm_config = smart_llm_config
        self.max_retries = max_retries
        self._callbacks = list(callbacks or [])
        self._panelists = [
            self._build_structured_judge(cfg) for cfg in self.panelist_llm_configs
        ]
        self._smart = self._build_structured_judge(smart_llm_config)

    async def _score(
        self,
        input: str,
        output: str,
        expected_output: str | None = None,
        trajectory: list[AgentStep] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float | None, str | None, str | None]:
        trace_fields, missing = self._resolve_trace_fields(
            output=output, metadata=metadata or {},
        )
        if missing:
            return (
                None,
                None,
                f"missing required trace field(s): {', '.join(missing)}",
            )

        user_payload = self._build_user_payload(input, output, trace_fields, trajectory)

        # Score every metric; each metric fans out across all panelists.
        outcomes = await asyncio.gather(*[
            self._score_metric(metric, user_payload)
            for metric in self.scenario.scenario_metrics
        ])

        # Aggregate: mean of per-metric scores; errored metrics count as 0.0.
        per_metric = [o.score if o.score is not None else 0.0 for o in outcomes]
        final_score = sum(per_metric) / len(per_metric)
        errors = [o.error for o in outcomes if o.error]
        error = "; ".join(errors) if errors else None
        explanation = self._render_explanation(outcomes)
        return final_score, explanation, error

    async def _score_metric(
        self,
        metric: ScenarioMetricConfig,
        user_payload: str,
    ) -> _MetricOutcome:
        system_prompt = self._render_system_prompt(metric)
        messages = to_langchain_messages([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ])

        panelist_results = await asyncio.gather(*[
            self._invoke_panelist(idx, panelist, messages)
            for idx, panelist in enumerate(self._panelists)
        ])

        successes = [r for r in panelist_results if r.score is not None]
        failures = [r for r in panelist_results if r.error is not None]

        agreement_tolerance = 1.0 - metric.threshold
        if successes:
            scores = [r.score for r in successes]
            # Consensus requires at least two surviving panelists within tolerance.
            # A single survivor is not a consensus — route to the smart judge.
            spread = max(scores) - min(scores) if len(successes) >= 2 else None
            if spread is not None and spread <= agreement_tolerance:
                consensus = sum(scores) / len(scores)
                source = "panel_partial" if failures else "panel"
                return _MetricOutcome(
                    name=metric.name,
                    score=consensus,
                    threshold=metric.threshold,
                    passed=consensus >= metric.threshold,
                    source=source,
                    reason=self._render_panel_reason(
                        successes, failures, spread=spread, consensus=consensus,
                    ),
                    error=None,
                )
            disagreement = self._describe_disagreement(
                spread=spread,
                agreement_tolerance=agreement_tolerance,
                successes=successes,
                total_panelists=len(self._panelists),
            )
        else:
            error_summary = "; ".join(
                f"p{r.panelist_index}: {r.error}" for r in failures
            )
            disagreement = (
                f"all panelists failed ({error_summary})"
                if error_summary else "all panelists failed"
            )
        smart_result = await self._invoke_smart(messages)
        if smart_result.score is None:
            return _MetricOutcome(
                name=metric.name,
                score=None,
                threshold=metric.threshold,
                passed=None,
                source="error",
                reason=f"{disagreement}; smart judge failed: {smart_result.error}",
                error=(
                    f"metric '{metric.name}': smart judge failed after "
                    f"{disagreement}: {smart_result.error}"
                ),
            )
        return _MetricOutcome(
            name=metric.name,
            score=smart_result.score,
            threshold=metric.threshold,
            passed=smart_result.score >= metric.threshold,
            source="smart_judge",
            reason=(
                f"escalated: {disagreement}; smart score={smart_result.score:.3f}; "
                f"reason: {_truncate(smart_result.reasoning)}"
            ),
            error=None,
        )

    @staticmethod
    def _describe_disagreement(
        *,
        spread: float | None,
        agreement_tolerance: float,
        successes: list[_PanelistResult],
        total_panelists: int,
    ) -> str:
        if spread is None:
            return (
                f"only {len(successes)}/{total_panelists} panelists produced "
                f"a score (need ≥ 2 for consensus)"
            )
        return (
            f"panel disagreement (spread={spread:.3f} > "
            f"{agreement_tolerance:.3f})"
        )

    async def _invoke_panelist(
        self,
        panelist_index: int,
        panelist: Any,
        messages: list,
    ) -> _PanelistResult:
        try:
            parsed: PanelJudgeResponse = await panelist.ainvoke(
                messages, config={"callbacks": self._callbacks},
            )
        except Exception as e:
            return _PanelistResult(
                panelist_index=panelist_index,
                score=None,
                reasoning=None,
                error=str(e),
            )
        return _PanelistResult(
            panelist_index=panelist_index,
            score=parsed.score / 10.0,
            reasoning=parsed.reasoning,
            error=None,
        )

    async def _invoke_smart(self, messages: list) -> _PanelistResult:
        try:
            parsed: PanelJudgeResponse = await self._smart.ainvoke(
                messages, config={"callbacks": self._callbacks},
            )
        except Exception as e:
            return _PanelistResult(
                panelist_index=-1, score=None, reasoning=None, error=str(e),
            )
        return _PanelistResult(
            panelist_index=-1,
            score=parsed.score / 10.0,
            reasoning=parsed.reasoning,
            error=None,
        )

    def _resolve_trace_fields(
        self,
        output: str,
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        trace_fields: dict[str, Any] = {}
        missing: list[str] = []
        for field in self.scenario.required_trace_fields:
            if field == "actual_output":
                if not output:
                    missing.append(field)
                else:
                    trace_fields[field] = output
                continue
            if field in metadata and metadata[field] is not None:
                trace_fields[field] = metadata[field]
            else:
                missing.append(field)
        return trace_fields, missing

    def _build_user_payload(
        self,
        input: str,
        output: str,
        trace_fields: dict[str, Any],
        trajectory: list[AgentStep] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "input": input,
            "output": output,
            "required_trace_fields": trace_fields,
        }
        if trajectory:
            payload["trajectory"] = [asdict(step) for step in trajectory]
        return json.dumps(payload, indent=2, default=str)

    def _build_structured_judge(self, llm_config: dict[str, Any]) -> Any:
        return (
            build_chat_model(llm_config)
            .with_structured_output(
                PanelJudgeResponse, method="json_schema", strict=True
            )
            .with_retry(
                retry_if_exception_type=_RETRY_EXCEPTIONS,
                stop_after_attempt=self.max_retries,
            )
        )

    def _render_system_prompt(self, metric: ScenarioMetricConfig) -> str:
        rubric_lines = [
            f"  - score {band.score_range[0]}-{band.score_range[1]}: "
            f"{band.expected_outcome.strip()}"
            for band in sorted(metric.rubric, key=lambda b: b.score_range[0])
        ]
        return (
            f"You are evaluating a response against the metric "
            f"\"{metric.name}\" defined by the scenario \"{self.scenario.name}\".\n\n"
            f"Criterion:\n{metric.criteria.strip()}\n\n"
            f"Rubric (pick the integer score whose band best matches the output):\n"
            + "\n".join(rubric_lines)
            + "\n\nRespond as JSON with fields `reasoning` (string) and "
            "`score` (integer 0-10). The score MUST fall inside one of the "
            "rubric bands above."
        )

    @staticmethod
    def _render_panel_reason(
        successes: list[_PanelistResult],
        failures: list[_PanelistResult],
        *,
        spread: float,
        consensus: float,
    ) -> str:
        parts = [
            f"consensus={consensus:.3f}, spread={spread:.3f}",
            *(
                f"p{r.panelist_index}={r.score:.2f} ({_truncate(r.reasoning)})"
                for r in successes
            ),
        ]
        if failures:
            parts.append(
                "failures: "
                + ", ".join(f"p{r.panelist_index}: {r.error}" for r in failures)
            )
        return "; ".join(parts)

    def _render_explanation(self, outcomes: list[_MetricOutcome]) -> str:
        lines = [f"Scenario: {self.scenario.name}"]
        for outcome in outcomes:
            if outcome.score is None:
                lines.append(
                    f"- {outcome.name}: ERROR (source={outcome.source}) — "
                    f"{outcome.reason}"
                )
                continue
            verdict = "PASS" if outcome.passed else "FAIL"
            lines.append(
                f"- {outcome.name}: score={outcome.score:.3f} "
                f"threshold={outcome.threshold:.2f} [{verdict}] "
                f"source={outcome.source}; {outcome.reason}"
            )
        return "\n".join(lines)


def _truncate(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"
