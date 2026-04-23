"""Unit tests for JudgePanelEvaluator.

Panelist and smart-judge LangChain runnables are replaced with stubs that
expose `async ainvoke(messages, config=None)` and return pre-set
`PanelJudgeResponse` objects — we never hit a real model.
"""

from __future__ import annotations

import asyncio

import pytest

from src.config.types import (
    ScenarioConfig,
    ScenarioMetricConfig,
    ScenarioRubricBand,
)
from src.evaluation.evaluators.judge_panel import JudgePanelEvaluator
from src.evaluation.types import PanelJudgeResponse


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Stub:
    """Minimal stand-in for a LangChain Runnable — captures invocation count
    and returns a pre-set `PanelJudgeResponse` (or raises)."""

    def __init__(
        self,
        response: PanelJudgeResponse | None = None,
        error: Exception | None = None,
    ):
        self._response = response
        self._error = error
        self.call_count = 0
        self.captured_messages: list | None = None

    async def ainvoke(self, messages, config=None):
        self.call_count += 1
        self.captured_messages = messages
        if self._error is not None:
            raise self._error
        return self._response


def _make_scenario(
    *,
    required_trace_fields: list[str] | None = None,
    metric_names: list[str] | None = None,
    threshold: float = 0.85,
) -> ScenarioConfig:
    trace_fields = required_trace_fields or ["actual_output"]
    metric_names = metric_names or ["m1"]
    # Contiguous 0-10 rubric (4 bands) reused across metrics.
    rubric = [
        ScenarioRubricBand(score_range=[0, 2], expected_outcome="bad"),
        ScenarioRubricBand(score_range=[3, 5], expected_outcome="meh"),
        ScenarioRubricBand(score_range=[6, 8], expected_outcome="good"),
        ScenarioRubricBand(score_range=[9, 10], expected_outcome="great"),
    ]
    metrics = [
        ScenarioMetricConfig(
            name=name,
            criteria=f"criterion for {name}",
            rubric=rubric,
            threshold=threshold,
        )
        for name in metric_names
    ]
    return ScenarioConfig(
        id="test",
        name="test scenario",
        description="scenario used by judge-panel unit tests",
        required_trace_fields=trace_fields,
        scenario_metrics=metrics,
    )


def _make_evaluator(
    *,
    scenario: ScenarioConfig | None = None,
    num_panelists: int = 3,
) -> JudgePanelEvaluator:
    scenario = scenario or _make_scenario()
    return JudgePanelEvaluator(
        results=[],
        scenario=scenario,
        panelist_llm_configs=[
            {"model": f"openai/panelist-{i}"} for i in range(num_panelists)
        ],
        smart_llm_config={"model": "openai/smart"},
    )


def _install_panelists(evaluator: JudgePanelEvaluator, stubs: list[_Stub]) -> None:
    evaluator._panelists = stubs  # type: ignore[assignment]


def _install_smart(evaluator: JudgePanelEvaluator, stub: _Stub) -> None:
    evaluator._smart = stub  # type: ignore[assignment]


def _run_score(evaluator: JudgePanelEvaluator, **kwargs):
    kwargs.setdefault("input", "q")
    kwargs.setdefault("output", "a")
    return asyncio.run(evaluator._score(**kwargs))


# ---------------------------------------------------------------------------
# Score normalization — 0-10 → [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [(0, 0.0), (5, 0.5), (8, 0.8), (10, 1.0)])
def test_score_normalizes_0_to_10_to_unit_interval(raw, expected):
    evaluator = _make_evaluator(num_panelists=2)
    _install_panelists(
        evaluator,
        [
            _Stub(PanelJudgeResponse(reasoning="r", score=raw)),
            _Stub(PanelJudgeResponse(reasoning="r", score=raw)),
        ],
    )
    score, _, error = _run_score(evaluator)
    assert score == pytest.approx(expected)
    assert error is None


# ---------------------------------------------------------------------------
# Agreement path — panel consensus, smart not called
# ---------------------------------------------------------------------------


def test_agreement_returns_panel_consensus_without_calling_smart():
    evaluator = _make_evaluator(num_panelists=3)
    panelists = [
        _Stub(PanelJudgeResponse(reasoning="r1", score=7)),
        _Stub(PanelJudgeResponse(reasoning="r2", score=8)),
        _Stub(PanelJudgeResponse(reasoning="r3", score=8)),
    ]
    smart = _Stub(PanelJudgeResponse(reasoning="smart", score=0))
    _install_panelists(evaluator, panelists)
    _install_smart(evaluator, smart)

    score, explanation, error = _run_score(evaluator)
    # Consensus = mean(0.7, 0.8, 0.8) = 0.7667 (spread 0.1 ≤ tolerance 0.15)
    assert score == pytest.approx((0.7 + 0.8 + 0.8) / 3.0)
    assert error is None
    assert smart.call_count == 0
    assert all(stub.call_count == 1 for stub in panelists)
    assert "source=panel" in explanation


# ---------------------------------------------------------------------------
# Disagreement path — escalate to smart model
# ---------------------------------------------------------------------------


def test_disagreement_escalates_to_smart_and_uses_its_score():
    evaluator = _make_evaluator(num_panelists=3)
    panelists = [
        _Stub(PanelJudgeResponse(reasoning="low", score=2)),
        _Stub(PanelJudgeResponse(reasoning="high", score=8)),
        _Stub(PanelJudgeResponse(reasoning="higher", score=9)),
    ]
    smart = _Stub(PanelJudgeResponse(reasoning="verdict", score=7))
    _install_panelists(evaluator, panelists)
    _install_smart(evaluator, smart)

    score, explanation, error = _run_score(evaluator)
    # Spread = 0.9 - 0.2 = 0.7 > tolerance 0.15 → smart wins.
    assert score == pytest.approx(0.7)
    assert error is None
    assert smart.call_count == 1
    assert "source=smart_judge" in explanation


# ---------------------------------------------------------------------------
# Missing trace field — no model calls
# ---------------------------------------------------------------------------


def test_missing_trace_field_returns_error_without_any_model_call():
    scenario = _make_scenario(
        required_trace_fields=["start_date", "end_date", "actual_output"],
    )
    evaluator = _make_evaluator(scenario=scenario)
    panelists = [
        _Stub(PanelJudgeResponse(reasoning="r", score=8)),
        _Stub(PanelJudgeResponse(reasoning="r", score=8)),
        _Stub(PanelJudgeResponse(reasoning="r", score=8)),
    ]
    smart = _Stub(PanelJudgeResponse(reasoning="r", score=8))
    _install_panelists(evaluator, panelists)
    _install_smart(evaluator, smart)

    score, explanation, error = _run_score(
        evaluator,
        metadata={"start_date": "2026-01-01"},  # end_date missing
    )
    assert score is None
    assert explanation is None
    assert "missing required trace field" in error
    assert "end_date" in error
    assert all(stub.call_count == 0 for stub in panelists)
    assert smart.call_count == 0


def test_required_trace_fields_resolved_from_metadata():
    scenario = _make_scenario(
        required_trace_fields=["query_domains", "actual_output"],
    )
    evaluator = _make_evaluator(scenario=scenario, num_panelists=2)
    _install_panelists(
        evaluator,
        [
            _Stub(PanelJudgeResponse(reasoning="r", score=9)),
            _Stub(PanelJudgeResponse(reasoning="r", score=9)),
        ],
    )

    score, _, error = _run_score(
        evaluator,
        metadata={"query_domains": ["example.com", "openai.com"]},
    )
    assert score == pytest.approx(0.9)
    assert error is None


# ---------------------------------------------------------------------------
# Panelist failures
# ---------------------------------------------------------------------------


def test_partial_panelist_failure_still_reaches_agreement():
    evaluator = _make_evaluator(num_panelists=3)
    panelists = [
        _Stub(PanelJudgeResponse(reasoning="a", score=8)),
        _Stub(error=RuntimeError("rate limit")),
        _Stub(PanelJudgeResponse(reasoning="c", score=8)),
    ]
    smart = _Stub(PanelJudgeResponse(reasoning="smart", score=0))
    _install_panelists(evaluator, panelists)
    _install_smart(evaluator, smart)

    score, explanation, error = _run_score(evaluator)
    assert score == pytest.approx(0.8)
    assert error is None
    assert smart.call_count == 0
    assert "source=panel_partial" in explanation
    assert "rate limit" in explanation  # surfaced in the per-metric reason


class _MetricRoutedStub:
    """Stub that keys its response on the metric name encoded in the system
    prompt — robust to asyncio scheduling order."""

    def __init__(self, by_metric: dict[str, PanelJudgeResponse | Exception]):
        self._by_metric = by_metric
        self.call_count = 0
        self.calls_by_metric: dict[str, int] = {}

    async def ainvoke(self, messages, config=None):
        self.call_count += 1
        system = str(messages[0].content)
        for metric_name, outcome in self._by_metric.items():
            if f'metric "{metric_name}"' in system:
                self.calls_by_metric[metric_name] = (
                    self.calls_by_metric.get(metric_name, 0) + 1
                )
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise AssertionError(f"no stub for metric in system prompt: {system!r}")


def test_all_panelists_fail_for_one_metric_but_other_metric_succeeds():
    # Every panelist AND the smart judge raise for the "broken" metric, so
    # there's no surviving score for that metric; the "ok" metric still
    # produces a normal panel consensus.
    scenario = _make_scenario(metric_names=["broken", "ok"])
    evaluator = _make_evaluator(scenario=scenario, num_panelists=3)

    ok_response = PanelJudgeResponse(reasoning="r", score=9)
    panelists = [
        _MetricRoutedStub({"broken": RuntimeError("boom"), "ok": ok_response})
        for _ in range(3)
    ]
    smart = _MetricRoutedStub({"broken": RuntimeError("smart-boom")})
    _install_panelists(evaluator, panelists)
    _install_smart(evaluator, smart)

    score, explanation, error = _run_score(evaluator)
    # Errored metric contributes 0.0; aggregate = mean(0.0, 0.9) = 0.45.
    assert score == pytest.approx((0.0 + 0.9) / 2)
    assert error is not None
    assert "broken" in error
    assert "ok" in explanation
    assert "broken" in explanation
    # Every panelist was asked about both metrics.
    for stub in panelists:
        assert stub.calls_by_metric == {"broken": 1, "ok": 1}


def test_single_surviving_panelist_escalates_to_smart():
    evaluator = _make_evaluator(num_panelists=3)
    panelists = [
        _Stub(PanelJudgeResponse(reasoning="lone", score=8)),
        _Stub(error=RuntimeError("boom-1")),
        _Stub(error=RuntimeError("boom-2")),
    ]
    smart = _Stub(PanelJudgeResponse(reasoning="verdict", score=6))
    _install_panelists(evaluator, panelists)
    _install_smart(evaluator, smart)

    score, explanation, error = _run_score(evaluator)
    # A single surviving panelist is not a consensus — smart wins (0.6).
    assert score == pytest.approx(0.6)
    assert error is None
    assert smart.call_count == 1
    assert "source=smart_judge" in explanation
    assert "only 1/3 panelists" in explanation


def test_mixed_metric_outcomes_in_single_score_call():
    # metric 'agree' → 3 panelists within tolerance → panel consensus.
    # metric 'disagree' → wide spread → smart judge escalation.
    scenario = _make_scenario(metric_names=["agree", "disagree"])
    evaluator = _make_evaluator(scenario=scenario, num_panelists=3)

    panelists = [
        _MetricRoutedStub(
            {
                "agree": PanelJudgeResponse(reasoning="r", score=8),
                "disagree": PanelJudgeResponse(reasoning="r", score=2),
            },
        ),
        _MetricRoutedStub(
            {
                "agree": PanelJudgeResponse(reasoning="r", score=8),
                "disagree": PanelJudgeResponse(reasoning="r", score=8),
            },
        ),
        _MetricRoutedStub(
            {
                "agree": PanelJudgeResponse(reasoning="r", score=8),
                "disagree": PanelJudgeResponse(reasoning="r", score=9),
            },
        ),
    ]
    smart = _MetricRoutedStub(
        {"disagree": PanelJudgeResponse(reasoning="verdict", score=5)},
    )
    _install_panelists(evaluator, panelists)
    _install_smart(evaluator, smart)

    score, explanation, error = _run_score(evaluator)
    # agree: mean(0.8,0.8,0.8) = 0.8; disagree (smart): 0.5; aggregate = 0.65
    assert score == pytest.approx((0.8 + 0.5) / 2)
    assert error is None
    # Smart was called exactly once, and only for the disagreeing metric.
    assert smart.call_count == 1
    assert smart.calls_by_metric == {"disagree": 1}
    assert "source=panel" in explanation
    assert "source=smart_judge" in explanation


def test_explanation_marks_pass_and_fail_per_metric():
    scenario = _make_scenario(metric_names=["pass_me", "fail_me"], threshold=0.85)
    evaluator = _make_evaluator(scenario=scenario, num_panelists=2)

    panelists = [
        _MetricRoutedStub(
            {
                "pass_me": PanelJudgeResponse(reasoning="r", score=9),   # 0.9 ≥ 0.85
                "fail_me": PanelJudgeResponse(reasoning="r", score=2),   # 0.2 < 0.85
            },
        ),
        _MetricRoutedStub(
            {
                "pass_me": PanelJudgeResponse(reasoning="r", score=9),
                "fail_me": PanelJudgeResponse(reasoning="r", score=2),
            },
        ),
    ]
    _install_panelists(evaluator, panelists)

    score, explanation, error = _run_score(evaluator)
    assert error is None
    assert score == pytest.approx((0.9 + 0.2) / 2)
    assert "pass_me: score=0.900" in explanation
    assert "[PASS]" in explanation
    assert "fail_me: score=0.200" in explanation
    assert "[FAIL]" in explanation


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_fewer_than_two_panelists():
    with pytest.raises(ValueError, match="at least 2 panelists"):
        JudgePanelEvaluator(
            results=[],
            scenario=_make_scenario(),
            panelist_llm_configs=[{"model": "only-one"}],
            smart_llm_config={"model": "openai/smart"},
        )
