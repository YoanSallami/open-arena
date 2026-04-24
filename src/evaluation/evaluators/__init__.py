import inspect

from src.evaluation.base import Evaluator
from src.evaluation.evaluators.llm_as_judge import LLMAsJudgeEvaluator
from src.evaluation.evaluators.llm_as_verifier import LLMAsVerifierEvaluator

_EVALUATORS: dict[str, type[Evaluator]] = {
    "llm_as_judge": LLMAsJudgeEvaluator,
    "llm_as_verifier": LLMAsVerifierEvaluator,
}

# Parameters the runner fills in itself (not user-configurable via
# EvaluationConfig). Excluded from `evaluator_init_params` so the
# config-to-kwargs dispatcher in main_cli never tries to thread them.
_RUNNER_MANAGED_PARAMS = frozenset({"self", "results", "groups"})


def _require_known_method(method: str) -> type[Evaluator]:
    if method not in _EVALUATORS:
        raise ValueError(
            f"Unknown evaluation method: {method!r}. Available: {sorted(_EVALUATORS)}"
        )
    return _EVALUATORS[method]


def build_evaluator(method: str, **kwargs) -> Evaluator:
    return _require_known_method(method)(**kwargs)


def evaluator_mode(method: str) -> str:
    return _require_known_method(method).mode


def evaluator_init_params(method: str) -> set[str]:
    """Return the set of parameter names the evaluator's ``__init__`` accepts,
    excluding runner-managed args (`self`, `results`, `groups`).

    Used by the CLI / benchmark runner to thread any matching
    `EvaluationConfig` field through to the evaluator constructor without
    hardcoding per-method dispatch — new evaluators get picked up as soon as
    they are registered in `_EVALUATORS`.
    """
    cls = _require_known_method(method)
    sig = inspect.signature(cls.__init__)
    return {name for name in sig.parameters if name not in _RUNNER_MANAGED_PARAMS}


__all__ = [
    "Evaluator",
    "LLMAsJudgeEvaluator",
    "LLMAsVerifierEvaluator",
    "build_evaluator",
    "evaluator_init_params",
    "evaluator_mode",
]
