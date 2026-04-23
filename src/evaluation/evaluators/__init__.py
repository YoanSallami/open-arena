from src.evaluation.base import Evaluator
from src.evaluation.evaluators.judge_panel import JudgePanelEvaluator
from src.evaluation.evaluators.llm_as_judge import LLMAsJudgeEvaluator
from src.evaluation.evaluators.llm_as_verifier import LLMAsVerifierEvaluator

_EVALUATORS: dict[str, type[Evaluator]] = {
    "llm_as_judge": LLMAsJudgeEvaluator,
    "llm_as_verifier": LLMAsVerifierEvaluator,
    "judge_panel": JudgePanelEvaluator,
}


def build_evaluator(method: str, **kwargs) -> Evaluator:
    if method not in _EVALUATORS:
        raise ValueError(f"Unknown evaluation method: {method!r}. Available: {sorted(_EVALUATORS)}")
    return _EVALUATORS[method](**kwargs)


def evaluator_mode(method: str) -> str:
    if method not in _EVALUATORS:
        raise ValueError(f"Unknown evaluation method: {method!r}. Available: {sorted(_EVALUATORS)}")
    return _EVALUATORS[method].mode


__all__ = [
    "Evaluator",
    "JudgePanelEvaluator",
    "LLMAsJudgeEvaluator",
    "LLMAsVerifierEvaluator",
    "build_evaluator",
    "evaluator_mode",
]
