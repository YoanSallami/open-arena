from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass
class EvaluationResult:
    input: str
    expected_output: str
    output: str
    model_name: str
    experiment_name: str = ""
    score: float | None = None
    explanation: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class JudgeResponse(BaseModel):
    thinking: str = Field(..., description="Step-by-step reasoning before scoring")
    score: Literal[1, 2, 3, 4, 5] = Field(..., description="Integer score from 1 to 5")


class PanelJudgeResponse(BaseModel):
    """Structured output for a single panelist scoring a single rubric metric.

    Scores are integers in [0, 10] matching the rubric band boundaries; the
    evaluator normalizes to [0, 1] before returning.
    """

    reasoning: str = Field(..., description="Short rationale for the chosen rubric band.")
    score: int = Field(..., ge=0, le=10, description="Integer score on the 0-10 rubric.")
