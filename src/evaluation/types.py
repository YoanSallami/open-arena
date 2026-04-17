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
    score: Literal[1, 2, 3, 4, 5] = Field(..., description="Integer score from 1 to 5")
    explanation: str = Field(..., description="Explanation for the score")
