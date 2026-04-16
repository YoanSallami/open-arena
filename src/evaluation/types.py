from dataclasses import dataclass, field
from typing import Any

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
    score: int = Field(..., description="Integer score from 1 to 5", ge=1, le=5)
    explanation: str = Field(..., description="Explanation for the score")
