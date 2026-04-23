from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class DatasetConfig(BaseModel):
    """Global dataset configuration. Rows are fetched via a provider adapter
    and shaped into Records using Jinja2 `input` / `expected_output` templates."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Logical dataset name (used for Langfuse dataset creation).")
    source: dict[str, Any] = Field(
        ...,
        description="Provider-specific source config. Must contain `provider` (e.g. 'local', 'huggingface').",
    )
    input: str = Field(..., min_length=1, description="Jinja2 template rendered per row to produce the user input.")
    expected_output: str = Field(..., min_length=1, description="Jinja2 template rendered per row to produce the ground truth.")
    limit: int | None = Field(default=None, ge=1, description="Optional cap on number of rows.")
    description: str | None = Field(default=None, description="Optional human-readable description.")


class MCPServer(BaseModel):
    name: str = Field(..., min_length=1)
    url: HttpUrl = Field(...)


class LiteLLMConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(..., min_length=1)
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Reasoning/thinking budget for reasoning models (o1, deepseek-r1, "
            "qwen3, ...). Typical values: 'low', 'medium', 'high', 'disable'. "
            "See https://docs.litellm.ai/docs/reasoning_content."
        ),
    )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    litellm: LiteLLMConfig = Field(...)
    mcp: list[MCPServer] | None = Field(default=None)
    timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Per-row wall-clock timeout for LLM calls. Row fails with a timeout error if exceeded.",
    )


class CriterionConfig(BaseModel):
    """A single evaluation criterion as in Kwok et al. 2026.

    `name` is the short label (e.g. "Root Cause Analysis") shown in the
    prompt header. `description` is the long rubric paragraph the verifier
    reads to decide how to score along this dimension.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class ScenarioRubricBand(BaseModel):
    """One band of a 0-10 scoring rubric.

    `score_range` is an inclusive `[lo, hi]` pair; `expected_outcome` is the
    rubric paragraph the judge reads when deciding whether to place a response
    in this band.
    """

    model_config = ConfigDict(extra="forbid")

    score_range: list[int] = Field(..., description="Inclusive [lo, hi] integer band in [0, 10].")
    expected_outcome: str = Field(..., min_length=1)

    @field_validator("score_range")
    @classmethod
    def _validate_score_range(cls, v: list[int]) -> list[int]:
        if len(v) != 2:
            raise ValueError("score_range must be a length-2 list [lo, hi].")
        lo, hi = v
        if not (0 <= lo <= hi <= 10):
            raise ValueError(
                f"score_range must satisfy 0 <= lo <= hi <= 10; got [{lo}, {hi}]."
            )
        return v


class ScenarioMetricConfig(BaseModel):
    """A single rubric-driven metric scored on a 0-10 scale.

    The rubric must cover the full 0-10 range with no gaps and no overlaps
    (bands sorted by `lo`; `hi_i + 1 == lo_{i+1}`).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    criteria: str = Field(..., min_length=1)
    rubric: list[ScenarioRubricBand] = Field(..., min_length=1)
    threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Normalized pass threshold in [0, 1] (applied after score/10).",
    )

    @model_validator(mode="after")
    def _validate_rubric_covers_0_to_10(self) -> "ScenarioMetricConfig":
        bands = sorted(self.rubric, key=lambda b: b.score_range[0])
        if bands[0].score_range[0] != 0:
            raise ValueError(
                f"rubric must start at 0; first band starts at {bands[0].score_range[0]}."
            )
        if bands[-1].score_range[1] != 10:
            raise ValueError(
                f"rubric must end at 10; last band ends at {bands[-1].score_range[1]}."
            )
        for prev, curr in zip(bands, bands[1:]):
            prev_hi = prev.score_range[1]
            curr_lo = curr.score_range[0]
            if curr_lo != prev_hi + 1:
                reason = "overlaps" if curr_lo <= prev_hi else "leaves a gap"
                raise ValueError(
                    f"rubric {reason} between [{prev.score_range[0]}, {prev_hi}] and "
                    f"[{curr_lo}, {curr.score_range[1]}]; bands must be contiguous."
                )
        return self


class ScenarioConfig(BaseModel):
    """Rubric-based evaluation scenario declared inline under `evaluation.scenario`.

    Consumed by the judge-panel evaluator: panelists score each metric against
    its rubric bands, agreement is computed per metric, and disagreeing metrics
    escalate to a smart judge.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_trace_fields: list[str] = Field(..., min_length=1)
    scenario_metrics: list[ScenarioMetricConfig] = Field(..., min_length=1)

    @field_validator("required_trace_fields")
    @classmethod
    def _dedupe_and_reject_empty(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for item in v:
            if not item or not item.strip():
                raise ValueError("required_trace_fields entries must be non-empty strings.")
            if item not in seen:
                seen.append(item)
        return seen


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., min_length=1)
    litellm: LiteLLMConfig = Field(...)
    score_name: str | None = Field(default="evaluation_score")
    max_concurrency: int | None = Field(default=10, ge=1)
    system_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "System prompt used in reference mode (when the row has a "
            "non-empty expected_output). Required — see config.example.yaml "
            "for ready-to-copy templates per evaluator method."
        ),
    )
    system_prompt_no_reference: str = Field(
        ...,
        min_length=1,
        description=(
            "System prompt used when the row has no expected_output. "
            "Required — see config.example.yaml for ready-to-copy templates."
        ),
    )
    granularity: int | None = Field(
        default=None,
        ge=2,
        le=26,
        description=(
            "'llm_as_verifier' only: number of score letters (A..) used "
            "for logprob-based scoring (G). Typical: 8."
        ),
    )
    repeats: int | None = Field(
        default=None,
        ge=1,
        description=(
            "'llm_as_verifier' only: number of repeated verification "
            "samples to average per pair-criterion call (K). Typical: 1-16."
        ),
    )
    criteria: list[CriterionConfig] | None = Field(
        default=None,
        min_length=1,
        description=(
            "'llm_as_verifier' only: list of criteria to decompose scoring "
            "over (C). Each item is {name, description} where `description` "
            "is the long rubric paragraph the verifier reads (Kwok et al. "
            "2026). Each criterion is scored in its own pairwise call and "
            "the per-pair reward is the mean across criteria. Omit for a "
            "single holistic call."
        ),
    )
    max_retries: int | None = Field(
        default=None,
        ge=1,
        description=(
            "'llm_as_judge' and 'judge_panel': retry budget on structured-output "
            "parse failures (OutputParserException). Default: 3."
        ),
    )
    timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Per-item wall-clock timeout for evaluator calls. Item fails with a timeout error if exceeded.",
    )
    panelists: list[LiteLLMConfig] | None = Field(
        default=None,
        description=(
            "'judge_panel' only: list of panelist LLM configs (≥ 2). Each "
            "panelist scores every scenario metric independently; the panel's "
            "consensus is the mean of agreeing panelists."
        ),
    )
    smart_litellm: LiteLLMConfig | None = Field(
        default=None,
        description=(
            "'judge_panel' only (required): smart-model config used to "
            "escalate when a metric's panelist spread exceeds "
            "(1 - metric.threshold), or when fewer than two panelists "
            "survived."
        ),
    )
    scenario: ScenarioConfig | None = Field(
        default=None,
        description=(
            "'judge_panel' only: inline scenario definition (rubric metrics + "
            "required_trace_fields). Required when method is 'judge_panel'."
        ),
    )

    @field_validator("criteria", mode="before")
    @classmethod
    def _reject_string_criteria(cls, v):
        # Catch the common foot-gun of `criteria: ["correctness", "clarity"]`
        # left over from the pre-decomposition format and surface a clear
        # error instead of an opaque pydantic validation message.
        if isinstance(v, list) and any(isinstance(item, str) for item in v):
            raise ValueError(
                "criteria entries must be objects with `name` and `description` "
                "fields (not bare strings). See config.example.yaml."
            )
        return v

    @model_validator(mode="after")
    def _validate_judge_panel_fields(self) -> "EvaluationConfig":
        if self.method == "judge_panel":
            if not self.panelists or len(self.panelists) < 2:
                raise ValueError(
                    "method='judge_panel' requires `panelists` with at least two entries."
                )
            if self.scenario is None:
                raise ValueError(
                    "method='judge_panel' requires an inline `scenario` block "
                    "(rubric metrics + required_trace_fields)."
                )
            if self.smart_litellm is None:
                raise ValueError(
                    "method='judge_panel' requires a `smart_litellm` block "
                    "(escalation model for panel disagreement or insufficient survivors)."
                )
            return self

        # Non-judge-panel methods: the judge-panel-specific fields should not
        # be set at all (otherwise they're silently ignored and the config
        # hides a copy-paste bug).
        extra_fields = [
            name
            for name in ("panelists", "smart_litellm", "scenario")
            if getattr(self, name) is not None
        ]
        if extra_fields:
            raise ValueError(
                f"Fields {extra_fields} are only valid when method='judge_panel'; "
                f"got method={self.method!r}."
            )
        return self


class ExperimentsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig = Field(...)
    system_prompt: str = Field(..., min_length=1)
    experiments: list[ExperimentConfig] = Field(..., min_length=1)
    evaluation: EvaluationConfig = Field(...)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentsFile":
        config_file = Path(yaml_path)
        if not config_file.exists():
            raise FileNotFoundError(f"File not found: {yaml_path}")
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
