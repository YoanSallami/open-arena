# License Apache 2.0: (c) 2026 Athena-Reply

"""DeepEval reward.

Wraps any `deepeval.metrics` metric (`GEval`, `AnswerRelevancyMetric`,
`HallucinationMetric`, `ToxicityMetric`, `BiasMetric`, …) into a
`synalinks.rewards.Reward` so it can be selected from `config.yaml`
under the snake_case name ``deep_eval``.

Install (optional): ``uv add deepeval``. The import is lazy — the module
itself loads even without DeepEval, but constructing a `DeepEval`
reward will raise `ImportError` until the dep is installed.

I/O contract:

- `y_true` / `y_pred` are synalinks `JsonDataModel` chat-message rows.
  The `input_field` (default ``"content"``) is extracted from each and
  passed to DeepEval as `expected_output` (from `y_true`) and
  `actual_output` (from `y_pred`).
- DeepEval's `LLMTestCase` requires an `input`. The open-arena CLI
  builds the eval program with `ChainOfThought(return_inputs=True)`,
  so `y_pred` carries the original input messages under the
  ``messages`` field; this reward extracts the **user-role messages**
  from there and concatenates their `content` as the `LLMTestCase`
  input. If `messages` is absent (e.g. a project that opts out of
  `return_inputs`), the `placeholder_input` is used as a fallback.
- The metric's `score` is returned clamped to `[0, 1]`.

YAML usage::

    - name: deep_eval
      alias: deep_eval_correctness
      metric: GEval
      metric_kwargs:
        name: Correctness
        criteria: |
          Determine whether the actual output matches the expected
          output in meaning, ignoring formatting differences.
        evaluation_params: [input, actual_output, expected_output]
        model: gpt-4o          # any litellm-routed model name
        threshold: 0.5
      # No `in_mask`: the judge needs the full row (input messages +
      # prediction). `in_mask: [content]` would discard the `messages`
      # field and the judge would lose the prompt.
"""

import asyncio

from synalinks.src.rewards.reward import Reward


_PARAM_ALIASES = {
    "input": "INPUT",
    "actual_output": "ACTUAL_OUTPUT",
    "expected_output": "EXPECTED_OUTPUT",
    "context": "CONTEXT",
    "retrieval_context": "RETRIEVAL_CONTEXT",
    "tools_called": "TOOLS_CALLED",
    "expected_tools": "EXPECTED_TOOLS",
    "reasoning": "REASONING",
}


def _resolve_evaluation_params(values):
    """Map snake_case YAML strings to `LLMTestCaseParams` enum members."""
    from deepeval.test_case import LLMTestCaseParams

    resolved = []
    for v in values:
        if isinstance(v, LLMTestCaseParams):
            resolved.append(v)
            continue
        if not isinstance(v, str):
            raise ValueError(
                f"`evaluation_params` entries must be strings or "
                f"LLMTestCaseParams; got {type(v).__name__}: {v!r}"
            )
        key = _PARAM_ALIASES.get(v.lower(), v.upper())
        try:
            resolved.append(LLMTestCaseParams[key])
        except KeyError as e:
            valid = ", ".join(sorted(_PARAM_ALIASES))
            raise ValueError(
                f"Unknown evaluation_params entry {v!r}. Valid: {valid}"
            ) from e
    return resolved


def _build_metric(metric, metric_kwargs):
    """Resolve `metric` (str class name or instance) into a metric object."""
    if not isinstance(metric, str):
        return metric

    try:
        from deepeval import metrics as _metrics
    except ImportError as e:
        raise ImportError(
            "DeepEval is not installed. Run `uv add deepeval` to enable "
            "the `deep_eval` reward."
        ) from e

    cls = getattr(_metrics, metric, None)
    if cls is None:
        available = sorted(n for n in dir(_metrics) if not n.startswith("_"))
        raise ValueError(
            f"DeepEval has no metric class named {metric!r}. "
            f"Available: {available}"
        )

    kwargs = dict(metric_kwargs or {})
    if "evaluation_params" in kwargs:
        kwargs["evaluation_params"] = _resolve_evaluation_params(
            kwargs["evaluation_params"]
        )
    return cls(**kwargs)


def _extract(dm, field):
    """Pull a string value out of a (masked) JsonDataModel."""
    if dm is None:
        return None
    if hasattr(dm, "get"):
        value = dm.get(field)
        if value is not None:
            return str(value)
    data = dm.get_json() if hasattr(dm, "get_json") else None
    if isinstance(data, dict):
        value = data.get(field)
        if value is not None:
            return str(value)
        return str(data)
    return str(dm)


def _extract_input_messages(dm, messages_field, user_role):
    """Pull the original prompt text out of `y_pred.messages`.

    Concatenates the `content` of every message whose role matches
    `user_role`. Falls back to all message contents if no user-role
    messages are present (e.g. system-only prompts). Returns `None`
    if the data model has no `messages_field`.
    """
    if dm is None:
        return None
    messages = None
    if hasattr(dm, "get"):
        messages = dm.get(messages_field)
    if messages is None:
        data = dm.get_json() if hasattr(dm, "get_json") else None
        if isinstance(data, dict):
            messages = data.get(messages_field)
    if not isinstance(messages, list) or not messages:
        return None

    def _role_content(msg):
        if isinstance(msg, dict):
            return msg.get("role"), msg.get("content")
        return getattr(msg, "role", None), getattr(msg, "content", None)

    user_chunks = [
        str(content)
        for role, content in (_role_content(m) for m in messages)
        if role == user_role and content is not None
    ]
    if user_chunks:
        return "\n\n".join(user_chunks)
    all_chunks = [
        str(content)
        for _role, content in (_role_content(m) for m in messages)
        if content is not None
    ]
    return "\n\n".join(all_chunks) if all_chunks else None


class DeepEval(Reward):
    """Wrap a DeepEval metric as a synalinks reward.

    See module docstring for the full I/O contract and YAML example.

    Args:
        metric: Either a DeepEval metric class name (string, e.g.
            ``"GEval"``) resolved against `deepeval.metrics`, or a
            pre-built metric instance for programmatic use.
        metric_kwargs: kwargs forwarded to the metric constructor when
            `metric` is a string. `evaluation_params` entries are
            auto-resolved from snake_case strings to
            `LLMTestCaseParams`.
        input_field: Field of the `y_true` / `y_pred` data model to
            extract as the DeepEval `expected_output` /
            `actual_output` string. Defaults to ``"content"`` (the
            chat-message body).
        messages_field: Field on `y_pred` that holds the original
            input messages (when the CoT was built with
            `return_inputs=True`). Defaults to ``"messages"``.
        user_role: Role to look for inside `messages_field` when
            extracting the input prompt. Defaults to ``"user"``.
        placeholder_input: Fallback `LLMTestCase.input` used only when
            `messages_field` is absent or empty (e.g. dataset opted
            out of `return_inputs`). Defaults to ``""``.
        name: Reward instance name.
        in_mask: list of keys to keep on `y_true` / `y_pred`. Leave
            unset for judge use — masking out `messages` would hide
            the prompt from the metric.
        out_mask: list of keys to drop.
    """

    def __init__(
        self,
        metric="GEval",
        metric_kwargs=None,
        input_field="content",
        messages_field="messages",
        user_role="user",
        placeholder_input="",
        name="deep_eval",
        in_mask=None,
        out_mask=None,
    ):
        super().__init__(name=name, in_mask=in_mask, out_mask=out_mask)
        self.metric = metric
        self.metric_kwargs = dict(metric_kwargs or {})
        self.input_field = input_field
        self.messages_field = messages_field
        self.user_role = user_role
        self.placeholder_input = placeholder_input
        self._metric_instance = _build_metric(metric, self.metric_kwargs)

    async def call(self, y_true, y_pred):
        if not y_pred:
            return 0.0

        from deepeval.test_case import LLMTestCase

        actual_output = _extract(y_pred, self.input_field) or ""
        expected_output = _extract(y_true, self.input_field) if y_true else None
        prompt = _extract_input_messages(
            y_pred, self.messages_field, self.user_role
        )

        test_case = LLMTestCase(
            input=prompt if prompt is not None else self.placeholder_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )

        a_measure = getattr(self._metric_instance, "a_measure", None)
        if a_measure is not None and asyncio.iscoroutinefunction(a_measure):
            await a_measure(test_case)
        else:
            await asyncio.to_thread(self._metric_instance.measure, test_case)

        score = getattr(self._metric_instance, "score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if score != score:  # NaN
            score = 0.0
        return max(0.0, min(1.0, score))

    def get_config(self):
        return {
            "metric": self.metric if isinstance(self.metric, str) else None,
            "metric_kwargs": self.metric_kwargs,
            "input_field": self.input_field,
            "messages_field": self.messages_field,
            "user_role": self.user_role,
            "placeholder_input": self.placeholder_input,
            "name": self.name,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
