# License Apache 2.0: (c) 2026 Athena-Reply

import json as _json

from synalinks.src.backend.common.json_data_model import JsonDataModel


class Dataset:
    """Base class for streaming datasets compatible with synalinks trainers.

    Synalinks' `Trainer.fit/evaluate/predict(x=...)` accepts a Python
    generator that yields `(inputs,)` or `(inputs, targets)` tuples — one
    tuple per batch. See
    `synalinks/src/trainers/data_adapters/generator_data_adapter.py` and
    the dispatch in `synalinks/src/trainers/data_adapters/__init__.py`.

    Subclasses implement `__iter__` as a generator method (using `yield`)
    that produces those tuples. Calling the dataset returns a fresh
    `types.GeneratorType` suitable for synalinks:

    ```python
    program.evaluate(x=my_dataset())
    ```

    The shape of the per-row input/target objects is controlled by either
    a Python `DataModel` class (`input_data_model` / `output_data_model`)
    OR a raw JSON Schema (`input_schema` / `output_schema`). The two are
    mutually exclusive on each side. With a class, rows are validated via
    `cls.model_validate_json(rendered)`. With a schema, rows are wrapped
    as `JsonDataModel(schema=..., json=json.loads(rendered))` — the schema
    flows directly into the LM as a structured-output constraint, so any
    JSON Schema feature (enum, const, oneOf, ...) is supported.

    Args:
        input_data_model (DataModel): Python class describing batch inputs.
            Defaults to `synalinks.ChatMessages` in subclasses when neither
            this nor `input_schema` is provided.
        input_schema (dict | str): Raw JSON Schema for batch inputs. May be
            given as a dict or as a JSON-encoded string. Mutually exclusive
            with `input_data_model`.
        input_template (str): Jinja2 template string used to render raw
            rows into the input shape. Required.
        output_data_model (DataModel): Python class describing batch targets.
            Defaults to `synalinks.ChatMessage` in subclasses when neither
            this nor `output_schema` is provided.
        output_schema (dict | str): Raw JSON Schema for batch targets.
            Mutually exclusive with `output_data_model`.
        output_template (str): Jinja2 template string used to render raw
            rows into the target shape. Required.
        batch_size (int): Number of examples per yielded batch. `None` lets
            the subclass decide (or yield the whole dataset as one batch).
        limit (int): Optional. Maximum number of examples to iterate over.
            `None` (default) means no cap. Useful for capping huge or
            streaming sources for quick experiments / smoke tests.
        **kwargs: Provider-specific fields forwarded by subclasses (e.g. HF
            dataset name, split, revision, API key, file path, ...).
    """

    def __init__(
        self,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        batch_size=None,
        limit: int = None,
        **kwargs,
    ):
        if input_template is None:
            raise ValueError("`input_template` is required (Jinja2 template).")
        if output_template is None:
            raise ValueError("`output_template` is required (Jinja2 template).")
        if input_data_model is not None and input_schema is not None:
            raise ValueError(
                "Pass either `input_data_model` or `input_schema`, not both."
            )
        if output_data_model is not None and output_schema is not None:
            raise ValueError(
                "Pass either `output_data_model` or `output_schema`, not both."
            )
        self.input_data_model = input_data_model
        self.input_schema = _coerce_schema(input_schema)
        self.input_template = input_template
        self.output_data_model = output_data_model
        self.output_schema = _coerce_schema(output_schema)
        self.output_template = output_template
        self.batch_size = batch_size
        self.limit = limit

    def _make_input(self, rendered: str):
        if self.input_schema is not None:
            return JsonDataModel(schema=self.input_schema, json=_json.loads(rendered))
        return self.input_data_model.model_validate_json(rendered)

    def _make_target(self, rendered: str):
        if self.output_schema is not None:
            return JsonDataModel(schema=self.output_schema, json=_json.loads(rendered))
        return self.output_data_model.model_validate_json(rendered)

    def __iter__(self):
        """Yield one `(inputs,)` or `(inputs, targets)` tuple per batch."""
        raise NotImplementedError

    def __call__(self):
        """Return a fresh generator over the dataset's batches."""
        return iter(self)

    def __len__(self):
        """Number of batches, if known. Override when the size is finite."""
        raise NotImplementedError


def _coerce_schema(schema):
    """Accept a JSON Schema as either a dict or a JSON-encoded string."""
    if schema is None:
        return None
    if isinstance(schema, str):
        return _json.loads(schema)
    if isinstance(schema, dict):
        return schema
    raise TypeError(
        f"`schema` must be a dict or JSON string; got {type(schema).__name__}."
    )
