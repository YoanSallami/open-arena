# License Apache 2.0: (c) 2026 Athena-Reply

from datetime import datetime
from datetime import timezone

import jinja2
import numpy as np
import synalinks

from src.datasets.dataset import Dataset


def _coerce_as_of(as_of):
    """Pass tag strings through; coerce naive datetimes to UTC."""
    if as_of is None or isinstance(as_of, str):
        return as_of
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=timezone.utc)
        return as_of
    raise TypeError(
        f"`as_of` must be str (version tag), datetime, or None — "
        f"got {type(as_of).__name__}"
    )


class LangSmithDataset(Dataset):
    """Streaming dataset backed by a LangSmith dataset.

    See https://docs.smith.langchain.com/evaluation/concepts#datasets. Each
    LangSmith `Example` exposes `inputs` (dict), `outputs` (dict | None),
    `metadata` (dict), and `id`; those names become the variables available
    to the Jinja2 `input_template` / `output_template`. Auth comes from env
    vars (`LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`); set them in `.env`.

    Example:

    ```python
    ds = LangSmithDataset(
        dataset_name="mmlu_subset",
        input_template='{"messages":[{"role":"user","content":{{ inputs.question | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ outputs.answer | tojson }}}',
        batch_size=1,
    )
    program.evaluate(x=ds)
    ```

    Args:
        dataset_name (str): The LangSmith dataset name.
        as_of (str | datetime): Optional. Point-in-time snapshot — either a
            version tag string (e.g. `"prod"`) or a UTC tz-aware datetime
            for time-travel. Defaults to `None` (latest state).
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many items are
            consumed; required for `__len__` since LangSmith doesn't
            expose a count under `as_of` without a full scan.
    """

    def __init__(
        self,
        dataset_name,
        *,
        as_of=None,
        input_data_model=None,
        input_template=None,
        output_data_model=None,
        output_template=None,
        batch_size=1,
        limit: int = None,
    ):
        super().__init__(
            input_data_model=input_data_model or synalinks.ChatMessages,
            input_template=input_template,
            output_data_model=output_data_model or synalinks.ChatMessage,
            output_template=output_template,
            batch_size=batch_size,
            limit=limit,
        )
        self.dataset_name = dataset_name
        self.as_of = _coerce_as_of(as_of)

        # Imported lazily so the project doesn't require the langsmith package
        # unless this dataset type is actually used.
        from langsmith import Client

        self._client = Client()

        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self._input_tmpl = env.from_string(input_template)
        self._output_tmpl = env.from_string(output_template)

    def _iter_items(self):
        for ex in self._client.list_examples(
            dataset_name=self.dataset_name,
            as_of=self.as_of,
        ):
            yield {
                "inputs": ex.inputs or {},
                "outputs": ex.outputs or {},
                "metadata": ex.metadata or {},
                "id": str(ex.id),
            }

    def __iter__(self):
        x_buf, y_buf = [], []
        seen = 0
        for row in self._iter_items():
            if self.limit is not None and seen >= self.limit:
                break
            seen += 1
            x = self.input_data_model.model_validate_json(
                self._input_tmpl.render(**row)
            )
            y = self.output_data_model.model_validate_json(
                self._output_tmpl.render(**row)
            )
            x_buf.append(x)
            y_buf.append(y)
            if len(x_buf) >= self.batch_size:
                yield (
                    np.array(x_buf, dtype="object"),
                    np.array(y_buf, dtype="object"),
                )
                x_buf, y_buf = [], []
        if x_buf:
            yield (
                np.array(x_buf, dtype="object"),
                np.array(y_buf, dtype="object"),
            )

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "LangSmith dataset length is unknown without a full scan; "
                "set `limit` for a bounded epoch."
            )
        return (self.limit + self.batch_size - 1) // self.batch_size
