# License Apache 2.0: (c) 2026 Athena-Reply

import jinja2
import numpy as np
import synalinks

from src.datasets.dataset import Dataset


class PhoenixDataset(Dataset):
    """Streaming dataset backed by an Arize Phoenix dataset.

    See https://docs.arize.com/phoenix/datasets-and-experiments. Each
    Phoenix `DatasetExample` exposes `id`, `input` (dict), `output` (dict),
    and `metadata` (dict); those names become the variables available to
    the Jinja2 `input_template` / `output_template`. Auth comes from env
    vars (`PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT` or
    `PHOENIX_HOST` + `PHOENIX_PORT`); set them in `.env`.

    Example:

    ```python
    ds = PhoenixDataset(
        dataset_name="mmlu_subset",
        input_template='{"messages":[{"role":"user","content":{{ input.question | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ output.answer | tojson }}}',
        batch_size=1,
    )
    program.evaluate(x=ds)
    ```

    Args:
        dataset_name (str): The Phoenix dataset name (or ID).
        version_id (str): Optional. The version ID to pin items to a
            snapshot. Defaults to `None` (latest version).
        splits (list): Optional. List of split names to restrict to.
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many items are
            consumed. Phoenix returns the full version in one call, so
            this is enforced client-side.
    """

    def __init__(
        self,
        dataset_name,
        *,
        version_id=None,
        splits=None,
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
        self.version_id = version_id
        self.splits = splits

        # Imported lazily so the project doesn't require the phoenix client
        # unless this dataset type is actually used.
        from phoenix.client import Client

        self._client = Client()
        self._dataset = self._client.datasets.get_dataset(
            dataset=dataset_name,
            version_id=version_id,
            splits=splits,
        )

        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self._input_tmpl = env.from_string(input_template)
        self._output_tmpl = env.from_string(output_template)

    def _iter_items(self):
        for ex in self._dataset.examples:
            yield {
                "id": ex["id"],
                "input": ex.get("input") or {},
                "output": ex.get("output") or {},
                "metadata": ex.get("metadata") or {},
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
        n = self.limit if self.limit is not None else self._dataset.example_count
        return (n + self.batch_size - 1) // self.batch_size
