# License Apache 2.0: (c) 2026 Athena-Reply

import jinja2
import numpy as np
import synalinks

from src.datasets.dataset import Dataset


class BraintrustDataset(Dataset):
    """Streaming dataset backed by a Braintrust dataset.

    See https://www.braintrust.dev/docs/guides/datasets. Each Braintrust
    dataset record exposes `id`, `input` (any JSON), `expected` (any JSON),
    and `metadata` (dict); those names become the variables available to
    the Jinja2 `input_template` / `output_template`. Auth comes from env
    vars (`BRAINTRUST_API_KEY`, optional `BRAINTRUST_APP_URL`,
    `BRAINTRUST_ORG_NAME`); set them in `.env`.

    Example:

    ```python
    ds = BraintrustDataset(
        project="my-project",
        dataset_name="mmlu_subset",
        input_template='{"messages":[{"role":"user","content":{{ input.question | tojson }}}]}',
        output_template='{"role":"assistant","content":{{ expected.answer | tojson }}}',
        batch_size=1,
    )
    program.evaluate(x=ds)
    ```

    Args:
        project (str): The Braintrust project name (or use `project_id`).
        dataset_name (str): The Braintrust dataset name within that
            project.
        version (str | int): Optional. Transaction id to pin items to a
            snapshot. Defaults to `None` (latest).
        project_id (str): Optional. Alternative to `project` — the
            project's UUID. Takes precedence over `project` when both are
            set.
        org_name (str): Optional. Organization to scope to (for users in
            multiple orgs). Falls back to `BRAINTRUST_ORG_NAME`.
        fetch_batch_size (int): Records per Braintrust API batch. Defaults
            to `1000` (the SDK default).
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_template (str): See `Dataset`.
        output_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessage`.
        output_template (str): See `Dataset`.
        batch_size (int): Examples per yielded batch. Defaults to `1`.
        limit (int): Optional. See `Dataset`. Caps how many records are
            consumed.
    """

    def __init__(
        self,
        project=None,
        dataset_name=None,
        *,
        version=None,
        project_id=None,
        org_name=None,
        fetch_batch_size=None,
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
        if project is None and project_id is None:
            raise ValueError("BraintrustDataset requires `project` or `project_id`.")
        if dataset_name is None:
            raise ValueError("BraintrustDataset requires `dataset_name`.")
        self.project = project
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.version = version
        self.org_name = org_name
        self.fetch_batch_size = fetch_batch_size

        # Imported lazily so the project doesn't require the braintrust
        # package unless this dataset type is actually used.
        from braintrust import init_dataset

        self._dataset = init_dataset(
            project=project,
            project_id=project_id,
            name=dataset_name,
            version=version,
            org_name=org_name,
        )

        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        self._input_tmpl = env.from_string(input_template)
        self._output_tmpl = env.from_string(output_template)

    def _iter_items(self):
        kwargs = {}
        if self.fetch_batch_size is not None:
            kwargs["batch_size"] = self.fetch_batch_size
        for rec in self._dataset.fetch(**kwargs):
            yield {
                "id": rec.get("id"),
                "input": rec.get("input"),
                "expected": rec.get("expected"),
                "metadata": rec.get("metadata") or {},
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
                "Braintrust dataset length is unknown without a full scan; "
                "set `limit` for a bounded epoch."
            )
        return (self.limit + self.batch_size - 1) // self.batch_size
