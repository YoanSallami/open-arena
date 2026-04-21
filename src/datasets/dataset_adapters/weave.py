from collections.abc import Iterator
from typing import Any

import weave

from src.datasets.base import Dataset


class WeaveDataset(Dataset):
    """
    Fetch rows from an existing Weights & Biases Weave dataset.

    Weave rows are free-form dicts — whatever keys you stored are exposed
    directly as top-level Jinja variables (e.g. `{{ input }}`,
    `{{ expected }}`, or any custom field). `weave_dataset_name` /
    `weave_dataset_version` / `weave_project` are injected for traceability.

    `weave.init(project)` is called on each iteration, which is idempotent;
    `WANDB_API_KEY` must be set in the environment.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        project: str,
        dataset_name: str | None = None,
        version: str = "latest",
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.project = project
        self.dataset_name = dataset_name or name
        self.version = version

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        weave.init(self.project)
        dataset = weave.ref(f"{self.dataset_name}:{self.version}").get()
        for row in dataset.rows:
            yield {
                **dict(row),
                "weave_dataset_name": self.dataset_name,
                "weave_dataset_version": self.version,
                "weave_project": self.project,
            }
