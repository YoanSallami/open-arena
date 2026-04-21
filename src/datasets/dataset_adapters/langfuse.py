from collections.abc import Iterator
from typing import Any

from langfuse import get_client

from src.datasets.base import Dataset


class LangfuseDataset(Dataset):
    """
    Fetch rows from an existing Langfuse dataset.

    Each Langfuse item's `input`, `expected_output`, and `metadata` fields are
    exposed to the Jinja templates as top-level variables, alongside
    `lf_item_id` / `lf_dataset_id` / `lf_dataset_name`. Metadata keys are
    spread at the top level so templates can reference them directly; any
    key not consumed by a template flows through as metadata — which lets
    the execution layer link traces back to the original Langfuse item
    without re-uploading.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        dataset_name: str | None = None,
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.dataset_name = dataset_name or name

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        langfuse = get_client()
        remote = langfuse.get_dataset(self.dataset_name)
        for item in remote.items:
            yield {
                **(item.metadata or {}),
                "input": item.input,
                "expected_output": item.expected_output,
                "lf_item_id": item.id,
                "lf_dataset_id": item.dataset_id,
                "lf_dataset_name": item.dataset_name,
            }
