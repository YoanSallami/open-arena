from collections.abc import Iterator
from typing import Any

from langsmith import Client

from src.datasets.base import Dataset


class LangSmithDataset(Dataset):
    """
    Fetch rows from an existing LangSmith dataset.

    Each LangSmith example's `inputs` / `outputs` dicts and `metadata` keys
    are exposed to the Jinja templates as top-level variables, alongside
    `ls_example_id` / `ls_dataset_id` / `ls_dataset_name`. Because LangSmith
    examples don't share ids with Langfuse, rows still go through the normal
    upload path so the executor can link traces to Langfuse items.
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
        client = Client()
        for example in client.list_examples(dataset_name=self.dataset_name, limit=self.limit):
            yield {
                **(example.metadata or {}),
                "inputs": example.inputs,
                "outputs": example.outputs,
                "ls_example_id": str(example.id),
                "ls_dataset_id": str(example.dataset_id),
                "ls_dataset_name": self.dataset_name,
            }
