from collections.abc import Iterator
from typing import Any

from phoenix.client import Client

from src.datasets.base import Dataset


class PhoenixDataset(Dataset):
    """
    Fetch rows from an existing Arize Phoenix dataset.

    Each Phoenix example's `input` / `output` dicts and `metadata` keys are
    exposed to the Jinja templates as top-level variables, alongside
    `phoenix_example_id` / `phoenix_dataset_id` / `phoenix_dataset_name`.
    Rows still flow through the Langfuse upload path since Phoenix and
    Langfuse are separate systems with separate ids.

    `base_url` / `api_key` are passed straight to `phoenix.client.Client`;
    omit them to fall back to the `PHOENIX_COLLECTOR_ENDPOINT` and
    `PHOENIX_CLIENT_HEADERS` environment variables.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        dataset_name: str | None = None,
        version_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.dataset_name = dataset_name or name
        self.version_id = version_id
        self.base_url = base_url
        self.api_key = api_key

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        client = Client(base_url=self.base_url, api_key=self.api_key)
        dataset = client.datasets.get_dataset(
            dataset=self.dataset_name,
            version_id=self.version_id,
        )
        for example in dataset.examples:
            yield {
                **(example.get("metadata") or {}),
                "input": example.get("input"),
                "output": example.get("output"),
                "phoenix_example_id": example.get("id"),
                "phoenix_dataset_id": dataset.id,
                "phoenix_dataset_name": dataset.name,
            }
