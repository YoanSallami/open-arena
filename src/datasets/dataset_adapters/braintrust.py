from collections.abc import Iterator
from typing import Any

from braintrust import init_dataset

from src.datasets.base import Dataset


class BraintrustDataset(Dataset):
    """
    Fetch rows from an existing Braintrust dataset.

    Braintrust records expose `input`, `expected`, optional `metadata`, and
    `tags`. Metadata keys are spread at the top level so templates can
    reference them directly; `braintrust_record_id` / `braintrust_dataset_id`
    / `braintrust_dataset_name` / `braintrust_project` are injected for
    traceability. Rows still go through the Langfuse upload path.

    `api_key` / `app_url` / `org_name` fall back to the standard
    `BRAINTRUST_API_KEY` / `BRAINTRUST_APP_URL` env vars when omitted.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        project: str,
        dataset_name: str | None = None,
        version: str | int | None = None,
        api_key: str | None = None,
        app_url: str | None = None,
        org_name: str | None = None,
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.project = project
        self.dataset_name = dataset_name or name
        self.version = version
        self.api_key = api_key
        self.app_url = app_url
        self.org_name = org_name

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        dataset = init_dataset(
            project=self.project,
            name=self.dataset_name,
            version=self.version,
            api_key=self.api_key,
            app_url=self.app_url,
            org_name=self.org_name,
        )
        try:
            for record in dataset.fetch():
                metadata = record.get("metadata") or {}
                yield {
                    **metadata,
                    "input": record.get("input"),
                    "expected": record.get("expected"),
                    "tags": record.get("tags") or [],
                    "braintrust_record_id": record.get("id"),
                    "braintrust_dataset_id": record.get("dataset_id") or dataset.id,
                    "braintrust_dataset_name": dataset.name,
                    "braintrust_project": self.project,
                }
        finally:
            dataset.close()
