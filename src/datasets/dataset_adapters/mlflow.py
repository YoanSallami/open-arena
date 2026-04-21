from collections.abc import Iterator
from typing import Any

import mlflow
from mlflow.genai.datasets import get_dataset

from src.datasets.base import Dataset


class MLflowDataset(Dataset):
    """
    Fetch rows from an existing MLflow GenAI evaluation dataset.

    Uses `mlflow.genai.datasets.get_dataset(name=...)` and yields one row
    per record. Each record's `inputs`, `expectations`, and `tags` dicts
    are exposed to the Jinja templates as top-level variables, alongside
    `mlflow_record_id` / `mlflow_dataset_id` / `mlflow_dataset_name`.

    Requires an MLflow 3 tracking server with the GenAI datasets feature
    enabled (e.g. Databricks, or self-hosted MLflow 3 with the relevant
    extras). Set `tracking_uri` / `registry_uri` here or via the standard
    `MLFLOW_TRACKING_URI` / `MLFLOW_REGISTRY_URI` env vars.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        dataset_name: str | None = None,
        dataset_id: str | None = None,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.dataset_name = dataset_name or name
        self.dataset_id = dataset_id
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        dataset = (
            get_dataset(dataset_id=self.dataset_id)
            if self.dataset_id
            else get_dataset(name=self.dataset_name)
        )
        for record in dataset.to_df().to_dict(orient="records"):
            yield {
                "inputs": record.get("inputs"),
                "expectations": record.get("expectations"),
                "tags": record.get("tags"),
                "mlflow_record_id": record.get("dataset_record_id"),
                "mlflow_dataset_id": dataset.dataset_id,
                "mlflow_dataset_name": dataset.name,
            }
