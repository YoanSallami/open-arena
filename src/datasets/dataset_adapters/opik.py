from collections.abc import Iterator
from typing import Any

from opik import Opik

from src.datasets.base import Dataset


class OpikDataset(Dataset):
    """
    Fetch rows from an existing Opik (Comet) dataset.

    Opik dataset items are free-form dicts — whatever keys you stored are
    exposed directly as top-level Jinja variables (e.g. `{{ input }}`,
    `{{ expected_output }}`, or any custom field). The item's `id` is
    surfaced as `opik_item_id`, alongside `opik_dataset_id` /
    `opik_dataset_name`, so the execution layer can link back if needed.

    `host` / `api_key` / `workspace` / `project_name` fall back to the
    standard Opik env vars (`OPIK_URL_OVERRIDE`, `OPIK_API_KEY`, ...) when
    omitted.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        dataset_name: str | None = None,
        host: str | None = None,
        api_key: str | None = None,
        workspace: str | None = None,
        project_name: str | None = None,
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.dataset_name = dataset_name or name
        self.host = host
        self.api_key = api_key
        self.workspace = workspace
        self.project_name = project_name

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        client = Opik(
            host=self.host,
            api_key=self.api_key,
            workspace=self.workspace,
            project_name=self.project_name,
        )
        dataset = client.get_dataset(name=self.dataset_name)
        for item in dataset.get_items(nb_samples=self.limit):
            item_id = item.pop("id", None)
            yield {
                **item,
                "opik_item_id": item_id,
                "opik_dataset_id": dataset.id,
                "opik_dataset_name": dataset.name,
            }
