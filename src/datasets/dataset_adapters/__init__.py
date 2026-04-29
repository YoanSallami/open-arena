from typing import Any

from src.datasets.base import Dataset
from src.datasets.dataset_adapters.braintrust import BraintrustDataset
from src.datasets.dataset_adapters.huggingface import HuggingFaceDataset
from src.datasets.dataset_adapters.langfuse import LangfuseDataset
from src.datasets.dataset_adapters.langsmith import LangSmithDataset
from src.datasets.dataset_adapters.local import LocalDataset
from src.datasets.dataset_adapters.local_json_folder import LocalJsonFolderDataset
from src.datasets.dataset_adapters.mlflow import MLflowDataset
from src.datasets.dataset_adapters.opik import OpikDataset
from src.datasets.dataset_adapters.phoenix import PhoenixDataset
from src.datasets.dataset_adapters.weave import WeaveDataset

_ADAPTERS: dict[str, type[Dataset]] = {
    "local": LocalDataset,
    "local_json_folder": LocalJsonFolderDataset,
    "huggingface": HuggingFaceDataset,
    "braintrust": BraintrustDataset,
    "langfuse": LangfuseDataset,
    "langsmith": LangSmithDataset,
    "mlflow": MLflowDataset,
    "opik": OpikDataset,
    "phoenix": PhoenixDataset,
    "weave": WeaveDataset,
}


def build_dataset(
    name: str,
    source: dict[str, Any],
    input_template: str,
    expected_output_template: str,
    limit: int | None = None,
) -> Dataset:
    provider = source.get("provider")
    if provider not in _ADAPTERS:
        raise ValueError(
            f"Unknown dataset provider: {provider!r}. "
            f"Available: {sorted(_ADAPTERS)}"
        )
    adapter_cls = _ADAPTERS[provider]
    source_args = {k: v for k, v in source.items() if k != "provider"}
    return adapter_cls(
        name=name,
        input_template=input_template,
        expected_output_template=expected_output_template,
        limit=limit,
        **source_args,
    )


__all__ = [
    "BraintrustDataset",
    "Dataset",
    "HuggingFaceDataset",
    "LangSmithDataset",
    "LangfuseDataset",
    "LocalDataset",
    "LocalJsonFolderDataset",
    "MLflowDataset",
    "OpikDataset",
    "PhoenixDataset",
    "WeaveDataset",
    "build_dataset",
]
