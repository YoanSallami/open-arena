from collections.abc import Iterator
from typing import Any

from src.datasets.base import Dataset


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        repo: str,
        config: str | None = None,
        split: str = "test",
        revision: str | None = None,
        streaming: bool = True,
        limit: int | None = None,
    ):
        if streaming and limit is None:
            raise ValueError(
                f"HuggingFace dataset '{repo}' has streaming=True but no limit set. "
                "Streaming datasets can be unbounded — set `limit` in the dataset config "
                "or use streaming: false to load the full split into memory."
            )
        super().__init__(name, input_template, expected_output_template, limit)
        self.repo = repo
        self.hf_config = config
        self.split = split
        self.revision = revision
        self.streaming = streaming

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(
            self.repo,
            self.hf_config,
            split=self.split,
            revision=self.revision,
            streaming=self.streaming,
        )
        for row in ds:
            yield dict(row)
