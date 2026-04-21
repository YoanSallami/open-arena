from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl

from src.datasets.base import Dataset


class LocalDataset(Dataset):
    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        path: str,
        format: str,
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.path = Path(path)
        self.format = format

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        if self.format == "csv":
            df = pl.read_csv(self.path)
        elif self.format == "excel":
            df = pl.read_excel(self.path)
        else:
            raise ValueError(f"Unsupported local format: {self.format!r}")

        for row in df.iter_rows(named=True):
            yield dict(row)
