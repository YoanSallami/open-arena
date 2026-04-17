"""
Offline tests for every dataset adapter.

Each remote adapter is exercised with its provider client mocked out, so
no network or credentials are required. The tests lock in the row shape
that `Dataset.__iter__` produces — specifically:
  - `input` and `expected_output` are rendered correctly from the raw row,
  - all provider id fields (`lf_item_id`, `ls_example_id`, ...) flow into
    metadata so the executor can link traces back to the source item,
  - unreferenced raw columns flow through as metadata,
  - `limit` truncates iteration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.datasets import build_dataset
from src.datasets.dataset_adapters import _ADAPTERS


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------

def test_every_documented_provider_is_registered():
    expected = {
        "local",
        "huggingface",
        "braintrust",
        "langfuse",
        "langsmith",
        "mlflow",
        "opik",
        "phoenix",
        "weave",
    }
    assert expected == set(_ADAPTERS)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown dataset provider"):
        build_dataset(
            name="x",
            source={"provider": "does-not-exist"},
            input_template="{{ a }}",
            expected_output_template="{{ b }}",
        )


# ---------------------------------------------------------------------------
# Local: CSV round-trip (no mocking — write a real tiny file)
# ---------------------------------------------------------------------------

def test_local_csv_adapter(tmp_path):
    csv = tmp_path / "rows.csv"
    csv.write_text("question,answer,category\nWhat?,42,trivia\nWhy?,because,philosophy\n")

    dataset = build_dataset(
        name="local-test",
        source={"provider": "local", "path": str(csv), "format": "csv"},
        input_template="Q: {{ question }}",
        expected_output_template="{{ answer }}",
    )
    rows = list(dataset)

    assert rows == [
        ("Q: What?", "42", {"category": "trivia"}),
        ("Q: Why?", "because", {"category": "philosophy"}),
    ]


def test_local_limit_truncates(tmp_path):
    csv = tmp_path / "rows.csv"
    csv.write_text("q,a\nq1,a1\nq2,a2\nq3,a3\n")

    dataset = build_dataset(
        name="local-test",
        source={"provider": "local", "path": str(csv), "format": "csv"},
        input_template="{{ q }}",
        expected_output_template="{{ a }}",
        limit=2,
    )
    assert len(list(dataset)) == 2


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

def test_huggingface_adapter():
    fake_rows = [
        {"question": "q1", "answer": "a1", "split": "test"},
        {"question": "q2", "answer": "a2", "split": "test"},
    ]

    with patch("datasets.load_dataset", return_value=fake_rows) as mocked:
        dataset = build_dataset(
            name="hf-test",
            source={
                "provider": "huggingface",
                "repo": "some/repo",
                "config": "subset",
                "split": "test",
            },
            input_template="{{ question }}",
            expected_output_template="{{ answer }}",
        )
        rows = list(dataset)

    mocked.assert_called_once_with(
        "some/repo", "subset", split="test", revision=None, streaming=True
    )
    assert rows == [
        ("q1", "a1", {"split": "test"}),
        ("q2", "a2", {"split": "test"}),
    ]


# ---------------------------------------------------------------------------
# Langfuse (source-is-langfuse: spread metadata at top level, inject lf_* ids)
# ---------------------------------------------------------------------------

def test_langfuse_adapter():
    fake_items = [
        MagicMock(
            input={"question": "q1"},
            expected_output="a1",
            metadata={"category": "cat1"},
            id="item-1",
            dataset_id="ds-1",
            dataset_name="mmlu",
        ),
        MagicMock(
            input={"question": "q2"},
            expected_output="a2",
            metadata={"category": "cat2"},
            id="item-2",
            dataset_id="ds-1",
            dataset_name="mmlu",
        ),
    ]
    fake_client = MagicMock()
    fake_client.get_dataset.return_value = MagicMock(items=fake_items)

    with patch("src.datasets.dataset_adapters.langfuse.get_client", return_value=fake_client):
        dataset = build_dataset(
            name="mmlu",
            source={"provider": "langfuse", "dataset_name": "mmlu"},
            input_template="{{ input.question }}",
            expected_output_template="{{ expected_output }}",
        )
        rows = list(dataset)

    fake_client.get_dataset.assert_called_once_with("mmlu")
    assert rows[0][0] == "q1"
    assert rows[0][1] == "a1"
    assert rows[0][2] == {
        "category": "cat1",
        "lf_item_id": "item-1",
        "lf_dataset_id": "ds-1",
        "lf_dataset_name": "mmlu",
    }


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------

def test_langsmith_adapter():
    fake_examples = [
        MagicMock(
            inputs={"question": "q1"},
            outputs={"answer": "a1"},
            metadata={"source": "manual"},
            id="ex-1",
            dataset_id="ds-1",
        ),
    ]
    fake_client = MagicMock()
    fake_client.list_examples.return_value = iter(fake_examples)

    with patch("src.datasets.dataset_adapters.langsmith.Client", return_value=fake_client):
        dataset = build_dataset(
            name="mmlu",
            source={"provider": "langsmith", "dataset_name": "mmlu"},
            input_template="{{ inputs.question }}",
            expected_output_template="{{ outputs.answer }}",
        )
        rows = list(dataset)

    fake_client.list_examples.assert_called_once_with(dataset_name="mmlu", limit=None)
    assert rows == [
        (
            "q1",
            "a1",
            {
                "source": "manual",
                "ls_example_id": "ex-1",
                "ls_dataset_id": "ds-1",
                "ls_dataset_name": "mmlu",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Phoenix
# ---------------------------------------------------------------------------

def test_phoenix_adapter():
    fake_dataset = MagicMock()
    fake_dataset.id = "phoenix-ds-id"
    fake_dataset.name = "mmlu-anatomy"
    fake_dataset.examples = [
        {
            "id": "ex-1",
            "input": {"question": "q1"},
            "output": {"answer": "a1"},
            "metadata": {"topic": "cells"},
        },
    ]
    fake_client = MagicMock()
    fake_client.datasets.get_dataset.return_value = fake_dataset

    with patch("src.datasets.dataset_adapters.phoenix.Client", return_value=fake_client):
        dataset = build_dataset(
            name="mmlu-anatomy",
            source={"provider": "phoenix", "base_url": "http://localhost:6006"},
            input_template="{{ input.question }}",
            expected_output_template="{{ output.answer }}",
        )
        rows = list(dataset)

    fake_client.datasets.get_dataset.assert_called_once_with(
        dataset="mmlu-anatomy", version_id=None
    )
    assert rows[0][0] == "q1"
    assert rows[0][1] == "a1"
    assert rows[0][2] == {
        "topic": "cells",
        "phoenix_example_id": "ex-1",
        "phoenix_dataset_id": "phoenix-ds-id",
        "phoenix_dataset_name": "mmlu-anatomy",
    }


# ---------------------------------------------------------------------------
# Opik (free-form records: spread raw, rename `id` -> opik_item_id)
# ---------------------------------------------------------------------------

def test_opik_adapter():
    fake_dataset = MagicMock()
    fake_dataset.id = "opik-ds-id"
    fake_dataset.name = "mmlu"
    fake_dataset.get_items.return_value = [
        {"id": "item-1", "question": "q1", "expected_output": "a1", "difficulty": "easy"},
    ]
    fake_client = MagicMock()
    fake_client.get_dataset.return_value = fake_dataset

    with patch("src.datasets.dataset_adapters.opik.Opik", return_value=fake_client):
        dataset = build_dataset(
            name="mmlu",
            source={"provider": "opik", "dataset_name": "mmlu"},
            input_template="{{ question }}",
            expected_output_template="{{ expected_output }}",
        )
        rows = list(dataset)

    fake_client.get_dataset.assert_called_once_with(name="mmlu")
    fake_dataset.get_items.assert_called_once_with(nb_samples=None)
    assert rows[0][0] == "q1"
    assert rows[0][1] == "a1"
    assert rows[0][2] == {
        "difficulty": "easy",
        "opik_item_id": "item-1",
        "opik_dataset_id": "opik-ds-id",
        "opik_dataset_name": "mmlu",
    }


# ---------------------------------------------------------------------------
# Braintrust
# ---------------------------------------------------------------------------

def test_braintrust_adapter():
    fake_dataset = MagicMock()
    fake_dataset.id = "bt-ds-id"
    fake_dataset.name = "mmlu"
    fake_dataset.fetch.return_value = [
        {
            "id": "rec-1",
            "input": {"question": "q1"},
            "expected": "a1",
            "metadata": {"topic": "cells"},
            "tags": ["v1"],
            "dataset_id": "bt-ds-id",
        },
    ]

    with patch(
        "src.datasets.dataset_adapters.braintrust.init_dataset",
        return_value=fake_dataset,
    ) as init:
        dataset = build_dataset(
            name="mmlu",
            source={"provider": "braintrust", "project": "proj-1", "dataset_name": "mmlu"},
            input_template="{{ input.question }}",
            expected_output_template="{{ expected }}",
        )
        rows = list(dataset)

    init.assert_called_once()
    fake_dataset.close.assert_called_once()
    assert rows[0][0] == "q1"
    assert rows[0][1] == "a1"
    assert rows[0][2] == {
        "topic": "cells",
        "tags": ["v1"],
        "braintrust_record_id": "rec-1",
        "braintrust_dataset_id": "bt-ds-id",
        "braintrust_dataset_name": "mmlu",
        "braintrust_project": "proj-1",
    }


# ---------------------------------------------------------------------------
# Weave
# ---------------------------------------------------------------------------

def test_weave_adapter():
    fake_dataset = MagicMock()
    fake_dataset.rows = [
        {"question": "q1", "answer": "a1", "topic": "cells"},
    ]
    fake_ref = MagicMock()
    fake_ref.get.return_value = fake_dataset

    with patch("src.datasets.dataset_adapters.weave.weave") as weave_mod:
        weave_mod.ref.return_value = fake_ref

        dataset = build_dataset(
            name="mmlu",
            source={
                "provider": "weave",
                "project": "ent/proj",
                "dataset_name": "mmlu",
                "version": "v2",
            },
            input_template="{{ question }}",
            expected_output_template="{{ answer }}",
        )
        rows = list(dataset)

    weave_mod.init.assert_called_once_with("ent/proj")
    weave_mod.ref.assert_called_once_with("mmlu:v2")
    assert rows[0][0] == "q1"
    assert rows[0][1] == "a1"
    assert rows[0][2] == {
        "topic": "cells",
        "weave_dataset_name": "mmlu",
        "weave_dataset_version": "v2",
        "weave_project": "ent/proj",
    }


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def test_mlflow_adapter():
    import polars as pl

    records = pl.DataFrame([
        {
            "dataset_record_id": "rec-1",
            "inputs": {"question": "q1"},
            "expectations": {"answer": "a1"},
            "tags": {"topic": "cells"},
        },
    ]).to_dicts()
    fake_df = MagicMock()
    fake_df.to_dict.return_value = records
    fake_dataset = MagicMock()
    fake_dataset.dataset_id = "mlflow-ds-id"
    fake_dataset.name = "mmlu"
    fake_dataset.to_df.return_value = fake_df

    with patch(
        "src.datasets.dataset_adapters.mlflow.get_dataset",
        return_value=fake_dataset,
    ) as mocked:
        dataset = build_dataset(
            name="mmlu",
            source={"provider": "mlflow", "dataset_name": "mmlu"},
            input_template="{{ inputs.question }}",
            expected_output_template="{{ expectations.answer }}",
        )
        rows = list(dataset)

    mocked.assert_called_once_with(name="mmlu")
    assert rows[0][0] == "q1"
    assert rows[0][1] == "a1"
    assert rows[0][2] == {
        "tags": {"topic": "cells"},
        "mlflow_record_id": "rec-1",
        "mlflow_dataset_id": "mlflow-ds-id",
        "mlflow_dataset_name": "mmlu",
    }


# ---------------------------------------------------------------------------
# `limit` works on every adapter (via the base class)
# ---------------------------------------------------------------------------

def _five(shape):
    return [shape(i) for i in range(5)]


def _setup_local(tmp_path):
    csv = tmp_path / "rows.csv"
    csv.write_text("q,a\n" + "\n".join(f"q{i},a{i}" for i in range(5)) + "\n")
    return {
        "source": {"provider": "local", "path": str(csv), "format": "csv"},
        "input_template": "{{ q }}",
        "expected_output_template": "{{ a }}",
        "patches": [],
    }


def _setup_huggingface(tmp_path):
    rows = _five(lambda i: {"q": f"q{i}", "a": f"a{i}"})
    return {
        "source": {"provider": "huggingface", "repo": "fake/repo"},
        "input_template": "{{ q }}",
        "expected_output_template": "{{ a }}",
        "patches": [patch("datasets.load_dataset", return_value=rows)],
    }


def _setup_langfuse(tmp_path):
    items = _five(lambda i: MagicMock(
        input=f"q{i}", expected_output=f"a{i}",
        metadata={}, id=f"i{i}", dataset_id="d", dataset_name="x",
    ))
    client = MagicMock()
    client.get_dataset.return_value = MagicMock(items=items)
    return {
        "source": {"provider": "langfuse"},
        "input_template": "{{ input }}",
        "expected_output_template": "{{ expected_output }}",
        "patches": [patch("src.datasets.dataset_adapters.langfuse.get_client", return_value=client)],
    }


def _setup_langsmith(tmp_path):
    examples = _five(lambda i: MagicMock(
        inputs={"q": f"q{i}"}, outputs={"a": f"a{i}"},
        metadata={}, id=f"e{i}", dataset_id="d",
    ))
    client = MagicMock()
    # Respect the limit kwarg the adapter now passes through to the SDK.
    client.list_examples.side_effect = lambda dataset_name, limit: iter(
        examples[:limit] if limit else examples
    )
    return {
        "source": {"provider": "langsmith"},
        "input_template": "{{ inputs.q }}",
        "expected_output_template": "{{ outputs.a }}",
        "patches": [patch("src.datasets.dataset_adapters.langsmith.Client", return_value=client)],
    }


def _setup_phoenix(tmp_path):
    dataset = MagicMock()
    dataset.id = "d"
    dataset.name = "x"
    dataset.examples = _five(lambda i: {
        "id": f"e{i}", "input": {"q": f"q{i}"}, "output": {"a": f"a{i}"}, "metadata": {},
    })
    client = MagicMock()
    client.datasets.get_dataset.return_value = dataset
    return {
        "source": {"provider": "phoenix"},
        "input_template": "{{ input.q }}",
        "expected_output_template": "{{ output.a }}",
        "patches": [patch("src.datasets.dataset_adapters.phoenix.Client", return_value=client)],
    }


def _setup_opik(tmp_path):
    items = _five(lambda i: {"id": f"i{i}", "q": f"q{i}", "a": f"a{i}"})
    dataset = MagicMock()
    dataset.id = "d"
    dataset.name = "x"
    dataset.get_items.side_effect = lambda nb_samples: items[:nb_samples] if nb_samples else items
    client = MagicMock()
    client.get_dataset.return_value = dataset
    return {
        "source": {"provider": "opik"},
        "input_template": "{{ q }}",
        "expected_output_template": "{{ a }}",
        "patches": [patch("src.datasets.dataset_adapters.opik.Opik", return_value=client)],
    }


def _setup_braintrust(tmp_path):
    records = _five(lambda i: {
        "id": f"r{i}", "input": f"q{i}", "expected": f"a{i}", "metadata": {}, "tags": [], "dataset_id": "d",
    })
    dataset = MagicMock()
    dataset.id = "d"
    dataset.name = "x"
    dataset.fetch.return_value = iter(records)
    return {
        "source": {"provider": "braintrust", "project": "p"},
        "input_template": "{{ input }}",
        "expected_output_template": "{{ expected }}",
        "patches": [patch("src.datasets.dataset_adapters.braintrust.init_dataset", return_value=dataset)],
    }


def _setup_weave(tmp_path):
    rows = _five(lambda i: {"q": f"q{i}", "a": f"a{i}"})
    dataset = MagicMock()
    dataset.rows = rows
    ref = MagicMock()
    ref.get.return_value = dataset
    return {
        "source": {"provider": "weave", "project": "ent/proj"},
        "input_template": "{{ q }}",
        "expected_output_template": "{{ a }}",
        "patches": [patch("src.datasets.dataset_adapters.weave.weave", MagicMock(ref=MagicMock(return_value=ref)))],
    }


def _setup_mlflow(tmp_path):
    import polars as pl

    records = pl.DataFrame(_five(lambda i: {
        "dataset_record_id": f"r{i}",
        "inputs": {"q": f"q{i}"},
        "expectations": {"a": f"a{i}"},
        "tags": {},
    })).to_dicts()
    fake_df = MagicMock()
    fake_df.to_dict.return_value = records
    dataset = MagicMock()
    dataset.dataset_id = "d"
    dataset.name = "x"
    dataset.to_df.return_value = fake_df
    return {
        "source": {"provider": "mlflow"},
        "input_template": "{{ inputs.q }}",
        "expected_output_template": "{{ expectations.a }}",
        "patches": [patch("src.datasets.dataset_adapters.mlflow.get_dataset", return_value=dataset)],
    }


@pytest.mark.parametrize(
    "setup",
    [
        _setup_local,
        _setup_huggingface,
        _setup_langfuse,
        _setup_langsmith,
        _setup_phoenix,
        _setup_opik,
        _setup_braintrust,
        _setup_weave,
        _setup_mlflow,
    ],
    ids=lambda f: f.__name__.removeprefix("_setup_"),
)
def test_limit_truncates_for_every_adapter(setup, tmp_path):
    cfg = setup(tmp_path)
    for p in cfg["patches"]:
        p.start()
    try:
        dataset = build_dataset(
            name="x",
            source=cfg["source"],
            input_template=cfg["input_template"],
            expected_output_template=cfg["expected_output_template"],
            limit=2,
        )
        rows = list(dataset)
    finally:
        for p in cfg["patches"]:
            p.stop()

    assert len(rows) == 2
