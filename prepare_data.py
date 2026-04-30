# License Apache 2.0: (c) 2026 Athena-Reply

"""Prepare datasets for the open-arena evaluation sweep.

Companion script to `evaluate.py`. The platform itself does not ship a
synthetic-data engine — generation, selection, deduplication, and any
other dataset-preparation logic lives here, per project. The expected
flow is:

    external generator (or hand-curation)
        -> rows written under `raw_data/`
        -> read at sweep time by a `local` / `folder` / `huggingface`
           dataset adapter declared in `config.yaml`

Recommended tooling for structured synthetic-data workflows: **NVIDIA
Data Designer** (https://github.com/NVIDIA/data-designer). It provides
declarative building blocks for prompt-driven generation, schema
constraints, multi-stage transforms, and evaluator-in-the-loop
filtering. A typical `prepare_data.py` using it looks like:

    from data_designer import DataDesigner  # or whatever the public API is
    # build the pipeline: source -> generate -> validate -> dedupe -> emit
    pipeline = DataDesigner(...)
    pipeline.run(output_dir="raw_data/<dataset_name>/")

Other tools (Distilabel, Argilla, Synthetic Data Vault, plain LLM
scripts) work the same way — emit rows into `raw_data/`, then point a
dataset adapter at them.

Anything useful to prepare the raw data is fair game here; this script
is intentionally free-form.
"""