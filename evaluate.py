# License Apache 2.0: (c) 2026 Athena-Reply

"""Run the open-arena evaluation sweep.

The program graph lives in `program.py` (`build_program()`) so it can
be edited in place — swap `Generator` for an agent, splice in a
retriever, layer a critique, etc., without touching the rest of the
trial-runner glue. This is the file `arena` (the console script)
executes; `un run evaluate.py` is the equivalent direct invocation.

Run with:

    arena -c config.yaml
    arena -v                       # show synalinks progress bar
    un run evaluate.py -c ...      # equivalent
"""

import asyncio
import json
import sys
import warnings
from pathlib import Path

import src.keras_stub  # noqa: F401  must precede `import keras_tuner`

import click
import keras_tuner as kt
import yaml

from program import build_program
from src.datasets import load_dataset_from_yaml
from src.rewards import get as get_reward


KT_DIR = Path(".kt")
KT_PROJECT = "open_arena"
TSV_PATH = KT_DIR / "last_run.tsv"


class OpenArenaTuner(kt.engine.base_tuner.BaseTuner):
    """Grid sweep over (`language_model`, `dataset`) pairs.

    Bypasses the Keras `model.fit`/`model.evaluate` path: each trial reads
    both choices from `hp`, runs an async synalinks eval against the
    matching dataset (with that dataset's own generator kwargs), and
    reports `reward` (the primary, dataset-defined metric) plus one extra
    metric per `experiment_rewards` entry. Each extra reward triggers a
    separate `program.evaluate()` pass — synalinks `compile(reward=...)`
    only accepts one — so K extras cost K× the model calls of the primary.
    """

    def run_trial(
        self,
        trial,
        datasets,
        generator_kwargs_by_ds,
        rewards_by_ds,
        experiment_rewards,
        verbose=0,
    ):
        # Register experiment-reward metrics explicitly so keras-tuner doesn't
        # fall back to its auto-direction inference, which pokes keras.metrics
        # — not stubbed in our keras-less setup. "reward" is already registered
        # via the oracle's Objective.
        for alias, _, direction in experiment_rewards:
            if not trial.metrics.exists(alias):
                trial.metrics.register(alias, direction=direction)

        model_id = trial.hyperparameters.get("language_model")
        ds_name = trial.hyperparameters.get("dataset")
        gen_kwargs = generator_kwargs_by_ds.get(ds_name, {})
        ds = datasets[ds_name]

        async def _eval(reward):
            program = await build_program(model_id, ds, gen_kwargs, reward)
            return await program.evaluate(x=ds, verbose=verbose)

        # Primary first: any failure here fails the trial outright (no fallback,
        # no silent 0.0 — let synalinks errors propagate).
        primary_out = asyncio.run(_eval(rewards_by_ds[ds_name]))
        result = {"reward": float(primary_out["reward"])}

        # Experiment rewards are best-effort: a flaky LM-judge or buggy reward
        # shouldn't void the primary score. Skip the alias on failure; the
        # missing metric surfaces as a `null` cell in the result matrix.
        for alias, reward, _ in experiment_rewards:
            try:
                out = asyncio.run(_eval(reward))
                result[alias] = float(out["reward"])
            except Exception as e:
                warnings.warn(
                    f"experiment reward {alias!r} failed for "
                    f"({model_id}, {ds_name}): {e}",
                    stacklevel=2,
                )
        return result


def _parse_experiment_rewards(specs):
    """Build a list of `(alias, Reward, direction)` from `experiments.rewards` YAML.

    `direction:` ∈ `{"max", "min"}` (default `"max"`) tells keras-tuner
    which way to rank the metric. Use `"min"` for loss-style rewards
    (cross-entropy, edit distance, ...).
    """
    out = []
    for spec in specs or []:
        if isinstance(spec, str):
            spec = {"name": spec}
        spec = dict(spec)
        alias = spec.pop("alias", None) or spec["name"]
        direction = spec.pop("direction", "max")
        if direction not in ("max", "min"):
            raise ValueError(
                f"Reward {alias!r}: `direction:` must be 'max' or 'min'; "
                f"got {direction!r}."
            )
        if alias == "reward":
            raise ValueError(
                "Experiment reward alias 'reward' collides with the primary "
                "metric. Pick a different `alias:`."
            )
        out.append((alias, get_reward(spec), direction))
    return out


def _collect_cells(oracle, metric_keys, primary_direction="max"):
    """Return `(cells, statuses)` for every (model, dataset) trial.

    Iterates *every* trial on the oracle (not `get_best_trials`, which
    silently drops FAILED/INVALID cells). For each (model, dataset) hp
    combo we keep one canonical trial: COMPLETED beats anything else;
    among COMPLETED, the better `score` (per `primary_direction`) wins;
    otherwise the later-seen wins. `cells` maps `(m, d, k) -> float | None`;
    `statuses` maps `(m, d) -> trial.status`.
    """
    canonical: dict[tuple[str, str], object] = {}
    sentinel = float("-inf") if primary_direction == "max" else float("inf")
    score_better = (lambda a, b: a >= b) if primary_direction == "max" else (lambda a, b: a <= b)

    def _better(a, b):
        a_done = a.status == "COMPLETED"
        b_done = b.status == "COMPLETED"
        if a_done != b_done:
            return a if a_done else b
        if a_done and b_done:
            sa = a.score if a.score is not None else sentinel
            sb = b.score if b.score is not None else sentinel
            return a if score_better(sa, sb) else b
        return b

    for trial in oracle.trials.values():
        m = trial.hyperparameters.get("language_model")
        d = trial.hyperparameters.get("dataset")
        if m is None or d is None:
            # Stale cache from a different HP space; skip.
            continue
        prev = canonical.get((m, d))
        canonical[(m, d)] = trial if prev is None else _better(prev, trial)

    cells: dict[tuple[str, str, str], float | None] = {}
    statuses: dict[tuple[str, str], str] = {}
    for (m, d), trial in canonical.items():
        statuses[(m, d)] = trial.status
        for k in metric_keys:
            try:
                cells[(m, d, k)] = trial.metrics.get_best_value(k)
            except (KeyError, ValueError):
                cells[(m, d, k)] = None
    return cells, statuses


def _render_matrix(matrix: dict) -> None:
    """Print one dataset's matrix as a markdown table to stdout.

    Rows = models, columns = metrics. Numeric cells render as `0.0000`;
    cells with no value render as the trial status (so FAILED / INVALID
    rows are still visible). Output is grep-able: `grep '^| ' run.log`
    lists every row across every dataset.
    """
    metric_keys = matrix["metrics"]
    headers = ["language_model", *metric_keys]
    rows = []
    for r in matrix["rows"]:
        row = [r["model"]]
        for k in metric_keys:
            v = r.get(k)
            row.append(f"{v:.4f}" if v is not None else (r.get("status") or "—"))
        rows.append(row)

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def _fmt(row):
        first = row[0].ljust(widths[0])
        rest = [c.rjust(widths[i + 1]) for i, c in enumerate(row[1:])]
        return "| " + " | ".join([first, *rest]) + " |"

    sep = ["-" * widths[0]] + ["-" * (widths[i] - 1) + ":" for i in range(1, len(headers))]

    print()
    print(f"## {matrix['dataset']}")
    print()
    print(_fmt(headers))
    print("| " + " | ".join(sep) + " |")
    for row in rows:
        print(_fmt(row))
    print()


def _write_tsv(result: dict, path: Path) -> None:
    """Write the sweep in long format: `model\\tdataset\\tmetric\\tvalue`.

    One row per `(model, dataset, metric)` cell with a value; failed
    cells are skipped. Easy to `cut -f` / pivot, no rich markup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("model\tdataset\tmetric\tvalue\n")
        for matrix in result["matrices"]:
            d = matrix["dataset"]
            for r in matrix["rows"]:
                m = r["model"]
                for k in matrix["metrics"]:
                    v = r.get(k)
                    if v is None:
                        continue
                    f.write(f"{m}\t{d}\t{k}\t{v:.6f}\n")


def _build_result(cells, statuses, model_ids, dataset_names, metric_keys, *, config_path=None):
    """Assemble a JSON-serializable view of the sweep result.

    `matrices` mirrors the markdown render one-to-one (one entry per
    dataset, in `experiments.datasets` order); `cells` is the flat
    long-form sidecar for ingestion / dataframes. Failed trials surface
    as `value: null` plus a non-COMPLETED `status`, so an API consumer
    can distinguish "not run" from "ran and got 0.0".
    """
    matrices = []
    for d in dataset_names:
        rows = []
        for m in model_ids:
            row = {"model": m, "status": statuses.get((m, d))}
            for k in metric_keys:
                row[k] = cells.get((m, d, k))
            rows.append(row)
        matrices.append(
            {
                "dataset": d,
                "models": list(model_ids),
                "metrics": list(metric_keys),
                "rows": rows,
            }
        )

    flat = [
        {
            "model": m,
            "dataset": d,
            "metric": k,
            "value": cells.get((m, d, k)),
            "status": statuses.get((m, d)),
        }
        for m in model_ids
        for d in dataset_names
        for k in metric_keys
    ]
    return {
        "config": str(config_path) if config_path is not None else None,
        "models": list(model_ids),
        "datasets": list(dataset_names),
        "metrics": list(metric_keys),
        "matrices": matrices,
        "cells": flat,
    }


def run_sweep(
    config_path: str,
    *,
    no_cache: bool = False,
    verbose: int = 0,
) -> dict:
    """Run the full sweep and return the JSON-serializable result dict.

    Programmatic entry point (e.g. for an API endpoint that needs the
    matrix without going through stdout). Trial state is still persisted
    under `.kt/`, the same cache the CLI reads, so resumed runs work.

    `no_cache=True` drops the `.kt/open_arena/` cache before starting —
    use it when the HP space (model / dataset / experiment-reward set)
    changed since the last run. `verbose` is forwarded to
    `synalinks.Program.evaluate` (0 silent, 1 progress bar, 2 per-batch).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_ids = config["experiments"]["language_models"]
    dataset_names = config["experiments"].get("datasets") or [config["default"]]

    datasets = {n: load_dataset_from_yaml(config_path, name=n) for n in dataset_names}
    generator_kwargs_by_ds = {
        n: config["datasets"][n].get("generator", {}) for n in dataset_names
    }
    rewards_by_ds = {}
    for n in dataset_names:
        spec = config["datasets"][n].get("reward")
        if spec is None:
            raise ValueError(f"{config_path}: dataset {n!r} is missing a `reward:` field.")
        rewards_by_ds[n] = get_reward(spec)

    experiment_rewards = _parse_experiment_rewards(config["experiments"].get("rewards"))

    primary_direction = config["experiments"].get("primary_direction", "max")
    if primary_direction not in ("max", "min"):
        raise ValueError(
            f"{config_path}: `experiments.primary_direction` must be 'max' or "
            f"'min'; got {primary_direction!r}."
        )

    hp = kt.HyperParameters()
    hp.Choice("language_model", values=model_ids)
    hp.Choice("dataset", values=dataset_names)

    max_trials = len(model_ids) * len(dataset_names)
    oracle = kt.oracles.GridSearchOracle(
        objective=kt.Objective("reward", direction=primary_direction),
        max_trials=max_trials,
        hyperparameters=hp,
    )

    tuner = OpenArenaTuner(
        oracle=oracle,
        directory=str(KT_DIR),
        project_name=KT_PROJECT,
        overwrite=no_cache,
    )
    tuner.search(
        datasets,
        generator_kwargs_by_ds,
        rewards_by_ds,
        experiment_rewards,
        verbose=verbose,
    )

    metric_keys = ["reward", *(alias for alias, _, _ in experiment_rewards)]
    cells, statuses = _collect_cells(oracle, metric_keys, primary_direction)

    return _build_result(
        cells, statuses, model_ids, dataset_names, metric_keys, config_path=config_path
    )


def _emit_json(result: dict, dest) -> None:
    """Write `result` as pretty-printed JSON with a trailing newline."""
    json.dump(result, dest, indent=2)
    dest.write("\n")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="config.yaml",
    show_default=True,
    help="Path to the YAML config.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help=(
        "Discard the keras-tuner trial cache before running. Use this when "
        "the HP space (model list, dataset list, or experiment-reward set) "
        "has changed since the last run."
    ),
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Show the per-trial progress bar. Repeat for more detail: `-v` is "
        "synalinks's progress bar (verbose=1), `-vv` is per-batch lines "
        "(verbose=2, recommended when piping to a log file)."
    ),
)
@click.option(
    "--json",
    "json_out",
    type=click.Path(dir_okay=False, writable=True, allow_dash=True),
    default=None,
    help=(
        "Write the result matrix as JSON to PATH (use `-` for stdout). "
        "Suppresses the markdown tables and TSV when set to `-`; otherwise "
        "writes alongside them."
    ),
)
def main(config_path: str, no_cache: bool, verbose: int, json_out: str | None) -> None:
    """Run the open-arena LLM evaluation sweep."""
    result = run_sweep(config_path, no_cache=no_cache, verbose=verbose)

    # `--json -` is the API-style invocation: emit one JSON document on
    # stdout and skip the human-facing table / TSV outputs entirely.
    if json_out == "-":
        _emit_json(result, sys.stdout)
        return

    for matrix in result["matrices"]:
        _render_matrix(matrix)

    _write_tsv(result, TSV_PATH)
    print(f"wrote {TSV_PATH}")

    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            _emit_json(result, f)
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
