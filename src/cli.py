# License Apache 2.0: (c) 2026 Athena-Reply

"""`arena` CLI entrypoint.

Wraps the keras-tuner sweep with a `click` interface so it can be invoked
as a console script after `uv sync`:

    arena -c config.yaml
    arena --no-cache         # ignore stale `.kt/` trial state
"""

import asyncio
from pathlib import Path

import src.keras_stub  # noqa: F401  must precede `import keras_tuner`

import click
import keras_tuner as kt
import synalinks
import yaml

from src.datasets import load_dataset_from_yaml
from src.rewards import get as get_reward


async def _evaluate(model_id: str, dataset, generator_kwargs: dict, reward) -> dict:
    language_model = synalinks.LanguageModel(model=model_id)

    # Build the program input from the dataset's schema or data_model.
    # `input_schema:` (raw JSON Schema in YAML) takes precedence; otherwise
    # fall back to the dataset's class (defaults to `synalinks.ChatMessages`).
    if dataset.input_schema is not None:
        inputs = synalinks.Input(schema=dataset.input_schema)
    else:
        inputs = synalinks.Input(data_model=dataset.input_data_model)

    # `return_inputs:` and `reasoning_effort:` are per-dataset (lives in the
    # dataset's `generator:` block). return_inputs=True concatenates input
    # fields onto the output so judge-style rewards see the original prompt
    # alongside the prediction; comparison-style primaries usually want it
    # False. reasoning_effort defaults to "low" — small ollama models are
    # slow under "high" especially with structured output constraints.
    cot_kwargs = {"reasoning_effort": "low", **generator_kwargs}
    # Constrain the LM's structured output to the dataset's `output_schema:`
    # when one is set. With no output schema we leave Generator's default
    # path alone (free-form chat-message shape) for backward compat.
    if dataset.output_schema is not None and "schema" not in cot_kwargs:
        cot_kwargs["schema"] = dataset.output_schema

    outputs = await synalinks.ChainOfThought(
        language_model=language_model,
        **cot_kwargs,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"eval_{model_id.replace('/', '_').replace(':', '_')}",
    )
    program.compile(reward=reward)
    return await program.evaluate(x=dataset, verbose=0)


class OpenArenaTuner(kt.engine.base_tuner.BaseTuner):
    """Grid sweep over (`language_model`, `dataset`) pairs.

    Bypasses the Keras `model.fit`/`model.evaluate` path: each trial reads
    both choices from `hp`, runs an async synalinks eval against the
    matching dataset (with that dataset's own generator kwargs), and
    reports `reward` (the primary, dataset-defined metric) to the oracle
    plus one extra metric per `experiment_rewards` entry. Each extra
    reward triggers a separate `program.evaluate()` pass — synalinks
    `compile(reward=...)` only accepts one — so K extras cost K× the
    model calls of the primary run.
    """

    def run_trial(
        self, trial, datasets, generator_kwargs_by_ds, rewards_by_ds, experiment_rewards
    ):
        # Register experiment-reward metrics explicitly so keras-tuner doesn't
        # fall back to its auto-direction inference, which pokes keras.metrics
        # — not stubbed in our keras-less setup. "reward" is already registered
        # via the oracle's Objective.
        for alias, _ in experiment_rewards:
            if not trial.metrics.exists(alias):
                trial.metrics.register(alias, direction="max")

        model_id = trial.hyperparameters.get("language_model")
        ds_name = trial.hyperparameters.get("dataset")
        gen_kwargs = generator_kwargs_by_ds.get(ds_name, {})
        ds = datasets[ds_name]

        primary = asyncio.run(_evaluate(model_id, ds, gen_kwargs, rewards_by_ds[ds_name]))
        result = {"reward": float(primary.get("reward", 0.0))}
        for alias, reward in experiment_rewards:
            extra = asyncio.run(_evaluate(model_id, ds, gen_kwargs, reward))
            result[alias] = float(extra.get("reward", 0.0))
        return result


def _parse_experiment_rewards(specs):
    """Build a list of `(alias, Reward)` from `experiments.rewards` YAML.

    Each spec is the same shape as a dataset `reward:` entry, with an
    optional `alias:` key that names the metric in the result table. The
    alias defaults to the reward's class identifier.
    """
    out = []
    for spec in specs or []:
        if isinstance(spec, str):
            spec = {"name": spec}
        spec = dict(spec)
        alias = spec.pop("alias", None) or spec["name"]
        if alias == "reward":
            raise ValueError(
                "Experiment reward alias 'reward' collides with the primary "
                "metric. Pick a different `alias:`."
            )
        out.append((alias, get_reward(spec)))
    return out


def _collect_cells(oracle, max_trials, metric_keys):
    """Pull `(model, dataset, metric) -> value` cells once for all metrics.

    Returns `(cells, statuses)`. `cells` maps to `float` for successful
    trials and to `None` when the metric is missing (crash, skip, …);
    `statuses` maps `(model, dataset) -> trial.status` so the renderer
    can show what went wrong in those cells.
    """
    cells: dict[tuple[str, str, str], float | None] = {}
    statuses: dict[tuple[str, str], str] = {}
    for trial in oracle.get_best_trials(num_trials=max_trials):
        m = trial.hyperparameters.get("language_model")
        d = trial.hyperparameters.get("dataset")
        statuses[(m, d)] = trial.status
        for k in metric_keys:
            try:
                cells[(m, d, k)] = trial.metrics.get_best_value(k)
            except (KeyError, ValueError):
                cells[(m, d, k)] = None
    return cells, statuses


def _render_matrix(cells, statuses, model_ids, dataset_names, metric_key, title):
    """Print one matrix as a markdown table to stdout.

    Numeric cells render as `0.0000`; failed cells render as the trial
    status. Output is grep-able: `grep '^| ' run.log` lists every row
    across every metric, including the header rows.
    """
    headers = ["language_model", *dataset_names]
    rows = []
    for m in model_ids:
        row = [m]
        for d in dataset_names:
            v = cells.get((m, d, metric_key))
            row.append(f"{v:.4f}" if v is not None else statuses.get((m, d), "—"))
        rows.append(row)

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def _fmt(row):
        first = row[0].ljust(widths[0])
        rest = [c.rjust(widths[i + 1]) for i, c in enumerate(row[1:])]
        return "| " + " | ".join([first, *rest]) + " |"

    sep = ["-" * widths[0]] + ["-" * (widths[i] - 1) + ":" for i in range(1, len(headers))]

    print()
    print(f"## {title}")
    print()
    print(_fmt(headers))
    print("| " + " | ".join(sep) + " |")
    for row in rows:
        print(_fmt(row))
    print()


def _write_tsv(cells, model_ids, dataset_names, metric_keys, path: Path) -> None:
    """Write the full sweep in long format: `model\\tdataset\\tmetric\\tvalue`.

    One row per `(model, dataset, metric)` cell that has a value; failed
    cells are skipped. Easy to `cut -f` / pivot, no rich markup.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("model\tdataset\tmetric\tvalue\n")
        for m in model_ids:
            for d in dataset_names:
                for k in metric_keys:
                    v = cells.get((m, d, k))
                    if v is None:
                        continue
                    f.write(f"{m}\t{d}\t{k}\t{v:.6f}\n")


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
def main(config_path: str, no_cache: bool) -> None:
    """Run the open-arena LLM evaluation sweep."""
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

    hp = kt.HyperParameters()
    hp.Choice("language_model", values=model_ids)
    hp.Choice("dataset", values=dataset_names)

    max_trials = len(model_ids) * len(dataset_names)
    oracle = kt.oracles.GridSearchOracle(
        objective=kt.Objective("reward", direction="max"),
        max_trials=max_trials,
        hyperparameters=hp,
    )

    tuner = OpenArenaTuner(
        oracle=oracle,
        directory=".kt",
        project_name="open_arena",
        overwrite=no_cache,
    )
    tuner.search(datasets, generator_kwargs_by_ds, rewards_by_ds, experiment_rewards)

    metric_keys = ["reward", *(alias for alias, _ in experiment_rewards)]
    cells, statuses = _collect_cells(oracle, max_trials, metric_keys)

    _render_matrix(cells, statuses, model_ids, dataset_names, "reward", "Reward (primary)")
    for alias, _ in experiment_rewards:
        _render_matrix(cells, statuses, model_ids, dataset_names, alias, alias)

    tsv_path = Path(".kt") / "last_run.tsv"
    _write_tsv(cells, model_ids, dataset_names, metric_keys, tsv_path)
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
