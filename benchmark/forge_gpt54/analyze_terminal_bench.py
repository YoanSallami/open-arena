#!/usr/bin/env python3
"""Run the forge_gpt54 benchmark and report pairwise accuracy."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import get_client

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.types import ExperimentsFile  # noqa: E402
from src.datasets.langfuse_upload import attach_existing_dataset, upload_rows  # noqa: E402
from src.evaluation import EvaluationResult  # noqa: E402
from src.main_cli import _load_rows, _run_evaluations, _run_experiments  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the replay-backed forge_gpt54 benchmark and report "
            "pairwise accuracy over reward-discriminating trial pairs."
        )
    )
    parser.add_argument(
        "--config",
        default="benchmark/forge_gpt54/verifier.yaml",
        help="Path to the benchmark YAML config (default: benchmark/forge_gpt54/verifier.yaml)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Reuse an existing Langfuse dataset instead of uploading rows again.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any replay row or verifier call fails.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full benchmark summary as JSON.",
    )
    return parser.parse_args()


def _classify_task(rewards: list[int]) -> str:
    reward_set = set(rewards)
    if reward_set == {1}:
        return "all_pass"
    if reward_set == {0}:
        return "all_fail"
    return "mixed"


def _group_results_by_item(all_evals: list[list[EvaluationResult]]) -> dict[str, dict[str, EvaluationResult]]:
    grouped: dict[str, dict[str, EvaluationResult]] = {}
    for per_experiment in all_evals:
        for result in per_experiment:
            key = str(result.metadata.get("lf_item_id") or result.input)
            grouped.setdefault(key, {})[result.experiment_name or result.model_name] = result
    return grouped


def _task_summary(
    config: ExperimentsFile,
    item_id: str,
    results_by_experiment: dict[str, EvaluationResult],
) -> dict[str, Any]:
    first = next(iter(results_by_experiment.values()))
    rewards = [int(value) for value in first.metadata.get("rewards") or []]
    trial_ids = [str(value) for value in first.metadata.get("trial_ids") or []]
    if len(rewards) != len(config.experiments):
        raise ValueError(
            f"Task {first.metadata.get('task_name', item_id)!r} has {len(rewards)} rewards, "
            f"but config declares {len(config.experiments)} replay experiments"
        )

    trials: list[dict[str, Any]] = []
    for index, exp in enumerate(config.experiments):
        result = results_by_experiment.get(exp.name)
        if result is None:
            raise ValueError(
                f"Missing evaluation result for experiment {exp.name!r} on "
                f"task {first.metadata.get('task_name', item_id)!r}"
            )
        trials.append({
            "experiment": exp.name,
            "trial_id": trial_ids[index] if index < len(trial_ids) else exp.name,
            "reward": rewards[index],
            "score": result.score,
            "error": result.error,
        })

    correct = 0
    incorrect = 0
    ties = 0
    skipped = 0
    pairs: list[dict[str, Any]] = []

    for left, right in combinations(range(len(trials)), 2):
        left_trial = trials[left]
        right_trial = trials[right]
        if left_trial["reward"] == right_trial["reward"]:
            continue

        left_score = left_trial["score"]
        right_score = right_trial["score"]
        pair_summary = {
            "left": left_trial["experiment"],
            "right": right_trial["experiment"],
            "left_reward": left_trial["reward"],
            "right_reward": right_trial["reward"],
            "left_score": left_score,
            "right_score": right_score,
            "outcome": "",
        }

        if left_score is None or right_score is None:
            skipped += 1
            pair_summary["outcome"] = "skipped"
            pairs.append(pair_summary)
            continue

        if left_score == right_score:
            ties += 1
            pair_summary["outcome"] = "tie"
            pairs.append(pair_summary)
            continue

        picked_left = left_score > right_score
        picked_reward = left_trial["reward"] if picked_left else right_trial["reward"]
        if picked_reward == 1:
            correct += 1
            pair_summary["outcome"] = "correct"
        else:
            incorrect += 1
            pair_summary["outcome"] = "incorrect"
        pairs.append(pair_summary)

    evaluated = correct + incorrect + ties
    accuracy = ((correct + 0.5 * ties) / evaluated) if evaluated else None

    return {
        "item_id": item_id,
        "task_name": str(first.metadata.get("task_name") or item_id),
        "classification": _classify_task(rewards),
        "trials": trials,
        "pairwise": {
            "accuracy": accuracy,
            "correct": correct,
            "incorrect": incorrect,
            "ties": ties,
            "skipped": skipped,
            "evaluated_pairs": evaluated,
            "discriminating_pairs": evaluated + skipped,
            "pairs": pairs,
        },
    }


def _benchmark_summary(
    config: ExperimentsFile,
    all_evals: list[list[EvaluationResult]],
    config_path: str,
) -> dict[str, Any]:
    tasks = [
        _task_summary(config, item_id, results)
        for item_id, results in sorted(_group_results_by_item(all_evals).items())
    ]

    counts = {"all_pass": 0, "all_fail": 0, "mixed": 0}
    correct = 0
    incorrect = 0
    ties = 0
    skipped = 0
    for task in tasks:
        counts[task["classification"]] += 1
        pairwise = task["pairwise"]
        correct += int(pairwise["correct"])
        incorrect += int(pairwise["incorrect"])
        ties += int(pairwise["ties"])
        skipped += int(pairwise["skipped"])

    evaluated = correct + incorrect + ties
    discriminating_pairs = evaluated + skipped
    accuracy = ((correct + 0.5 * ties) / evaluated) if evaluated else None

    return {
        "config": {
            "path": config_path,
            "dataset_name": config.dataset.name,
            "experiments": [exp.name for exp in config.experiments],
            "score_name": config.evaluation.score_name or "evaluation_score",
            "granularity": config.evaluation.granularity,
            "repeats": config.evaluation.repeats,
        },
        "tasks": tasks,
        "summary": {
            "task_counts": counts,
            "pairwise_accuracy": accuracy,
            "correct_pairs": correct,
            "incorrect_pairs": incorrect,
            "tied_pairs": ties,
            "skipped_pairs": skipped,
            "evaluated_pairs": evaluated,
            "discriminating_pairs": discriminating_pairs,
        },
    }


def _print_summary(summary: dict[str, Any]) -> None:
    meta = summary["summary"]
    counts = meta["task_counts"]
    accuracy = meta["pairwise_accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.3f}"
    print("TERMINAL-BENCH PAIRWISE ACCURACY")
    print(
        "  Tasks:"
        f" {sum(counts.values())}"
        f"  All-pass: {counts['all_pass']}"
        f"  All-fail: {counts['all_fail']}"
        f"  Mixed: {counts['mixed']}"
    )
    print(
        "  Pairwise accuracy:"
        f" {accuracy_text}"
        f"  ({meta['correct_pairs']} correct,"
        f" {meta['incorrect_pairs']} incorrect,"
        f" {meta['tied_pairs']} ties,"
        f" {meta['skipped_pairs']} skipped)"
    )

    mixed_tasks = [task for task in summary["tasks"] if task["classification"] == "mixed"]
    if not mixed_tasks:
        return

    print("\nMixed-task breakdown:")
    for task in mixed_tasks:
        pairwise = task["pairwise"]
        task_accuracy = pairwise["accuracy"]
        accuracy_text = "n/a" if task_accuracy is None else f"{task_accuracy:.3f}"
        print(
            f"  - {task['task_name']}: {accuracy_text}"
            f" over {pairwise['evaluated_pairs']}/{pairwise['discriminating_pairs']} pairs"
        )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = ExperimentsFile.from_yaml(args.config)
    rows = _load_rows(config)
    if config.dataset.source.get("provider") == "langfuse":
        _logger.info("Source is Langfuse; skipping upload because rows already exist remotely")
    elif args.skip_upload:
        rows = attach_existing_dataset(rows, config.dataset.name)
    else:
        rows = await upload_rows(rows, dataset_name=config.dataset.name, description=config.dataset.description or "")

    results = await _run_experiments(config, rows, fail_fast=args.fail_fast)
    all_evals = await _run_evaluations(config, results)
    get_client().flush()
    return _benchmark_summary(config, all_evals, args.config)


def main() -> None:
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        summary = asyncio.run(_run(args))
    except Exception as exc:
        _logger.error("Benchmark run failed: %s", exc, exc_info=args.debug)
        raise SystemExit(1) from exc

    _print_summary(summary)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
