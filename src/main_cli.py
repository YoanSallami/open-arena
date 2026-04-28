import asyncio
import logging
import sys
import warnings
from typing import Any

import click
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import ValidationError

from src.config.types import ExperimentsFile
from src.datasets import Row, build_dataset
from src.datasets.langfuse_upload import attach_existing_dataset, upload_rows
from src.evaluation import EvaluationResult, build_evaluator
from src.evaluation.evaluators import evaluator_init_params, evaluator_mode
from src.execution import ExecutionResult, Executor
from src.llms import AgentCaller, ReplayCaller, SimpleCaller
from src.llms.types import MCPServerConfig

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_logger = logging.getLogger(__name__)

load_dotenv()

def _load_rows(config: ExperimentsFile) -> list[Row]:
    _logger.info(f"Loading dataset: {config.dataset.name} (provider={config.dataset.source.get('provider')})")
    dataset = build_dataset(
        name=config.dataset.name,
        source=config.dataset.source,
        input_template=config.dataset.input,
        expected_output_template=config.dataset.expected_output,
        limit=config.dataset.limit,
    )
    rows = list(dataset)
    _logger.info(f"Fetched {len(rows)} rows")
    return rows


def _build_replay_lookup(rows: list[Row], trial_index: int) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    trial_number = trial_index + 1
    output_key = f"trial_{trial_number}_output"
    trajectory_key = f"trial_{trial_number}_trajectory"
    lookup: dict[str, tuple[str, list[dict[str, Any]]]] = {}

    for input_text, _expected, metadata in rows:
        if input_text in lookup:
            raise ValueError(
                "Replay lookup requires unique rendered inputs, but found a duplicate: "
                f"{input_text[:120]!r}"
            )
        if output_key not in metadata or trajectory_key not in metadata:
            raise ValueError(
                f"Replay trial index {trial_index} missing expected metadata keys "
                f"{output_key!r} / {trajectory_key!r}"
            )

        trajectory = metadata[trajectory_key]
        if not isinstance(trajectory, list):
            raise ValueError(f"Expected {trajectory_key!r} to be a list, got {type(trajectory).__name__}")

        lookup[input_text] = (str(metadata[output_key] or ""), trajectory)

    return lookup




async def _run_experiments(
    config: ExperimentsFile, rows: list[Row], fail_fast: bool = False
) -> list[list[ExecutionResult]]:
    _logger.info(f"Running {len(config.experiments)} experiments")
    all_results: list[list[ExecutionResult]] = []

    for exp_config in config.experiments:
        _logger.info(f"Experiment '{exp_config.name}' with model {exp_config.litellm.model}")
        callbacks = [CallbackHandler()]
        if exp_config.replay_trial_index is not None:
            if exp_config.mcp:
                raise ValueError("Replay mode does not support MCP servers")
            caller_cls = ReplayCaller
            caller_kwargs: dict[str, Any] = {
                "llm_config": exp_config.litellm.model_dump(),
                "lookup": _build_replay_lookup(rows, exp_config.replay_trial_index),
                "callbacks": callbacks,
            }
        else:
            mcp_servers: list[MCPServerConfig] = (
                [{"server_name": m.name, "url": str(m.url)} for m in exp_config.mcp]
                if exp_config.mcp
                else []
            )
            caller_cls = AgentCaller if mcp_servers else SimpleCaller
            caller_kwargs = {
                "llm_config": exp_config.litellm.model_dump(),
                "callbacks": callbacks,
            }
            if mcp_servers:
                caller_kwargs["mcp_servers"] = mcp_servers

        async with caller_cls(**caller_kwargs) as client:
            executor = Executor(
                dataset=rows,
                llm_client=client,
                system_prompt=config.system_prompt,
                experiment_name=exp_config.name,
                experiment_description=f"Experiment: {exp_config.name} with model {exp_config.litellm.model}",
                fail_fast=fail_fast,
                timeout_s=exp_config.timeout_s,
            )
            results = await executor.execute()

        errors = sum(1 for r in results if r.error)
        if errors:
            _logger.warning(f"Experiment '{exp_config.name}' completed: {len(results)} items, {errors} errors")
        else:
            _logger.info(f"Experiment '{exp_config.name}' completed: {len(results)} items")
        all_results.append(results)

    return all_results


async def _run_evaluations(
    config: ExperimentsFile, all_results: list[list[ExecutionResult]]
) -> list[list[EvaluationResult]]:
    _logger.info(f"Evaluating {len(all_results)} experiments with {config.evaluation.method}")
    mode = evaluator_mode(config.evaluation.method)

    # Runner-managed kwargs (not sourced from EvaluationConfig fields).
    common_kwargs: dict[str, Any] = {
        "method": config.evaluation.method,
        "llm_config": config.evaluation.litellm.model_dump(),
        "callbacks": [CallbackHandler()],
    }

    # Thread every declared EvaluationConfig field whose name matches a
    # parameter on the evaluator's __init__. This keeps the dispatcher
    # evaluator-agnostic: registering a new evaluator (e.g. a judge panel) and
    # declaring any extra fields it needs on EvaluationConfig is enough to
    # wire it in — no edits to this function required.
    accepted_params = evaluator_init_params(config.evaluation.method)
    for field_name in config.evaluation.model_fields:
        if field_name in ("method", "litellm"):
            continue
        if field_name not in accepted_params:
            continue
        value = getattr(config.evaluation, field_name)
        if value is None:
            continue
        common_kwargs[field_name] = value

    all_evals: list[list[EvaluationResult]] = []

    if mode == "pointwise":
        for exp_config, results in zip(config.experiments, all_results):
            _logger.info(f"Evaluating experiment: {exp_config.name}")
            evaluator = build_evaluator(results=results, **common_kwargs)
            eval_results = await evaluator.evaluate()
            _log_summary(exp_config.name, eval_results)
            all_evals.append(eval_results)
        return all_evals

    # group mode: bundle results across experiments by lf_item_id
    _logger.info(f"Group evaluation across {len(config.experiments)} experiments")
    groups = _group_by_item(config.experiments, all_results)
    evaluator = build_evaluator(groups=groups, **common_kwargs)
    flat_results = await evaluator.evaluate()

    by_exp: dict[str, list[EvaluationResult]] = {exp.name: [] for exp in config.experiments}
    for r in flat_results:
        by_exp.setdefault(r.experiment_name or r.model_name, []).append(r)
    for exp_config in config.experiments:
        _log_summary(exp_config.name, by_exp.get(exp_config.name, []))
        all_evals.append(by_exp.get(exp_config.name, []))
    return all_evals


def _group_by_item(
    experiments, all_results: list[list[ExecutionResult]]
) -> list[dict[str, ExecutionResult]]:
    groups: dict[str, dict[str, ExecutionResult]] = {}
    for exp_config, results in zip(experiments, all_results):
        for r in results:
            groups.setdefault(r.metadata["lf_item_id"], {})[exp_config.name] = r
    return list(groups.values())


def _log_summary(name: str, eval_results: list[EvaluationResult]) -> None:
    scored = sum(1 for r in eval_results if r.score is not None)
    errors = sum(1 for r in eval_results if r.error is not None)
    if scored:
        avg = sum(r.score for r in eval_results if r.score is not None) / scored
        msg = f"'{name}': {scored} scored (avg: {avg:.2f})"
    else:
        msg = f"'{name}': 0 scored"
    if errors:
        _logger.warning(f"{msg}, {errors} errors")
    else:
        _logger.info(msg)


@click.command()
@click.option("--config", "-c", "config_path", required=True, type=click.Path(exists=True), help="Path to YAML configuration file")
@click.option("--skip-upload", is_flag=True, default=False, help="Skip dataset upload (reuse existing Langfuse dataset)")
@click.option("--fail-fast", is_flag=True, default=False, help="Re-raise on the first row failure instead of recording the error and continuing")
@click.option("--debug", "-v", is_flag=True, default=False, help="Enable DEBUG-level logging")
def main(config_path: str, skip_upload: bool, fail_fast: bool, debug: bool):
    """Run the Open Arena CLI workflow from a YAML configuration file."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        _logger.info(f"Loading configuration from: {config_path}")
        try:
            config = ExperimentsFile.from_yaml(config_path)
        except ValidationError as e:
            _logger.error("Configuration validation failed:")
            for error in e.errors():
                _logger.error(f"  {error['loc']}: {error['msg']}")
            sys.exit(1)
        _logger.info(f"Validated: {len(config.experiments)} experiments")

        async def workflow():
            rows = _load_rows(config)
            if config.dataset.source.get("provider") == "langfuse":
                _logger.info("Source is Langfuse; skipping upload (items already exist remotely)")
            elif skip_upload:
                rows = attach_existing_dataset(rows, config.dataset.name)
            else:
                rows = await upload_rows(rows, dataset_name=config.dataset.name, description=config.dataset.description or "")
            results = await _run_experiments(config, rows, fail_fast=fail_fast)
            await _run_evaluations(config, results)
            get_client().flush()

        asyncio.run(workflow())
        _logger.info("Execution completed. View results in Langfuse.")
        sys.exit(0)

    except FileNotFoundError as e:
        _logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        _logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
