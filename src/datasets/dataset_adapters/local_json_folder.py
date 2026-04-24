import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.datasets.base import Dataset

_EXPECTED_TRIAL_COUNT = 5
_MAX_TRACE_AGENT_STEPS = 12


class LocalJsonFolderDataset(Dataset):
    """Load one task folder of pre-recorded agent trials into one dataset row.

    Expects each task directory to contain one `trials_metadata.json` plus
    exactly `_EXPECTED_TRIAL_COUNT` `*_trajectory.json` files. Yields one row
    per task with the task prompt, per-trial formatted trace strings, rewards,
    and trial IDs, suitable for replay-based benchmarking.
    """

    def __init__(
        self,
        name: str,
        input_template: str,
        expected_output_template: str,
        path: str,
        mixed_only: bool = False,
        tasks: list[str] | None = None,
        limit: int | None = None,
    ):
        super().__init__(name, input_template, expected_output_template, limit)
        self.path = Path(path)
        self.mixed_only = mixed_only
        self.tasks = set(tasks or [])

    def iter_raw_rows(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_dir():
            raise FileNotFoundError(f"JSON-folder dataset path not found: {self.path}")

        task_dirs = {path.name: path for path in self.path.iterdir() if path.is_dir()}
        if self.tasks:
            missing = sorted(self.tasks - set(task_dirs))
            if missing:
                raise ValueError(f"Requested task(s) not found under {self.path}: {missing}")
            names = sorted(self.tasks)
        else:
            names = sorted(task_dirs)

        for task_name in names:
            row = self._load_task(task_dirs[task_name])
            if row is not None:
                yield row

    def _load_task(self, task_dir: Path) -> dict[str, Any] | None:
        metadata_path = task_dir / "trials_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing trials metadata for task {task_dir.name}: {metadata_path}")

        trial_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(trial_metadata) != _EXPECTED_TRIAL_COUNT:
            raise ValueError(
                f"Expected {_EXPECTED_TRIAL_COUNT} trial metadata rows in {metadata_path}, "
                f"found {len(trial_metadata)}"
            )
        by_trial_id = {record["trialId"]: record for record in trial_metadata}
        if len(by_trial_id) != len(trial_metadata):
            raise ValueError(f"Duplicate trialId values in {metadata_path}")

        trajectory_files = sorted(task_dir.glob("*_trajectory.json"))
        if not trajectory_files:
            raise ValueError(f"No trajectory JSON files found in {task_dir}")
        if len(trajectory_files) != _EXPECTED_TRIAL_COUNT:
            raise ValueError(
                f"Expected {_EXPECTED_TRIAL_COUNT} trajectory files in {task_dir}, "
                f"found {len(trajectory_files)}"
            )

        trials: list[dict[str, Any]] = []
        task_prompt: str | None = None
        task_name: str | None = None

        for trajectory_path in trajectory_files:
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))

            trial_id = str(payload["trial_id"])
            meta = by_trial_id.get(trial_id)
            if meta is None:
                raise ValueError(f"Trial {trial_id} from {trajectory_path.name} missing from {metadata_path.name}")

            current_task_name = str(payload["task_name"])
            if task_name is None:
                task_name = current_task_name
            elif current_task_name != task_name:
                raise ValueError(f"Inconsistent task_name values under {task_dir}: {current_task_name!r} != {task_name!r}")

            prompt = _extract_task_prompt(payload)
            if task_prompt is None:
                task_prompt = prompt
            elif prompt != task_prompt:
                raise ValueError(f"Inconsistent task prompt across trials for task {task_dir.name}")

            reward = int(meta["reward"])
            if int(payload["reward"]) != reward:
                raise ValueError(
                    f"Reward mismatch for trial {trial_id}: "
                    f"trajectory={payload['reward']!r} metadata={meta['reward']!r}"
                )

            trials.append({
                "trial_id": trial_id,
                "reward": reward,
                "output": _format_trace(payload["trajectory"]),
                # Keep replay execution lightweight: the benchmark prompt is
                # based on the formatted trace text, so we do not need to
                # ship the bulky structured trajectory into evaluator calls.
                "trajectory": [],
            })

        trials.sort(key=lambda trial: trial["trial_id"])
        if len(trials) != _EXPECTED_TRIAL_COUNT:
            raise ValueError(
                f"Expected {_EXPECTED_TRIAL_COUNT} loaded trials for task {task_dir.name}, "
                f"found {len(trials)}"
            )
        rewards = [int(trial["reward"]) for trial in trials]
        if self.mixed_only and len(set(rewards)) <= 1:
            return None

        if task_name is None or task_prompt is None:
            raise ValueError(f"Task {task_dir.name} did not yield a valid name and prompt")

        row: dict[str, Any] = {
            "task_name": task_name,
            "task_prompt": task_prompt,
            "trial_ids": [trial["trial_id"] for trial in trials],
            "rewards": rewards,
        }
        for index, trial in enumerate(trials, start=1):
            row[f"trial_{index}_output"] = trial["output"]
            row[f"trial_{index}_trajectory"] = trial["trajectory"]
        return row


def _extract_task_prompt(payload: dict[str, Any]) -> str:
    steps = payload["trajectory"]["steps"]
    task_name = str(payload.get("task_name") or "<unknown>")
    for step in steps:
        if step.get("source") != "user":
            continue
        message = str(step.get("message") or "")
        if message and not (message.startswith("$") and len(message) < 5):
            return message

    # Upstream terminal-bench falls back to the agent's first analysis when
    # the original user task is missing from the trace.
    parts: list[str] = []
    for step in steps:
        if step.get("source") != "agent":
            continue
        message = str(step.get("message") or "").strip()
        if not message:
            continue
        parts.append(message)
        if len(parts) >= 2:
            break
    if parts:
        joined = "\n\n".join(parts)
        return (
            f"[Task: {task_name}]\n"
            "The original task instruction was not captured. Below is the "
            "agent's initial analysis:\n\n"
            f"{joined}"
        )

    raise ValueError(f"No usable task prompt found for trial {payload.get('trial_id', '<unknown>')}")


def _format_trace(trajectory: dict[str, Any] | None) -> str:
    """Render the upstream terminal-bench trace text for one trial."""
    if not trajectory:
        return "(no trajectory data)"

    parts: list[str] = []
    agent_steps = [
        step
        for step in trajectory.get("steps", [])
        if step.get("source") == "agent"
    ]
    if len(agent_steps) > _MAX_TRACE_AGENT_STEPS:
        parts.append(
            f"[Trace truncated to final {_MAX_TRACE_AGENT_STEPS} agent steps]"
        )

    for step in agent_steps[-_MAX_TRACE_AGENT_STEPS:]:
        source = step.get("source", "")
        message = str(step.get("message") or "")
        step_id = step.get("step_id", "?")
        if source != "agent":
            continue

        parts.append(f"--- Agent Step {step_id} ---")
        if message:
            parts.append(message)

        for tool_call in step.get("tool_calls", []):
            arguments = tool_call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            keystrokes = str(arguments.get("keystrokes") or "").rstrip()
            if keystrokes:
                parts.append(f"[Command] {keystrokes}")

        observation = step.get("observation") or {}
        for result in observation.get("results", []):
            if not isinstance(result, dict):
                continue
            content = str(result.get("content") or "")
            if content:
                parts.append(f"[Output]\n{content}")

        parts.append("")

    return "\n".join(parts) if parts else "(no trajectory data)"
