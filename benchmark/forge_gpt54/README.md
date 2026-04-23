# `forge_gpt54` Benchmark

Replay-backed terminal-bench benchmark for the `forge_gpt54` trajectories in
`resources/data/llm-as-a-verifier/data/terminal_trajs/forge_gpt54`.

This benchmark keeps Open Arena's replay and generic verifier machinery, but
aligns the benchmark semantics with upstream `run_terminal_bench.py`:

- use the upstream terminal-bench criteria
- use the upstream "terminal output is ground truth" framing
- keep pairwise comparison over reward-discriminating trial pairs as the
  primary metric for now

## Files

- `verifier.yaml` — replay benchmark config using `llm_as_verifier`
- `analyze_terminal_bench.py` — benchmark-local runner that reports pairwise
  accuracy from the verifier scores
- `run_smoke.sh` — one-task or three-task smoke runner built on the analyzer

## Requirements

- Langfuse environment variables must be set
- Ollama must be running on `http://localhost:11434`
- The grader model used by this benchmark is `qwen2.5-coder:7b-instruct`

## Metric

The primary benchmark result is pairwise accuracy over discriminating trial
pairs within each task. Ties count as half credit. By default the config uses
`mixed_only: true`, so the benchmark focuses on the 17 mixed tasks where
pairwise discrimination matters.

## Smoke tests

Run a one-task smoke test:

```bash
benchmark/forge_gpt54/run_smoke.sh benchmark/forge_gpt54/verifier.yaml one
```

Run the three-task smoke set:

```bash
benchmark/forge_gpt54/run_smoke.sh benchmark/forge_gpt54/verifier.yaml three
```

Additional analyzer flags are forwarded after the mode argument:

```bash
benchmark/forge_gpt54/run_smoke.sh benchmark/forge_gpt54/verifier.yaml one --debug
```

## Full run

Run the full mixed-task verifier benchmark and print the pairwise summary:

```bash
uv run python benchmark/forge_gpt54/analyze_terminal_bench.py --config benchmark/forge_gpt54/verifier.yaml
```

On repeated runs, use `--skip-upload` to attach to the existing Langfuse
dataset items instead of uploading duplicates:

```bash
uv run python benchmark/forge_gpt54/analyze_terminal_bench.py --config benchmark/forge_gpt54/verifier.yaml --skip-upload
```

To capture the raw summary for later inspection:

```bash
uv run python benchmark/forge_gpt54/analyze_terminal_bench.py \
  --config benchmark/forge_gpt54/verifier.yaml \
  --json-out benchmark/forge_gpt54/results/latest.json
```
