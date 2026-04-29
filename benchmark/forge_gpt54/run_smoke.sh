#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-benchmark/forge_gpt54/verifier.yaml}"
MODE="${2:-one}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

case "$MODE" in
  one)
    ;;
  three)
    ;;
  *)
    echo "Usage: benchmark/forge_gpt54/run_smoke.sh [config_path] [one|three] [analyzer flags...]" >&2
    exit 1
    ;;
esac

TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/forge-gpt54-smoke.XXXXXX.yaml")"
cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

uv run python - "$CONFIG_PATH" "$MODE" "$TMP_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config_path, mode, output_path = sys.argv[1:4]
tasks_by_mode = {
    "one": ["regex-chess"],
    "three": ["regex-chess", "mailman", "extract-moves-from-video"],
}

with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["dataset"]["source"]["tasks"] = tasks_by_mode[mode]
config["dataset"]["source"]["mixed_only"] = True
config["dataset"]["name"] = f"{config['dataset']['name']}-{Path(config_path).stem}-{mode}"

with open(output_path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"Prepared smoke config: {output_path}")
print(f"Dataset name: {config['dataset']['name']}")
print(f"Tasks: {tasks_by_mode[mode]}")
PY

uv run python benchmark/forge_gpt54/analyze_terminal_bench.py --config "$TMP_CONFIG" "$@"
