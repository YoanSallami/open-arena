# License Apache 2.0: (c) 2026 Athena-Reply

"""Analyze a finished sweep — for every candidate experiment-level reward,
report best-model agreement and mean per-dataset Spearman correlation
against the primary `reward` column.

Reads `.kt/last_run.tsv` (the long-format sidecar written by the sweep)
by default; pass an alternate path as a positional arg. Output is one
line per candidate, `<alias>\\t<agreement>\\t<spearman>`, formatted to
drop straight into `results.tsv`.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PRIMARY = "reward"


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    keys = list(a)
    n = len(keys)
    if n < 2:
        return 1.0
    rank_a = {m: r for r, m in enumerate(sorted(keys, key=lambda m: a[m]))}
    rank_b = {m: r for r, m in enumerate(sorted(keys, key=lambda m: b[m]))}
    d2 = sum((rank_a[m] - rank_b[m]) ** 2 for m in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        sys.stderr.write(f"analyze: {path} is missing or empty\n")
        sys.exit(1)

    cols: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    with path.open() as f:
        next(f)
        for model, dataset, metric, value in csv.reader(f, delimiter="\t"):
            cols[(dataset, metric)][model] = float(value)

    datasets = sorted({d for d, _ in cols})
    candidates = sorted({m for _, m in cols} - {PRIMARY})

    if not candidates:
        sys.stderr.write(
            "analyze: no candidate metrics in TSV — add an entry to "
            "`experiments.rewards` in config.yaml.\n"
        )
        sys.exit(1)

    for cand in candidates:
        hits = 0
        sps: list[float] = []
        for d in datasets:
            prim = cols.get((d, PRIMARY))
            ccol = cols.get((d, cand))
            if not prim or not ccol or set(prim) != set(ccol):
                continue
            if max(prim, key=prim.get) == max(ccol, key=ccol.get):
                hits += 1
            sps.append(_spearman(prim, ccol))
        agreement = hits / len(datasets) if datasets else 0.0
        spearman = mean(sps) if sps else 0.0
        print(f"{cand}\t{agreement:.6f}\t{spearman:.6f}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".kt/last_run.tsv")
    main(path)
