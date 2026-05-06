# License Apache 2.0: (c) 2026 Athena-Reply

"""Analyze a finished sweep.

Two outputs:

* **Model selection** (stderr) — for each dataset, the best model under
  the primary `reward` column plus the margin to the runner-up.
* **Reward R&D** (stderr summary + stdout TSV) — for every candidate
  experiment-level reward, report:
    - `top1`      — fraction of usable datasets where the candidate's #1
                    model matches the primary's #1
    - `pairwise`  — fraction of (model_i, model_j) pairs across usable
                    datasets where primary and candidate agree on which
                    is the stronger model (a Kendall-style score; ties
                    on the same side count as agreement, ties on only
                    one side count as half)
    - `sp_min` / `sp_med` / `sp_max`  — distribution of per-dataset
                    Spearman ρ (primary rank vs candidate rank), instead
                    of just the mean which hid catastrophic single-dataset
                    failures
    - `n_usable`  — number of datasets contributing to the stats (i.e.
                    those with both primary and candidate scores for the
                    same set of models)

Plus a stderr disagreement breakdown listing the datasets where the
candidate would have crowned a different #1 model.

stdout shape (one row per candidate, drops into `results.tsv`):

    <alias>\\t<top1>\\t<pairwise>\\t<sp_min>\\t<sp_med>\\t<sp_max>\\t<n_usable>

Reads `.kt/last_run.tsv` (the long-format sidecar written by the sweep)
by default; pass an alternate path as a positional arg.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

PRIMARY = "reward"


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Spearman rank correlation with average-rank tie handling.

    Pearson correlation of the two average-rank vectors — the standard
    Spearman definition. The simplified `1 - 6Σd²/(n(n²-1))` form only
    holds when there are no ties; with ties it spuriously reports
    disagreement that depends on input ordering.

    Returns 1.0 in degenerate cases where there is no ordering signal at
    all (n < 2, or both sides are constant). Returns 0.0 when exactly one
    side is constant — Pearson is undefined there and "no correlation" is
    the least-misleading scalar to fold into min/median/max stats.
    """
    keys = list(a)
    n = len(keys)
    if n < 2:
        return 1.0
    ra = _average_ranks([a[k] for k in keys])
    rb = _average_ranks([b[k] for k in keys])
    mean = (n + 1) / 2  # average rank is always (n+1)/2 with fractional ranks
    num = sum((x - mean) * (y - mean) for x, y in zip(ra, rb))
    da = sum((x - mean) ** 2 for x in ra)
    db = sum((y - mean) ** 2 for y in rb)
    if da == 0 and db == 0:
        return 1.0
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db) ** 0.5


def _average_ranks(values: list[float]) -> list[float]:
    """Fractional ranks (1-based): tied values share the average of the
    positions they would occupy. E.g. `[10, 20, 20, 30]` → `[1, 2.5, 2.5, 4]`.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pairwise_agreement(a: dict[str, float], b: dict[str, float]) -> float:
    """Kendall-style pairwise agreement between two metric scorings.

    For every (m_i, m_j) model pair, count whether the two metrics agree
    on which is better. Same direction (or both tied) → 1; one tied and
    one decisive → 0.5; actively inverted → 0. Returns the average. With
    `n < 2` models, returns 1.0 (vacuously consistent).
    """
    models = list(a)
    n = len(models)
    if n < 2:
        return 1.0
    total = n * (n - 1) // 2
    score = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = models[i], models[j]
            sa = (a[mi] > a[mj]) - (a[mi] < a[mj])
            sb = (b[mi] > b[mj]) - (b[mi] < b[mj])
            if sa == sb:
                score += 1.0
            elif sa == 0 or sb == 0:
                score += 0.5
    return score / total


def _top_set(scores: dict[str, float]) -> frozenset[str]:
    """Models tied at `max(scores.values())`."""
    top = max(scores.values())
    return frozenset(m for m, s in scores.items() if s == top)


def _best_per_dataset(cols, datasets):
    """`{dataset: (best_models, best_score, margin_or_None)}` under primary.

    `best_models` is a tuple of all models tied at the top score (one entry
    when there's a unique winner). `margin` is the gap to the next
    *distinct* score, or `None` when every model has the same score.
    """
    out = {}
    for d in datasets:
        prim = cols.get((d, PRIMARY))
        if not prim:
            continue
        ranked = sorted(prim.items(), key=lambda kv: -kv[1])
        best_score = ranked[0][1]
        best_models = tuple(m for m, s in ranked if s == best_score)
        next_distinct = next(
            (s for _, s in ranked if s < best_score), None
        )
        margin = (best_score - next_distinct) if next_distinct is not None else None
        out[d] = (best_models, best_score, margin)
    return out


def _print_best_per_dataset(best, datasets):
    """Render the model-selection summary to stderr."""
    sys.stderr.write(f"best model per dataset (primary={PRIMARY!r}):\n")
    if not best:
        sys.stderr.write(f"  (no datasets with primary {PRIMARY!r} scores)\n\n")
        return
    name_w = max(len(d) for d in datasets)
    model_w = max(len(", ".join(ms)) for ms, _, _ in best.values())
    for d in datasets:
        if d not in best:
            sys.stderr.write(f"  {d.ljust(name_w)}  (no successful trials)\n")
            continue
        models, score, margin = best[d]
        models_s = ", ".join(models)
        if margin is None:
            margin_s = "(all tied)" if len(models) > 1 else "(only model)"
        else:
            margin_s = f"Δ +{margin:.4f}"
            if len(models) > 1:
                margin_s += f"  [{len(models)}-way tie]"
        sys.stderr.write(
            f"  {d.ljust(name_w)}  {models_s.ljust(model_w)}  {score:.4f}  {margin_s}\n"
        )
    sys.stderr.write("\n")


def _candidate_stats(cols, datasets, cand):
    """Compute (top1, pairwise, sp_min, sp_med, sp_max, n_usable, disagreements).

    Top1 is the fraction of datasets where the two metrics' top-scoring
    sets *overlap* — i.e. there exists at least one model tied for #1
    under both metrics. Strict equality of the top sets would penalise
    benign tie-break differences (the metrics agreeing on which models
    are best, but ranking them differently within the tied group).

    `disagreements` is a list of `(dataset, primary_top_set, candidate_top_set)`
    tuples for the datasets where the top sets are disjoint.
    """
    top1_hits = 0
    pairwise_per_ds: list[float] = []
    sps: list[float] = []
    disagreements: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    for d in datasets:
        prim = cols.get((d, PRIMARY))
        ccol = cols.get((d, cand))
        if not prim or not ccol or set(prim) != set(ccol):
            continue
        ptop = _top_set(prim)
        ctop = _top_set(ccol)
        if ptop & ctop:
            top1_hits += 1
        else:
            disagreements.append((d, tuple(sorted(ptop)), tuple(sorted(ctop))))
        pairwise_per_ds.append(_pairwise_agreement(prim, ccol))
        sps.append(_spearman(prim, ccol))

    n = len(sps)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0, [])
    pairwise = sum(pairwise_per_ds) / n
    return (top1_hits / n, pairwise, min(sps), median(sps), max(sps), n, disagreements)


def _print_reward_analysis(stats):
    """Render the per-candidate analysis table + disagreement list to stderr."""
    if not stats:
        return
    name_w = max(len("candidate"), max(len(s[0]) for s in stats))
    sys.stderr.write(f"reward analysis vs primary={PRIMARY!r}:\n")
    sys.stderr.write(
        f"  {'candidate'.ljust(name_w)}  {'top1':>6}  {'pairwise':>8}  "
        f"{'ρ_min':>7}  {'ρ_med':>7}  {'ρ_max':>7}  {'n':>3}\n"
    )
    for alias, top1, pw, sp_min, sp_med, sp_max, n, _diss in stats:
        sys.stderr.write(
            f"  {alias.ljust(name_w)}  {top1:6.3f}  {pw:8.3f}  "
            f"{sp_min:+7.3f}  {sp_med:+7.3f}  {sp_max:+7.3f}  {n:>3}\n"
        )
    sys.stderr.write("\n")

    diss = [(s[0], s[7]) for s in stats if s[7]]
    if diss:
        sys.stderr.write(
            "disagreements (datasets where the candidate's top set is disjoint from primary's):\n"
        )
        for cand, ds_list in diss:
            for d, ptop, ctop in ds_list:
                sys.stderr.write(
                    f"  {cand}: {d} — primary→{{{', '.join(ptop)}}}, "
                    f"candidate→{{{', '.join(ctop)}}}\n"
                )
        sys.stderr.write("\n")


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

    # Model selection: always emit, even with zero candidates — picking the
    # right model per dataset doesn't depend on having extra rewards to audit.
    _print_best_per_dataset(_best_per_dataset(cols, datasets), datasets)

    if not candidates:
        sys.stderr.write(
            "analyze: no candidate metrics in TSV — add an entry to "
            "`experiments.rewards` in config.yaml to enable reward analysis.\n"
        )
        return

    stats = [(c, *_candidate_stats(cols, datasets, c)) for c in candidates]
    _print_reward_analysis(stats)

    # stdout TSV: one journal-friendly row per candidate.
    for alias, top1, pw, sp_min, sp_med, sp_max, n, _diss in stats:
        print(
            f"{alias}\t{top1:.6f}\t{pw:.6f}\t"
            f"{sp_min:.6f}\t{sp_med:.6f}\t{sp_max:.6f}\t{n}"
        )


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".kt/last_run.tsv")
    main(path)
