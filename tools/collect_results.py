"""
Aggregate evaluation runs into the tables the manuscript needs.

``dreamerv3/eval_stats.py`` reports a fixed list of metrics for a single run.
The revision needs mean +/- standard error *across seeds*, for a dozen
configurations, including the new conflict and EVT metrics.  This script scans
``metrics.jsonl`` generically -- every ``stats/*`` and ``episode/*`` key is
picked up, so metrics added to the environment info dict appear here without
further changes -- groups runs by configuration name, and emits a Markdown or
LaTeX table.

A run directory is expected to be named ``<config>_s<seed>``.

Examples::

    python tools/collect_results.py --logdir logdir --pattern 'prosafeav_*'
    python tools/collect_results.py --logdir logdir --pattern 'prosafeav_*' \\
        --metrics is_collision out_of_lane ttc drac evt_severity \\
        --latex logdir/results_table.tex
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Metric name -> (label, unit scale, higher_is_better, decimals)
KNOWN = {
    "is_collision": ("Collision rate (\\%)", 100.0, False, 2),
    "out_of_lane": ("Lane departure rate (\\%)", 100.0, False, 2),
    "destination_reached": ("Success rate (\\%)", 100.0, True, 2),
    "travel_distance": ("Travel distance (m)", 1.0, True, 2),
    "ttc": ("Time-to-collision (s)", 1.0, True, 2),
    "drac": ("DRAC (m/s$^2$)", 1.0, False, 3),
    "evt_severity": ("EVT severity", 1.0, False, 4),
    "evt_tail_prob": ("Joint tail probability", 1.0, False, 5),
    "evt_crash_prob": ("EVT crash probability", 1.0, False, 6),
    "speed_norm": ("Speed (m/s)", 1.0, True, 2),
    "n_interacting": ("Interacting vehicles", 1.0, None, 2),
    "score": ("Episode return", 1.0, True, 2),
    "length": ("Episode length", 1.0, None, 1),
}

DEFAULT_METRICS = [
    "is_collision", "out_of_lane", "travel_distance", "ttc", "drac",
    "evt_severity", "speed_norm", "score",
]


def read_run(run_dir):
    """Per-episode series for one run, keyed by bare metric name."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return {}
    series = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, value in row.items():
                if not isinstance(value, (int, float)):
                    continue
                if key.startswith("stats/sum_"):
                    series[key[len("stats/sum_"):]].append(value)
                elif key.startswith("stats/mean_"):
                    series[key[len("stats/mean_"):]].append(value)
                elif key in ("episode/score", "episode/length"):
                    series[key.split("/", 1)[1]].append(value)
    return {k: np.asarray(v, dtype=float) for k, v in series.items() if len(v)}


def aggregate(runs, metric):
    """Mean over episodes per seed, then mean +/- standard error over seeds."""
    per_seed = []
    for series in runs:
        if metric in series and series[metric].size:
            per_seed.append(float(np.mean(series[metric])))
    if not per_seed:
        return None
    arr = np.asarray(per_seed)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sem, arr.size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default="logdir")
    ap.add_argument("--pattern", default="*", help="glob over run directory names")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--latex", default=None, help="also write a LaTeX table here")
    ap.add_argument("--json", default=None, help="also write the raw numbers here")
    args = ap.parse_args()

    root = Path(args.logdir)
    groups = defaultdict(list)
    for run_dir in sorted(root.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        match = re.match(r"^(.*)_s(\d+)$", run_dir.name)
        config = match.group(1) if match else run_dir.name
        series = read_run(run_dir)
        if series:
            groups[config].append(series)

    if not groups:
        print(f"no runs with metrics.jsonl under {root}/{args.pattern}")
        return

    configs = sorted(groups)
    table = {}
    for metric in args.metrics:
        row = {}
        for config in configs:
            row[config] = aggregate(groups[config], metric)
        if any(v is not None for v in row.values()):
            table[metric] = row

    # ---- console ---------------------------------------------------------- #
    width = max(len(c) for c in configs) + 2
    header = "metric".ljust(28) + "".join(c.ljust(width) for c in configs)
    print(header)
    print("-" * len(header))
    for metric, row in table.items():
        label, scale, _, dec = KNOWN.get(metric, (metric, 1.0, None, 3))
        line = re.sub(r"\\\\.|\\$|\\{|\\}", "", label)[:26].ljust(28)
        for config in configs:
            cell = row[config]
            if cell is None:
                line += "-".ljust(width)
            else:
                mean, sem, n = cell
                line += f"{mean * scale:.{dec}f}+-{sem * scale:.{dec}f}".ljust(width)
        print(line)
    print()
    for config in configs:
        print(f"  {config}: {len(groups[config])} seed(s)")

    # ---- LaTeX ------------------------------------------------------------ #
    if args.latex:
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Task performance (mean $\\pm$ standard error over seeds).}",
            "\\label{tab:revision_results}",
            "\\resizebox{\\linewidth}{!}{",
            "\\begin{tabular}{l" + "c" * len(configs) + "}",
            "\\toprule",
            "\\textbf{Metric} & "
            + " & ".join("\\textbf{" + c.replace("_", "\\_") + "}" for c in configs)
            + " \\\\",
            "\\midrule",
        ]
        for metric, row in table.items():
            label, scale, higher, dec = KNOWN.get(metric, (metric.replace("_", "\\_"), 1.0, None, 3))
            arrow = "" if higher is None else (" $\\uparrow$" if higher else " $\\downarrow$")
            values = {c: (row[c][0] * scale if row[c] else None) for c in configs}
            present = [v for v in values.values() if v is not None]
            best = None
            if present and higher is not None:
                best = max(present) if higher else min(present)
            cells = []
            for config in configs:
                cell = row[config]
                if cell is None:
                    cells.append("--")
                    continue
                mean, sem, _ = cell
                text = f"{mean * scale:.{dec}f} $\\pm$ {sem * scale:.{dec}f}"
                if best is not None and abs(mean * scale - best) < 1e-12:
                    text = "\\textbf{" + text + "}"
                cells.append(text)
            lines.append(f"{label}{arrow} & " + " & ".join(cells) + " \\\\")
        lines += [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table*}",
        ]
        Path(args.latex).write_text("\n".join(lines) + "\n")
        print(f"\nLaTeX table written to {args.latex}")

    if args.json:
        payload = {
            m: {c: (None if v is None else {"mean": v[0], "sem": v[1], "n_seeds": v[2]})
                for c, v in row.items()}
            for m, row in table.items()
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"raw numbers written to {args.json}")


if __name__ == "__main__":
    main()
