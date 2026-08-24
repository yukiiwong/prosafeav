"""Summarise the sweep so far: one row per configuration, at a common horizon.

Reports the mean over the last fifth of each run, averaged across seeds, so a
configuration is judged on where it ended up rather than on its whole history.
Every configuration is truncated at the same step count, because runs in this
sweep were launched under different budgets.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

KEYS = [
    ("episode/score", "return", 1),
    ("stats/max_collision", "collision", 1),
    ("train/imag_evt_risk_mean", "evt_risk", 5),
    ("train/imag_evt_active_frac", "evt_active", 5),
    ("train/imag_evt_penalty", "evt_penalty", 4),
    ("train/safety_loss_mean", "safety_loss", 4),
]


def load(run_dir, cap):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    out = defaultdict(list)
    for line in open(path, errors="ignore"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step", 0)
        if cap and step > cap:
            continue
        for k, v in row.items():
            if isinstance(v, (int, float)):
                out[k].append((step, v))
    return out


def tail_mean(series, key, frac=0.2):
    if not series or key not in series:
        return None
    arr = np.asarray(series[key], dtype=float)
    if not len(arr):
        return None
    cut = arr[:, 0].max() * (1 - frac)
    sel = arr[arr[:, 0] >= cut]
    return float(sel[:, 1].mean()) if len(sel) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logdir")
    ap.add_argument("--cap", type=int, default=60000)
    args = ap.parse_args()

    groups = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(args.logdir, "*_s[0-9]"))):
        m = re.match(r"^(.*)_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        series = load(d, args.cap)
        if series:
            groups[m.group(1)].append(series)

    if not groups:
        print("no runs found")
        return

    head = f"{'configuration':26s} {'seeds':>5s} " + " ".join(f"{lbl:>12s}" for _, lbl, _ in KEYS)
    print(head)
    print("-" * len(head))
    for name in sorted(groups):
        runs = groups[name]
        cells = []
        for key, _, dec in KEYS:
            vals = [v for v in (tail_mean(r, key) for r in runs) if v is not None]
            if not vals:
                cells.append(f"{'-':>12s}")
                continue
            mean = np.mean(vals)
            if len(vals) > 1:
                sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
                cells.append(f"{mean:.{dec}f}+-{sem:.{dec}f}".rjust(12))
            else:
                cells.append(f"{mean:.{dec}f}".rjust(12))
        print(f"{name:26s} {len(runs):5d} " + " ".join(cells))
    print(f"\nlast 20% of each run, truncated at step {args.cap}")


if __name__ == "__main__":
    main()
