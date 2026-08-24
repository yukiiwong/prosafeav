"""What would a shorter budget actually cost?

Rather than asking when a curve "converges" -- which is close to tautological for
a monotonically rising curve -- this reports the smoothed level reached at
several fractions of the budget, relative to the level at the end.  That is the
number the decision turns on: if stopping at two thirds costs a couple of
percent, the sweep is paying for training that changes nothing.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

FRACTIONS = [0.33, 0.50, 0.67, 0.83, 1.00]


def series(run_dir, key, cap=None):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    pts = []
    for line in open(path, errors="ignore"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if key in row and isinstance(row[key], (int, float)):
            step = row.get("step", 0)
            if cap and step > cap:
                continue
            pts.append((step, row[key]))
    return np.asarray(pts, dtype=float) if pts else None


def level_at(arr, frac, width=0.12):
    """Mean of the metric in a window ending at `frac` of the run."""
    last = arr[:, 0].max()
    hi = last * frac
    lo = max(0.0, hi - last * width)
    sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] <= hi)]
    return float(sel[:, 1].mean()) if len(sel) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logdir")
    ap.add_argument("--key", default="episode/score")
    ap.add_argument("--min-steps", type=int, default=50000,
                    help="ignore runs shorter than this")
    args = ap.parse_args()

    rows, rel = [], defaultdict(list)
    for d in sorted(glob.glob(os.path.join(args.logdir, "*_s[0-9]"))):
        arr = series(d, args.key)
        if arr is None or arr[:, 0].max() < args.min_steps or len(arr) < 30:
            continue
        levels = [level_at(arr, f) for f in FRACTIONS]
        if any(v is None for v in levels):
            continue
        final = levels[-1]
        rows.append((os.path.basename(d), int(arr[:, 0].max()), levels))
        span = max(abs(final), 1e-9)
        for f, v in zip(FRACTIONS, levels):
            # Express as a shortfall against the final level, in percent.
            rel[f].append(100.0 * (v - final) / span)

    if not rows:
        print("no completed runs")
        return

    head = f"{'run':30s} {'steps':>7s} " + " ".join(f"{int(f*100):>8d}%" for f in FRACTIONS)
    print(f"{args.key}\n")
    print(head)
    print("-" * len(head))
    for name, last, levels in rows:
        print(f"{name:30s} {last:7d} " + " ".join(f"{v:9.1f}" for v in levels))

    print("\nshortfall against the final level, median across runs:")
    for f in FRACTIONS:
        vals = np.asarray(rel[f])
        print(f"  at {int(f*100):3d}% of the budget: {np.median(vals):+6.1f}%"
              f"   (worst run {np.min(vals):+6.1f}%)")


if __name__ == "__main__":
    main()
