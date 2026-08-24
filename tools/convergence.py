"""When does a run stop improving?

Decides the step budget from the data rather than from a guess.  For each
completed run, the episode return is smoothed, and the earliest step is found at
which the smoothed curve first reaches a given fraction of its final level and
stays there.  If that step is well short of the budget the sweep is paying for
training that changes nothing.

Reported per run and per configuration, alongside the same statistic for the
metrics the manuscript actually cares about, since return can plateau while the
safety behaviour is still moving.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


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


def smooth(arr, window=25):
    """Running mean over episodes, then resampled onto a step grid."""
    if len(arr) < window:
        window = max(3, len(arr) // 3)
    if len(arr) < 3:
        return arr
    kernel = np.ones(window) / window
    vals = np.convolve(arr[:, 1], kernel, mode="valid")
    steps = arr[window - 1:, 0]
    return np.column_stack([steps, vals])


def convergence_step(arr, frac=0.95, window=25):
    """First step at which the smoothed curve reaches `frac` of its final level
    and never falls back below it."""
    sm = smooth(arr, window)
    if len(sm) < 5:
        return None, None, None
    final = sm[-max(1, len(sm) // 10):, 1].mean()
    lo = sm[:, 1].min()
    # Work in the run's own range so a negative baseline does not break the ratio.
    target = lo + frac * (final - lo)
    hits = np.where(sm[:, 1] >= target)[0]
    if not len(hits):
        return None, final, sm[-1, 0]
    # Require it to stay above from that point on.
    for i in hits:
        if (sm[i:, 1] >= target * 0.97).mean() > 0.8:
            return float(sm[i, 0]), float(final), float(sm[-1, 0])
    return float(sm[hits[-1], 0]), float(final), float(sm[-1, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logdir")
    ap.add_argument("--frac", type=float, default=0.95)
    ap.add_argument("--cap", type=int, default=0, help="0 uses each run's full history")
    ap.add_argument("--keys", nargs="*",
                    default=["episode/score", "stats/max_collision",
                             "train/imag_evt_risk_mean"])
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(args.logdir, "*_s[0-9]")))
    if not runs:
        print("no runs")
        return

    print(f"step at which the smoothed curve reaches {args.frac:.0%} of its final level\n")
    per_key = defaultdict(list)
    for key in args.keys:
        print(f"--- {key}")
        print(f"  {'run':32s} {'converged':>10s} {'of':>8s} {'final':>10s}")
        for d in runs:
            arr = series(d, key, args.cap or None)
            if arr is None or len(arr) < 10:
                continue
            step, final, last = convergence_step(arr, args.frac)
            if step is None:
                continue
            per_key[key].append(step / last if last else 1.0)
            print(f"  {os.path.basename(d):32s} {int(step):10d} {int(last):8d} {final:10.3g}")
        if per_key[key]:
            frac = np.median(per_key[key])
            print(f"  median: converged at {frac:.0%} of the budget\n")
        else:
            print("  (no usable runs)\n")


if __name__ == "__main__":
    main()
