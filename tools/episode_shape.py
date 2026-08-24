"""Are the episodes ending because the task was completed, or because it failed?

Episode length is the cheapest sanity check on a driving scenario that is
available from the training log alone.  The route is a fixed distance, so at the
desired speed there is an expected number of steps to finish it.  Much shorter
means episodes are being cut off by collisions or lane departures; a pile-up at
the time limit means the vehicle is too slow to finish at all.  Either way the
scenario is not measuring what it is supposed to.
"""
import glob
import json
import os
import sys

import numpy as np

logdir = sys.argv[1] if len(sys.argv) > 1 else "logdir"
cap = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
LIMIT = 500  # terminal.time_limit

print(f"{'run':26s} {'eps':>5s} {'median':>7s} {'p10':>6s} {'p90':>6s} "
      f"{'at limit':>9s} {'<100 steps':>11s} {'score@short':>12s} {'score@long':>11s}")
print("-" * 104)
for d in sorted(glob.glob(os.path.join(logdir, "*_s[0-9]"))):
    p = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(p):
        continue
    lens, scores = [], []
    for line in open(p, errors="ignore"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("step", 0) > cap:
            continue
        if "episode/length" in r:
            lens.append(r["episode/length"])
        if "episode/score" in r:
            scores.append(r["episode/score"])
    if len(lens) < 20:
        continue
    L = np.asarray(lens, dtype=float)
    S = np.asarray(scores[:len(L)], dtype=float) if len(scores) >= len(L) else None
    at_limit = float((L >= LIMIT - 2).mean())
    short = float((L < 100).mean())
    if S is not None and len(S) == len(L):
        s_short = S[L < 100].mean() if (L < 100).any() else float("nan")
        s_long = S[L >= LIMIT - 2].mean() if (L >= LIMIT - 2).any() else float("nan")
    else:
        s_short = s_long = float("nan")
    print(f"{os.path.basename(d):26s} {len(L):5d} {np.median(L):7.0f} "
          f"{np.percentile(L, 10):6.0f} {np.percentile(L, 90):6.0f} "
          f"{at_limit:8.0%} {short:10.0%} {s_short:12.1f} {s_long:11.1f}")

print(f"\ntime limit {LIMIT} steps; the 100 m route takes about 200 steps at the "
      f"desired 5 m/s")
