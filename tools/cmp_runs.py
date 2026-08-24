"""Compare a few runs on the raw episode statistics available during training."""
import json
import sys

import numpy as np

runs = sys.argv[1:] or ["prosafeav_fourlane_s0", "prosafeav_d15_s0", "prosafeav_s0"]

print(f"{'run':28s} {'eps':>5s} {'score':>9s} {'length':>8s} {'collision':>10s}")
print("-" * 66)
for r in runs:
    sc, ln, co = [], [], []
    try:
        fh = open(f"logdir/{r}/metrics.jsonl", errors="ignore")
    except FileNotFoundError:
        print(f"{r:28s} (missing)")
        continue
    for line in fh:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("step", 0) > 60000:
            continue
        if "episode/score" in d:
            sc.append(d["episode/score"])
        if "episode/length" in d:
            ln.append(d["episode/length"])
        if "stats/max_collision" in d:
            co.append(d["stats/max_collision"])
    if not sc:
        print(f"{r:28s} (no episodes)")
        continue
    n = max(1, len(sc) // 5)
    coll = np.mean(co[-max(1, len(co) // 5):]) if co else float("nan")
    print(f"{r:28s} {len(sc):5d} {np.mean(sc[-n:]):9.1f} "
          f"{np.mean(ln[-n:]):8.1f} {coll:10.1f}")
