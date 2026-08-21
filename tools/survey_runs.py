"""One-line-per-run survey of what is on disk: how far each got and how it did.

Useful after a sweep has been stopped, restarted or reconfigured, when the run
directories no longer all correspond to the same budget.
"""
import glob
import json
import os
import sys

logdir = sys.argv[1] if len(sys.argv) > 1 else "logdir"
rows = []
for d in sorted(glob.glob(os.path.join(logdir, "*_s[0-9]"))):
    p = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(p):
        rows.append((os.path.basename(d), 0, 0, float("nan"), False, "no metrics"))
        continue
    steps, scores = [], []
    for line in open(p):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in r:
            steps.append(r["step"])
        if "episode/score" in r:
            scores.append(r["episode/score"])
    restarts = 0
    log = os.path.join(d, "run.log")
    if os.path.exists(log):
        restarts = sum(1 for line in open(log, errors="ignore") if "restart " in line)
    tail = scores[-20:]
    rows.append((
        os.path.basename(d),
        max(steps) if steps else 0,
        len(scores),
        sum(tail) / len(tail) if tail else float("nan"),
        os.path.isdir(os.path.join(d, "replay")),
        f"{restarts} restarts" if restarts else "",
    ))

print(f"{len(rows)} run directories\n")
print(f"{'run':32s} {'max_step':>9s} {'episodes':>8s} {'last20':>9s}  {'replay':6s} note")
print("-" * 90)
for n, s, e, sc, rp, note in rows:
    print(f"{n:32s} {s:9d} {e:8d} {sc:9.1f}  {str(rp):6s} {note}")
