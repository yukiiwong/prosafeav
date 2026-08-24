"""Live aggregate throughput across the running jobs, with the dataset share."""
import glob
import json
import os
import statistics
import sys
import time

logdir = sys.argv[1] if len(sys.argv) > 1 else "logdir"
total, n = 0.0, 0
for d in sorted(glob.glob(os.path.join(logdir, "*_s[0-9]"))):
    p = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(p) or os.path.getmtime(p) < time.time() - 900:
        continue
    fps, step, ds = [], 0, None
    for line in open(p, errors="ignore"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "fps" in r:
            fps.append(r["fps"])
        if "timer/dataset_frac" in r:
            ds = r["timer/dataset_frac"]
        step = max(step, r.get("step", 0))
    if len(fps) >= 2:
        m = statistics.median(fps[-8:])
        total += m
        n += 1
        share = f"{ds:.2f}" if ds is not None else "-"
        print(f"{os.path.basename(d):32s} step={step:6d} fps={m:5.2f} dataset={share}")
    else:
        print(f"{os.path.basename(d):32s} step={step:6d} (warming up)")
print(f"\naggregate {total:.2f} steps/s across {n} jobs")
