"""Which observations does each task store, and which does the model actually read?

Storing an observation the encoder never looks at costs replay bandwidth on every
step for nothing, and the replay pipeline is the throughput bottleneck here.
"""
import glob
import os
import sys

sys.path.insert(0, ".")

import car_dreamer

TASKS = sys.argv[1:] or [
    "carla_overtake_d15",
    "carla_four_lane_prosafeav",
    "carla_four_lane_prosafeav_dense",
    "carla_lane_merge_prosafeav",
    "carla_right_turn_prosafeav",
    "carla_roundabout_prosafeav",
]

print(f"{'task':36s} {'town':8s} {'stored observations':46s} {'encoded':16s} {'unused'}")
print("-" * 130)
for t in TASKS:
    try:
        c = car_dreamer.load_task_configs(t)
    except Exception as exc:
        print(f"{t:36s} (failed: {exc})")
        continue
    stored = list(c.env.observation.enabled)
    enc = c.dreamerv3.get("encoder.cnn_keys", "?")
    # Image observations are the expensive ones; the rest are a few floats.
    heavy = [k for k in stored
             if tuple(c.env.observation.get(k, {}).get("shape", [])) and
             len(c.env.observation.get(k, {}).get("shape", [])) == 3]
    unused = [k for k in heavy if k != enc]
    print(f"{t:36s} {c.env.world.town:8s} {str(stored):46s} {str(enc):16s} {unused}")

print("\nreplay chunk sizes on disk:")
for d in sorted(glob.glob("logdir/*_s0/replay")):
    files = glob.glob(os.path.join(d, "*.npz"))
    if not files:
        continue
    size = sum(os.path.getsize(f) for f in files) / max(len(files), 1)
    run = os.path.basename(os.path.dirname(d))
    print(f"  {run:34s} {len(files):4d} chunks, {size/1e6:6.2f} MB each")
