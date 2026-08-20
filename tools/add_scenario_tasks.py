"""Derive ProSafeAV task variants for scenarios beyond overtaking.

Reviewer 1 comment 6 is that the study rests on a single overtaking scenario, so
generalisation cannot be judged.  Varying traffic density inside that one
scenario only answers half of it: the *conflict geometry* stays the same
throughout -- a leader ahead in the ego's lane, overtaken on the left.

Every driving environment in CarDreamer inherits ``CarlaWptEnv``, so the
``safety`` and ``evt_params`` observations and the EVT reward term are available
in all of them without further code.  This script derives a ProSafeAV variant of
each selected base task by injecting the ``evt`` configuration block and the EVT
reward weight, which keeps the derived tasks in lockstep with their bases
instead of duplicating a hundred lines of geometry per scenario.

The scenarios chosen span four distinct conflict types:

    overtake     rear-end conflict resolved by a discretionary lane change
    four_lane    multi-lane free driving; conflicts arise from lane changes of
                 both the ego and the surrounding autopilot traffic
    lane_merge   forced merge; the conflict is a converging-path gap acceptance
    right_turn   unsignalised crossing conflict against an oncoming flow
    roundabout   yielding and circulating conflicts with continuous curvature

Crossing geometries matter for the EVT model specifically: the car-following
definition of DRAC is meaningless there, so they exercise the two-dimensional
encroachment indicator that the revision added.
"""

import copy
import sys

import ruamel.yaml as yaml

PATH = "/home/yukai/CarDreamer_prosafeav/car_dreamer/configs/tasks.yaml"

EVT_BLOCK = {
    "mode": "both",
    "copula": "logistic",
    "threshold_method": "stability",
    "threshold_ttc": None,
    "threshold_drac": None,
    "update_interval": 2000,
    "buffer_size": 20000,
    "min_sample": 300,
    "min_exceedances": 30,
    "risk_tolerance": 0.0,
    "indicator_mode": "max",
    "interaction_radius": 50.0,
    "ttc_cap": 30.0,
    "crash_drac": 8.5,
}

# base task -> (derived name, description, overrides applied to env)
DERIVED = [
    (
        "carla_four_lane",
        "carla_four_lane_prosafeav",
        "Multi-lane free driving with autopilot traffic; conflicts come from lane "
        "changes on both sides rather than from a single scripted leader.",
        {"num_vehicles": 40},
    ),
    (
        "carla_four_lane",
        "carla_four_lane_prosafeav_dense",
        "As above at a higher vehicle count, to separate the effect of density "
        "from the effect of the conflict geometry.",
        {"num_vehicles": 80},
    ),
    (
        "carla_lane_merge",
        "carla_lane_merge_prosafeav",
        "Forced merge: the conflict is gap acceptance on a converging path, where "
        "the car-following DRAC is undefined and the encroachment indicator governs.",
        {},
    ),
    (
        "carla_right_turn_hard",
        "carla_right_turn_prosafeav",
        "Unsignalised crossing conflict against an oncoming flow; a pure test of "
        "the two-dimensional conflict indicators.",
        {},
    ),
    (
        "carla_roundabout",
        "carla_roundabout_prosafeav",
        "Roundabout yielding and circulating conflicts under continuous curvature.",
        {},
    ),
]

BEGIN = "# >>> ProSafeAV cross-scenario tasks (generated) >>>"
END = "# <<< ProSafeAV cross-scenario tasks (generated) <<<"


def main():
    loader = yaml.YAML(typ="safe")
    with open(PATH, encoding="utf-8") as fh:
        cfg = loader.load(fh)

    out = {}
    missing = []
    for base, name, description, overrides in DERIVED:
        if base not in cfg:
            missing.append(base)
            continue
        block = copy.deepcopy(cfg[base])
        env = block.setdefault("env", {})
        env["evt"] = copy.deepcopy(EVT_BLOCK)
        env.update(overrides)

        # The EVT weight lives with the other reward scales.  Some base tasks pull
        # their reward block in through a YAML anchor, which the safe loader has
        # already expanded, so writing into it here is safe.
        scales = env.setdefault("reward", {}).setdefault("scales", {})
        scales.setdefault("evt", 3.0)
        scales["evt"] = 3.0

        for backend in ("dreamerv3",):
            if backend in block:
                block[backend]["run.log_keys_mean"] = (
                    "(log_entropy|ttc|drac|evt_|speed_|conflict_|closing_|"
                    "n_interacting|n_background|initial_gap|wpt_dis)"
                )
                block[backend]["evt.mode"] = "both"
                block[backend]["evt.imag_weight"] = 3.0

        block["__description__"] = description
        out[name] = block

    if missing:
        print(f"WARNING: base tasks absent, skipped: {missing}")
    if not out:
        print("ERROR: no tasks derived")
        sys.exit(1)

    dumper = yaml.YAML()
    dumper.default_flow_style = False
    dumper.width = 4096

    import io

    chunks = []
    for name, block in out.items():
        description = block.pop("__description__")
        buf = io.StringIO()
        dumper.dump({name: block}, buf)
        chunks.append(f"\n# {description}\n{buf.getvalue()}")
    generated = BEGIN + "\n" + "".join(chunks) + "\n" + END + "\n"

    with open(PATH, encoding="utf-8") as fh:
        src = fh.read()
    if BEGIN in src:
        head, rest = src.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        src = head + generated + tail
        action = "replaced"
    else:
        src = src.rstrip() + "\n\n" + generated
        action = "appended"
    with open(PATH, "w", encoding="utf-8") as fh:
        fh.write(src)

    with open(PATH, encoding="utf-8") as fh:
        reloaded = loader.load(fh)
    bad = [n for n in out if n not in reloaded]
    if bad:
        print("ERROR: tasks missing after write:", bad)
        sys.exit(1)
    print(f"{action} {len(out)} cross-scenario tasks; yaml parses; total = {len(reloaded)}")
    for name in out:
        print("  " + name)


if __name__ == "__main__":
    main()
