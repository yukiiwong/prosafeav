"""
Record what a scenario actually does, for inspection and for the manuscript figure.

The scenario changes in this revision -- heterogeneous IDM/MOBIL traffic, randomised
initial geometry, and injected pre-crash events -- are only claims until someone can
see the trajectories they produce.  This script drives a task with a simple
waypoint-following controller (no trained agent required, so it can run while
training occupies the rest of the card), and records per step:

  * the position, speed and lane of every vehicle, ego included,
  * the conflict indicators and their conflict partner,
  * the fitted EVT quantities,
  * which injected event is active,
  * optionally the bird's-eye-view frames the agent would receive.

Output is an .npz per episode plus a JSON summary, which ``tools/plot_scenario.py``
turns into the time-space and conflict-indicator figures.

Example::

    python tools/record_scenario.py --task carla_overtake_critical \\
        --port 2100 --episodes 3 --out logdir/scenario_recordings --save-bev
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")


def pure_pursuit_action(info, env_config, obs=None):
    """A minimal lane-following controller, expressed in the task's discrete actions.

    This is not meant to drive well.  It only has to keep the ego moving along its
    route so that the *background* traffic and the injected events can be observed;
    a trained policy would confound the two.
    """
    acc_values = list(env_config.action.discrete_acc)
    steer_values = list(env_config.action.discrete_steer)
    n_steer = len(steer_values)

    speed = float(info.get("speed_norm", 0.0))
    desired = float(env_config.reward.desired_speed)
    # Longitudinal: close the gap to the desired speed, but yield to an imminent
    # conflict.  Without this the driver ploughs into the first hard-braking lead
    # vehicle and the episode ends in a few steps, which shows that the injected
    # events bite but leaves nothing to look at.
    ttc = float(info.get("ttc", np.inf))
    # Graduated response rather than a brake/accelerate switch.  A binary rule
    # brakes on every dip below the threshold and never builds speed again: in a
    # conflict-rich scenario the ego ends up parked at 0.2 m/s, which says
    # nothing about the scenario and everything about the controller.  Braking is
    # also pointless once nearly stopped, so it is gated on actually moving.
    if np.isfinite(ttc) and ttc < 1.0 and speed > 0.5:
        acc_target = acc_values[0]
    elif np.isfinite(ttc) and ttc < 2.5 and speed > 0.5:
        acc_target = 0.0
    else:
        acc_target = np.clip(desired - speed, acc_values[0], acc_values[-1])
    acc_idx = int(np.argmin([abs(a - acc_target) for a in acc_values]))

    # Lateral: steer toward the next waypoint, using the signed offset the env
    # already computes for its lane-keeping reward.
    perp = float(info.get("speed_perpendicular", 0.0))
    steer_target = float(np.clip(-0.05 * perp, steer_values[0], steer_values[-1]))
    steer_idx = int(np.argmin([abs(s - steer_target) for s in steer_values]))
    return acc_idx * n_steer + steer_idx


def vehicle_snapshot(env):
    """Positions and speeds of every vehicle this step, ego first."""
    import math

    rows = []

    def row(actor, role):
        tf = actor.get_transform()
        vel = actor.get_velocity()
        return {
            "id": int(actor.id),
            "role": role,
            "x": float(tf.location.x),
            "y": float(tf.location.y),
            "yaw": float(math.radians(tf.rotation.yaw)),
            "speed": float(math.hypot(vel.x, vel.y)),
        }

    rows.append(row(env.ego, "ego"))
    if getattr(env, "nonego", None) is not None and env.nonego.is_alive:
        rows.append(row(env.nonego, "target"))
    for actor in getattr(env, "background_vehicles", []):
        if actor.is_alive:
            role = "stopped" if actor.id == getattr(env, "stopped_vehicle_id", None) else "background"
            rows.append(row(actor, role))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="carla_overtake_critical")
    ap.add_argument("--port", type=int, default=2100)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="logdir/scenario_recordings")
    ap.add_argument("--save-bev", action="store_true",
                    help="also store the BEV frames (larger files)")
    ap.add_argument("--bev-key", default="birdeye_wpt")
    args = ap.parse_args()

    import car_dreamer

    argv = ["--env.world.carla_port", str(args.port), "--env.display.enable", "False"]
    env, config = car_dreamer.create_task(args.task, argv)
    env_config = config.env

    os.makedirs(args.out, exist_ok=True)
    summaries = []

    for ep in range(args.episodes):
        obs = env.reset()
        # create_task returns the gym env; the CarDreamer env is the unwrapped one.
        inner = env.unwrapped if hasattr(env, "unwrapped") else env

        traj, indicators, frames = [], [], []
        info = {"speed_norm": 0.0, "speed_perpendicular": 0.0}
        done = False
        step = 0
        while not done and step < args.max_steps:
            action = pure_pursuit_action(info, env_config)
            obs, reward, done, info = env.step(action)

            traj.append(vehicle_snapshot(inner))
            active = inner.events.active_overrides if hasattr(inner, "events") else {}
            indicators.append({
                "step": step,
                "reward": float(reward),
                "ttc": float(info.get("ttc", np.inf)),
                "drac": float(info.get("drac", 0.0)),
                "evt_severity": float(info.get("evt_severity", 0.0)),
                "evt_tail_prob": float(info.get("evt_tail_prob", 0.0)),
                "r_evt": float(info.get("r_evt", 0.0)),
                "conflict_partner": int(info.get("conflict_partner", -1)),
                "conflict_role": str(info.get("conflict_role", "none")),
                "n_interacting": int(info.get("n_interacting", 0)),
                "speed": float(info.get("speed_norm", 0.0)),
                "active_event": next(iter(e.kind for e in active.values()), "none"),
            })
            if args.save_bev and args.bev_key in obs:
                frames.append(np.asarray(obs[args.bev_key], dtype=np.uint8))
            step += 1

        events = inner.events.summary() if hasattr(inner, "events") else {}
        profile = getattr(inner, "traffic_profile", {})
        path = os.path.join(args.out, f"{args.task}_ep{ep}.npz")
        payload = {
            "trajectory": np.array(json.dumps(traj)),
            "indicators": np.array(json.dumps(indicators)),
            "events": np.array(json.dumps(events)),
        }
        if frames:
            payload["bev"] = np.stack(frames)
        np.savez_compressed(path, **payload)

        conflicts = [i for i in indicators if np.isfinite(i["ttc"])]
        summary = {
            "episode": ep,
            "task": args.task,
            "steps": step,
            "file": path,
            "n_vehicles": len(traj[0]) if traj else 0,
            "initial_gap": float(getattr(inner, "initial_gap", 0.0)),
            "ego_start_lane_offset": int(getattr(inner, "ego_start_lane_offset", 0)),
            "events_scheduled": events.get("n_events", 0),
            "events_fired": events.get("n_fired", 0),
            "event_kinds": [e["kind"] for e in events.get("events", [])],
            "steps_with_conflict": len(conflicts),
            "min_ttc": float(min((i["ttc"] for i in conflicts), default=float("inf"))),
            "max_drac": float(max((i["drac"] for i in indicators), default=0.0)),
            "background_density": profile.get("density_veh_per_km_lane"),
        }
        summaries.append(summary)
        print(
            f"episode {ep}: {step} steps, {summary['n_vehicles']} vehicles, "
            f"gap {summary['initial_gap']:.1f} m, events {summary['events_fired']}/"
            f"{summary['events_scheduled']} {summary['event_kinds']}, "
            f"conflict steps {summary['steps_with_conflict']}, "
            f"min TTC {summary['min_ttc']:.2f}s, max DRAC {summary['max_drac']:.2f}"
        )

    with open(os.path.join(args.out, f"{args.task}_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nwrote {len(summaries)} episodes to {args.out}")
    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
