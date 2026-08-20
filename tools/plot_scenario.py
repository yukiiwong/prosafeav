"""
Turn a recorded episode into the scenario figures for the manuscript.

Three panels, each answering one of the reviewer's questions about the scenario:

  1. Time-space diagram -- every vehicle's longitudinal position against time, ego
     highlighted.  Shows the traffic is multi-vehicle and heterogeneous, and marks
     where an injected pre-crash event is active.  (Reviewer 1, comments 5 and 6.)
  2. Lateral trajectory -- the same episode in plan view, which is where the
     overtaking and cut-in geometry is legible.
  3. Conflict indicators -- TTC and DRAC against time with the fitted POT
     thresholds, showing that the exceedances the EVT model is fitted on actually
     occur, and when.  (Reviewer 2, comments 2 and 4.)

Usage::

    python tools/plot_scenario.py logdir/scenario_recordings/<task>_ep2.npz
    python tools/plot_scenario.py --auto logdir/scenario_recordings   # pick the
                                                                     # best episode

The numbers behind each panel are written next to the figure as CSV, so the
figure is reproducible without re-running CARLA.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evtstyle import FIGSIZE, FIGSIZE_WIDE, dash, gray, mark, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# Roles are a fixed, known vocabulary, so their encoding is fixed too: the ego is
# the only series the reader must never lose, so it is black and solid.
ROLE_STYLE = {
    "ego":        dict(color="black", ls="-",  lw=3.0, zorder=5, label="Ego vehicle"),
    "target":     dict(color="0.35",  ls="--", lw=2.5, zorder=4, label="Overtaken vehicle"),
    "stopped":    dict(color="0.25",  ls="-.", lw=2.5, zorder=4, label="Stopped vehicle"),
    "background": dict(color="0.70",  ls=":",  lw=1.8, zorder=2, label="Background traffic"),
}


def load(path):
    d = np.load(path, allow_pickle=True)
    traj = json.loads(str(d["trajectory"]))
    ind = json.loads(str(d["indicators"]))
    events = json.loads(str(d["events"]))
    bev = d["bev"] if "bev" in d.files else None
    return traj, ind, events, bev


def by_vehicle(traj):
    """Reshape the per-step snapshots into one series per vehicle id."""
    out = {}
    for t, rows in enumerate(traj):
        for r in rows:
            v = out.setdefault(r["id"], {"role": r["role"], "t": [], "x": [], "y": [], "speed": []})
            v["t"].append(t)
            v["x"].append(r["x"])
            v["y"].append(r["y"])
            v["speed"].append(r["speed"])
            # A vehicle keeps the most specific role it is ever seen with.
            if r["role"] != "background":
                v["role"] = r["role"]
    return out


def event_spans(ind):
    """Contiguous step ranges during which some injected event was active."""
    spans, start, kind = [], None, None
    for i, row in enumerate(ind):
        active = row.get("active_event", "none")
        if active != "none" and start is None:
            start, kind = i, active
        elif active == "none" and start is not None:
            spans.append((start, i, kind))
            start, kind = None, None
    if start is not None:
        spans.append((start, len(ind), kind))
    return spans


def shade_events(ax, spans, dt=0.1):
    for s, e, _ in spans:
        ax.axvspan(s * dt, e * dt, color="0.85", zorder=0)


def plot_episode(path, outdir, dt=0.1):
    traj, ind, events, bev = load(path)
    vehicles = by_vehicle(traj)
    spans = event_spans(ind)
    stem = os.path.splitext(os.path.basename(path))[0]
    t = np.arange(len(ind)) * dt

    # ---------------------------------------------------------------- panel 1+2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=False)
    fig.subplots_adjust(wspace=0.28)

    seen = set()
    for vid, v in vehicles.items():
        style = dict(ROLE_STYLE.get(v["role"], ROLE_STYLE["background"]))
        label = style.pop("label")
        ax1.plot(np.array(v["t"]) * dt, v["y"], label=label if label not in seen else None, **style)
        seen.add(label)
        style2 = dict(ROLE_STYLE.get(v["role"], ROLE_STYLE["background"]))
        style2.pop("label")
        ax2.plot(v["x"], v["y"], **style2)

    shade_events(ax1, spans, dt)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Longitudinal position (m)")
    ax1.invert_yaxis()          # the road runs along -y, so ahead is downward
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Lateral position (m)")
    ax2.set_ylabel("Longitudinal position (m)")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=len(labels), frameon=False)
    save(fig, f"{stem}_trajectories", path)
    plt.close(fig)

    # ---------------------------------------------------------------- panel 3
    ttc = np.array([r["ttc"] for r in ind], dtype=float)
    drac = np.array([r["drac"] for r in ind], dtype=float)
    ttc_plot = np.where(np.isfinite(ttc), ttc, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=False)
    fig.subplots_adjust(wspace=0.28)

    ax1.plot(t, ttc_plot, color="black", ls="-", lw=2.5, label="TTC")
    shade_events(ax1, spans, dt)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Time-to-collision (s)")
    ax1.grid(alpha=0.3)

    ax2.plot(t, drac, color="black", ls="-", lw=2.5, label="DRAC")
    shade_events(ax2, spans, dt)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("DRAC (m/s$^2$)")
    ax2.grid(alpha=0.3)

    save(fig, f"{stem}_indicators", path)
    plt.close(fig)

    # ---------------------------------------------------------------- BEV strip
    if bev is not None and len(bev) > 0:
        # Sample frames around the first event if there is one, otherwise evenly.
        if spans:
            centre = spans[0][0]
            raw = [centre - 20, centre - 8, centre, centre + 8, centre + 20]
            idx = sorted({int(np.clip(i, 0, len(bev) - 1)) for i in raw})
            # Clamping alone collapses frames onto each other when the event fires
            # near the start, which produced a strip with two panels both labelled
            # t = 0.0 s.  Top up from an even spread instead.
            if len(idx) < 5:
                extra = np.linspace(0, len(bev) - 1, 5).astype(int)
                idx = sorted(set(idx) | set(int(i) for i in extra))[:5]
        else:
            idx = sorted(set(np.linspace(0, len(bev) - 1, 5).astype(int).tolist()))
        fig, axes = plt.subplots(1, len(idx), figsize=(FIGSIZE[0], FIGSIZE[0] / len(idx)))
        for ax, i in zip(np.atleast_1d(axes), idx):
            ax.imshow(bev[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"t = {i * dt:.1f} s")
        save(fig, f"{stem}_bev", path)
        plt.close(fig)

    # ---------------------------------------------------------------- numbers
    csv = os.path.join(os.path.dirname(path), "figures", f"{stem}_indicators.csv")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    with open(csv, "w") as fh:
        fh.write("time_s,ttc_s,drac_mps2,evt_severity,speed_mps,n_interacting,active_event\n")
        for i, r in enumerate(ind):
            fh.write(f"{i * dt:.2f},{r['ttc']:.4f},{r['drac']:.4f},"
                     f"{r['evt_severity']:.6f},{r['speed']:.4f},"
                     f"{r['n_interacting']},{r.get('active_event', 'none')}\n")

    return {
        "file": path,
        "steps": len(ind),
        "vehicles": len(vehicles),
        "event_spans": [(s, e, k) for s, e, k in spans],
        "scheduled": [e["kind"] for e in events.get("events", [])],
        "min_ttc": float(np.nanmin(ttc_plot)) if np.isfinite(ttc_plot).any() else float("inf"),
        "max_drac": float(drac.max()) if drac.size else 0.0,
        "csv": csv,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="an .npz recording")
    ap.add_argument("--auto", metavar="DIR",
                    help="pick the most illustrative episode in DIR: the longest one "
                         "in which an injected event actually fired")
    ap.add_argument("--dt", type=float, default=0.1)
    args = ap.parse_args()

    if args.auto:
        best, best_key = None, (-1, -1)
        for p in sorted(glob.glob(os.path.join(args.auto, "*_ep*.npz"))):
            traj, ind, events, _ = load(p)
            fired = events.get("n_fired", 0)
            key = (fired, len(ind))
            if key > best_key:
                best, best_key = p, key
        if best is None:
            print(f"no recordings under {args.auto}")
            sys.exit(1)
        print(f"selected {best} (events fired {best_key[0]}, {best_key[1]} steps)")
        path = best
    elif args.path:
        path = args.path
    else:
        ap.error("give a recording path or --auto DIR")

    info = plot_episode(path, os.path.dirname(path), dt=args.dt)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
