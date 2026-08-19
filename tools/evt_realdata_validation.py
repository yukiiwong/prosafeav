"""
Validate the ProSafeAV conflict/EVT pipeline on a real-world trajectory dataset.

Reviewer 1 asks whether a framework trained only in simulation says anything
about real traffic.  The part of ProSafeAV that makes a falsifiable claim about
the real world is the *risk model*: the bivariate POT model of (TTC, DRAC) and
the crash probability it extrapolates.  That model can be fitted to real
trajectories directly, with no simulator and no policy involved, and the fitted
tail parameters compared against the ones obtained in CARLA.

This script

  1. streams a real trajectory dataset (TGSIM Foggy Bottom by default; pNEUMA and
     any CSV with the same columns work through ``--columns``),
  2. computes TTC and DRAC with *the same definitions* used in the simulator --
     the vectorised implementation here is checked against
     ``car_dreamer.toolkit.carla_manager.conflict`` on random inputs before it is
     used, so the two cannot silently diverge,
  3. fits the same GPD margins and logistic extreme-value copula, and
  4. writes a JSON report with thresholds, shape/scale parameters, the dependence
     parameter, return levels and the extrapolated crash probability, plus the
     observed time-headway distribution used to justify the IDM parameters of the
     background traffic.

Example::

    python tools/evt_realdata_validation.py \\
        --csv /home/yukai/datasets/trajectory_external/TGSIM_Foggy_Bottom/TGSIM_Foggy_Bottom_Trajectories.csv \\
        --out logdir/evt_real_tgsim.json --max-frames 4000

    # then compare against a model fitted during training
    python tools/evt_realdata_validation.py --compare logdir/<run>/evt_model.json \\
        --out logdir/evt_real_tgsim.json
"""

import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from car_dreamer.evt_module import CopulaEVTModel
from car_dreamer.toolkit.carla_manager.conflict import (
    MAX_DECELERATION,
    VehicleState,
    longitudinal_indicators,
    planar_indicators,
)

DEFAULT_COLUMNS = {
    "id": "id",
    "time": "time",
    "x": "xloc_kf",
    "y": "yloc_kf",
    "vx": "speed_kf_x",
    "vy": "speed_kf_y",
    "length": "length_smoothed",
    "width": "width_smoothed",
}


# --------------------------------------------------------------------------- #
# Vectorised indicators (must agree with conflict.py)
# --------------------------------------------------------------------------- #
def pairwise_indicators(x, y, vx, vy, length, width, max_distance=50.0,
                        min_speed=0.0, heading_tol_deg=180.0, lateral_bound=np.inf):
    """All-pairs TTC and DRAC within one frame.

    Returns ``(ttc, drac)`` arrays of shape ``[n]``: for every vehicle, the most
    critical values against every other vehicle that survives the interaction
    filter.  The combination rule is the ``"max"`` mode of
    :func:`conflict.compute_conflict`: the smaller TTC and the larger DRAC of the
    car-following and the planar (encroachment) variants.

    The interaction filter selects the pairs for which the surrogate measures are
    *defined*, and it matters a great deal on real data.  Urban datasets are
    dominated by queued vehicles standing bumper to bumper at a red light and by
    crossing or opposing movements; neither is a car-following or lane-change
    conflict, but both produce spuriously extreme indicator values that swamp the
    genuine tail.  Three portable criteria are applied, none of which depends on a
    dataset-specific lane encoding:

    :param min_speed: both vehicles must exceed this speed (m/s), which removes
        standing queues.
    :param heading_tol_deg: the two headings must agree to within this angle,
        which removes crossing and opposing traffic.
    :param lateral_bound: the lateral offset in the ego frame must be within this
        many metres, which removes parallel traffic several lanes away.

    Passing the defaults disables the filter and reproduces the unfiltered
    all-pairs computation, which is what the equivalence check against
    ``conflict.py`` uses.
    """
    n = len(x)
    if n < 2:
        return np.full(n, np.inf), np.zeros(n)

    pos = np.stack([x, y], axis=1)
    vel = np.stack([vx, vy], axis=1)
    speed = np.hypot(vx, vy)
    # Heading from the velocity vector; stationary vehicles keep the +x axis.
    heading = np.zeros_like(pos)
    moving = speed > 0.1
    heading[moving] = vel[moving] / speed[moving, None]
    heading[~moving] = np.array([1.0, 0.0])

    dp = pos[None, :, :] - pos[:, None, :]  # [i, j, 2] = p_j - p_i
    dv = vel[None, :, :] - vel[:, None, :]
    dist = np.linalg.norm(dp, axis=2)

    # -- longitudinal (car following) ------------------------------------- #
    s = np.einsum("ijk,ik->ij", dp, heading)  # projection of the offset on ego heading
    v_own = np.einsum("ik,ik->i", vel, heading)[:, None]
    v_oth = np.einsum("jk,ik->ij", vel, heading)
    half_sum = 0.5 * (length[:, None] + length[None, :])
    lead = s > 0
    gap_lon = np.where(lead, s, -s) - half_sum
    dv_lon = np.where(lead, v_own - v_oth, v_oth - v_own)

    closing_lon = dv_lon > 0
    safe_gap = np.maximum(gap_lon, 1e-2)
    ttc_lon = np.where(closing_lon & (gap_lon > 0), safe_gap / np.maximum(dv_lon, 1e-9), np.inf)
    ttc_lon = np.where(closing_lon & (gap_lon <= 0), 0.0, ttc_lon)
    drac_lon = np.where(
        closing_lon & (gap_lon > 0),
        np.minimum(dv_lon**2 / (2.0 * safe_gap), MAX_DECELERATION),
        0.0,
    )
    drac_lon = np.where(closing_lon & (gap_lon <= 0), MAX_DECELERATION, drac_lon)

    # -- planar (encroachment), elliptical exclusion zone ------------------ #
    normal = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
    semi_l = 0.5 * (length[:, None] + length[None, :])
    semi_w = 0.5 * (width[:, None] + width[None, :])
    q = np.stack([
        np.einsum("ijk,ik->ij", dp, heading) / semi_l,
        np.einsum("ijk,ik->ij", dp, normal) / semi_w,
    ], axis=2)
    w = np.stack([
        np.einsum("ijk,ik->ij", dv, heading) / semi_l,
        np.einsum("ijk,ik->ij", dv, normal) / semi_w,
    ], axis=2)

    q_norm = np.linalg.norm(q, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        closing = -np.einsum("ijk,ijk->ij", dv, dp) / np.where(dist > 1e-6, dist, np.inf)
        gap_pl = dist * (q_norm - 1.0) / np.where(q_norm > 1e-9, q_norm, np.inf)
    inside = q_norm <= 1.0
    gap_pl = np.where(inside, 0.0, gap_pl)

    a = np.einsum("ijk,ijk->ij", w, w)
    b = 2.0 * np.einsum("ijk,ijk->ij", q, w)
    c = np.einsum("ijk,ijk->ij", q, q) - 1.0
    disc = b**2 - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(np.maximum(disc, 0.0))
        safe_a = np.where(a > 1e-12, a, np.inf)
        t1 = (-b - root) / (2 * safe_a)
        t2 = (-b + root) / (2 * safe_a)
    valid = (disc >= 0) & (a > 1e-12)
    t1 = np.where(valid & (t1 > 0), t1, np.inf)
    t2 = np.where(valid & (t2 > 0), t2, np.inf)
    ttc_pl = np.where(inside, 0.0, np.minimum(t1, t2))

    safe_gap_pl = np.maximum(gap_pl, 1e-2)
    drac_pl = np.where(
        (closing > 0) & ~inside,
        np.minimum(closing**2 / (2.0 * safe_gap_pl), MAX_DECELERATION),
        0.0,
    )
    drac_pl = np.where(inside, MAX_DECELERATION, drac_pl)

    # -- aggregate --------------------------------------------------------- #
    ttc = np.minimum(ttc_lon, ttc_pl)
    drac = np.maximum(drac_lon, drac_pl)

    excluded = dist > max_distance
    if min_speed > 0:
        slow = speed < min_speed
        excluded |= slow[:, None] | slow[None, :]
    if heading_tol_deg < 180.0:
        cos_tol = math.cos(math.radians(heading_tol_deg))
        excluded |= (heading @ heading.T) < cos_tol
    if np.isfinite(lateral_bound):
        lateral = np.abs(np.einsum("ijk,ik->ij", dp, np.stack(
            [-heading[:, 1], heading[:, 0]], axis=1)))
        excluded |= lateral > lateral_bound
    np.fill_diagonal(excluded, True)

    ttc = np.where(excluded, np.inf, ttc)
    drac = np.where(excluded, 0.0, drac)
    return ttc.min(axis=1), drac.max(axis=1)


def verify_against_reference(seed=0, n_trials=200):
    """Assert the vectorised implementation matches the per-pair reference."""
    rng = np.random.default_rng(seed)
    worst_ttc, worst_drac = 0.0, 0.0
    for _ in range(n_trials):
        n = 3
        x = rng.uniform(-40, 40, n)
        y = rng.uniform(-40, 40, n)
        sp = rng.uniform(0.5, 25.0, n)
        th = rng.uniform(-math.pi, math.pi, n)
        vx, vy = sp * np.cos(th), sp * np.sin(th)
        length = rng.uniform(3.5, 12.0, n)
        width = rng.uniform(1.6, 2.6, n)

        vec_ttc, vec_drac = pairwise_indicators(x, y, vx, vy, length, width)

        states = [
            VehicleState(x[i], y[i], vx[i], vy[i], math.atan2(vy[i], vx[i]),
                         length[i], width[i], id=i)
            for i in range(n)
        ]
        for i in range(n):
            ref_ttc, ref_drac = np.inf, 0.0
            for j in range(n):
                if i == j:
                    continue
                if np.hypot(x[j] - x[i], y[j] - y[i]) > 50.0:
                    continue
                lt, ld, _ = longitudinal_indicators(states[i], states[j])
                pt, pd, _, _ = planar_indicators(states[i], states[j])
                ref_ttc = min(ref_ttc, lt, pt)
                ref_drac = max(ref_drac, ld, pd)
            if math.isinf(ref_ttc) and math.isinf(vec_ttc[i]):
                d_ttc = 0.0
            else:
                d_ttc = abs(ref_ttc - vec_ttc[i])
            worst_ttc = max(worst_ttc, d_ttc)
            worst_drac = max(worst_drac, abs(ref_drac - vec_drac[i]))
    return worst_ttc, worst_drac


# --------------------------------------------------------------------------- #
# Dataset streaming
# --------------------------------------------------------------------------- #
def stream_frames(csv_path, columns, max_frames=None, min_vehicles=2, stride=1):
    """Yield ``(t, arrays)`` per timestamp.  Requires the file to be time sorted
    within a reasonable window; rows are grouped by their time column."""
    import pandas as pd

    usecols = list(columns.values())
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=500_000)
    buffer = None
    frames = 0
    for chunk in reader:
        chunk = chunk.rename(columns={v: k for k, v in columns.items()})
        buffer = chunk if buffer is None else pd.concat([buffer, chunk], ignore_index=True)
        times = np.sort(buffer["time"].unique())
        if len(times) < 2:
            continue
        # Everything but the last (possibly incomplete) timestamp is safe to emit.
        complete, last = times[:-1], times[-1]
        for i, t in enumerate(complete):
            if i % stride:
                continue
            grp = buffer[buffer["time"] == t]
            if len(grp) >= min_vehicles:
                yield t, grp
                frames += 1
                if max_frames and frames >= max_frames:
                    return
        buffer = buffer[buffer["time"] == last].copy()


def headway_statistics(gaps, speeds):
    """Observed time headway ``gap / speed`` for the moving vehicles."""
    ok = (speeds > 1.0) & np.isfinite(gaps) & (gaps > 0)
    if not ok.any():
        return {}
    th = gaps[ok] / speeds[ok]
    th = th[th < 10.0]
    if not th.size:
        return {}
    return {
        "n": int(th.size),
        "mean": float(th.mean()),
        "std": float(th.std()),
        "p05": float(np.percentile(th, 5)),
        "p25": float(np.percentile(th, 25)),
        "p50": float(np.percentile(th, 50)),
        "p75": float(np.percentile(th, 75)),
        "p95": float(np.percentile(th, 95)),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="/home/yukai/datasets/trajectory_external/"
                                     "TGSIM_Foggy_Bottom/TGSIM_Foggy_Bottom_Trajectories.csv")
    ap.add_argument("--dataset-name", default="TGSIM Foggy Bottom")
    ap.add_argument("--out", default="logdir/evt_real_validation.json")
    ap.add_argument("--max-frames", type=int, default=5000,
                    help="0 processes the whole file")
    ap.add_argument("--stride", type=int, default=1, help="use every k-th frame")
    ap.add_argument("--max-distance", type=float, default=50.0)
    ap.add_argument("--min-speed", type=float, default=2.0,
                    help="both vehicles of a pair must exceed this speed (m/s); "
                         "removes standing queues, which dominate urban datasets")
    ap.add_argument("--heading-tol", type=float, default=30.0,
                    help="maximum heading difference (deg) for a pair to count as "
                         "an interaction; removes crossing and opposing movements")
    ap.add_argument("--lateral-bound", type=float, default=7.0,
                    help="maximum lateral offset (m) in the ego frame; keeps the "
                         "own and the immediately adjacent lanes")
    ap.add_argument("--copula", default="logistic", choices=["logistic", "frank"])
    ap.add_argument("--threshold-method", default="stability",
                    choices=["stability", "mrl", "quantile"])
    ap.add_argument("--compare", default=None,
                    help="JSON produced by CopulaEVTModel.save() during training")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    if not args.skip_verify:
        print("Verifying the vectorised indicators against the simulator definitions ...")
        d_ttc, d_drac = verify_against_reference()
        print(f"  max |dTTC| = {d_ttc:.3e} s, max |dDRAC| = {d_drac:.3e} m/s^2")
        if d_ttc > 1e-6 or d_drac > 1e-6:
            print("ERROR: the vectorised indicators disagree with conflict.py; aborting.")
            sys.exit(1)
        print("  definitions agree.\n")

    model = CopulaEVTModel(
        copula=args.copula,
        threshold_method=args.threshold_method,
        buffer_size=5_000_000,
        min_sample=1000,
        min_exceedances=100,
    )

    print(f"Streaming {args.csv} ...")
    t0 = time.time()
    n_frames, n_obs = 0, 0
    all_gaps, all_speeds = [], []
    for t, grp in stream_frames(
        args.csv, DEFAULT_COLUMNS,
        max_frames=(args.max_frames or None), stride=args.stride,
    ):
        ttc, drac = pairwise_indicators(
            grp["x"].to_numpy(float), grp["y"].to_numpy(float),
            grp["vx"].to_numpy(float), grp["vy"].to_numpy(float),
            grp["length"].to_numpy(float), grp["width"].to_numpy(float),
            max_distance=args.max_distance,
            min_speed=args.min_speed,
            heading_tol_deg=args.heading_tol,
            lateral_bound=args.lateral_bound,
        )
        speeds = np.hypot(grp["vx"].to_numpy(float), grp["vy"].to_numpy(float))
        finite = np.isfinite(ttc)
        all_gaps.append(ttc[finite] * np.maximum(speeds[finite], 1e-3))
        all_speeds.append(speeds[finite])
        for a, b in zip(ttc, drac):
            model.add_sample(a, b)
            n_obs += 1
        n_frames += 1
        if n_frames % 1000 == 0:
            print(f"  {n_frames} frames, {len(model.buffer)} conflict samples, "
                  f"{time.time() - t0:.0f}s")

    print(f"\n{n_frames} frames processed, {n_obs} vehicle-steps, "
          f"{len(model.buffer)} retained conflict samples "
          f"({time.time() - t0:.0f}s)\n")

    if not model.update_model(verbose=True):
        print("ERROR: the EVT model could not be fitted on this sample.")
        sys.exit(1)

    report = {
        "dataset": args.dataset_name,
        "source_file": args.csv,
        "n_frames": n_frames,
        "n_vehicle_steps": n_obs,
        "interaction_filter": {
            "max_distance_m": args.max_distance,
            "min_speed_mps": args.min_speed,
            "heading_tol_deg": args.heading_tol,
            "lateral_bound_m": args.lateral_bound,
        },
        "n_conflict_samples": len(model.buffer),
        "evt": model.summary(),
        "return_levels": {
            "ttc_1e-3": float(model.margin_ttc.return_level(1e-3)),
            "ttc_1e-4": float(model.margin_ttc.return_level(1e-4)),
            "drac_1e-3": float(model.margin_drac.return_level(1e-3)),
            "drac_1e-4": float(model.margin_drac.return_level(1e-4)),
        },
        # With surrogate measures that are physically bounded (TTC cannot fall
        # below 0, DRAC cannot exceed the tyre-road limit), the fitted GPD has a
        # finite upper endpoint sitting on that bound, so the tail probability
        # evaluated *exactly* at the boundary degenerates to zero.  The
        # informative quantity is the probability of reaching a near-crash
        # boundary, so a small sweep is reported instead of a single number.
        "crash_probability_sweep": {
            f"ttc<={t}s,drac>={d}": model.joint_exceedance_prob(t, d)
            for t, d in [(0.0, 8.5), (0.2, 8.0), (0.5, 7.0), (0.5, 6.0), (1.0, 5.0)]
        },
        "observed_contact_rate": (
            model.n_contact / max(model.n_contact + model.n_saturated + len(model.buffer), 1)
        ),
        "n_contact_excluded": model.n_contact,
        "n_saturated_excluded": model.n_saturated,
        "headway": headway_statistics(
            np.concatenate(all_gaps) if all_gaps else np.array([]),
            np.concatenate(all_speeds) if all_speeds else np.array([]),
        ),
    }

    if args.compare:
        with open(args.compare) as fh:
            sim = json.load(fh)
        real, comp = report["evt"], {}
        for side, key in (("ttc", "margin_ttc"), ("drac", "margin_drac")):
            if sim.get(key) and real.get(side):
                for p in ("u", "xi", "sigma"):
                    comp[f"{side}_{p}"] = {
                        "real": real[side][p],
                        "sim": sim[key][p],
                        "abs_diff": abs(real[side][p] - sim[key][p]),
                    }
        sim_alpha = (sim.get("copula") or {}).get("alpha")
        if sim_alpha is not None and real.get("alpha") is not None:
            comp["alpha"] = {"real": real["alpha"], "sim": sim_alpha,
                             "abs_diff": abs(real["alpha"] - sim_alpha)}
        report["comparison_with_simulation"] = comp
        report["simulation_source"] = args.compare

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 66)
    print(f"Dataset            : {report['dataset']}")
    ttc_m, drac_m = report["evt"]["ttc"], report["evt"]["drac"]
    print(f"TTC   threshold    : {-ttc_m['u']:.3f} s   "
          f"(xi = {ttc_m['xi']:+.3f}, sigma = {ttc_m['sigma']:.3f}, "
          f"n_exc = {ttc_m['n_exc']})")
    print(f"DRAC  threshold    : {drac_m['u']:.3f} m/s^2 "
          f"(xi = {drac_m['xi']:+.3f}, sigma = {drac_m['sigma']:.3f}, "
          f"n_exc = {drac_m['n_exc']})")
    print(f"Dependence alpha   : {report['evt']['alpha']:.4f}  "
          f"(0 = complete dependence, 1 = independence)")
    print(f"Joint exc. rate    : {report['evt']['zeta_joint']:.5f}")
    print("P(near-crash)      :")
    for label, value in report["crash_probability_sweep"].items():
        print(f"    {label:24s} {value:.3e} per interacting vehicle-step")
    print(f"Observed contacts  : {report['n_contact_excluded']} "
          f"({report['observed_contact_rate']:.3e} of interactions), "
          f"{report['n_saturated_excluded']} saturated DRAC excluded")
    if report["headway"]:
        h = report["headway"]
        print(f"Time headway (s)   : median {h['p50']:.2f}, "
              f"IQR [{h['p25']:.2f}, {h['p75']:.2f}], p05 {h['p05']:.2f}")
    print(f"\nreport written to {args.out}")
    print("=" * 66)


if __name__ == "__main__":
    main()
