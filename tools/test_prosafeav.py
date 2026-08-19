"""Self-contained sanity checks for the ProSafeAV EVT / conflict / traffic modules.

Run with:  python -m pytest tools/test_prosafeav.py -q
or simply: python tools/test_prosafeav.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")

from car_dreamer.evt_module import CopulaEVTModel, GPDMargin, LogisticEVCopula, select_threshold
from car_dreamer.toolkit.carla_manager.conflict import (
    VehicleState,
    compute_conflict,
    longitudinal_indicators,
    planar_indicators,
)
from car_dreamer.toolkit.traffic_models import (
    IDMParams,
    IDMMobilController,
    MOBILParams,
    idm_acceleration,
    mobil_decision,
)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


# --------------------------------------------------------------------------- #
def test_logistic_copula():
    print("\n[logistic extreme-value copula]")
    cop = LogisticEVCopula(alpha=0.5)

    # Boundary conditions of a copula.
    check("C(u,1) = u", abs(cop.cdf(0.4, 1.0 - 1e-12) - 0.4) < 1e-3,
          f"got {cop.cdf(0.4, 1.0 - 1e-12)}")
    check("C(1,v) = v", abs(cop.cdf(1.0 - 1e-12, 0.7) - 0.7) < 1e-3,
          f"got {cop.cdf(1.0 - 1e-12, 0.7)}")
    check("alpha=1 is independence",
          abs(LogisticEVCopula(alpha=0.999).cdf(0.5, 0.5) - 0.25) < 1e-2,
          f"got {LogisticEVCopula(alpha=0.999).cdf(0.5, 0.5)}")

    # Monotonicity in both arguments.
    grid = np.linspace(0.05, 0.95, 19)
    mono = all(cop.cdf(grid[i], 0.6) <= cop.cdf(grid[i + 1], 0.6) + 1e-12 for i in range(len(grid) - 1))
    check("C is non-decreasing in u1", mono)

    # The density must integrate to one over the unit square.
    n = 400
    u = (np.arange(n) + 0.5) / n
    U1, U2 = np.meshgrid(u, u)
    mass = float(np.exp(cop.logpdf(U1.ravel(), U2.ravel())).sum() / (n * n))
    check("density integrates to 1", abs(mass - 1.0) < 0.02, f"mass={mass:.4f}")

    # Parameter recovery: sample from a Gumbel copula and refit.
    rng = np.random.default_rng(0)
    true_alpha = 0.4
    # Sample via the Marshall-Olkin / stable mixture representation.
    theta = 1.0 / true_alpha
    # positive stable(1/theta) variate through Chambers-Mallows-Stuck
    w = rng.uniform(0, math.pi, 40000)
    e = rng.exponential(1.0, 40000)
    a = 1.0 / theta
    s = (np.sin(a * w) / np.sin(w) ** a) * (np.sin((1 - a) * w) / e) ** ((1 - a) / a)
    e1, e2 = rng.exponential(1.0, 40000), rng.exponential(1.0, 40000)
    u1 = np.exp(-((e1 / s) ** a))
    u2 = np.exp(-((e2 / s) ** a))
    fit = LogisticEVCopula().fit(u1, u2)
    check("alpha recovered from samples", abs(fit.alpha - true_alpha) < 0.06,
          f"true={true_alpha} fitted={fit.alpha:.4f}")


def test_gpd_margin():
    print("\n[GPD margin / POT]")
    rng = np.random.default_rng(1)
    # Body: normal; tail: exponential excesses above 3 (xi = 0).
    body = rng.normal(0, 1, 20000)
    data = np.concatenate([body, 3.0 + rng.exponential(0.8, 3000)])

    m = GPDMargin()
    ok = m.fit(data, threshold=None, threshold_method="stability", min_exceedances=100)
    check("margin fits", ok)
    if not ok:
        return
    check("cdf is monotone", m.cdf(np.array([1.0])) < m.cdf(np.array([5.0])))
    check("cdf in [0,1]", 0.0 <= float(m.cdf(np.array([4.0]))) <= 1.0)
    check("sf decreasing", float(m.sf(np.array([4.0]))) > float(m.sf(np.array([8.0]))))
    check("xi near 0 for exponential tail", abs(m.xi) < 0.25, f"xi={m.xi:.3f}")
    rl = m.return_level(1e-4)
    check("return level above threshold", rl > m.u, f"u={m.u:.2f} rl={rl:.2f}")

    u, diag = select_threshold(data, method="mrl", min_exceedances=100)
    check("mrl threshold is finite", np.isfinite(u), f"u={u}")


def test_evt_model_direction():
    print("\n[CopulaEVTModel risk direction and scale]")
    rng = np.random.default_rng(2)
    model = CopulaEVTModel(min_sample=300, min_exceedances=60)

    # Synthetic conflicts: dangerous states have small TTC and large DRAC.
    for _ in range(6000):
        ttc = float(np.clip(rng.gamma(3.0, 1.2), 0.15, 9.9))
        drac = float(np.clip(6.0 / ttc + rng.normal(0, 0.4), 0.0, 14.0))
        model.add_sample(ttc, drac)
    ok = model.update_model(verbose=False)
    check("model fits", ok)
    if not ok:
        return

    # Probe points are placed relative to the *fitted* thresholds; hard-coded
    # values would silently fall outside the joint tail whenever the threshold
    # selection moves.
    ttc_thr = -model.margin_ttc.u  # the threshold expressed as a TTC in seconds
    drac_thr = model.margin_drac.u
    print(f"    fitted thresholds: TTC <= {ttc_thr:.2f} s, DRAC >= {drac_thr:.2f} m/s^2, "
          f"alpha = {model.copula.alpha:.3f}, zeta_joint = {model.zeta_joint:.4f}")

    mild = model.severity(ttc_thr + 3.0, drac_thr - 3.0)
    medium = model.severity(ttc_thr * 0.7, drac_thr * 1.05)
    severe = model.severity(ttc_thr * 0.2, drac_thr * 1.35)
    # Severity is zero outside the joint tail and grows continuously inside it.
    check("severity is non-decreasing in danger", mild <= medium <= severe,
          f"{mild:.4f} <= {medium:.4f} <= {severe:.4f}")
    check("severity is positive in the joint tail", severe > 0.0, f"{severe:.4f}")
    check("severity is zero outside the joint tail", mild == 0.0, f"{mild:.4f}")
    check("severity bounded in [0,1]", 0.0 <= mild and severe <= 1.0)

    # The parameter vector handed to the world model must round-trip.
    pv = model.param_vector()
    check("param vector has the declared width", pv.shape == (model.PARAM_DIM,), f"{pv.shape}")
    check("param vector marks the model as fitted", pv[0] == 1.0, f"{pv[0]}")
    check("param vector is finite", bool(np.all(np.isfinite(pv))), f"{pv}")

    p_mild = model.joint_exceedance_prob(ttc_thr + 3.0, drac_thr - 3.0)
    p_severe = model.joint_exceedance_prob(ttc_thr * 0.2, drac_thr * 1.35)
    check("tail probability decreases with danger", p_severe < p_mild,
          f"{p_severe:.3e} vs {p_mild:.3e}")

    pc = model.crash_probability()
    check("crash probability in (0,1)", 0.0 <= pc < 1.0, f"P_crash={pc:.3e}")

    r_mild = model.get_evt_reward(ttc_thr + 3.0, drac_thr - 3.0, weight=3.0)
    r_severe = model.get_evt_reward(ttc_thr * 0.2, drac_thr * 1.35, weight=3.0)
    check("penalty magnitude ordered", r_severe < r_mild <= 0.0,
          f"severe={r_severe:.3f} mild={r_mild:.3f}")
    check("penalty bounded by weight", r_severe >= -3.0 - 1e-9, f"{r_severe:.3f}")

    # Risk tolerance gates small severities.
    model.risk_tolerance = 0.999
    check("tolerance suppresses mild risk",
          model.get_risk(ttc_thr * 0.7, drac_thr * 1.05) == 0.0)
    model.risk_tolerance = 0.0

    # Frank comparator must still fit.
    frank = CopulaEVTModel(copula="frank", min_sample=300, min_exceedances=60)
    frank.buffer = model.buffer
    check("frank comparator fits", frank.update_model(verbose=False))


def test_conflict_indicators():
    print("\n[conflict indicators]")
    # Ego at origin heading +x at 20 m/s, leader 30 m ahead at 10 m/s.
    ego = VehicleState(0, 0, 20, 0, 0.0, id=1)
    lead = VehicleState(30, 0, 10, 0, 0.0, id=2)
    ttc, drac, gap = longitudinal_indicators(ego, lead)
    # gap = 30 - 4.5 = 25.5, dv = 10  ->  ttc = 2.55, drac = 100/51 = 1.96
    check("longitudinal ttc", abs(ttc - 2.55) < 1e-6, f"{ttc}")
    check("longitudinal drac", abs(drac - 100.0 / 51.0) < 1e-6, f"{drac}")
    check("gap", abs(gap - 25.5) < 1e-9, f"{gap}")

    # Separating traffic yields no conflict.
    fast_lead = VehicleState(30, 0, 30, 0, 0.0, id=3)
    ttc2, drac2, _ = longitudinal_indicators(ego, fast_lead)
    check("no conflict when separating", math.isinf(ttc2) and drac2 == 0.0)

    # A vehicle holding its lane 3.5 m to the side is never hit if both keep
    # their heading, and the elliptical clearance model says so.
    parallel_side = VehicleState(20, 3.5, 12, 0, 0.0, id=6)
    par_ttc, _, _, _ = planar_indicators(ego, parallel_side)
    check("parallel adjacent-lane traffic is not a conflict", math.isinf(par_ttc),
          f"ttc={par_ttc}")

    # The overtaking case: the same vehicle while the ego encroaches laterally.
    # This is what the original same-lane-only TTC could never see.
    side = VehicleState(20, 3.5, 12, 0.0, 0.0, id=4)
    encroaching = VehicleState(0, 0, 20, 1.5, 0.07, id=1)
    p_ttc, p_drac, _, closing = planar_indicators(encroaching, side)
    check("lane-change conflict is finite", math.isfinite(p_ttc), f"ttc={p_ttc}")
    check("lane-change closing speed positive", closing > 0, f"{closing}")
    check("lane-change produces a positive drac", p_drac > 0, f"{p_drac}")

    # A queued pair standing bumper to bumper must not read as a live conflict,
    # which is what makes the model usable on real urban data.
    q1 = VehicleState(0, 0, 0.0, 0, 0.0, length=6.2, width=2.1, id=7)
    q2 = VehicleState(8.2, 0, 0.0, 0, 0.0, length=6.2, width=2.1, id=8)
    ql_ttc, ql_drac, _ = longitudinal_indicators(q1, q2)
    qp_ttc, qp_drac, _, _ = planar_indicators(q1, q2)
    check("stationary queue is not a conflict",
          math.isinf(ql_ttc) and math.isinf(qp_ttc) and ql_drac == 0 and qp_drac == 0,
          f"lon={ql_ttc}/{ql_drac} planar={qp_ttc}/{qp_drac}")

    res = compute_conflict(ego, [lead, fast_lead, side], max_distance=50.0)
    check("scene aggregation counts all vehicles", res.n_interacting == 3, f"{res.n_interacting}")
    check("scene ttc is the minimum", res.ttc <= ttc + 1e-9, f"{res.ttc}")
    check("scene drac is at least the longitudinal one", res.drac >= drac - 1e-9, f"{res.drac}")
    check("partner is identified", res.partner_id in (2, 4), f"{res.partner_id}")

    # Far vehicles are excluded.
    far = VehicleState(400, 0, 0, 0, 0.0, id=5)
    res2 = compute_conflict(ego, [lead, far], max_distance=50.0)
    check("interaction radius respected", res2.n_interacting == 1, f"{res2.n_interacting}")

    # Neither variant dominates the other: the longitudinal projection ignores the
    # lateral offset, the planar one accounts for it.  What matters is that the
    # default "max" mode is the conservative combination of the two.
    res3 = compute_conflict(ego, [side], max_distance=50.0, mode="longitudinal")
    res4 = compute_conflict(ego, [side], max_distance=50.0, mode="planar")
    res5 = compute_conflict(ego, [side], max_distance=50.0, mode="max")
    check("max mode takes the smaller ttc", abs(res5.ttc - min(res3.ttc, res4.ttc)) < 1e-9,
          f"max={res5.ttc} lon={res3.ttc} planar={res4.ttc}")
    check("max mode takes the larger drac", abs(res5.drac - max(res3.drac, res4.drac)) < 1e-9,
          f"max={res5.drac} lon={res3.drac} planar={res4.drac}")
    # The longitudinal projection ignores lateral separation, so it still reports a
    # conflict with a vehicle that is safely one lane over; the planar variant does
    # not.  Combining them keeps the conservative answer, which is the point.
    check("longitudinal projection is conservative for offset traffic",
          math.isfinite(res3.ttc) and math.isinf(res4.ttc), f"{res3.ttc} {res4.ttc}")


def test_idm_mobil():
    print("\n[IDM / MOBIL]")
    p = IDMParams(v0=12.0, T=1.5, s0=2.0, a=1.4, b=2.0)

    # Free road at the desired speed: zero acceleration.
    check("free flow at v0 gives ~0 acc",
          abs(idm_acceleration(12.0, float("inf"), 0.0, p)) < 1e-9)
    check("free flow below v0 accelerates", idm_acceleration(5.0, float("inf"), 0.0, p) > 0)

    # Equilibrium spacing s_e = s0 + v T should give zero acceleration at v = v0.
    v = 8.0
    s_e = p.s0 + v * p.T
    a_eq = idm_acceleration(v, s_e, 0.0, p)
    check("equilibrium spacing is near zero acc", abs(a_eq) < 0.6, f"a={a_eq:.3f}")

    # Closing on a stopped leader must brake hard.
    check("hard braking when closing", idm_acceleration(15.0, 5.0, 15.0, p) < -3.0,
          f"{idm_acceleration(15.0, 5.0, 15.0, p):.2f}")
    # Tighter gap => stronger braking.
    check("monotone in gap",
          idm_acceleration(10.0, 6.0, 5.0, p) < idm_acceleration(10.0, 20.0, 5.0, p))

    # MOBIL: an empty adjacent lane in front of a slow leader is attractive.
    cur = {"lead_gap": 8.0, "lead_dv": 4.0, "follow_gap": 40.0, "follow_dv": 0.0, "follow_speed": 8.0}
    cand = {"lead_gap": 120.0, "lead_dv": 0.0, "follow_gap": 60.0, "follow_dv": 0.0, "follow_speed": 8.0}
    change, incentive, safe = mobil_decision(10.0, cur, cand, p, MOBILParams())
    check("mobil accepts a clearly better lane", change and safe, f"incentive={incentive:.3f}")

    # An adjacent lane with a fast follower right behind is unsafe.
    unsafe = {"lead_gap": 120.0, "lead_dv": 0.0, "follow_gap": 0.6, "follow_dv": 12.0, "follow_speed": 22.0}
    change2, _, safe2 = mobil_decision(10.0, cur, unsafe, p, MOBILParams())
    check("mobil rejects an unsafe lane", (not change2) and (not safe2))

    # Controller integration on a 4-lane road along -y.
    ctrl = IDMMobilController(lane_centres=[5.8, 9.0, 12.2, 15.6], rng=np.random.default_rng(3))
    ego = VehicleState(x=5.8, y=100.0, vx=0.0, vy=-10.0, yaw=-math.pi / 2, id=1)
    slow_lead = VehicleState(x=5.8, y=90.0, vx=0.0, vy=-3.0, yaw=-math.pi / 2, id=2)
    acc, target_x, diag = ctrl.step(ego, [slow_lead])
    check("controller brakes behind a slow leader", acc < 0, f"acc={acc:.2f}")
    check("controller reports the lane", diag["lane"] == 0, f"{diag}")
    check("controller returns a valid lane centre", target_x in ctrl.lane_centres, f"{target_x}")
    check("driver parameters are heterogeneous",
          ctrl.describe()["idm"]["v0"] != IDMParams().v0)


if __name__ == "__main__":
    test_logistic_copula()
    test_gpd_margin()
    test_evt_model_direction()
    test_conflict_indicators()
    test_idm_mobil()
    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
        sys.exit(1)
    print("all checks passed")
