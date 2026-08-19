"""Check that the device-side EVT map matches the host-side one exactly.

The EVT model is *fitted* on the host with SciPy and *evaluated* on the device
inside the jitted imagination rollout.  If the two evaluations disagree, the
penalty the policy is optimising against is not the penalty the manuscript
describes, so this equivalence is worth asserting explicitly.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp

from car_dreamer.carla_wpt_env import DRAC_SCALE, TTC_HORIZON
from car_dreamer.evt_module import CopulaEVTModel
from dreamerv3 import evt_jax

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


def normalise(ttc, drac):
    """Reproduce the environment's safety observation."""
    ttc_n = 0.0 if not np.isfinite(ttc) else float(np.clip(1.0 - ttc / TTC_HORIZON, 0.0, 1.0))
    drac_n = float(np.clip(drac / DRAC_SCALE, 0.0, 1.0))
    return np.array([ttc_n, drac_n], dtype=np.float32)


def main():
    print("[host vs device EVT evaluation]")
    check("PARAM_DIM agrees", CopulaEVTModel.PARAM_DIM == evt_jax.PARAM_DIM,
          f"{CopulaEVTModel.PARAM_DIM} vs {evt_jax.PARAM_DIM}")
    check("PARAM_KEYS agree", tuple(CopulaEVTModel.PARAM_KEYS) == tuple(evt_jax.PARAM_KEYS))
    check("TTC_HORIZON agrees", TTC_HORIZON == evt_jax.TTC_HORIZON)
    check("DRAC_SCALE agrees", DRAC_SCALE == evt_jax.DRAC_SCALE)

    rng = np.random.default_rng(7)
    model = CopulaEVTModel(min_sample=300, min_exceedances=60)
    for _ in range(8000):
        ttc = float(np.clip(rng.gamma(2.5, 1.1), 0.1, 9.9))
        drac = float(np.clip(3.5 / ttc + rng.normal(0, 0.6), 0.0, 14.0))
        model.add_sample(ttc, drac)
    ok = model.update_model(verbose=True)
    check("model fits", ok)
    if not ok:
        return

    params = model.param_vector()
    ttc_thr, drac_thr = -model.margin_ttc.u, model.margin_drac.u

    # A grid that straddles both thresholds, so the branch structure is exercised.
    ttcs = np.concatenate([
        np.linspace(0.05, ttc_thr, 15),
        np.linspace(ttc_thr, min(ttc_thr + 4.0, 9.9), 10),
    ])
    dracs = np.concatenate([
        np.linspace(max(drac_thr - 4.0, 0.0), drac_thr, 10),
        np.linspace(drac_thr, DRAC_SCALE, 15),
    ])

    host_risk, host_sev, host_tail, safety = [], [], [], []
    for t in ttcs:
        for d in dracs:
            host_risk.append(model.get_risk(float(t), float(d)))
            host_sev.append(model.severity(float(t), float(d)))
            host_tail.append(model.joint_exceedance_prob(float(t), float(d)))
            safety.append(normalise(float(t), float(d)))

    safety = jnp.asarray(np.stack(safety))
    p = jnp.asarray(np.broadcast_to(params, (safety.shape[0], params.shape[0])))

    dev_risk = np.asarray(jax.jit(evt_jax.evt_risk)(safety, p))
    dev_sev = np.asarray(jax.jit(evt_jax.severity)(safety, p))
    dev_tail = np.asarray(jax.jit(evt_jax.joint_exceedance_prob)(safety, p))

    host_risk = np.array(host_risk)
    host_sev = np.array(host_sev)
    host_tail = np.array(host_tail)

    # float32 on the device against float64 on the host, so a small tolerance is
    # expected; anything larger means the two formulations have drifted apart.
    for name, h, dv, tol in [
        ("severity", host_sev, dev_sev, 2e-4),
        ("risk", host_risk, dev_risk, 2e-4),
        ("tail probability", host_tail, dev_tail, 2e-5),
    ]:
        err = float(np.max(np.abs(h - dv)))
        check(f"{name} matches host", err < tol, f"max |diff| = {err:.3e}")

    check("device output is finite", bool(np.all(np.isfinite(dev_risk))))
    check("device risk within [0,1]", bool(dev_risk.min() >= 0.0 and dev_risk.max() <= 1.0),
          f"[{dev_risk.min():.4f}, {dev_risk.max():.4f}]")
    check("some probes land in the joint tail", bool((dev_risk > 0).any()),
          f"{int((dev_risk > 0).sum())} of {dev_risk.size}")

    # An unfitted model must contribute nothing, and must not produce NaNs.
    zeros = jnp.zeros((4, evt_jax.PARAM_DIM))
    unfitted = np.asarray(jax.jit(evt_jax.evt_risk)(safety[:4], zeros))
    check("unfitted model gives zero risk", bool(np.all(unfitted == 0.0)), f"{unfitted}")
    check("unfitted model gives no NaN", bool(np.all(np.isfinite(unfitted))))

    # The penalty must be differentiable w.r.t. the predicted safety indicators,
    # otherwise it cannot shape the actor through the imagination gradient.
    grad = jax.grad(lambda s: evt_jax.evt_risk(s, params).sum())(jnp.asarray(np.stack(
        [normalise(ttc_thr * 0.3, drac_thr * 1.2)]
    )))
    grad = np.asarray(grad)
    check("gradient is finite", bool(np.all(np.isfinite(grad))), f"{grad}")
    check("gradient is non-zero inside the tail", bool(np.any(np.abs(grad) > 0)), f"{grad}")

    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
        sys.exit(1)
    print("host and device EVT evaluations agree")


if __name__ == "__main__":
    main()
