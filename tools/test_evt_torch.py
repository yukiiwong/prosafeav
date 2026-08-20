"""Check that all three EVT evaluations agree: NumPy host, JAX, and PyTorch.

The EVT map is fitted once on the host and then evaluated in three places -- by
the environment (NumPy), inside the DreamerV3 imagination rollout (JAX), and
inside the PyTorch variants' imagination rollouts (Torch).  If they disagree,
the ablation table is not comparing the same penalty across backbones, so the
equivalence is asserted rather than assumed.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

import torch

from car_dreamer.carla_wpt_env import DRAC_SCALE, TTC_HORIZON
from car_dreamer.evt_module import CopulaEVTModel
from dreamerv3 import evt_torch

try:
    import jax
    import jax.numpy as jnp

    from dreamerv3 import evt_jax

    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


def normalise(ttc, drac):
    ttc_n = 0.0 if not np.isfinite(ttc) else float(np.clip(1.0 - ttc / TTC_HORIZON, 0.0, 1.0))
    drac_n = float(np.clip(drac / DRAC_SCALE, 0.0, 1.0))
    return np.array([ttc_n, drac_n], dtype=np.float32)


def main():
    print("[host / JAX / PyTorch EVT agreement]")
    check("PARAM_DIM agrees", CopulaEVTModel.PARAM_DIM == evt_torch.PARAM_DIM,
          f"{CopulaEVTModel.PARAM_DIM} vs {evt_torch.PARAM_DIM}")
    check("PARAM_KEYS agree", tuple(CopulaEVTModel.PARAM_KEYS) == tuple(evt_torch.PARAM_KEYS))
    check("TTC_HORIZON agrees", TTC_HORIZON == evt_torch.TTC_HORIZON)
    check("DRAC_SCALE agrees", DRAC_SCALE == evt_torch.DRAC_SCALE)

    rng = np.random.default_rng(11)
    model = CopulaEVTModel(min_sample=300, min_exceedances=60)
    for _ in range(8000):
        ttc = float(np.clip(rng.gamma(2.5, 1.1), 0.1, 9.9))
        drac = float(np.clip(3.5 / ttc + rng.normal(0, 0.6), 0.0, 8.4))
        model.add_sample(ttc, drac)
    ok = model.update_model(verbose=True)
    check("model fits", ok)
    if not ok:
        return

    params = model.param_vector()
    ttc_thr, drac_thr = -model.margin_ttc.u, model.margin_drac.u

    ttcs = np.concatenate([np.linspace(0.05, ttc_thr, 15),
                           np.linspace(ttc_thr, min(ttc_thr + 4.0, 9.9), 10)])
    dracs = np.concatenate([np.linspace(max(drac_thr - 4.0, 0.0), drac_thr, 10),
                            np.linspace(drac_thr, DRAC_SCALE, 15)])

    host_risk, safety = [], []
    for t in ttcs:
        for d in dracs:
            host_risk.append(model.get_risk(float(t), float(d)))
            safety.append(normalise(float(t), float(d)))
    host_risk = np.array(host_risk)
    safety_np = np.stack(safety)
    p_np = np.broadcast_to(params, (safety_np.shape[0], params.shape[0]))

    torch_risk = evt_torch.evt_risk(
        torch.from_numpy(safety_np), torch.from_numpy(np.ascontiguousarray(p_np))
    ).numpy()
    err = float(np.max(np.abs(host_risk - torch_risk)))
    check("torch risk matches host", err < 2e-4, f"max |diff| = {err:.3e}")

    if HAVE_JAX:
        jax_risk = np.asarray(jax.jit(evt_jax.evt_risk)(
            jnp.asarray(safety_np), jnp.asarray(p_np)))
        err_tj = float(np.max(np.abs(jax_risk - torch_risk)))
        check("torch matches jax", err_tj < 2e-5, f"max |diff| = {err_tj:.3e}")
    else:
        print("  SKIP  jax comparison (jax unavailable)")

    check("torch risk within [0,1]",
          bool(torch_risk.min() >= 0.0 and torch_risk.max() <= 1.0),
          f"[{torch_risk.min():.4f}, {torch_risk.max():.4f}]")
    check("some probes land in the joint tail", bool((torch_risk > 0).any()),
          f"{int((torch_risk > 0).sum())} of {torch_risk.size}")

    # Unfitted model must contribute nothing and produce no NaN.
    zeros = torch.zeros(4, evt_torch.PARAM_DIM, dtype=torch.float32)
    unfitted = evt_torch.evt_risk(torch.from_numpy(safety_np[:4]), zeros).numpy()
    check("unfitted gives zero risk", bool(np.all(unfitted == 0.0)), f"{unfitted}")
    check("unfitted gives no NaN", bool(np.all(np.isfinite(unfitted))))

    # The penalty must be differentiable so it can shape the policy through the
    # imagination gradient.
    probe = torch.tensor(normalise(ttc_thr * 0.3, drac_thr * 1.2)[None], requires_grad=True)
    p_t = torch.from_numpy(np.ascontiguousarray(params[None]))
    evt_torch.evt_risk(probe, p_t).sum().backward()
    grad = probe.grad.numpy()
    check("gradient is finite", bool(np.all(np.isfinite(grad))), f"{grad}")
    check("gradient is non-zero inside the tail", bool(np.any(np.abs(grad) > 0)), f"{grad}")

    # SafetyHead must emit values in the range the EVT map expects.
    head = evt_torch.SafetyHead(32)
    out = head(torch.randn(64, 32))
    check("SafetyHead output shape", tuple(out.shape) == (64, 2), f"{tuple(out.shape)}")
    check("SafetyHead output in [0,1]",
          bool(out.min().item() >= 0.0 and out.max().item() <= 1.0))

    # The mode switch must gate the penalty, otherwise `evt.mode=env` would
    # double-count the term that is already inside the environment reward.
    # Probe with states that are known to sit inside the joint tail, so a zero
    # here means the gate fired rather than that the probes were benign.
    tail_idx = np.flatnonzero(torch_risk > 0)[:8]
    check("tail probes available for the mode test", tail_idx.size > 0, f"{tail_idx.size}")
    tail_safety = torch.from_numpy(safety_np[tail_idx])
    tail_params = torch.from_numpy(np.ascontiguousarray(p_np[tail_idx]))
    for mode, expect in [("both", True), ("imagine", True), ("env", False), ("none", False)]:
        pen = evt_torch.EVTImaginationPenalty({"evt_mode": mode, "evt_imag_weight": 3.0})
        risk = pen.risk(tail_safety, tail_params)
        active = bool((risk > 0).any())
        check(f"mode '{mode}' {'applies' if expect else 'suppresses'} the penalty",
              active == expect, f"risk max = {risk.max().item():.4f}")

    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
        sys.exit(1)
    print("all three EVT implementations agree")


if __name__ == "__main__":
    main()
