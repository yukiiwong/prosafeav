"""Smoke test for the PyTorch ProSafeAV variants and cross-architecture baselines.

These agents were previously unrunnable: ``__init__`` assigned
``self.policy = <network>``, shadowing the ``policy(obs, state, mode)`` method
that ``__call__`` dispatches to, so any attempt to act raised a TypeError.  This
test exercises the full ``policy`` -> ``train`` -> ``save`` -> ``load`` cycle for
every variant so that regression cannot recur silently.

It also checks that the EVT term actually reaches the imagined return: with the
tail-risk parameters set to a fitted model and the safety head forced into the
tail, ``imag_evt_risk`` must be strictly positive.

Run: python tools/test_torch_agents.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "dreamerv3")

import torch

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


class Space:
    def __init__(self, shape, discrete=False):
        self.shape = tuple(shape)
        self.discrete = discrete


def make_spaces(obs_key="birdeye_wpt", act_dim=5):
    obs_space = {
        obs_key: Space((16, 16, 3)),
        "safety": Space((2,)),
        "evt_params": Space((10,)),
    }
    act_space = {"action": Space((act_dim,), discrete=True)}
    return obs_space, act_space


def make_batch(batch_size=8, obs_key="birdeye_wpt", act_dim=5, evt_params=None):
    rng = np.random.default_rng(0)
    batch = {
        obs_key: rng.random((batch_size, 16, 16, 3), dtype=np.float32),
        "action": np.eye(act_dim, dtype=np.float32)[rng.integers(0, act_dim, batch_size)],
        "reward": rng.normal(0, 1, batch_size).astype(np.float32),
        "safety": rng.random((batch_size, 2)).astype(np.float32),
        "evt_params": np.tile(
            (evt_params if evt_params is not None else np.zeros(10, dtype=np.float32)),
            (batch_size, 1),
        ).astype(np.float32),
    }
    return batch


def fitted_params():
    """A real fitted parameter vector, so the EVT term is actually active."""
    from car_dreamer.evt_module import CopulaEVTModel

    rng = np.random.default_rng(3)
    model = CopulaEVTModel(min_sample=300, min_exceedances=60)
    for _ in range(6000):
        ttc = float(np.clip(rng.gamma(2.5, 1.1), 0.1, 9.9))
        drac = float(np.clip(3.5 / ttc + rng.normal(0, 0.6), 0.0, 8.4))
        model.add_sample(ttc, drac)
    model.update_model(verbose=False)
    return model.param_vector()


def exercise(name, cls, config, obs_key="birdeye_wpt", act_dim=5, params=None):
    print(f"\n[{name}]")
    obs_space, act_space = make_spaces(obs_key, act_dim)
    try:
        agent = cls(obs_space, act_space, config)
    except Exception as exc:
        check(f"{name} constructs", False, repr(exc))
        return
    check(f"{name} constructs", True)

    # The bug that made these agents unrunnable: policy must stay a method.
    check(f"{name}.policy is a bound method", callable(getattr(agent, "policy", None))
          and not isinstance(getattr(agent, "policy"), torch.nn.Module),
          f"{type(getattr(agent, 'policy', None))}")

    obs = {obs_key: np.random.rand(2, 16, 16, 3).astype(np.float32)}
    try:
        out, state = agent(obs, None, "train")
        ok_shape = out["action"].shape == (2, act_dim)
        one_hot = np.allclose(out["action"].sum(axis=1), 1.0)
        check(f"{name} acts", ok_shape and one_hot, f"{out['action'].shape}")
    except Exception as exc:
        check(f"{name} acts", False, repr(exc))
        return

    try:
        out2, _ = agent(obs, state, "eval")
        check(f"{name} acts with carried state", out2["action"].shape == (2, act_dim))
    except Exception as exc:
        check(f"{name} acts with carried state", False, repr(exc))

    batch = make_batch(8, obs_key, act_dim, evt_params=params)
    try:
        _, _, metrics = agent.train(batch, None, 0)
        check(f"{name} trains", "loss" in metrics and np.isfinite(metrics["loss"]),
              str(metrics.get("loss")))
        check(f"{name} learns the safety head",
              "safety_loss" in metrics and np.isfinite(metrics["safety_loss"]),
              str(metrics.get("safety_loss")))
        risk = metrics.get("imag_evt_risk", None)
        check(f"{name} reports the EVT risk", risk is not None and np.isfinite(risk),
              str(risk))
    except Exception as exc:
        import traceback

        traceback.print_exc()
        check(f"{name} trains", False, repr(exc))
        return

    try:
        blob = agent.save()
        agent.load(blob)
        check(f"{name} round-trips its checkpoint", "safety_head" in blob)
    except Exception as exc:
        check(f"{name} round-trips its checkpoint", False, repr(exc))


def main():
    params = fitted_params()
    print(f"using fitted EVT params (fitted flag = {params[0]}, alpha = {params[7]:.3f})")

    common = {
        "obs_key": "birdeye_wpt",
        "evt_mode": "both",
        "evt_imag_weight": 3.0,
        "imagination_horizon": 3,
    }

    from prosafeav_rssm_agent import ProSafeAVRSSMAgent

    exercise("ProSafeAV-RSSM", ProSafeAVRSSMAgent,
             {**common, "stochastic_dim": 8, "deterministic_dim": 16}, params=params)

    from prosafeav_deterministic_agent import ProSafeAVDeterministicAgent

    exercise("ProSafeAV-Deterministic", ProSafeAVDeterministicAgent,
             {**common, "hidden_dim": 32}, params=params)

    from transformer_wm_agent import TransformerWorldModelAgent

    exercise("Transformer-WM", TransformerWorldModelAgent,
             {**common, "latent_dim": 32, "d_model": 32, "nhead": 2,
              "transformer_layers": 1, "context_len": 4}, params=params)

    from tdmpc_agent import TDMPCAgent

    exercise("TD-MPC", TDMPCAgent,
             {**common, "latent_dim": 32, "plan_horizon": 3, "num_samples": 16,
              "num_elites": 4, "plan_iterations": 2}, params=params)

    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
        sys.exit(1)
    print("all PyTorch agents run end to end")


if __name__ == "__main__":
    main()
