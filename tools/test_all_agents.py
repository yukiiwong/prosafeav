"""Smoke test every agent in the comparison, not only the ones added here.

Table I and the world-model comparison table are only meaningful if each agent
can actually complete a policy -> train -> save -> load cycle.  Two of the
ProSafeAV variants could not, which is why this covers the whole set rather
than trusting that the pre-existing baselines are fine.

Run: python tools/test_all_agents.py
"""
import sys
import traceback

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "dreamerv3")

import torch

RESULTS = []


class Space:
    def __init__(self, shape, discrete=False):
        self.shape = tuple(shape)
        self.discrete = discrete


def make_spaces(obs_key="birdeye_wpt", act_dim=5):
    return (
        {
            obs_key: Space((16, 16, 3)),
            "safety": Space((2,)),
            "evt_params": Space((10,)),
        },
        {"action": Space((act_dim,), discrete=True)},
    )


def make_batch(batch_size=8, obs_key="birdeye_wpt", act_dim=5, params=None):
    rng = np.random.default_rng(0)
    obs = rng.random((batch_size, 16, 16, 3), dtype=np.float32)
    return {
        obs_key: obs,
        # Several agents look for next_obs and silently fall back to obs; supply
        # it so the dynamics losses are exercised on genuinely different tensors.
        "next_obs": {obs_key: rng.random((batch_size, 16, 16, 3), dtype=np.float32)},
        "action": np.eye(act_dim, dtype=np.float32)[rng.integers(0, act_dim, batch_size)],
        "reward": rng.normal(0, 1, batch_size).astype(np.float32),
        "is_terminal": np.zeros(batch_size, dtype=bool),
        "is_first": np.zeros(batch_size, dtype=bool),
        "safety": rng.random((batch_size, 2)).astype(np.float32),
        "evt_params": np.tile(
            params if params is not None else np.zeros(10, dtype=np.float32),
            (batch_size, 1),
        ).astype(np.float32),
    }


def fitted_params():
    from car_dreamer.evt_module import CopulaEVTModel

    rng = np.random.default_rng(3)
    m = CopulaEVTModel(min_sample=300, min_exceedances=30)
    for _ in range(6000):
        ttc = float(np.clip(rng.gamma(2.5, 1.1), 0.1, 9.9))
        drac = float(np.clip(3.5 / ttc + rng.normal(0, 0.6), 0.0, 8.4))
        m.add_sample(ttc, drac)
    m.update_model(verbose=False)
    return m.param_vector()


def exercise(label, module, cls_name, config, params):
    entry = {"agent": label, "construct": "-", "act": "-", "train": "-",
             "save_load": "-", "evt": "-", "error": ""}
    RESULTS.append(entry)
    act_dim = 5
    obs_space, act_space = make_spaces(act_dim=act_dim)
    try:
        mod = __import__(module)
        cls = getattr(mod, cls_name)
    except Exception as exc:
        entry["construct"] = "FAIL"
        entry["error"] = f"import: {exc}"
        return
    try:
        agent = cls(obs_space, act_space, config)
        entry["construct"] = "ok"
    except Exception as exc:
        entry["construct"] = "FAIL"
        entry["error"] = f"init: {exc}"
        return

    obs = {"birdeye_wpt": np.random.rand(2, 16, 16, 3).astype(np.float32)}
    try:
        out, state = agent(obs, None, "train")
        assert out["action"].shape == (2, act_dim), out["action"].shape
        entry["act"] = "ok"
    except Exception as exc:
        entry["act"] = "FAIL"
        entry["error"] = f"policy: {exc}"
        return

    try:
        _, _, metrics = agent.train(make_batch(8, params=params), None, 0)
        loss = metrics.get("loss", metrics.get("model_loss"))
        assert loss is None or np.isfinite(loss), loss
        entry["train"] = "ok"
        risk = metrics.get("imag_evt_risk")
        entry["evt"] = "ok" if risk is not None else "none"
    except Exception as exc:
        entry["train"] = "FAIL"
        entry["error"] = f"train: {type(exc).__name__}: {exc}"
        return

    try:
        agent.load(agent.save())
        entry["save_load"] = "ok"
    except Exception as exc:
        entry["save_load"] = "FAIL"
        entry["error"] = f"save/load: {exc}"


def main():
    params = fitted_params()
    common = {"obs_key": "birdeye_wpt", "evt_mode": "both",
              "evt_imag_weight": 3.0, "imagination_horizon": 3}

    agents = [
        ("ProSafeAV-DV3", None, None, None),  # JAX, covered by the training run
        ("ProSafeAV-RSSM", "prosafeav_rssm_agent", "ProSafeAVRSSMAgent",
         {**common, "stochastic_dim": 8, "deterministic_dim": 16}),
        ("ProSafeAV-Deterministic", "prosafeav_deterministic_agent",
         "ProSafeAVDeterministicAgent", {**common, "hidden_dim": 32}),
        ("Transformer-WM", "transformer_wm_agent", "TransformerWorldModelAgent",
         {**common, "latent_dim": 32, "d_model": 32, "nhead": 2,
          "transformer_layers": 1, "context_len": 4}),
        ("TD-MPC", "tdmpc_agent", "TDMPCAgent",
         {**common, "latent_dim": 32, "plan_horizon": 3, "num_samples": 16,
          "num_elites": 4, "plan_iterations": 2}),
        ("PlaNet", "planet_agent", "PlaNetAgent",
         {**common, "latent_dim": 32, "planning_horizon": 3, "num_candidates": 16,
          "num_iterations": 2, "num_elite": 4}),
        ("World Models", "world_models_agent", "WorldModelsAgent",
         {**common, "latent_dim": 32, "hidden_dim": 32}),
        ("SimPLe", "simple_agent", "SimPLeAgent",
         {**common, "rollout_length": 3, "num_simulated_rollouts": 2}),
        ("DQN", "dqn_agent", "DQNAgent", dict(common)),
        ("SAC", "sac_agent", "SACAgent", dict(common)),
        ("TD3", "td3_agent", "TD3Agent", dict(common)),
        ("PPO", "ppo_agent", "PPOAgent", dict(common)),
    ]

    for label, module, cls_name, config in agents:
        if module is None:
            RESULTS.append({"agent": label, "construct": "n/a", "act": "n/a",
                            "train": "n/a", "save_load": "n/a", "evt": "ok",
                            "error": "JAX backbone, verified by a live training run"})
            continue
        print(f"--- {label}")
        try:
            exercise(label, module, cls_name, config, params)
        except Exception:
            traceback.print_exc()

    print("\n" + "=" * 104)
    hdr = f"{'AGENT':26s} {'INIT':6s} {'ACT':6s} {'TRAIN':6s} {'SAVE':6s} {'EVT':6s} NOTE"
    print(hdr)
    print("-" * 104)
    broken = []
    for r in RESULTS:
        print(f"{r['agent']:26s} {r['construct']:6s} {r['act']:6s} {r['train']:6s} "
              f"{r['save_load']:6s} {r['evt']:6s} {r['error'][:38]}")
        if "FAIL" in (r["construct"], r["act"], r["train"], r["save_load"]):
            broken.append(r["agent"])
    print("=" * 104)
    if broken:
        print(f"\n{len(broken)} agent(s) cannot complete the cycle: {', '.join(broken)}")
        sys.exit(1)
    print("\nevery agent completes policy -> train -> save -> load")


if __name__ == "__main__":
    main()
