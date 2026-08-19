"""Apply the ProSafeAV EVT-in-imagination patch to the DreamerV3 sources.

Idempotent: every replacement checks whether it has already been applied.
"""
import sys

ROOT = "/home/yukai/CarDreamer_prosafeav/dreamerv3"
applied, skipped = [], []


def patch(path, old, new, tag):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if new in src:
        skipped.append(tag)
        return
    if old not in src:
        print(f"ERROR: anchor not found for {tag} in {path}")
        sys.exit(1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new, 1))
    applied.append(tag)


# --------------------------------------------------------------------------- #
# agent.py
# --------------------------------------------------------------------------- #
AGENT = f"{ROOT}/agent.py"

patch(
    AGENT,
    "from . import behaviors, jaxagent, jaxutils, nets\nfrom . import ninjax as nj",
    "from . import behaviors, evt_jax, jaxagent, jaxutils, nets\nfrom . import ninjax as nj",
    "agent:import",
)

patch(
    AGENT,
    """        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }""",
    """        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }
        # Safety head: regresses the normalised surrogate safety measures
        # [1 - TTC/TTC_HORIZON, DRAC/DRAC_SCALE] from the latent state, so the EVT
        # tail risk can be evaluated on imagined rollouts instead of only on
        # realised transitions.  Trained jointly with the other heads.
        self.use_safety_head = (
            config.evt.mode in ("imagine", "both") and "safety" in shapes
        )
        if self.use_safety_head:
            self.heads["safety"] = nets.MLP(
                shapes["safety"], **config.safety_head, name="safety"
            )""",
    "agent:safety_head",
)

patch(
    AGENT,
    """        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales""",
    """        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        if self.use_safety_head:
            scales["safety"] = config.loss_scales.safety
        self.scales = scales""",
    "agent:safety_scale",
)

# Carry the fitted EVT parameters along the imagined rollout.  They are constant
# within a rollout: the risk model is fixed, only the predicted state evolves.
IMAGINE_OLD = """    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)"""
IMAGINE_NEW = """    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        evt_params = start.get("evt_params", None)
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)"""
patch(AGENT, IMAGINE_OLD, IMAGINE_NEW, "agent:imagine_capture")

IMAGINE_TAIL_OLD = """        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()"""
IMAGINE_TAIL_NEW = """        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        if evt_params is not None:
            traj["evt_params"] = jnp.broadcast_to(
                evt_params[None], (horizon + 1,) + evt_params.shape
            )
        cont = self.heads["cont"](traj).mode()"""
patch(AGENT, IMAGINE_TAIL_OLD, IMAGINE_TAIL_NEW, "agent:imagine_carry_params")

CARRY_OLD = """    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}"""
CARRY_NEW = """    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        evt_params = start.get("evt_params", None)
        start = {k: v for k, v in start.items() if k in keys}"""
patch(AGENT, CARRY_OLD, CARRY_NEW, "agent:imagine_carry_capture")

CARRY_TAIL_OLD = """        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        cont = self.heads["cont"](traj).mode()"""
CARRY_TAIL_NEW = """        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        if evt_params is not None:
            traj["evt_params"] = jnp.broadcast_to(
                evt_params[None], (horizon + 1,) + evt_params.shape
            )
        cont = self.heads["cont"](traj).mode()"""
patch(AGENT, CARRY_TAIL_OLD, CARRY_TAIL_NEW, "agent:imagine_carry_params2")


# --------------------------------------------------------------------------- #
# behaviors.py
# --------------------------------------------------------------------------- #
BEHAVIORS = f"{ROOT}/behaviors.py"

patch(
    BEHAVIORS,
    "from . import agent, expl, jaxutils\nfrom . import ninjax as nj",
    "from . import agent, evt_jax, expl, jaxutils\nfrom . import ninjax as nj",
    "behaviors:import",
)

GREEDY_OLD = """class Greedy(nj.Module):
    def __init__(self, wm, act_space, config):
        rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
        if config.critic_type == "vfunction":"""
GREEDY_NEW = '''class Greedy(nj.Module):
    def __init__(self, wm, act_space, config):
        # ProSafeAV: subtract the EVT tail-risk penalty evaluated on the *imagined*
        # latent rollout.  The safety indicators come from the world model's safety
        # head and the fitted GPD/copula parameters ride along the trajectory as
        # ``evt_params``, so the penalty is a forward-looking risk estimate rather
        # than a replay of what the environment already observed.
        #
        # ``config.evt.mode``:
        #   none    -- no EVT anywhere (the DreamerV3 ablation)
        #   env     -- penalty applied only in the environment reward (post hoc)
        #   imagine -- penalty applied only here, on imagined rollouts
        #   both    -- applied in both places
        use_imag = config.evt.mode in ("imagine", "both")
        w_evt = float(config.evt.imag_weight)

        def rewfn(s):
            reward = wm.heads["reward"](s).mean()[1:]
            if use_imag and "safety" in wm.heads and "evt_params" in s:
                safety = wm.heads["safety"](s).mean()[1:]
                risk = evt_jax.evt_risk(safety, s["evt_params"][1:])
                reward = reward - w_evt * risk
            return reward

        if config.critic_type == "vfunction":'''
patch(BEHAVIORS, GREEDY_OLD, GREEDY_NEW, "behaviors:greedy_rewfn")


# --------------------------------------------------------------------------- #
# dreamerv3.yaml
# --------------------------------------------------------------------------- #
YAML = f"{ROOT}/dreamerv3.yaml"

patch(
    YAML,
    """  cont_head:
    {
      layers: 5,""",
    """  # ProSafeAV: head predicting the normalised surrogate safety measures
  # [1 - TTC/10s, DRAC/8.5] from the latent state.  Its output feeds the EVT
  # tail-risk penalty during imagination.
  safety_head:
    {
      layers: 4,
      units: 512,
      act: silu,
      norm: layer,
      dist: mse,
      outscale: 1.0,
      outnorm: False,
      inputs: [deter, stoch],
      winit: normal,
      fan: avg,
    }

  # ProSafeAV EVT integration.
  #   mode        none | env | imagine | both
  #   imag_weight weight of the tail-risk penalty inside imagination; the
  #               environment-side weight stays in env.reward.scales.evt
  evt: { mode: both, imag_weight: 3.0 }

  cont_head:
    {
      layers: 5,""",
    "yaml:safety_head",
)

patch(
    YAML,
    """      reward: 1.0,
      cont: 1.0,
      dyn: 0.5,""",
    """      reward: 1.0,
      cont: 1.0,
      safety: 1.0,
      dyn: 0.5,""",
    "yaml:loss_scale",
)

print("applied:", applied)
print("already present:", skipped)
