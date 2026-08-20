import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from . import agent, evt_jax, expl, jaxutils
from . import ninjax as nj


class Greedy(nj.Module):
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
            if use_imag:
                # Fail loudly at trace time rather than silently returning the
                # unpenalised reward: a missing safety head or missing evt_params
                # would turn the whole EVT-in-imagination path into a no-op that
                # only shows up as an unexplained result days later.
                if "safety" not in wm.heads:
                    raise RuntimeError(
                        f"evt.mode={config.evt.mode} needs the world model safety "
                        "head, which is only built when the environment provides a "
                        "'safety' observation. Check env.evt.mode in the task config."
                    )
                if "evt_params" not in s:
                    raise RuntimeError(
                        f"evt.mode={config.evt.mode} needs the fitted EVT parameters "
                        "to reach imagination, but 'evt_params' is absent from the "
                        "trajectory. The environment must publish it as an "
                        "observation and WorldModel.imagine must carry it through."
                    )
                safety = wm.heads["safety"](s).mean()[1:]
                risk = evt_jax.evt_risk(safety, s["evt_params"][1:])
                reward = reward - w_evt * risk
            return reward

        if config.critic_type == "vfunction":
            critics = {"extr": agent.VFunction(rewfn, config, name="critic")}
        else:
            raise NotImplementedError(config.critic_type)
        self.ac = agent.ImagActorCritic(critics, {"extr": 1.0}, act_space, config, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}


class Random(nj.Module):
    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space

    def initial(self, batch_size):
        return jnp.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist = jaxutils.OneHotDist(jnp.zeros(shape))
        else:
            dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
            dist = tfd.Independent(dist, 1)
        return {"action": dist}, state

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}


class Explore(nj.Module):
    REWARDS = {
        "disag": expl.Disag,
    }

    def __init__(self, wm, act_space, config):
        self.config = config
        self.rewards = {}
        critics = {}
        for key, scale in config.expl_rewards.items():
            if not scale:
                continue
            if key == "extr":
                rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
                critics[key] = agent.VFunction(rewfn, config, name=key)
            else:
                rewfn = self.REWARDS[key](wm, act_space, config, name=key + "_reward")
                critics[key] = agent.VFunction(rewfn, config, name=key)
                self.rewards[key] = rewfn
        scales = {k: v for k, v in config.expl_rewards.items() if v}
        self.ac = agent.ImagActorCritic(critics, scales, act_space, config, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        metrics = {}
        for key, rewfn in self.rewards.items():
            mets = rewfn.train(data)
            metrics.update({f"{key}_k": v for k, v in mets.items()})
        traj, mets = self.ac.train(imagine, start, data)
        metrics.update(mets)
        return traj, metrics

    def report(self, data):
        return {}
