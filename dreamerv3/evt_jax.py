"""
JAX evaluation of the fitted EVT tail-risk model inside latent imagination.

The manuscript claims that EVT is applied *forward*, to the trajectories the world
model imagines, rather than post hoc to realised trajectories.  Making that claim
literally true requires evaluating the EVT map inside the jitted actor-critic
update, where SciPy cannot be called.

The split used here is:

  * **Fitting** (GPD maximum likelihood, threshold selection, copula maximum
    likelihood) stays on the host, in :mod:`car_dreamer.evt_module`.  It runs every
    ``evt.update_interval`` environment steps.
  * **Evaluation** is a closed-form expression -- a Generalised Pareto CDF per
    margin and a logistic extreme-value copula -- reproduced here in ``jnp``.

The fitted parameters reach the device as an ordinary observation
(``evt_params``), so they travel through the replay buffer with the transitions
they were fitted on and are held fixed along an imagined rollout: within one
rollout the risk *model* is constant and only the predicted *state* evolves,
which is exactly the intended semantics.

The safety indicators themselves come from the world model's ``safety`` head,
which regresses the normalised ``[1 - TTC/TTC_HORIZON, DRAC/DRAC_SCALE]`` pair
from the latent state.  Nothing in this path depends on the reward head, so the
EVT term and the task reward have independent error sources.
"""

import jax.numpy as jnp

# Must match car_dreamer.evt_module.CopulaEVTModel.PARAM_KEYS.
PARAM_KEYS = (
    "fitted",
    "u_ttc",
    "xi_ttc",
    "sigma_ttc",
    "u_drac",
    "xi_drac",
    "sigma_drac",
    "alpha",
    "zeta_joint",
    "risk_tolerance",
)
PARAM_DIM = len(PARAM_KEYS)

# Must match car_dreamer.carla_wpt_env.
TTC_HORIZON = 10.0
DRAC_SCALE = 8.5

EPS = 1e-9


def gpd_cond_cdf(x, u, xi, sigma):
    """Conditional GPD CDF ``G(x - u)``, zero at or below the threshold.

    The ``xi -> 0`` branch is the exponential limit.  Both branches are always
    evaluated and selected with ``where`` so the expression stays jittable and
    free of NaNs when the model has not been fitted yet (all parameters zero).
    """
    sigma = jnp.maximum(sigma, 1e-6)
    z = jnp.maximum(x - u, 0.0)
    exp_branch = jnp.exp(-z / sigma)
    safe_xi = jnp.where(jnp.abs(xi) < 1e-6, 1.0, xi)
    base = jnp.maximum(1.0 + safe_xi * z / sigma, EPS)
    pow_branch = base ** (-1.0 / safe_xi)
    sf = jnp.where(jnp.abs(xi) < 1e-6, exp_branch, pow_branch)
    return jnp.clip(1.0 - sf, 0.0, 1.0)


def logistic_copula_cdf(u1, u2, alpha):
    """``C(u1, u2; alpha) = exp(-((-log u1)^(1/a) + (-log u2)^(1/a))^a)``."""
    a = jnp.clip(alpha, 0.05, 0.999)
    u1 = jnp.clip(u1, EPS, 1.0 - EPS)
    u2 = jnp.clip(u2, EPS, 1.0 - EPS)
    t1, t2 = -jnp.log(u1), -jnp.log(u2)
    s = t1 ** (1.0 / a) + t2 ** (1.0 / a)
    return jnp.exp(-(s ** a))


def denormalise_safety(safety):
    """Map the normalised safety observation back onto ``(-TTC, DRAC)``.

    ``safety[..., 0] = clip(1 - TTC / TTC_HORIZON, 0, 1)`` and
    ``safety[..., 1] = clip(DRAC / DRAC_SCALE, 0, 1)``, so ``-TTC`` is recovered
    as ``TTC_HORIZON * (safety0 - 1)``.  Both channels are clipped because the
    head's prediction is unconstrained.
    """
    ttc_n = jnp.clip(safety[..., 0], 0.0, 1.0)
    drac_n = jnp.clip(safety[..., 1], 0.0, 1.0)
    neg_ttc = TTC_HORIZON * (ttc_n - 1.0)
    drac = drac_n * DRAC_SCALE
    return neg_ttc, drac


def severity(safety, params):
    """``C(u1, u2)`` for the predicted safety indicators.  Shape ``[...]``."""
    neg_ttc, drac = denormalise_safety(safety)
    u1 = gpd_cond_cdf(neg_ttc, params[..., 1], params[..., 2], params[..., 3])
    u2 = gpd_cond_cdf(drac, params[..., 4], params[..., 5], params[..., 6])
    sev = logistic_copula_cdf(u1, u2, params[..., 7])
    # Outside the joint tail there is no extreme to speak of.
    in_tail = (u1 > 0.0) & (u2 > 0.0)
    return jnp.where(in_tail, sev, 0.0)


def joint_exceedance_prob(safety, params):
    """``P(-TTC > x, DRAC > y)``, the classical tail probability.  Diagnostic only."""
    neg_ttc, drac = denormalise_safety(safety)
    u1 = gpd_cond_cdf(neg_ttc, params[..., 1], params[..., 2], params[..., 3])
    u2 = gpd_cond_cdf(drac, params[..., 4], params[..., 5], params[..., 6])
    surv = jnp.clip(1.0 - u1 - u2 + logistic_copula_cdf(u1, u2, params[..., 7]), 0.0, 1.0)
    return params[..., 8] * surv


def evt_risk(safety, params):
    """Normalised tail risk in ``[0, 1]``, gated by the risk tolerance.

    Mirrors :meth:`car_dreamer.evt_module.CopulaEVTModel.get_risk` exactly, and
    returns zero whenever the host model has not produced a fit yet (the
    ``fitted`` flag in ``params[..., 0]``).
    """
    sev = severity(safety, params)
    tol = params[..., 9]
    risk = jnp.clip((sev - tol) / jnp.maximum(1.0 - tol, 1e-6), 0.0, 1.0)
    return params[..., 0] * risk
