"""
PyTorch evaluation of the fitted EVT tail-risk model inside latent imagination.

This is the exact counterpart of :mod:`dreamerv3.evt_jax` for the PyTorch agents
(``prosafeav_rssm_agent``, ``prosafeav_deterministic_agent``, the transformer
world model and the TD-MPC style planner).  Keeping a single closed-form
definition in three places -- NumPy on the host for fitting, JAX for the
DreamerV3 backbone, PyTorch for the rest -- is what makes the ablation table an
apples-to-apples comparison: every variant is penalised by the same function of
its own predicted safety indicators.

The equivalence of the three implementations is asserted in
``tools/test_evt_torch.py``.

Fitting always happens on the host in :mod:`car_dreamer.evt_module`; only the
evaluation lives here.  The fitted parameters arrive as the ``evt_params``
observation and are held fixed along a rollout, so within one imagined
trajectory the risk *model* is constant and only the predicted *state* moves.
"""

import torch

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

    Both the ``xi -> 0`` exponential limit and the general branch are evaluated
    and selected with ``where`` so the expression stays differentiable and free
    of NaNs when the model has not been fitted yet (all parameters zero).
    """
    sigma = torch.clamp(sigma, min=1e-6)
    z = torch.clamp(x - u, min=0.0)
    exp_branch = torch.exp(-z / sigma)
    small = xi.abs() < 1e-6
    safe_xi = torch.where(small, torch.ones_like(xi), xi)
    base = torch.clamp(1.0 + safe_xi * z / sigma, min=EPS)
    pow_branch = base ** (-1.0 / safe_xi)
    sf = torch.where(small, exp_branch, pow_branch)
    return torch.clamp(1.0 - sf, 0.0, 1.0)


def logistic_copula_cdf(u1, u2, alpha):
    """``C(u1, u2; alpha) = exp(-((-log u1)^(1/a) + (-log u2)^(1/a))^a)``."""
    a = torch.clamp(alpha, 0.05, 0.999)
    u1 = torch.clamp(u1, EPS, 1.0 - EPS)
    u2 = torch.clamp(u2, EPS, 1.0 - EPS)
    t1, t2 = -torch.log(u1), -torch.log(u2)
    s = t1 ** (1.0 / a) + t2 ** (1.0 / a)
    return torch.exp(-(s ** a))


def denormalise_safety(safety):
    """Map the normalised safety prediction back onto ``(-TTC, DRAC)``."""
    ttc_n = torch.clamp(safety[..., 0], 0.0, 1.0)
    drac_n = torch.clamp(safety[..., 1], 0.0, 1.0)
    neg_ttc = TTC_HORIZON * (ttc_n - 1.0)
    drac = drac_n * DRAC_SCALE
    return neg_ttc, drac


def severity(safety, params):
    """``C(u1, u2)`` for the predicted safety indicators.  Shape ``[...]``."""
    neg_ttc, drac = denormalise_safety(safety)
    u1 = gpd_cond_cdf(neg_ttc, params[..., 1], params[..., 2], params[..., 3])
    u2 = gpd_cond_cdf(drac, params[..., 4], params[..., 5], params[..., 6])
    sev = logistic_copula_cdf(u1, u2, params[..., 7])
    in_tail = (u1 > 0.0) & (u2 > 0.0)
    return torch.where(in_tail, sev, torch.zeros_like(sev))


def joint_exceedance_prob(safety, params):
    """``P(-TTC > x, DRAC > y)``, the classical tail probability.  Diagnostic only."""
    neg_ttc, drac = denormalise_safety(safety)
    u1 = gpd_cond_cdf(neg_ttc, params[..., 1], params[..., 2], params[..., 3])
    u2 = gpd_cond_cdf(drac, params[..., 4], params[..., 5], params[..., 6])
    surv = torch.clamp(1.0 - u1 - u2 + logistic_copula_cdf(u1, u2, params[..., 7]), 0.0, 1.0)
    return params[..., 8] * surv


def evt_risk(safety, params):
    """Normalised tail risk in ``[0, 1]``, gated by the risk tolerance.

    Mirrors :meth:`car_dreamer.evt_module.CopulaEVTModel.get_risk`, and returns
    zero whenever the host model has not produced a fit yet (the ``fitted`` flag
    in ``params[..., 0]``).
    """
    sev = severity(safety, params)
    tol = params[..., 9]
    risk = torch.clamp((sev - tol) / torch.clamp(1.0 - tol, min=1e-6), 0.0, 1.0)
    return params[..., 0] * risk


class SafetyHead(torch.nn.Module):
    """Predicts the normalised ``[1 - TTC/10s, DRAC/8.5]`` pair from a latent state.

    Deliberately small: the target is two bounded scalars, and keeping the head
    cheap means the EVT term costs almost nothing at inference, which matters for
    the computational-efficiency table.  The sigmoid enforces the ``[0, 1]`` range
    the EVT map expects.
    """

    def __init__(self, latent_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden),
            torch.nn.ELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ELU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, latent):
        return torch.sigmoid(self.net(latent))


class EVTImaginationPenalty:
    """Bookkeeping for the EVT term inside a PyTorch imagination loop.

    Usage inside an agent::

        self.evt = EVTImaginationPenalty(config)
        ...
        # per imagined step
        risk = self.evt.risk(self.safety_head(latent), evt_params)
        imagined_reward = imagined_reward + reward - self.evt.weight * risk

    ``mode`` mirrors ``car_dreamer``'s ``evt.mode``:  the penalty is applied here
    only for ``imagine`` and ``both``, because in ``env`` mode it is already
    inside the reward the reward model was trained to predict, and applying it
    twice would double-count it.
    """

    def __init__(self, config):
        self.mode = config.get("evt_mode", "both")
        self.weight = float(config.get("evt_imag_weight", 3.0))
        self.enabled = self.mode in ("imagine", "both")

    def risk(self, safety_pred, evt_params):
        if not self.enabled:
            return torch.zeros(safety_pred.shape[:-1], device=safety_pred.device)
        return evt_risk(safety_pred, evt_params)

    def penalty(self, safety_pred, evt_params):
        return self.weight * self.risk(safety_pred, evt_params)
