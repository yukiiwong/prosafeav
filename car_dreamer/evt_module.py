"""
Bivariate peaks-over-threshold (POT) extreme value model for traffic conflict risk.

This module implements the EVT component of ProSafeAV exactly as specified in the
manuscript:

  * Generalised Pareto (GPD) margins fitted to exceedances of the two surrogate
    safety measures (negated TTC and DRAC) over automatically selected thresholds.
  * A *logistic* (Gumbel-Hougaard) extreme-value copula for the dependence
    structure, with the dependence parameter ``alpha`` estimated by maximum
    likelihood.  ``alpha -> 0`` is complete dependence, ``alpha -> 1`` is
    independence.
  * Three distinct risk read-outs, kept separate on purpose because they answer
    different questions:

      ``severity(x, y)``            = C(u1, u2)              in [0, 1], increases
                                      with how extreme the current state is.
                                      This is what shapes the RL reward.
      ``joint_exceedance_prob``     = P(X > x, Y > y)        the classical tail
                                      probability; small for extreme states.
      ``crash_probability()``       = P(X > x_c, Y > y_c)    the scenario level
                                      crash probability at the physical crash
                                      boundary, i.e. the quantity reported in the
                                      traffic-safety EVT literature.

Rationale for using ``severity`` (and not the raw tail probability) as the reward
penalty: the raw joint exceedance probability is *monotonically decreasing* in the
severity of the current state, so using ``-p`` as a penalty would punish mild
conflicts more than severe ones, and its magnitude (1e-3 .. 1e-6) is negligible
next to the collision penalty.  ``C(u1, u2)`` is the probability that a randomly
drawn conflict is jointly less severe than the current one; it is bounded in
[0, 1], monotonically increasing in severity, and therefore commensurate with the
other reward terms once multiplied by ``w_evt``.
"""

from __future__ import annotations

import json
import warnings
from collections import deque

import numpy as np

try:
    from scipy.optimize import minimize_scalar
    from scipy.stats import genpareto

    _SCIPY = True
except ImportError:  # pragma: no cover - scipy is a hard requirement for training
    _SCIPY = False


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #
def select_threshold(
    data,
    method="stability",
    quantile=0.90,
    candidates=None,
    tol=0.25,
    min_exceedances=50,
):
    """Select the POT threshold ``u`` for a 1-D sample.

    :param data: 1-D array of observations (larger = more dangerous).
    :param method: ``"quantile"``, ``"mrl"`` or ``"stability"``.

        * ``quantile``  -- fixed empirical quantile, the simplest defensible rule.
        * ``mrl``       -- mean residual life: the smallest threshold beyond which
          the mean excess is approximately linear in ``u``.
        * ``stability`` -- GPD parameter stability: the smallest threshold beyond
          which the modified scale ``sigma* = sigma_u - xi * u`` and the shape
          ``xi`` stay within ``tol`` of their values at the highest candidate.

    :param quantile: quantile used by ``"quantile"`` and as the fallback.
    :param candidates: candidate quantiles scanned by ``mrl`` / ``stability``.
    :param tol: relative tolerance for the stability criterion.
    :param min_exceedances: a threshold is only admissible if it leaves at least
        this many exceedances.
    :return: ``(threshold, diagnostics_dict)``.  ``diagnostics_dict`` carries the
        full scan so a threshold-sensitivity figure can be produced for the paper.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < min_exceedances:
        return (float(np.quantile(data, quantile)) if data.size else 0.0), {}

    if candidates is None:
        candidates = np.linspace(0.70, 0.98, 15)

    if method == "quantile":
        return float(np.quantile(data, quantile)), {"method": "quantile", "q": quantile}

    scan = []
    for q in candidates:
        u = float(np.quantile(data, q))
        exc = data[data > u] - u
        if exc.size < min_exceedances:
            continue
        entry = {"q": float(q), "u": u, "n_exc": int(exc.size), "mean_excess": float(exc.mean())}
        if _SCIPY:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    xi, _, sigma = genpareto.fit(exc, floc=0.0)
                entry["xi"] = float(xi)
                entry["sigma"] = float(sigma)
                entry["sigma_star"] = float(sigma - xi * u)
            except Exception:
                pass
        scan.append(entry)

    if not scan:
        return float(np.quantile(data, quantile)), {"method": "fallback_quantile", "q": quantile}

    if method == "mrl":
        # Pick the smallest threshold from which the mean-excess plot is linear:
        # fit a line to the remaining tail and keep the most linear starting point.
        us = np.array([e["u"] for e in scan])
        me = np.array([e["mean_excess"] for e in scan])
        best, best_err = scan[0], np.inf
        for i in range(max(len(scan) - 3, 1)):
            coef = np.polyfit(us[i:], me[i:], 1)
            err = float(np.mean((np.polyval(coef, us[i:]) - me[i:]) ** 2))
            err /= max(float(np.var(me[i:])), 1e-9)
            if err < best_err:
                best_err, best = err, scan[i]
        return best["u"], {"method": "mrl", "scan": scan, "chosen": best}

    # "stability": walk up from the lowest admissible threshold and take the first
    # one whose (xi, sigma*) already agree with the highest candidate.
    ref = scan[-1]
    if "xi" not in ref:
        return float(np.quantile(data, quantile)), {"method": "fallback_quantile", "scan": scan}
    chosen = ref
    for entry in scan:
        if "xi" not in entry:
            continue
        d_xi = abs(entry["xi"] - ref["xi"]) / max(abs(ref["xi"]), 0.1)
        d_ss = abs(entry["sigma_star"] - ref["sigma_star"]) / max(abs(ref["sigma_star"]), 1e-3)
        if d_xi <= tol and d_ss <= tol:
            chosen = entry
            break
    return chosen["u"], {"method": "stability", "scan": scan, "chosen": chosen, "tol": tol}


# --------------------------------------------------------------------------- #
# GPD margin with the classical POT tail formula
# --------------------------------------------------------------------------- #
class GPDMargin:
    """POT margin: empirical below the threshold, GPD tail above it.

    ``F(x) = 1 - zeta * (1 + xi (x - u) / sigma) ** (-1 / xi)`` for ``x > u``,
    where ``zeta = P(X > u)`` is the empirical exceedance rate.
    """

    def __init__(self):
        self.u = None
        self.xi = None
        self.sigma = None
        self.zeta = None
        self.n = 0
        self.n_exc = 0
        self._ecdf_x = None
        self.diagnostics = {}

    def fit(self, data, threshold=None, threshold_method="stability", min_exceedances=50):
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size < min_exceedances:
            return False
        if threshold is None:
            threshold, self.diagnostics = select_threshold(
                data, method=threshold_method, min_exceedances=min_exceedances
            )
        exc = data[data > threshold] - threshold
        if exc.size < min_exceedances or not _SCIPY:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xi, _, sigma = genpareto.fit(exc, floc=0.0)
        if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
            return False
        self.u, self.xi, self.sigma = float(threshold), float(xi), float(sigma)
        self.n, self.n_exc = int(data.size), int(exc.size)
        self.zeta = self.n_exc / self.n
        self._ecdf_x = np.sort(data[data <= threshold])
        return True

    @property
    def fitted(self):
        return self.u is not None

    def cdf(self, x):
        """Non-exceedance probability F(x), monotonically increasing in danger."""
        x = np.asarray(x, dtype=float)
        below = 1.0 - self.zeta
        if self._ecdf_x is not None and self._ecdf_x.size:
            rank = np.searchsorted(self._ecdf_x, x, side="right") / float(self._ecdf_x.size)
        else:
            rank = np.zeros_like(x)
        low = below * rank
        tail = 1.0 - self.zeta * self._sf_gpd(x)
        return np.where(x > self.u, tail, low)

    def cond_cdf(self, x):
        """Conditional GPD CDF ``G(x - u) = P(X <= x | X > u)``.

        Zero at or below the threshold, approaching one deep in the tail.  This is
        the transform used for the copula: fitting and scoring both operate on the
        joint exceedance set, so the same closed form is evaluated on the host and
        inside the jitted imagination rollout with no empirical component to carry.
        """
        return 1.0 - self._sf_gpd(x)

    def _sf_gpd(self, x):
        """Conditional survival P(X > x | X > u) from the fitted GPD."""
        z = np.maximum(np.asarray(x, dtype=float) - self.u, 0.0)
        if abs(self.xi) < 1e-8:
            return np.exp(-z / self.sigma)
        base = np.maximum(1.0 + self.xi * z / self.sigma, 0.0)
        return base ** (-1.0 / self.xi)

    def sf(self, x):
        """Unconditional exceedance probability P(X > x)."""
        x = np.asarray(x, dtype=float)
        return np.where(x > self.u, self.zeta * self._sf_gpd(x), 1.0 - self.cdf(x))

    def return_level(self, p):
        """Level exceeded with probability ``p`` (requires ``p < zeta``)."""
        p = float(p)
        if p <= 0 or p >= self.zeta:
            return float("nan")
        if abs(self.xi) < 1e-8:
            return self.u + self.sigma * np.log(self.zeta / p)
        return self.u + self.sigma / self.xi * ((p / self.zeta) ** (-self.xi) - 1.0)

    def state_dict(self):
        return {
            "u": self.u,
            "xi": self.xi,
            "sigma": self.sigma,
            "zeta": self.zeta,
            "n": self.n,
            "n_exc": self.n_exc,
            "diagnostics": self.diagnostics,
        }

    def load_state_dict(self, d, ecdf_x=None):
        self.u, self.xi = d["u"], d["xi"]
        self.sigma, self.zeta = d["sigma"], d["zeta"]
        self.n, self.n_exc = d.get("n", 0), d.get("n_exc", 0)
        self.diagnostics = d.get("diagnostics", {})
        self._ecdf_x = np.asarray(ecdf_x) if ecdf_x is not None else np.array([])


# --------------------------------------------------------------------------- #
# Logistic (Gumbel-Hougaard) extreme-value copula
# --------------------------------------------------------------------------- #
class LogisticEVCopula:
    """C(u1, u2; a) = exp(-((-log u1)^(1/a) + (-log u2)^(1/a))^a),  a in (0, 1].

    The bivariate logistic / Gumbel-Hougaard extreme-value copula used in the
    traffic-conflict EVT literature.  ``a -> 0`` is perfect extremal dependence,
    ``a = 1`` is independence.  The parameter is estimated by maximising the exact
    copula log-density

        log c = -S^a + (1/a - 1)(log t1 + log t2) - log u1 - log u2
                + (a - 2) log S + log(S^a + (1 - a)/a),

    with ``t_i = -log u_i`` and ``S = t1^(1/a) + t2^(1/a)``.
    """

    EPS = 1e-9

    def __init__(self, alpha=0.5):
        self.alpha = float(alpha)
        self.loglik = None
        self.n = 0

    @staticmethod
    def _prep(u1, u2):
        eps = LogisticEVCopula.EPS
        u1 = np.clip(np.asarray(u1, dtype=float), eps, 1.0 - eps)
        u2 = np.clip(np.asarray(u2, dtype=float), eps, 1.0 - eps)
        return u1, u2

    def cdf(self, u1, u2, alpha=None):
        a = self.alpha if alpha is None else alpha
        u1, u2 = self._prep(u1, u2)
        t1, t2 = -np.log(u1), -np.log(u2)
        s = t1 ** (1.0 / a) + t2 ** (1.0 / a)
        return np.exp(-(s ** a))

    def survival(self, u1, u2, alpha=None):
        """P(U1 > u1, U2 > u2) = 1 - u1 - u2 + C(u1, u2)."""
        u1, u2 = self._prep(u1, u2)
        return np.clip(1.0 - u1 - u2 + self.cdf(u1, u2, alpha), 0.0, 1.0)

    def logpdf(self, u1, u2, alpha=None):
        a = self.alpha if alpha is None else alpha
        u1, u2 = self._prep(u1, u2)
        t1, t2 = -np.log(u1), -np.log(u2)
        inv = 1.0 / a
        s = t1 ** inv + t2 ** inv
        log_s = np.log(np.maximum(s, self.EPS))
        term = np.log(np.maximum(s ** a + (1.0 - a) / a, self.EPS))
        return (
            -(s ** a)
            + (inv - 1.0) * (np.log(t1) + np.log(t2))
            - np.log(u1)
            - np.log(u2)
            + (a - 2.0) * log_s
            + term
        )

    def fit(self, u1, u2):
        u1, u2 = self._prep(u1, u2)

        def nll(a):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = -float(np.sum(self.logpdf(u1, u2, alpha=a)))
            return val if np.isfinite(val) else 1e12

        if _SCIPY:
            res = minimize_scalar(nll, bounds=(0.05, 0.999), method="bounded")
            self.alpha = float(res.x)
            self.loglik = float(-res.fun)
        else:  # coarse grid fallback
            grid = np.linspace(0.05, 0.999, 96)
            vals = [nll(a) for a in grid]
            self.alpha = float(grid[int(np.argmin(vals))])
            self.loglik = float(-min(vals))
        self.n = int(np.size(u1))
        return self


class FrankCopula:
    """Frank copula, retained only as a dependence-family sensitivity comparator.

    Frank is *not* an extreme-value copula; it is kept so the manuscript can show
    that the choice of dependence family does not drive the conclusions.
    """

    EPS = 1e-9

    def __init__(self, theta=1.0):
        self.theta = float(theta)
        self.loglik = None
        self.n = 0

    def cdf(self, u1, u2, theta=None):
        th = self.theta if theta is None else theta
        u1, u2 = LogisticEVCopula._prep(u1, u2)
        if abs(th) < 1e-8:
            return u1 * u2
        num = (np.exp(-th * u1) - 1.0) * (np.exp(-th * u2) - 1.0)
        return -1.0 / th * np.log(np.maximum(1.0 + num / (np.exp(-th) - 1.0), self.EPS))

    def survival(self, u1, u2, theta=None):
        u1, u2 = LogisticEVCopula._prep(u1, u2)
        return np.clip(1.0 - u1 - u2 + self.cdf(u1, u2, theta), 0.0, 1.0)

    def logpdf(self, u1, u2, theta=None):
        th = self.theta if theta is None else theta
        u1, u2 = LogisticEVCopula._prep(u1, u2)
        if abs(th) < 1e-8:
            return np.zeros_like(u1)
        e = np.exp(-th)
        num = th * (1.0 - e) * np.exp(-th * (u1 + u2))
        den = (1.0 - e - (1.0 - np.exp(-th * u1)) * (1.0 - np.exp(-th * u2))) ** 2
        return np.log(np.maximum(num, self.EPS)) - np.log(np.maximum(den, self.EPS))

    def fit(self, u1, u2):
        def nll(th):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = -float(np.sum(self.logpdf(u1, u2, theta=th)))
            return val if np.isfinite(val) else 1e12

        if _SCIPY:
            res = minimize_scalar(nll, bounds=(-30.0, 30.0), method="bounded")
            self.theta, self.loglik = float(res.x), float(-res.fun)
        else:
            grid = np.linspace(-30.0, 30.0, 241)
            vals = [nll(t) for t in grid]
            self.theta, self.loglik = float(grid[int(np.argmin(vals))]), float(-min(vals))
        self.n = int(np.size(u1))
        return self


_COPULAS = {"logistic": LogisticEVCopula, "gumbel": LogisticEVCopula, "frank": FrankCopula}


# --------------------------------------------------------------------------- #
# The model used by the environment and by the world model
# --------------------------------------------------------------------------- #
class CopulaEVTModel:
    """Online bivariate POT model over (-TTC, DRAC).

    :param copula: ``"logistic"`` (default, matches the manuscript) or ``"frank"``
        for the dependence-family sensitivity analysis.
    :param threshold_method: ``"stability"`` (default), ``"mrl"`` or ``"quantile"``.
    :param threshold_ttc / threshold_drac: fix the thresholds manually; when
        ``None`` (default) they are selected automatically from the buffer.
    :param risk_tolerance: ``u`` in Eq. (6) of the manuscript.  Severity at or
        below this value incurs no penalty.
    :param crash_ttc / crash_drac: the physical crash boundary used by
        :meth:`crash_probability`.  ``TTC = 0`` s and ``DRAC = 8.5 m/s^2`` (the
        deceleration a passenger car can achieve on dry asphalt) are the usual
        choices.
    :param drac_saturation: DRAC values at or above this are treated as saturated
        and excluded from the peaks-over-threshold sample.
    :param contact_ttc: TTC values at or below this count as contact rather than
        as a conflict observation, and are excluded likewise.
    """

    def __init__(
        self,
        threshold_ttc=None,
        threshold_drac=None,
        buffer_size=20000,
        min_sample=500,
        min_exceedances=50,
        copula="logistic",
        threshold_method="stability",
        risk_tolerance=0.0,
        crash_ttc=0.0,
        crash_drac=8.5,
        ttc_clip=10.0,
        drac_saturation=8.5,
        contact_ttc=0.0,
        frozen=False,
    ):
        self.threshold_ttc = threshold_ttc
        self.threshold_drac = threshold_drac
        self.buffer = deque(maxlen=buffer_size)
        self.min_sample = min_sample
        self.min_exceedances = min_exceedances
        self.copula_name = copula
        self.threshold_method = threshold_method
        self.risk_tolerance = float(risk_tolerance)
        self.crash_ttc = float(crash_ttc)
        self.crash_drac = float(crash_drac)
        self.ttc_clip = float(ttc_clip)
        self.drac_saturation = float(drac_saturation)
        self.contact_ttc = float(contact_ttc)
        self.frozen = bool(frozen)

        self.margin_ttc = GPDMargin()
        self.margin_drac = GPDMargin()
        self.copula = None
        self.n_updates = 0
        self.zeta_joint = 0.0
        self.n_joint = 0
        # Boundary observations excluded from the POT sample; reported because the
        # observed contact rate is the empirical counterpart of crash_probability().
        self.n_contact = 0
        self.n_saturated = 0

    # -- data ------------------------------------------------------------- #
    def add_sample(self, ttc, drac):
        """Buffer one observation.  TTC is negated so that larger = more dangerous.

        Three classes of observation are excluded, and the exclusions matter:

        * **No conflict** -- the calculator returns ``+inf`` when no collision
          course exists.  Nothing to model.
        * **Contact** -- ``TTC <= 0`` means the vehicles already overlap.  These
          are the events the model extrapolates *to*, not observations of the
          tail, and they are counted separately in ``n_contact``.
        * **Saturated DRAC** -- ``DRAC`` is capped at the physical deceleration
          limit, so saturated values form a point mass at the boundary.

        Keeping either boundary class in the sample is fatal for a peaks-over-
        threshold fit: the Generalised Pareto distribution then reports a shape
        parameter whose implied upper endpoint is exactly the clamp, i.e. it fits
        the clamp rather than the tail.  On the TGSIM data that produced
        ``xi = -1.84`` with ``sigma / |xi|`` equal to the clamp distance to four
        significant figures.
        """
        if self.frozen or ttc is None or drac is None:
            return
        if not (np.isfinite(ttc) and np.isfinite(drac)):
            return
        if ttc > self.ttc_clip or drac < 0:
            return
        if ttc <= self.contact_ttc:
            self.n_contact += 1
            return
        if drac >= self.drac_saturation:
            self.n_saturated += 1
            return
        self.buffer.append((-float(ttc), float(drac)))

    # -- fitting ---------------------------------------------------------- #
    def update_model(self, verbose=True):
        if self.frozen or len(self.buffer) < self.min_sample:
            return False
        arr = np.asarray(self.buffer, dtype=float)
        ok_t = self.margin_ttc.fit(
            arr[:, 0], self.threshold_ttc, self.threshold_method, self.min_exceedances
        )
        ok_d = self.margin_drac.fit(
            arr[:, 1], self.threshold_drac, self.threshold_method, self.min_exceedances
        )
        if not (ok_t and ok_d):
            if verbose:
                print("[EVT] not enough exceedances to fit the GPD margins.")
            return False

        # The copula is fitted on the *joint* exceedance set, i.e. the observations
        # that are extreme in both indicators, with each margin transformed by its
        # conditional GPD CDF.  This is the censored-likelihood practice of the
        # bivariate POT literature and keeps the transform identical to the one
        # evaluated inside latent imagination.
        joint = (arr[:, 0] > self.margin_ttc.u) & (arr[:, 1] > self.margin_drac.u)
        self.n_joint = int(joint.sum())
        self.zeta_joint = self.n_joint / float(arr.shape[0])
        if self.n_joint < self.min_exceedances:
            if verbose:
                print(f"[EVT] only {self.n_joint} joint exceedances, need {self.min_exceedances}.")
            self.copula = None
            return False

        u1 = self.margin_ttc.cond_cdf(arr[joint, 0])
        u2 = self.margin_drac.cond_cdf(arr[joint, 1])
        try:
            self.copula = _COPULAS[self.copula_name]().fit(u1, u2)
        except Exception as exc:
            print(f"[EVT] copula fitting failed: {exc}")
            self.copula = None
            return False

        self.n_updates += 1
        if verbose:
            par = getattr(self.copula, "alpha", None)
            par = par if par is not None else getattr(self.copula, "theta", float("nan"))
            print(
                f"[EVT] update #{self.n_updates} n={len(self.buffer)} | "
                f"u_ttc={self.margin_ttc.u:.3f} xi={self.margin_ttc.xi:.3f} "
                f"sigma={self.margin_ttc.sigma:.3f} n_exc={self.margin_ttc.n_exc} | "
                f"u_drac={self.margin_drac.u:.3f} xi={self.margin_drac.xi:.3f} "
                f"sigma={self.margin_drac.sigma:.3f} n_exc={self.margin_drac.n_exc} | "
                f"{self.copula_name}={par:.4f} P_crash={self.crash_probability():.3e}"
            )
        return True

    @property
    def fitted(self):
        return (
            self.copula is not None
            and self.margin_ttc.fitted
            and self.margin_drac.fitted
            and self.zeta_joint > 0.0
        )

    # -- risk read-outs --------------------------------------------------- #
    def _uniforms(self, ttc, drac):
        """Conditional GPD CDFs of the pair, restricted to the joint tail."""
        return (
            float(self.margin_ttc.cond_cdf(np.asarray(-float(ttc)))),
            float(self.margin_drac.cond_cdf(np.asarray(float(drac)))),
        )

    def severity(self, ttc, drac):
        """C(u1, u2): probability that a joint extreme is less severe than this one.

        Bounded in [0, 1], zero unless *both* indicators exceed their thresholds,
        and increasing continuously from the threshold corner into the tail.  This
        is the quantity that shapes the reward.
        """
        if not self.fitted:
            return 0.0
        if ttc is None or drac is None or not np.isfinite(ttc) or not np.isfinite(drac):
            return 0.0
        u1, u2 = self._uniforms(ttc, drac)
        if u1 <= 0.0 or u2 <= 0.0:
            return 0.0
        return float(np.clip(self.copula.cdf(u1, u2), 0.0, 1.0))

    def joint_exceedance_prob(self, ttc, drac):
        """P(-TTC > -ttc, DRAC > drac): the classical joint tail probability.

        Obtained by scaling the survival copula of the joint-exceedance model by
        the empirical rate ``zeta_joint`` at which both thresholds are exceeded.
        """
        if not self.fitted:
            return 0.0
        u1, u2 = self._uniforms(ttc, drac)
        return float(self.zeta_joint * self.copula.survival(u1, u2))

    def crash_probability(self):
        """EVT estimate of P(TTC <= crash_ttc AND DRAC >= crash_drac).

        The scenario-level crash probability reported in the traffic-safety EVT
        literature.  It depends only on the fitted model, not on the current
        state, and is the natural quantity to compare against real-world data.
        """
        if not self.fitted:
            return 0.0
        return self.joint_exceedance_prob(self.crash_ttc, self.crash_drac)

    def return_period(self, ttc, drac):
        """Expected number of observations between events at least this severe."""
        p = self.joint_exceedance_prob(ttc, drac)
        return float("inf") if p <= 0 else 1.0 / p

    # -- reward ----------------------------------------------------------- #
    def get_risk(self, ttc, drac):
        """Normalised risk in [0, 1] after applying the risk-tolerance threshold.

        Implements Eq. (6): zero at or below the tolerance ``u``, then linearly
        rescaled onto [0, 1] so the penalty magnitude does not depend on the
        chosen tolerance and stays comparable with the other reward terms.
        """
        sev = self.severity(ttc, drac)
        if sev <= self.risk_tolerance:
            return 0.0
        return float((sev - self.risk_tolerance) / max(1.0 - self.risk_tolerance, 1e-6))

    def get_evt_reward(self, ttc, drac, weight=1.0):
        return -float(weight) * self.get_risk(ttc, drac)

    # -- export to the world model ---------------------------------------- #
    #: Layout of :meth:`param_vector`.
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

    def param_vector(self):
        """Flat vector of the fitted parameters, published as an observation.

        The world model carries this vector along an imagined rollout so the EVT
        map applied to the predicted safety indicators is exactly the one fitted
        on the host, without scipy ever being called inside the jitted graph.
        Only the logistic family is exported; the Frank comparator is a host-side
        analysis, not a training-time option.
        """
        if not self.fitted:
            return np.zeros(self.PARAM_DIM, dtype=np.float32)
        alpha = getattr(self.copula, "alpha", 1.0)
        return np.array(
            [
                1.0,
                self.margin_ttc.u,
                self.margin_ttc.xi,
                self.margin_ttc.sigma,
                self.margin_drac.u,
                self.margin_drac.xi,
                self.margin_drac.sigma,
                alpha if alpha is not None else 1.0,
                self.zeta_joint,
                self.risk_tolerance,
            ],
            dtype=np.float32,
        )

    # -- persistence ------------------------------------------------------ #
    def state_dict(self):
        par = {}
        if self.copula is not None:
            par = {
                "name": self.copula_name,
                "alpha": getattr(self.copula, "alpha", None),
                "theta": getattr(self.copula, "theta", None),
                "loglik": self.copula.loglik,
            }
        return {
            "margin_ttc": self.margin_ttc.state_dict() if self.margin_ttc.fitted else None,
            "margin_drac": self.margin_drac.state_dict() if self.margin_drac.fitted else None,
            "copula": par,
            "risk_tolerance": self.risk_tolerance,
            "crash_ttc": self.crash_ttc,
            "crash_drac": self.crash_drac,
            "zeta_joint": self.zeta_joint,
            "n_joint": self.n_joint,
            "n_contact": self.n_contact,
            "n_saturated": self.n_saturated,
            "n_updates": self.n_updates,
            "n_samples": len(self.buffer),
        }

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.state_dict(), fh, indent=2)

    def load(self, path, freeze=True):
        """Load a previously fitted model.

        Used at evaluation time so the policy is scored against a *frozen* risk
        model instead of one that keeps adapting during the evaluation episodes.
        """
        with open(path) as fh:
            d = json.load(fh)
        if d.get("margin_ttc"):
            self.margin_ttc.load_state_dict(d["margin_ttc"])
        if d.get("margin_drac"):
            self.margin_drac.load_state_dict(d["margin_drac"])
        cop = d.get("copula") or {}
        if cop.get("name"):
            self.copula_name = cop["name"]
            self.copula = _COPULAS[self.copula_name]()
            if cop.get("alpha") is not None:
                self.copula.alpha = cop["alpha"]
            if cop.get("theta") is not None:
                self.copula.theta = cop["theta"]
        self.risk_tolerance = d.get("risk_tolerance", self.risk_tolerance)
        self.zeta_joint = d.get("zeta_joint", 0.0)
        self.n_joint = d.get("n_joint", 0)
        self.frozen = freeze
        return self

    # -- diagnostics for the manuscript ----------------------------------- #
    def summary(self):
        """Everything needed for the EVT diagnostics table and threshold figure."""
        return {
            "n_samples": len(self.buffer),
            "n_updates": self.n_updates,
            "ttc": self.margin_ttc.state_dict() if self.margin_ttc.fitted else None,
            "drac": self.margin_drac.state_dict() if self.margin_drac.fitted else None,
            "copula": self.copula_name,
            "zeta_joint": self.zeta_joint,
            "n_joint": self.n_joint,
            "n_contact": self.n_contact,
            "n_saturated": self.n_saturated,
            "alpha": getattr(self.copula, "alpha", None) if self.copula else None,
            "theta": getattr(self.copula, "theta", None) if self.copula else None,
            "loglik": self.copula.loglik if self.copula else None,
            "crash_probability": self.crash_probability(),
        }
