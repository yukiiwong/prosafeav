"""
Behavioural models for background traffic: IDM car-following and MOBIL lane
changing.

The original overtaking scenario drove its single background vehicle with a
hand-tuned sinusoidal steering "swing" plus a PID lane keeper.  That produces a
repeatable but physically unmotivated trajectory, and gives no way to reason
about whether the surrounding traffic behaves like real traffic.  This module
replaces it with the two models that the traffic-flow literature calibrates
against real trajectory data:

  * **IDM** (Intelligent Driver Model, Treiber, Hennecke & Helbing, 2000) for
    longitudinal control, and
  * **MOBIL** (Kesting, Treiber & Helbing, 2007) for discretionary and mandatory
    lane changes.

Default parameter ranges follow published calibrations on the highD and NGSIM
datasets, so the manuscript can state the provenance of every number rather than
"a PID controller".  Each background vehicle draws its own parameters at spawn
time, which is what produces heterogeneous, non-deterministic traffic.

The models are written against the plain :class:`VehicleState` used by
``conflict.py`` so they can be exercised without CARLA, and re-used to simulate
the same traffic in an offline validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------- #
# Driver parameters
# --------------------------------------------------------------------------- #
@dataclass
class IDMParams:
    """IDM parameters.  Defaults are the highD-calibrated passenger-car values.

    :param v0: desired free-flow speed (m/s).
    :param T: desired time headway (s).
    :param s0: minimum bumper-to-bumper spacing at standstill (m).
    :param a: maximum acceleration (m/s^2).
    :param b: comfortable deceleration (m/s^2).
    :param delta: free-acceleration exponent.
    """

    v0: float = 12.0
    T: float = 1.4
    s0: float = 2.0
    a: float = 1.4
    b: float = 2.0
    delta: float = 4.0

    @staticmethod
    def sample(rng=None, v0_mean=12.0, v0_std=1.5, aggressive_frac=0.25):
        """Draw a heterogeneous driver.

        ``aggressive_frac`` of the drivers get a short headway and a high desired
        speed, which is what generates the rare close-following events the EVT
        model needs; the remainder are sampled around the calibrated means.
        """
        rng = np.random.default_rng() if rng is None else rng
        aggressive = rng.random() < aggressive_frac
        if aggressive:
            return IDMParams(
                v0=float(rng.normal(v0_mean * 1.15, v0_std)),
                T=float(np.clip(rng.normal(0.9, 0.15), 0.5, 1.3)),
                s0=float(np.clip(rng.normal(1.5, 0.3), 0.8, 2.5)),
                a=float(np.clip(rng.normal(1.9, 0.3), 1.0, 3.0)),
                b=float(np.clip(rng.normal(2.6, 0.4), 1.5, 4.0)),
            )
        return IDMParams(
            v0=float(rng.normal(v0_mean, v0_std)),
            T=float(np.clip(rng.normal(1.5, 0.3), 0.8, 2.5)),
            s0=float(np.clip(rng.normal(2.2, 0.4), 1.0, 4.0)),
            a=float(np.clip(rng.normal(1.3, 0.25), 0.6, 2.2)),
            b=float(np.clip(rng.normal(2.0, 0.3), 1.2, 3.0)),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class MOBILParams:
    """MOBIL parameters.

    :param politeness: weight ``p`` given to the acceleration of the affected
        neighbours.  ``0`` is purely selfish, ``0.5`` is the calibrated value for
        European motorway traffic.
    :param delta_a_th: switching threshold (m/s^2).
    :param b_safe: maximum deceleration that may be imposed on the prospective
        follower (m/s^2).
    :param bias_right: keep-right bias (m/s^2); ``0`` disables it.
    """

    politeness: float = 0.35
    delta_a_th: float = 0.1
    b_safe: float = 4.0
    bias_right: float = 0.0
    cooldown: float = 2.0  # seconds between two lane changes

    @staticmethod
    def sample(rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return MOBILParams(
            politeness=float(np.clip(rng.normal(0.35, 0.15), 0.0, 0.8)),
            delta_a_th=float(np.clip(rng.normal(0.15, 0.08), 0.02, 0.4)),
            b_safe=float(np.clip(rng.normal(4.0, 0.8), 2.0, 6.0)),
        )

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# IDM
# --------------------------------------------------------------------------- #
def idm_acceleration(v, gap, dv, params: IDMParams):
    """IDM acceleration.

    :param v: own speed (m/s).
    :param gap: bumper-to-bumper spacing to the leader (m); ``inf`` if none.
    :param dv: approach rate ``v - v_leader`` (m/s), positive when closing.
    :param params: :class:`IDMParams`.
    :return: acceleration in m/s^2.
    """
    v = max(float(v), 0.0)
    free = 1.0 - (v / max(params.v0, 1e-3)) ** params.delta
    if not math.isfinite(gap):
        return params.a * free
    gap = max(float(gap), 1e-2)
    s_star = params.s0 + max(
        0.0, v * params.T + v * dv / (2.0 * math.sqrt(max(params.a * params.b, 1e-6)))
    )
    return params.a * (free - (s_star / gap) ** 2)


# --------------------------------------------------------------------------- #
# MOBIL
# --------------------------------------------------------------------------- #
def mobil_decision(
    own_speed,
    current: dict,
    candidate: dict,
    idm: IDMParams,
    mobil: MOBILParams,
    right_lane=False,
):
    """Decide whether a lane change to ``candidate`` is worthwhile and safe.

    ``current`` and ``candidate`` each describe one lane as

        ``{"lead_gap": float, "lead_dv": float,
           "follow_gap": float, "follow_dv": float, "follow_speed": float}``

    where ``*_dv`` is the approach rate seen by the *following* party of that
    pair (positive when closing) and gaps are bumper-to-bumper in metres.
    ``inf`` marks an absent neighbour.

    :return: ``(change: bool, incentive: float, safe: bool)``.
    """
    a_cur = idm_acceleration(own_speed, current["lead_gap"], current["lead_dv"], idm)
    a_new = idm_acceleration(own_speed, candidate["lead_gap"], candidate["lead_dv"], idm)

    # Safety: the prospective follower must not be forced to brake harder than b_safe.
    a_new_follower = idm_acceleration(
        candidate.get("follow_speed", own_speed),
        candidate["follow_gap"],
        candidate["follow_dv"],
        idm,
    )
    safe = a_new_follower >= -mobil.b_safe
    if not safe:
        return False, 0.0, False

    # Politeness term: how the change affects the old and the new follower.
    a_old_follower_before = idm_acceleration(
        current.get("follow_speed", own_speed), current["follow_gap"], current["follow_dv"], idm
    )
    # After the change the old follower inherits the gap to the old leader.
    old_follow_gap_after = current["follow_gap"] + current["lead_gap"]
    a_old_follower_after = idm_acceleration(
        current.get("follow_speed", own_speed), old_follow_gap_after, current["follow_dv"], idm
    )
    a_new_follower_before = idm_acceleration(
        candidate.get("follow_speed", own_speed),
        candidate["follow_gap"] + candidate["lead_gap"],
        candidate["follow_dv"],
        idm,
    )

    incentive = (
        (a_new - a_cur)
        + mobil.politeness
        * ((a_new_follower - a_new_follower_before) + (a_old_follower_after - a_old_follower_before))
        + (mobil.bias_right if right_lane else 0.0)
    )
    return bool(incentive > mobil.delta_a_th), float(incentive), True


# --------------------------------------------------------------------------- #
# Neighbour extraction
# --------------------------------------------------------------------------- #
def lane_neighbours(ego, others, lane_centre_x, lane_half_width, forward_axis=-1):
    """Find the leader and the follower of ``ego`` within one lane.

    The overtaking map (CARLA Town04) runs along the world ``y`` axis with the
    vehicles heading in the negative ``y`` direction, so ``forward_axis = -1``
    means "smaller y is further ahead".  Lanes are identified by their centre
    ``x`` coordinate.

    :return: ``(leader, follower)``; either may be ``None``.
    """
    leader, follower = None, None
    best_lead, best_follow = float("inf"), float("inf")
    for oth in others:
        if oth.id == ego.id:
            continue
        if abs(oth.x - lane_centre_x) > lane_half_width:
            continue
        ds = forward_axis * (oth.y - ego.y)  # >0 means ahead of the ego
        if ds > 0 and ds < best_lead:
            best_lead, leader = ds, oth
        elif ds < 0 and -ds < best_follow:
            best_follow, follower = -ds, oth
    return leader, follower


def pair_gap_dv(follower, leader, forward_axis=-1):
    """Bumper-to-bumper gap and approach rate for one follower/leader pair."""
    if follower is None or leader is None:
        return float("inf"), 0.0
    ds = forward_axis * (leader.y - follower.y)
    gap = ds - 0.5 * (follower.length + leader.length)
    v_f = forward_axis * follower.vy
    v_l = forward_axis * leader.vy
    return max(gap, 0.0), v_f - v_l


class IDMMobilController:
    """Per-vehicle IDM + MOBIL controller with its own sampled driver parameters.

    The controller is deliberately stateless apart from the lane-change cooldown
    so it can be driven step by step from the environment.
    """

    def __init__(self, lane_centres, rng=None, v0_mean=12.0, aggressive_frac=0.25,
                 lane_half_width=1.75, forward_axis=-1, dt=0.1):
        self.rng = np.random.default_rng() if rng is None else rng
        self.idm = IDMParams.sample(self.rng, v0_mean=v0_mean, aggressive_frac=aggressive_frac)
        self.mobil = MOBILParams.sample(self.rng)
        self.lane_centres = list(lane_centres)
        self.lane_half_width = lane_half_width
        self.forward_axis = forward_axis
        self.dt = dt
        self.target_lane = None
        self._cooldown = 0.0

    def current_lane(self, state):
        return int(np.argmin([abs(state.x - c) for c in self.lane_centres]))

    def _lane_context(self, state, others, lane_idx):
        centre = self.lane_centres[lane_idx]
        leader, follower = lane_neighbours(
            state, others, centre, self.lane_half_width, self.forward_axis
        )
        lead_gap, lead_dv = pair_gap_dv(state, leader, self.forward_axis)
        follow_gap, follow_dv = pair_gap_dv(follower, state, self.forward_axis)
        return {
            "lead_gap": lead_gap,
            "lead_dv": lead_dv,
            "follow_gap": follow_gap,
            "follow_dv": follow_dv,
            "follow_speed": abs(follower.vy) if follower is not None else state.speed,
        }

    def step(self, state, others):
        """Return ``(acceleration, target_lane_centre_x, diagnostics)``."""
        self._cooldown = max(0.0, self._cooldown - self.dt)
        lane = self.current_lane(state)
        ctx = self._lane_context(state, others, lane)
        acc = idm_acceleration(state.speed, ctx["lead_gap"], ctx["lead_dv"], self.idm)

        chosen, best_incentive = lane, 0.0
        if self._cooldown <= 0.0:
            for cand in (lane - 1, lane + 1):
                if not 0 <= cand < len(self.lane_centres):
                    continue
                cand_ctx = self._lane_context(state, others, cand)
                change, incentive, _ = mobil_decision(
                    state.speed, ctx, cand_ctx, self.idm, self.mobil, right_lane=cand > lane
                )
                if change and incentive > best_incentive:
                    chosen, best_incentive = cand, incentive
            if chosen != lane:
                self._cooldown = self.mobil.cooldown

        self.target_lane = chosen
        diagnostics = {
            "lane": lane,
            "target_lane": chosen,
            "lead_gap": ctx["lead_gap"],
            "lead_dv": ctx["lead_dv"],
            "incentive": best_incentive,
            "acc": acc,
        }
        return float(acc), float(self.lane_centres[chosen]), diagnostics

    def describe(self):
        return {"idm": self.idm.to_dict(), "mobil": self.mobil.to_dict()}
