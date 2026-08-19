"""
Surrogate safety measures (TTC and DRAC) between the ego vehicle and *all*
surrounding vehicles.

The original CarDreamer ``TTCCalculator`` only considered vehicles that share the
ego vehicle's road id **and** lane id and lie ahead of it.  In a lane-change or
overtaking manoeuvre that set is empty for most of the manoeuvre, so both
indicators are undefined exactly when the conflict is most severe.  This module
replaces it with a pairwise formulation that

  1. keeps the classical car-following definitions of the manuscript
     (Eq. 1) for the same-lane leader, and
  2. generalises them to two dimensions so that a vehicle in an adjacent lane
     which the ego is encroaching upon still produces a finite TTC and DRAC.

For every vehicle within ``max_distance`` the pair indicators are evaluated and
the most critical values (minimum TTC, maximum DRAC) are returned together with
the identity and the relative role of the conflicting partner.

Both the longitudinal ("car-following") and the two-dimensional ("encroachment")
variants are exposed so the manuscript can state precisely which one is used
where, and an ablation over the choice is possible.

The core maths is pure NumPy and does not import CARLA, so it can be unit tested
and reused when fitting the same indicators on real trajectory datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Deceleration a passenger car can realise on dry asphalt; used to cap DRAC.
MAX_DECELERATION = 8.5


@dataclass
class VehicleState:
    """Minimal planar state of a vehicle, in a right-handed world frame."""

    x: float
    y: float
    vx: float
    vy: float
    yaw: float  # radians
    length: float = 4.5
    width: float = 1.9
    id: int = -1

    @property
    def pos(self):
        return np.array([self.x, self.y], dtype=float)

    @property
    def vel(self):
        return np.array([self.vx, self.vy], dtype=float)

    @property
    def speed(self):
        return float(np.hypot(self.vx, self.vy))

    @property
    def heading(self):
        return np.array([math.cos(self.yaw), math.sin(self.yaw)], dtype=float)


@dataclass
class ConflictResult:
    """Most critical conflict indicators over all surrounding vehicles."""

    ttc: float = float("inf")
    drac: float = 0.0
    partner_id: int = -1
    partner_role: str = "none"
    gap: float = float("inf")
    closing_speed: float = 0.0
    n_interacting: int = 0
    per_vehicle: list = field(default_factory=list)

    def as_dict(self):
        return {
            "ttc": self.ttc,
            "drac": self.drac,
            "conflict_partner": self.partner_id,
            "conflict_role": self.partner_role,
            "conflict_gap": self.gap,
            "closing_speed": self.closing_speed,
            "n_interacting": self.n_interacting,
        }


# --------------------------------------------------------------------------- #
# Pairwise indicators
# --------------------------------------------------------------------------- #
def longitudinal_indicators(ego: VehicleState, other: VehicleState):
    """Car-following TTC and DRAC along the ego heading (manuscript Eq. 1).

    ``DRAC = (v_FV - v_LV)^2 / (2 (x_LV - x_FV - D_LV))`` with the positions
    projected onto the ego heading and ``D_LV`` the leading vehicle length.
    Returns ``(ttc, drac, gap)``; ``ttc`` is ``inf`` and ``drac`` is ``0`` when
    the pair is not on a converging longitudinal course.
    """
    e_hat = ego.heading
    delta = other.pos - ego.pos
    s = float(np.dot(delta, e_hat))  # longitudinal spacing, >0 means ahead
    v_fv = float(np.dot(ego.vel, e_hat))
    v_lv = float(np.dot(other.vel, e_hat))

    if s > 0:  # the other vehicle leads
        gap = s - 0.5 * (ego.length + other.length)
        dv = v_fv - v_lv
    else:  # the ego vehicle leads; swap the roles
        gap = -s - 0.5 * (ego.length + other.length)
        dv = v_lv - v_fv

    if dv <= 0:  # separating
        return float("inf"), 0.0, gap
    if gap <= 0:  # already overlapping longitudinally
        return 0.0, MAX_DECELERATION, gap
    ttc = gap / dv
    drac = min(dv * dv / (2.0 * gap), MAX_DECELERATION)
    return ttc, drac, gap


def planar_indicators(ego: VehicleState, other: VehicleState, inflate=1.0):
    """Two-dimensional encroachment TTC and DRAC.

    The pair is modelled by an *elliptical* exclusion zone oriented along the ego
    heading, with semi-axes ``L = (l_ego + l_other) / 2`` longitudinally and
    ``W = (w_ego + w_other) / 2`` laterally.  Working in the normalised frame
    ``q = (ds / L, dd / W)`` turns the contact condition into ``|q| = 1``, so TTC
    is the smallest positive root of ``|q + w t| = 1``.

    A circumscribed *disc* was the obvious alternative, but its radius is
    ``sqrt(l^2 + w^2) / 2`` -- about 3.3 m for a passenger car, so two vehicles in
    adjacent lanes 3.5 m apart register as already in contact.  On dense real
    traffic that fires on the large majority of pairs and destroys the tail
    structure the EVT model is supposed to estimate.  The ellipse keeps the
    longitudinal and lateral clearances on their own scales, which is the standard
    treatment in the surrogate-safety literature.

    The orientation of the *other* vehicle is ignored, which is the usual
    approximation for near-parallel motorway and arterial traffic.

    :return: ``(ttc, drac, gap, closing_speed)``.  ``gap`` is the distance from
        the current position to the ellipse boundary along the line of centres.
    """
    e_hat = ego.heading
    n_hat = np.array([-e_hat[1], e_hat[0]])
    dp = other.pos - ego.pos
    dv = other.vel - ego.vel

    dist = float(np.linalg.norm(dp))
    if dist < 1e-6:
        return 0.0, MAX_DECELERATION, 0.0, 0.0

    semi_l = inflate * 0.5 * (ego.length + other.length)
    semi_w = inflate * 0.5 * (ego.width + other.width)
    q = np.array([float(np.dot(dp, e_hat)) / semi_l, float(np.dot(dp, n_hat)) / semi_w])
    w = np.array([float(np.dot(dv, e_hat)) / semi_l, float(np.dot(dv, n_hat)) / semi_w])

    q_norm = float(np.linalg.norm(q))
    closing = -float(np.dot(dv, dp)) / dist  # >0 when approaching
    if q_norm <= 1.0:  # already inside the exclusion zone
        return 0.0, MAX_DECELERATION, 0.0, closing

    # Physical clearance along the line of centres, out to the ellipse boundary.
    gap = dist * (q_norm - 1.0) / q_norm

    a = float(np.dot(w, w))
    b = 2.0 * float(np.dot(q, w))
    c = float(np.dot(q, q)) - 1.0
    ttc = float("inf")
    if a > 1e-12:
        disc = b * b - 4 * a * c
        if disc >= 0:
            root = math.sqrt(disc)
            for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)):
                if t > 0:
                    ttc = min(ttc, t)
    drac = 0.0
    if closing > 0 and gap > 0:
        drac = min(closing * closing / (2.0 * gap), MAX_DECELERATION)
    return ttc, drac, gap, closing


def classify_role(ego: VehicleState, other: VehicleState, lane_width=3.5):
    """Label the relative position of ``other`` with respect to ``ego``."""
    e_hat = ego.heading
    n_hat = np.array([-e_hat[1], e_hat[0]])
    delta = other.pos - ego.pos
    s = float(np.dot(delta, e_hat))
    d = float(np.dot(delta, n_hat))
    lateral = "same" if abs(d) < 0.5 * lane_width else ("left" if d > 0 else "right")
    longitudinal = "lead" if s >= 0 else "follow"
    return f"{lateral}_{longitudinal}"


# --------------------------------------------------------------------------- #
# Scene-level aggregation
# --------------------------------------------------------------------------- #
def compute_conflict(
    ego: VehicleState,
    others,
    max_distance=50.0,
    mode="max",
    lane_width=3.5,
    ttc_cap=10.0,
):
    """Aggregate the pairwise indicators over the surrounding vehicles.

    :param others: iterable of :class:`VehicleState`.
    :param max_distance: interaction radius, in metres.
    :param mode: how the longitudinal and planar variants are combined.

        * ``"max"`` (default) -- take the more critical of the two, i.e. the
          smaller TTC and the larger DRAC.  This is what the manuscript reports:
          the car-following definition governs while the ego trails the target,
          the encroachment definition takes over during the lane change.
        * ``"longitudinal"`` -- car-following definitions only (the behaviour of
          the original implementation, kept for the ablation).
        * ``"planar"`` -- two-dimensional definitions only.

    :param ttc_cap: TTC values above this are reported as ``inf`` (no conflict).
    :return: :class:`ConflictResult`.
    """
    result = ConflictResult()
    best_ttc, best_drac = float("inf"), 0.0

    for oth in others:
        if oth.id == ego.id:
            continue
        if float(np.linalg.norm(oth.pos - ego.pos)) > max_distance:
            continue
        result.n_interacting += 1

        lon_ttc, lon_drac, lon_gap = longitudinal_indicators(ego, oth)
        pl_ttc, pl_drac, pl_gap, closing = planar_indicators(ego, oth)

        if mode == "longitudinal":
            ttc, drac, gap = lon_ttc, lon_drac, lon_gap
        elif mode == "planar":
            ttc, drac, gap = pl_ttc, pl_drac, pl_gap
        else:
            ttc = min(lon_ttc, pl_ttc)
            drac = max(lon_drac, pl_drac)
            gap = min(lon_gap, pl_gap)

        role = classify_role(ego, oth, lane_width)
        result.per_vehicle.append(
            {"id": oth.id, "role": role, "ttc": ttc, "drac": drac, "gap": gap}
        )

        if ttc < best_ttc:
            best_ttc = ttc
            result.partner_id, result.partner_role = oth.id, role
            result.gap, result.closing_speed = gap, closing
        best_drac = max(best_drac, drac)

    result.ttc = best_ttc if best_ttc <= ttc_cap else float("inf")
    result.drac = best_drac
    return result


# --------------------------------------------------------------------------- #
# CARLA adapter
# --------------------------------------------------------------------------- #
def state_from_carla(actor):
    """Build a :class:`VehicleState` from a ``carla.Actor``."""
    tf = actor.get_transform()
    vel = actor.get_velocity()
    try:
        extent = actor.bounding_box.extent
        length, width = 2.0 * extent.x, 2.0 * extent.y
    except Exception:  # some blueprints expose no bounding box
        length, width = 4.5, 1.9
    return VehicleState(
        x=tf.location.x,
        y=tf.location.y,
        vx=vel.x,
        vy=vel.y,
        yaw=math.radians(tf.rotation.yaw),
        length=length,
        width=width,
        id=actor.id,
    )


class ConflictIndicatorCalculator:
    """Drop-in replacement for ``TTCCalculator`` that also yields DRAC.

    Usage inside an environment::

        conflict = ConflictIndicatorCalculator.evaluate(ego, carla_world)
        ttc, drac = conflict.ttc, conflict.drac
    """

    MAX_DISTANCE = 50.0

    @staticmethod
    def evaluate(ego_actor, carla_world, max_distance=None, mode="max", lane_width=3.5):
        max_distance = ConflictIndicatorCalculator.MAX_DISTANCE if max_distance is None else max_distance
        ego = state_from_carla(ego_actor)
        others = []
        for actor in carla_world.get_actors().filter("vehicle.*"):
            if actor.id == ego_actor.id:
                continue
            others.append(state_from_carla(actor))
        return compute_conflict(ego, others, max_distance=max_distance, mode=mode, lane_width=lane_width)

    @staticmethod
    def get_ttc(ego_actor, carla_world, carla_map=None, max_distance=None):
        """Backward-compatible helper returning only the minimum TTC."""
        return ConflictIndicatorCalculator.evaluate(ego_actor, carla_world, max_distance).ttc
