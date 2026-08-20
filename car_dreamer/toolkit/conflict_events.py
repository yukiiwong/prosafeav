"""
Injected pre-crash scenarios, so that safety-critical events actually occur.

Free-flowing IDM/MOBIL traffic is realistic but placid: vehicles keep safe
headways by construction, so an episode may pass without a single conflict.  That
is fatal for this framework specifically -- with no exceedances there is nothing
for the peaks-over-threshold model to fit, the EVT term stays inert, and the
agent is trained by an ordinary reward while the manuscript claims otherwise.

The fix used in scenario-based safety assessment is to inject the manoeuvres that
actually precede crashes.  The event types here follow the pre-crash scenario
typology used for light-vehicle crashes (Najm et al., NHTSA, 2007), restricted to
the rear-end and lane-change families that a highway overtaking task can express:

    lead_brake      the vehicle ahead decelerates hard              (LVD)
    stopped_vehicle a stationary vehicle blocks the lane            (LVS)
    cut_in          a neighbour changes into the ego's lane at a
                    short gap                                       (LC)

Each event is sampled per episode with its own probability and randomised
trigger distance, duration and intensity, so the resulting conflicts are varied
rather than a fixed scripted sequence -- the criticism the original single
sinusoidal "swing" manoeuvre attracted.

Events are transient overrides of the IDM/MOBIL controllers, not replacements:
outside the trigger window the background vehicles behave normally, so the
traffic remains calibrated while still producing a usable tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EventSpec:
    """One scheduled event and the state needed to run it."""

    kind: str
    trigger_distance: float          # fires when the ego is this close (m)
    duration_steps: int
    intensity: float                 # brake fraction, or lateral gain for cut-in
    target_id: int = -1              # which background vehicle performs it
    fired: bool = False
    steps_left: int = 0
    fire_step: int = -1

    def as_dict(self):
        return {
            "kind": self.kind,
            "trigger_distance": self.trigger_distance,
            "duration_steps": self.duration_steps,
            "intensity": self.intensity,
            "target_id": self.target_id,
            "fired": self.fired,
            "fire_step": self.fire_step,
        }


class ConflictEventScheduler:
    """Samples and runs the injected pre-crash events for one episode.

    :param config: mapping with the per-event probabilities and ranges, e.g.

        ``{"lead_brake": {"prob": 0.5, "trigger_distance": [12, 30],
                          "duration": [8, 20], "intensity": [0.6, 1.0]},
           "cut_in": {...}, "stopped_vehicle": {...}}``

    :param rng: numpy generator, so a scenario seed reproduces the episode.
    :param dt: simulation timestep, used to convert durations in seconds.
    """

    DEFAULTS = {
        "lead_brake": {
            "prob": 0.5,
            "trigger_distance": [12.0, 30.0],
            "duration": [0.8, 2.0],      # seconds
            "intensity": [0.6, 1.0],     # brake fraction
        },
        "cut_in": {
            "prob": 0.4,
            "trigger_distance": [8.0, 22.0],
            "duration": [1.5, 3.0],
            "intensity": [0.6, 1.2],     # lateral steering gain
        },
        "stopped_vehicle": {
            "prob": 0.25,
            "trigger_distance": [0.0, 0.0],   # placed at reset, no trigger
            "duration": [0.0, 0.0],
            "intensity": [0.0, 0.0],
        },
    }

    def __init__(self, config=None, rng=None, dt=0.1):
        self.rng = np.random.default_rng() if rng is None else rng
        self.dt = float(dt)
        cfg = dict(self.DEFAULTS)
        for key, value in (config or {}).items():
            if key in cfg and isinstance(value, dict):
                merged = dict(cfg[key])
                merged.update(value)
                cfg[key] = merged
            else:
                cfg[key] = value
        self.config = cfg
        self.events: list[EventSpec] = []
        self.active_overrides: dict[int, EventSpec] = {}
        self.n_fired = 0

    # ------------------------------------------------------------------ #
    def sample_episode(self, candidate_ids, lead_id=None):
        """Draw this episode's events.  Call once per reset.

        :param candidate_ids: ids of background vehicles that may perform a
            cut-in.
        :param lead_id: id of the vehicle directly ahead of the ego, which is the
            only sensible actor for a lead-brake event.
        """
        self.events = []
        self.active_overrides = {}
        self.n_fired = 0

        for kind, spec in self.config.items():
            if kind not in self.DEFAULTS:
                continue
            if float(self.rng.random()) >= float(spec.get("prob", 0.0)):
                continue
            if kind == "lead_brake":
                if lead_id is None:
                    continue
                target = lead_id
            elif kind == "cut_in":
                ids = [i for i in candidate_ids if i != lead_id]
                if not ids:
                    continue
                target = int(self.rng.choice(ids))
            else:  # stopped_vehicle is realised at spawn time
                target = -1

            dur = self.rng.uniform(*spec["duration"])
            self.events.append(
                EventSpec(
                    kind=kind,
                    trigger_distance=float(self.rng.uniform(*spec["trigger_distance"])),
                    duration_steps=max(1, int(round(dur / self.dt))),
                    intensity=float(self.rng.uniform(*spec["intensity"])),
                    target_id=int(target),
                )
            )
        return self.events

    def wants_stopped_vehicle(self):
        return any(e.kind == "stopped_vehicle" for e in self.events)

    # ------------------------------------------------------------------ #
    def update(self, step, ego_state, states):
        """Advance the scheduler.  Returns ``{vehicle_id: EventSpec}`` overrides.

        :param step: current episode timestep.
        :param ego_state: :class:`VehicleState` of the ego vehicle.
        :param states: ``{id: VehicleState}`` of the background vehicles.
        """
        # Expire running overrides.
        for vid in list(self.active_overrides):
            event = self.active_overrides[vid]
            event.steps_left -= 1
            if event.steps_left <= 0:
                del self.active_overrides[vid]

        for event in self.events:
            if event.fired or event.kind == "stopped_vehicle":
                continue
            state = states.get(event.target_id)
            if state is None:
                continue
            distance = float(np.hypot(state.x - ego_state.x, state.y - ego_state.y))
            if distance > event.trigger_distance:
                continue
            # Only fire on an actor the ego is approaching, otherwise a vehicle
            # that is already behind would brake for no reason.
            approaching = (
                (state.vx - ego_state.vx) * (state.x - ego_state.x)
                + (state.vy - ego_state.vy) * (state.y - ego_state.y)
            ) < 0
            if not approaching and event.kind == "lead_brake":
                continue
            event.fired = True
            event.fire_step = int(step)
            event.steps_left = event.duration_steps
            self.active_overrides[event.target_id] = event
            self.n_fired += 1

        return self.active_overrides

    def override_for(self, vehicle_id):
        return self.active_overrides.get(vehicle_id)

    def summary(self):
        return {
            "n_events": len(self.events),
            "n_fired": self.n_fired,
            "events": [e.as_dict() for e in self.events],
        }


def apply_override(event: EventSpec, acc, target_x, ego_x, lane_width=3.4):
    """Bend one IDM/MOBIL command according to an active event.

    :return: ``(acc, target_x)`` after the override.
    """
    if event is None:
        return acc, target_x
    if event.kind == "lead_brake":
        # A hard deceleration, expressed on the same scale the environment
        # converts into throttle and brake.
        return -3.0 * event.intensity, target_x
    if event.kind == "cut_in":
        # Steer toward the ego's lane, keeping speed so the gap closes laterally
        # rather than by braking.
        return acc, float(ego_x)
    return acc, target_x
