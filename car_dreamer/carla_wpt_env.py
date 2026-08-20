from abc import abstractmethod

import numpy as np
from gym import spaces

from .carla_base_env import CarlaBaseEnv
from .evt_module import CopulaEVTModel
from .toolkit import (
    BasePlanner,
    ConflictIndicatorCalculator,
    get_location_distance,
    get_vehicle_pos,
    get_vehicle_velocity,
)

# Scales used to normalise the surrogate safety measures before they are handed to
# the world model.  Keeping them explicit means the safety head of the world model
# and the EVT module always agree on the units.
TTC_HORIZON = 10.0  # s, beyond which a conflict is considered absent
DRAC_SCALE = 8.5  # m/s^2, the deceleration a passenger car can realise on dry asphalt


class CarlaWptEnv(CarlaBaseEnv):
    """
    This is the base env for all waypoint following tasks.
    An ``ego_planner`` is required to provide waypoints for the ego vehicle.
    **DO NOT** instantiate this class directly.

    All envs that inherit from this class also inherits the following config parameters:

    * ``reward``: Reward configuration.

        * ``desired_speed``: Desired speed for the ego vehicle.
        * ``scales``: Dictionary of reward scales.

            * ``waypoint``: Reward for reaching waypoints.
            * ``speed``: Reward for speed.
            * ``collision``: Penalty for collision.
            * ``out_of_lane``: Penalty for going out of lane.
            * ``time``: Penalty for each time step.
            * ``evt``: Weight of the EVT tail-risk penalty.

    * ``terminal``: Terminal condition configuration.

        * ``time_limit``: Maximum number of time steps.
        * ``out_lane_thres``: Distance threshold for going out of lane.

    * ``evt``: EVT conflict-risk configuration.

        * ``enabled``: Whether to compute the EVT penalty at all.
        * ``copula``: ``logistic`` (default) or ``frank`` for the sensitivity study.
        * ``threshold_method``: ``stability``, ``mrl`` or ``quantile``.
        * ``threshold_ttc`` / ``threshold_drac``: fixed thresholds; ``null``
          selects them automatically from the observed exceedances.
        * ``update_interval``: number of environment steps between refits.
        * ``risk_tolerance``: ``u`` in Eq. (6); severity at or below it is free.
        * ``indicator_mode``: ``max`` (car-following and encroachment combined),
          ``longitudinal`` or ``planar``.
        * ``interaction_radius``: radius within which surrounding vehicles are
          considered conflict partners.
        * ``load_from``: path to a fitted model to freeze at evaluation time.
    """

    @abstractmethod
    def get_ego_planner(self) -> BasePlanner:
        """
        Override this method to return the ego vehicle planner.
        The default behavior is to return self.ego_planner.
        """
        return self.ego_planner

    def get_state(self):
        return {"ego_waypoints": self.waypoints, "timesteps": self._time_step}

    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        self.get_ego_vehicle().apply_control(control)

    def on_step(self) -> None:
        self.waypoints, self.planner_stats = self.get_ego_planner().run_step()
        self.num_completed = self.planner_stats["num_completed"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        evt_cfg = self._config.get("evt", {}) or {}
        # ``mode`` mirrors ``dreamerv3.evt.mode``:
        #   none    -- no EVT term anywhere (the DreamerV3 ablation)
        #   env     -- penalty in the environment reward only (post hoc EVT)
        #   imagine -- penalty applied only inside latent imagination
        #   both    -- applied in both places
        self.evt_mode = evt_cfg.get("mode", "both")
        # The model is always fitted, because its parameters are published to the
        # world model even when the environment reward itself carries no penalty.
        self.evt_fit = self.evt_mode != "none"
        self.evt_enabled = evt_cfg.get("enabled", self.evt_mode in ("env", "both"))
        self.evt_update_interval = int(evt_cfg.get("update_interval", 2000))
        self.indicator_mode = evt_cfg.get("indicator_mode", "max")
        self.interaction_radius = float(evt_cfg.get("interaction_radius", 50.0))
        # Beyond this there is no interaction to speak of.  Deliberately generous:
        # the conflict threshold is chosen by the EVT model from the data, and a
        # tight cap here would starve that choice of the distribution body.
        self.ttc_cap = float(evt_cfg.get("ttc_cap", 30.0))

        self.evt_model = CopulaEVTModel(
            threshold_ttc=evt_cfg.get("threshold_ttc", None),
            threshold_drac=evt_cfg.get("threshold_drac", None),
            buffer_size=int(evt_cfg.get("buffer_size", 20000)),
            min_sample=int(evt_cfg.get("min_sample", 300)),
            min_exceedances=int(evt_cfg.get("min_exceedances", 30)),
            copula=evt_cfg.get("copula", "logistic"),
            threshold_method=evt_cfg.get("threshold_method", "stability"),
            risk_tolerance=float(evt_cfg.get("risk_tolerance", 0.0)),
            crash_drac=float(evt_cfg.get("crash_drac", DRAC_SCALE)),
            ttc_clip=self.ttc_cap,
        )
        if evt_cfg.get("load_from"):
            self.evt_model.load(evt_cfg["load_from"], freeze=True)

        # Total steps across episodes: the refit cadence has to be measured on the
        # full data stream, not on the per-episode counter.
        self._total_steps = 0
        self._conflict = None
        self._safety_vec = np.zeros(2, dtype=np.float32)

        # Expose the surrogate safety measures as an observation so the world model
        # can learn to predict them and the EVT penalty can be evaluated inside
        # latent imagination rather than only on realised transitions.
        self._observer.register_simple_handler(
            "safety",
            self._safety_observation,
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        )
        # Publish the fitted GPD/copula parameters alongside every transition so the
        # jitted imagination rollout can evaluate the same EVT map that was fitted
        # here, without SciPy ever being called on the device.
        self._observer.register_simple_handler(
            "evt_params",
            self._evt_param_observation,
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(CopulaEVTModel.PARAM_DIM,),
                dtype=np.float32,
            ),
        )
        self.observation_space = self._get_observation_space()

    # ------------------------------------------------------------------ #
    # Conflict indicators
    # ------------------------------------------------------------------ #
    def compute_conflict_indicators(self):
        """Most critical TTC and DRAC against every surrounding vehicle.

        Both indicators are evaluated pairwise against all vehicles within
        ``interaction_radius`` and the most critical values are kept, so they stay
        defined throughout a lane change instead of only while a same-lane leader
        exists.
        """
        return ConflictIndicatorCalculator.evaluate(
            self.get_ego_vehicle(),
            self._world.carla_world,
            max_distance=self.interaction_radius,
            mode=self.indicator_mode,
            ttc_cap=self.ttc_cap,
        )

    def _safety_observation(self, env_state=None):
        """Normalised ``[-TTC, DRAC]`` observation consumed by the world model.

        The first channel is ``1 - TTC / TTC_HORIZON`` clipped to ``[0, 1]``, so it
        grows with danger and saturates at 0 when no conflict exists; the second
        is ``DRAC / DRAC_SCALE``.  Both are monotone in risk and bounded, which is
        what the safety head of the world model regresses.
        """
        self._conflict = self.compute_conflict_indicators()
        ttc, drac = self._conflict.ttc, self._conflict.drac
        ttc_n = 0.0 if not np.isfinite(ttc) else float(np.clip(1.0 - ttc / TTC_HORIZON, 0.0, 1.0))
        drac_n = float(np.clip(drac / DRAC_SCALE, 0.0, 1.0))
        self._safety_vec = np.array([ttc_n, drac_n], dtype=np.float32)
        return self._safety_vec

    def _evt_param_observation(self, env_state=None):
        """Fitted EVT parameters, consumed by :mod:`dreamerv3.evt_jax`."""
        return self.evt_model.param_vector()

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def reward(self):
        reward_scales = self._config.reward.scales
        ego = self.get_ego_vehicle()
        ego_location = np.array([*get_vehicle_pos(ego)])
        ego_velocity = np.array([*get_vehicle_velocity(ego)])
        speed_norm = np.linalg.norm(ego_velocity)

        r_waypoints = reward_scales["waypoint"] if self.num_completed > 0 else 0.0

        # Speed reward
        r_speed = 0.0
        speed_parallel = 0.0
        speed_perpendicular = 0.0
        perp_direction_norm = 0.0
        if len(self.waypoints) > 0:
            next_waypoint = self.waypoints[0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]])
            yaw_radius = next_waypoint[2] * np.pi / 180
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)])
            goal_offset = next_location - ego_location
            perp_direction = goal_offset - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            perp_direction_norm = np.linalg.norm(perp_direction)
            perp_direction = perp_direction / perp_direction_norm if perp_direction_norm > 0.05 else np.array([0.0, 0.0])
            desired_speed = self._config.reward.desired_speed
            speed_parallel = np.dot(ego_velocity, waypoint_direction)
            speed_perpendicular = np.dot(ego_velocity, perp_direction)
            r_speed = (desired_speed - np.abs(speed_parallel - desired_speed) - 2 * max(speed_perpendicular, -0.5)) * reward_scales["speed"]

        r_collision = -reward_scales["collision"] * np.abs(speed_norm) if reward_scales["collision"] > 0 and self.is_collision() else 0.0

        r_out_of_lane = -reward_scales["out_of_lane"] * (perp_direction_norm - 0.5) if perp_direction_norm > 0.5 else 0.0

        r_destination = reward_scales["destination_reached"] if self.is_destination_reached() else 0.0

        time_penalty = -reward_scales["time"]

        # EVT tail-risk penalty over the joint (TTC, DRAC) extremes.
        conflict = self._conflict if self._conflict is not None else self.compute_conflict_indicators()
        ttc, drac = conflict.ttc, conflict.drac

        r_evt = 0.0
        evt_severity = 0.0
        evt_tail_prob = 0.0
        if self.evt_fit:
            # The refit cadence is counted over the whole data stream rather than
            # the per-episode step counter, so it does not reset every episode.
            self._total_steps += 1
            self.evt_model.add_sample(ttc, drac)
            if self.evt_update_interval > 0 and self._total_steps % self.evt_update_interval == 0:
                if self.evt_model.update_model() and self.evt_model.margin_ttc.fitted:
                    # The safety observation saturates at TTC_HORIZON, so a fitted
                    # threshold beyond it can never be exceeded by the world
                    # model's prediction and the imagination-side penalty would be
                    # permanently zero while the environment-side one fires.
                    thr = -self.evt_model.margin_ttc.u
                    if thr > TTC_HORIZON:
                        print(
                            f"[EVT] warning: fitted TTC threshold {thr:.2f}s exceeds "
                            f"the {TTC_HORIZON:.0f}s observation horizon; the "
                            "imagination-side penalty cannot fire. Raise TTC_HORIZON "
                            "or tighten evt.threshold_method."
                        )
            evt_severity = self.evt_model.severity(ttc, drac)
            evt_tail_prob = self.evt_model.joint_exceedance_prob(ttc, drac)
            if self.evt_enabled:
                r_evt = self.evt_model.get_evt_reward(
                    ttc, drac, weight=reward_scales.get("evt", 1.0)
                )

        total_reward = r_waypoints + r_speed + r_collision + r_out_of_lane + r_destination + time_penalty + r_evt

        info = {
            **self.planner_stats,
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": speed_parallel,
            "speed_perpendicular": speed_perpendicular,
            "speed_norm": speed_norm,
            "wpt_dis": self.get_wpt_dist(ego_location),
            "r_waypoints": r_waypoints,
            "r_speed": r_speed,
            "r_collision": r_collision,
            "r_out_of_lane": r_out_of_lane,
            "r_evt": r_evt,
            "ttc": ttc if np.isfinite(ttc) else TTC_HORIZON,
            "drac": drac,
            "evt_severity": evt_severity,
            "evt_tail_prob": evt_tail_prob,
            "evt_crash_prob": self.evt_model.crash_probability() if self.evt_fit else 0.0,
            **conflict.as_dict(),
        }

        return total_reward, info

    def is_destination_reached(self):
        return len(self.waypoints) <= 3

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        ego_location = get_vehicle_pos(self.get_ego_vehicle())
        conds = {
            "is_collision": self.is_collision(),
            "time_exceeded": self._time_step > terminal_config.time_limit,
            "out_of_lane": self.get_wpt_dist(ego_location) > terminal_config.out_lane_thres,
            "destination_reached": self.is_destination_reached(),
        }
        return conds

    def get_wpt_dist(self, ego_location):
        if len(self.waypoints) == 0:
            return 0
        else:
            return get_location_distance(ego_location, self.waypoints[0])

    def evt_summary(self):
        """Fitted EVT parameters, for the diagnostics table in the manuscript."""
        return self.evt_model.summary()
