import math

import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedEndingPlanner, get_vehicle_pos
from .toolkit.carla_manager.conflict import state_from_carla
from .toolkit.conflict_events import ConflictEventScheduler, apply_override
from .toolkit.traffic_models import IDMMobilController


class CarlaOvertakeEnv(CarlaWptEnv):
    """
    This task places slower traffic in front of the ego vehicle for overtaking.

    **Provided Tasks**: ``carla_overtake``, ``carla_overtake_d{05,15,30,45}``

    Background traffic can be driven in one of two ways, selected with
    ``background_controller``:

    * ``idm_mobil`` (default) -- every background vehicle follows the Intelligent
      Driver Model longitudinally and MOBIL laterally, with driver parameters
      drawn per vehicle from the highD/NGSIM-calibrated distributions in
      :mod:`car_dreamer.toolkit.traffic_models`.  Traffic density, the number of
      vehicles, their initial gaps and their desired speeds are randomised each
      episode, which is what makes the scenario stochastic and lets the same task
      be instantiated at several densities.
    * ``swing`` -- the original single-vehicle behaviour: one background vehicle
      holding a constant speed, kept in lane by a PID controller and made to
      swing sinusoidally once the ego vehicle closes in.  Retained so the
      previously reported results remain reproducible.

    Available config parameters:

    * ``background_controller``: ``idm_mobil`` or ``swing``.
    * ``traffic_density``: background vehicles per kilometre per lane.  Used when
      ``num_background_vehicles`` is not given.
    * ``num_background_vehicles``: fixed count, or ``[min, max]`` to randomise.
    * ``lane_centres``: x coordinates of the lane centres.
    * ``background_speed_range``: ``[min, max]`` desired speed (m/s) of background
      traffic; each vehicle draws its own IDM ``v0`` around a value in this range.
    * ``spawn_gap_range``: ``[min, max]`` initial longitudinal gap (m).
    * ``aggressive_fraction``: share of background drivers using short headways.
    * ``swing_steer``: The background vehicle steer for swing.
    * ``swing_amplitude``: The y-axis amplitude of background vehicle steer.
    * ``swing_trigger_dist``: The distance between ego and background vehicle that triggers swing.
    * ``pid_coeffs``: The PID controller parameter for background vehicle lane keeping.
    * ``reward_overtake_dist``: The distance from background vehicle to ego vehicle that triggers overtake reward.
    * ``early_lane_change_dist``: The distance that penalizes early lane change.
    * ``lane_width``: The width of the lane.
    * ``stay_same_lane``: The penalty for stay in the same lane when approaching the background vehicle.
    * ``overtake``: The reward for overtaking.
    * ``early_lane_change``: The reward for early lane change.

    """

    # The road runs along -y in Town04, so "ahead" means a smaller y coordinate.
    FORWARD_AXIS = -1

    # ------------------------------------------------------------------ #
    # Scenario construction
    # ------------------------------------------------------------------ #
    def _lane_centres(self):
        centres = self._config.get("lane_centres")
        if centres:
            return [float(c) for c in centres]
        return [float(p[0]) for p in self._config.lane_start_points]

    def _num_background_vehicles(self, road_length_m, n_lanes):
        cfg_n = self._config.get("num_background_vehicles")
        if cfg_n is not None:
            if isinstance(cfg_n, (list, tuple)):
                return int(self._rng.integers(int(cfg_n[0]), int(cfg_n[1]) + 1))
            return int(cfg_n)
        density = float(self._config.get("traffic_density", 15.0))  # veh/km/lane
        expected = density * (road_length_m / 1000.0) * n_lanes
        # Poisson arrivals give a realistic spread of vehicle counts at a target density.
        return int(max(1, self._rng.poisson(max(expected, 1e-3))))

    def on_reset(self) -> None:
        assert self._config.get("lane_start_points"), "Missing lane_start_points in config"
        seed = self._config.get("scenario_seed")
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        self.background_controller = self._config.get("background_controller", "idm_mobil")
        self.lane_centres = self._lane_centres()

        # The target vehicle the ego has to overtake, kept as ``self.nonego`` so the
        # overtaking reward terms are unchanged.
        self.nonego_spawn_point = self._config.nonego_spawn_points[
            int(self._rng.integers(len(self._config.nonego_spawn_points)))
        ]
        nonego_transform = carla.Transform(
            carla.Location(*self.nonego_spawn_point[:3]),
            carla.Rotation(*self.nonego_spawn_point[-3:]),
        )
        self.nonego = self._world.spawn_actor(transform=nonego_transform)

        self.ego_src = self._config.lane_start_points[
            int(self._rng.integers(len(self._config.lane_start_points)))
        ]

        # Randomise where the ego vehicle starts relative to the vehicle it has to
        # overtake.  Previously the pair always began in the same lane exactly
        # 20 m apart, so the policy could learn a single fixed manoeuvre; reviewer
        # 1 comment 6 asks precisely about this.  The gap is drawn from
        # ``ego_gap_range`` and, with probability ``ego_offset_lane_prob``, the ego
        # starts one lane over so it must merge back before overtaking.
        gap_range = self._config.get("ego_gap_range", [20.0, 20.0])
        gap = float(self._rng.uniform(*gap_range))
        ego_y = self.nonego_spawn_point[1] + gap  # +y is behind, the road runs along -y

        ego_x = self.nonego_spawn_point[0]
        self.ego_start_lane_offset = 0
        if float(self._rng.random()) < float(self._config.get("ego_offset_lane_prob", 0.0)):
            lane_idx = int(np.argmin([abs(ego_x - c) for c in self.lane_centres]))
            choices = [i for i in (lane_idx - 1, lane_idx + 1) if 0 <= i < len(self.lane_centres)]
            if choices:
                new_idx = int(self._rng.choice(choices))
                self.ego_start_lane_offset = new_idx - lane_idx
                ego_x = self.lane_centres[new_idx]

        ego_transform = carla.Transform(
            carla.Location(x=float(ego_x), y=float(ego_y), z=self.ego_src[2]),
            carla.Rotation(yaw=-90),
        )  # always behind the vehicle to be overtaken
        self.ego = self._world.spawn_actor(transform=ego_transform)
        self.initial_gap = gap

        # A non-zero initial speed avoids every episode starting from standstill,
        # which would make the early conflict distribution unrepresentative.
        v0_range = self._config.get("ego_initial_speed_range", [0.0, 0.0])
        v_init = float(self._rng.uniform(*v0_range))
        if v_init > 0:
            try:
                self.ego.set_target_velocity(carla.Vector3D(0.0, self.FORWARD_AXIS * v_init, 0.0))
            except Exception:
                pass

        self.background_vehicles = []
        self.background_controllers = {}
        self.traffic_profile = {}
        self.stopped_vehicle_id = None

        # Injected pre-crash events.  Free-flowing IDM/MOBIL traffic keeps safe
        # headways by construction and can run a whole episode without a single
        # conflict, which leaves the EVT model with nothing to fit.
        self.events = ConflictEventScheduler(
            config=self._config.get("conflict_events", {}) or {},
            rng=np.random.default_rng(self._rng.integers(0, 2**31 - 1)),
            dt=float(self._config.world.fixed_delta_seconds),
        )
        if self.background_controller == "idm_mobil":
            self._spawn_idm_traffic()
            self.events.sample_episode(
                candidate_ids=[v.id for v in self.background_vehicles],
                lead_id=self.nonego.id,
            )
            if self.events.wants_stopped_vehicle():
                self._spawn_stopped_vehicle()

        # Path planning
        ego_dest = self._config.lane_end_points
        dest_location = carla.Location(x=self.nonego_spawn_point[0], y=ego_dest[0][1], z=ego_dest[0][2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        self.exceeding = False
        self.overtake = False
        self.last_ego_y = self.ego_src[1]

        # Set spectator for debugging
        spectator = self._world._world.get_spectator()
        ego_transform.location.z += 150
        ego_transform.rotation.pitch = -70
        spectator.set_transform(ego_transform)
        self.swing_direction = 1

        self.prev_errors = {"last_error": 0.0, "integral": 0.0}  # For PID controller
        self._bg_prev_errors = {}

    def _spawn_idm_traffic(self):
        """Populate the road with heterogeneous IDM/MOBIL background traffic."""
        y_start = float(self.ego_src[1])
        y_end = float(self._config.lane_end_points[0][1])
        road_length = abs(y_start - y_end)
        speed_range = self._config.get("background_speed_range", [6.0, 12.0])
        gap_range = self._config.get("spawn_gap_range", [12.0, 45.0])
        aggressive = float(self._config.get("aggressive_fraction", 0.25))
        dt = float(self._config.world.fixed_delta_seconds)

        n_target = self._num_background_vehicles(road_length, len(self.lane_centres))

        # Lay the vehicles out lane by lane with randomised gaps, skipping the slot
        # immediately behind the ego vehicle so the episode never starts in a crash.
        candidates = []
        for lane_x in self.lane_centres:
            y = min(y_start, self.nonego_spawn_point[1]) - float(self._rng.uniform(*gap_range))
            while y > y_end + 10.0:
                candidates.append((lane_x, y))
                y -= float(self._rng.uniform(*gap_range))
        self._rng.shuffle(candidates)

        for lane_x, y in candidates:
            if len(self.background_vehicles) >= n_target:
                break
            if abs(lane_x - self.nonego_spawn_point[0]) < 1.0 and abs(y - self.nonego_spawn_point[1]) < 12.0:
                continue  # too close to the overtaking target
            transform = carla.Transform(
                carla.Location(x=float(lane_x), y=float(y), z=0.1),
                carla.Rotation(yaw=-90.0),
            )
            actor = self._world.spawn_actor(transform=transform)
            if actor is None:
                continue
            self.background_vehicles.append(actor)

        # The overtaking target is also IDM driven, but with a deliberately low
        # desired speed so that overtaking remains the rational manoeuvre.
        all_bg = [self.nonego] + self.background_vehicles
        for idx, actor in enumerate(all_bg):
            v0_mean = (
                float(self._config.get("target_speed", 4.0))
                if actor is self.nonego
                else float(self._rng.uniform(*speed_range))
            )
            ctrl = IDMMobilController(
                lane_centres=self.lane_centres,
                rng=np.random.default_rng(self._rng.integers(0, 2**31 - 1)),
                v0_mean=v0_mean,
                aggressive_frac=0.0 if actor is self.nonego else aggressive,
                lane_half_width=float(self._config.get("lane_width", 3.4)) / 2.0,
                forward_axis=self.FORWARD_AXIS,
                dt=dt,
            )
            if actor is self.nonego:
                # The target keeps its lane; the manoeuvre under test is the ego's.
                ctrl.mobil.delta_a_th = 1e9
            self.background_controllers[actor.id] = ctrl
            # Give every vehicle a running start so the traffic state is not
            # transient at the beginning of the episode.
            v_init = min(ctrl.idm.v0, float(self._rng.uniform(*speed_range)))
            try:
                actor.set_target_velocity(carla.Vector3D(0.0, self.FORWARD_AXIS * v_init, 0.0))
            except Exception:
                pass

        self.traffic_profile = {
            "n_background": len(self.background_vehicles),
            "density_veh_per_km_lane": len(all_bg) / max(road_length / 1000.0, 1e-6) / len(self.lane_centres),
            "controllers": {str(k): v.describe() for k, v in self.background_controllers.items()},
        }

    # ------------------------------------------------------------------ #
    # Control
    # ------------------------------------------------------------------ #
    def apply_control(self, action) -> None:
        control = self.get_vehicle_control(action)
        self.ego.apply_control(control)

        if self.background_controller == "idm_mobil":
            self._apply_idm_controls()
        else:
            self.nonego.apply_control(self.get_nonego_vehicle_control())

    def _spawn_stopped_vehicle(self):
        """Place a stationary vehicle in a lane ahead, past the overtaking target.

        The lead-vehicle-stopped pre-crash scenario: the ego must detect a
        non-moving obstacle and change lane in time, which produces the sharpest
        deceleration demands in the whole task.
        """
        y = float(self.nonego_spawn_point[1]) - float(self._rng.uniform(35.0, 70.0))
        lane_x = float(self._rng.choice(self.lane_centres))
        transform = carla.Transform(
            carla.Location(x=lane_x, y=y, z=0.1), carla.Rotation(yaw=-90.0)
        )
        actor = self._world.spawn_actor(transform=transform)
        if actor is None:
            return
        self.background_vehicles.append(actor)
        self.stopped_vehicle_id = actor.id
        # No controller: the vehicle simply never moves.

    def _apply_idm_controls(self):
        states = {}
        actors = [self.nonego] + list(self.background_vehicles)
        for actor in actors:
            if not actor.is_alive:
                continue
            states[actor.id] = state_from_carla(actor)
        ego_state = state_from_carla(self.ego)
        all_states = list(states.values()) + [ego_state]

        overrides = self.events.update(self._time_step, ego_state, states)

        for actor in actors:
            if not actor.is_alive or actor.id not in states:
                continue
            if actor.id == getattr(self, "stopped_vehicle_id", None):
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                continue
            ctrl = self.background_controllers.get(actor.id)
            if ctrl is None:
                continue
            state = states[actor.id]
            others = [s for s in all_states if s.id != state.id]
            acc, target_x, _ = ctrl.step(state, others)
            # An injected event transiently overrides the calibrated behaviour;
            # outside its window the vehicle drives normally again.
            acc, target_x = apply_override(
                overrides.get(actor.id), acc, target_x, ego_state.x,
                lane_width=float(self._config.get("lane_width", 3.4)),
            )
            actor.apply_control(self._to_carla_control(actor, state, acc, target_x))

    def _to_carla_control(self, actor, state, acc, target_x):
        """Convert an IDM acceleration and a target lane centre into a CARLA control."""
        coeffs = self._config.get("pid_coeffs", [0.03, 0.0, 0.03])
        errors = self._bg_prev_errors.setdefault(actor.id, {"last_error": 0.0, "integral": 0.0})
        steer, updated = self.pid_controller(float(target_x), float(state.x), errors, coeffs)
        errors.update(updated)

        # Damp the steering with the heading error so lane changes are smooth
        # rather than a step change in lateral position.
        yaw_err = math.atan2(state.vx, self.FORWARD_AXIS * state.vy + 1e-6) if state.speed > 0.5 else 0.0
        steer = steer + 0.5 * yaw_err

        if acc > 0:
            throttle, brake = float(np.clip(acc / 3.0, 0.0, 1.0)), 0.0
        else:
            throttle, brake = 0.0, float(np.clip(-acc / 3.0, 0.0, 1.0))
        return carla.VehicleControl(
            throttle=throttle, steer=float(np.clip(-steer, -1.0, 1.0)), brake=brake
        )

    def get_nonego_vehicle_control(self):
        """
        Legacy scripted behaviour of the single background vehicle, used when
        ``background_controller`` is ``swing``.
        """
        ego_loc = self.ego.get_transform().location
        nonego_loc = self.nonego.get_transform().location

        # Keep constant speed
        if abs(self.nonego.get_velocity().y) < 2:
            acc = 2
        else:
            acc = 0

        dist = math.sqrt((ego_loc.x - nonego_loc.x) ** 2 + (ego_loc.y - nonego_loc.y) ** 2)
        swing_steer = self._config.swing_steer
        swing_amplitude = self._config.swing_amplitude
        swing_trigger_dist = self._config.swing_trigger_dist
        if dist < swing_trigger_dist:
            # Swing when ego vehicle approaching
            if self.nonego_spawn_point[0] + swing_amplitude <= nonego_loc.x:
                self.swing_direction = 1
            if self.nonego_spawn_point[0] - swing_amplitude >= nonego_loc.x:
                self.swing_direction = -1
            steer = swing_steer * self.swing_direction
            self.prev_errors = {
                "last_error": 0.0,
                "integral": 0.0,
            }  # Reset the prev_error
        else:
            # Implement PID controller for lane keeping
            coeffs = self._config.pid_coeffs
            steer, updated_errors = self.pid_controller(self.nonego_spawn_point[0], nonego_loc.x, self.prev_errors, coeffs)
            self.prev_errors.update(updated_errors)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def pid_controller(self, target, current, prev_errors, coeffs):
        """
        Calculate the PID control output to minimize the deviation.

        Args:
        target (float): The target for the PID controller (central line x-coordinate).
        current (float): The current measurement of the process variable (vehicle x-coordinate).
        prev_errors (dict): A dictionary holding the last error and the integral of errors.
        coeffs (tuple): A tuple of PID coefficients (Kp, Ki, Kd).

        Returns:
        float: The control output (steering angle adjustment).
        dict: Updated dictionary with the last error and integral.
        """
        Kp, Ki, Kd = coeffs
        error = current - target
        integral = prev_errors["integral"] + error
        derivative = error - prev_errors["last_error"]

        output = (Kp * error) + (Ki * integral) + (Kd * derivative)

        # Update the errors for the next call
        updated_errors = {"last_error": error, "integral": integral}

        return output, updated_errors

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def reward(self):
        total_reward, info = super().reward()
        # remove the out of lane penalty
        total_reward -= info["r_out_of_lane"]
        del info["r_out_of_lane"]

        reward_scales = self._config.reward.scales
        ego = self.ego
        ego_x, ego_y = get_vehicle_pos(ego)
        nonego_spawn_x = self.nonego_spawn_point[0]
        nonego_y = self.nonego.get_transform().location.y

        # Reward vehicle to stay in the lane, while penalize vehicle staying in the lane when overtaking.
        if (
            ego_y - self._config.reward.early_lane_change_dist < nonego_y
            and ego_y + self._config.reward.reward_overtake_dist > nonego_y
        ):
            p_stay_same_lane = -1 / (0.5 + abs(ego_y - nonego_y)) * reward_scales["stay_same_lane"]
        else:
            p_stay_same_lane = 1 / (0.5 + abs(ego_y - nonego_y)) * reward_scales["stay_same_lane"]

        # Penalty for early lane change before overtake
        p_early_lane_change = 0.0
        if (
            ego_y - self._config.reward.early_lane_change_dist > nonego_y
            and abs(ego_x - nonego_spawn_x) > self._config.reward.lane_width
        ):
            p_early_lane_change = -reward_scales["early_lane_change"]

        # Exceeding reward
        r_exceeding = 0.0
        if ego_y < nonego_y and not self.exceeding:
            r_exceeding = reward_scales["exceeding"]
            self.exceeding = True

        # Overtake reward (exceed and come back to the same lane)
        r_overtake = 0.0
        if (
            ego_y + self._config.reward.reward_overtake_dist < nonego_y
            and abs(ego_x - nonego_spawn_x) < self._config.terminal.lane_width / 5
            and not self.overtake
        ):
            r_overtake = reward_scales["overtake"]
            self.overtake = True

        # Total reward
        total_reward += p_stay_same_lane + p_early_lane_change + r_exceeding + r_overtake

        info.update(
            {
                "p_stay_same_lane": p_stay_same_lane,
                "r_exceeding": r_exceeding,
                "r_overtake": r_overtake,
                "p_early_lane_change": p_early_lane_change,
                "n_background": len(getattr(self, "background_vehicles", [])),
                # Logged so results can be stratified by the initial geometry
                # rather than only averaged over it.
                "initial_gap": getattr(self, "initial_gap", 0.0),
                "ego_start_lane_offset": getattr(self, "ego_start_lane_offset", 0),
                # How many injected pre-crash events actually triggered, so the
                # conflict rate can be attributed rather than guessed at.
                "n_events_scheduled": len(self.events.events) if hasattr(self, "events") else 0,
                "n_events_fired": self.events.n_fired if hasattr(self, "events") else 0,
            }
        )

        return total_reward, info

    def get_terminal_conditions(self):
        ego_x = self.ego.get_location().x
        ego_location = get_vehicle_pos(self.get_ego_vehicle())
        terminal_config = self._config.terminal
        info = super().get_terminal_conditions()
        info["out_of_lane"] = (
            self.get_wpt_dist(ego_location) > terminal_config.out_lane_thres
            or ego_x < terminal_config.left_lane_boundry
            or ego_x > terminal_config.right_lane_boundry
        )
        return info
