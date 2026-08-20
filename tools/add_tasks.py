"""Append the ProSafeAV revision task variants to car_dreamer/configs/tasks.yaml.

Idempotent: re-running replaces the generated block rather than duplicating it.
"""
import sys

PATH = "/home/yukai/CarDreamer_prosafeav/car_dreamer/configs/tasks.yaml"
BEGIN = "# >>> ProSafeAV revision tasks (generated) >>>"
END = "# <<< ProSafeAV revision tasks (generated) <<<"

COMMON_ENV = """    name: CarlaOvertakeEnv-v0
    observation.enabled: [collision, {bev}]
    nonego_spawn_points:
      - [5.8, 80.0, 0.1, 0.0, -90.0, 0.0]
      - [12.2, 80.0, 0.1, 0.0, -90.0, 0.0]
    lane_start_points:
      - [5.8, 100, 0.1]
    lane_end_points:
      - [5.8, 0.0, 0.1]
    lane_centres: [5.8, 9.0, 12.2, 15.6]
    background_controller: {controller}
    traffic_density: {density}          # background vehicles per km per lane
    background_speed_range: [3.0, 9.0]  # brackets the ego's desired speed, so some
                                        # traffic is caught up to and some catches up
    spawn_behind_fraction: 0.4          # share of traffic starting behind the ego
    target_speed: 4.0                   # desired speed of the overtaken vehicle (m/s)
    spawn_gap_range: [12.0, 45.0]
    aggressive_fraction: 0.25
    ego_gap_range: {ego_gap}          # initial ego-to-target gap (m)
    ego_offset_lane_prob: {ego_off}   # chance the ego starts one lane over
    ego_initial_speed_range: {ego_v0}
    # Injected pre-crash scenarios (Najm et al., NHTSA 2007 typology).
    # Free-flowing IDM/MOBIL traffic keeps safe headways by construction and
    # can run a whole episode without a conflict, leaving the EVT model with
    # nothing to fit.  These transiently override the calibrated behaviour.
    conflict_events:
      lead_brake:      {{ prob: {p_brake}, trigger_distance: [12.0, 30.0], duration: [0.8, 2.0], intensity: [0.6, 1.0] }}
      cut_in:          {{ prob: {p_cutin}, trigger_distance: [8.0, 22.0], duration: [1.5, 3.0], intensity: [0.6, 1.2] }}
      stopped_vehicle: {{ prob: {p_stop}, trigger_distance: [0.0, 0.0], duration: [0.0, 0.0], intensity: [0.0, 0.0] }}
    swing_steer: 0.04
    swing_amplitude: 0.2
    swing_trigger_dist: 20
    pid_coeffs: [0.03, 0.0, 0.03]
    evt:
      mode: {evt_mode}          # none | env | imagine | both
      copula: {copula}          # logistic (manuscript) | frank (sensitivity check)
      threshold_method: {thr_method}   # stability | mrl | quantile
      threshold_ttc: null       # null selects the threshold from the data
      threshold_drac: null
      update_interval: 2000     # environment steps between refits
      buffer_size: 20000
      min_sample: 500
      min_exceedances: 50
      risk_tolerance: {tol}     # u in Eq. (6)
      indicator_mode: {ind_mode}  # max | longitudinal | planar
      interaction_radius: 50.0
      crash_drac: 8.5
    reward:
      desired_speed: 5
      reward_overtake_dist: 8
      early_lane_change_dist: 10
      lane_width: 3.4
      scales:
        {{
          waypoint: 2.0,
          speed: 0.5,
          stay_same_lane: 0.3,
          out_of_lane: 3.0,
          collision: 30.0,
          time: 0.0,
          evt: {w_evt},
          exceeding: 200.0,
          overtake: 200.0,
          early_lane_change: 0.0,
          destination_reached: 20.0,
        }}
    terminal:
      out_lane_thres: 5
      time_limit: 500
      left_lane_boundry: 3.7
      right_lane_boundry: 17.7
      lane_width: 3.4
      terminal_dist: 100
"""

TRAILER = """
  dreamerv3:
    encoder.cnn_keys: "{bev}"
    decoder.cnn_keys: "{bev}"
    run.log_keys_video: [{bev}]
    run.log_keys_max: "collision"
    evt.mode: {evt_mode}
    evt.imag_weight: {w_imag}

  dreamerv2:
    encoder.cnn_keys: "{bev}"
    decoder.cnn_keys: "{bev}"
    decoder.cnn_kernels: [5, 5, 5, 6, 6]
    train.log_keys_video: [{bev}]
"""


def task(name, comment, bev="birdeye_wpt", controller="idm_mobil", density=15,
         evt_mode="both", copula="logistic", thr_method="stability", tol=0.0,
         ind_mode="max", w_evt=3.0, w_imag=3.0, extra_env="",
         ego_gap="[15.0, 45.0]", ego_off=0.3, ego_v0="[2.0, 6.0]",
         p_brake=0.5, p_cutin=0.4, p_stop=0.25):
    body = COMMON_ENV.format(
        bev=bev, controller=controller, density=density, evt_mode=evt_mode,
        copula=copula, thr_method=thr_method, tol=tol, ind_mode=ind_mode, w_evt=w_evt,
        ego_gap=ego_gap, ego_off=ego_off, ego_v0=ego_v0,
        p_brake=p_brake, p_cutin=p_cutin, p_stop=p_stop,
    )
    trailer = TRAILER.format(bev=bev, evt_mode=evt_mode, w_imag=w_imag)
    return f"\n# {comment}\n{name}:\n  env:\n{body}{extra_env}{trailer}"


blocks = [
    task(
        "carla_overtake_prosafeav",
        "ProSafeAV main configuration: heterogeneous IDM/MOBIL traffic at a medium "
        "density, logistic extreme-value copula, EVT penalty applied both in the "
        "environment reward and inside latent imagination.",
    ),
    # ---- Reviewer 1 comment 6: generalisation across traffic density -------- #
    task("carla_overtake_d05", "Density sweep: 5 veh/km/lane (sparse).", density=5),
    task("carla_overtake_d15", "Density sweep: 15 veh/km/lane (medium).", density=15),
    task("carla_overtake_d30", "Density sweep: 30 veh/km/lane (dense).", density=30),
    task("carla_overtake_d45", "Density sweep: 45 veh/km/lane (congested).", density=45),
    # ---- Reviewer 1 comment 3: onboard-perception realism ------------------ #
    task(
        "carla_overtake_fov",
        "Field-of-view limited BEV: the raster is built only from what an onboard "
        "sensor suite can observe, so occluded and out-of-range vehicles are simply "
        "absent from it.",
        extra_env="""    observation:
      birdeye_wpt:
        observability: fov
        sight_fov: 150
        sight_range: 32
""",
    ),
    task(
        "carla_overtake_noisy",
        "Perception-degraded BEV: detections are perturbed to emulate the tracking "
        "errors of a real onboard stack.",
        bev="birdeye_wpt_with_errors",
    ),
    # ---- Ablations over where EVT acts ------------------------------------- #
    task("carla_overtake_noevt", "Ablation: no EVT term anywhere (DreamerV3 baseline).",
         evt_mode="none", w_evt=0.0, w_imag=0.0),
    task("carla_overtake_evtenv", "Ablation: EVT penalty on realised transitions only.",
         evt_mode="env", w_imag=0.0),
    task("carla_overtake_evtimag", "Ablation: EVT penalty inside latent imagination only.",
         evt_mode="imagine", w_evt=0.0),
    # ---- Sensitivity studies ----------------------------------------------- #
    task("carla_overtake_frank", "Sensitivity: Frank copula instead of the logistic one.",
         copula="frank"),
    task("carla_overtake_thrq", "Sensitivity: fixed 90th-percentile threshold rule.",
         thr_method="quantile"),
    task("carla_overtake_lonly", "Sensitivity: car-following conflict indicators only "
         "(reproduces the same-lane-only definition).", ind_mode="longitudinal"),
    task("carla_overtake_w1", "Sensitivity: EVT weight w_evt = 1.", w_evt=1.0, w_imag=1.0),
    task("carla_overtake_w10", "Sensitivity: EVT weight w_evt = 10.", w_evt=10.0, w_imag=10.0),
    # ---- Conflict-rich variants -------------------------------------------- #
    task("carla_overtake_critical",
         "Conflict-rich: every episode injects a hard-braking lead, a cut-in and a "
         "stopped vehicle. Used to populate the EVT tail quickly and to test the "
         "policy against the pre-crash scenarios directly.",
         density=25, p_brake=1.0, p_cutin=1.0, p_stop=0.6,
         ego_gap="[10.0, 25.0]"),
    task("carla_overtake_calm",
         "Conflict-poor control: no injected events, sparse traffic. Shows how much "
         "of the EVT tail is due to the injected scenarios rather than to density.",
         density=8, p_brake=0.0, p_cutin=0.0, p_stop=0.0),

    # ---- Legacy ------------------------------------------------------------ #
    task(
        "carla_overtake_legacy",
        "The originally published scenario: a single background vehicle driven by the "
        "scripted swing plus PID controller.  Kept so the previous results remain "
        "reproducible.",
        controller="swing", density=0, ego_gap="[20.0, 20.0]", ego_off=0.0,
        ego_v0="[0.0, 0.0]", p_brake=0.0, p_cutin=0.0, p_stop=0.0,
    ),
]

GENERATED = BEGIN + "\n" + "".join(blocks) + "\n" + END + "\n"

with open(PATH, encoding="utf-8") as fh:
    src = fh.read()

if BEGIN in src:
    head, rest = src.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    src = head + GENERATED + tail
    action = "replaced"
else:
    src = src.rstrip() + "\n\n" + GENERATED
    action = "appended"

with open(PATH, "w", encoding="utf-8") as fh:
    fh.write(src)

# Validate.
try:
    import yaml

    with open(PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    names = [b.split(":\n  env:")[0].rsplit("\n", 1)[-1] for b in blocks]
    missing = [n for n in names if n not in cfg]
    if missing:
        print("ERROR: tasks missing after write:", missing)
        sys.exit(1)
    print(f"{action} {len(names)} tasks; yaml parses; total tasks = {len(cfg)}")
    print("  " + ", ".join(names))
except ImportError:
    print(f"{action} the generated block (pyyaml unavailable, not validated)")
