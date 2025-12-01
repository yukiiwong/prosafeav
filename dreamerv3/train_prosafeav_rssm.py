import datetime
import warnings
import ruamel.yaml as yaml
import embodied

import car_dreamer
from prosafeav_rssm_agent import ProSafeAVRSSMAgent
from embodied.envs import from_gym

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def main(argv=None):
    # === Load and flatten config ===
    config_path = embodied.Path(__file__).parent / "dreamerv3.yaml"
    yaml_loader = yaml.YAML(typ="safe")
    raw = yaml_loader.load(config_path.read())

    # Flatten default + model size config
    config_base = embodied.Config(raw["defaults"])
    config_flat = config_base.update(raw["small"])

    # Add ProSafeAV-RSSM parameters (lightweight)
    rssm_params = {
        "stochastic_dim": 16,      # Small stochastic dimension
        "deterministic_dim": 64,   # Small deterministic dimension
        "model_lr": 1e-3,
        "policy_lr": 3e-4,
        "imagination_horizon": 10,
        "kl_weight": 0.1,
        "batch_length": 64,
        "batch_size": 16,
        "obs_key": "birdeye_wpt",
    }
    config_flat = config_flat.update(rssm_params)

    # Wrap back to nested config
    config = embodied.Config({"dreamerv3": config_flat})

    # === Task loading ===
    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for task_name in parsed.task:
        print("Using task:", task_name)
        env, task_config = car_dreamer.create_task(task_name, argv)
        config = config.update(task_config)

    config = embodied.Flags(config).parse(other)
    dreamerv3_config = config.dreamerv3

    # === Logging ===
    logdir = embodied.Path(dreamerv3_config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        outputs=[
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    # === Wrap environment ===
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    # === Save config snapshot ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_file = logdir / f"config_{timestamp}.yaml"
    config.save(config_file)
    print(f"[Train ProSafeAV-RSSM] Config saved to {config_file}")
    print("OBS SPACE:", env.obs_space)
    print("=" * 60)
    print("ABLATION STUDY: ProSafeAV-RSSM (Lightweight RSSM)")
    print("- Stochastic + Deterministic structure maintained")
    print("- Reduced network sizes for efficiency")
    print("- Testing if lightweight RSSM is sufficient")
    print("=" * 60)

    # === Initialize Agent and Training ===
    agent = ProSafeAVRSSMAgent(env.obs_space, env.act_space, dreamerv3_config)
    replay = embodied.replay.Uniform(
        dreamerv3_config.batch_length,
        dreamerv3_config.replay_size,
        logdir / "replay"
    )
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )

    embodied.run.train(agent, env, replay, logger, args)


if __name__ == "__main__":
    main()
