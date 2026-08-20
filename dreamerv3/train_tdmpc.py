import datetime
import warnings

import ruamel.yaml as yaml

import embodied
import car_dreamer
from tdmpc_agent import TDMPCAgent
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
    config_path = embodied.Path(__file__).parent / "dreamerv3.yaml"
    yaml_loader = yaml.YAML(typ="safe")
    raw = yaml_loader.load(config_path.read())

    config_base = embodied.Config(raw["defaults"])
    config_flat = config_base.update(raw["small"])
    config_flat = config_flat.update({
        "latent_dim": 64,
        "plan_horizon": 8,
        "num_samples": 128,
        "num_elites": 16,
        "plan_iterations": 4,
        "discount": 0.99,
        "consistency_weight": 2.0,
        "model_lr": 3e-4,
        "batch_length": 64,
        "batch_size": 16,
        "obs_key": "birdeye_wpt",
        "evt_mode": "both",
        "evt_imag_weight": 3.0,
    })

    config = embodied.Config({"dreamerv3": config_flat})

    parsed, other = embodied.Flags(task=["carla_overtake_prosafeav"]).parse_known(argv)
    for task_name in parsed.task:
        print("Using task:", task_name)
        env, task_config = car_dreamer.create_task(task_name, argv)
        config = config.update(task_config)

    config = embodied.Flags(config).parse(other)
    dreamerv3_config = config.dreamerv3

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

    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config.save(logdir / f"config_{timestamp}.yaml")
    print("=" * 60)
    print("CROSS-ARCHITECTURE BASELINE: TD-MPC style planner")
    print("Decision-time trajectory optimisation; EVT risk enters the planner objective")
    print("=" * 60)

    agent = TDMPCAgent(env.obs_space, env.act_space, dreamerv3_config)
    replay = embodied.replay.Uniform(
        dreamerv3_config.batch_length,
        dreamerv3_config.replay_size,
        logdir / "replay",
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
