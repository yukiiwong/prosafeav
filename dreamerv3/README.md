# Guide for Training on ProSafeAV

ProSafeAV is built on the CarDreamer platform and extends it with multiple reinforcement learning algorithms and world model variants for autonomous driving research.

This guide assumes you have installed `car_dreamer`. If not, please follow the instructions in the [main README](../README.md).

## Available Algorithms

ProSafeAV includes the following algorithms:

### World Model-Based Methods
- **DreamerV3** - State-of-the-art world model with RSSM (Recurrent State Space Model)
- **ProSafeAV RSSM** - Custom RSSM variant for safer autonomous driving
- **ProSafeAV Deterministic** - Deterministic world model variant
- **PlaNet** - Deep Planning Network
- **World Models** - Classic world model architecture
- **Simple World Model** - Lightweight world model implementation

### Model-Free RL Methods
- **DQN** - Deep Q-Network
- **SAC** - Soft Actor-Critic
- **TD3** - Twin Delayed Deep Deterministic Policy Gradient
- **PPO** - Proximal Policy Optimization

## Installation

First, install the required dependencies for DreamerV3 and world model-based methods:

```bash
cd dreamerv3
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.34" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Set up CARLA and environment variables:

```bash
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export CUSOLVER_PATH=$(dirname $(python -c "import nvidia.cusolver;print(nvidia.cusolver.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUSOLVER_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Training

### Using ProSafeAV Training Script (Recommended)

ProSafeAV provides an improved training script with automatic CARLA server management and crash recovery:

```bash
cd ..
# Basic usage
bash train_prosafeav.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane

# Override task and model parameters
bash train_prosafeav.sh 2000 0 --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/carla_right_turn_simple \
    --dreamerv3.run.steps=5e6
```

**Parameters:**
- `2000` - CARLA server port number (script will automatically start the server)
- `0` - GPU device number
- `--task` - Task name (see [documentation](https://car-dreamer.readthedocs.io/en/latest/tasks.html) for available tasks)
- `--dreamerv3.logdir` - Directory to save training logs

**Features:**
- Automatic CARLA server startup and restart on crashes
- Training script auto-restart on failures
- Logs saved to `log_<port>.log`

### Training Different Algorithms

#### World Model-Based Methods

**DreamerV3 (Default):**
```bash
bash train_prosafeav.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/dreamerv3
```

**ProSafeAV RSSM Variant:**
```bash
python dreamerv3/train_prosafeav_rssm.py --task carla_four_lane --dreamerv3.logdir ./logdir/prosafeav_rssm
```

**ProSafeAV Deterministic:**
```bash
python dreamerv3/train_prosafeav_deterministic.py --task carla_four_lane --dreamerv3.logdir ./logdir/prosafeav_det
```

**PlaNet:**
```bash
python dreamerv3/train_planet.py --task carla_four_lane --dreamerv3.logdir ./logdir/planet
```

**World Models:**
```bash
python dreamerv3/train_worldmodels.py --task carla_four_lane --dreamerv3.logdir ./logdir/worldmodels
```

**Simple World Model:**
```bash
python dreamerv3/train_simple.py --task carla_four_lane --dreamerv3.logdir ./logdir/simple
```

#### Model-Free RL Methods

**DQN:**
```bash
python dreamerv3/train_dqn.py --task carla_four_lane --dreamerv3.logdir ./logdir/dqn
```

**SAC:**
```bash
python dreamerv3/train_sac.py --task carla_four_lane --dreamerv3.logdir ./logdir/sac
```

**TD3:**
```bash
python dreamerv3/train_td3.py --task carla_four_lane --dreamerv3.logdir ./logdir/td3
```

**PPO:**
```bash
python dreamerv3/train_ppo.py --task carla_four_lane --dreamerv3.logdir ./logdir/ppo
```

### Algorithm Selection Guide

- **DreamerV3**: Best overall performance, recommended for most tasks. Learns world model and policy jointly.
- **ProSafeAV Variants**: Experimental variants focusing on safety and reliability for autonomous driving.
- **Model-Free Methods (DQN/SAC/TD3/PPO)**: Faster training but may require more environment interactions. Good for simpler tasks.
- **Simple/PlaNet/World Models**: Lightweight alternatives for resource-constrained environments or ablation studies.

### Notes

- World model-based methods generally require more GPU memory (10-20GB) but achieve better sample efficiency
- Model-free methods are faster to train but may need more episodes to converge
- All methods support the same tasks and configuration options as DreamerV3
- For running multiple experiments, you can use different CARLA ports (e.g., 2000, 2002, 2004) on different GPUs

## Visualization

Online data monitoring can be accessed on website on `http://localhost:9000/`, where the port number should be changed to `<carla-port> + 7000` if you don't use the default port number `2000` for CARLA server.

Offline data logging can be accessed through TensorBoard or WandB:

```bash
tensorboard --logdir ./logdir/carla_four_lane
```

Go to `http://localhost:6006/` in your browser to see the output.

To use `wandb` for visualization, add the WandB logger to `dreamerv3/train.py`:

```python
logger = embodied.Logger(step, [
    # ...
    embodied.logger.WandBOutput(logdir.name, config),
])
```

Once you log in `wandb`, put your project and entity name in `dreamerv3/embodied/logger.py`:

```python
class WandBOutput:

  def __init__(self, run_name, config, pattern=r'.*'):
    self._pattern = re.compile(pattern)
    import wandb
    wandb.init(
        project="project_name",
        name=run_name,
        entity='entity_name',
        config=dict(config),
    )
    self._wandb = wandb
```

# Evaluation

Run the following commands to evaluate the trained model where the third argument is the path to the checkpoint:

```bash
bash eval_dm3.sh 2000 0 ./logdir/carla_four_lane/checkpoint.ckpt --task carla_four_lane --dreamerv3.logdir ./logdir/eval_carla_four_lane
```

After running for some episodes, you can visualize the evaluation results using TensorBoard or WandB as described above. Furthermore, you can get the evaluation metrics by running the following command:

```bash
python dreamerv3/eval_stats.py --logdir ./logdir/eval_carla_four_lane
```
