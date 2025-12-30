# ProSafeAV: Proactive Risk Modelling for Autonomous Driving

**Integrating World Models with Extreme Value Theory for Safety-Critical Decision Making**

---

## 📖 Overview

ProSafeAV is a novel world-model-centered framework that combines latent-space prediction with probabilistic tail-risk estimation using **Extreme Value Theory (EVT)** for autonomous driving. Unlike traditional reactive safety approaches, ProSafeAV enables proactive risk assessment by predicting and preventing high-risk traffic conflicts before they escalate into collisions.

### Key Innovation

- 🔮 **Predictive Risk Assessment**: Uses world models to generate imagined future trajectories in latent space
- 📊 **Extreme Value Theory**: Applies EVT to convert traffic conflicts into tail-risk probability estimates
- 🛡️ **Proactive Safety**: Identifies safety-critical events before they occur, not after crashes happen
- 🎯 **Model-Agnostic Design**: Works with any world model architecture as a black-box dynamics component

## 🌟 Why ProSafeAV?

Traditional autonomous driving systems face critical limitations:

1. **Reactive Safety**: Most methods analyze crashes after they occur, which is insufficient for real-time AV decision-making
2. **Data Scarcity**: Limited AV crash data makes traditional safety assessment unreliable
3. **Missing Risk Layer**: Current world models excel at predicting likely scenarios but lack explicit representation of rare, safety-critical events

ProSafeAV addresses these challenges by:
- Using **traffic conflicts** (near-miss events) as surrogate safety measures
- Applying **EVT** to model tail distributions and estimate crash probabilities from conflict data
- Integrating risk estimates directly into the **reinforcement learning** reward function for risk-aware policy learning

## 🏗️ Architecture

ProSafeAV consists of four main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    External Environment                      │
└────────────────────┬────────────────────────────────────────┘
                     │ Observations (Camera, LiDAR, Maps)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Perception Module                         │
│  • Multi-sensor fusion                                       │
│  • Object detection and tracking                             │
└────────────────────┬────────────────────────────────────────┘
                     │ Fused sensor data
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    World Model (RSSM)                        │
│  • Encoder: obs → latent state                               │
│  • Dynamics: predict future latent states                    │
│  • Decoder: reconstruct observations                         │
│  • Generates imagined trajectories in latent space           │
└────────────────────┬────────────────────────────────────────┘
                     │ Latent rollouts
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVT Risk Estimation                       │
│  • Extract surrogate safety measures (TTC, DRAC)             │
│  • Fit Generalized Pareto Distribution (GPD)                 │
│  • Logistic copula for joint tail-risk modeling              │
│  • Output: Tail-risk probability P_joint                     │
└────────────────────┬────────────────────────────────────────┘
                     │ Risk predictions
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Reinforcement Learning (Actor-Critic)           │
│  • Actor: proposes actions based on states + risk            │
│  • Critic: evaluates long-term value with EVT penalty        │
│  • Risk-aware reward: r_t - λ * I(P_joint > threshold)      │
│  • Learns safe, efficient driving policies                   │
└────────────────────┬────────────────────────────────────────┘
                     │ Actions
                     ▼
                Environment
```

### Component Details

**1. Perception Module**
- Fuses multi-sensor data (cameras, LiDAR, HD maps)
- Detects road users, obstacles, and traffic conditions
- Creates comprehensive environmental understanding

**2. World Model (based on DreamerV3 RSSM)**
- **Encoder**: Compresses high-dimensional observations into compact latent states
- **Latent Dynamics**: Predicts state evolution: `z_{t+1} = f(z_t, a_t)`
- **Reward Predictor**: Estimates task utility without environment interaction
- Enables efficient planning in latent space with reduced computational cost

**3. EVT Risk Estimation Module**
- **Surrogate Safety Measures**:
  - **TTC (Time-to-Collision)**: Time until collision if current speeds maintained
  - **DRAC (Deceleration Rate to Avoid Collision)**: Required deceleration to prevent crash
- **Peaks-Over-Threshold (POT)**: Extracts exceedances above threshold
- **GPD Fitting**: Models extreme value distribution for each measure
- **Logistic Copula**: Captures joint tail dependence between TTC and DRAC
- **Risk Probability**: Computes `P_joint(TTC, DRAC)` for tail-risk assessment

**4. Reinforcement Learning Module**
- **Actor-Critic Architecture**:
  - Actor proposes actions based on latent states and risk predictions
  - Critic evaluates actions considering both task reward and safety risk
- **Risk-Aware Reward**:
  ```
  r_total = r_task - λ * I(P_joint > threshold)
  ```
- Learns policies that balance task efficiency with proactive safety

## 🔬 Technical Approach

### Extreme Value Theory Integration

ProSafeAV uses a bivariate EVT framework:

1. **Marginal Distribution** (Generalized Pareto Distribution):
   ```
   F(x) = 1 - (1 + ξ(x-u)/σ)^(-1/ξ)
   ```
   where `u` is threshold, `σ` is scale, `ξ` is shape parameter

2. **Joint Distribution** (Logistic Copula):
   ```
   C(u₁, u₂) = exp{-[(−log u₁)^θ + (−log u₂)^θ]^(1/θ)}
   ```
   where `θ` controls extremal dependence strength

3. **Tail Risk Probability**:
   ```
   P_joint = 1 - C(F₁(TTC), F₂(DRAC))
   ```

### World Model-Based RL

Formulated as MDP `(S, A, T, R, γ)`:

- **Latent State**: `z_t = enc(o_t)`
- **Dynamics**: `z_{t+1} = f_θ(z_t, a_t)`
- **EVT-Augmented Reward**: `r_t = r_task(z_t, a_t) - λ * penalty(P_joint)`

The agent learns entirely in latent space through imagined rollouts:
```
z_t → a_t → z_{t+1} → ... → z_{t+H}
```

## 📊 Experimental Results

### Benchmark Comparison

ProSafeAV is evaluated against representative baselines:

**Model-Free Methods:**
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
- PPO (Proximal Policy Optimization)

**Model-Based Methods:**
- DreamerV3 (baseline world model)
- ProSafeAV variants (RSSM, Deterministic)
- PlaNet, World Models, Simple World Model

### Performance Highlights

- ✅ **Zero Collisions**: ProSafeAV eliminates collision incidents entirely
- ✅ **Zero Lane Departures**: No lane-departure events in testing
- 📈 **Superior Safety**: Outperforms all model-free baselines in safety metrics
- 🎯 **Task Efficiency**: Maintains competitive task performance while ensuring safety

### Testing Environment

- **Simulator**: CARLA (high-fidelity autonomous driving simulator)
- **Scenario**: Dynamic overtaking with multiple traffic participants
- **Evaluation**: Collision rate, lane departures, task completion, comfort metrics

## 🚀 Quick Start

### Prerequisites

ProSafeAV builds on the CarDreamer platform. First, ensure you have:

- Python 3.10+
- CARLA Simulator 0.9.15
- CUDA 11.8+ (for GPU acceleration)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ProSafeAV
cd ProSafeAV
```

2. **Download and setup CARLA**:
```bash
# Download CARLA 0.9.15 from https://github.com/carla-simulator/carla/releases
export CARLA_ROOT="/path/to/carla"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
```

3. **Create conda environment and install dependencies**:
```bash
conda create python=3.10 --name prosafeav
conda activate prosafeav
pip install flit
flit install --symlink
```

4. **Install DreamerV3 dependencies**:
```bash
cd dreamerv3
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.34" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

5. **Set up environment variables**:
```bash
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export CUSOLVER_PATH=$(dirname $(python -c "import nvidia.cusolver;print(nvidia.cusolver.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUSOLVER_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Training

#### ProSafeAV with EVT Risk Estimation

```bash
# Train ProSafeAV RSSM variant
bash train_prosafeav.sh 2000 0 --task carla_overtake \
    --dreamerv3.logdir ./logdir/prosafeav_rssm

# Train ProSafeAV Deterministic variant
python dreamerv3/train_prosafeav_deterministic.py \
    --task carla_overtake \
    --dreamerv3.logdir ./logdir/prosafeav_det
```


### Evaluation

```bash
# Evaluate trained ProSafeAV model
bash eval_dm3.sh 2000 0 ./logdir/prosafeav_rssm/checkpoint.ckpt \
    --task carla_overtake \
    --dreamerv3.logdir ./logdir/eval_prosafeav

# Get evaluation statistics
python dreamerv3/eval_stats.py --logdir ./logdir/eval_prosafeav
```

### Visualization

**Real-time monitoring** (during training):
```bash
# Access web interface at http://localhost:9000/
# Port = CARLA_PORT + 7000 (e.g., 2000 + 7000 = 9000)
```

**TensorBoard** (offline analysis):
```bash
tensorboard --logdir ./logdir/prosafeav_rssm
# Open http://localhost:6006/ in browser
```

## 📚 Available Tasks

ProSafeAV supports all CarDreamer built-in tasks:

- `carla_overtake` - Overtaking scenario (primary evaluation scenario)
- `carla_four_lane` - Four-lane highway driving
- `carla_right_turn_simple` - Right turn at intersection
- `carla_left_turn_hard` - Left turn with heavy traffic
- `carla_lane_merge` - Lane merging scenario
- `carla_roundabout` - Roundabout navigation
- `carla_traffic_lights` - Traffic light compliance
- `carla_stop_sign` - Stop sign scenario

For complete task configurations, see [CarDreamer Documentation](https://car-dreamer.readthedocs.io/en/latest/tasks.html).

## 🔧 Customization

### Adjusting EVT Parameters

Modify risk threshold and penalty weight in configuration:

```python
# In task config YAML
evt:
  threshold: 0.95        # Risk tolerance threshold
  lambda: 1.0            # Penalty weight
  safety_measures:
    - TTC                # Time-to-Collision
    - DRAC               # Deceleration Rate to Avoid Collision
```

### Creating Custom Tasks

Follow the [CarDreamer Task Development Guide](https://car-dreamer.readthedocs.io/en/latest/customization.html) to create new driving scenarios with ProSafeAV safety features.

## 📄 Citation

If you use ProSafeAV in your research, please cite:

```bibtex
@article{wang2025prosafeav,
  title={Proactive Risk Modelling for Autonomous Driving with World Models and Extreme Value Theory},
  author={Wang, Yukai and Jang, Kitae and Luo, Yuhao and Chen, Sikai and Chen, Tiantian},
  journal={[Under Review]},
  year={2025}
}
```

## 🤝 Acknowledgments

This project is built upon the excellent [CarDreamer](https://github.com/ucd-dare/CarDreamer) platform. We are deeply grateful to the CarDreamer team for providing a robust, open-source framework for world model-based autonomous driving research.

**CarDreamer Citation:**
```bibtex
@ARTICLE{10714437,
  author={Gao, Dechen and Cai, Shuangyu and Zhou, Hanchu and Wang, Hang and Soltani, Iman and Zhang, Junshan},
  journal={IEEE Internet of Things Journal},
  title={CarDreamer: Open-Source Learning Platform for World Model Based Autonomous Driving},
  year={2024},
  doi={10.1109/JIOT.2024.3479088}
}
```

We also acknowledge the following projects that contributed to this work:
- [DreamerV3](https://github.com/danijar/dreamerv3) - World model architecture
- [CARLA Simulator](https://carla.org/) - High-fidelity autonomous driving simulation
- CarDreamer community for their valuable contributions

## 📧 Contact

For questions or collaborations:

- **Yukai Wang**: yukai@kaist.ac.kr


## 📜 License

This project inherits the license from CarDreamer. CarDreamer © 2024 The Regents of the University of California, Davis campus.

---

**ProSafeAV** - Advancing Proactive Safety for Autonomous Vehicles through Predictive Risk Modeling
