"""
SimPLe Agent (Simulated Policy Learning, Kaiser et al. 2019)
Simple model-based RL: Learn world model + train policy on simulated data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class SimpleDynamicsModel(nn.Module):
    """Simple deterministic dynamics model: (s, a) -> (s', r)"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_next_obs = nn.Linear(128, obs_dim)
        self.fc_reward = nn.Linear(128, 1)

    def forward(self, obs, action):
        obs_flat = obs.reshape(obs.size(0), -1)
        x = torch.cat([obs_flat, action], dim=-1)
        h = self.net(x)
        next_obs = self.fc_next_obs(h)
        reward = self.fc_reward(h)
        return next_obs, reward


class SimplePolicy(nn.Module):
    """Simple policy network"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs):
        obs_flat = obs.reshape(obs.size(0), -1)
        return self.net(obs_flat)


class SimpleValueNetwork(nn.Module):
    """Value network for policy gradient"""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        obs_flat = obs.reshape(obs.size(0), -1)
        return self.net(obs_flat)


class SimPLeAgent:
    """
    SimPLe: Simple Model-based Policy Learning
    1. Learn dynamics model from real data
    2. Generate simulated rollouts using the model
    3. Train policy on mixture of real + simulated data
    """
    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Observation and action spaces
        obs_key = 'birdeye_wpt'
        obs_shape = obs_space[obs_key].shape
        self.obs_key = obs_key
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space['action'].shape[0]
        self.num_actions = self.act_dim

        # Networks
        self.dynamics = SimpleDynamicsModel(self.obs_dim, self.act_dim).to(self.device)
        self.policy = SimplePolicy(self.obs_dim, self.act_dim).to(self.device)
        self.value = SimpleValueNetwork(self.obs_dim).to(self.device)

        # Optimizers
        self.dynamics_optimizer = optim.Adam(
            self.dynamics.parameters(),
            lr=config.get('dynamics_lr', 1e-3)
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.get('policy_lr', 3e-4)
        )
        self.value_optimizer = optim.Adam(
            self.value.parameters(),
            lr=config.get('value_lr', 1e-3)
        )

        # Training parameters
        self.rollout_length = config.get('rollout_length', 5)
        self.num_simulated_rollouts = config.get('num_simulated_rollouts', 10)
        self.gamma = config.get('gamma', 0.99)

        # Exploration
        self.epsilon = config.get('epsilon_start', 0.2)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.999)

        self.train_step_counter = 0

    def _preprocess_obs(self, obs):
        x = obs[self.obs_key]
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x.reshape(x.size(0), -1)

    def __call__(self, obs, state=None, mode='train'):
        return self.policy(obs, state, mode)

    def policy(self, obs, state=None, mode='train'):
        obs_tensor = self._preprocess_obs(obs)
        batch_size = obs_tensor.size(0)

        with torch.no_grad():
            logits = self.policy(obs_tensor)

            if mode == 'train' and np.random.rand() < self.epsilon:
                # Random exploration
                action_idx = np.random.randint(0, self.num_actions, size=batch_size)
            else:
                # Greedy/stochastic policy
                if mode == 'eval':
                    action_idx = logits.argmax(dim=-1).cpu().numpy()
                else:
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample().cpu().numpy()

        one_hot = np.zeros((len(action_idx), self.num_actions), dtype=np.float32)
        one_hot[np.arange(len(action_idx)), action_idx] = 1.0

        return {'action': one_hot, 'reset': np.array(False)}, state

    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        actions = torch.tensor(batch['action'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)

        # Get next observations
        if 'next_obs' in batch:
            next_obs_tensor = torch.tensor(batch['next_obs'][self.obs_key], dtype=torch.float32).to(self.device)
            next_obs_tensor = next_obs_tensor.reshape(next_obs_tensor.size(0), -1)
        else:
            next_obs_tensor = obs_tensor

        # ===== Train Dynamics Model =====
        pred_next_obs, pred_reward = self.dynamics(obs_tensor, actions)

        dynamics_obs_loss = F.mse_loss(pred_next_obs, next_obs_tensor)
        dynamics_reward_loss = F.mse_loss(pred_reward.squeeze(-1), rewards)
        dynamics_loss = dynamics_obs_loss + dynamics_reward_loss

        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        # ===== Generate Simulated Rollouts =====
        with torch.no_grad():
            simulated_obs = []
            simulated_actions = []
            simulated_rewards = []

            # Start from current observations
            current_obs = obs_tensor.detach()

            for _ in range(self.rollout_length):
                # Get action from current policy
                action_logits = self.policy(current_obs)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample()
                action_one_hot = F.one_hot(action_idx, num_classes=self.num_actions).float()

                # Simulate next state
                next_obs, reward = self.dynamics(current_obs, action_one_hot)

                simulated_obs.append(current_obs)
                simulated_actions.append(action_one_hot)
                simulated_rewards.append(reward.squeeze(-1))

                current_obs = next_obs

            simulated_obs = torch.stack(simulated_obs, dim=1)  # [batch, rollout_length, obs_dim]
            simulated_actions = torch.stack(simulated_actions, dim=1)
            simulated_rewards = torch.stack(simulated_rewards, dim=1)

        # ===== Train Policy on Real + Simulated Data =====
        # Real data
        real_logits = self.policy(obs_tensor)
        real_values = self.value(obs_tensor).squeeze(-1)
        real_action_idx = actions.argmax(dim=-1)

        # Simulated data (flatten batch and rollout dimensions)
        batch_size, rollout_len, obs_dim = simulated_obs.shape
        sim_obs_flat = simulated_obs.reshape(batch_size * rollout_len, obs_dim)
        sim_actions_flat = simulated_actions.reshape(batch_size * rollout_len, self.num_actions)
        sim_rewards_flat = simulated_rewards.reshape(batch_size * rollout_len)

        sim_logits = self.policy(sim_obs_flat)
        sim_values = self.value(sim_obs_flat).squeeze(-1)
        sim_action_idx = sim_actions_flat.argmax(dim=-1)

        # Compute advantages (simple TD error)
        with torch.no_grad():
            next_values = self.value(next_obs_tensor).squeeze(-1)
            real_advantages = rewards + self.gamma * next_values - real_values

            # For simulated data, use returns
            sim_advantages = sim_rewards_flat - sim_values

        # Policy loss (policy gradient)
        real_log_probs = F.log_softmax(real_logits, dim=-1)
        real_selected_log_probs = real_log_probs.gather(1, real_action_idx.unsqueeze(1)).squeeze(1)
        real_policy_loss = -(real_selected_log_probs * real_advantages.detach()).mean()

        sim_log_probs = F.log_softmax(sim_logits, dim=-1)
        sim_selected_log_probs = sim_log_probs.gather(1, sim_action_idx.unsqueeze(1)).squeeze(1)
        sim_policy_loss = -(sim_selected_log_probs * sim_advantages.detach()).mean()

        policy_loss = real_policy_loss + 0.5 * sim_policy_loss

        # Value loss
        real_value_loss = F.mse_loss(real_values, rewards + self.gamma * next_values.detach())
        sim_value_loss = F.mse_loss(sim_values, sim_rewards_flat)
        value_loss = real_value_loss + 0.5 * sim_value_loss

        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Optimize value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.train_step_counter += 1

        return {}, state, {
            'loss': (dynamics_loss + policy_loss + value_loss).item(),
            'dynamics_loss': dynamics_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'epsilon': self.epsilon,
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            'dynamics': self.dynamics.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'epsilon': self.epsilon,
        }

    def load(self, data):
        self.dynamics.load_state_dict(data['dynamics'])
        self.policy.load_state_dict(data['policy'])
        self.value.load_state_dict(data['value'])
        self.epsilon = data.get('epsilon', self.epsilon)

    def sync(self):
        pass
