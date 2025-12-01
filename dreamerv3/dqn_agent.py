import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.q_values = nn.Linear(64, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        return self.q_values(x)


class DQNAgent:
    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use birdeye_wpt as input (non-image)
        obs_key = 'birdeye_wpt'
        obs_shape = obs_space[obs_key].shape
        self.obs_key = obs_key
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space['action'].shape[0]  # one-hot action dimension
        self.num_actions = self.act_dim

        # Q-networks (main and target)
        self.q_network = QNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.target_network = QNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.get('learning_rate', 1e-3))

        # DQN hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.train_step_counter = 0

    def _preprocess_obs(self, obs):
        x = obs[self.obs_key]
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x.reshape(x.size(0), -1)  # flatten to (batch, obs_dim)

    def __call__(self, obs, state=None, mode='train'):
        return self.policy(obs, state, mode)

    def policy(self, obs, state=None, mode='train'):
        obs_tensor = self._preprocess_obs(obs)

        # Epsilon-greedy policy
        if mode == 'train' and np.random.rand() < self.epsilon:
            # Random action
            action_idx = np.random.randint(0, self.num_actions, size=obs_tensor.size(0))
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action_idx = q_values.argmax(dim=1).cpu().numpy()

        # Convert to one-hot
        one_hot = np.zeros((len(action_idx), self.num_actions), dtype=np.float32)
        one_hot[np.arange(len(action_idx)), action_idx] = 1.0

        return {'action': one_hot, 'reset': np.array(False)}, state

    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        actions = torch.tensor(batch['action'], dtype=torch.float32).to(self.device)
        actions_idx = torch.argmax(actions, dim=1)
        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)

        # Get next observations (assuming batch contains transitions)
        # For simplicity, we assume batch['next_obs'] exists
        if 'next_obs' in batch:
            next_obs_tensor = torch.tensor(batch['next_obs'][self.obs_key], dtype=torch.float32).to(self.device)
            next_obs_tensor = next_obs_tensor.reshape(next_obs_tensor.size(0), -1)
            dones = torch.tensor(batch.get('done', np.zeros_like(rewards.cpu().numpy())), dtype=torch.float32).to(self.device)
        else:
            # Fallback: treat current obs as next obs (not ideal)
            next_obs_tensor = obs_tensor
            dones = torch.zeros_like(rewards)

        # Current Q values
        current_q = self.q_network(obs_tensor).gather(1, actions_idx.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_obs_tensor).argmax(dim=1)
            next_q = self.target_network(next_obs_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {}, state, {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'epsilon': self.epsilon,
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
        }

    def load(self, data):
        self.q_network.load_state_dict(data['q_network'])
        self.target_network.load_state_dict(data['target_network'])
        self.epsilon = data.get('epsilon', self.epsilon)

    def sync(self):
        pass  # for multi-GPU setting
