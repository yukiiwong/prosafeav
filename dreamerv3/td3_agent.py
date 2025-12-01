import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeterministicActor(nn.Module):
    """Deterministic policy for discrete actions (outputs logits)"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.logits = nn.Linear(128, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        logits = self.logits(x)
        # Use softmax for discrete action probabilities
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """Twin Q-networks for TD3"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class TD3Agent:
    """
    TD3-style agent adapted for discrete actions.
    Key features:
    - Twin Q-networks (like TD3)
    - Delayed policy updates
    - Target policy smoothing (adapted for discrete actions)
    """
    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use birdeye_wpt as input
        obs_key = 'birdeye_wpt'
        obs_shape = obs_space[obs_key].shape
        self.obs_key = obs_key
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space['action'].shape[0]
        self.num_actions = self.act_dim

        # Actor networks
        self.actor = DeterministicActor(self.obs_dim, self.act_dim).to(self.device)
        self.actor_target = DeterministicActor(self.obs_dim, self.act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Twin Critics
        self.critic1 = CriticNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(self.device)

        # Target Critics
        self.critic1_target = CriticNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get('actor_lr', 3e-4))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.get('critic_lr', 3e-4))
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.get('critic_lr', 3e-4))

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.policy_delay = config.get('policy_delay', 2)
        self.policy_noise = config.get('policy_noise', 0.1)  # for exploration
        self.noise_clip = config.get('noise_clip', 0.2)

        # Exploration noise (for discrete actions, we use epsilon-greedy style)
        self.epsilon = config.get('epsilon_start', 0.3)
        self.epsilon_min = config.get('epsilon_min', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.9995)

        self.train_step_counter = 0

    def _preprocess_obs(self, obs):
        x = obs[self.obs_key]
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x.reshape(x.size(0), -1)

    def __call__(self, obs, state=None, mode='train'):
        return self.policy(obs, state, mode)

    def policy(self, obs, state=None, mode='train'):
        obs_tensor = self._preprocess_obs(obs)

        with torch.no_grad():
            action_probs = self.actor(obs_tensor)

            if mode == 'train' and np.random.rand() < self.epsilon:
                # Exploration: random action
                action_idx = np.random.randint(0, self.num_actions, size=obs_tensor.size(0))
            else:
                # Exploitation: select action with highest probability
                action_idx = action_probs.argmax(dim=-1).cpu().numpy()

        one_hot = np.zeros((len(action_idx), self.num_actions), dtype=np.float32)
        one_hot[np.arange(len(action_idx)), action_idx] = 1.0

        return {'action': one_hot, 'reset': np.array(False)}, state

    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        actions = torch.tensor(batch['action'], dtype=torch.float32).to(self.device)
        actions_idx = torch.argmax(actions, dim=1)
        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)

        # Get next observations
        if 'next_obs' in batch:
            next_obs_tensor = torch.tensor(batch['next_obs'][self.obs_key], dtype=torch.float32).to(self.device)
            next_obs_tensor = next_obs_tensor.reshape(next_obs_tensor.size(0), -1)
            dones = torch.tensor(batch.get('done', np.zeros_like(rewards.cpu().numpy())), dtype=torch.float32).to(self.device)
        else:
            next_obs_tensor = obs_tensor
            dones = torch.zeros_like(rewards)

        # ===== Update Critics =====
        with torch.no_grad():
            # Target policy smoothing (adapted for discrete: use target actor with slight randomness)
            next_action_probs = self.actor_target(next_obs_tensor)

            # Add noise to probabilities for smoothing
            noise = torch.randn_like(next_action_probs) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action_probs = F.softmax(torch.log(next_action_probs + 1e-8) + noise, dim=-1)

            # Get Q-values from target critics
            target_q1 = self.critic1_target(next_obs_tensor)
            target_q2 = self.critic2_target(next_obs_tensor)

            # Minimum of twin Q-values (expected value over action distribution)
            target_q = torch.min(target_q1, target_q2)
            next_v = (next_action_probs * target_q).sum(dim=-1)

            target_q_value = rewards + self.gamma * next_v * (1 - dones)

        # Current Q estimates
        current_q1 = self.critic1(obs_tensor).gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        current_q2 = self.critic2(obs_tensor).gather(1, actions_idx.unsqueeze(1)).squeeze(1)

        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q_value)
        critic2_loss = F.mse_loss(current_q2, target_q_value)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ===== Delayed Policy Update =====
        actor_loss_value = 0.0
        if self.train_step_counter % self.policy_delay == 0:
            # Actor loss: maximize Q-value
            action_probs = self.actor(obs_tensor)
            q1_values = self.critic1(obs_tensor)

            # Policy loss: maximize expected Q-value
            actor_loss = -(action_probs * q1_values).sum(dim=-1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_value = actor_loss.item()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        # Decay exploration noise
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.train_step_counter += 1

        return {}, state, {
            'loss': (critic1_loss + critic2_loss).item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss_value,
            'epsilon': self.epsilon,
        }

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'epsilon': self.epsilon,
        }

    def load(self, data):
        self.actor.load_state_dict(data['actor'])
        self.actor_target.load_state_dict(data['actor_target'])
        self.critic1.load_state_dict(data['critic1'])
        self.critic2.load_state_dict(data['critic2'])
        self.critic1_target.load_state_dict(data['critic1_target'])
        self.critic2_target.load_state_dict(data['critic2_target'])
        self.epsilon = data.get('epsilon', self.epsilon)

    def sync(self):
        pass
