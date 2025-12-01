"""
PlaNet Agent (Deep Planning Network, Hafner et al. 2019)
Simplified latent dynamics model with planning via CEM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Encoder(nn.Module):
    """Encode observations to latent space"""
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_std = nn.Linear(128, latent_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.fc_mu(h)
        std = F.softplus(self.fc_std(h)) + 0.1
        return mu, std


class TransitionModel(nn.Module):
    """Deterministic state transition model"""
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_std = nn.Linear(128, latent_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)
        mu = self.fc_mu(h)
        std = F.softplus(self.fc_std(h)) + 0.1
        return mu, std


class ObservationModel(nn.Module):
    """Decode latent state to observation"""
    def __init__(self, latent_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, state):
        return self.net(state)


class RewardModel(nn.Module):
    """Predict reward from latent state"""
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state)


class PlaNetAgent:
    """
    Simplified PlaNet agent with Cross-Entropy Method (CEM) planning
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

        # Model parameters
        self.latent_dim = config.get('latent_dim', 32)

        # World model components
        self.encoder = Encoder(self.obs_dim, self.latent_dim).to(self.device)
        self.transition = TransitionModel(self.latent_dim, self.act_dim).to(self.device)
        self.observation = ObservationModel(self.latent_dim, self.obs_dim).to(self.device)
        self.reward = RewardModel(self.latent_dim).to(self.device)

        # Optimizer
        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.transition.parameters()) +
            list(self.observation.parameters()) +
            list(self.reward.parameters()),
            lr=config.get('learning_rate', 1e-3)
        )

        # CEM planning parameters
        self.planning_horizon = config.get('planning_horizon', 12)
        self.num_candidates = config.get('num_candidates', 100)
        self.num_iterations = config.get('num_iterations', 10)
        self.num_elite = config.get('num_elite', 10)

        # Current state
        self.current_state = None

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
            # Encode current observation
            state_mu, state_std = self.encoder(obs_tensor)
            current_state = state_mu  # Use mean for planning

            if mode == 'train':
                # Simple random action for exploration during training
                if np.random.rand() < 0.1:
                    action_idx = np.random.randint(0, self.num_actions, size=batch_size)
                else:
                    # Plan with CEM
                    action_idx = self._plan_cem(current_state)
            else:
                # Plan with CEM during evaluation
                action_idx = self._plan_cem(current_state)

        one_hot = np.zeros((len(action_idx), self.num_actions), dtype=np.float32)
        one_hot[np.arange(len(action_idx)), action_idx] = 1.0

        return {'action': one_hot, 'reset': np.array(False)}, current_state

    def _plan_cem(self, initial_state):
        """Cross-Entropy Method planning"""
        batch_size = initial_state.size(0)
        device = initial_state.device

        # Initialize action distribution (uniform over discrete actions)
        action_probs = torch.ones(batch_size, self.planning_horizon, self.num_actions, device=device)
        action_probs = action_probs / self.num_actions

        for _ in range(self.num_iterations):
            # Sample action sequences
            action_sequences = []
            for b in range(batch_size):
                seq = []
                for h in range(self.planning_horizon):
                    dist = torch.distributions.Categorical(probs=action_probs[b, h])
                    action_idx = dist.sample((self.num_candidates,))
                    seq.append(F.one_hot(action_idx, num_classes=self.num_actions).float())
                action_sequences.append(torch.stack(seq, dim=1))  # [num_candidates, horizon, act_dim]

            # Evaluate each candidate
            returns = []
            for b in range(batch_size):
                state = initial_state[b:b+1].expand(self.num_candidates, -1)
                total_reward = torch.zeros(self.num_candidates, device=device)

                for h in range(self.planning_horizon):
                    action = action_sequences[b][:, h]  # [num_candidates, act_dim]
                    next_state_mu, _ = self.transition(state, action)
                    reward = self.reward(next_state_mu).squeeze(-1)
                    total_reward += reward
                    state = next_state_mu

                returns.append(total_reward)

            # Select elite candidates and update distribution
            best_actions = []
            for b in range(batch_size):
                elite_idx = torch.topk(returns[b], self.num_elite).indices
                elite_actions = action_sequences[b][elite_idx]  # [num_elite, horizon, act_dim]

                # Update action distribution based on elite samples
                for h in range(self.planning_horizon):
                    elite_h = elite_actions[:, h].argmax(dim=-1)
                    counts = torch.bincount(elite_h, minlength=self.num_actions).float()
                    action_probs[b, h] = (counts + 1) / (counts.sum() + self.num_actions)

                # Select first action from best candidate
                best_idx = elite_idx[0]
                best_first_action = action_sequences[b][best_idx, 0].argmax().item()
                best_actions.append(best_first_action)

        return np.array(best_actions)

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

        # Encode observations
        state_mu, state_std = self.encoder(obs_tensor)
        state = state_mu + torch.randn_like(state_std) * state_std

        next_state_mu, next_state_std = self.encoder(next_obs_tensor)

        # Predict next state via transition model
        pred_next_mu, pred_next_std = self.transition(state, actions)

        # Predict observation from latent state
        pred_obs = self.observation(state)

        # Predict reward
        pred_reward = self.reward(state).squeeze(-1)

        # Compute losses
        # Transition loss (match predicted next state with encoded next state)
        transition_loss = F.mse_loss(pred_next_mu, next_state_mu.detach())

        # Observation reconstruction loss
        obs_loss = F.mse_loss(pred_obs, obs_tensor)

        # Reward prediction loss
        reward_loss = F.mse_loss(pred_reward, rewards)

        # Total loss
        total_loss = transition_loss + obs_loss + reward_loss

        # Optimize
        self.model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.transition.parameters()) +
            list(self.observation.parameters()) +
            list(self.reward.parameters()),
            10.0
        )
        self.model_optimizer.step()

        return {}, state, {
            'loss': total_loss.item(),
            'transition_loss': transition_loss.item(),
            'obs_loss': obs_loss.item(),
            'reward_loss': reward_loss.item(),
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            'encoder': self.encoder.state_dict(),
            'transition': self.transition.state_dict(),
            'observation': self.observation.state_dict(),
            'reward': self.reward.state_dict(),
        }

    def load(self, data):
        self.encoder.load_state_dict(data['encoder'])
        self.transition.load_state_dict(data['transition'])
        self.observation.load_state_dict(data['observation'])
        self.reward.load_state_dict(data['reward'])

    def sync(self):
        pass
