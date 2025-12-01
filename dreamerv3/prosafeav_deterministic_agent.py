"""
ProSafeAV-Deterministic: Fully deterministic latent dynamics for ablation study
NO stochastic sampling - pure deterministic state transitions

Key differences from RSSM:
- No stochastic latent variables (no sampling)
- Only deterministic recurrent state (GRU)
- Simpler, faster, but potentially less expressive

Research question: Is stochasticity necessary for world models?
Hypothesis: With EVT handling uncertainty, deterministic latent may be sufficient
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeterministicEncoder(nn.Module):
    """Encode observations to embedding"""
    def __init__(self, obs_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


class DeterministicDynamics(nn.Module):
    """
    Fully deterministic dynamics model
    h_t = GRU(h_{t-1}, [e_t, a_{t-1}])
    No stochastic sampling!
    """
    def __init__(self, embed_dim=64, action_dim=5, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Deterministic transition (GRU)
        self.gru = nn.GRUCell(embed_dim + action_dim, hidden_dim)

    def init_state(self, batch_size, device):
        """Initialize hidden state"""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, h, embed, action):
        """
        Deterministic state transition
        No sampling, no distributions - pure deterministic!
        """
        x = torch.cat([embed, action], dim=-1)
        h_next = self.gru(x, h)
        return h_next


class DeterministicDecoder(nn.Module):
    """Decode deterministic state to observation"""
    def __init__(self, hidden_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim),
        )

    def forward(self, h):
        return self.net(h)


class DeterministicRewardModel(nn.Module):
    """Predict reward from deterministic state"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return self.net(h)


class DeterministicPolicy(nn.Module):
    """Policy network (can still be stochastic in action space)"""
    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, h):
        return self.net(h)


class ProSafeAVDeterministicAgent:
    """
    Fully deterministic world model for ProSafeAV

    Ablation study: Testing if stochastic latent is necessary
    - Deterministic latent dynamics (GRU only)
    - No VAE, no sampling, no KL divergence
    - Simpler and potentially faster

    Motivation:
    - EVT module already handles uncertainty in safety
    - Deterministic models may be sufficient for planning
    - Easier to interpret and debug
    """
    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Observation and action spaces
        obs_key = 'birdeye_wpt'
        obs_shape = obs_space[obs_key].shape
        self.obs_key = obs_key
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space['action'].shape[0]

        # Hyperparameters
        self.embed_dim = config.get('embed_dim', 64)
        self.hidden_dim = config.get('hidden_dim', 128)

        # Components (all deterministic!)
        self.encoder = DeterministicEncoder(self.obs_dim, self.embed_dim).to(self.device)
        self.dynamics = DeterministicDynamics(self.embed_dim, self.act_dim, self.hidden_dim).to(self.device)
        self.decoder = DeterministicDecoder(self.hidden_dim, self.obs_dim).to(self.device)
        self.reward_model = DeterministicRewardModel(self.hidden_dim).to(self.device)
        self.policy = DeterministicPolicy(self.hidden_dim, self.act_dim).to(self.device)

        # Optimizers
        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()),
            lr=config.get('model_lr', 1e-3)
        )
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.get('policy_lr', 3e-4)
        )

        # Training parameters
        self.imagination_horizon = config.get('imagination_horizon', 10)

        # State
        self.hidden_state = None

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
            # Encode observation
            embed = self.encoder(obs_tensor)

            # Initialize or use existing state
            if state is None:
                h = self.dynamics.init_state(batch_size, self.device)
            else:
                h = state

            # Update deterministic state
            action_dummy = torch.zeros(batch_size, self.act_dim, device=self.device)
            h = self.dynamics(h, embed, action_dummy)

            # Get action from policy
            logits = self.policy(h)

            if mode == 'eval':
                action_idx = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()

        action_idx = action_idx.cpu().numpy()
        one_hot = np.zeros((len(action_idx), self.act_dim), dtype=np.float32)
        one_hot[np.arange(len(action_idx)), action_idx] = 1.0

        return {'action': one_hot, 'reset': np.array(False)}, h

    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        actions = torch.tensor(batch['action'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)
        batch_size = obs_tensor.size(0)

        # Get next observations for dynamics training
        if 'next_obs' in batch:
            next_obs_tensor = torch.tensor(batch['next_obs'][self.obs_key], dtype=torch.float32).to(self.device)
            next_obs_tensor = next_obs_tensor.reshape(next_obs_tensor.size(0), -1)
        else:
            next_obs_tensor = obs_tensor

        # ===== Train World Model =====
        # Encode
        embed = self.encoder(obs_tensor)
        next_embed = self.encoder(next_obs_tensor)

        # Initialize state
        h = self.dynamics.init_state(batch_size, self.device)

        # Predict next state (deterministic)
        h_next = self.dynamics(h, embed, actions)

        # Reconstruction loss (decode current state)
        recon_obs = self.decoder(h_next)
        recon_loss = F.mse_loss(recon_obs, next_obs_tensor)

        # Consistency loss (next hidden state should match next observation embedding)
        # This replaces the KL loss from RSSM
        consistency_loss = F.mse_loss(h_next, next_embed.detach())

        # Reward prediction loss
        pred_reward = self.reward_model(h_next).squeeze(-1)
        reward_loss = F.mse_loss(pred_reward, rewards)

        # Total model loss (no KL divergence - we're deterministic!)
        model_loss = recon_loss + 0.1 * consistency_loss + reward_loss

        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.dynamics.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()),
            10.0
        )
        self.model_optimizer.step()

        # ===== Train Policy via Imagination =====
        with torch.no_grad():
            h_start = self.dynamics.init_state(batch_size, self.device)
            embed_start = self.encoder(obs_tensor)
            h_start = self.dynamics(h_start, embed_start, actions)

        imagined_reward = 0
        h_imag = h_start.detach()

        for _ in range(self.imagination_horizon):
            # Get action from policy
            action_logits = self.policy(h_imag)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            action_one_hot = F.one_hot(action_idx, num_classes=self.act_dim).float()

            # Imagine next state (deterministic transition)
            # For imagination, we use zero embedding (or could use decoder output)
            embed_imag = torch.zeros(batch_size, self.embed_dim, device=self.device)
            h_imag = self.dynamics(h_imag, embed_imag, action_one_hot)

            # Predict reward
            reward = self.reward_model(h_imag).squeeze(-1)
            imagined_reward += reward

        # Policy loss: maximize imagined reward
        policy_loss = -imagined_reward.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {}, state, {
            'loss': (model_loss + policy_loss).item(),
            'model_loss': model_loss.item(),
            'recon_loss': recon_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'reward_loss': reward_loss.item(),
            'policy_loss': policy_loss.item(),
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            'encoder': self.encoder.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'policy': self.policy.state_dict(),
        }

    def load(self, data):
        self.encoder.load_state_dict(data['encoder'])
        self.dynamics.load_state_dict(data['dynamics'])
        self.decoder.load_state_dict(data['decoder'])
        self.reward_model.load_state_dict(data['reward_model'])
        self.policy.load_state_dict(data['policy'])

    def sync(self):
        pass
