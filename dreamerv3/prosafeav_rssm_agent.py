"""
ProSafeAV-RSSM: Lightweight RSSM variant for ablation study
Simplified RSSM with reduced parameters while maintaining stochastic + deterministic structure

Key differences from DreamerV3:
- Smaller network sizes (256→128→64 instead of 512+)
- Simpler RSSM structure (single GRU + small stochastic layer)
- Focused on efficiency while keeping core RSSM benefits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LightweightEncoder(nn.Module):
    """Lightweight encoder for observations"""
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


class LightweightRSSM(nn.Module):
    """
    Simplified RSSM: Recurrent State-Space Model
    - Deterministic path: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    - Stochastic path: z_t ~ p(z_t | h_t)
    """
    def __init__(self, stochastic_dim=16, deterministic_dim=64, action_dim=5):
        super().__init__()
        self.stochastic_dim = stochastic_dim
        self.deterministic_dim = deterministic_dim

        # Deterministic state (GRU)
        self.gru = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)

        # Prior: p(z_t | h_t)
        self.fc_prior_mean = nn.Linear(deterministic_dim, stochastic_dim)
        self.fc_prior_std = nn.Linear(deterministic_dim, stochastic_dim)

        # Posterior: q(z_t | h_t, e_t) where e_t is encoded observation
        self.fc_posterior = nn.Linear(deterministic_dim + 64, 64)  # 64 is embed_dim
        self.fc_posterior_mean = nn.Linear(64, stochastic_dim)
        self.fc_posterior_std = nn.Linear(64, stochastic_dim)

    def init_state(self, batch_size, device):
        """Initialize RSSM state"""
        h = torch.zeros(batch_size, self.deterministic_dim, device=device)
        z = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return h, z

    def observe(self, prev_h, prev_z, action, embed):
        """Observation step: compute posterior given observation"""
        # Update deterministic state
        x = torch.cat([prev_z, action], dim=-1)
        h = self.gru(x, prev_h)

        # Compute prior
        prior_mean = self.fc_prior_mean(h)
        prior_std = F.softplus(self.fc_prior_std(h)) + 0.1

        # Compute posterior
        post_input = torch.cat([h, embed], dim=-1)
        post_hidden = F.relu(self.fc_posterior(post_input))
        post_mean = self.fc_posterior_mean(post_hidden)
        post_std = F.softplus(self.fc_posterior_std(post_hidden)) + 0.1

        # Sample from posterior
        z = post_mean + torch.randn_like(post_std) * post_std

        return h, z, prior_mean, prior_std, post_mean, post_std

    def imagine(self, prev_h, prev_z, action):
        """Imagination step: predict next state without observation"""
        # Update deterministic state
        x = torch.cat([prev_z, action], dim=-1)
        h = self.gru(x, prev_h)

        # Sample from prior
        prior_mean = self.fc_prior_mean(h)
        prior_std = F.softplus(self.fc_prior_std(h)) + 0.1
        z = prior_mean + torch.randn_like(prior_std) * prior_std

        return h, z


class LightweightDecoder(nn.Module):
    """Decode RSSM state to observation"""
    def __init__(self, state_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim),
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.net(state)


class LightweightRewardModel(nn.Module):
    """Predict reward from RSSM state"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.net(state)


class LightweightPolicy(nn.Module):
    """Policy network"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, h, z):
        state = torch.cat([h, z], dim=-1)
        return self.net(state)


class ProSafeAVRSSMAgent:
    """
    Lightweight RSSM agent for ProSafeAV
    - Maintains RSSM structure but with smaller networks
    - For ablation study: testing if lightweight RSSM is sufficient
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
        self.stochastic_dim = config.get('stochastic_dim', 16)
        self.deterministic_dim = config.get('deterministic_dim', 64)
        self.embed_dim = 64

        # Components
        self.encoder = LightweightEncoder(self.obs_dim, self.embed_dim).to(self.device)
        self.rssm = LightweightRSSM(self.stochastic_dim, self.deterministic_dim, self.act_dim).to(self.device)
        self.decoder = LightweightDecoder(self.stochastic_dim + self.deterministic_dim, self.obs_dim).to(self.device)
        self.reward_model = LightweightRewardModel(self.stochastic_dim + self.deterministic_dim).to(self.device)
        self.policy = LightweightPolicy(self.stochastic_dim + self.deterministic_dim, self.act_dim).to(self.device)

        # Optimizers
        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
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
        self.kl_weight = config.get('kl_weight', 0.1)

        # State
        self.rssm_state = None

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

            # Initialize or use existing RSSM state
            if state is None:
                h, z = self.rssm.init_state(batch_size, self.device)
            else:
                h, z = state

            # Observe and update state (using posterior)
            action_dummy = torch.zeros(batch_size, self.act_dim, device=self.device)
            h, z, _, _, _, _ = self.rssm.observe(h, z, action_dummy, embed)

            # Get action from policy
            logits = self.policy(h, z)

            if mode == 'eval':
                action_idx = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()

        action_idx = action_idx.cpu().numpy()
        one_hot = np.zeros((len(action_idx), self.act_dim), dtype=np.float32)
        one_hot[np.arange(len(action_idx)), action_idx] = 1.0

        return {'action': one_hot, 'reset': np.array(False)}, (h, z)

    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        actions = torch.tensor(batch['action'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)
        batch_size = obs_tensor.size(0)

        # ===== Train World Model =====
        # Encode
        embed = self.encoder(obs_tensor)

        # Initialize RSSM state
        h, z = self.rssm.init_state(batch_size, self.device)

        # Observe and compute losses
        h, z, prior_mean, prior_std, post_mean, post_std = self.rssm.observe(h, z, actions, embed)

        # Reconstruction loss
        recon_obs = self.decoder(h, z)
        recon_loss = F.mse_loss(recon_obs, obs_tensor)

        # KL divergence loss
        kl_loss = torch.distributions.kl_divergence(
            torch.distributions.Normal(post_mean, post_std),
            torch.distributions.Normal(prior_mean, prior_std)
        ).mean()

        # Reward prediction loss
        pred_reward = self.reward_model(h, z).squeeze(-1)
        reward_loss = F.mse_loss(pred_reward, rewards)

        # Total model loss
        model_loss = recon_loss + self.kl_weight * kl_loss + reward_loss

        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.decoder.parameters()) +
            list(self.reward_model.parameters()),
            10.0
        )
        self.model_optimizer.step()

        # ===== Train Policy via Imagination =====
        with torch.no_grad():
            h_start, z_start = self.rssm.init_state(batch_size, self.device)
            embed_start = self.encoder(obs_tensor)
            h_start, z_start, _, _, _, _ = self.rssm.observe(h_start, z_start, actions, embed_start)

        imagined_reward = 0
        h_imag, z_imag = h_start.detach(), z_start.detach()

        for _ in range(self.imagination_horizon):
            # Get action from policy
            action_logits = self.policy(h_imag, z_imag)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            action_one_hot = F.one_hot(action_idx, num_classes=self.act_dim).float()

            # Imagine next state
            h_imag, z_imag = self.rssm.imagine(h_imag, z_imag, action_one_hot)

            # Predict reward
            reward = self.reward_model(h_imag, z_imag).squeeze(-1)
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
            'kl_loss': kl_loss.item(),
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
            'rssm': self.rssm.state_dict(),
            'decoder': self.decoder.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'policy': self.policy.state_dict(),
        }

    def load(self, data):
        self.encoder.load_state_dict(data['encoder'])
        self.rssm.load_state_dict(data['rssm'])
        self.decoder.load_state_dict(data['decoder'])
        self.reward_model.load_state_dict(data['reward_model'])
        self.policy.load_state_dict(data['policy'])

    def sync(self):
        pass
