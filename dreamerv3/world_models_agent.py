"""
World Models Agent (Ha & Schmidhuber, 2018)
Simplified world model: VAE (encoder) + RNN (dynamics) + Controller (policy)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class VAE(nn.Module):
    """Variational Autoencoder for vision encoding"""
    def __init__(self, obs_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class RNNDynamics(nn.Module):
    """RNN-based world model dynamics"""
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # RNN for state transition
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Predict next latent state
        self.fc_next_latent = nn.Linear(hidden_dim, latent_dim)

        # Predict reward
        self.fc_reward = nn.Linear(hidden_dim, 1)

        # Predict done
        self.fc_done = nn.Linear(hidden_dim, 1)

    def forward(self, latent, action, hidden):
        x = torch.cat([latent, action], dim=-1)
        hidden = self.rnn(x, hidden)

        next_latent = self.fc_next_latent(hidden)
        reward = self.fc_reward(hidden)
        done = torch.sigmoid(self.fc_done(hidden))

        return next_latent, reward, done, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class Controller(nn.Module):
    """Simple MLP controller (policy)"""
    def __init__(self, latent_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, latent, hidden):
        x = torch.cat([latent, hidden], dim=-1)
        return self.net(x)


class WorldModelsAgent:
    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use birdeye_wpt as input
        obs_key = 'birdeye_wpt'
        obs_shape = obs_space[obs_key].shape
        self.obs_key = obs_key
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space['action'].shape[0]
        self.num_actions = self.act_dim

        # Hyperparameters
        self.latent_dim = config.get('latent_dim', 32)
        self.hidden_dim = config.get('hidden_dim', 128)

        # Components
        self.vae = VAE(self.obs_dim, self.latent_dim).to(self.device)
        self.dynamics = RNNDynamics(self.latent_dim, self.act_dim, self.hidden_dim).to(self.device)
        self.controller = Controller(self.latent_dim, self.hidden_dim, self.act_dim).to(self.device)

        # Optimizers
        self.vae_optimizer = optim.Adam(self.vae.parameters(), lr=config.get('vae_lr', 1e-3))
        self.dynamics_optimizer = optim.Adam(self.dynamics.parameters(), lr=config.get('dynamics_lr', 1e-3))
        self.controller_optimizer = optim.Adam(self.controller.parameters(), lr=config.get('controller_lr', 3e-4))

        # State
        self.rnn_hidden = None

        # Training parameters
        self.imagination_horizon = config.get('imagination_horizon', 15)
        self.kl_weight = config.get('kl_weight', 0.1)

    def _preprocess_obs(self, obs):
        x = obs[self.obs_key]
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x.reshape(x.size(0), -1)

    def __call__(self, obs, state=None, mode='train'):
        return self.policy(obs, state, mode)

    def policy(self, obs, state=None, mode='train'):
        obs_tensor = self._preprocess_obs(obs)
        batch_size = obs_tensor.size(0)

        # Encode observation
        with torch.no_grad():
            mu, _ = self.vae.encode(obs_tensor)
            latent = mu  # Use mean for deterministic encoding

            # Initialize hidden state if needed
            if state is None or self.rnn_hidden is None:
                self.rnn_hidden = self.dynamics.init_hidden(batch_size, self.device)
            else:
                self.rnn_hidden = state

            # Get action from controller
            logits = self.controller(latent, self.rnn_hidden)

            if mode == 'eval':
                action_idx = logits.argmax(dim=-1)
            else:
                # Stochastic policy
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample()

        action_idx = action_idx.cpu().numpy()
        one_hot = np.zeros((len(action_idx), self.num_actions), dtype=np.float32)
        one_hot[np.arange(len(action_idx)), action_idx] = 1.0

        return {'action': one_hot, 'reset': np.array(False)}, self.rnn_hidden

    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        actions = torch.tensor(batch['action'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)

        batch_size = obs_tensor.size(0)

        # ===== Train VAE =====
        recon, mu, logvar = self.vae(obs_tensor)
        recon_loss = F.mse_loss(recon, obs_tensor)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        vae_loss = recon_loss + self.kl_weight * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # ===== Train Dynamics Model =====
        with torch.no_grad():
            latent, _ = self.vae.encode(obs_tensor)

        hidden = self.dynamics.init_hidden(batch_size, self.device)

        # Next observation (for dynamics training)
        if 'next_obs' in batch:
            next_obs_tensor = torch.tensor(batch['next_obs'][self.obs_key], dtype=torch.float32).to(self.device)
            next_obs_tensor = next_obs_tensor.reshape(next_obs_tensor.size(0), -1)
            with torch.no_grad():
                target_next_latent, _ = self.vae.encode(next_obs_tensor)
        else:
            target_next_latent = latent  # Fallback

        pred_next_latent, pred_reward, pred_done, hidden = self.dynamics(latent, actions, hidden)

        latent_loss = F.mse_loss(pred_next_latent, target_next_latent.detach())
        reward_loss = F.mse_loss(pred_reward.squeeze(), rewards)
        dynamics_loss = latent_loss + reward_loss

        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        # ===== Train Controller via Imagination =====
        with torch.no_grad():
            start_latent, _ = self.vae.encode(obs_tensor)

        imagined_reward = 0
        current_latent = start_latent.detach()
        current_hidden = self.dynamics.init_hidden(batch_size, self.device)

        for _ in range(self.imagination_horizon):
            # Get action from controller
            action_logits = self.controller(current_latent, current_hidden)
            action_probs = F.softmax(action_logits, dim=-1)

            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            action_one_hot = F.one_hot(action_idx, num_classes=self.num_actions).float()

            # Predict next state and reward
            next_latent, reward, done, current_hidden = self.dynamics(
                current_latent, action_one_hot, current_hidden
            )

            imagined_reward += reward.squeeze()
            current_latent = next_latent

        # Maximize imagined reward
        controller_loss = -imagined_reward.mean()

        self.controller_optimizer.zero_grad()
        controller_loss.backward()
        self.controller_optimizer.step()

        return {}, state, {
            'loss': (vae_loss + dynamics_loss + controller_loss).item(),
            'vae_loss': vae_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'dynamics_loss': dynamics_loss.item(),
            'controller_loss': controller_loss.item(),
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            'vae': self.vae.state_dict(),
            'dynamics': self.dynamics.state_dict(),
            'controller': self.controller.state_dict(),
        }

    def load(self, data):
        self.vae.load_state_dict(data['vae'])
        self.dynamics.load_state_dict(data['dynamics'])
        self.controller.load_state_dict(data['controller'])

    def sync(self):
        pass
