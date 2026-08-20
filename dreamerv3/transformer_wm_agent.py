"""
Transformer world model baseline for ProSafeAV.

Reviewer 1 asks for cross-architecture comparison against other mainstream world
models.  The variants already in this repository (DreamerV3, PlaNet, the
VAE+RNN World Models agent) all propagate the latent state through a *recurrent*
cell, so comparing them says little about whether the EVT coupling depends on
the recurrence.  This agent replaces the recurrence with a **causal Transformer**
over the latent sequence, which is the backbone used by TransDreamer and by the
IRIS/STORM family.

Architecture
    obs -> encoder -> latent token
    [latent, action] tokens -> causal Transformer -> next-latent prediction
    next latent -> reward head, safety head, decoder

Compared with an RSSM the differences that matter for this study are:
  * the state is the *sequence* of past tokens rather than a fixed-size carry, so
    the model can attend directly to a conflict several steps back instead of
    having to keep it alive in a hidden vector;
  * imagination is autoregressive over a growing context rather than a repeated
    cell application, so prediction error accumulates differently -- which is
    exactly what the EVT term is sensitive to.

The EVT coupling is identical to every other variant: a safety head regresses
the normalised (TTC, DRAC) pair from the latent, and the tail risk of the
*imagined* latents is subtracted from the imagined return.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from evt_torch import EVTImaginationPenalty, SafetyHead


class ObsEncoder(nn.Module):
    """Flattened BEV -> latent token."""

    def __init__(self, obs_dim, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class ObsDecoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, latent):
        return self.net(latent)


class CausalTransformerDynamics(nn.Module):
    """Causal Transformer over interleaved latent/action tokens.

    A single ``nn.TransformerEncoder`` with a causal mask is used rather than a
    decoder stack: there is no cross-attention to a separate memory here, the
    sequence attends only to its own past.
    """

    def __init__(self, latent_dim, act_dim, d_model=128, nhead=4, layers=2,
                 max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token = nn.Linear(latent_dim + act_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, latent_dim)

    def forward(self, latents, actions):
        """``latents``/``actions``: ``[B, T, *]``.  Returns next latents ``[B, T, latent]``."""
        seq_len = latents.size(1)
        assert seq_len <= self.max_len, f"sequence {seq_len} exceeds max_len {self.max_len}"
        x = self.token(torch.cat([latents, actions], dim=-1)) + self.pos[:, :seq_len]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=latents.device)
        return self.head(self.blocks(x, mask=mask, is_causal=True))

    def step(self, latent_hist, action_hist):
        """One autoregressive step: predict the latent after the last token."""
        if latent_hist.size(1) > self.max_len:
            latent_hist = latent_hist[:, -self.max_len:]
            action_hist = action_hist[:, -self.max_len:]
        return self.forward(latent_hist, action_hist)[:, -1]


class RewardHead(nn.Module):
    def __init__(self, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent):
        return self.net(latent)


class Policy(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, latent):
        return self.net(latent)


class TransformerWorldModelAgent:
    """ProSafeAV with a causal-Transformer latent dynamics backbone."""

    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_key = config.get("obs_key", "birdeye_wpt")
        obs_shape = obs_space[self.obs_key].shape
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space["action"].shape[0]

        self.latent_dim = config.get("latent_dim", 64)
        self.context_len = config.get("context_len", 16)
        self.imagination_horizon = config.get("imagination_horizon", 10)

        self.encoder = ObsEncoder(self.obs_dim, self.latent_dim).to(self.device)
        self.decoder = ObsDecoder(self.latent_dim, self.obs_dim).to(self.device)
        self.dynamics = CausalTransformerDynamics(
            self.latent_dim, self.act_dim,
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            layers=config.get("transformer_layers", 2),
            max_len=self.context_len + self.imagination_horizon + 2,
        ).to(self.device)
        self.reward_model = RewardHead(self.latent_dim).to(self.device)
        self.safety_head = SafetyHead(self.latent_dim, hidden=64).to(self.device)
        self.policy_net = Policy(self.latent_dim, self.act_dim).to(self.device)
        self.evt = EVTImaginationPenalty(config)

        self.model_optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_model.parameters())
            + list(self.safety_head.parameters()),
            lr=config.get("model_lr", 3e-4),
        )
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.get("policy_lr", 3e-4)
        )
        self._last_batch_size = 1

    # ------------------------------------------------------------------ #
    def _preprocess_obs(self, obs):
        x = torch.tensor(np.asarray(obs[self.obs_key]), dtype=torch.float32, device=self.device)
        return x.reshape(x.size(0), -1)

    def _evt_batch(self, batch, key, dim, default=0.0):
        if key not in batch:
            return torch.full((self._last_batch_size, dim), default, device=self.device)
        value = torch.tensor(np.asarray(batch[key]), dtype=torch.float32, device=self.device)
        if value.dim() == 3:
            value = value[:, 0]
        return value.reshape(-1, dim)[: self._last_batch_size]

    def __call__(self, obs, state=None, mode="train"):
        return self.policy(obs, state, mode)

    def policy(self, obs, state=None, mode="train"):
        """``state`` carries the (latent, action) token history for the context window."""
        obs_tensor = self._preprocess_obs(obs)
        batch_size = obs_tensor.size(0)

        with torch.no_grad():
            latent = self.encoder(obs_tensor)
            if state is None:
                lat_hist = latent.unsqueeze(1)
                act_hist = torch.zeros(batch_size, 1, self.act_dim, device=self.device)
            else:
                lat_hist, act_hist = state
                if lat_hist.size(0) != batch_size:  # episode boundary
                    lat_hist = latent.unsqueeze(1)
                    act_hist = torch.zeros(batch_size, 1, self.act_dim, device=self.device)
                else:
                    lat_hist = torch.cat([lat_hist, latent.unsqueeze(1)], dim=1)[:, -self.context_len:]
                    act_hist = act_hist[:, -self.context_len:]
                    if act_hist.size(1) < lat_hist.size(1):
                        pad = torch.zeros(batch_size, lat_hist.size(1) - act_hist.size(1),
                                          self.act_dim, device=self.device)
                        act_hist = torch.cat([act_hist, pad], dim=1)

            logits = self.policy_net(latent)
            if mode == "eval":
                action_idx = logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                action_idx = dist.sample()
            one_hot = F.one_hot(action_idx, num_classes=self.act_dim).float()
            act_hist = torch.cat([act_hist[:, :-1], one_hot.unsqueeze(1)], dim=1)

        out = np.zeros((batch_size, self.act_dim), dtype=np.float32)
        out[np.arange(batch_size), action_idx.cpu().numpy()] = 1.0
        return {"action": out, "reset": np.array(False)}, (lat_hist, act_hist)

    # ------------------------------------------------------------------ #
    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        batch_size = obs_tensor.size(0)
        self._last_batch_size = batch_size

        actions = torch.tensor(np.asarray(batch["action"]), dtype=torch.float32, device=self.device)
        actions = actions.reshape(batch_size, -1)[:, : self.act_dim]
        rewards = torch.tensor(np.asarray(batch["reward"]), dtype=torch.float32, device=self.device)
        rewards = rewards.reshape(batch_size)

        # ---- world model ------------------------------------------------ #
        latent = self.encoder(obs_tensor)
        lat_seq = latent.unsqueeze(1)
        act_seq = actions.unsqueeze(1)
        next_latent = self.dynamics(lat_seq, act_seq)[:, -1]

        recon_loss = F.mse_loss(self.decoder(latent), obs_tensor)
        reward_loss = F.mse_loss(self.reward_model(next_latent).squeeze(-1), rewards)
        # Latent consistency keeps the predicted latent on the encoder manifold;
        # without it the Transformer is free to drift into an unconstrained space.
        consistency_loss = F.mse_loss(next_latent, latent.detach())

        safety_target = self._evt_batch(batch, "safety", 2)
        safety_loss = F.mse_loss(self.safety_head(latent), safety_target)

        model_loss = recon_loss + reward_loss + 0.1 * consistency_loss + safety_loss
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_model.parameters())
            + list(self.safety_head.parameters()),
            10.0,
        )
        self.model_optimizer.step()

        # ---- policy via autoregressive imagination ---------------------- #
        evt_params = self._evt_batch(batch, "evt_params", 10)
        with torch.no_grad():
            lat_hist = self.encoder(obs_tensor).unsqueeze(1)
        act_hist = torch.zeros(batch_size, 1, self.act_dim, device=self.device)

        imagined_reward, imagined_risk = 0.0, 0.0
        for _ in range(self.imagination_horizon):
            cur = lat_hist[:, -1]
            logits = self.policy_net(cur)
            dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            one_hot = F.one_hot(dist.sample(), num_classes=self.act_dim).float()
            act_hist = torch.cat([act_hist[:, :-1], one_hot.unsqueeze(1)], dim=1)

            nxt = self.dynamics.step(lat_hist, act_hist)
            lat_hist = torch.cat([lat_hist, nxt.unsqueeze(1)], dim=1)
            act_hist = torch.cat([act_hist, torch.zeros_like(one_hot).unsqueeze(1)], dim=1)

            reward = self.reward_model(nxt).squeeze(-1)
            risk = self.evt.risk(self.safety_head(nxt), evt_params)
            imagined_risk = imagined_risk + risk
            imagined_reward = imagined_reward + reward - self.evt.weight * risk

        policy_loss = -imagined_reward.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.policy_optimizer.step()

        return {}, state, {
            "loss": (model_loss + policy_loss).item(),
            "model_loss": model_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "safety_loss": safety_loss.item(),
            "imag_evt_risk": float(imagined_risk.mean()) if torch.is_tensor(imagined_risk) else 0.0,
            "policy_loss": policy_loss.item(),
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "safety_head": self.safety_head.state_dict(),
            "policy": self.policy_net.state_dict(),
        }

    def load(self, data):
        self.encoder.load_state_dict(data["encoder"])
        self.decoder.load_state_dict(data["decoder"])
        self.dynamics.load_state_dict(data["dynamics"])
        self.reward_model.load_state_dict(data["reward_model"])
        self.policy_net.load_state_dict(data["policy"])
        if "safety_head" in data:
            self.safety_head.load_state_dict(data["safety_head"])

    def sync(self):
        pass
