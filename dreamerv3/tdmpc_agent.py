"""
TD-MPC style planning baseline for ProSafeAV.

Every other world model in this study learns an *amortised* policy by
backpropagating through imagined rollouts.  This agent is the other main
model-based paradigm: it plans at decision time.  The latent dynamics are
trained without any reconstruction objective -- the latent is shaped purely by
what it needs to predict (reward, safety, its own next value) -- and actions are
chosen by sampling trajectories through the model and taking the best, in the
manner of TD-MPC / MPPI.

Why this matters for the manuscript's claim.  ProSafeAV is presented as a
*model-agnostic* coupling of EVT with a world model.  In a policy-gradient agent
the EVT term enters as a differentiable penalty on imagined states; here there
is no policy gradient at all, and the same EVT term enters directly into the
planner's objective as a cost on candidate trajectories.  Showing that both work
is what makes "model-agnostic" more than a word.

Planner
    For each decision, sample ``num_samples`` action sequences of length
    ``plan_horizon``, roll them through the latent model, score them by
    ``sum_t (reward_t - w_evt * evt_risk_t) + terminal_value``, keep the top
    ``num_elites``, refit a categorical distribution over actions, and iterate.
    The first action of the best sequence is executed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from evt_torch import EVTImaginationPenalty, SafetyHead


class LatentEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class LatentDynamics(nn.Module):
    """Deterministic latent transition.  No decoder: the latent is never asked to
    reconstruct the observation, only to support the prediction heads."""

    def __init__(self, latent_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, latent, action):
        return self.net(torch.cat([latent, action], dim=-1))


class ScalarHead(nn.Module):
    def __init__(self, latent_dim, hidden=256, act_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class TDMPCAgent:
    """ProSafeAV with decision-time planning instead of an amortised policy."""

    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_key = config.get("obs_key", "birdeye_wpt")
        self.obs_dim = int(np.prod(obs_space[self.obs_key].shape))
        self.act_dim = act_space["action"].shape[0]

        self.latent_dim = config.get("latent_dim", 64)
        self.plan_horizon = config.get("plan_horizon", 8)
        self.num_samples = config.get("num_samples", 128)
        self.num_elites = config.get("num_elites", 16)
        self.plan_iterations = config.get("plan_iterations", 4)
        self.discount = config.get("discount", 0.99)
        self.consistency_weight = config.get("consistency_weight", 2.0)

        self.encoder = LatentEncoder(self.obs_dim, self.latent_dim).to(self.device)
        self.dynamics = LatentDynamics(self.latent_dim, self.act_dim).to(self.device)
        self.reward_model = ScalarHead(self.latent_dim, act_dim=self.act_dim).to(self.device)
        self.value_model = ScalarHead(self.latent_dim).to(self.device)
        self.target_value = ScalarHead(self.latent_dim).to(self.device)
        self.target_value.load_state_dict(self.value_model.state_dict())
        self.safety_head = SafetyHead(self.latent_dim, hidden=64).to(self.device)
        self.evt = EVTImaginationPenalty(config)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_model.parameters())
            + list(self.value_model.parameters())
            + list(self.safety_head.parameters()),
            lr=config.get("model_lr", 3e-4),
        )
        self.tau = config.get("target_tau", 0.01)
        self._last_batch_size = 1
        self._last_evt_params = torch.zeros(1, 10, device=self.device)

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

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def plan(self, latent, evt_params, greedy=False):
        """Sample-based planner.  Returns the first action of the best sequence."""
        batch_size = latent.size(0)
        n, h = self.num_samples, self.plan_horizon

        # Categorical action distribution per horizon step, refit from the elites.
        logits = torch.zeros(batch_size, h, self.act_dim, device=self.device)

        for _ in range(self.plan_iterations):
            probs = F.softmax(logits, dim=-1)  # [B, H, A]
            flat = probs.unsqueeze(1).expand(batch_size, n, h, self.act_dim)
            idx = torch.distributions.Categorical(flat).sample()  # [B, n, H]
            actions = F.one_hot(idx, num_classes=self.act_dim).float()

            z = latent.unsqueeze(1).expand(batch_size, n, self.latent_dim).reshape(-1, self.latent_dim)
            params = evt_params.unsqueeze(1).expand(batch_size, n, 10).reshape(-1, 10)
            ret = torch.zeros(batch_size * n, device=self.device)
            gamma = 1.0
            for t in range(h):
                a = actions[:, :, t].reshape(-1, self.act_dim)
                reward = self.reward_model(torch.cat([z, a], dim=-1)).squeeze(-1)
                # The EVT tail risk enters the planner objective directly; there is
                # no policy gradient here for it to flow through.
                risk = self.evt.risk(self.safety_head(z), params)
                ret = ret + gamma * (reward - self.evt.weight * risk)
                z = self.dynamics(z, a)
                gamma *= self.discount
            ret = ret + gamma * self.target_value(z).squeeze(-1)

            ret = ret.reshape(batch_size, n)
            elite_idx = ret.topk(self.num_elites, dim=1).indices  # [B, E]
            elite_actions = torch.gather(
                actions, 1,
                elite_idx[:, :, None, None].expand(batch_size, self.num_elites, h, self.act_dim),
            )
            # Refit: elite action frequencies become the next proposal.
            counts = elite_actions.sum(dim=1) + 1e-3
            logits = torch.log(counts / counts.sum(dim=-1, keepdim=True))

        probs = F.softmax(logits[:, 0], dim=-1)
        return probs.argmax(dim=-1) if greedy else torch.distributions.Categorical(probs).sample()

    def policy(self, obs, state=None, mode="train"):
        obs_tensor = self._preprocess_obs(obs)
        batch_size = obs_tensor.size(0)
        with torch.no_grad():
            latent = self.encoder(obs_tensor)
            params = self._last_evt_params
            if params.size(0) != batch_size:
                params = params[:1].expand(batch_size, 10)
            action_idx = self.plan(latent, params, greedy=(mode == "eval"))
        out = np.zeros((batch_size, self.act_dim), dtype=np.float32)
        out[np.arange(batch_size), action_idx.cpu().numpy()] = 1.0
        return {"action": out, "reset": np.array(False)}, state

    # ------------------------------------------------------------------ #
    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        batch_size = obs_tensor.size(0)
        self._last_batch_size = batch_size

        actions = torch.tensor(np.asarray(batch["action"]), dtype=torch.float32, device=self.device)
        actions = actions.reshape(batch_size, -1)[:, : self.act_dim]
        rewards = torch.tensor(np.asarray(batch["reward"]), dtype=torch.float32, device=self.device)
        rewards = rewards.reshape(batch_size)

        if "next_obs" in batch:
            nxt = torch.tensor(np.asarray(batch["next_obs"][self.obs_key]),
                               dtype=torch.float32, device=self.device)
            next_obs_tensor = nxt.reshape(batch_size, -1)
        else:
            next_obs_tensor = obs_tensor

        evt_params = self._evt_batch(batch, "evt_params", 10)
        self._last_evt_params = evt_params.detach()
        safety_target = self._evt_batch(batch, "safety", 2)

        latent = self.encoder(obs_tensor)
        pred_next = self.dynamics(latent, actions)
        with torch.no_grad():
            target_next = self.encoder(next_obs_tensor)
            td_target = rewards + self.discount * self.target_value(target_next).squeeze(-1)

        # No reconstruction term: the latent is shaped only by what it must predict.
        consistency_loss = F.mse_loss(pred_next, target_next)
        reward_loss = F.mse_loss(
            self.reward_model(torch.cat([latent, actions], dim=-1)).squeeze(-1), rewards
        )
        value_loss = F.mse_loss(self.value_model(latent).squeeze(-1), td_target)
        safety_loss = F.mse_loss(self.safety_head(latent), safety_target)

        loss = (
            self.consistency_weight * consistency_loss
            + reward_loss
            + value_loss
            + safety_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_model.parameters())
            + list(self.value_model.parameters())
            + list(self.safety_head.parameters()),
            10.0,
        )
        self.optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.value_model.parameters(), self.target_value.parameters()):
                tp.data.lerp_(p.data, self.tau)

        with torch.no_grad():
            imag_risk = self.evt.risk(self.safety_head(pred_next), evt_params).mean()

        return {}, state, {
            "loss": loss.item(),
            "consistency_loss": consistency_loss.item(),
            "reward_loss": reward_loss.item(),
            "value_loss": value_loss.item(),
            "safety_loss": safety_loss.item(),
            "imag_evt_risk": float(imag_risk),
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {
            "encoder": self.encoder.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "target_value": self.target_value.state_dict(),
            "safety_head": self.safety_head.state_dict(),
        }

    def load(self, data):
        self.encoder.load_state_dict(data["encoder"])
        self.dynamics.load_state_dict(data["dynamics"])
        self.reward_model.load_state_dict(data["reward_model"])
        self.value_model.load_state_dict(data["value_model"])
        self.target_value.load_state_dict(data["target_value"])
        if "safety_head" in data:
            self.safety_head.load_state_dict(data["safety_head"])

    def sync(self):
        pass
