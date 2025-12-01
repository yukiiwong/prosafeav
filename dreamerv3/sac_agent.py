import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    """Stochastic policy network for discrete actions"""
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
        return logits

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        log_prob = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_prob).sum(dim=-1)

        return action, probs, entropy


class QNetwork(nn.Module):
    """Q-network for discrete actions"""
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


class SACAgent:
    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use birdeye_wpt as input
        obs_key = 'birdeye_wpt'
        obs_shape = obs_space[obs_key].shape
        self.obs_key = obs_key
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space['action'].shape[0]
        self.num_actions = self.act_dim

        # Networks
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)

        # Twin Q-networks
        self.q1 = QNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.act_dim).to(self.device)

        # Target Q-networks
        self.q1_target = QNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.q2_target = QNetwork(self.obs_dim, self.act_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.get('actor_lr', 3e-4))
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.get('critic_lr', 3e-4))
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.get('critic_lr', 3e-4))

        # Automatic entropy tuning
        self.target_entropy = -np.log(1.0 / self.act_dim) * 0.98  # target entropy for discrete
        self.log_alpha = torch.tensor(np.log(config.get('init_alpha', 0.2)), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.get('alpha_lr', 3e-4))

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
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
            action_idx, probs, _ = self.actor.get_action(obs_tensor, deterministic=(mode == 'eval'))

        action_idx = action_idx.cpu().numpy()
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

        alpha = self.log_alpha.exp()

        # ===== Update Critics =====
        with torch.no_grad():
            _, next_probs, next_entropy = self.actor.get_action(next_obs_tensor)
            next_q1 = self.q1_target(next_obs_tensor)
            next_q2 = self.q2_target(next_obs_tensor)
            next_q = torch.min(next_q1, next_q2)

            # V(s') = E[Q(s', a') - α log π(a'|s')] for discrete actions
            next_v = (next_probs * (next_q - alpha * F.log_softmax(next_q, dim=-1))).sum(dim=-1)
            target_q = rewards + self.gamma * next_v * (1 - dones)

        current_q1 = self.q1(obs_tensor).gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        current_q2 = self.q2(obs_tensor).gather(1, actions_idx.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ===== Update Actor =====
        _, probs, entropy = self.actor.get_action(obs_tensor)
        q1_val = self.q1(obs_tensor)
        q2_val = self.q2(obs_tensor)
        q_val = torch.min(q1_val, q2_val)

        # Policy loss: maximize E[Q(s,a) + α H(π(·|s))]
        # For discrete: sum over actions of π(a|s) * [Q(s,a) - α log π(a|s)]
        actor_loss = (probs * (alpha * F.log_softmax(q_val, dim=-1) - q_val)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Update Alpha =====
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy).mean())

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ===== Soft update target networks =====
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self.train_step_counter += 1

        return {}, state, {
            'loss': (q1_loss + q2_loss + actor_loss).item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item(),
            'entropy': entropy.mean().item(),
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
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'log_alpha': self.log_alpha.item(),
        }

    def load(self, data):
        self.actor.load_state_dict(data['actor'])
        self.q1.load_state_dict(data['q1'])
        self.q2.load_state_dict(data['q2'])
        self.q1_target.load_state_dict(data['q1_target'])
        self.q2_target.load_state_dict(data['q2_target'])
        self.log_alpha.data.fill_(data.get('log_alpha', np.log(0.2)))

    def sync(self):
        pass
