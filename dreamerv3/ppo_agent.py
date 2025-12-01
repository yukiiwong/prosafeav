import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.logits = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.net(obs)
        return self.logits(x), self.value_head(x)

class PPOAgent:
    def __init__(self, obs_space, act_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用 birdeye_wpt 作为输入键（非图像）
        obs_key = 'birdeye_wpt'
        obs_shape = obs_space[obs_key].shape
        self.obs_key = obs_key
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = act_space['action'].shape[0]  # one-hot 动作维度
        self.num_actions = self.act_dim

        self.model = MLPPolicy(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _preprocess_obs(self, obs):
        x = obs[self.obs_key]  # e.g., shape (batch, D)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return x.reshape(x.size(0), -1)  # flatten to (batch, obs_dim)

    def __call__(self, obs, state=None, mode='train'):
        return self.policy(obs, state, mode)

    def policy(self, obs, state=None, mode='train'):
        obs_tensor = self._preprocess_obs(obs)
        logits, _ = self.model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        one_hot = torch.nn.functional.one_hot(action, num_classes=self.num_actions).float()
        return {'action': one_hot.cpu().numpy(), 'reset': np.array(False)}, state

    def train(self, batch, state=None, step=None):
        obs_tensor = self._preprocess_obs(batch)
        actions = torch.tensor(batch['action'], dtype=torch.float32).to(self.device)
        actions_idx = torch.argmax(actions, dim=1)

        rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)
        logits, values = self.model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_idx)
        advantages = rewards - values.squeeze()

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}, state, {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def dataset(self, make_replay_dataset):
        return make_replay_dataset()

    def report(self, batch):
        return {}

    def save(self):
        return {'model': self.model.state_dict()}

    def load(self, data):
        self.model.load_state_dict(data['model'])


    def sync(self):
        pass  # for multi-GPU setting, can be implemented later
