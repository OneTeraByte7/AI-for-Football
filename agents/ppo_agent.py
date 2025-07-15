import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, device='cpu'):
        self.device = device
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr
        )

    def to(self, device):
        self.policy_net = self.policy_net.to(device)
        self.value_net = self.value_net.to(device)
        return self

    
    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, obs_dim)
        # DEBUG print can be toggled on/off outside if needed
        # print(f"[DEBUG] act input tensor shape: {obs.shape}")
        probs = self.policy_net(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate(self, obs, actions):
        probs = self.policy_net(obs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(obs).squeeze(-1)
        return log_probs, entropy, values
