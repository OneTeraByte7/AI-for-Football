import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent(nn.Module):
    def __init__(self,obs_dim, action_dim, lr = 2.5e-4):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def act(self,obs):
        obs = torch.tensor(obs, dtype = torch.float32)
        probs = self.policy_net(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    
    def evaluate(self,obs,action):
        probs = self.policy_net(obs)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_net(obs).squeeze(-1)
        return log_prob, entropy, value