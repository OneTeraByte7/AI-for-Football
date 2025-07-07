# test_agent.py

from agents.ppo_agent import PPOAgent
import numpy as np
import torch

obs_dim = 28  # Based on football_env (4 + 4 * (2 * 3 - 1)) = 28
action_dim = 7  # 7 discrete actions

agent = PPOAgent(obs_dim, action_dim)

# Create a dummy observation (normalized like in the env)
obs = np.random.rand(obs_dim).astype(np.float32)

# Test action selection
action, log_prob, entropy = agent.act(obs)
print(f"Sampled Action: {action}")
print(f"Log Prob: {log_prob.item():.4f}")
print(f"Entropy: {entropy.item():.4f}")

# Test evaluate (batch of obs)
obs_batch = torch.tensor(np.stack([obs]*4), dtype=torch.float32)  # batch of 4
action_batch = torch.tensor([action]*4)

log_probs, entropies, values = agent.evaluate(obs_batch, action_batch)
print(f"Evaluate -> LogProbs: {log_probs.shape}, Values: {values.shape}")
