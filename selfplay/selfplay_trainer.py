import torch
import numpy as np
from environments.football_ENV import FootballEnv
from agents.ppo_agent import PPOAgent
from selfplay.checkpoint_manager import CheckpointManager

class SelfPlayTrainer:
    def __init__(self, episodes=1000, snapshot_interval=50, device=None):
        self.episodes = episodes
        self.snapshot_interval = snapshot_interval
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoints = CheckpointManager()
        self.env = FootballEnv()
        obs = self.env.reset()
        first_agent = list(obs.keys())[0]
        self.obs_dim = obs[first_agent].shape[0]
        self.act_dim = self.env.action_space[first_agent].n
        self.team_0_ids = [a for a in self.env.agents if "team_0" in a]
        self.team_1_ids = [a for a in self.env.agents if "team_1" in a]
        self.agents = {aid: PPOAgent(self.obs_dim, self.act_dim).to(self.device) for aid in self.team_0_ids}
        self.opponents = {aid: PPOAgent(self.obs_dim, self.act_dim).to(self.device) for aid in self.team_1_ids}