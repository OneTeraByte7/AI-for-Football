from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from gymnasium import spaces
import numpy as np

GRID_WIDTH = 20
GRID_HEIGHT = 10
NUM_PLAYERS_PER_TEAM = 3
MAX_EXPISODE_STEPS = 300
class FootballEnv(ParallelEnv):
    def __inti__(self):
        super().__init__()
        self.agents = [f"team_0_player_{i}" for i in range(NUM_PLAYERS_PER_TEAM)] + \
                      [f"team_1_player_{i}" for i in range(NUM_PLAYERS_PER_TEAM)]
                      
        self.possible_agents = self.agents[:]
        self.agent_order = self.agents[:]
        self.step_count = 0