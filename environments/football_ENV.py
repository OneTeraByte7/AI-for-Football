from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from gymnasium import spaces
import numpy as np

GRID_WIDTH = 20
GRID_HEIGHT = 10
NUM_PLAYERS_PER_TEAM = 3
MAX_EPISODE_STEPS = 300
class FootballEnv(ParallelEnv):
    def __init__(self):
        super().__init__()

        # Initialize agents BEFORE reset()
        self.agents = [f"team_0_player_{i}" for i in range(NUM_PLAYERS_PER_TEAM)] + \
                      [f"team_1_player_{i}" for i in range(NUM_PLAYERS_PER_TEAM)]
        self.possible_agents = self.agents[:]
        self.agent_order = self.agents[:]

        self.action_space = {agent: spaces.Discrete(7) for agent in self.agents}
        obs_dim = 4 + 4 * (2 * NUM_PLAYERS_PER_TEAM - 1)
        self.observation_space = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }

        # Just init counters
        self.step_count = 0
        self.dones = {}
        self.infos = {}
        self.truncations = {}
        self.positions = {}
        self.ball_pos = None
        self.ball_owner = None

        
    def reset(self, seed=None, options=None):
        self.positions = {agent: np.array([
            np.random.randint(0, GRID_WIDTH),
            np.random.randint(0, GRID_HEIGHT)
        ]) for agent in self.agents}

        self.ball_pos = np.array([GRID_WIDTH // 2, GRID_HEIGHT // 2])
        self.ball_owner = np.random.choice(self.agents)  # Random kickoff
        self.step_count = 0
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        return self._get_obs()

    def step(self, actions):
        rewards = {agent: 0.0 for agent in self.agents}
        self.step_count += 1

        # Move agents
        for agent, action in actions.items():
            if action <= 3:  # Movement
                dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
                self.positions[agent] += [dx, dy]
                self.positions[agent] = np.clip(self.positions[agent], [0, 0], [GRID_WIDTH - 1, GRID_HEIGHT - 1])
            elif action == 4 and self.ball_owner == agent:  # Pass
                teammates = [a for a in self.agents if a != agent and a.startswith(agent[:6])]
                target = np.random.choice(teammates)
                self.ball_owner = target
                rewards[agent] += 0.1
            elif action == 5 and self.ball_owner == agent:  # Shoot
                team = int(agent[5])
                if self._is_goal(team):
                    rewards[agent] += 1.0
                    opp_team = 1 - team
                    for a in self.agents:
                        if a.startswith(f"team_{opp_team}"):
                            rewards[a] -= 1.0
                    self._reset_positions()
         
        self.ball_pos = self.positions[self.ball_owner].copy()

        terminated = self.step_count >= MAX_EPISODE_STEPS
        self.dones = {agent: terminated for agent in self.agents}
        self.truncations = {agent: terminated for agent in self.agents}

        obs = self._get_obs()
        return obs, rewards, self.dones, self.truncations, self.infos

    def _is_goal(self, team):
        # Define goal lines (left and right edges)
        if team == 0 and self.ball_pos[0] >= GRID_WIDTH - 1:
            return True
        if team == 1 and self.ball_pos[0] <= 0:
            return True
        return False

    def _reset_positions(self):
        for agent in self.agents:
            self.positions[agent] = np.array([
                np.random.randint(0, GRID_WIDTH),
                np.random.randint(0, GRID_HEIGHT)
            ])
        self.ball_owner = np.random.choice(self.agents)

    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            agent_pos = self.positions[agent] / [GRID_WIDTH, GRID_HEIGHT]
            ball_pos = self.ball_pos / [GRID_WIDTH, GRID_HEIGHT]
            others = [a for a in self.agents if a != agent]
            others_pos = np.concatenate([
                self.positions[a] / [GRID_WIDTH, GRID_HEIGHT] for a in others
            ])
            obs[agent] = np.concatenate([agent_pos, ball_pos, others_pos]).astype(np.float32)
        return obs

    def render(self):
        grid = np.full((GRID_HEIGHT, GRID_WIDTH), ".", dtype=str)
        for agent, pos in self.positions.items():
            symbol = "A" if agent.startswith("team_0") else "B"
            x, y = pos
            grid[y, x] = symbol
        bx, by = self.ball_pos
        grid[by, bx] = "O"
        print("\n".join("".join(row) for row in grid[::-1]))
        print()