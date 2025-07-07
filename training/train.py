from environments.football_ENV import FootballEnv
from agents import PPOAgent
import numpy as np
import torch
from collections import deque


NUM_EPISODES = 1000
GAMMA = 0.99
CLIP_EPS = 0.2
UPDATES_EPOCHS = 4
ENTROPY_COEF = 0.01
BATCH_SIZE = 32

env = FootballEnv()
obs_space = env.observations_space[env.agents[0]].shape[0]
act_space = env.action_space[env.agents[0]].n

def compute_returns(rewards, values, dones, gamma=GAMMA):
    returns = []
    R = 0;
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        R = r + gamma *R *(1-d)
        returns.insert(0,R)
    return returns

for episodes in range(NUM_EPISODES):
    obs = env.reset()
    done = {agent: False for agent in env.agents}
    
    episodes_data = {agent: {
        "obs":[], "actions":[], "log_probs":[],
        "rewards":[], "values":[], "dones":[]
    } for agent in env.agents}
    
    while not all(done.values()):
        actions = {}
        log_probs = {}
        entropies = {}
        values = {}
        
        
        for agent_id, agent in agents.items():
            if done[agent_id]:
                continue
            ob = obs[agent_id]
            action, log_prob, entropy = agent.act(ob)
            value = agent.value_net(torch.tensor(ob,dtype=torch.float32)).item()