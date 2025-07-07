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