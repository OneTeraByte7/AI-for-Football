# utils/rollout.py
# MIT License
# Copyright (c) 2025 Soham
# This file is part of the football simulation project and is licensed under the MIT License.
# See the LICENSE file in the root directory for full license text.


import torch
import numpy as np
from environments.football_ENV import FootballEnv
from agents.ppo_agent import PPOAgent
from tqdm import trange


def evaluate(agent_path="trained_agent.pth", num_episodes=5, render=False):
    env = FootballEnv()
    obs = env.reset()
    first_agent = list(obs.keys())[0]
    obs_dim = obs[first_agent].shape[0]
    act_dim = env.action_space[first_agent].n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")

    agents = {
        agent_id: PPOAgent(obs_dim, act_dim).to(device)
        for agent_id in env.agents
    }
    for agent_id, agent in agents.items():
        agent.policy_net.load_state_dict(torch.load(agent_path, map_location=device))
        agent.policy_net.eval()

    episode_scores = []

    
    for ep in trange(num_episodes, desc="Evaluating"):
        obs = env.reset()
        done = {agent: False for agent in env.agents}
        rewards = {agent: 0.0 for agent in env.agents}

        while not all(done.values()):
            actions = {}
            for agent_id in env.agents:
                if not done[agent_id]:
                    action, _, _ = agents[agent_id].act(obs[agent_id])
                    actions[agent_id] = action

            obs, reward, done, _, _ = env.step(actions)

            for k in reward:
                rewards[k] += reward[k]

            if render:
                env.render()

        avg_ep_reward = sum(rewards.values()) / len(rewards)
        episode_scores.append(avg_ep_reward)
        print(f"[EVAL] Episode {ep}: Avg Reward = {avg_ep_reward:.3f}")

    final_score = np.mean(episode_scores)
    print(f"\n[EVAL] Final Avg Reward over {num_episodes} episodes: {final_score:.3f}")
    return final_score

if __name__ == "__main__":
    evaluate(num_episodes=5, render=True)
