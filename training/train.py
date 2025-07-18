# MIT License
# Copyright (c) 2025 Soham
# This file is part of the football simulation project and is licensed under the MIT License.
# See the LICENSE file in the root directory for full license text.


import logging
import time
import numpy as np
import torch
from environments.football_ENV import FootballEnv
from agents.ppo_agent import PPOAgent

def train(num_episodes=100, save_path=None):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    GAMMA = 0.99
    CLIP_EPS = 0.2
    UPDATE_EPOCHS = 4
    ENTROPY_COEF = 0.01
    DEBUG_PRINT_FREQ = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    env = FootballEnv()
    obs = env.reset()
    first_agent = list(obs.keys())[0]
    logger.info(f"Observation shape on reset for {first_agent}: {obs[first_agent].shape}")

    obs_space = obs[first_agent].shape[0]
    act_space = env.action_space[first_agent].n

    agents = {
        agent_id: PPOAgent(obs_space, act_space).to(device)
        for agent_id in env.agents
    }

    def compute_returns(rewards, dones, gamma=GAMMA):
        returns = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    try:
        from tqdm import trange
        for episode in trange(1, num_episodes + 1, desc="Training Episodes"):
            obs = env.reset()
            done = {agent: False for agent in env.agents}
            step_count = 0

            episode_data = {
                agent: {
                    "obs": [], "actions": [], "log_probs": [],
                    "rewards": [], "values": [], "dones": []
                } for agent in env.agents
            }

            start_time = time.time()

            while not all(done.values()):
                step_count += 1
                actions, log_probs, entropies, values = {}, {}, {}, {}

                for agent_id, agent in agents.items():
                    if done[agent_id]:
                        continue

                    ob = obs[agent_id]
                    if step_count % DEBUG_PRINT_FREQ == 0:
                        logger.debug(f"Agent {agent_id} obs shape before act: {np.array(ob).shape}")

                    action, log_prob, entropy = agent.act(ob)
                    value = agent.value_net(torch.tensor(ob, dtype=torch.float32).unsqueeze(0).to(device)).item()

                    actions[agent_id] = action
                    log_probs[agent_id] = log_prob
                    entropies[agent_id] = entropy
                    values[agent_id] = value

                next_obs, rewards, done, _, _ = env.step(actions)

                for agent_id in env.agents:
                    episode_data[agent_id]["obs"].append(obs[agent_id])
                    episode_data[agent_id]["actions"].append(actions.get(agent_id, 0))
                    episode_data[agent_id]["log_probs"].append(
                        log_probs.get(agent_id, torch.tensor(0.0, dtype=torch.float32))
                    )
                    episode_data[agent_id]["rewards"].append(rewards[agent_id])
                    episode_data[agent_id]["values"].append(values.get(agent_id, 0.0))
                    episode_data[agent_id]["dones"].append(done[agent_id])

                obs = next_obs

            # PPO update per agent
            for agent_id, agent in agents.items():
                data = episode_data[agent_id]
                obs_tensor = torch.tensor(np.stack(data["obs"]), dtype=torch.float32).to(device)
                actions_tensor = torch.tensor(data["actions"], dtype=torch.long).to(device)
                old_log_probs = torch.stack(data["log_probs"]).to(device)
                values_tensor = torch.tensor(data["values"], dtype=torch.float32).to(device)
                returns = torch.tensor(compute_returns(data["rewards"], data["dones"]), dtype=torch.float32).to(device)

                advantages = returns - values_tensor.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                for _ in range(UPDATE_EPOCHS):
                    new_log_probs, entropy, new_values = agent.evaluate(obs_tensor, actions_tensor)
                    ratio = (new_log_probs - old_log_probs.detach()).exp()

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = ((returns - new_values) ** 2).mean()
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + 0.5 * critic_loss + ENTROPY_COEF * entropy_loss
                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()

            if episode % 10 == 0:
                avg_reward = np.mean([sum(d["rewards"]) for d in episode_data.values()])
                elapsed = time.time() - start_time
                logger.info(f"[Episode {episode}] Avg Reward: {avg_reward:.3f}, Steps: {step_count}, Time: {elapsed:.2f}s")

        if save_path:
            torch.save(agents[first_agent].policy_net.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
