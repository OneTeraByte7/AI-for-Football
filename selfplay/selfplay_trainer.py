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
        
    def train(self):
        for ep in range(1, self.episodes + 1):
            obs = self.env.reset()
            done = {a: False for a in self.env.agents}
            step_count = 0

            # --- Optionally load older opponent snapshot ---
            if ep > self.snapshot_interval and ep % self.snapshot_interval == 0:
                version = np.random.randint(1, ep // self.snapshot_interval + 1)
                opponent_agents = self.checkpoints.load_version(version, self.team_0_ids, self.obs_dim, self.act_dim, self.device, PPOAgent)
                self.opponents = {
                    f"team_1_player_{i}": opponent_agents[f"team_0_player_{i}"]
                    for i in range(len(self.team_1_ids))
                }

            # Initialize training data
            episode_data = {aid: {"obs": [], "actions": [], "rewards": [], "log_probs": [], "values": []} for aid in self.team_0_ids}

            while not all(done.values()):
                actions = {}
                for aid in self.env.agents:
                    if done[aid]:
                        continue
                    ob = torch.tensor(obs[aid], dtype=torch.float32).unsqueeze(0).to(self.device)
                    agent = self.agents[aid] if "team_0" in aid else self.opponents[aid]
                    with torch.no_grad():
                        action, log_prob, _ = agent.act(obs[aid])
                        value = agent.value_net(ob).item()
                    actions[aid] = action
                    if "team_0" in aid:
                        episode_data[aid]["obs"].append(obs[aid])
                        episode_data[aid]["actions"].append(action)
                        episode_data[aid]["log_probs"].append(log_prob)
                        episode_data[aid]["values"].append(value)

                obs, rewards, done, trunc, _ = self.env.step(actions)

                for aid in self.team_0_ids:
                    episode_data[aid]["rewards"].append(rewards[aid])

            self._ppo_update(episode_data)

            if ep % self.snapshot_interval == 0:
                self.checkpoints.save(self.agents)

            if ep % 10 == 0:
                avg_reward = np.mean([sum(data["rewards"]) for data in episode_data.values()])
                print(f"[EP {ep}] Avg Team 0 Reward: {avg_reward:.2f}")

    def _ppo_update(self, episode_data):
        gamma = 0.99
        clip_eps = 0.2
        update_epochs = 4
        entropy_coef = 0.01

        for aid, agent in self.agents.items():
            data = episode_data[aid]
            obs = torch.tensor(np.array(data["obs"]), dtype=torch.float32).to(self.device)
            actions = torch.tensor(data["actions"]).to(self.device)
            old_log_probs = torch.stack(data["log_probs"]).to(self.device)
            values = torch.tensor(data["values"]).to(self.device)

            # GAE or return computation
            returns = []
            R = 0
            for r in reversed(data["rewards"]):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(update_epochs):
                new_log_probs, entropy, new_values = agent.evaluate(obs, actions)
                ratio = (new_log_probs - old_log_probs.detach()).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns - new_values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()