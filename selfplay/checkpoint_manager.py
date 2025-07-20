import os
import torch

class CheckpointManager:
    def __init__(self, base_dir="snapshots"):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.version = 0
        
    def save(self, agents):
        self.version += 1
        version_dir = os.path.join(self.base_dir, f"v{self.version}")
        os.makedirs(version_dir, exist_ok=True)
        for agent_id, agent in agents.items():
            path = os.path.join(version_dir, f"{agent_id}.pth")
            torch.save(agent.policy_net.state_dict(), path)
            
    def load_version(self, version, agent_ids, obs_dim, act_dim, device, agent_class):
        version_dir = os.path.join(self.base_dir, f"{version}")
        agents = {}
        for agent_id in agent_ids:
            agent = agent_class(obs_dim, act_dim).to(device)
            path = os.path.join(version_dir, f"{agent_id}.pth")
            agent.policy_net.load_state_dict(torch.load(path, map_location=device))
            agent.policy_net.eval()
            agents[agent_id] = agent
        return agents