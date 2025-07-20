# main.py
# MIT License
# Copyright (c) 2025 Soham
# This file is part of the football simulation project and is licensed under the MIT License.
# See the LICENSE file in the root directory for full license text.


import argparse
import os
from utils.rollout import evaluate
from visualization.visualize import visualize
from training import train  # assuming train.py has a train() method optionally
from selfplay.selfplay_trainer import SelfPlayTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "visualize", "selfplay"], required=True)
    parser.add_argument("--model_path", type=str, default="trained_agent.pth")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--record", action="store_true", help="Record gameplay when visulaising")
    args = parser.parse_args()

    if args.mode == "train":
        print("[MAIN] Starting training...")
        train.train(num_episodes=args.episodes, save_path=args.model_path)

    elif args.mode == "evaluate":
        print("[MAIN] Evaluating agent...")
        evaluate(agent_path=args.model_path, num_episodes=args.episodes)

    elif args.mode == "visualize":
        print("[MAIN] Visualizing gameplay...")
        visualize(agent_path=args.model_path)
        
    elif args.mode == "selfplay":
        print("[MAIN] Starting self-play training...")
        trainer = SelfPlayTrainer(episodes=args.episodes)
        trainer.train()

if __name__ == "__main__":
    main()
