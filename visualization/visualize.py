import pygame
import torch
import time
import imageio
import numpy as np
from environments.football_ENV import FootballEnv
from agents.ppo_agent import PPOAgent

# Config
CELL_SIZE = 40
MARGIN = 2
GRID_WIDTH = 20
GRID_HEIGHT = 10
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = ((GRID_HEIGHT * CELL_SIZE + 60 + 15) // 16) * 16
FPS = 30
RECORD_PATH = "football_sim.mp4"

# Colors
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
TEAM0_COLOR = (0, 102, 255)
TEAM1_COLOR = (255, 51, 51)
BALL_COLOR = (255, 165, 0)
BLACK = (0, 0, 0)

def load_trained_agent(agent_path, obs_dim, act_dim, device):
    agent = PPOAgent(obs_dim, act_dim).to(device)
    agent.policy_net.load_state_dict(torch.load(agent_path, map_location=device))
    agent.policy_net.eval()
    return agent

def draw_game(screen, font, env, step_count, score):
    screen.fill(GREEN)

    # Grid
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            pygame.draw.rect(screen, WHITE, 
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN), 1)

    # Goals
    pygame.draw.rect(screen, WHITE, (0, 3 * CELL_SIZE, MARGIN, 4 * CELL_SIZE))
    pygame.draw.rect(screen, WHITE, (WINDOW_WIDTH - MARGIN, 3 * CELL_SIZE, MARGIN, 4 * CELL_SIZE))

    # Agents
    for agent_id, pos in env.positions.items():
        x, y = pos
        screen_x = x * CELL_SIZE + CELL_SIZE // 2
        screen_y = y * CELL_SIZE + CELL_SIZE // 2
        color = TEAM0_COLOR if "team_0" in agent_id else TEAM1_COLOR
        pygame.draw.circle(screen, color, (screen_x, screen_y), CELL_SIZE // 3)
        text = font.render(agent_id[-1], True, WHITE)
        screen.blit(text, (screen_x - 6, screen_y - 10))

    # Ball
    bx, by = env.ball_pos
    pygame.draw.circle(screen, BALL_COLOR, 
                       (bx * CELL_SIZE + CELL_SIZE // 2, by * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 6)

    # Scoreboard
    pygame.draw.rect(screen, BLACK, (0, WINDOW_HEIGHT - 60, WINDOW_WIDTH, 60))
    score_text = font.render(f"Score: Team 0 = {score[0]} | Team 1 = {score[1]}", True, WHITE)
    time_text = font.render(f"Steps: {step_count}", True, WHITE)
    screen.blit(score_text, (10, WINDOW_HEIGHT - 50))
    screen.blit(time_text, (10, WINDOW_HEIGHT - 25))

    pygame.display.flip()

def capture_frame(screen):
    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))

def visualize(agent_path="trained_agent.pth", record=True):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Football Simulation")
    font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()

    writer = imageio.get_writer(RECORD_PATH, fps=FPS) if record else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    def reset_game():
        env = FootballEnv()
        obs = env.reset()
        score = [0, 0]
        done = {agent: False for agent in env.agents}

        agents = {
            agent_id: load_trained_agent(agent_path, obs[first_agent].shape[0], env.action_space[first_agent].n, device)
            for agent_id in env.agents
        }
        return env, obs, score, done, agents

    first_agent = "team_0_player_0"
    env, obs, score, done, agents = reset_game()
    step_count = 0
    running = True

    while running:
        step_count += 1
        actions = {}

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env, obs, score, done, agents = reset_game()
                    step_count = 0
                elif event.key == pygame.K_s:
                    if writer:
                        writer.close()
                        writer = None
                        print("[INFO] Stopped recording.")
                    else:
                        writer = imageio.get_writer(RECORD_PATH, fps=FPS)
                        print("[INFO] Started recording...")

        if all(done.values()):
            break

        for agent_id in env.agents:
            if not done[agent_id]:
                action, _, _ = agents[agent_id].act(obs[agent_id])
                actions[agent_id] = action

        obs, rewards, done, trunc, _ = env.step(actions)

        for agent_id, r in rewards.items():
            if r >= 1.0:
                team = 0 if "team_0" in agent_id else 1
                score[team] += 1

        draw_game(screen, font, env, step_count, score)

        if writer:
            frame = capture_frame(screen)
            writer.append_data(frame)

        clock.tick(FPS)

    if writer:
        writer.close()
    pygame.quit()

if __name__ == "__main__":
    visualize()
