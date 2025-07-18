import pygame
import torch
import time
import imageio
import numpy as np
from collections import deque
from environments.football_ENV import FootballEnv
from agents.ppo_agent import PPOAgent

# Config
CELL_SIZE = 40
MARGIN = 2
GRID_WIDTH = 20
GRID_HEIGHT = 10
WINDOW_HEIGHT = ((GRID_HEIGHT * CELL_SIZE + 150 + 15) // 16) * 16  # Increased bottom panel space
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
FPS = 30
RECORD_PATH = "football_sim.mp4"
TRAIL_LENGTH = 10

# Colors
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
TEAM0_COLOR = (0, 102, 255)
TEAM1_COLOR = (255, 51, 51)
BALL_COLOR = (255, 165, 0)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
YELLOW = (255, 255, 0)

def load_trained_agent(agent_path, obs_dim, act_dim, device):
    agent = PPOAgent(obs_dim, act_dim).to(device)
    agent.policy_net.load_state_dict(torch.load(agent_path, map_location=device))
    agent.policy_net.eval()
    return agent

def capture_frame(screen):
    return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))

def render_commentary(screen, font, lines):
    y_offset = WINDOW_HEIGHT - 60  # Commentary starts here
    lines_list = list(lines)
    for i, line in enumerate(lines_list[-3:]):
        text = font.render(line, True, YELLOW)
        screen.blit(text, (10, y_offset + i * 20))

def draw_game(screen, font, env, step_count, score, player_goals, possession, trails, commentary):
    screen.fill(GREEN)

    # Draw grid
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            pygame.draw.rect(screen, WHITE, 
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN), 1)

    # Goals
    pygame.draw.rect(screen, WHITE, (0, 3 * CELL_SIZE, MARGIN, 4 * CELL_SIZE))
    pygame.draw.rect(screen, WHITE, (WINDOW_WIDTH - MARGIN, 3 * CELL_SIZE, MARGIN, 4 * CELL_SIZE))

    # Draw agent trails
    for agent_id, trail in trails.items():
        for i, (x, y) in enumerate(trail):
            alpha = int(255 * (i + 1) / TRAIL_LENGTH)
            color = TEAM0_COLOR if "team_0" in agent_id else TEAM1_COLOR
            faded_color = tuple(int(c * (i + 1) / TRAIL_LENGTH) for c in color)
            pygame.draw.circle(screen, faded_color, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 4)

    # Draw agents
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
    pygame.draw.circle(screen, BALL_COLOR, (bx * CELL_SIZE + CELL_SIZE // 2, by * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 6)

    # Scoreboard background (increased height)
    pygame.draw.rect(screen, BLACK, (0, WINDOW_HEIGHT - 150, WINDOW_WIDTH, 150))

    # Score and steps text
    score_text = font.render(f"Score: Team 0 = {score[0]} | Team 1 = {score[1]}", True, WHITE)
    steps_text = font.render(f"Steps: {step_count}", True, WHITE)
    screen.blit(score_text, (10, WINDOW_HEIGHT - 140))
    screen.blit(steps_text, (10, WINDOW_HEIGHT - 115))

    # Player goals with spacing (max 4 per row)
    x_offset = 300
    y_base = WINDOW_HEIGHT - 140
    for i, (agent_id, g) in enumerate(player_goals.items()):
        row = i // 4
        col = i % 4
        txt = font.render(f"{agent_id[-3:]}: {g}", True, WHITE)
        screen.blit(txt, (x_offset + col * 120, y_base + row * 20))

    # Commentary text (last 3 lines)
    render_commentary(screen, font, commentary)

    pygame.display.flip()

def visualize(agent_path="trained_agent.pth", record=True):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Football Simulation")
    font = pygame.font.SysFont("Arial", 18)
    clock = pygame.time.Clock()
    writer = imageio.get_writer(RECORD_PATH, fps=FPS) if record else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    def reset_game():
        env = FootballEnv()
        obs = env.reset()
        score = [0, 0]
        player_goals = {agent: 0 for agent in env.agents}
        possession = {agent: 0 for agent in env.agents}
        trails = {agent: deque(maxlen=TRAIL_LENGTH) for agent in env.agents}
        commentary = deque(maxlen=5)
        done = {agent: False for agent in env.agents}
        agents = {
            agent_id: load_trained_agent(agent_path, obs[agent_id].shape[0], env.action_space[agent_id].n, device)
            for agent_id in env.agents
        }
        return env, obs, score, done, agents, player_goals, possession, trails, commentary

    env, obs, score, done, agents, player_goals, possession, trails, commentary = reset_game()
    step_count = 0
    running = True

    while running and not all(done.values()):
        step_count += 1
        actions = {}

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env, obs, score, done, agents, player_goals, possession, trails, commentary = reset_game()
                    step_count = 0
                elif event.key == pygame.K_s:
                    if writer:
                        writer.close()
                        writer = None
                        print("[INFO] Stopped recording.")
                    else:
                        writer = imageio.get_writer(RECORD_PATH, fps=FPS)
                        print("[INFO] Started recording...")

        for agent_id in env.agents:
            if not done[agent_id]:
                action, _, _ = agents[agent_id].act(obs[agent_id])
                actions[agent_id] = action

        obs, rewards, done, trunc, _ = env.step(actions)

        for agent_id, r in rewards.items():
            if r >= 1.0:
                team = 0 if "team_0" in agent_id else 1
                score[team] += 1
                player_goals[agent_id] += 1
                commentary.append(f"Goal by {agent_id}! Team {team} scores!")

            elif r > 0.05:
                commentary.append(f"{agent_id} makes a great pass!")

            if env.ball_owner == agent_id:
                possession[agent_id] += 1

        for agent_id in env.agents:
            trails[agent_id].append(tuple(env.positions[agent_id]))

        draw_game(screen, font, env, step_count, score, player_goals, possession, trails, commentary)

        if writer:
            frame = capture_frame(screen)
            writer.append_data(frame)

        clock.tick(FPS)

    if writer:
        writer.close()

    # End of match summary
    pygame.time.wait(1000)
    screen.fill(BLACK)
    summary_font = pygame.font.SysFont("Arial", 24)
    winner = "Team 0" if score[0] > score[1] else "Team 1" if score[1] > score[0] else "Draw"
    summary = [
        f"Match Over!",
        f"Final Score: Team 0 = {score[0]} | Team 1 = {score[1]}",
        f"Winner: {winner}",
        f"Steps: {step_count}",
        f"Possession: Team 0 = {sum(possession[k] for k in possession if 'team_0' in k)} | Team 1 = {sum(possession[k] for k in possession if 'team_1' in k)}"
    ]
    for i, line in enumerate(summary):
        text = summary_font.render(line, True, WHITE)
        screen.blit(text, (WINDOW_WIDTH // 6, 80 + i * 40))
    pygame.display.flip()
    pygame.time.wait(5000)
    pygame.quit()

if __name__ == "__main__":
    visualize()
