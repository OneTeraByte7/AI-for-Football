import pygame
import torch
import time
from environments.football_ENV import FootballEnv
from agents.ppo_agent import PPOAgent

# Pygame setup
CELL_SIZE = 40
MARGIN = 2
GRID_WIDTH = 20
GRID_HEIGHT = 10
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE + 60  # Extra for scoreboard

# Colors
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
TEAM0_COLOR = (0, 102, 255)
TEAM1_COLOR = (255, 51, 51)
BALL_COLOR = (255, 165, 0)
BLACK = (0, 0, 0)

FPS = 60

def load_trained_agent(agent_path, obs_dim, act_dim, device):
    agent = PPOAgent(obs_dim, act_dim).to(device)
    agent.policy_net.load_state_dict(torch.load(agent_path, map_location=device))
    agent.policy_net.eval()
    return agent

def draw_game(screen, font, env, step_count, score):
    screen.fill(GREEN)

    # Grid lines
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            pygame.draw.rect(screen, WHITE, 
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN), 1)

    # Goals
    pygame.draw.rect(screen, WHITE, (0, 3 * CELL_SIZE, MARGIN, 4 * CELL_SIZE))
    pygame.draw.rect(screen, WHITE, (WINDOW_WIDTH - MARGIN, 3 * CELL_SIZE, MARGIN, 4 * CELL_SIZE))

    # Draw agents
    for agent_id, pos in env.positions.items():
        x, y = pos
        screen_x = x * CELL_SIZE + CELL_SIZE // 2
        screen_y = y * CELL_SIZE + CELL_SIZE // 2
        team_color = TEAM0_COLOR if "team_0" in agent_id else TEAM1_COLOR
        pygame.draw.circle(screen, team_color, (screen_x, screen_y), CELL_SIZE // 3)
        player_num = agent_id.split("_")[-1]
        text = font.render(player_num, True, WHITE)
        screen.blit(text, (screen_x - 6, screen_y - 10))

    # Ball
    bx, by = env.ball_pos
    pygame.draw.circle(screen, BALL_COLOR, 
                       (bx * CELL_SIZE + CELL_SIZE // 2, by * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 6)

    # Score & time
    pygame.draw.rect(screen, BLACK, (0, WINDOW_HEIGHT - 60, WINDOW_WIDTH, 60))
    score_text = font.render(f"Score: Team 0 = {score[0]} | Team 1 = {score[1]}", True, WHITE)
    time_text = font.render(f"Steps: {step_count}", True, WHITE)
    screen.blit(score_text, (10, WINDOW_HEIGHT - 50))
    screen.blit(time_text, (10, WINDOW_HEIGHT - 25))

    pygame.display.flip()

def visualize(agent_path="trained_agent.pth"):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Football Simulation")
    font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()

    env = FootballEnv()
    obs = env.reset()
    step_count = 0
    score = [0, 0]

    first_agent = list(obs.keys())[0]
    obs_dim = obs[first_agent].shape[0]
    act_dim = env.action_space[first_agent].n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents = {
        agent_id: load_trained_agent(agent_path, obs_dim, act_dim, device)
        for agent_id in env.agents
    }

    done = {agent: False for agent in env.agents}
    running = True

    while not all(done.values()) and running:
        step_count += 1
        actions = {}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        for agent_id in env.agents:
            if done[agent_id]:
                continue
            ob = obs[agent_id]
            action, _, _ = agents[agent_id].act(ob)
            actions[agent_id] = action

        obs, rewards, done, trunc, _ = env.step(actions)

        # Live scoring logic (basic goal detection by reward spike)
        for aid, r in rewards.items():
            team = 0 if "team_0" in aid else 1
            if r >= 1.0:
                score[team] += 1

        draw_game(screen, font, env, step_count, score)
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    visualize()
