import pygame
import random
import numpy as np
import sys
import pickle
import os

# ---------- Config ----------
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
BLOCK_SIZE = WIDTH // GRID_SIZE
FPS = 30

# Colors
BLACK, WHITE, RED, GREEN = (0,0,0), (255,255,255), (255,0,0), (0,255,0)

# Q-learning hyper-parameters
EPSILON = 1.0
MIN_EPSILON = 0.05
GAMMA = 0.9
LEARNING_RATE = 0.1
EPISODES = 10000
MAX_STEPS = 5000
EPSILON_DECAY = 0.995
SAVE_INTERVAL = 500  # Save every 500 episodes
SAVE_FILE = "q_table.pkl"

# ---------- Pygame ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Snake Game (Training)")
clock = pygame.time.Clock()

# ---------- Q-table save/load ----------
def save_q_table():
    """Save Q-table to file."""
    with open(SAVE_FILE, "wb") as f:
        pickle.dump((q_table, EPSILON), f)
    print(f"[💾] Q-table saved to {SAVE_FILE}")

def load_q_table():
    """Load Q-table from file if it exists."""
    global q_table, EPSILON
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple):
                q_table, EPS = data
                EPSILON = EPS
            else:
                q_table = data
        print(f"[✅] Loaded Q-table from {SAVE_FILE} (entries: {len(q_table)})")
    else:
        q_table = {}
        print("[ℹ️] No previous Q-table found — starting fresh.")

# ---------- Snake ----------
class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.body = [(GRID_SIZE//2, GRID_SIZE//2)]
        self.length = 1

    def move(self, action):
        row, col = self.body[0]

        # 0=up, 1=down, 2=left, 3=right
        if action == 0:
            d_row, d_col = -1, 0
        elif action == 1:
            d_row, d_col = 1, 0
        elif action == 2:
            d_row, d_col = 0, -1
        else:
            d_row, d_col = 0, 1

        new_head = (row + d_row, col + d_col)

        # Collision with wall or self
        if (new_head in self.body or
            not 0 <= new_head[0] < GRID_SIZE or
            not 0 <= new_head[1] < GRID_SIZE):
            return True, False  # game over, did not eat food

        ate_food = (new_head == food)
        self.body.insert(0, new_head)
        if ate_food:
            self.length += 1
            generate_food()
        else:
            self.body.pop()

        return False, ate_food

    def draw(self):
        for (row, col) in self.body:
            pygame.draw.rect(screen, GREEN,
                             (col*BLOCK_SIZE, row*BLOCK_SIZE, BLOCK_SIZE-1, BLOCK_SIZE-1))

# ---------- Food ----------
def generate_food():
    global food
    while True:
        food = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if food not in snake.body:
            break

# ---------- State ----------
def get_state():
    row, col = snake.body[0]
    f_row, f_col = food

    # Direction to food
    food_d_row = np.sign(f_row - row)
    food_d_col = np.sign(f_col - col)

    # Wall proximity
    wall_up = (row == 0)
    wall_down = (row == GRID_SIZE - 1)
    wall_left = (col == 0)
    wall_right = (col == GRID_SIZE - 1)

    # Obstacles (snake body)
    obstacle_up    = (row-1, col) in snake.body[1:]
    obstacle_down  = (row+1, col) in snake.body[1:]
    obstacle_left  = (row, col-1) in snake.body[1:]
    obstacle_right = (row, col+1) in snake.body[1:]

    return (food_d_row, food_d_col,
            wall_up, wall_down, wall_left, wall_right,
            obstacle_up, obstacle_down, obstacle_left, obstacle_right)

# ---------- Q-learning ----------
q_table = {}
load_q_table()

def choose_action(state, epsilon_override=None):
    legal_actions = [0, 1, 2, 3]
    eps = epsilon_override if epsilon_override is not None else EPSILON
    if random.random() < eps:
        return random.choice(legal_actions)
    q_vals = [q_table.get((state, a), 0) for a in range(4)]
    return int(np.argmax(q_vals))

def update_q_table(state, action, reward, next_state):
    current_q = q_table.get((state, action), 0)
    next_max_q = max([q_table.get((next_state, a), 0) for a in range(4)])
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + GAMMA * next_max_q)
    q_table[(state, action)] = new_q

# ---------- Training ----------
snake = Snake()
generate_food()

try:
    for episode in range(EPISODES):
        snake.reset()
        generate_food()
        state = get_state()
        total_reward = 0

        for step in range(MAX_STEPS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("[👋] Quitting — saving Q-table...")
                    save_q_table()
                    pygame.quit()
                    sys.exit()

            prev_dist = np.linalg.norm(np.array(snake.body[0]) - np.array(food))
            action = choose_action(state)
            done, ate_food = snake.move(action)
            new_dist = np.linalg.norm(np.array(snake.body[0]) - np.array(food))
            next_state = get_state()

            # Reward shaping
            if done:
                reward = -10
            elif ate_food:
                reward = +10
            elif new_dist < prev_dist:
                reward = +0.1
            else:
                reward = -0.1

            update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                break

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"Episode {episode} | Total Reward: {total_reward:.1f} | Epsilon: {EPSILON:.3f}")

        if episode % SAVE_INTERVAL == 0 and episode > 0:
            save_q_table()

    save_q_table()

except KeyboardInterrupt:
    print("[🛑] Training interrupted — saving progress...")
    save_q_table()

# ---------- Evaluation ----------
snake.reset()
generate_food()

print("[🎮] Training done — entering evaluation mode.")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    state = get_state()
    # Deterministic policy for evaluation
    action = choose_action(state, epsilon_override=0)
    done, _ = snake.move(action)

    if done:
        snake.reset()
        generate_food()

    screen.fill(BLACK)
    snake.draw()

    # Food
    f_row, f_col = food
    pygame.draw.rect(screen, RED,
                     (f_col*BLOCK_SIZE, f_row*BLOCK_SIZE, BLOCK_SIZE-1, BLOCK_SIZE-1))

    pygame.display.flip()
    clock.tick(FPS)
