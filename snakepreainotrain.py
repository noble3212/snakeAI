import pygame
import random
from collections import deque

# --- Game settings ---
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
CELL_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE
FPS = 10

# --- Snake game ---
class SnakeGameAI:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)  # moving right
        self.head = (GRID_WIDTH//2, GRID_HEIGHT//2)
        self.snake = deque([self.head,
                            (self.head[0]-1, self.head[1]),
                            (self.head[0]-2, self.head[1])])
        self.score = 0
        self.spawn_food()
        self.frame_iteration = 0

    def spawn_food(self):
        while True:
            x = random.randint(0, GRID_WIDTH-1)
            y = random.randint(0, GRID_HEIGHT-1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def play_step(self, action):
        self.frame_iteration += 1
        # handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # move snake
        self.direction = action
        self.head = (self.head[0]+self.direction[0], self.head[1]+self.direction[1])
        self.snake.appendleft(self.head)

        # check death
        reward = 0
        game_over = False
        if (self.head[0]<0 or self.head[0]>=GRID_WIDTH or
            self.head[1]<0 or self.head[1]>=GRID_HEIGHT or
            self.head in list(self.snake)[1:]):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # check food
        if self.head == self.food:
            self.score += 1
            reward = +10
            self.spawn_food()
        else:
            self.snake.pop()

        # draw everything
        self.display.fill(pygame.Color('black'))
        for pt in self.snake:
            pygame.draw.rect(self.display, pygame.Color('green'),
                             pygame.Rect(pt[0]*CELL_SIZE, pt[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.display, pygame.Color('red'),
                         pygame.Rect(self.food[0]*CELL_SIZE, self.food[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()
        self.clock.tick(FPS)

        return reward, game_over, self.score

    def get_next_action(self):
        # Simple AI: move towards food with basic collision avoidance
        head_x, head_y = self.head
        food_x, food_y = self.food
        # Possible directions: right, left, down, up
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        safe_moves = []
        for dir in directions:
            nx, ny = head_x + dir[0], head_y + dir[1]
            if (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and
                (nx, ny) not in self.snake):
                safe_moves.append(dir)
        # move towards food if safe
        best_move = self.direction
        if safe_moves:
            best_move = min(safe_moves, key=lambda d: abs((head_x+d[0])-food_x)+abs((head_y+d[1])-food_y))
        return best_move

# --- Main loop ---
def main():
    game = SnakeGameAI()
    while True:
        action = game.get_next_action()
        reward, done, score = game.play_step(action)
        if done:
            print("Game over! Score:", score)
            pygame.time.delay(1000)
            game.reset()

if __name__ == "__main__":
    main()
