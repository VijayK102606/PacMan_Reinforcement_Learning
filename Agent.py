import torch
import random
import numpy as np
from collections import deque
from Game import PacMan, Ghost
from Model import Linear_QNet, QTrainer
import pygame

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
TILE_SIZE = 20
SCREEN_WIDTH = 420
SCREEN_HEIGHT = 460
FPS = 30
BLUE = (0, 0, 255)
BLACK = (255, 0, 0)
PINK = (255, 105, 180)
RED = (0, 0, 0)
duration = 7500
legal_moves = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 7, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 2, 2, 2, 2],
        [2, 2, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(17, 256, 4)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, pac_man, ghosts):
        x, y = pac_man.x // TILE_SIZE, pac_man.y // TILE_SIZE
        dx, dy = pac_man.dx, pac_man.dy
        isFood = pac_man.legal_moves[y][x] == 0

        wall_up = pac_man.legal_moves[y - 1][x] if y > 0 else 1
        wall_down = pac_man.legal_moves[y + 1][x] if y < len(pac_man.legal_moves) - 1 else 1
        wall_left = pac_man.legal_moves[y][x - 1] if x > 0 else 1
        wall_right = pac_man.legal_moves[y][x + 1] if x < len(pac_man.legal_moves[0]) - 1 else 1

        ghost_positions = [0] * 8
        for i, ghost in enumerate(ghosts):
            ghost_positions[i * 2] = ghost.x // TILE_SIZE
            ghost_positions[i * 2 + 1] = ghost.y // TILE_SIZE

        state = [
            x, y, dx, dy,
            isFood,
            wall_up, wall_down, wall_left, wall_right,
            *ghost_positions,
        ]
        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            min_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            min_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*min_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def draw_background(screen):
    for row in range(len(legal_moves)):
        for col in range(len(legal_moves[row])):
            val = legal_moves[row][col]
            if val == 1:
                pygame.draw.rect(screen, BLUE, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Blue
            elif val == 0:
                pygame.draw.rect(screen, BLACK, ((col * TILE_SIZE) + 7.5, row * TILE_SIZE + 7.5, 5, 5), 0)
            elif val == 7:
                pygame.draw.rect(screen, PINK,
                                 (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Pink
            elif val == 2:
                pygame.draw.rect(screen, RED, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))  # Black
    return legal_moves

def reset(pacman, a):
    pacman.reset()
    a.n_games += 1
    a.train_long_memory()

def learn(a, pacman, ghosts):
    state_old = a.get_state(pacman, ghosts)
    final_move = a.get_action(state_old)
    action_index = final_move.index(1)

    reward, done = pacman.play_step(action_index, ghosts)
    state_new = a.get_state(pacman, ghosts)

    a.train_short_memory(state_old, final_move, reward, state_new, done)
    a.remember(state_old, final_move, reward, state_new, done)

def train():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    a = Agent()

    start_time = None

    pacman = PacMan(20, 20, legal_moves)
    pacman.legal_moves = draw_background(screen)
    ghosts = []
    ghosts.append(Ghost(100, 100, legal_moves))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        state_old = a.get_state(pacman, ghosts)
        final_move = a.get_action(state_old)
        action_index = final_move.index(1)

        reward, done = pacman.play_step(action_index, ghosts)

        if pacman.power_up and start_time is None:
            start_time = pygame.time.get_ticks()

        if start_time is not None:
            curr_time = pygame.time.get_ticks()
            if curr_time - start_time >= duration:
                pacman.power_up = False
                start_time = None

        state_new = a.get_state(pacman, ghosts)

        a.train_short_memory(state_old, final_move, reward, state_new, done)
        a.remember(state_old, final_move, reward, state_new, done)


        for ghost in ghosts:
            ghost.move()

        if len(ghosts) < 4:
            new_ghost_x = random.randint(0, (SCREEN_WIDTH // TILE_SIZE) - 1) * TILE_SIZE
            new_ghost_y = random.randint(0, (SCREEN_HEIGHT // TILE_SIZE) - 1) * TILE_SIZE
            ghosts.append(Ghost(new_ghost_x, new_ghost_y, legal_moves))

        if done: reset(pacman, a)

        screen.fill((0, 0, 0))
        draw_background(screen)
        pacman.draw(screen)
        for ghost in ghosts:
            ghost.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

train()





