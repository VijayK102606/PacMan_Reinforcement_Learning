import pygame
import random

# Constants
TILE_SIZE = 20
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
PACMAN_COLOR = (255, 255, 0)
GHOST_COLOR = (255, 0, 0)
DOT_RADIUS = 10
SCREEN_WIDTH = 420
SCREEN_HEIGHT = 460

class PacMan:
    def __init__(self, x, y, legal_moves):
        self.x = x
        self.y = y
        self.dx = 0
        self.dy = 0
        self.legal_moves = legal_moves
        self.power_up = False

    def draw(self, screen):
        pygame.draw.circle(screen, PACMAN_COLOR, (self.x + TILE_SIZE // 2, self.y + TILE_SIZE // 2), DOT_RADIUS)

    def play_step(self, action, ghosts):
        dx, dy = ACTIONS[action]
        x = self.x + (dx * TILE_SIZE)
        y = self.y + (dy * TILE_SIZE)
        reward = 0

        if self.legalMove(x, y):
            self.x = x
            self.y = y
            curr_x = x // TILE_SIZE
            curr_y = y // TILE_SIZE
            val = self.legal_moves[curr_y][curr_x]
            if val == 0:
                reward += 1
                self.legal_moves[curr_y][curr_x] = 2
            if val == 7:
                reward += 10
                self.legal_moves[curr_y][curr_x] = 2
                self.power_up = True


        done = False
        collision, hit = self.isCollision(ghosts)

        if collision and self.power_up:
            reward += 30
            ghosts.remove(hit)
        elif collision and not self.power_up:
            reward -= 30
            done = True

        return reward, done

    def reset(self):
        self.x = 20
        self.y = 20

    def isCollision(self, ghosts):
        for i in range(len(ghosts)):
            if self.x == ghosts[i].getX() and self.y == ghosts[i].getY():
                return True, ghosts[i]
        return False, 0

    def legalMove(self, x, y):
        col = x // TILE_SIZE
        row = y // TILE_SIZE
        return self.legal_moves[row][col] != 1

class Ghost:
    def __init__(self, x, y, legal_moves):
        self.x = x
        self.y = y
        self.legal_moves = legal_moves

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def draw(self, screen):
        pygame.draw.circle(screen, GHOST_COLOR, (self.x + TILE_SIZE // 2, self.y + TILE_SIZE // 2), DOT_RADIUS)

    def move(self):
        dx, dy = random.choice(ACTIONS)
        x = self.x + dx * TILE_SIZE
        y = self.y + dy * TILE_SIZE

        if self.legalMove(x, y):
            self.x = x
            self.y = y

    def legalMove(self, x, y):
        col = x // TILE_SIZE
        row = y // TILE_SIZE
        return self.legal_moves[row][col] != 1