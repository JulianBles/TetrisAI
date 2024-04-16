import pygame
import random

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym

from gym import Env, spaces
import time
import copy
import json
from json import JSONEncoder

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Figure:
    x = 0
    y = 0

    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        [[4, 5, 9, 10], [2, 6, 5, 9]],
        [[6, 7, 9, 10], [1, 5, 6, 10]],
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        # self.type = 6
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris(Env):
    def __init__(self, height, width):
        global q_table
        super(Tetris, self).__init__()

        self.height = height
        self.width = width

        self.initialize()

        self.observation_shape = (height, width)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape))
        
        self.nr_of_actions = 6
        self.action_space = spaces.Discrete(self.nr_of_actions) # Left, right, soft drop, hard drop, rotate clockwise, rotate counter-clockwise

        self.nr_of_drop_states = 6
        self.nr_of_pieces = 7
        self.nr_of_rotations = 4
        self.nr_of_x_locations = 10
        # Drop state = punishment, nothing, 1 row, 2, 3, 4
        q_table = np.zeros([self.nr_of_drop_states * self.nr_of_pieces * self.nr_of_rotations * self.nr_of_x_locations * self.height,
                            self.nr_of_actions]) # Drop state TIMES the nr of different pieces TIMES nr of rotations TIMES nr of x locations

    def initialize(self):
        self.level = 10 # Change this to change the speed
        self.score = 0
        self.state = "start"
        self.field = []
        self.x = 100
        self.y = 60
        self.zoom = 20
        self.figure = None

        self.nr_of_holes = 0
        self.nr_of_holes_changed = False
        self.current_reward = 0
        self.real_reward = False
        self.game_state = 1
        self.previous_highest_y = 0
        self.current_highest_y = 0
    
        self.score = 0
        self.state = "start"
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)

        self.new_figure()

    def new_figure(self):
        self.figure = Figure(3, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def break_lines(self, real_reward=True):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]

        self.score += lines ** 2
        self.current_reward = lines
        self.real_reward = real_reward

        if real_reward:
            old_nr_of_holes = self.nr_of_holes
            self.nr_of_holes = self.check_holes(self.field)
            self.nr_of_holes_changed = self.nr_of_holes > old_nr_of_holes

            self.previous_highest_y = self.current_highest_y # Exactly lowest because y goes the other way
            new_highest_y_found = False
            for row in range(self.height):
                for column in range(self.width):
                    if self.field[row][column] > 0:
                        self.current_highest_y = row
                        new_highest_y_found = True
                        break
                if new_highest_y_found:
                    break

    def go_space(self, freeze=True):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        
        if freeze:
            self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color

        self.break_lines()
        self.new_figure()
    
        if self.intersects():
            self.state = "gameover"

    def go_side(self, dx):
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x

    def rotate(self):
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation

    def check_holes(self, field):
        nr_of_holes = 0
        for row in range(1, self.height):
            for column in range(self.width):
                if field[row][column] == 0 and field[row - 1][column] > 0:
                    nr_of_holes += 1

        return nr_of_holes


    def calculate_game_state(self):
        current_x = self.figure.x
        current_y = self.figure.y
        current_field = copy.deepcopy(self.field)

        self.go_space(False)
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines(False)

        drop_state = 0
        # Check if nr of holes has increased
        if self.nr_of_holes_changed:
            drop_state = 0
        else:
            drop_state = self.current_reward + 1

        self.game_state = drop_state 
        + self.nr_of_drop_states * self.figure.type 
        + self.figure.rotation * self.nr_of_drop_states * self.nr_of_pieces 
        + self.figure.x * self.nr_of_drop_states * self.nr_of_pieces * self.nr_of_rotations 
        + self.figure.y * self.nr_of_drop_states * self.nr_of_pieces * self.nr_of_rotations * self.width

        self.figure.x = current_x
        self.figure.y = current_y
        self.field = current_field

        # print("Gamestate:", self.game_state)

        return drop_state

    def calculate_reward(self):
        reward = 0

        if self.real_reward == True:
            # Just placed a piece

            if self.nr_of_holes_changed:
                reward = -20
            else:
                if self.current_reward == 0: # No lines
                    reward = 100
                elif self.current_reward == 1: # One line
                    reward = 300
                elif self.current_reward == 2: # Two lines
                    reward = 400
                elif self.current_reward == 3: # Three lines
                    reward = 500
                elif self.current_reward == 4: # Four lines (tetris)
                    reward = 1000

            # If highest piece has become higher, add -20 to reward
            if self.current_highest_y < self.previous_highest_y:
                reward += -200
        
        old_game_state = self.game_state
        drop_state = self.calculate_game_state()
        if reward == 0:
            if drop_state == 0: # Punishment
                reward = -5
            elif drop_state == 1: # No lines
                reward = 0.5
            elif drop_state == 2: # One line
                reward = 3
            elif drop_state == 3: # Two lines
                reward = 5
            elif drop_state == 4: # Three lines
                reward = 7
            elif drop_state == 5: # Four lines
                reward = 10

        if old_game_state == self.game_state:
            reward += -50

        if self.state == "gameover":
            reward += -1000

        return reward

    def reset(self):
        self.initialize()

        self.calculate_game_state()
        return self.game_state

    def step(self, action):
        self.go_down()

        assert self.action_space.contains(action), "Invalid Action"

        if action == 0: # Left
            self.go_side(-1)
        elif action == 1: # Right
            self.go_side(1)
        elif action == 2: # Soft drop
            self.go_down()
        elif action == 3: # Hard drop
            self.go_space()
        elif action == 4: # Rotate clockwise
            self.rotate()
            self.rotate()
            self.rotate()
        elif action == 5: # Rotate counter-clockwise
            self.rotate()

        reward = self.calculate_reward()
        
        return (self.game_state, reward)

# Initialize the game engine
pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

size = (400, 500)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Tetris")

# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()
fps = 25
q_table = None
game = Tetris(20, 10)
counter = 0

pressing_down = False


with open("tetris_AI_V3.1.json", "r") as file:
    decoded_data = json.loads(file.read())
    q_table = np.asarray(decoded_data)
    
# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

epochs, penalties, reward = 0, 0, 0

state = game.reset()

while not done:
    counter += 1
    if counter > 100000:
        counter = 0

    if counter % (fps // game.level // 2) == 0 or pressing_down:
        if game.state == "start":
            # action = game.action_space.sample()
            # game.step(action)

            if random.uniform(0, 1) < epsilon:
                action = game.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            # print("Action:", action)

            next_state, reward = game.step(action)

            print("Reward:", reward)
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1
                print("Penalties:", penalties)

            state = next_state
            epochs += 1

            # print(q_table)
        elif game.state == "gameover":
            json_data = json.dumps(q_table, cls=NumpyArrayEncoder)
            with open("tetris_AI_V3.1.json", "w") as file:
                file.write(json_data + "\n")

            game.reset()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                game.rotate()
                game.rotate()
                game.rotate()
            if event.key == pygame.K_DOWN:
                pressing_down = True
            if event.key == pygame.K_LEFT:
                game.go_side(-1)
            if event.key == pygame.K_RIGHT:
                game.go_side(1)
            if event.key == pygame.K_SPACE:
                game.go_space()
            if event.key == pygame.K_ESCAPE:
                game.reset()

    if event.type == pygame.KEYUP:
            if event.key == pygame.K_DOWN:
                pressing_down = False

    screen.fill(WHITE)

    for i in range(game.height):
        for j in range(game.width):
            pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
            if game.field[i][j] > 0:
                pygame.draw.rect(screen, colors[game.field[i][j]],
                                 [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

    if game.figure is not None:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in game.figure.image():
                    pygame.draw.rect(screen, colors[game.figure.color],
                                     [game.x + game.zoom * (j + game.figure.x) + 1,
                                      game.y + game.zoom * (i + game.figure.y) + 1,
                                      game.zoom - 2, game.zoom - 2])

    font = pygame.font.SysFont('Calibri', 25, True, False)
    font1 = pygame.font.SysFont('Calibri', 65, True, False)
    text = font.render("Score: " + str(game.score), True, BLACK)
    text_game_over = font1.render("Game Over", True, (255, 125, 0))
    text_game_over1 = font1.render("Press ESC", True, (255, 215, 0))

    screen.blit(text, [0, 0])
    if game.state == "gameover":
        screen.blit(text_game_over, [20, 200])
        screen.blit(text_game_over1, [25, 265])

    pygame.display.flip()
    clock.tick(fps)

pygame.quit()