import pygame
import random

import gymnasium as gym
import numpy as np
import copy
import torch
import sys

colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


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
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, height, width, track_steps=False, env_type=0):
        super(Tetris, self).__init__()

        self.track_steps = track_steps
        self.steps_count = 0

        self.env_type = env_type # 0 = nothing special, 1 = train, 2 = test
        self.TRAIN = 1
        self.TEST = 2

        self.screen = None

        self.height = height
        self.width = width

        self.initialize()

        self.observation_shape = (height, width)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_shape)
        
        self.action_space = gym.spaces.Discrete(6) # Left, right, soft drop, hard drop, rotate clockwise, rotate counter-clockwise

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
        self.new_reward_gotten = False
        self.placed_height = self.height
        self.height_difference = (0, 0, 0) # Height difference, max height, min height
        self.steps_per_episode = 0
    
        self.score = 0
        for i in range(self.height):
            new_line = []
            for j in range(self.width):
                new_line.append(0)
            self.field.append(new_line)

        self.new_figure()

    def seed(self, seed):
        np.random.seed(seed)

    def get_steps_count(self):
        temp = self.steps_count
        self.steps_count = 0
        return temp
    
    def retrieve_state(self):
        state = []

        # First 10 elements are height of each column
        for column in range(self.width):
            max_height = self.height
            for row in range(self.height):
                if self.field[row][column] > 0:
                    max_height = row
            state.append(self.height - max_height)

        state.append(self.figure.x)
        state.append(self.figure.y)
        state.append(self.figure.type)
        state.append(self.figure.rotation)

        return state

        # figure_coordinates = []
        # for i in range(4):
        #     for j in range(4):
        #         if i * 4 + j in self.figure.image():
        #             figure_coordinates.append((i + self.figure.y, j + self.figure.x))

        # new_field = []
        # for row in range(self.height):
        #     # temp_field = []
        #     for column in range(self.width):
        #         if ((row, column) in figure_coordinates and self.field[row][column] == 0) or self.field[row][column] > 0:
        #             new_field.append(1)
        #         else:
        #             new_field.append(0)

        #     # new_field.append(temp_field)

        # return new_field
    
    def print_field(self, field):
        print("Printing field:")
        for row in range(len(field)):
            for column in range(len(field[row])):
                print(field[row][column], end='\t')
            print()
        print("\n")


    ##### Env functions #####
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.initialize()

        return self.retrieve_state()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        if action == 0: # Left
            self.go_side(-1)
        elif action == 1: # Right
            self.go_side(1)
        elif action == 2: # Soft drop
            self.go_down()
        elif action == 3: # Hard drop
            self.go_space()
        elif action == 4: # Rotate counter-clockwise
            self.rotate()

        self.go_down()

        reward = self.calculate_reward()

        if self.figure is None:
            self.new_figure()

        self.steps_per_episode += 1
        if self.track_steps:
            self.steps_count += 1
        
        return self.retrieve_state(), reward, (self.state == "gameover")
    
    def render(self):
        if self.screen == None:
            # Initialize the game engine
            pygame.init()
            self.size = (400, 500)
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Tetris")
            self.clock = pygame.time.Clock()

        # Define some colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (128, 128, 128)

        # Loop until the user clicks the close button.
        fps = 60

        self.screen.fill(WHITE)

        for i in range(self.height):
            for j in range(self.width):
                pygame.draw.rect(self.screen, GRAY, [self.x + self.zoom * j, self.y + self.zoom * i, self.zoom, self.zoom], 1)
                if self.field[i][j] > 0:
                    pygame.draw.rect(self.screen, colors[self.field[i][j]],
                                    [self.x + self.zoom * j + 1, self.y + self.zoom * i + 1, self.zoom - 2, self.zoom - 1])

        if self.figure is not None:
            for i in range(4):
                for j in range(4):
                    p = i * 4 + j
                    if p in self.figure.image():
                        pygame.draw.rect(self.screen, colors[self.figure.color],
                                        [self.x + self.zoom * (j + self.figure.x) + 1,
                                        self.y + self.zoom * (i + self.figure.y) + 1,
                                        self.zoom - 2, self.zoom - 2])

        font = pygame.font.SysFont('Calibri', 25, True, False)
        text = font.render("Score: " + str(self.score), True, BLACK)

        self.screen.blit(text, [0, 0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    ##########################


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

    def break_lines(self):
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

        old_nr_of_holes = self.nr_of_holes
        self.nr_of_holes = self.check_holes(self.field)
        self.nr_of_holes_changed = self.nr_of_holes > old_nr_of_holes

        self.score += lines ** 2
        self.current_reward = lines
        self.new_reward_gotten = True
        self.height_difference = self.check_height_difference()

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
                    self.placed_height = min(self.placed_height, self.height - (i + self.figure.y + 1))

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


    def calculate_reward(self):
        reward = -0.1

        piece_placed = self.new_reward_gotten
        if self.new_reward_gotten == True:
            # Just placed a piece, so reset reward variable
            self.new_reward_gotten = False

            if self.nr_of_holes_changed:
                reward = -20
                reward = self.calculate_piece_placement_reward(reward)
            else:
                if self.current_reward > 0:
                    print("LINE CLEARED")

                if self.current_reward == 0: # No lines
                    reward = 15
                    reward = self.calculate_piece_placement_reward(reward)
                elif self.current_reward == 1: # One line
                    reward = 60
                elif self.current_reward == 2: # Two lines
                    reward = 70
                elif self.current_reward == 3: # Three lines
                    reward = 80
                elif self.current_reward == 4: # Four lines (tetris)
                    reward = 100

        if self.state == "gameover":
            reward = -100

        return float(reward), piece_placed
    
    def calculate_piece_placement_reward(self, reward):
        if self.placed_height >= self.height_difference[1]:
            reward -= self.height_difference[1]
        elif self.placed_height < self.height_difference[1]:
            reward += (self.height - self.placed_height)

        # reward -= (self.height_difference[0] - 4)

        self.reset_rewards()

        return reward
    
    def reset_rewards(self):
        self.height_difference = (0, 0, 0)
        self.placed_height = self.height
    
    def check_height_difference(self):
        heights = [0] * self.width

        for column in range(self.width):
            height_found = False

            for row in range(self.height):
                if self.field[row][column] > 0:
                    heights[column] = self.height - row
                    height_found = True
                    break

            if not height_found:
                heights[column] = 0

        min_height = min(heights)
        max_height = max(heights)

        height_difference = max_height - min_height

        return (height_difference, max_height, min_height)
    
    def print_Q(self, Q):
        print(f"Left: {Q[0]}, right: {Q[1]}, soft drop: {Q[2]}, hard drop: {Q[3]}, rotate: {Q[4]}")