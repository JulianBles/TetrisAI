from tetris_lib import *
from lib import *
from DQNAgent import *

import gymnasium as gym
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import tianshou as ts
import torch, numpy as np
from torch import nn
import signal
import sys
import math

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import multiprocessing


env = Tetris(20, 10, True)

observation_size = 200
action_size = 5

agent = DQNAgent(observation_size, action_size)

nr_of_episodes = 2000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

def select_action(steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

    if sample > eps_threshold:
        return -1
    else:
        return random.randint(0, action_size - 1)

def plot_reward(rewards, episode):
    plt.plot(rewards)
    plt.title(f"Rewards episode: {episode + 1}")
    plt.xlabel("Nr of episodes")
    plt.ylabel("Average reward")
    plt.pause(0.1)  # Pause to allow time for the plot to update (adjust as needed)
    plt.clf()  # Clear the plot for the next update

average_reward = []
i = 0
for episode in range(nr_of_episodes):
    done = False

    experiences = []

    # print(f"Running episode {episode}")

    j = i
    reward = 0.0

    Q = None
    # Sample episode and start filling experiences
    s = torch.FloatTensor(env.reset())
    while not done:
        i += 1

        Q = agent.get_Q(s)

        a = select_action(i)
        if a == -1:
            a = torch.argmax(Q).item()

        env.render()

        s1, (r, piece_placed), done = env.step(a)
        reward += r

        agent.add_experience(s, a, r, s1, piece_placed)
        agent.learn_from_experience()

        s = s1        

    # env.print_Q(Q)

    average_reward.append(reward / float(i - j))

    if episode % 10 == 0 and episode != 0:
        plot_reward(average_reward, episode)

agent.save_model()