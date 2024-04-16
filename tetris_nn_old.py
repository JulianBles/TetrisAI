from tetris_lib import *
from lib import *

import gymnasium as gym
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import tianshou as ts
import torch, numpy as np
from torch import nn
import signal
import sys
import matplotlib.pyplot as plt


env = Tetris(20, 10, True)

observation_size = 14
action_size = 5

model = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

# print("State dict")
# print(model.state_dict()) # TODO: use this for storing model parameters

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_reward = 1000.0

nr_of_epochs = 1000

alpha = 1.0
epsilon = 0.9
gamma = 1.0

for epoch in range(nr_of_epochs):
    done = False

    experiences = []

    # Sample episode and start filling experiences
    s = torch.FloatTensor(env.reset())
    while not done:
        Q = model(torch.Tensor(s))
        
        a = torch.argmax(Q).item() # Greedy action selection

        s1, r, done = env.step(a)

        experiences.append([Q, s, a, r, s1])

        s = s1

    # Learn from experiences

    # Randomly shuffle experiences
    # random.shuffle(experiences)

    for Q, s, a, r, s1 in experiences:
        # TODO: update neural network based on this information
        Qsa = torch.max(Q).item()
        maxQ = torch.max(model(torch.Tensor(s1))).item()
        optimal_Q = Qsa + alpha * (r + gamma * maxQ - Qsa)
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(torch.Tensor(Qsa), torch.Tensor(optimal_Q))

        optimizer.zero_grad()
        loss.backward()