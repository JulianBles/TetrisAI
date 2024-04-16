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

observation_size = 200 #14
action_size = 5

# model = nn.Sequential(
#             nn.Linear(observation_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, action_size)
#         )

model = nn.Sequential(
                nn.Linear(observation_size, 200), nn.ReLU(),
                nn.Linear(200, 200), nn.ReLU(),
                nn.Linear(200, 1000), nn.ReLU(),
                nn.Linear(1000, 1000), nn.ReLU(),
                nn.Linear(1000, 200), nn.ReLU(),
                nn.Linear(200, action_size)
            )


# loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

max_reward = 1000.0

total_losses = []
nr_of_epochs = 10000
for epoch in range(nr_of_epochs):
    done = False
    running_loss = 0.0
    index = 0.0

    state = env.reset()
    actions_used = []
    rewards = []
    current_reward = 0.0
    while not done:
        # optimizer.zero_grad()

        # Only required when state is list of lists
        # state = [item for row in state for item in row]
        state = torch.FloatTensor(state)
        # action = model(state)
        action_values = model(state.unsqueeze(0))
        # print("Action:", action_values)

        action_index = torch.argmax(action_values)
        actions_used.append(action_index.item())
        # print("Action index:", action_index.item())
        state, (reward, give_reward), done = env.step(action_index.item())

        env.render()

        rewards.append((action_values, action_index))
        current_reward += reward

        if give_reward:
            for av, ai in rewards:
                print(ai)
                running_loss += custom_backward(av, ai, current_reward, optimizer)
            
            rewards = []
            current_reward = 0.0
        
        # running_loss += custom_backward(action_values, action_index, reward, optimizer)
        index += 1.0

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

    average_loss = running_loss / index
    print("Average loss:", average_loss)
    print("Percentages:", calculate_percentage(actions_used))

    # Append the average loss to the list of training losses
    total_losses.append(average_loss)

    # Update the plot
    plt.plot(total_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.pause(0.1)  # Pause to allow time for the plot to update (adjust as needed)
    plt.clf()  # Clear the plot for the next update

    if epoch % 25 == 0:
        torch.save(model.state_dict(), 'model.pth')

    # # Code for loading the model:
    # # Load the saved model parameters
    # loaded_model.load_state_dict(torch.load('model.pth'))

    # # Set the model to evaluation mode (if needed)
    # loaded_model.eval()
