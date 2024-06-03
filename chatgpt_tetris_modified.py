import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf
from collections import deque
from torch import nn
import matplotlib.pyplot as plt

from tetris_lib import *

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 1.0  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
                nn.Linear(self.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 24),
                nn.ReLU(),
                nn.Linear(24, self.action_size)
            )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0])
            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

def plot_reward(rewards, episode):
    plt.plot(rewards)
    plt.title(f"Rewards episode: {episode + 1}")
    plt.xlabel("Nr of episodes")
    plt.ylabel("Average reward")
    plt.pause(0.1)  # Pause to allow time for the plot to update (adjust as needed)
    plt.clf()  # Clear the plot for the next update

if __name__ == "__main__":
    env = Tetris(20, 10, True)
    state_size = 14 # env.observation_space.shape[0]
    action_size = 5 # env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    update_target_every = 10

    average_reward = []
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0.0
        total_time = 0
        for time in range(500):
            # if e % 10 == 0:  # Render only every 10 episodes
            #     env.render()

            action = agent.act(state)
            next_state, (reward, piece_placed), done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_time = time

            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 25 == 0:
            agent.save(f"saved_models/tetris-dqn-torch3-{e}.weights.h5")

        if e % update_target_every == 0:
            agent.update_target_model()

        average_reward.append(total_reward / total_time)

        if e % 10 == 0 and e != 0:
            plot_reward(average_reward, e)
