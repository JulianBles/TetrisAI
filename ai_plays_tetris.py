import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tetris_lib import *

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def load(self, name):
        self.model.load_weights(name)

    def act(self, state):
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

if __name__ == "__main__":
    env = Tetris(20, 10, True)
    state_size = 200 # env.observation_space.shape[0]
    action_size = 5 # env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("tetris-dqn-15.weights.h5")  # replace with your actual weights file

    while True:
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            env.render()
            action = agent.act(state)
            next_state, (reward, piece_placed), done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
