import torch
from torch import nn
import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent():
    def __init__(self, observation_size, action_size):
        self.batch_size = 16

        self.model = nn.Sequential(
                nn.Linear(observation_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_size)
            )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.alpha = 1.0
        self.epsilon = 0.9
        self.gamma = 1.0

        self.experiences = []

        self.memory = ReplayMemory(10000)

    def get_Q(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def add_experience(self, s, a, r, s1, piece_placed):
        # self.experiences.append([s, a, r, s1, piece_placed])
        self.memory.push(s, a, s1, r)

    def _recall_experience(self):
        indices = random.sample(range(len(self.experiences)), self.batch_size)
        # indices = random.choices(range(len(self.experiences)), k=self.batch_size)

        # print(f"Indices: {indices}")

        s = torch.empty(self.batch_size)
        a = torch.empty(self.batch_size)
        r = torch.empty(self.batch_size)
        s1 = torch.empty(self.batch_size)
        pp = torch.empty(self.batch_size)

        for i in indices:
            s = torch.cat((s, torch.tensor([self.experiences[i][0]])))
            s.append(self.experiences[i][0])
            a.append(self.experiences[i][1])
            r.append(self.experiences[i][2])
            s1.append(self.experiences[i][3])
            pp.append(self.experiences[i][4])

        indices.sort()
        indices.reverse()
        for i in indices:
            del self.experiences[i]

        print(s)
        s = torch.Tensor(s)
        a = torch.Tensor(a)
        r = torch.Tensor(r)
        s1 = torch.Tensor(s1)
        pp = torch.Tensor(pp)

        return s, a, r, s1, pp

    def learn_from_experience(self):
        if self.batch_size > len(self.experiences):
            return
        
        # s, a, r, s1, piece_placed = self._recall_experience()
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        s = batch.state
        a = batch.action
        s1 = batch.next_state
        r = batch.reward

        state_action_values = self.model(s).gather(1, torch.reshape(a, (len(a), 1)).long())
        state_action_values = torch.reshape(state_action_values, (len(a),))

        # current = []
        # for i in range(len(Q)):
        #     current.append(Q[i][a[i]])
        # current = torch.tensor(current)

        # new_Q = self.get_Q(s1).max(1).values
        with torch.no_grad():
            next_state_values = self.model(torch.Tensor(s1)).max(1).values

        # target = current + self.alpha * (torch.tensor(r) + (self.gamma * new_Q) - current)
        excepted_state_action_values = r + (self.gamma * next_state_values)

        # In case of a piece being placed
        # for i in range(len(target)):
        #     if piece_placed[i]:
        #         target[i] = r[i]

        criterion = nn.SmoothL1Loss()

        # loss = criterion(current, target)
        loss = criterion(state_action_values, excepted_state_action_values)
        loss.backward() # Compute gradients
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step() # Backpropagate error

    def save_model(self):
        torch.save(self.model.state_dict(), 'model.pth')