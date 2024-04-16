from tetris_lib import *

import gymnasium as gym
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import tianshou as ts
import torch, numpy as np
from torch import nn
import signal
import sys

def save_policy(policy, filename='dqn.pth'):
    torch.save(policy.state_dict(), filename)

def main(load_from_file='', render_tests=0):
    # Setup the environments
    env = Tetris(20, 10)
    train_envs = Tetris(20, 10, True, 1)
    test_envs = Tetris(20, 10, True, 2)

    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.action_shape = action_shape

            self.input_shape = state_shape
            self.output_shape = action_shape
            self.hidden_shape = 200 #int(self.input_shape / 2)

            self.model = nn.Sequential(
                nn.Linear(self.input_shape, 200), nn.ReLU(inplace=True),
                nn.Linear(200, 200), nn.ReLU(inplace=True),
                nn.Linear(200, 1000), nn.ReLU(inplace=True),
                nn.Linear(1000, 1000), nn.ReLU(inplace=True),
                nn.Linear(1000, 200), nn.ReLU(inplace=True),
                nn.Linear(200, self.output_shape)
            )

        def forward(self, obs, state=None, info={}):
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float)
            batch = obs.shape[0]
            logits = self.model(obs.view(batch, -1))
            logits = logits.view(batch, self.action_shape)
            return logits, state
    
    # Create the network
    state_shape = 200 #14
    action_shape = 5
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)

    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=1,
        target_update_freq=0,
        reward_normalization=False,
        is_double=True,
        clip_loss_grad=False
    )

    if load_from_file == 'd': # Default
        print("Loaded default file")
        policy.load_state_dict(torch.load('dqn.pth'))
    elif load_from_file != '' and load_from_file != ' ':
        print("Loaded file:", load_from_file)
        policy.load_state_dict(torch.load(load_from_file))

    # Create the collectors
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

    # Variables:
    total_steps = 10000
    number_of_episodes_per_training = 50
    number_of_episodes_per_testing = 50
    number_of_steps_before_testing = 50

    reward_threshold = 100.0

    # policy.set_eps(0.1)
    for i in range(int(total_steps)):  # total step
        try:
            print("New step:", i, end='\t')
            collect_result = train_collector.collect(n_episode=number_of_episodes_per_training)#, render=1 / 35)
            print(train_envs.get_steps_count(), end='\t')
            print(collect_result['rew'])

            # once if the collected episodes' mean returns reach the threshold,
            # or every 1000 steps, we test it on test_collector
            if collect_result['rews'].mean() >= reward_threshold or i % number_of_steps_before_testing == 0:
                print("Following collected training result made it:", collect_result)
                # policy.set_eps(0.05)
                result = test_collector.collect(n_episode=number_of_episodes_per_testing)#, render=1 / 35)
                print("Collected testing result:", result)
                print("Steps in testing total:", test_envs.get_steps_count())

                if render_tests > 0:
                    test_collector.collect(n_episode=render_tests, render=1 / 35)
                    test_envs.close()

                save_policy(policy)

                if result['rews'].mean() >= reward_threshold * 5:
                    print(f'Finished training! Test mean returns: {result["rews"].mean()}')
                    break
                # else:
                    # back to training eps
                    # policy.set_eps(0.1)

            # train policy with a sampled batch data from buffer
            losses = policy.update(64, train_collector.buffer)
        except KeyboardInterrupt:
            print("Interrupted but still saving network")
            save_policy(policy, 'dqn_interrupted.pth')

            received_input = input("What now?\t")

            splitted_input = received_input.split(' ')

            if splitted_input[0] == 'r' or splitted_input[0] == 'render':
                nr_of_episodes = 1
                if len(splitted_input) > 1:
                    nr_of_episodes = int(splitted_input[1])
                policy.set_eps(0.05)
                test_collector.collect(n_episode=nr_of_episodes, render=1 / 35)
                test_envs.close()
                policy.set_eps(0.1)
            elif splitted_input[0] == 'e' or splitted_input[0] == 'exit':
                exit(0)

    # Save network
    print("Now saving network")
    save_policy(policy)

if __name__ == "__main__":
    load_from_file = ''
    render_tests = 0
    if len(sys.argv) >= 2:
        load_from_file = sys.argv[1]
    if len(sys.argv) >= 3:
        render_tests = int(sys.argv[2])
    main(load_from_file, render_tests)