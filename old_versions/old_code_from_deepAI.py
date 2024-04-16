# Define the network
    # class Net(nn.Module):
    #     def __init__(self, state_shape, action_shape):
    #         super().__init__()
    #         self.model = nn.Sequential(
    #             nn.Linear(state_shape, 128), nn.ReLU(inplace=True),
    #             nn.Linear(128, 64), nn.ReLU(inplace=True),
    #             nn.Linear(64, 32), nn.ReLU(inplace=True),
    #             nn.Linear(32, 16), nn.ReLU(inplace=True),
    #             nn.Linear(16, action_shape)
    #         )

    #     def forward(self, obs, state=None, info={}):
    #         print("first obs shape:", obs.shape)
    #         if not isinstance(obs, torch.Tensor):
    #             obs = torch.tensor(obs, dtype=torch.float)
    #         batch = obs.shape[0]
    #         print("obs:", obs)
    #         print("obs shape:", obs.shape)
    #         print("batch:", batch)
    #         print("???:", obs.view(batch, -1).shape)
    #         input = obs.view(batch, -1)
    #         print("Idk man:", getattr(input, "mask", None))
    #         logits = self.model(obs.view(batch, -1))
    #         return logits, state



# Create the DQN policy
    # policy = ts.policy.DQNPolicy(
    #     model=net,
    #     optim=optim,
    #     action_space=env.action_space,
    #     discount_factor=0.9,
    #     estimation_step=3,
    #     target_update_freq=320
    # )


# Logger
    # writer = SummaryWriter('log/dqn')
    # logger = TensorboardLogger(writer)

    # Watch agents performance
    # policy.eval()
    # policy.set_eps(0.05)
    # collector = ts.data.Collector(policy, env, exploration_noise=True)
    # collector.collect(n_episode=10000, render=1 / 35)

    # assume obs is a single environment observation
    # action = policy(Batch(obs=np.array([obs]))).act[0]

    # Create the DQN offpolicty trainer and start
    # result = ts.trainer.OffpolicyTrainer(
    #     policy=policy,
    #     train_collector=train_collector,
    #     test_collector=test_collector,
    #     max_epoch=10, step_per_epoch=10000, step_per_collect=10,
    #     update_per_step=0.1, episode_per_test=100, batch_size=64,
    #     train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    #     test_fn=lambda epoch, env_step: policy.set_eps(0.05)
    #     # stop_fn=lambda mean_rewards: mean_rewards >= env.reward_threshold
    # ).run()
    # print(f'Finished training! Use {result["duration"]}')

    # pre-collect at least 5000 transitions with random action before training
    # pre_collected = train_collector.collect(n_step=5000, random=True)
    # print("Pre collected:", pre_collected)




# def check_height_difference(self):
    #     first_height_found = False
    #     first_height = 0

    #     lowest_height = 0

    #     for row in range(self.height):
    #         count_lowest_height = 0
    #         for column in range(self.width):                
    #             if self.field[row][column] > 0 and not first_height_found:
    #                 first_height = self.height - (row + 1)
    #                 first_height_found = True

    #             if self.field[row][column] > 0 and first_height_found:
    #                 count_lowest_height += 1

    #         if count_lowest_height >= 2:
    #             lowest_height = self.height - (row + 1)

    #     reward = 0
    #     if first_height - lowest_height > 5:
    #         reward = -15

    #     return reward