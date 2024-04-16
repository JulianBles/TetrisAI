import gym

env = gym.make("Taxi-v3").env

env.reset() # reset environment to a new, random state
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()

print(env.P[328]) # {action: [(probability, nextstate, reward, done)]}