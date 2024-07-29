from ANNarchy import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy.sparse
import gymnasium as gym
from scipy.special import erf
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
max_steps = 1000
terminated = False
truncated = False
episodes = 100
h=0
#Final fitness 
final_fitness = 0
while h < episodes:
    j=0
    returns = []
    actions_done = []
    terminated = False
    truncated = False
    while j < max_steps and not terminated and not truncated:
        #Choose the action with the most spikes
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("R:",reward)
        returns.append(reward)
        actions_done.append(action)
        j += 1
    env.reset()
    final_fitness += np.sum(returns)
    print("H:",h)
    h += 1

final_fitness = final_fitness/episodes
env.close()
print(final_fitness)


