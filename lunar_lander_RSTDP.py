import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate

env = gym.make("MountainCarContinuous-v0")

scheduler = ContinuousScheduler()
update_function = IncrementUpdate(scheduler, k=0.01)

tunable_params = {"power": update_function}

ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True)

obs, info = ns_env.reset()
done = False
truncated = False
total_reward = 0

while not done and not truncated:
    action = ns_env.action_space.sample()
    obs, reward, done, truncated, info = ns_env.step(action)
    total_reward += reward.reward

print("MountainCarContinuous reward:", total_reward)
