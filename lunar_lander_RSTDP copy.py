import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler
from ns_gym.update_functions import IncrementUpdate

env = gym.make("Pendulum-v1")

scheduler = ContinuousScheduler()
update_g = IncrementUpdate(scheduler, k=0.01)
update_m = IncrementUpdate(scheduler, k=0.01)
update_l = IncrementUpdate(scheduler, k=0.01)
update_dt = IncrementUpdate(scheduler, k=0.001)

tunable_params = {
    "g": update_g,
    "m": update_m,
    "l": update_l,
    "dt": update_dt
}

ns_env = NSClassicControlWrapper(env, tunable_params, change_notification=True)

obs, info = ns_env.reset()
done = False
truncated = False
total_reward = 0

while not done and not truncated:
    action = ns_env.action_space.sample()
    obs, reward, done, truncated, info = ns_env.step(action)
    total_reward += reward.reward

print("Pendulum reward:", total_reward)
