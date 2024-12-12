#random lunar lander gymnasium
import gymnasium as gym




env = gym.make('Acrobot-v1', render_mode="rgb_array")
env.reset()
fitness = 0
episode = 0
returns = []
while episode < 2:
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, done, truncated, info = env.step(action)
        fitness += reward
        if done or truncated:
            print(fitness)
            returns.append(fitness)
            fitness = 0
            episode += 1
            break
env.close()

print("final:",sum(returns)/len(returns))

