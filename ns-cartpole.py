
###### Step 1: Import necessary gym and ns_gym modules
import gymnasium as gym
import ns_gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate


###### Step 2: Create a standard gym environment ####
env = gym.make("CartPole-v1", render_mode="human") 
#############

########## Step 3: to describe the evolution of the non-stationary parameters, 
# we define the two schedulers and update functions that model the semi-Markov chain over the relevant parameters
############
scheduler_1 = ContinuousScheduler()
scheduler_2 = PeriodicScheduler(period=3)

update_function1= IncrementUpdate(scheduler_1, k=0.5)
update_function2 = RandomWalk(scheduler_2)

##### Step 4: map parameters to update functions
tunable_params = {"masspole":update_function1, "gravity": update_function2}


######## Step 5: set notification level and pass environment and parameters into wrapper
ns_env = NSClassicControlWrapper(env,tunable_params,change_notification=True)



import matplotlib.pyplot as plt

# Lista para guardar la evolución de la gravedad
gravedad_por_step = []

# Reiniciar entorno
obs, info = ns_env.reset()
done = False
truncated = False
episode_reward = 0

while not done and not truncated:
    # Guardar el valor actual de gravedad
    gravedad_actual = ns_env.unwrapped.gravity

    gravedad_por_step.append(gravedad_actual)

    planning_env = ns_env.get_planning_env()
    mcts_agent = ns_gym.benchmark_algorithms.MCTS(env=planning_env, state=obs.state, gamma=1, d=500, m=100, c=2)
    action, action_vals = mcts_agent.search()
    obs, reward, done, truncated, info = ns_env.step(action)
    episode_reward += reward.reward

# Mostrar gráfica de la evolución de la gravedad
plt.plot(gravedad_por_step)
plt.title("Evolución de la gravedad por step")
plt.xlabel("Step")
plt.ylabel("Gravedad")
plt.grid(True)
plt.tight_layout()
plt.show()
