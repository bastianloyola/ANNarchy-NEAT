from ANNarchy import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import PeriodicScheduler
from ns_gym.update_functions import IncrementUpdate

class R_STDP(Synapse):
    _instantiated = []

    def __init__(self, tau_c=20.0, a=0.1,
                 A_plus=0.01, A_minus=0.01,
                 tau_plus=20.0, tau_minus=20.0,
                 w_min=0.0, w_max=1.0):

        parameters = """
            tau_c = %(tau_c)s : projection
            a = %(a)s : projection
            A_plus = %(A_plus)s : projection
            A_minus = %(A_minus)s : projection
            tau_plus = %(tau_plus)s : projection
            tau_minus = %(tau_minus)s : projection
            w_min = %(w_min)s : projection
            w_max = %(w_max)s : projection
            reward = 0.0 : projection
        """ % locals()

        equations = """
            tau_c * dc/dt = -c : event-driven
            tau_plus  * dx/dt = -x : event-driven
            tau_minus * dy/dt = -y : event-driven
        """

        pre_spike = """
            g_target += w
            x += A_plus
            c += y
            w = w + clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a)
        """

        post_spike = """
            y -= A_minus
            c += x
            w = w + clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a)
        """

        Synapse.__init__(self,
                         parameters=parameters,
                         equations=equations,
                         pre_spike=pre_spike,
                         post_spike=post_spike,
                         name="R-STDP")

        self._instantiated.append(True)

LIF = Neuron(
    parameters="""
        tau = 50.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        tau * dv/dt = -v + g_exc - g_inh + (I - 65) : init=0
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
    """,
    spike="v >= -40.0",
    reset="v = -65"
)

IZHIKEVICH = Neuron(
    parameters="""
        a = 0.02 : population
        b = 0.2 : population
        c = -65.0 : population
        d = 8.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh : init=-65
        du/dt = a*(b*v - u) : init=-14.0
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)

pop = Population(10, IZHIKEVICH)
syn = Projection(pop, pop, target='exc')

input_index = [0, 1, 2, 3, 4, 5, 6, 7]
output_index = [8, 9]

matrix = np.zeros((10, 10))
for i in range(8):
    for j in range(2):
        np.random.seed(i + j)
        matrix[i][j + 8] = np.random.uniform(0, 110)

from scipy import sparse
matrix = sparse.csr_matrix(matrix)
syn.connect_from_sparse(matrix)

compile(directory="cartpole-ns")

weights = syn.w[0]
print("Pesos de INICIO: ", weights)

# Crear entorno no estacionario
base_env = gym.make("CartPole-v1")
scheduler = PeriodicScheduler(period=3)
update_function = IncrementUpdate(scheduler, k=0.1)
tunable_params = {"gravity": update_function}
env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True)

limites = [
    (-4.8, 4.8),
    (-10.0, 10.0),
    (-0.418, 0.418),
    (-10.0, 10.0)
]

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

syn.reset(synapses=True)

np.random.seed(10)
inputWeights = np.random.uniform(0, 150, 4)
M = Monitor(pop, ['spike', 'v'])
gravedades = []
retornos = []
acciones = []
episodes = 500
total_return = 0.0

for ep in range(episodes):
    observation, _ = env.reset()
    terminated = False
    truncated = False
    episode_return = 0.0
    acciones2 = []

    while not terminated and not truncated:
        i = 0
        k = 0
        for val in observation.state:
            val = normalize(val, limites[k][0], limites[k][1])
            if val < 0.5:
                pop[input_index[i]].I = val * 20
                pop[input_index[i + 1]].I = 0
            else:
                pop[input_index[i]].I = 0
                pop[input_index[i + 1]].I = val * 20
            i += 2
            k += 1

        gravedades.append(env.unwrapped.gravity)
        simulate(50.0)

        spikes = M.get('spike')
        output1 = np.size(spikes[output_index[0]])
        output2 = np.size(spikes[output_index[1]])

        action = env.action_space.sample()
        if output1 > output2:
            action = 0
        elif output2 > output1:
            action = 1
        else:
            action = env.action_space.sample()

        acciones2.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward.reward if hasattr(reward, 'reward') else reward

        M.reset()
        pop.reset()
        syn.reset(synapses=True)

    acciones.append(acciones2)
    print("Episode %d reward: %f" % (ep + 1, episode_return))
    retornos.append(episode_return)
    total_return += episode_return
    simulate(50.0)
    M.reset()
    pop.reset()
    syn.reset(synapses=True)

env.close()
print("Total reward across episodes:", total_return / episodes)

accion1 = sum(a == 0 for ep in acciones for a in ep)
accion2 = sum(a == 1 for ep in acciones for a in ep)
accion3 = sum(a == 2 for ep in acciones for a in ep)

acciones_labels = ['Accion 1', 'Accion 2', 'Accion 3']
valores = [accion1, accion2, accion3]

plt.bar(acciones_labels, valores)
plt.xlabel('Acciones')
plt.ylabel('Frecuencia')
plt.title('Frecuencia de acciones en episodios')
plt.show()

plt.figure()
plt.plot(gravedades)
plt.xlabel("Paso")
plt.ylabel("Valor de gravedad")
plt.title("Evolución del parámetro 'gravity'")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(retornos)
plt.xlabel("Episodio")
plt.ylabel("Retorno")
plt.title("Retorno por episodio")
plt.grid(True)
plt.show()