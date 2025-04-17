from ANNarchy import *
import numpy as np
import gymnasium as gym
from scipy import sparse

# --- Sinapsis R-STDP ---
class R_STDP(Synapse):
    _instantiated = []

    def __init__(self, tau_c=100.0, a=0.005,
                 A_plus=1.0, A_minus=1.0,
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
            w = clip(w + a * c * reward, w_min, w_max)
        """

        post_spike = """
            y -= A_minus
            c += x
            w = clip(w + a * c * reward, w_min, w_max)
        """

        Synapse.__init__(self,
                         parameters=parameters,
                         equations=equations,
                         pre_spike=pre_spike,
                         post_spike=post_spike,
                         name="R-STDP")
        self._instantiated.append(True)

# --- Neurona IZHIKEVICH ---
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

# --- Preparar red neuronal ---
pop = Population(20, IZHIKEVICH)  # 8 obs * 2 + 4 salidas = 20 neuronas

input_index = list(range(16))      # 8 observaciones → 16 neuronas codificadas
output_index = [16, 17, 18, 19]    # 4 salidas posibles

# Matriz de conexión 16 -> 4
matrix = np.zeros((20, 20))
for i in range(16):
    for j in range(4):
        matrix[i][j + 16] = np.random.uniform(-20, 80)
matrix = sparse.csr_matrix(matrix)

syn = Projection(pop, pop, target='exc', synapse=R_STDP)
syn.connect_from_sparse(matrix)


compile()

# --- Configurar entorno LunarLander ---
env = gym.make('LunarLander-v2', render_mode='human')
observation = env.reset()[0]
terminated = False
truncated = False

# Límites para normalizar
limites = [
    (-1.5, 1.5),      # x position
    (-.5, 1.5),       # y position
    (-2.0, 2.0),      # x velocity
    (-2.0, 2.0),      # y velocity
    (-3.14, 3.14),    # angle
    (-4.0, 4.0),      # angular velocity
    (0.0, 1.0),       # leg 1 contact
    (0.0, 1.0),       # leg 2 contact
]

# Función para normalizar
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Pesos de entrada aleatorios
np.random.seed(4)
inputWeights = np.random.uniform(0, 150, 8)

# --- Simulación ---
returns = []
M = Monitor(pop, ['spike', 'v'])
episodes = 50
total_return = 0.0

for ep in range(episodes):
    observation, _ = env.reset()
    terminated = False
    truncated = False
    episode_return = 0.0

    while not terminated and not truncated:
        # Codificar observaciones
        i = 0
        k = 0
        for val in observation:
            val = normalize(val, limites[k][0], limites[k][1])
            if val < 0.5:
                pop[input_index[i]].I = (0.5 - val) * inputWeights[k]
                pop[input_index[i + 1]].I = 0
            else:
                pop[input_index[i]].I = 0
                pop[input_index[i + 1]].I = (val - 0.5) * inputWeights[k]
            i += 2
            k += 1

        simulate(100.0)
        spikes = M.get('spike')
        out_spikes = [np.size(spikes[i]) for i in output_index]
        action = np.argmax(out_spikes)

        observation, reward, terminated, truncated, info = env.step(action)
        syn.reward = reward
        simulate(100.0)
        episode_return += reward
        M.reset()

    print(f"Episode {ep + 1} reward: {episode_return}")
    total_return += episode_return
    simulate(100.0)
    M,reset()
    reset(pop)

env.close()
print("Total reward across episodes:", total_return/episodes)


w_after = syn.w.copy()
delta_w = w_after - w_before
print("Weight changes:")
print(delta_w)
print("Mean weight change:", np.mean(delta_w))
