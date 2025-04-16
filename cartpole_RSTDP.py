from ANNarchy import *

class R_STDP(Synapse):
    """
    R-STDP con trazas pre y post, y modulación por recompensa.
    La actualización del peso depende de la coincidencia temporal (STDP) y el refuerzo externo.
    """

    _instantiated = []

    def __init__(self, tau_c=100.0, a=0.1,
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
            c -= y
            w += clip(a * c * reward, -w*a, w*a)
        """

        post_spike = """
            y -= A_minus
            c += x
            w += clip(a * c * reward, -w*a, w*a)
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


syn = Projection(pop, pop, target='exc', synapse=R_STDP())

#matriz de conexion de los 8 de enntradas a los 2 de salida
input_index = [0, 1, 2, 3, 4, 5, 6, 7]
output_index = [8, 9]

#crear matrix 15x15
matrix = np.zeros((10, 10))
#conectar los 12 de entrada con los 3 de salida
for i in range(8):
    for j in range(2):
        matrix[i][j+8] = np.random.uniform(-20, 80)

#convertir en sparse matrix
from scipy import sparse
matrix = sparse.csr_matrix(matrix)

syn.connect_from_sparse(matrix)


compile()

import gymnasium as gym

#cartpole
env = gym.make('CartPole-v1')
i = 0
limites = [
        (-4.8, 4.8),  
        (-10.0, 10.0),  
        (-0.418, 0.418), 
        (-10.0, 10.0)  
    ]

simulate(1)
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

dends = list(syn.dendrites)
print(dends[0])
for i, dend in enumerate(syn.dendrites):
    print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")
syn.reset()
dends = list(syn.dendrites)
for i, dend in enumerate(syn.dendrites):
    print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")

j = 0
returns = []
actions_done = []
terminated = False
truncated = False
observation = env.reset()[0]
np.random.seed(4)
inputWeights = np.random.uniform(0,150,4)
print(observation)

M = Monitor(pop, ['spike','v'])
num_steps = 1000
i = 0

episodes = 2
total_return = 0.0
for ep in range(episodes):
    observation, _ = env.reset()
    terminated = False
    truncated = False
    episode_return = 0.0
    print("Comienzo del episodio %d" % (ep + 1))
    for i, dend in enumerate(syn.dendrites):
        print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")
    #Prediccion de recompensa en base a la media movil de la recompensa
    recompensas = []
    while not terminated:
        # Codificar observación
        i = 0
        k = 0
        for val in observation:
            if val < 0:
                #Normalizar val
                val = normalize(val, limites[k][0], limites[k][1])
                pop[int(input_index[i])].I = -val*inputWeights[k]
                pop[int(input_index[i+1])].I = 0
            else:
                #Normalizar val
                val = normalize(val, limites[k][0], limites[k][1])
                pop[int(input_index[i])].I = 0
                pop[int(input_index[i+1])].I = val*inputWeights[k]
            i += 2
            k += 1

        simulate(50.0)
        spikes = M.get('spike')
        #Output from 2vneurons, one for each action
        output1 = np.size(spikes[output_index[0]])
        output2 = np.size(spikes[output_index[1]])
        #Choose the action with the most spikes
        action = env.action_space.sample()
        if output1 > output2:
            action = 0
        elif output2 > output1:
            action = 1

        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        #La recomensa será la distancia de la vara desde el punto optimo cuando esta a 90ª
        # reward = 90º - abs(observation[2])
        reward = 1 - abs(np.rad2deg(observation[2]))
        recompensas.append(reward)
        r = reward - np.mean(recompensas)
        syn.reward = r
        simulate(50.0)
        M.rebset()

    print("Episode %d reward: %f" % (ep + 1, episode_return))
    total_return += episode_return
    dends = list(syn.dendrites)
    print(dends[0])
    for i, dend in enumerate(syn.dendrites):
        print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")

    simulate(50.0)
    M.reset()
    pop.reset()
    syn.reset(synapses=True)
env.close()
print("Total reward across episodes:", total_return/episodes)


