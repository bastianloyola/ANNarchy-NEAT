from ANNarchy import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from ns_gym import base
import ns_gym.utils as utils
from typing import Union, Any, Optional,Type
import random as rd

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



class R_STDP(Synapse):
    """
    R-STDP con trazas pre y post, y modulación por recompensa.
    La actualización del peso depende de la coincidencia temporal (STDP) y el refuerzo externo.
    """

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
            w += ite(
                (c < 0.0) and (reward < 0.0),
                clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a),     
                abs(clip(a * c * reward, -abs(w)*a, abs(w)*a)))
        """

        post_spike = """
            y -= A_minus
            c += x
            w += ite(
                (c < 0.0) and (reward < 0.0),
                clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a),     
                abs(clip(a * c * reward, -abs(w)*a, abs(w)*a)))
        """

        Synapse.__init__(self,
                         parameters=parameters,
                         equations=equations,
                         pre_spike=pre_spike,
                         post_spike=post_spike,
                         name="R-STDP")

        self._instantiated.append(True)



class BoundedRandomWalk(base.UpdateFn):
    def __init__(self, scheduler: Type[base.Scheduler], mu: float = 0, sigma: float = 1,
                 min_val: Optional[float] = None, max_val: Optional[float] = None, seed=None):
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.min_val = min_val
        self.max_val = max_val
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, param: float, t: float) -> tuple[float, bool]:
        return super().__call__(param, t)

    def update(self, param: float, t: float) -> float:
        #get random value between min val and max val
        updated_param = rd.uniform(self.min_val,self.max_val)
        while updated_param == param:
            updated_param = rd.uniform(self.min_val,self.max_val)
        return updated_param



base_env = gym.make("Acrobot-v1")

scheduler = PeriodicScheduler(period=5)
scheduler2 = PeriodicScheduler(period=5)
scheduler3 = PeriodicScheduler(period=5)
update1 = BoundedRandomWalk(scheduler, min_val=0.8, max_val=2.0)
update2 = BoundedRandomWalk(scheduler2, min_val=0.8, max_val=3.0)
update3 = BoundedRandomWalk(scheduler3, min_val=0.2, max_val=0.8)



tunable_params = {
    "LINK_LENGTH_1": update1,
    "LINK_LENGTH_2": update1,
    "LINK_MASS_1": update2,
    "LINK_MASS_2": update2,
    "LINK_COM_POS_1": update3,
    "LINK_COM_POS_2": update3,
}

env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True, render_mode="human")

obs, info = env.reset()
done = False
truncated = False
total_reward = 0



#matriz de conexion de los 8 de enntradas a los 2 de salida
input_index = [0, 1, 2, 3, 4, 5, 6, 7]
output_index = [8, 9]

#matriz de conexion de los 12 de enntradas a los 3 de salida
input_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
output_index = [12, 13, 14]

#crear matrix 15x15
matrix = np.zeros((15, 15))
#conectar los 12 de entrada con los 3 de salida
for i in range(12):
    for j in range(3):
        #random positivo o negativo
        #random entre 0 y 1
        np.random.seed(i+j)
        matrix[i][j+8] = np.random.uniform(0,110)

#print(matrix)




print("Pesos de INICIO: ", matrix)

limites = [
    (-1, 1),  # cos(theta1)
    (-1, 1),  # sin(theta1)
    (-1, 1),  # cos(theta2)
    (-1, 1),  # sin(theta2)
    (-12.5663706, 12.5663706),  # theta1_dot
    (-28.2743339, 28.2743339)  # theta2_dot
]

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)



# Entorno base
i = 0


trials = 1
from scipy import sparse
matrix = sparse.csr_matrix(matrix)



retornos2 = []
gravedades2 = []
total_return2 = []
for trial in range(trials):
    #convertir en sparse matrix

    pop = Population(15, IZHIKEVICH)

    #syn = Projection(pop, pop, target='exc')
    syn = Projection(pop, pop, target='exc', synapse=R_STDP(A_plus=0.01, A_minus=0.01, tau_plus=20.0, tau_minus=20.0, tau_c=20.0))


    syn.connect_from_sparse(matrix)


    compile(directory="ns-cartpole-rstdp-33")
    j = 0
    returns = []
    actions_done = []
    terminated = False
    truncated = False
    observation = env.reset()[0].state
    #print(observation)

    #print("Pesos de entrada: ", inputWeights)
    #print(observation)

    gravedades = []
    retornos = []


    M = Monitor(pop, ['spike','v'])
    i = 0
    acciones = []
    episodes = 500
    total_return = 0.0
    for ep in range(episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0
        #print("Comienzo del episodio %d" % (ep + 1))
        #for i, dend in enumerate(syn.dendrites):
         #   print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")
        #Prediccion de recompensa en base a la media movil de la recompensa
        distancias = []
        acciones2  = []
        rs = []
        while not terminated and not truncated:
            # Codificar observación
            i = 0
            k = 0
            for val in observation.state:
                if val < 0:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = val*30 #inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    #Normalizar val
                    val = normalize(val, limites[k][0], limites[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*30 #inputWeights[k]
                i += 2
                k += 1
            theta1 = np.arccos(observation.state[0])
            theta2 = np.arccos(observation.state[2])
            reward = 1 - (-np.cos(theta1) - np.cos(theta2 + theta1))

            returns.append(reward)
            r = reward - np.mean(returns)
            syn.reward = r
            rs.append(r)
            simulate(50.0)
            #print("Recompensa: ", syn.reward)
            weights = syn.w[0]
            #print("Pesos de salida: ", weights)
            spikes = M.get('spike')
            #Output from 2vneurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            action = env.action_space.sample()
            if output1 > output2 and output1 > output3:
                action = 0
            elif output2 > output1 and output2 > output3:
                action = 1
            elif output3 > output1 and output3 > output2:
                action = 2
            #print("Output1: ", output1)
            #print("Output2: ", output2)
            #print("-------")

            #graficar actividad de las neuronas
            t, n = M.raster_plot(spikes)
            #plt.plot(t, n, 'b.')
            #plt.title('Raster plot')
            #plt.show()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward.reward
            #La recomensa será la distancia de la vara desde el punto optimo cuando esta a 90ª
            # reward = 90º - abs(observation[2])
            #print(np.rad2deg(observation[2]))
            #print(observation[2])







            M.reset()
            pop.reset()
            syn.reward = 0.0
        acciones.append(acciones2)

        #print("Episode %d reward: %f" % (ep + 1, episode_return))
        retornos.append(episode_return)
        #print("Recompensas: ", distancias)
        #print("Acciones: ", rs)
        total_return += episode_return
        simulate(50.0)
        M.reset()
        pop.reset()
        #syn.reset(synapses=True)
    retornos2.append(retornos)
    total_return2.append(total_return / episodes)
    print(total_return2)
    clear()
env.close()




# Convertir la lista de listas en un array de numpy
retornos2 = np.array(retornos2)
nombre = "force-rstdp-hidden1"
np.save(f"retornos2_{nombre}.npy", retornos2)

promedios_trials = np.mean(retornos2, axis=1)

plt.figure(figsize=(8, 5))
plt.boxplot(promedios_trials, vert=True, patch_artist=True)
plt.title(f'Distribución de la recompensa promedio por trial - {nombre}')
plt.ylabel('Recompensa promedio')
plt.xticks([1], [nombre])
plt.tight_layout()
plt.savefig(f'boxplot_{nombre}.png')
plt.close()

mean = np.mean(retornos2, axis=0)
q25 = np.percentile(retornos2, 25, axis=0)
q75 = np.percentile(retornos2, 75, axis=0)
median = np.median(retornos2, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(mean, label='Media', color='blue')
plt.fill_between(range(len(mean)), q25, q75, color='blue', alpha=0.2, label='IQR')
plt.plot(median, label='Mediana', color='red')
plt.title(f'Convergencia de la recompensa media por episodio - {nombre}')
plt.xlabel('Episodio')
plt.ylabel('Recompensa media')
plt.legend()
plt.tight_layout()
plt.savefig(f'convergencia_{nombre}.png')
plt.close()