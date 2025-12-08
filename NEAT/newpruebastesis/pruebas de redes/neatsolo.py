from ANNarchy import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler, CustomScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from ns_gym import base
import ns_gym.utils as utils
from typing import Union, Any, Optional,Type


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
            w = w + (c * reward)
        """

        post_spike = """
            y -= A_minus
            c += x
            w = w + (c * reward)
        """

        Synapse.__init__(self,
                         parameters=parameters,
                         equations=equations,
                         pre_spike=pre_spike,
                         post_spike=post_spike,
                         name="R-STDP")

        self._instantiated.append(True)


import random as rd

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


#1;9;43.5712
#1;10;42.4252
#2;9;-15.8397
#2;10;0.221279
#3;9;49.8525
#3;10;-15.7113
#4;9;55.313
#4;10;45.7265
#5;9;-4.33209
#5;10;3.1428
#6;9;-7.02096
#6;10;67.7384
#7;9;53.6828
#7;10;-20.626
#8;9;29.6074
#8;10;73.9112
#7;3;-45.3084

file_matrix = open("best0.txt", "r")
lines = file_matrix.readlines()
# Convertir la lista de listas en un array de numpy

max_node = 0
for line in lines:
    con = line.split(";")
    in_node = int(con[0]) -1
    out_node = int(con[1]) -1
    weight = float(con[2])
    if in_node > max_node:
        max_node = in_node
    if out_node > max_node:
        max_node = out_node
# Crear una matriz de ceros de tamaño max_node x max_node
matrix = np.zeros((max_node+1, max_node+1))
# Llenar la matriz con los pesos
for line in lines:
    con = line.split(";")
    in_node = int(con[0]) -1
    out_node = int(con[1]) -1
    weight = float(con[2])
    matrix[in_node][out_node] = weight

input_index = [0, 1, 2, 3]
output_index = [4, 5, 6]




print("Pesos de INICIO: ", matrix)


change_state = {
    "active": False,
    "remaining": 0
}

def event_function(t):
    # Si está activo, consumir duración
    if change_state["active"]:
        if change_state["remaining"] > 0:
            change_state["remaining"] -= 1
            return True
        else:
            change_state["active"] = False
            return False
    return False



# Entorno base
base_env = gym.make("MountainCar")
scheduler = CustomScheduler(event_function)
#scheduler2 = PeriodicScheduler(period=500)
update_function = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=0.00025, max_val=0.0010)
update_function2 = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=0.003, max_val=0.01)
tunable_params = {"force": update_function, "gravity": update_function2}
env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True)


i = 0
limites = [
        (-4.8, 4.8),  
        (-10.0, 10.0),  
        (-0.418, 0.418), 
        (-10.0, 10.0)  
    ]

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)




from scipy import sparse

matrix = sparse.csr_matrix(matrix)




    #convertir en sparse matrix
#pop = Population(max_node+1, IZHIKEVICH)
pop = Population(max_node+1, IZHIKEVICH)
#parameters: {'tau_c': 20.576914068937768, 'a_plus': 0.06598918612549612, 'a_minus': 0.04050097614812863, 'tau_plus': 10.12782991797178, 'tau_minus': 11.603141475053995}.
tau_c = 20.576914068937768
a_plus = 0.06598918612549612
a_minus = 0.04050097614812863
tau_plus = 10.12782991797178
tau_minus = 11.603141475053995
syn = Projection(pre=pop, post=pop, target='exc', synapse=R_STDP(tau_c=tau_c, A_plus=a_plus, A_minus=a_minus, tau_minus=tau_minus, tau_plus=tau_plus))
    
syn.connect_from_sparse(matrix)
compile(directory="neat-og")
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
fuerzas = []
retornos = []
M = Monitor(pop, ['spike','v'])


pop[16].I += 20
syn.reward = -50
simulate(1000.0)

spikes = M.get('spike')
t, n = M.raster_plot(spikes)
plt.plot(t, n, 'b.')
plt.title('Raster plot')
print(spikes)
plt.show()


#get weights

weights = syn.connectivity_matrix()

#numpy array to list


wheightssparse = sparse.csr_matrix(weights.T)

print("Pesos INICIALES: ", matrix)
print("Pesos FINALES: ", wheightssparse)