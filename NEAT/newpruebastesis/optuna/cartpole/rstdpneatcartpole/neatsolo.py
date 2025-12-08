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
import optuna


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



# Entorno base
def objetive(trial):
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

    input_index = [0, 1, 2, 3, 4, 5, 6, 7]
    output_index = [8, 9]




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

    base_env = gym.make("CartPole-v1")
    scheduler = CustomScheduler(event_function)
    #scheduler2 = CustomScheduler(event_function)
    update_function = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=2.0, max_val=30.0)
    #update_function2 = BoundedRandomWalk(scheduler, mu=0, sigma=10, min_val=9.0, max_val=20.0)
    tunable_params = {"force_mag": update_function}
    #tunable_params = {"force_mag": update_function, "gravity": update_function2}
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




    pruebas = 33
    from scipy import sparse
    matrix = sparse.csr_matrix(matrix)

    print("Pesos de INICIO: ", matrix)




    retornos2 = []
    gravedades2 = []
    total_return2 = []

    gravedad_og = base_env.gravity
    fuerza_og = base_env.force_mag
    largo_og = base_env.length
    masa_og = base_env.masspole

    gravedad_og = base_env.gravity
    fuerza_og = base_env.force_mag
    largo_og = base_env.length
    masa_og = base_env.masspole
            
            
    episodes = 200
    change_episode = 10


    for prueba in range(pruebas):
        #convertir en sparse matrix

        pop = Population(max_node+1, IZHIKEVICH)

        #syn = Projection(pop, pop, target='exc')
        tau_c = trial.suggest_float('tau_c', 10, 30)
        a_plus = trial.suggest_float('a_plus', 0.001, 0.09)
        a_minus = trial.suggest_float('a_minus', 0.001, 0.09)
        tau_plus = trial.suggest_float('tau_plus', 10, 30)
        tau_minus = trial.suggest_float('tau_minus', 10, 30)



        syn = Projection(pop, pop, target='exc', synapse=R_STDP(tau_c, a_plus, a_minus, tau_plus, tau_minus))


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
        i = 0
        acciones = []
        total_return = 0.0


        base_env.gravity = gravedad_og
        base_env.length = largo_og
        base_env.masspole = masa_og
        base_env.force_mag = fuerza_og


        for ep in range(episodes//2):
            observation, _ = base_env.reset()
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
                for val in observation:
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
                distance = - abs(observation[2])
                distancias.append(distance)
                r = distance - np.mean(distancias)
                #syn.reward = r
                rs.append(r)
                simulate(50.0)
                #print("Recompensa: ", syn.reward)
                weights = syn.w[0]
                #print("Pesos de salida: ", weights)
                spikes = M.get('spike')
                #Output from 2vneurons, one for each action
                output1 = np.size(spikes[output_index[0]])
                output2 = np.size(spikes[output_index[1]])
                #print("Output1: ", output1)
                #print("Output2: ", output2)
                #print("-------")

                #graficar actividad de las neuronas
                t, n = M.raster_plot(spikes)
                #plt.plot(t, n, 'b.')
                #plt.title('Raster plot')
                #plt.show()
                #Choose the action with the most spikes
                action = base_env.action_space.sample()
                if output1 > output2:
                    action = 0
                    acciones2.append(0)
                    #print("Accion 0")
                elif output2 > output1:
                    action = 1
                    acciones2.append(1)
                    #print("Accion 1")
                else:
                    acciones2.append(2)
                    #print("Accion Aleatoria")
                observation, reward, terminated, truncated, info = base_env.step(action)
                episode_return += reward
                #La recomensa será la distancia de la vara desde el punto optimo cuando esta a 90ª
                # reward = 90º - abs(observation[2])
                #print(np.rad2deg(observation[2]))
                #print(observation[2])

                M.reset()
                pop.reset()
            acciones.append(acciones2)

            #print("Episode %d reward: %f" % (ep + 1, episode_return))
            retornos.append(episode_return)
            #print("Recompensas: ", distancias)
            #print("Acciones: ", rs)
            total_return += episode_return
            simulate(50.0)
            M.reset()
            pop.reset()
            syn.reset(synapses=True)

        for ep in range(episodes//2):
            observation, _ = env.reset()
            if gravedades != []:
                base_env.gravity = gravedades[-1]
                env.unwrapped.gravity = gravedades[-1]
            if fuerzas != []:
                env.unwrapped.force_mag = fuerzas[-1]
                base_env.force_mag = fuerzas[-1]
            if ep % change_episode == 0:
                change_state["active"] = True
                change_state["remaining"] = change_episode
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
                distance = - abs(observation.state[2])
                distancias.append(distance)
                r = distance - np.mean(distancias)
                #syn.reward = r
                rs.append(r)
                simulate(50.0)
                #print("Recompensa: ", syn.reward)
                weights = syn.w[0]
                #print("Pesos de salida: ", weights)
                spikes = M.get('spike')
                #Output from 2vneurons, one for each action
                output1 = np.size(spikes[output_index[0]])
                output2 = np.size(spikes[output_index[1]])
                #print("Output1: ", output1)
                #print("Output2: ", output2)
                #print("-------")

                #graficar actividad de las neuronas
                t, n = M.raster_plot(spikes)
                #plt.plot(t, n, 'b.')
                #plt.title('Raster plot')
                #plt.show()
                #Choose the action with the most spikes
                action = env.action_space.sample()
                if output1 > output2:
                    action = 0
                    acciones2.append(0)
                    #print("Accion 0")
                elif output2 > output1:
                    action = 1
                    acciones2.append(1)
                    #print("Accion 1")
                else:
                    acciones2.append(2)
                    #print("Accion Aleatoria")
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward.reward
                #La recomensa será la distancia de la vara desde el punto optimo cuando esta a 90ª
                # reward = 90º - abs(observation[2])
                #print(np.rad2deg(observation[2]))
                #print(observation[2])







                M.reset()
                pop.reset()
            acciones.append(acciones2)
            gravedades.append(env.unwrapped.gravity)
            fuerzas.append(env.unwrapped.force_mag)
            #print("Episode %d reward: %f" % (ep + 1, episode_return))
            retornos.append(episode_return)
            #print("Recompensas: ", distancias)
            #print("Acciones: ", rs)
            total_return += episode_return
            simulate(50.0)
            M.reset()
            pop.reset()
            syn.reset(synapses=True)

        retornos2.append(retornos[episodes//2:])
        print(retornos2)
        print("fuerzas: ", fuerzas)
        clear()
    env.close()
    
    promedios_trials = np.mean(retornos2, axis=1)

    return np.mean(promedios_trials)


study_name = "cartpole-rstdp"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    load_if_exists=True
)

study.optimize(objetive, n_trials=100)
print("Mejores parámetros encontrados:", study.best_params)
print("Mejor valor:", study.best_value)
