from ANNarchy import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy.sparse
import gymnasium as gym
from scipy.special import erf

def get_function(trial):
    #open config file and get the parameter "function"
    with open('results/trial-' + str(int(trial)) + '/config.txt') as f:
        lines = f.readlines()
        for line in lines:
            if "function" in line:
                return line.split('=')[1].strip()
    return None



LIF = Neuron(  #I = 75
    parameters = """
    tau = 50.0 : population
    I = 0.0
    tau_I = 10.0 : population
    """,
    equations = """
    tau * dv/dt = -v + g_exc - g_inh + (I-65) : init=0
    tau_I * dg_exc/dt = -g_exc
    tau_I * dg_inh/dt = -g_inh
    """,
    spike = "v >= -40.0",
    reset = "v = -65"
)

IZHIKEVICH = Neuron(  #I = 20
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
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
        du/dt = a*(b*v - u) : init=-14.0
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)
I = 0
def snn(n_entrada, n_salida, n, i, matrix, inputWeights, trial, genome_id):
    try:
        I = i
        clear()
        pop = Population(geometry=n, neuron=IZHIKEVICH)
        proj = Projection(pre=pop, post=pop, target='exc')
        #Matrix to numpy array
         # Verificar el tamaño de la matrix
        if matrix.size == 0:
            raise ValueError("matrix is empty")
        #lil_matrix scipy nxn with values of matrix
        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))

        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        lil_matrix[:n_rows, :n_cols] = matrix
        proj.connect_from_sparse(lil_matrix)
        nombre = 'annarchy/annarchy-'+str(int(trial))+'/annarchy-'+str(int(i))
        compile(directory=nombre, clean=False, silent=True)
        M = Monitor(pop, ['spike','v'])
        input_index = []
        output_index = []
        n_entrada = int(n_entrada)
        n_salida = int(n_salida)
        for i in range(n_entrada):
            input_index.append(i)
        for i in range(n_entrada,n_salida+n_entrada):
            output_index.append(i)
        # Verificar el tamaño de inputWeights
        if inputWeights.size == 0:
            raise ValueError("inputWeights is empty")

        funcion = get_function('results/trial-'+ str(int(trial)))
        fit = fitness(pop,M,input_index,output_index, funcion, inputWeights, genome_id*int(trial))
        #return fit

        return fit
    except Exception as e:
        # Capturar y manejar excepciones
        print("Error en annarchy:", e)

def fitness(pop, Monitor, input_index, output_index, funcion, inputWeights, genome_id):
    if funcion == "xor":
        return xor(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "cartpole":
        return cartpole(pop, Monitor, input_index, output_index, inputWeights, genome_id)
    elif funcion == "lunar_lander":
        return lunar_lander(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "cartpole2":
        return cartpole2(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "cartpole3":
        return cartpole3(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "lunar_lander2":
        return lunar_lander2(pop, Monitor, input_index, output_index, inputWeights)
    elif funcion == "acrobot":
        return acrobot(pop, Monitor, input_index, output_index, inputWeights, genome_id)
    elif funcion == "acrobot2":
        return acrobot2(pop, Monitor, input_index, output_index, inputWeights, genome_id)
    else:
        raise ValueError(f"Unknown function: {funcion}")


def get_function(folder):
    # Open config file and get the parameter "function"
    config_path = folder + '/config.cfg'
    with open(config_path) as f:
        lines = f.readlines()
        for line in lines:
            if "function" in line:
                return line.split('=')[1].strip()
    return None
     

def xor(pop,Monitor,input_index,output_index,inputWeights):
    Monitor.reset()
    entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    fitness = 0
    for entrada in entradas:
        for i, val in zip(input_index, entrada):
            if val == 1:
                pop[int(i)].I = 15.1*inputWeights[i]
            else:
                pop[int(i)].I = 0
        simulate(10.0)
        spikes = Monitor.get('spike')
        #print("spikes: ",spikes) 
        #Get the output
        output = 0
        for i in output_index:
            output += np.size(spikes[i])
        #print("spike output: ",output)

        decode_output = 0
        if output > 1:
            decode_output = 1

        pop.reset()
        Monitor.reset()
        #comparar las entradas y la salida esperada con el output
        if entrada[0] ^ entrada[1] == decode_output:
            fitness += 1
    return fitness



def cartpole(pop,Monitor,input_index,output_index,inputWeights, genome_id):
    env = gym.make("CartPole-v1")
    observation, info = env.reset()
    terminated = False
    truncated = False
    maxInput = inputWeights[1]
    minInput = inputWeights[0]
    #Generate 4 input weights for each input
    np.random.seed(int(genome_id))
    inputWeights = np.random.uniform(minInput,maxInput,4)
    #Number of episodes
    episodes = 100
    h=0
    #Final fitness 
    final_fitness = 0

    # Limits for each observation variable
    limits = [
        (-4.8, 4.8),  # Cart position
        (-10.0, 10.0),  # Cart velocity (estimated)
        (-0.418, 0.418),  # Pole angle in radians
        (-10.0, 10.0)  # Pole angular velocity (estimated)
    ]
    
    while h < episodes:
        j=0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        env.reset()
        while not terminated and not truncated:
            #encode observation, 4 values split in 8 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = -val*inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
                    val = normalize(val, limits[k][0], limits[k][1])
                    pop[int(input_index[i])].I = 0
                    pop[int(input_index[i+1])].I = val*inputWeights[k]
                i += 2
                k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            #Output from 2 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2: #left
                action = 0
            elif output1 < output2: #right
                action = 1
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        #The fitness is the sum of the rewards for each episode
        final_fitness += np.sum(returns)
        h += 1
    #The final fitness is the mean of the fitness for each episode
    final_fitness = final_fitness/episodes
    env.close()
    return final_fitness


def cartpole2(pop, Monitor, input_index, output_index, inputWeights):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    max_steps = 1000
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 100
    h = 0
    # Final fitness 
    final_fitness = 0
    
    # Definir límites para cada variable de observación
    limites = [
        (-4.8, 4.8),  # Posición del carro
        (-10.0, 10.0),  # Velocidad del carro (estimado)
        (-0.418, 0.418),  # Ángulo del poste en radianes
        (-10.0, 10.0)  # Velocidad angular del poste (estimado)
    ]
    
    num_neuronas_por_variable = 20
    intervals = []

    for low, high in limites:
        # Generar valores centrados en 0 siguiendo una distribución normal
        values = np.random.normal(loc=0, scale=1, size=1000)
        z = np.linspace(low, high, num_neuronas_por_variable + 1)
        interval_limits = np.percentile(values, (0.5 * (1 + erf(z / np.sqrt(2)))) * 100)
        # Dividir los valores en intervalos
        intervals = [values[(values >= interval_limits[i]) & (values < interval_limits[i+1])] for i in range(num_neuronas_por_variable)]
        intervals[-1] = np.append(intervals[-1], values[-1])  # Asegurar que el último intervalo incluye el valor máximo

    
    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        while j < max_steps and not terminated and not truncated:
            # Codificar observación
            for i, obs in enumerate(observation):  # Primer ciclo: Itera sobre cada observación
                for j in range(num_neuronas_por_variable):
                    if obs >= interval_limits[j] and obs < interval_limits[j + 1]:
                        pop[input_index[i * num_neuronas_por_variable + j]].I = 20 # Activa la neurona correspondiente
                        break
            simulate(50.0)

            # Decodificar la acción basada en la cual neurona de salida tuvo la primera spike
            spikes = Monitor.get('spike')
            min_left = np.inf
            for i in output_index[:20]:
                if len(spikes[i]) > 0:
                    if min(spikes[i]) < min_left:
                        min_left = min(spikes[i])
            min_right = np.inf
            for i in output_index[20:]:
                if len(spikes[i]) > 0:
                    if min(spikes[i]) < min_right:
                        min_right = min(spikes[i])


            action = env.action_space.sample()
            if min_left < min_right:
                action = 0
            elif min_right < min_left:
                action = 1
        

            
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        env.reset()
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


def lunar_lander(pop, Monitor, input_index, output_index, inputWeights):
    #funcion similar a cartpole, solo que con el entorno de lunar lander 16 entradas y 4 salidas
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=42)
    max_steps = 1000
    terminated = False
    truncated = False
    maxInput = inputWeights[1]
    minInput = inputWeights[0]

    #Generar 8 input weights para cada input
    inputWeights = np.random.uniform(minInput,maxInput,8)
    #Number of episodes
    episodes = 31
    h=0
    #Final fitness
    final_fitness = 0

    limites = [
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-6.2831855, 6.2831855),
        (-10.0, 10.0),
        (-0.0, 1.0),
        (-0.0, 1.0)
    ]
    while h < episodes:
        j=0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        env.reset()
        while j < max_steps and not terminated and not truncated:
            #encode observation, 8 values split in 16 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
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
            spikes = Monitor.get('spike')
            #Output from 4 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            output4 = np.size(spikes[output_index[3]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2 and output1 > output3 and output1 > output4:
                action = 0
            elif output2 > output1 and output2 > output3 and output2 > output4:
                action = 1
            elif output3 > output1 and output3 > output2 and output3 > output4:
                action = 2
            elif output4 > output1 and output4 > output2 and output4 > output3:
                action = 3
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            Monitor.reset()
            j += 1
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness/episodes
    env.close()
    return final_fitness

def cartpole3(pop, Monitor, input_index, output_index, inputWeights):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    max_steps = 1000
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 100
    h = 0
    # Final fitness 
    final_fitness = 0
    
    # Definir límites para cada variable de observación
    limites = [
    (-4.8, 4.8),  # Posición del carro
    (-10.0, 10.0),  # Velocidad del carro (estimado)
    (-0.418, 0.418),  # Ángulo del poste en radianes
    (-10.0, 10.0)  # Velocidad angular del poste (estimado)
]

    num_neuronas_por_variable = 20
    std_dev = 1  # Controla cuán concentrados están los incrementos en el centro
    interval_limits = []

    for low, high in limites:
        # Crear una distribución gaussiana normalizada en el rango [-1, 1]
        x = np.linspace(-1, 1, num_neuronas_por_variable)
        gaussian_weights = np.exp(-0.5 * (x / std_dev) ** 2)
        gaussian_weights /= gaussian_weights.sum()

        increments = gaussian_weights * (high - low)

        limites_acumulados = np.concatenate([[low], low + np.cumsum(increments)])
        intervalos = [[limites_acumulados[i], limites_acumulados[i+1]] for i in range(len(limites_acumulados) - 1)]
        interval_limits.append(intervalos)
    flag=True
    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        while j < max_steps and not terminated and not truncated:
            # Codificar observación
            for i, obs in enumerate(observation):  # Primer ciclo: Itera sobre cada observación
                for j in range(num_neuronas_por_variable):
                    if obs >= interval_limits[j] and obs < interval_limits[j + 1]:
                        pop[input_index[i * num_neuronas_por_variable + j]].I = 75 # Activa la neurona correspondiente
                        break

            simulate(50.0)
            spikes = Monitor.get('spike')
            # Decodificar la acción basada en el número de picos en las neuronas de salida
            left_spikes = sum(np.size(spikes[idx]) for idx in output_index[:20])  # Neuronas que controlan el movimiento a la izquierda
            right_spikes = sum(np.size(spikes[idx]) for idx in output_index[20:])  # Neuronas que controlan el movimiento a la derecha
            
            action = env.action_space.sample()
            if left_spikes > right_spikes:
                action = 0  # Mover a la izquierda
            elif left_spikes < right_spikes:
                action = 1  # Mover a la derecha

            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            #resetear I=0, resetear a -65 (Iz valor de descanso)
            j += 1
        env.reset()
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness


#lunar_lander2 based on cartpole3
def lunar_lander2(pop, Monitor, input_index, output_index, inputWeights):
    env = gym.make("LunarLander-v2")
    observation, info = env.reset()
    terminated = False
    truncated = False
    max_steps = 1000
    # Number of episodes
    episodes = 31
    h = 0
    # Final fitness
    final_fitness = 0

    # Definir límites para cada variable de observación
    limites = [
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-6.2831855, 6.2831855),
        (-10.0, 10.0),
        (-0.0, 1.0),
        (-0.0, 1.0)
    ]

    num_neuronas_por_variable = 20
    std_dev = 1  # Controla cuán concentrados están los incrementos en el centro
    intervalos_por_variable = []

    for low, high in limites:
        # Crear una distribución gaussiana normalizada en el rango [-1, 1]
        x = np.linspace(-1, 1, num_neuronas_por_variable)
        gaussian_weights = np.exp(-0.5 * (x / std_dev) ** 2)

        # Normalizar los pesos para que sumen 1
        gaussian_weights /= gaussian_weights.sum()

        # Escalar los pesos al rango total
        increments = gaussian_weights * (high - low)

        # Calcular los límites acumulados
        limites_acumulados = np.concatenate([[low], low + np.cumsum(increments)])

        # Guardar los intervalos como listas anidadas
        intervalos = [[limites_acumulados[i], limites_acumulados[i+1]] for i in range(len(limites_acumulados) - 1)]
        intervalos_por_variable.append(intervalos)
    while h < episodes:
        l = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        while not terminated and not truncated and l < max_steps:
            # Codificar observación
            for i, intervalos in enumerate(intervalos_por_variable):
                min_val, max_val = intervalos[0]
                for j, (low, high) in enumerate(intervalos):
                    if observation[i] >= low and observation[i] < high:
                        pop[input_index[i * num_neuronas_por_variable + j]].I = 20
                    elif observation[i] > max_val:
                        pop[input_index[i * num_neuronas_por_variable + j]].I = 20
                    elif observation[i] < min_val:
                        pop[input_index[i * num_neuronas_por_variable + j]].I = 20
            simulate(50.0)
            spikes = Monitor.get('spike')
            # Decodificar la acción basada en el número de picos en las neuronas de salida
            left_spikes = sum(np.size(spikes[idx]) for idx in output_index[:20])  
            middle_spikes = sum(np.size(spikes[idx]) for idx in output_index[20:40])
            right_spikes = sum(np.size(spikes[idx]) for idx in output_index[40:])
            no_action_spikes = sum(np.size(spikes[idx]) for idx in output_index[60:]) 

            action = env.action_space.sample()
            if left_spikes > middle_spikes and left_spikes > right_spikes and left_spikes > no_action_spikes:
                action = 1
            elif middle_spikes > left_spikes and middle_spikes > right_spikes and middle_spikes > no_action_spikes:
                action = 2
            elif right_spikes > left_spikes and right_spikes > middle_spikes and right_spikes > no_action_spikes:
                action = 3
            elif no_action_spikes > left_spikes and no_action_spikes > middle_spikes and no_action_spikes > right_spikes:
                action = 0

            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            l += 1
        env.reset()
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness



def acrobot(pop, Monitor, input_index, output_index, inputWeights, genome_id):
    env = gym.make("Acrobot-v1")
    observation, info = env.reset()
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 31
    h = 0
    # Final fitness 
    final_fitness = 0

    maxInput = inputWeights[1]
    minInput = inputWeights[0]
    
    # Definir límites para cada variable de observación
    limites = [
        (-1, 1),  # cos(theta1)
        (-1, 1),  # sin(theta1)
        (-1, 1),  # cos(theta2)
        (-1, 1),  # sin(theta2)
        (-12.5663706, 12.5663706),  # theta1_dot
        (-28.2743339, 28.2743339)  # theta2_dot
    ]
    np.random.seed(int(genome_id))
    inputWeights = np.random.uniform(minInput,maxInput,6)
    while h < episodes:
        j = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        env.reset()
        while not terminated and not truncated:
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
            spikes = Monitor.get('spike')
            #Output from 3 neurons, one for each action
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            #Choose the action with the most spikes
            action = env.action_space.sample()
            if output1 > output2 and output1 > output3:
                action = 0
            elif output2 > output1 and output2 > output3:
                action = 1
            elif output3 > output1 and output3 > output2:
                action = 2
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            Monitor.reset()
            j += 1
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness


def acrobot2(pop, Monitor, input_index, output_index, inputWeights, genome_id): #based on lunar_lander2
    env = gym.make("Acrobot-v1")
    observation, info = env.reset()
    terminated = False
    truncated = False
    # Number of episodes
    episodes = 31
    h = 0
    # Final fitness
    final_fitness = 0
    
    # Definir límites para cada variable de observación
    limites = [
        (-1, 1),  # cos(theta1)
        (-1, 1),  # sin(theta1)
        (-1, 1),  # cos(theta2)
        (-1, 1),  # sin(theta2)
        (-12.5663706, 12.5663706),  # theta1_dot
        (-28.2743339, 28.2743339)  # theta2_dot
    ]

    num_neuronas_por_variable = 20
    std_dev = 1  # Controla cuán concentrados están los incrementos en el centro
    interval_limits = []
    for low, high in limites:
        # Crear una distribución gaussiana normalizada en el rango [-1, 1]
        x = np.linspace(-1, 1, num_neuronas_por_variable)
        gaussian_weights = np.exp(-0.5 * (x / std_dev) ** 2)
        gaussian_weights /= gaussian_weights.sum()

        increments = gaussian_weights * (high - low)

        limites_acumulados = np.concatenate([[low], low + np.cumsum(increments)])
        interval_limits.append(limites_acumulados)
    while h < episodes:
        l = 0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        env.reset()
        while not terminated and not truncated:
            # Codificar observación
            for i, intervalos in enumerate(interval_limits):
                min_val, max_val = intervalos[0], intervalos[-1]
                k = 0
                while k < len(intervalos) - 1:
                    if observation[i] >= intervalos[k] and observation[i] < intervalos[k+1]:
                        pop[input_index[i * num_neuronas_por_variable + k]].I = 20
                        break
                    elif observation[i] > max_val:
                        pop[input_index[i * num_neuronas_por_variable + 19]].I = 20
                    elif observation[i] < min_val:
                        pop[input_index[i * num_neuronas_por_variable + 0]].I = 20
                    k += 1
            simulate(50.0)
            spikes = Monitor.get('spike')
            # Decodificar la acción basada en el número de picos en las neuronas de salida
            output1 = sum(np.size(spikes[idx]) for idx in output_index[:20])  # Neuronas que controlan el movimiento a la izquierda
            output2 = sum(np.size(spikes[idx]) for idx in output_index[20:40])  # Neuronas que controlan el movimiento a la derecha
            output3 = sum(np.size(spikes[idx]) for idx in output_index[40:])  # Neuronas que controlan el movimiento a la derecha

            action = env.action_space.sample()
            if output1 > output2 and output1 > output3:
                action = 0
            elif output2 > output1 and output2 > output3:
                action = 1
            elif output3 > output1 and output3 > output2:
                action = 2

            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            l += 1
        final_fitness += np.sum(returns)
        h += 1

    final_fitness = final_fitness / episodes
    env.close()
    return final_fitness

