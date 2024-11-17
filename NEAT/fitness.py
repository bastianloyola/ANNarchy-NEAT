
from ANNarchy import *
import gymnasium as gym
import numpy as np


def cartpole(pop,Monitor,input_index,output_index,inputWeights):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    max_steps = 500
    terminated = False
    truncated = False
    maxInput = inputWeights[1]
    minInput = inputWeights[0]
    #Generar 4 input weights para cada input
    inputWeights = np.random.uniform(minInput,maxInput,4)
    #Number of episodes
    episodes = 100
    h=0
    #Final fitness 
    final_fitness = 0
    while h < episodes:
        j=0
        returns = []
        actions_done = []
        terminated = False
        truncated = False
        while j < max_steps and not terminated and not truncated:
            #encode observation, 4 values split in 8 neurons (2 for each value), if value is negative the left neuron is activated, if positive the right neuron is activated
            i = 0
            k = 0
            for val in observation:
                if val < 0:
                    pop[int(input_index[i])].I = -val*inputWeights[k]
                    pop[int(input_index[i+1])].I = 0
                else:
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
            if output1 > output2:
                action = 0
            elif output1 < output2:
                action = 1
            observation, reward, terminated, truncated, info = env.step(action)
            returns.append(reward)
            actions_done.append(action)
            pop.reset()
            Monitor.reset()
            j += 1
        env.reset()
        #print("Episode: ",h," Fitness: ",np.sum(returns))
        final_fitness += np.sum(returns)
        h += 1
    final_fitness = final_fitness/episodes
    #print("Final mean fitness: ",final_fitness,"\n")
    env.close()
    #print("Returns: ",returns)
    #print("Actions: ",actions_done)
    return final_fitness

import random as rd
from scipy.special import erf

def cartpoleB(pop, Monitor, input_index, output_index, inputWeights):
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
