from ANNarchy import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate


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

#syn = Projection(pop, pop, target='exc')
syn = Projection(pop, pop, target='exc', synapse=R_STDP)

#matriz de conexion de los 8 de enntradas a los 2 de salida
input_index = [0, 1, 2, 3, 4, 5, 6, 7]
output_index = [8, 9]

#crear matrix 15x15
matrix = np.zeros((10, 10))
#conectar los 12 de entrada con los 3 de salida
for i in range(8):
    for j in range(2):
        #random positivo o negativo
        #random entre 0 y 1
        np.random.seed(i+j)
        matrix[i][j+8] = np.random.uniform(0,110)

#print(matrix)

#convertir en sparse matrix
from scipy import sparse
matrix = sparse.csr_matrix(matrix)

syn.connect_from_sparse(matrix)


compile(directory="ns-cartpole-rstdp-LTDFIX")

weights = syn.w[0]
print("Pesos de INICIO: ", weights)





# Entorno base
base_env = gym.make("CartPole-v1")
scheduler = PeriodicScheduler(period=3)
update_function = IncrementUpdate(scheduler, k=0.1)
tunable_params = {"gravity": update_function}
env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True)

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

#dends = list(syn.dendrites)
#print(dends[0])
#for i, dend in enumerate(syn.dendrites):
#    print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")
syn.reset(synapses=True)
#dends = list(syn.dendrites)
#for i, dend in enumerate(syn.dendrites):
#    print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")#

j = 0
returns = []
actions_done = []
terminated = False
truncated = False
observation = env.reset()[0].state
print(observation)
np.random.seed(10)
inputWeights = np.random.uniform(0,150,4)
#print("Pesos de entrada: ", inputWeights)
#print(observation)

gravedades = []
retornos = []


M = Monitor(pop, ['spike','v'])
num_steps = 1000
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
                pop[int(input_index[i])].I = val*20 #inputWeights[k]
                pop[int(input_index[i+1])].I = 0
            else:
                #Normalizar val
                val = normalize(val, limites[k][0], limites[k][1])
                pop[int(input_index[i])].I = 0
                pop[int(input_index[i+1])].I = val*20 #inputWeights[k]
            i += 2
            k += 1
        distance = - abs(observation.state[2])
        distancias.append(distance)
        r = distance - np.mean(distancias)
        syn.reward = r
        rs.append(r)
        gravedades.append(env.unwrapped.gravity)
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
        syn.reward = 0.0
    acciones.append(acciones2)

    print("Episode %d reward: %f" % (ep + 1, episode_return))
    retornos.append(episode_return)
    #print("Recompensas: ", distancias)
    #print("Acciones: ", rs)
    total_return += episode_return
    dends = list(syn.dendrites)
    #print(dends[0])
    #for i, dend in enumerate(syn.dendrites):
     #   print(f"Dendrita {i}: conecta con pre = {dend.pre}, pesos = {dend.w}")
    simulate(50.0)
    M.reset()
    pop.reset()
    #syn.reset(synapses=True)
env.close()
print("Total reward across episodes:", total_return/episodes)


#grafico de barras de las acciones, segun las màs frecuentes en todos los episodios



# Convertir la lista de acciones en un array de numpy

accion1 = 0
accion2 = 0
accion3 = 0

for i in range(episodes):
    for j in range(len(acciones[i])):
        if acciones[i][j] == 0:
            accion1 += 1
        elif acciones[i][j] == 1:
            accion2 += 1
        elif acciones[i][j] == 2:
            accion3 += 1
# Crear una lista con los nombres de las acciones
acciones = ['Accion 1', 'Accion 2', 'Accion 3']
# Crear una lista con los valores de las acciones
valores = [accion1, accion2, accion3]
# Crear el gráfico de barras
plt.bar(acciones, valores)
# Añadir etiquetas y título
plt.xlabel('Acciones')
plt.ylabel('Frecuencia')
plt.title('Frecuencia de acciones en episodios')
# Mostrar el gráfico
plt.show()


#print(valores)
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
