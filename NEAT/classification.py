from ANNarchy import *
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import KFold
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import random as rd
import pandas as pd

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


def rate_code(datos, max_rate=10):
    # Normalize data to the range [0, 1]
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(datos)
    encoded = datos*max_rate
    return encoded

def bootstrap_data(data_x, data_y,n_bootstrap, n_samples):
    # Lista para guardar los conjuntos de datos generados
    bootstrap_datasets = []
    # Generar los conjuntos de datos usando bootstrapping
    for i in range(n_bootstrap):
        X_bootstrap, y_bootstrap = resample(data_x, data_y, replace=True, n_samples=n_samples)
        bootstrap_datasets.append((X_bootstrap, y_bootstrap))
    return bootstrap_datasets

def estadistica(conf_matrix):   
    precision = []
    recall = []
    f1 = []
    for i in range(len(conf_matrix)):
        if sum(conf_matrix[i]) == 0:
            if sum(conf_matrix[:,i]) == 0:
                precision.append(1)
                recall.append(1)
                f1.append(1)
            else:
                precision.append(0)
        else:
            precision.append(conf_matrix[i][i]/sum(conf_matrix[i]))

        if sum(conf_matrix[:,i]) == 0:
            recall.append(0)
        else:
            recall.append(conf_matrix[i][i]/sum(conf_matrix[:,i]))
        if precision[i] + recall[i] == 0:
            f1.append(0)
        else:
            f1.append(2*precision[i]*recall[i]/(precision[i]+recall[i]))

    average_f1 = sum(f1)/len(f1)
    return precision, recall, f1, average_f1

def fitness(pop, M, n_input, n_output, flag=False):

    spike_rates = rate_code(data_x, max_rate=100)
    subsets = bootstrap_data(spike_rates, data_y,100,100)

    conf_matrix = np.zeros((n_output,n_output))
    '''    
    precision = []
    recall = []
    f1_cl = []
    f1 = []
    '''
    total = 0
    n = len(subsets)
    for i in range(n):
        x = subsets[i][0]
        y = subsets[i][1]

        sum = 0
        samples = len(x)
        for j in range(samples):
            sample = x[j]
            target = y[j]
            
            for k in range(n_input):
                pop[k].I = sample[k]

            simulate(100.0)
            if flag:
                graficos(M)
                if j == 3:
                    return 0
            spikes = M.get('spike')

            output = []
            for k in range(n_output):
                output.append(len(spikes[k]))

            max_spikes = max(output)
            indices_maximos = [i for i, x in enumerate(output) if x == max_spikes]
            
            if len(indices_maximos) == 1:
                index = indices_maximos[0]
            else:
                index = rd.choice(indices_maximos)

            if index == target:
                    sum += 1.0

            conf_matrix[index][target] += 1.0
            #print(index, target)
            #print(conf_matrix)
            pop.reset()
            M.reset()
        total += (sum/samples)
        #p,r,f1_c,f = estadistica(conf_matrix)
        #precision.append(p)
        #recall.append(r)
        #f1_cl.append(f1_c)
        #f1.append(f)

    #print("total:", total, " n:", n)
    fitness = total/n
    #fitness = np.mean(f1)
    return fitness

def snn(n_entrada, n_salida, n, i, matrix, inputWeights, trial):
    try:
        clear()
        pop = Population(geometry=n, neuron=IZHIKEVICH)
        proj = Projection(pre=pop, post=pop, target='exc')

        if matrix.size == 0:
            raise ValueError("matrix is empty")

        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))
        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        lil_matrix[:n_rows, :n_cols] = matrix
        proj.connect_from_sparse(lil_matrix)
        nombre = 'annarchy/annarchy-'+str(int(trial))+'/annarchy-'+str(int(i))

        compile(directory=nombre, clean=False, silent=True)
        M = Monitor(pop, ['spike','v'])
        
        fit = fitness(pop,M,int(n_entrada),int(n_salida))
        return fit
    except Exception as e:
        # Capturar y manejar excepciones
        print("Error en annarchy:", e)
        return -1

def graficos(M):
    spikes = M.get('spike')
    v = M.get('v')
    t, n = M.raster_plot(spikes)
    fr = M.histogram(spikes)
    print(spikes[0])
    fig = plt.figure(figsize=(17, 17))

    # First plot: raster plot
    plt.subplot(311)
    plt.plot(t, n, 'b.')
    plt.title('Raster plot')

    # Second plot: membrane potential of a single excitatory cell
    plt.subplot(312)
    plt.plot(v[5]) # for example
    plt.title('Membrane potential')

    # Third plot: number of spikes per step in the population.
    plt.subplot(313)
    plt.plot(fr)
    plt.title('Number of spikes')
    plt.xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()
    return 0

def test(n_entrada, n_salida):
    total = n_entrada + n_salida
    lil_matrix = scipy.sparse.lil_matrix((int(total), int(total)))
    # Asignar los valores deseados (1's en el bloque superior derecho)
    lil_matrix[0:n_entrada, n_entrada:total] = 110

    pop = Population(geometry=n_entrada+n_salida, neuron=LIF)
    proj = Projection(pre=pop, post=pop, target='exc')

    proj.connect_from_sparse(lil_matrix)
    nombre = 'annarchy/annarchy-tests/annarchy-0'

    compile(directory=nombre, clean=False, silent=True)
    M = Monitor(pop, ['spike','v'])

    fitness(pop, M, n_entrada, n_salida, True)

# Cargar datos
## Iris
#dataset = load_iris()
#dataset = load_wine()
#dataset = load_breast_cancer()

# Cargar los datos del archivo CSV
df_sample = pd.read_csv('breast_cancer_sample.csv')
data_x = df_sample.drop(columns=['target']).to_numpy()
data_y = df_sample['target'].to_numpy()

#data_x = dataset.data
#data_y = dataset.target