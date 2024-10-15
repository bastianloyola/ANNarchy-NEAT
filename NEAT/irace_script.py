import numpy as np
import pandas as pd

import os
import sys
import subprocess
import threading

from irace import irace

# Se recibe el tipo de neurona de Izhikevich que se usará en la búsqueda, debe estar dentro de los definidos en la constante
# IZHIKEVICH_KEYS
IZHIKEVICH_KEYS = ["X", "A", "B", "E", "F", "G", "H", "M", "P", "S"]
izhikevich_neuron_type = str(sys.argv[1])
if izhikevich_neuron_type not in IZHIKEVICH_KEYS:
    sys.exit(0)

# Se obtiene la ruta base para el archivo de logs .csv
base_path_log_file = str(sys.argv[2])

lock = threading.Lock()

def unpack_params(
        keep,
        threshold,
        interSpeciesRate,
        noCrossoverOff,
        probabilityWeightMutated,
        probabilityAddNodeSmall,
        probabilityAddLink_small,
        probabilityAddNodeLarge,
        probabilityAddLink_Large,
        c1,
        c2,
        c3
):
    return (keep,
        threshold,
        interSpeciesRate,
        noCrossoverOff,
        probabilityWeightMutated,
        probabilityAddNodeSmall,
        probabilityAddLink_small,
        probabilityAddNodeLarge,
        probabilityAddLink_Large,
        c1,
        c2,
        c3)

def params_to_str(keep,
        threshold,
        interSpeciesRate,
        noCrossoverOff,
        probabilityWeightMutated,
        probabilityAddNodeSmall,
        probabilityAddLink_small,
        probabilityAddNodeLarge,
        probabilityAddLink_Large,
        c1,
        c2,
        c3):
    params_str = f"\n[keep: {keep},\n \
        threshold: {threshold},\n \
        interSpeciesRate: {interSpeciesRate},\n \
        noCrossoverOff: {noCrossoverOff},\n \
        probabilityWeightMutated: {probabilityWeightMutated},\n \
        probabilityAddNodeSmall: {probabilityAddNodeSmall},\n \
        probabilityAddLink_small: {probabilityAddLink_small},\n \
        probabilityAddNodeLarge: {probabilityAddNodeLarge},\n \
        probabilityAddLink_Large: {probabilityAddLink_Large},\n \
        c1: {c1},\n \
        c2: {c2},\n \
        c3: {c3}]"
    return params_str

def write_experiment(experiment: dict, func_value: float):
    keep, \
    threshold, \
    interespeciesRate, \
    noCrossoverOff, \
    probabilityWeightMutated, \
    probabilityAddNodeSmall, \
    probabilityAddLink_small, \
    probabilityAddNodeLarge, \
    probabilityAddLink_Large, \
    c1, \
    c2, \
    c3 = unpack_params(**experiment['configuration'])
    row = pd.DataFrame({
        "neuron_type": [izhikevich_neuron_type],
        "id_configuration": [experiment['id.configuration']],
        "id_instance": [experiment['id.instance']],
        "seed": [experiment['seed']],
        "p_keep": [keep],
        "p_threshold": [threshold],
        "p_interespeciesRate": [interespeciesRate],
        "p_noCrossoverOff": [noCrossoverOff],
        "p_probabilityWeightMutated": [probabilityWeightMutated],
        "p_probabilityAddNodeSmall": [probabilityAddNodeSmall],
        "p_probabilityAddLink_small": [probabilityAddLink_small],
        "p_probabilityAddNodeLarge": [probabilityAddNodeLarge],
        "p_probabilityAddLink_Large": [probabilityAddLink_Large],
        "p_c1": [c1],
        "p_c2": [c2],
        "p_c3": [c3],
        "func_value": [func_value]
    })
    filepath = f"{base_path_log_file}/{izhikevich_neuron_type}/irace-detail.csv"
    with lock:
        file_exists = os.path.isfile(filepath)
        row.to_csv(filepath, mode="a", header=not file_exists, index=False)

def target_runner(experiment, scenario):
    keep, \
    threshold, \
    interespeciesRate, \
    noCrossoverOff, \
    probabilityWeightMutated, \
    probabilityAddNodeSmall, \
    probabilityAddLink_small, \
    probabilityAddNodeLarge, \
    probabilityAddLink_Large, \
    c1, \
    c2, \
    c3 = unpack_params(**experiment['configuration'])

    p = subprocess.Popen(["./NEAT", str(keep), str(threshold), str(interespeciesRate),
                          str(noCrossoverOff), str(probabilityWeightMutated), str(probabilityAddNodeSmall), 
                          str(probabilityAddLink_small), str(probabilityAddNodeLarge), str(probabilityAddLink_Large), 
                          str(c1), str(c2), str(c3), str(izhikevich_neuron_type)],
                        stderr=subprocess.PIPE, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    p.wait()

    # Capturar la salida estándar y de error
    stdout, _ = p.communicate()

    # Imprimir la salida estándar del proceso NEAT
    output = stdout.decode('utf-8')


    try:
        func_value = float(output.strip().split("\n")[-1])
    except ValueError:
        func_value = -np.inf  # En caso de error, retorna un valor muy bajo

    # with open("./irace-output-0.log", "a") as file:
    #     file.write(f"ID_CONFIGURATION: {experiment['id.configuration']}\nID_INSTANCE: {experiment['id.instance']}\nSEED: {experiment['seed']}\nPARAMETERS: {params_to_str(**experiment['configuration'])}\nFUNC_VALUE: {func_value}\n\n")

    write_experiment(experiment, func_value)

    return dict(cost=(-1 * func_value)) # Se multiplica por -1 porque irace minimiza el valor de la función objetivo

parameters_table = '''
keep       "" r (0.4, 0.6)
threshold "" r (2.0, 4.0)
interSpeciesRate  "" r     (0.0005, 0.0015)
noCrossoverOff      "" r     (0.15, 0.35)
probabilityWeightMutated    "" r     (0.7, 0.9) 
probabilityAddNodeSmall   "" r (0.02, 0.04)
probabilityAddLink_small  "" r (0.01, 0.05)
probabilityAddNodeLarge  "" r (0.02, 0.4)
probabilityAddLink_Large  "" r (0.05, 0.2)
c1 "" r (0.5, 1.5)
c2 "" r (0.5, 1.5)
c3 "" r (0.3, 0.5)
'''

default_values = '''
keep threshold interSpeciesRate noCrossoverOff probabilityWeightMutated probabilityAddNodeSmall probabilityAddLink_small probabilityAddNodeLarge probabilityAddLink_Large c1 c2 c3
0.4 2.0 0.0005 0.15 0.7 0.02 0.01 0.02 0.05 0.5 0.5 0.3
'''

# Para este caso no existen instancias específicas que "resolver"
instances = np.array([1]) 

# See https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
scenario = dict(
    instances = instances,
    nbConfigurations = 6,
    maxExperiments = 500,
    debugLevel = 3,
    digits = 3,
    parallel=1, # It can run in parallel ! 
    logFile = f"{base_path_log_file}/{izhikevich_neuron_type}/log-0.Rdata"
)


tuner = irace(scenario, parameters_table, target_runner)
tuner.set_initial_from_str(default_values)
best_confs = tuner.run()
# Pandas DataFrame
# print(best_confs)

# with open("./irace-output-0.log", "a") as file:
#     file.write(f"BEST_CONFS: {best_confs.to_string()}\n")

filepath = f"{base_path_log_file}/{izhikevich_neuron_type}/irace-results.csv"
with lock:
    file_exists = os.path.isfile(filepath)
    best_confs.to_csv(filepath, mode="a", header=not file_exists, index=False)