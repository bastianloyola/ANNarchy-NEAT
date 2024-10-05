import numpy as np

import subprocess

from irace import irace

# FIXME: Use benchmark instances from scipy

# LB = [-5.12]
# UB = [5.12]

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
    params_str = f"""\[keep: {keep},
        threshold: {threshold},
        interSpeciesRate: {interSpeciesRate},
        noCrossoverOff: {noCrossoverOff},
        probabilityWeightMutated: {probabilityWeightMutated},
        probabilityAddNodeSmall: {probabilityAddNodeSmall},
        probabilityAddLink_small: {probabilityAddLink_small},
        probabilityAddNodeLarge: {probabilityAddNodeLarge},
        probabilityAddLink_Large: {probabilityAddLink_Large},
        c1: {c1},
        c2: {c2},
        c3: {c3}]"""
    return params_str


# This target_runner is over-complicated on purpose to show what is possible.
def target_runner(experiment, scenario):
    print(experiment['configuration'])
    # keep = experiment['configuration']['keep']
    # threshold = experiment['configuration']['threshold']
    # interespeciesRate = experiment['configuration']['interespeciesRate']
    # noCrossoverOff = experiment['configuration']['noCrossoverOff']
    # probabilityWeightMutated = experiment['configuration']['probabilityWeightMutated']
    # probabilityAddNodeSmall = experiment['configuration']['probabilityAddNodeSmall']
    # probabilityAddLink_small = experiment['configuration']['probabilityAddLink_small']
    # probabilityAddNodeLarge = experiment['configuration']['probabilityAddNodeLarge']
    # probabilityAddLink_Large = experiment['configuration']['probabilityAddLink_Large']
    # c1 = experiment['configuration']['c1']
    # c2 = experiment['configuration']['c2']
    # c3 = experiment['configuration']['c3']

    keep, threshold, interespeciesRate, noCrossoverOff, probabilityWeightMutated, probabilityAddNodeSmall, probabilityAddLink_small, probabilityAddNodeLarge, probabilityAddLink_Large, c1, c2, c3 = unpack_params(**experiment['configuration'])

    p = subprocess.Popen(["./NEAT", str(keep), str(threshold), str(interespeciesRate),
                          str(noCrossoverOff), str(probabilityWeightMutated), str(probabilityAddNodeSmall), 
                          str(probabilityAddLink_small), str(probabilityAddNodeLarge), str(probabilityAddLink_Large), 
                          str(c1), str(c2), str(c3), str(0)],
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

    with open("./irace-output.log", "a") as file:
        file.write(f"PARAMETERS: {params_to_str(**experiment['configuration'])} - FUNC_VALUE: {func_value} \n")

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

# These are dummy "instances", we are tuning only on a single function.
instances = np.array([1]) # np.arange(100)

# See https://mlopez-ibanez.github.io/irace/reference/defaultScenario.html
scenario = dict(
    instances = instances,
    nbConfigurations = 6,
    maxExperiments = 500,
    debugLevel = 3,
    digits = 3,
    parallel=1, # It can run in parallel ! 
    logFile = "./log.Rdata")


tuner = irace(scenario, parameters_table, target_runner)
tuner.set_initial_from_str(default_values)
best_confs = tuner.run()
# Pandas DataFrame
print(best_confs)

with open("./irace-output.log", "a") as file:
    file.write(f"BEST_CONFS: {best_confs.to_string()}\n")