import optuna 
import sys
import logging
import subprocess
import numpy as np
import os

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

# Función de objetivo para Optuna
def objective(trial):

    # Crear un directorio y sus subdirectorios si no existen
    os.makedirs("results/trial-"+str(trial.number), exist_ok=True)

    # Trial: single execution of the objective function
    # Suggest call parameters uniformly within the range 
    # Definir los hiperparámetros que Optuna debe optimizar
    keep = 0.577
    threshold = 3.241
    interespeciesRate = 0.000549
    noCrossoverOff = 0.329
    probabilityWeightMutated=0.851
    probabilityAddNodeSmall=0.0305
    probabilityAddLink_small=0.0436
    probabilityAddNodeLarge=0.0417
    probabilityAddLink_Large=0.0789
    c1=0.53
    c2=0.959
    c3=0.306
    tau_c = trial.suggest_float('tau_c', 10, 30)
    a_plus = trial.suggest_float('a_plus', 0.001, 0.09)
    a_minus = trial.suggest_float('a_minus', 0.001, 0.09)
    tau_plus = trial.suggest_float('tau_plus', 10, 30)
    tau_minus = trial.suggest_float('tau_minus', 10, 30)
    p = subprocess.Popen(["./NEAT", str(keep), str(threshold), str(interespeciesRate),
                          str(noCrossoverOff), str(probabilityWeightMutated), str(probabilityAddNodeSmall), 
                          str(probabilityAddLink_small), str(probabilityAddNodeLarge), str(probabilityAddLink_Large), 
                          str(c1), str(c2), str(c3), str(trial.number), str(tau_c), str(a_plus), str(a_minus), str(tau_plus), str(tau_minus)],
                        stderr=subprocess.PIPE, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    p.wait()

    # Capturar la salida estándar y de error
    stdout, stderr = p.communicate()

    # Imprimir la salida estándar del proceso NEAT
    output = stdout.decode('utf-8')
    #print(f"Output del trial {trial.number}:\n{output}")
    
    # Imprimir la salida de error si existe
    if stderr:
        print(f"Errores del trial {trial.number}:\n{stderr.decode('utf-8')}")

    # Asumir que el fitness es la última línea de la salida estándar
    try:
        FuncValue = float(output.strip().split("\n")[-1])
    except ValueError:
        FuncValue = -np.inf  # En caso de error, retorna un valor muy bajo

    '''
    FuncValue = p.returncode
    #get final line of output
    if FuncValue != None:
        FuncValue = float(FuncValue)
    else:
        FuncValue = -np.inf
    '''
    return FuncValue
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

study_name = "example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

# Create the study object (an optimization session = set of trials)
study = optuna.create_study(study_name=study_name,
                            storage=storage_name,
                            direction='maximize', 
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner(),
                            load_if_exists=True)
# Pass the objective function method
study.optimize(objective, n_trials=100) #timeout in seconds

print(f'Mejor valor: {study.best_value}')
print(f'Mejores parámetros: {study.best_params}')

# Get the best parameter
found_params = study.best_params
found_value  = study.best_value
found_trial  = study.best_trial

"""
# Visualization options 
fig = optuna.visualization.plot_optimization_history(study)
fig = optuna.visualization.plot_parallel_coordinate(study)
fig = optuna.visualization.plot_slice(study)
fig = optuna.visualization.plot_param_importances(study)
fig = optuna.visualization.plot_edf(study)
fig.show()
"""


#https://adambaskerville.github.io/posts/PythonSubprocess