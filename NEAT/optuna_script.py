import optuna 
import sys
import logging
import subprocess
import numpy as np

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice



# Función de objetivo para Optuna
def objective(trial):
    # Trial: single execution of the objective function
    # Suggest call parameters uniformly within the range 
    # Definir los hiperparámetros que Optuna debe optimizar
    keep = trial.suggest_float('keep', 0.1, 0.7)
    threshold = trial.suggest_float('threshold', 2.0, 5.0)
    interespeciesRate = trial.suggest_float('interespeciesRate', 0.01, 0.5)
    noCrossoverOff = trial.suggest_float('noCrossoverOff', 0.2, 0.6)
    probabilityWeightMutated = trial.suggest_float('probabilityWeightMutated', 0.4, 0.9)
    probabilityAddNodeSmall = trial.suggest_float('probabilityAddNodeSmall', 0.01, 0.1)
    probabilityAddLink_small = trial.suggest_float('probabilityAddLink_small', 0.01, 0.1)
    probabilityAddNodeLarge = trial.suggest_float('probabilityAddNodeLarge', 0.01, 0.1)
    probabilityAddLink_Large = trial.suggest_float('probabilityAddLink_Large', 0.1, 0.4)
    c1 = trial.suggest_float('c1', 0.1, 3.0)
    c2 = trial.suggest_float('c2', 0.1, 3.0)
    c3 = trial.suggest_float('c3', 0.1, 3.0)
    p = subprocess.Popen(["./NEAT", str(keep), str(threshold), str(interespeciesRate), str(noCrossoverOff), str(probabilityWeightMutated), str(probabilityAddNodeSmall), str(probabilityAddLink_small), str(probabilityAddNodeLarge), str(probabilityAddLink_Large), str(c1), str(c2), str(c3)],
                        stderr=subprocess.PIPE, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)


    p.wait()
    FuncValue = p.returncode

    #get final line of output
    if FuncValue != None:
        FuncValue = float(FuncValue)
    else:
        FuncValue = -np.inf

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

# Visualization options 
fig = optuna.visualization.plot_optimization_history(study)
fig = optuna.visualization.plot_parallel_coordinate(study)
fig = optuna.visualization.plot_slice(study)
fig = optuna.visualization.plot_param_importances(study)
fig = optuna.visualization.plot_edf(study)
fig.show()

#https://adambaskerville.github.io/posts/PythonSubprocess