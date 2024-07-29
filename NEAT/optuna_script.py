import optuna 
import sys
import logging
import subprocess

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
    keep = trial.suggest_float('keep', 0.0, 1.0)
    threshold = trial.suggest_float('threshold', 0.0, 10.0)
    probabilityInterespecies = trial.suggest_float('probabilityInterespecies', 0.0, 0.01)
    probabilityNoCrossoverOff = trial.suggest_float('probabilityNoCrossoverOff', 0.0, 1.0)
    probabilityWeightMutated = trial.suggest_float('probabilityWeightMutated', 0.0, 1.0)
    probabilityAddNodeSmall = trial.suggest_float('probabilityAddNodeSmall', 0.0, 0.1)
    probabilityAddLink_small = trial.suggest_float('probabilityAddLink_small', 0.0, 0.1)
    probabilityAddNodeLarge = trial.suggest_float('probabilityAddNodeLarge', 0.0, 0.1)
    probabilityAddLink_Large = trial.suggest_float('probabilityAddLink_Large', 0.0, 1.0)
    largeSize = trial.suggest_int('largeSize', 1, 100)  
    c1 = trial.suggest_float('c1', 0.0, 10.0)
    c2 = trial.suggest_float('c2', 0.0, 10.0)
    c3 = trial.suggest_float('c3', 0.0, 1.0)
    p = subprocess.Popen(["./NEAT", str(keep), str(threshold), str(probabilityInterespecies), str(probabilityNoCrossoverOff), str(probabilityWeightMutated), str(probabilityAddNodeSmall), str(probabilityAddLink_small), str(probabilityAddNodeLarge), str(probabilityAddLink_Large), str(largeSize), str(c1), str(c2), str(c3)],
                        stderr=subprocess.PIPE, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    FuncValue = p.stdout.readlines()
    #get final line of output
    FuncValue = FuncValue[-1]
    #convert to float
    FuncValue = float(FuncValue)

    return FuncValue

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Create the study object (an optimization session = set of trials)
study = optuna.create_study(study_name='Test-study',
                            direction='maximize', 
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner(),
                            load_if_exists=True)
# Pass the objective function method
study.optimize(objective, n_trials=100, timeout=60) #timeout in seconds

print(f'Mejor valor: {study.best_value}')
print(f'Mejores parámetros: {study.best_params}')

