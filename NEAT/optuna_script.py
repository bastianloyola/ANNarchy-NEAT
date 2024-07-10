import optuna
import subprocess

# Función para modificar config.cfg
def modify_config(params):
    with open('config.cfg', 'w') as f:
        for key, value in params.items():
            f.write(f'{key}={value}\n')

# Función de objetivo para Optuna
def objective(trial):
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
    largeSize = trial.suggest_int('largeSize', 1, 100)  # Cambiado a entero
    c1 = trial.suggest_float('c1', 0.0, 10.0)
    c2 = trial.suggest_float('c2', 0.0, 10.0)
    c3 = trial.suggest_float('c3', 0.0, 1.0)

    # Mantener los parámetros que no deben cambiar
    params = {
        'keep': keep,
        'threshold': threshold,
        'probabilityInterespecies': probabilityInterespecies,
        'probabilityNoCrossoverOff': probabilityNoCrossoverOff,
        'probabilityWeightMutated': probabilityWeightMutated,
        'probabilityAddNodeSmall': probabilityAddNodeSmall,
        'probabilityAddLink_small': probabilityAddLink_small,
        'probabilityAddNodeLarge': probabilityAddNodeLarge,
        'probabilityAddLink_Large': probabilityAddLink_Large,
        'largeSize': largeSize,
        'c1': c1,
        'c2': c2,
        'c3': c3,
        'initial_weight': 30.0,  # No cambiar
        'numberGenomes': 2,  # No cambiar
        'numberInputs': 80,  # No cambiar
        'numberOutputs': 40,  # No cambiar
        'evolutions': 3,  # No cambiar
        'n_max': 200,  # No cambiar
        'learningRate': 10.0,  # No cambiar
        'inputWeights': '0.0,150.0',  # No cambiar
        'weightsRange': '-20.0,80.0',  # No cambiar
        'process_max': 2,  # No cambiar
        'function': 'cartpole2'  # No cambiar
    }
    modify_config(params)

    # Llamar a la función main desde C++
    result = subprocess.run(['./NEAT'], capture_output=True, text=True)

    # Verificar si hubo algún error en la ejecución
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return float('inf')  # Devolver un valor alto en caso de error

    # Extraer el resultado de la salida del programa
    try:
        fitness = float(result.stdout.strip())
    except ValueError:
        print("Error: No se pudo convertir la salida a float.")
        return float('inf')  # Devolver un valor alto en caso de error

    return fitness

# Crear y ejecutar el estudio de Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f'Mejor valor: {study.best_value}')
print(f'Mejores parámetros: {study.best_params}')
