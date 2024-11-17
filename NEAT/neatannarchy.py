import subprocess
def write_annarchy(neuron_model, func):
    # Leer el contenido dinámico de neuronmodel.py
    with open("neuronmodel.py", "r") as neuron_file:
        neuron_model_text = neuron_file.read()

    # Leer el contenido dinámico de fitness.py
    with open("fitness.py", "r") as func_file:
        func_text = func_file.read()

    # Construcción del código base usando listas
    lines = [
        "import numpy as np",
        "import random as rd",
        "import scipy.sparse",
        f"from neuronmodel import {neuron_model}",
        "",
        neuron_model_text.strip(),  # Añadir el contenido de neuronmodel.py
        "",
        "def snn(n_entrada, n_salida, n, i, matrix, inputWeights, trial):",
        "    try:",
        "        clear()",
        "        # Se crea una población de n neuronas",
        f"        pop = Population(geometry=n, neuron={neuron_model})",
        "        # Se crea una proyección de las neuronas a sí mismas",
        "        proj = Projection(pre=pop, post=pop, target='exc')",
        "        # Verificar el tamaño de la matrix",
        "        if matrix.size == 0:",
        "            raise ValueError('matrix is empty')",
        "        # lil_matrix scipy nxn con los valores de la matrix",
        "        lil_matrix = scipy.sparse.lil_matrix((int(n), int(n)))",
        "        n_rows = matrix.shape[0]",
        "        n_cols = matrix.shape[1]",
        "        lil_matrix[:n_rows, :n_cols] = matrix",
        "        proj.connect_from_sparse(lil_matrix)",
        "        nombre = 'annarchy/annarchy-' + str(int(trial)) + '/annarchy-' + str(int(i))",
        "        compile(directory=nombre, clean=False, silent=True)",
        "        M = Monitor(pop, ['spike', 'v'])",
        "        input_index = []",
        "        output_index = []",
        "        n_entrada = int(n_entrada)",
        "        n_salida = int(n_salida)",
        "        for i in range(n_entrada):",
        "            input_index.append(i)",
        "        for i in range(n_entrada, n_salida + n_entrada):",
        "            output_index.append(i)",
        "        # Verificar el tamaño de inputWeights",
        "        if inputWeights.size == 0:",
        "            raise ValueError('inputWeights is empty')",
        "",
        f"        fit = {func}(pop, M, input_index, output_index, inputWeights)",
        "        return fit",
        "    except Exception as e:",
        "        # Capturar y manejar excepciones",
        "        print('Error en annarchy:', e)",
        "",
        func_text.strip()  # Añadir el contenido de fitness.py
    ]

    # Escribir las líneas en el archivo final
    with open("annarchy.py", "w", newline='\n') as f:
        f.write("\n".join(lines))
        f.write("\n")  # Asegurar una línea final


def get_function():
    #open config file and get the parameter "function"
    with open('config/config.cfg') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('function='):
                return line.split('=')[1].strip()
            

def get_neuron_model():
    #open config file and get the parameter "neuronModel"
    with open('config/config.cfg') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('neuronModel='):
                return line.split('=')[1].strip()
            

def runNEAT(trial, func, neuron_model, procesos, evolutions, population):
    #open config file and write the parameters
    with open('config/config.cfg', 'a') as f:
        f.write(f'function={func}\n')
        f.write(f'neuronModel={neuron_model}\n')
        f.write(f'process_max={procesos}\n')
        f.write(f'evolutions={evolutions}\n')
        f.write(f'numberGenomes={population}\n')

    #write the annarchy file
    write_annarchy(neuron_model, func)
    fitness = None
    process = subprocess.Popen(["./NEAT", str(trial)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output, error = process.communicate()

    process.wait()

    if process.returncode != 0:
        print("Error durante la ejecución:", error)
    else:
        fitness = float(output.strip().split("\n")[-1])
        return fitness
