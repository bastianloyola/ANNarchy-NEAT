import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns

def leer_json(directorio):
    with open(directorio+'/output.json', 'r') as f:
        info = json.load(f)
        data = info['Info']['FitGeneraciones']
        best = info['Info']['BestGenome']
        print(best)
        print()
        print(data)

        gbest = []
        for b in best:
            gbest.append(b[2])
    return data, gbest

def graficar_convergencia(gbest_history, directorio, valor_optimo=None):

    # Gráfico de convergencia (gbest_history)
    plt.figure(figsize=(20, 10))
    plt.plot(gbest_history, label='Gbest', color='blue')
    
    if valor_optimo is not None:
        plt.axhline(y=valor_optimo, color='red', linestyle='--', label='Valor óptimo')

    plt.title('Convergencia del Mejor Fitness')
    plt.xlabel('Iteración')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    plt.savefig(directorio+'/convergencia_gbest.png')
    #plt.show()

def graficar_boxplot(iterations_history, directorio, valor_optimo=None):
    # Gráfico de boxplot por iteración (iterations_history)
    plt.figure(figsize=(30, 15))
    sns.boxplot(data=iterations_history, width=0.3)

    if valor_optimo is not None:
        plt.axhline(y=valor_optimo, color='red', linestyle='--', label='Valor óptimo')

    plt.title('Distribución del Fitness por Iteración', fontsize=18)
    plt.xlabel('Iteración', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)
    plt.xticks(ticks=range(len(iterations_history)), labels=range(1, len(iterations_history) + 1), fontsize=14)  # Valores enteros en X
    plt.yticks(fontsize=14)  # Valores flotantes en Y
    plt.grid()
    plt.legend(fontsize=14)
    plt.savefig(directorio + '/boxplot_iterations.png')
    # plt.show()


def graficar_convergencia(gbest_history, directorio, valor_optimo=None):
    # Gráfico de convergencia (gbest_history)
    plt.figure(figsize=(30, 15))
    plt.plot(gbest_history, label='Gbest', color='blue', linewidth=2)

    if valor_optimo is not None:
        plt.axhline(y=valor_optimo, color='red', linestyle='--', label='Valor óptimo')

    plt.title('Convergencia del Mejor Fitness', fontsize=18)
    plt.xlabel('Iteración', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)
    plt.xticks(ticks=range(len(gbest_history)), labels=range(1, len(gbest_history) + 1), fontsize=14)  # Valores enteros en X
    plt.yticks(fontsize=14)  # Valores flotantes en Y
    plt.legend(fontsize=14)
    plt.grid()
    plt.savefig(directorio + '/convergencia_gbest.png')
    # plt.show()


def graficar_combinado(gbest_history, iterations_history, directorio, valor_optimo=None):
    # Crear figura
    plt.figure(figsize=(30, 15))  # Mantener tamaño similar al gráfico de boxplot

    # Gráfico de boxplot
    sns.boxplot(data=iterations_history, width=0.3, boxprops={'alpha': 0.6})
    plt.xlabel('Iteración', fontsize=16)
    plt.ylabel('Fitness', fontsize=16)
    plt.xticks(ticks=range(len(iterations_history)), labels=range(1, len(iterations_history) + 1), fontsize=12)  # Valores enteros en X
    plt.yticks(fontsize=14)  # Valores flotantes en Y
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # Gráfico de convergencia superpuesto
    plt.plot(gbest_history, label='Gbest', color='blue', linewidth=2)

    # Línea de referencia para el valor óptimo
    if valor_optimo is not None:
        plt.axhline(y=valor_optimo, color='red', linestyle='--', label='Valor óptimo', alpha=0.8)

    # Ajustar límites del eje Y
    min_fitness = min(min(min(iterations_history)), min(gbest_history))
    max_fitness = max(max(max(iterations_history)), max(gbest_history))
    if valor_optimo is not None:
        min_fitness = min(min_fitness, valor_optimo)
        max_fitness = max(max_fitness, valor_optimo)
    plt.ylim(min_fitness - 0.1 * abs(min_fitness), max_fitness + 0.1 * abs(max_fitness))

    # Agregar leyenda
    plt.legend(loc='upper right', fontsize=16)

    # Título
    plt.title('Convergencia del Mejor Fitness y Distribución por Iteración', fontsize=20)
    plt.tight_layout(pad=3)

    # Guardar gráfico
    plt.savefig(directorio + '/combinado_gbest_boxplot_superpuesto.png')
    # plt.show()








def grenerar_graficos(directorio, valor_optimo=None):
    iterations_history, gbest_history = leer_json(directorio)

    graficar_convergencia(gbest_history, directorio, valor_optimo)
    graficar_boxplot(iterations_history, directorio, valor_optimo)
    graficar_combinado(gbest_history, iterations_history, directorio, valor_optimo)

    print("Gráficos generados en la carpeta:", directorio)

grenerar_graficos('results/trial-0')
"""carpetas  = ['basico','hibrido']
instancias = ['eil76','dj38','ch130','berlin52','qa194']
valores_optimos = [525, 6656, 6110, 7544, 9352]
cantidad = 11

for c in carpetas:
    for i in instancias:
        for j in range(1,cantidad+1):
            directorio = c+"/"+i+"/T"+str(j)
            grenerar_graficos(directorio, valores_optimos[instancias.index(i)])"""