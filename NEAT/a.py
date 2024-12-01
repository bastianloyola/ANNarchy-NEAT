import numpy as np

# Definir límites para cada variable de observación
limites = [
    (-4.8, 4.8),  # Posición del carro
    (-10.0, 10.0),  # Velocidad del carro (estimado)
    (-0.418, 0.418),  # Ángulo del poste en radianes
    (-10.0, 10.0)  # Velocidad angular del poste (estimado)
]

num_neuronas_por_variable = 20
std_dev = 0.4  # Controla cuán concentrados están los incrementos en el centro
intervalos_por_variable = []

for low, high in limites:
    # Crear una distribución gaussiana normalizada en el rango [-1, 1]
    x = np.linspace(-1, 1, num_neuronas_por_variable)
    gaussian_weights = np.exp(-0.5 * (x / std_dev) ** 2)
    
    # Normalizar los pesos para que sumen 1
    gaussian_weights /= gaussian_weights.sum()
    
    # Escalar los pesos al rango total
    increments = gaussian_weights * (high - low)
    
    # Calcular los límites acumulados
    limites_acumulados = np.concatenate([[low], low + np.cumsum(increments)])
    
    # Guardar los intervalos como listas anidadas
    intervalos = [[limites_acumulados[i], limites_acumulados[i+1]] for i in range(len(limites_acumulados) - 1)]
    intervalos_por_variable.append(intervalos)

# Imprimir los intervalos generados
for i, intervalos in enumerate(intervalos_por_variable):
    print(f"Variable {i + 1}:")
    print(intervalos)
