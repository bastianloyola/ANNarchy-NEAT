import json
import matplotlib.pyplot as plt

# Listas de carpetas y rango de trials por carpeta
carpetas = ['A-IZ/', 'A-LIF/']  # Añadir las carpetas necesarias
trials_por_carpeta = [(0, 11), (0, 11)]  # Añadir rangos según corresponda

# Configuración del gráfico
plt.figure(figsize=(12, 8))

# Recorre cada carpeta y sus respectivos trials
for k in range(len(carpetas)):
    nodos_cantidad = []
    conexiones_cantidad = []
    trials = []

    # Iterar sobre el rango de trials para la carpeta actual
    for j in range(trials_por_carpeta[k][0], trials_por_carpeta[k][1]):
        directorio = carpetas[k] + 'trial-' + str(j + 1)
        file = directorio + '/output.json'
        
        # Leer el archivo JSON
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Obtener la cantidad de nodos y conexiones
                nodos = data.get('Nodos', [])
                conexiones = data.get('Conexiones', [])
                
                # Guardar los valores en las listas
                trials.append(j + 1)
                nodos_cantidad.append(len(nodos))
                conexiones_cantidad.append(len(conexiones))
        
        except FileNotFoundError:
            print(f"Archivo no encontrado: {file}")
            continue

    # Graficar la fluctuación para nodos y conexiones
    plt.plot(trials, nodos_cantidad, label=f'Nodos ({carpetas[k].strip("/")})', marker='o')
    plt.plot(trials, conexiones_cantidad, label=f'Conexiones ({carpetas[k].strip("/")})', marker='x')

# Configurar el gráfico
plt.xlabel('Trial')
plt.ylabel('Cantidad')
plt.title('Cantidad de Nodos y Conexiones por configuración')
plt.legend()
plt.grid(True)
plt.xticks(trials)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
