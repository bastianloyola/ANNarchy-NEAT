import matplotlib.pyplot as plt
import json
import numpy as np

# Cargar datos desde el archivo JSON
outputFile = 'A-IZ/trial-1/output.json'
loaded = json.load(open(outputFile))
info = loaded['Info']
operadores = loaded['Operadores']

# Asumimos que el número de generaciones es la longitud de una de las listas de datos
generaciones = range(len(info['Eliminados']))

# Primera página de gráficos (sin cambios)
fig1 = plt.figure(figsize=(12, 12))
gs1 = fig1.add_gridspec(4, 1)
fig1.suptitle('Estadísticas por Generación - Página 1', fontsize=20)

# Gráfico de eliminados
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.plot(generaciones, info['Eliminados'], label='Eliminados', color='red')
ax1.set_title('Eliminados por Generación')
ax1.set_xlabel('Generación')
ax1.set_ylabel('Cantidad')
ax1.legend()

# Gráfico de reproducidos
ax2 = fig1.add_subplot(gs1[1, 0])
reproducidos_inter, reproducidos_intra, reproducidos_muta = zip(*info['Reproducidos'])
ax2.plot(generaciones, reproducidos_inter, label='Inter-especie', color='blue')
ax2.plot(generaciones, reproducidos_intra, label='Intra-especie', color='green')
ax2.plot(generaciones, reproducidos_muta, label='Mutación', color='purple')
ax2.set_title('Reproducidos por Generación')
ax2.set_xlabel('Generación')
ax2.set_ylabel('Cantidad')
ax2.legend()

# Gráfico de especies
ax3 = fig1.add_subplot(gs1[2, 0])
ax3.plot(generaciones, [len(species) for species in info['Species']], label='Especies', color='orange')
ax3.set_title('Número de Especies por Generación')
ax3.set_xlabel('Generación')
ax3.set_ylabel('Cantidad de Especies')
ax3.legend()

# Gráfico del mejor genoma
ax4 = fig1.add_subplot(gs1[3, 0])
fitness = [best_genome[2] for best_genome in info['BestGenome']]
ax4.plot(generaciones, fitness, label='Fitness', color='cyan')
ax4.set_title('Fitness del Mejor Genoma por Generación')
ax4.set_xlabel('Generación')
ax4.set_ylabel('Fitness')
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig('page1_estadisticas.png')

# Segunda página combinada de operadores y distribución de genomas por especie
fig2 = plt.figure(figsize=(12, 12))
gs2 = fig2.add_gridspec(2, 1)
fig2.suptitle('Estadísticas por Generación - Página 2', fontsize=20)

# Gráfico combinado de operadores
ax_op = fig2.add_subplot(gs2[0, 0])
ax_op.plot(generaciones, operadores['mutacionPeso'], label='Mutación Peso', color='magenta')
ax_op.plot(generaciones, operadores['mutacionPesoInput'], label='Mutación Peso Input', color='gray')
ax_op.plot(generaciones, operadores['agregarNodos'], label='Agregar Nodos', color='blue')
ax_op.plot(generaciones, operadores['agregarLinks'], label='Agregar Links', color='green')
ax_op.set_title('Operadores por Generación')
ax_op.set_xlabel('Generación')
ax_op.set_ylabel('Cantidad')
ax_op.legend()

# Gráfico de distribución de genomas por especie
ax_barras = fig2.add_subplot(gs2[1, 0])
max_especies_por_generacion = max(len(species) for species in info['Species'])
genomas_por_especie = []

for i in range(max_especies_por_generacion):
    especie_data = [gen_species[i] if i < len(gen_species) else 0 for gen_species in info['Species']]
    genomas_por_especie.append(especie_data)

x = np.arange(len(generaciones))
bottom = np.zeros(len(generaciones))

for i, especie_data in enumerate(genomas_por_especie):
    ax_barras.bar(x, especie_data, bottom=bottom, label=f'Especie {i+1}')
    bottom += especie_data

ax_barras.set_title('Distribución de Genomas por Especie')
ax_barras.set_xlabel('Generación')
ax_barras.set_ylabel('Cantidad de Genomas')
ax_barras.legend(title='Especies', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig('page2_operadores_distribucion.png')

plt.show()
