import numpy as np
import matplotlib.pyplot as plt

nombres = [
    "force-rstdp-hidden1",
    "force-rstdp-hidden2",
    "force-rstdp-rand1",
    "rstdphidden1",
    "rstdphidden2",
    "rstdprand1"
]

colores = [
    "cyan",
    "magenta",
    "gold",
    "pink",
    "gray",
    "olive"
]

retornos = {}
for nombre in nombres:
    retornos[nombre] = np.load(f"{nombre}/retornos2_{nombre}.npy", allow_pickle=True)

plt.figure()
for nombre, color in zip(nombres, colores):
    datos = np.array(retornos[nombre])[:, :400]
    media = np.mean(datos, axis=0)
    plt.plot(media, label=nombre, color=color)
plt.title("Convergencia comparada")
plt.xlabel("Episodio")
plt.ylabel("Recompensa promedio")
plt.legend()
plt.show()

plt.figure()
datos_box = [np.mean(np.array(retornos[nombre])[:, :400], axis=1) for nombre in nombres]
plt.boxplot(datos_box, labels=nombres)
plt.title("Boxplot comparado")
plt.ylabel("Recompensa promedio por trial")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

