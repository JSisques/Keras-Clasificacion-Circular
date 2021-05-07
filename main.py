import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import tensorflow.keras as ks

from sklearn.datasets import make_circles

# Creamos los datos artificiales donde buscaremos clasificar dos anillos concentricos de datos
X, Y =  make_circles(n_samples=500, factor=0.5, noise=0.05)

# Resolucion del mapa de prediccion
res = 100

# Coordenadas del mapa de prediccion
_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)

# Input con cada combo de coordenadas del mapa de predicción
_pX = np.array(np.meshgrid(_x0, _x1)).T.reshape(-1,2)

# Objeto vacio a 0.5 del mapa de prediccion
_pY = np.zeros((res, res)) + 0.5

# Visualizacion del mapa de prediccion
plt.figure(figsize=(8,8))
plt.pcolormesh(_x0, _x1, _pY, cmap="coolwarm", vmin=0, vmax=1, shading='auto')

# Visualizacion de la nube de datos
plt.scatter(X[Y == 0,0], X[Y == 0,1], c="skyblue")
plt.scatter(X[Y == 1,0], X[Y == 1,1], c="salmon")

plt.tick_params(labelbottom=False, labelleft=False)
plt.show()

# Programacion red neuronal con keras
lr = 0.01 # learning rate
nn = [2, 16, 8, 1] # Numero de neuronas por cada capa en la red

# Creamos la estructura que tendra nuestro modelo
model = ks.Sequential()

# Capa 1
model.add(ks.layers.Dense(nn[1], activation='relu'))

# Capa 2
model.add(ks.layers.Dense(nn[2], activation='relu'))

# Capa 3
model.add(ks.layers.Dense(nn[3], activation='sigmoid')) #Activacion sigmoide ya que es la ultima capa y queremos valores entre 0 y 1

# Compilamos el modelo
model.compile(
    loss='mse',
    optimizer=ks.optimizers.SGD(lr=lr),
    metrics=['acc']
)

# Entrenamos el modelo
model.fit(X, Y, epochs=500)