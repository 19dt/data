# OTRO EJEMPLO DE ALGORITMO KKN
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

wine = load_wine()
print(wine.data.shape)
print(wine.target.shape)

print(wine.data)   # nombre de los datos de vino

wine.target  #numero de etiquetas
n_classes = len(np.unique(wine.target))
print("n classes: ",n_classes)

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# definimos el modelo
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)


# calculamos las predicciones de y
y_pred = clf.predict(x_test)
print(y_pred)

# matriz de confusion
matriz_confusion = confusion_matrix(y_test, y_pred)
print(matriz_confusion)