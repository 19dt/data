# ALGORITMO K VECINOS MAS CERCANOS DONDE DEPENDIENDO DE DONDE SE DEFINA
# LA K, IRÁ A BUSCAR LOS DATOS DE LOS VECINOS MÁS CERCANOS

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

x, y = make_classification(n_samples=200)

plt.scatter(x[:,0], x[:,1], c=y)
plt.grid()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y)

plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.grid()
plt.show()

# PARA VER LAS PRUEBAS EN AZULES
plt.scatter(x_test[:,0], x_test[:,1])
plt.grid()
plt.show()

# PARA VER LOS DATOS PARA ENTRENAR
plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.scatter(x_test[:,0], x_test[:,1])
plt.grid()
plt.show()

# VAMOS CON EL CLASIFICADOR
clf = KNeighborsClassifier()

clf.fit(x_train,y_train)
clf.score(x_test,y_test)

y_pred = clf.predict(x_test)
print(y_pred)

print(y_test)

# vamos a ver con la matriz de confusion como se equivoco el modelo
matriz_confusion = confusion_matrix(y_test,y_pred)
print(matriz_confusion)

plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.scatter(x_test[0,0], x_test[0,1], s=120)
plt.grid()
plt.show()