import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## EJECUTAMOS UNA TABLA DE DATOS DE EXCEL PARA VER DIFERENTES EJERCICIOS

datos = pd.read_excel('ReduccionSolidosDemandaOxigeno.xlsx')


x = datos[["Reduccion de solidos"]]


y = datos[["Reduccion de la demanda de oxigeno"]]

## VAMOS A GRAFICAR LOS DATOS

plt.scatter(x,y)
plt.xlabel("Reduccion de solidos")
plt.ylabel("Reduccion de la demanda de oxigeno")
plt.grid()
plt.show()    

## CONVERTIR DATAFRAME A NUMPY
matriz = datos.to_numpy()

n = len(matriz)
sumatoria_x = np.sum(matriz[:,0])
sumatoria_y = np.sum(matriz[:,1])
sumatoria_producto = np.sum(matriz[:,0] * (matriz[:,1]))
sumatoria_cuadrado_x = np.sum(matriz[:,0] * (matriz[:,0]))


print("n:", n)
print("sumatoria x:", sumatoria_x)
print("sumatoria y:", sumatoria_y)
print("sumatoria xy:", sumatoria_producto)
print("sumatoria x^2:", sumatoria_cuadrado_x)


#-------------------------------------------
b1 = (n*sumatoria_producto - sumatoria_x*sumatoria_y) / (n*sumatoria_cuadrado_x - sumatoria_x*sumatoria_x)
b0 = (sumatoria_y - b1*sumatoria_x) / (n)
print("b1:",b1)
print("b2:", b0)

### ----------------> REALIZADO AHORA CON SCIKIT-LEARN <--------------------
# CREAMOS EL MODELO
clf = LinearRegression()
clf.fit(x,y)

# Con clf.coef_ despues de entrenarlo con fit nos da el valor directamente de b1
clf.coef_
# Con clf.intercept_ despues de entrenarlo con fit nos da el valor directamente de b0
clf.intercept_

# Para predecir valores que no están en la tabla de excel o csv probamos con otro numero
clf.predict([[100]])

# VAMOS A PINTAR VALORES PARA QUE SE VEA LA REGRESION LINEAL
plt.plot(x,y)
plt.plot(x, clf.predict(x))
plt.title("Regresion Lineal Simple")
plt.xlabel("Reduccion de solidos")
plt.ylabel("Reduccion de la demanda de oxigeno")
plt.legend(["y", "Predicciones"])
plt.grid()
plt.show()

# SOLUCION DEL PROBLEMA y = 3.83 + 0.9x