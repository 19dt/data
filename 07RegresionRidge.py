# USAREMOS EL DATASET BOSTON_HOUSING PARA PREDECIR EL PRECIO DE LAS CASAS DE BOSTON CON 3 VARIABLES CON EL ALGORITMO DE REGRESION
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

''' iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = Ridge()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

df_y_test = pd.DataFrame(y_test, columns=['y_test'])
df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
pd.concat([df_y_test, df_y_pred], axis=1)

plt.plot(y_test)
plt.plot(y_pred)
plt.grid()
plt.xlabel('N_casa')
plt.ylabel('Precio')
plt.legend(['y_test', 'predicciones'])
plt.show() '''

# AHORA CON LA REGRESION LINEAL EN VEZ DE CON RIDGE


iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = LinearRegression()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

df_y_test = pd.DataFrame(y_test, columns=['y_test'])
df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
pd.concat([df_y_test, df_y_pred], axis=1)

plt.plot(y_test)
plt.plot(y_pred)
plt.grid()
plt.xlabel('N_casa')
plt.ylabel('Precio')
plt.legend(['y_test', 'predicciones'])
plt.show()