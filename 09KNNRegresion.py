# PARA KNN DE REGRESION UTILIZAREMOS LOS DATOS DE LA DIABETES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error # para ver el valor cuadratico medio de las predicciones

data = load_diabetes()
print(data.data.shape) # para ver los datos
print(data.target.shape) # para ver las etiquetas

data.feature_names
print(data.data)

print(data.target) # etiquetas a predecir

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target)

#vamos a ver la forma de los datos
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = KNeighborsRegressor()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# Vamos a hacer un dataframe concatenado para comparar los valores predichos con los del testing
df_y_test = pd.DataFrame(y_test,columns=['y_test'])
df_y_pred = pd.DataFrame(y_pred,columns=['y_pred'])
pd.concat([df_y_test,df_y_pred], axis=1)

# Vamos a graficarlos
plt.plot(y_test)
plt.plot(y_pred)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['y_test', 'predicciones'])
plt.show()

# Para tener un medida mas cuantitativa usamos mean_squared_error
mean_squared_error(y_test,y_pred)