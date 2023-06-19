from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
iris

print(iris.data.shape)
print(iris.data)

print(iris.target.shape)
print(iris.target)

## ENTRENAMOS LAS VARIABLES DE X E Y PARA VER LAS VARIABLES (DATA) Y ETIQUETAS (TARGET)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

## SI QUEREMOS CONTROLAR EL PORCENTAJE al 30% o directamente el numero que quiero en test_size Y QUE NO MEZCLE LOS DATOS
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, shuffle=False )
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
