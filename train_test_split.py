from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
iris

print(iris.data.shape)
print(iris.data)

print(iris.target.shape)
print(iris.target)