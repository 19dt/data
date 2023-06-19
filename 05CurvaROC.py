import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# SON TODOS DATOS DE CLASIFICACION
# RESULTADOS EN LA CURVA ROC 1-> BIEN, 0.5-> NO ES CAPAZ DE SEPARAR, 0-> LO ESTÁ HACIENDO AL REVES

x,y = make_moons(n_samples=128)
plt.scatter(x[:,0], x[:,1], c=y)
plt.grid()
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x,y)
# modelo red neuronal
clf = MLPClassifier()
# entreno
clf.fit(x_train, y_train)
# resultado
clf.score(x_test, y_test)
# Cojo la primera columna
probabilidades = clf.predict_proba(x_test)
probabilidades = probabilidades[:,1]

# vemos la curva y sale 0.98 lo cual se acerca bastante
auc = roc_auc_score(y_test, probabilidades) 

# calculamos los ejes x(fpr), y(tpr) y thresholds
fpr, tpr, thresholds = roc_curve(y_test, probabilidades)

# vamos a graficar
plt.plot(fpr, tpr, marker = '.', label='MLP')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()