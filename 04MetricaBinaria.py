# VAMOS A APRENDER METRICAS DE EVALUACION: ACCAURACY, MATRIZ DE CONFUSION, PRECISION, RECALL Y F1-SCORE
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,precision_score,
                             recall_score,f1_score, classification_report)

# x, y = make_moons(n_samples=200) ----------> DATOS BALANCEADOS
#x, y = make_moons(n_samples=(150,50)) ------> DATOS DESBALANCEADOS
# plt.scatter(x[:,0], x[:,1], c = y)
# plt.grid()
# plt.show()

# # Entrenamos el modelo con los datos dados de make_moons
# x_train, x_test, y_train, y_test = train_test_split(x,y)
# clf = MLPClassifier()
# clf.fit(x_train, y_train)

# clf.score(x_test, y_test)

# y_pred = clf.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred)

# matriz_confusion = confusion_matrix(y_test, y_pred)

# # PARA VER LAS OPERACION PARA CONSEGUIR LA PRECISION, RECALL, F1 SCORE, TENEMOS LA FORMULA EN EL PROYECTO (METRICAS DE EVALUACION.PNG)
# # ([[26,8,                 TN,FP
# #    0,16]])      ----->    FN,TP


# # PRECISION = TP(true positive)/TP(true positive) + FP(False positive)
# #precision = 16/23
# precision = precision_score(y_test,y_pred)

# # RECALL = TP(True positive)/TP(True Positive) + FN (false negative)
# #recall = 16/16
# recall = recall_score(y_test,y_pred)

# # F1 Score = 2 * (precision*recall) / 2* (precision+recall)
# #f1 = 0.40
# f1 = f1_score(y_test, y_pred)

# # Classification Reports nos da los datos anteriores en una tabla
# report = classification_report(y_test, y_pred)


#-------------------------> OTRO EJEMPLO<--------------------------
from sklearn.datasets import load_wine

wine = load_wine()
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target)

clf = MLPClassifier()
clf.fit(x_train, y_train)

clf.score(x_test, y_test)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

matriz_confusion = confusion_matrix(y_test, y_pred)

# Cuando hay mas de dos clases, hay que pasarle tambien el average = 'macro'
precision = precision_score(y_test,y_pred, average='macro')

recall = recall_score(y_test,y_pred, average='macro')

f1 = f1_score(y_test, y_pred, average='macro')

report = classification_report(y_test, y_pred)