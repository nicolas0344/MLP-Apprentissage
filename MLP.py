# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:01:00 2022

@author: Nicolas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



#Question 1
# code pour generer le graphique XOR
x = np.array([0, 1, 0, 1])
y = np.array([0, 0, 1, 1])
groupXOR = np.array([0, 1, 1, 0])
cdict = {0: 'red', 1: 'green'}
n = ['(0,1)', '(1,1)', '(0,0)', '(1,0)']
loc = [(0.0275,0.98), (0.875,0.98), (0.0275,-0.015), (0.875,-0.015)]

fig, ax = plt.subplots()
for g in np.unique(groupXOR):
    ix = np.where(groupXOR == g)
    ax.scatter(x[ix], y[ix], c = cdict[g], label = g, s=170)
for i, txt, l in zip(range(4), n, loc):
    ax.annotate(txt, xy=(x[i], y[i]), xytext=l, fontsize = 13)
ax.legend(loc = 'center right')
plt.title('XOR (OU exclusif)')
plt.show()

# code pour generer le graphique AND
groupAND = np.array([0, 0, 0, 1])
fig, ax = plt.subplots()
for g in np.unique(groupAND):
    ix = np.where(groupAND == g)
    ax.scatter(x[ix], y[ix], c = cdict[g], label = g, s=170)
for i, txt, l in zip(range(4), n, loc):
    ax.annotate(txt, xy=(x[i], y[i]), xytext=l, fontsize = 13)
ax.legend(loc = 'center right')
plt.title('AND (ET)')
plt.show()

# code pour generer le graphique OR
groupOR = np.array([0, 1, 1, 1])
fig, ax = plt.subplots()
for g in np.unique(groupOR):
    ix = np.where(groupOR == g)
    ax.scatter(x[ix], y[ix], c = cdict[g], label = g, s=170)
for i, txt, l in zip(range(4), n, loc):
    ax.annotate(txt, xy=(x[i], y[i]), xytext=l, fontsize = 13)
ax.legend(loc = 'center right')
plt.title('OR (OU inclusif)')
plt.show()

#Question 2
xtrainAND = [[0., 0.], [0., 1.], [1., 0.],[1., 1.]]
ytrainAND = [0, 0, 0, 1]
xtestAND = [[1., 0.], [1., 1.], [0., 1.],[0., 0.]]
ytestAND = [0, 1, 0, 0]
clf = MLPClassifier(hidden_layer_sizes=(), activation='identity',
                    solver='lbfgs').fit(xtrainAND, ytrainAND)
print("Score obtenu sur les données de test pour l'apprentissage de l'operator AND: ",
      clf.score(xtestAND, ytestAND))

#Question 3
xtrainOR = [[0., 0.], [0., 1.], [1., 0.],[1., 1.]]
ytrainOR = [0, 1, 1, 1]
xtestOR = [[1., 1.], [1., 1.], [0., 1.],[0., 0.]]
ytestOR = [1, 1, 1, 0]
clf = MLPClassifier(hidden_layer_sizes=(), activation='identity',
                    solver='lbfgs').fit(xtrainOR, ytrainOR)
print("Score obtenu sur les données de test pour l'apprentissage de l'operator OR: ", 
      clf.score(xtestOR, ytestOR))

#Question 4a
xtrainXOR = [[0., 0.], [0., 1.], [1., 0.],[1., 1.]]
ytrainXOR = [0, 1, 1, 0]
xtestXOR = [[1., 1.], [1., 0.], [0., 1.],[0., 0.]]
ytestXOR = [0, 1, 1, 0]
clf = MLPClassifier(hidden_layer_sizes=(), activation='identity',
                    solver='lbfgs').fit(xtrainXOR, ytrainXOR)
print("Score obtenu sur les données de test pour l'apprentissage de l'operator XOR sans couche cachée: ", 
      clf.score(xtestXOR, ytestXOR))

#Question 4b
clf_2l = MLPClassifier(hidden_layer_sizes=(4,2), activation='identity',
                    solver='lbfgs').fit(xtrainXOR, ytrainXOR)
print("Score obtenu sur les données de test pour l'apprentissage de l'operator XOR avec les 2 couches cachées: ", 
      clf_2l.score(xtestXOR, ytestXOR))

#Question 4c
clf_hyp = MLPClassifier(hidden_layer_sizes=(4,2), activation='tanh',
                    solver='lbfgs').fit(xtrainXOR, ytrainXOR)
print("Score obtenu avec la fonction d'activation tanh: ", 
      clf_hyp.score(xtestXOR, ytestXOR))

#Question 5

dataset = load_digits()
X = dataset.data # Entrees
Y = dataset.target # Resultats attendus
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
clf5 = MLPClassifier(hidden_layer_sizes=(4,2), activation='relu',
                    solver='lbfgs').fit(X_train, Y_train)

activ = ['identity', 'logistic', 'tanh', 'relu']
solve = ['lbfgs', 'sgd', 'adam']
layers = [(),(4,2),(3,3,2)]
index = []

for j in range(len(solve)) :
    for i in range(len(layers)) :
        index.append((solve[j],layers[i]))

a = []
M = pd.DataFrame(columns = activ)

for j in range(len(activ)) :
    for i in range(len(solve)) : 
        for k in range(len(layers)) :
            classifier = MLPClassifier(hidden_layer_sizes=layers[k],
                                      activation=activ[j],
                                      solver=solve[i])
            classifier.fit(X_train,Y_train)
            Y_pred = classifier.predict(X_test)
            a.append(accuracy_score(Y_test, Y_pred))     
    M.iloc[:,j] = a
    a = []


classifier = MLPClassifier(hidden_layer_sizes=(),
                                      activation='relu',
                                      solver='lbfgs')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred)
classifier.score(X_test,Y_test)

M.columns = activ
M.index = index

M.idxmax()




#Question 6 

#svm
# classifier_svm_poly = SVC(kernel = "poly")
# classifier_svm_lin = SVC(kernel ="linear")

# scores_lin = cross_val_score(classifier_svm_lin, X_test, Y_test, cv = RepeatedKFold(
#     n_splits=5, n_repeats=1000))
# scores_poly = cross_val_score(classifier_svm_poly, X_test, Y_test, cv = RepeatedKFold(
#     n_splits=5, n_repeats=1000))

# print("Cross-validation: {}",scores_lin.mean())
# print("Cross-validation: {}",scores_poly.mean())



parameters = {'kernel': ['linear']}
model_svm = SVC()
svm_grid = GridSearchCV(model_svm, parameters, n_jobs=-1, cv = RepeatedKFold(
    n_splits=5, n_repeats=100))
svm_grid.fit(X_train, Y_train)
print(svm_grid.best_params_)
print('Score : %s' % svm_grid.score(X_test, Y_test))


parameters2 = {'kernel': ['poly']}
svm_grid2 = GridSearchCV(model_svm, parameters2, n_jobs=-1, cv = RepeatedKFold(
    n_splits=5, n_repeats=100))
svm_grid2.fit(X_train, Y_train)

print('Score : %s' % svm_grid2.score(X_test, Y_test))



