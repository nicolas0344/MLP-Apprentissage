# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 19:14:46 2022

@author: olivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

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
           a.append(classifier.score(X_test,Y_test))
    M[:,j] = a


classifier = MLPClassifier(hidden_layer_sizes=(),
                                      activation='identity',
                                      solver='sgd')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
classifier.score(X_test,Y_test)

M_results = pd.concat([pd.DataFrame(M[0]), pd.DataFrame(M[1]), 
                       pd.DataFrame(M[2])])
M_results.columns = activ
M_results.index = index