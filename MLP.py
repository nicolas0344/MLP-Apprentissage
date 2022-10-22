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
from sklearn.svm import SVC


#Question 1

# code pour generer le graphique XOR

X1 = np.array([0, 1, 0, 1])
X2 = np.array([0, 0, 1, 1])
groupXOR = np.array([0, 1, 1, 0])
cdict = {0: 'red', 1: 'green'}
n = ['(0,1)', '(1,1)', '(0,0)', '(1,0)']
loc = [(0.0275,0.98), (0.875,0.98), (0.0275,-0.015), (0.875,-0.015)]
fig, ax = plt.subplots()

for g in np.unique(groupXOR):
    ix = np.where(groupXOR == g)
    ax.scatter(X1[ix], X2[ix], c = cdict[g], label = g, s=170)

for i, txt, l in zip(range(4), n, loc):
    ax.annotate(txt, xy=(X1[i], X2[i]), xytext=l, fontsize = 13)

ax.legend(loc = 'center right')
plt.title('XOR (OU exclusif)')
plt.show()

 

# code pour generer le graphique AND

groupAND = np.array([0, 0, 0, 1])
fig, ax = plt.subplots()

for g in np.unique(groupAND):
    ix = np.where(groupAND == g)
    ax.scatter(X1[ix], X2[ix], c = cdict[g], label = g, s=170)

for i, txt, l in zip(range(4), n, loc):
    ax.annotate(txt, xy=(X1[i], X2[i]), xytext=l, fontsize = 13)

ax.legend(loc = 'center right')
plt.title('AND (ET)')
plt.show()
 

# code pour generer le graphique OR

groupOR = np.array([0, 1, 1, 1])
fig, ax = plt.subplots()

for g in np.unique(groupOR):
    ix = np.where(groupOR == g)
    ax.scatter(X1[ix], X2[ix], c = cdict[g], label = g, s=170)
    
for i, txt, l in zip(range(4), n, loc):
    ax.annotate(txt, xy=(X1[i], X2[i]), xytext=l, fontsize = 13)
    
ax.legend(loc = 'center right')
plt.title('OR (OU inclusif)')
plt.show()


classifier = MLPClassifier(hidden_layer_sizes=(),activation='identity',
                           solver='lbfgs')
#Question 2 

xtrainAND = [[0., 0.], [0., 1.], [1., 0.],[1., 1.]]
ytrainAND = [0, 0, 0, 1]

classifier.fit(xtrainAND, ytrainAND)

X_test_AND= [[1., 1.],[0.,0.],[1.,1.],[0.,1.]]
Y_test_AND = [1.,0.,1.,0.]
Y_predict_AND = classifier.predict(X_test_AND)

#Verification des résultats du classifieur
np.array(Y_test_AND) - Y_predict_AND
classifier.score(X_test_AND,Y_test_AND)


#Question 3
xtrainOR = [[0., 0.], [0., 1.], [1., 0.],[1., 1.]]
ytrainOR = [0, 1, 1, 1]

classifier.fit(xtrainOR, ytrainOR)

X_test_OR = [[1., 1.],[0.,0.],[0.,1.]]
Y_test_OR = [1.,0.,1.]
Y_predict_OR = classifier.predict(X_test_OR)

#Verification des résultats du classifieur
classifier.score(X_test_OR,Y_test_OR)


#Question 4
#a)

xtrainXOR = [[0., 0.], [0., 1.], [1., 0.],[1., 1.]]
ytrainXOR = [0, 1, 1, 0.]

classifier.fit(xtrainXOR, ytrainXOR)

X_test_XOR = [[1., 1.],[0.,0.],[0.,1.],[1.,0.]]
Y_test_XOR = [0.,0.,1.,1.]
Y_predict_XOR = classifier.predict(X_test_XOR)

classifier.score(X_test_XOR,Y_test_XOR)

#b)
classifier_XOR_id = MLPClassifier(hidden_layer_sizes=(4,2),activation='identity',
                           solver='lbfgs')

classifier_XOR_id.fit(xtrainXOR, ytrainXOR)
Y_predict_XOR_id = classifier_XOR_id.predict(X_test_XOR)

classifier_XOR_id.score(X_test_XOR,Y_test_XOR)
classifier_XOR_id.coefs_

#c)
classifier_XOR_tanh = MLPClassifier(hidden_layer_sizes=(4,2),activation='tanh',
                           solver='lbfgs')

classifier_XOR_tanh.fit(xtrainXOR, ytrainXOR)
Y_predict_XOR_tanh = classifier_XOR_tanh.predict(X_test_XOR)

classifier_XOR_tanh.score(X_test_XOR,Y_test_XOR)
classifier_XOR_tanh.coefs_


#Question 5 

dataset = load_digits()
X = dataset.data 
Y = dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


activ = ['identity', 'logistic', 'tanh', 'relu']
solve = ['lbfgs', 'sgd', 'adam']
layers = [(),(4,2),(3,3,2)]
index = []

for j in range(len(solve)) :
    for i in range(len(layers)) :
        index.append((solve[j],layers[i]))

a = np.zeros((3,4))
M = [a,a,a]

for j in range(len(activ)) :
    for i in range(len(solve)) : 
        for k in range(len(layers)) :
           classifier = MLPClassifier(hidden_layer_sizes=layers[k],
                                      activation=activ[j],
                                      solver=solve[i])
           classifier.fit(X_train,Y_train)
           Y_pred = classifier.predict(X_test)
           M[k][i,j] = classifier.score(X_test,Y_test)


classifier = MLPClassifier(hidden_layer_sizes=(),
                                      activation='identity',
                                      solver='sgd')
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred)


M_results = pd.concat([pd.DataFrame(M[0]), pd.DataFrame(M[1]), 
                       pd.DataFrame(M[2])])
M_results.columns = activ
M_results.index = index



#Question 6 

M_results.idxmax()

#svm
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
model_svm = SVC()
svm_grid = GridSearchCV(model_svm, parameters, n_jobs=-1, cv = 5)
svm_grid.fit(X_train, Y_train)
print('Score : %s' % svm_grid.score(X_test, Y_test))