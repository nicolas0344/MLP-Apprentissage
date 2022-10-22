# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:01:00 2022

@author: Nicolas
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
 

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


