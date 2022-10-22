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

 
#Question 2 

xtrainAND = [[0., 0.], [0., 1.], [1., 0.],[1., 1.]]
ytrainAND = [0, 0, 0, 1]
plt.plot(xtrainAND, ytrainAND)

classifier = MLPClassifier(hidden_layer_sizes=(),activation='identity',
                           solver='lbfgs')
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
plt.plot(xtrainOR, ytrainOR)

classifier = MLPClassifier(hidden_layer_sizes=(),activation='identity',
                           solver='lbfgs')
classifier.fit(xtrainOR, ytrainOR)

X_test_OR = [[1., 1.],[0.,0.]]
classifier.predict(X_test_OR)

#Verification des résultats du classifieur
np.array(ytrainOR) - classifier.predict(xtrainOR)


#Question 4



