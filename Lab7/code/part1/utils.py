"""
Learning on Sets - ALTEGRAD - Jan 2021
"""

import numpy as np


import random as rd
def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    X_train = np.zeros( n_train, max_train_card)
    cards = np.random.randint(1,10,  n_train)
    ############## Task 1
    for i in n_train:
        X_train[i,- cards[i]] = np.radom.randint(1,11, cards[i]) 
    y_train = np.sum( X_train, axis = 1)
    
    ##################
    # your code here #
    ##################
    return X_train, y_train


def create_test_dataset():
	
    ############## Task 2
    
    ##################
    # your code here #
    ##################
    X_test = list()
    y_test = list()
    for i in range((5, 101, 5)):
        X_test.append(np.random.randint((1,11, (10000, i))))
        y_test.append(np.sum(X_test[-1],axis=1))
        
    return X_test, y_test