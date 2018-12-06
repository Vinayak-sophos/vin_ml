# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import numpy as np
import vin_ml.kernals as ker

DATA_SIZE = 20
# np.random.seed(2)

def kernal(x, y):
	return ker.polynomial(x, y, 10)

def load_data():
	X = np.empty((0, 2), int)
	Y = np.empty((0), int)
	for i in range(DATA_SIZE):
		x = np.random.random(2)
		if np.random.randint(10000) % 2 == 0:
			y = 1
		else: y = -1
		X = np.append(X, [x], axis=0)
		Y = np.append(Y, [y], axis=0)
	return (X[:int(0.80 * len(X))], Y[:int(0.80 * len(Y))]), (X[int(0.80 * len(X)):], Y[int(0.80 * len(Y)):]), (-1, -1)