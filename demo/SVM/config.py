# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import numpy as np
import vin_ml.kernals as ker

m, c = 0, 0
# np.random.seed(6)

DATA_SIZE = 20

def kernal(x, y):
	return ker.linear(x, y)

def load_data():
	global m, c
	m, c = np.random.random(2)
	X = np.empty((0, 2), int)
	Y = np.empty((0), int)
	for i in range(DATA_SIZE):
		x = np.random.random(2)
		y = 1
		if x[0] * m - x[1] + c == 0:
			i -= 1
			continue
		elif x[0] * m - x[1] + c > 0: y = 1
		else: y = -1
		X = np.append(X, [x], axis=0)
		Y = np.append(Y, [y], axis=0)
	return (X[:int(0.80 * len(X))], Y[:int(0.80 * len(Y))]), (X[int(0.80 * len(X)):], Y[int(0.80 * len(Y)):]), (m, c)