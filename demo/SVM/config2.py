# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import numpy as np
import vin_ml.kernals as ker

def kernal(x, y):
	return ker.linear(x, y)

def load_data():
	return (np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), \
	np.array([1, 1, -1, -1])), \
	(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), \
	np.array([1, 1, -1, -1])), \
	(0, 0.5)