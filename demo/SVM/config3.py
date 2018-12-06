# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import numpy as np
import vin_ml.kernals as ker

def kernal(x, y):
	return ker.polynomial(x, y, 10)

def load_data():
	return (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), \
	np.array([1, -1, -1, 1])), \
	(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), \
	np.array([1, -1, -1, 1])), (-1, -1)