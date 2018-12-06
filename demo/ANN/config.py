# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

# configuration file for neaural network
import numpy as np

number_of_layers = 10
# number of neurons at each layer
layers = np.array([256, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10])
# learning rate
lr = 0.1
# activations to be used at each layer
activations = np.array(["sigmoid"] * (number_of_layers + 1))
max_error = 1.0
# configuration number
configuration = 1
PROPORTION = 0.80

def permute_data(X, Y):
	count = 0
	times = 1000
	while(count < times):
		(x, y) = np.random.randint(low = 0, high = len(X) - 1, size = 2)
		x_row = np.copy(X[x, :])
		X[x, :] = X[y, :]
		X[y, :] = x_row
		y_row = np.copy(Y[x, :])
		Y[x, :] = Y[y, :]
		Y[y, :] = y_row
		count += 1
	return X, Y

def load_data():
	X_train = []
	Y_train = []
	with open("test.txt") as file:
		lines = file.read().split('\n')
		for line in lines:
			if line == "": continue
			X_train.append([float(ele.strip()) for ele in line[1:].split()])
			Y = np.zeros(10)
			Y[int(line[0])] = 1
			Y_train.append(Y)
	X_train, Y_train = permute_data(np.array(X_train), np.array(Y_train))
	X_test = X_train[int(PROPORTION * len(X_train)):]
	Y_test = Y_train[int(PROPORTION * len(Y_train)):]
	X_train = X_train[:int(PROPORTION * len(X_train))]
	Y_train = Y_train[:int(PROPORTION * len(Y_train))]
	X_train = np.expand_dims(X_train, axis = -1)
	Y_train = np.expand_dims(Y_train, axis = -1)
	X_test = np.expand_dims(X_test, axis = -1)
	Y_test = np.expand_dims(Y_test, axis = -1)
	return X_train, Y_train, X_test, Y_test
