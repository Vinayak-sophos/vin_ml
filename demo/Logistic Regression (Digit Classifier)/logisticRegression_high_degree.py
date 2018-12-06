# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, multiply, divide
from math import log, exp, inf, floor, ceil
import os
from scipy.optimize import fmin_cg, fmin
from vin_ml.models import logistic_regression

plt.ion()

# find features from given digit features file 'test.txt'
# type depending on which python script to use to extract features
# Types
# 1 --> Directy use 256 image points to train
# 2 --> Average Brightness and Average Boundaries
# 3 --> Average Brightness and Average Symmetry
FEATURE_TYPE = 3
PLOT_DATA = True
NORMALIZE_FEATURE = False
PERMUTE_DATA = True
TRAINING_RATIO = 80 / 100
MAX_DEG = 10

# extract features to file 'features.txt'
os.system("python extractFeatures%d.py" % FEATURE_TYPE)

def setup_plot(title, xlabel, ylabel, minx = -inf, maxx = inf, miny = -inf, maxy = inf):
	plt.clf()
	if minx != -inf: plt.xlim((minx, maxx))
	if miny != -inf: plt.ylim((miny, maxy))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

def permute(X, Y):
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

def normalizeFeature(X):
	for ind in range(len(X[0])):
		range_of_values = X[:, ind].max() - X[:, ind].min()
		if (range_of_values) < 0.5:
			continue
		X[:, ind] -= np.mean(X[:, ind])
		X[:, ind] = divide(X[:, ind], range_of_values)
	return X

def plot_data(Xin, Yin, Xout, Yout):
	global to_draw
	if not to_draw:
		return
	setup_plot('Logistic Regression', 'Average Brightness', 'Average Boundaries' if FEATURE_TYPE == 2 else 'Average Symmetry', -1, 0, floor((X[:,2]).min()) - 0.2, ceil((X[:,2]).max()) + 0.2)
	done_p, done_n = False, False
	for ind in range(len(Xin)):
		points = plt.scatter(Xin[ind][1], Xin[ind][2],
			color = 'red' if Yin[ind] == 1 else 'blue',
			marker = 'x',
			label='Training, y=+1' if not done_p and Yin[ind] == 1 else 'Training, y=-1' if not done_n and Yin[ind] == -1 else '')
		if Yin[ind] == 1: done_p = True
		else: done_n = True
	done_p, done_n = False, False
	for ind in range(len(Xout)):
		plt.scatter(Xout[ind][1], Xout[ind][2],
			color = 'purple' if Yout[ind] == 1 else 'green',
			marker = '+',
			label='Testing, y=+1' if not done_p and Yout[ind] == 1 else 'Testing, y=-1' if not done_n and Yout[ind] == -1 else '')
		if Yout[ind] == 1: done_p = True
		else: done_n = True
	plt.draw()
	plt.pause(1e-40)
	plt.legend()

# evaluate final equation of classifier (W) to plot non-linear decision boundary, using contours
def equation(X, Y, W):
	global power_set
	val = np.zeros(X.shape)
	# summation of X^a * Y^b, iterating over all possible values of a and b
	for i, mat in enumerate(power_set):
		val += W.item(i) * (X ** mat[0]) * (Y ** mat[1])
	return val

cont = 0

def plot_separator(W):
	global to_draw, cont
	if not to_draw:
		return
	axes = plt.gca()
	for line in axes.lines:
		line.remove()
	delta = 0.001
	xrange = np.arange(-1.0, 0.0, delta)
	yrange = np.arange(0.0, 8.0, delta)
	X, Y = np.meshgrid(xrange,yrange)
	if cont: cont.collections[0].remove()
	cont = plt.contour(X, Y, equation(X, Y, W), [0], colors='brown')
	cont.collections[0].set_label('Predicted Classifier')
	plt.draw()
	plt.pause(1e-40)
	plt.legend()

def plot_errors():
	global E_in, E_out
	plt.figure()
	setup_plot('Cost vs Degree', 'Degree', 'Cost')
	x_vals = np.arange(1, len(E_in) + 1)
	plt.plot(x_vals, E_in, color='blue', marker = '+', label='Training Cost')
	plt.plot(x_vals, E_out, color='red', marker = '+', label='Testing Cost')
	plt.legend()

def get_data():
	global NORMALIZE_FEATURE, PERMUTE_DATA
	X = np.array([])
	Y = np.array([])
	num = 0
	data = open("features.txt", "r")
	for line in data.readlines():
		y = [1 if int(line[0]) == 1 else -1]
		x = np.append([1], [float(ele) for ele in line[2:].split()])
		if not X.size:
			X = np.empty((0, len(x)))
			Y = np.empty((0, 1))
		X = np.append(X, [x], axis = 0)
		Y = np.append(Y, [y], axis = 0)
		num += 1
	if PERMUTE_DATA:
		(X, Y) = permute(X, Y)
	if NORMALIZE_FEATURE:
		X = normalizeFeature(X)
	return (X, Y)

def train(log_reg, X, Y):
	global E_in
	W = log_reg.fit(X, Y)
	E = log_reg.loss(X, Y)
	Accuracy = log_reg.evaluate(X, Y)
	print("Training Cost: %f" % E)
	print("Training Accuracy: %f%%" % Accuracy)
	print("Training Error: %f%%" % (100 - Accuracy))
	print()
	E_in.append(E)
	return W

def test(log_reg, X, Y):
	global E_out
	E = log_reg.loss(X, Y)
	Accuracy = log_reg.evaluate(X, Y)
	print("Testing Cost: %f" % E)
	print("Testing Accuracy: %f%%" % Accuracy)
	print("Testing Error: %f%%" % (100 - Accuracy))
	E_out.append(E)
	print()

# to efficiently check which combination of degrees are used
power_map = {}
# list of combination of degrees which are used
power_set = []
# increase degree and accordingly update X and Y (next possible terms of X^a * Y^b, with a + b <= required degree)
def update_data_sets(Xin, Xout):
	global initial_Xin, initial_Xout
	length = len(Xin[0])
	for i in range(len(initial_Xin[0])):
		for j in range(length):
			mat = power_set[j].copy()
			mat[i] += 1
			if str(mat) not in power_map:
				power_set.append(mat)
				power_map[str(mat)] = 1
				new_col = initial_Xin[:, i] * Xin[:, j]
				new_col = new_col.reshape(len(new_col), 1)
				Xin = np.append(Xin, new_col, axis = 1)
				new_col = initial_Xout[:, i] * Xout[:, j]
				new_col = new_col.reshape(len(new_col), 1)
				Xout = np.append(Xout, new_col, axis = 1)
	return (Xin, Xout)

log_reg = logistic_regression.model()

X, Y = get_data()
to_draw = ((len(X[0]) == 3) & PLOT_DATA)
TRAINING_SIZE = floor(TRAINING_RATIO * len(X))
Xin = X[:TRAINING_SIZE, :]
Xout = X[TRAINING_SIZE:, :]
Yin = Y[:TRAINING_SIZE, :]
Yout = Y[TRAINING_SIZE:, :]
E_in = []
E_out = []

initial_Xin = Xin[:, 1:]
initial_Xout = Xout[:, 1:]

power_map[str(np.zeros(len(initial_Xin[0])))] = 1
power_set.append(np.zeros(len(initial_Xin[0])))

for i in range(len(initial_Xin[0])):
	mat = np.zeros(len(initial_Xin[0]))
	mat[i] = 1
	power_map[str(mat)] = 1
	power_set.append(mat)

plot_data(Xin, Yin, Xout, Yout)

deg = 1
while True:
	print('Degree:', deg, '\n')
	train(log_reg, Xin, Yin)
	test(log_reg, Xout, Yout)
	plot_separator(log_reg.W)
	if input('Do you want to move to next degree[Y/n] ') not in ['Y', 'y', '']: break
	print()
	Xin, Xout = update_data_sets(Xin, Xout)
	deg += 1
plot_errors()
plt.pause(1e-40)
input('\nPress Enter Key to Quit ')