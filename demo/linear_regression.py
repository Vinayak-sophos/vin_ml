# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from numpy import dot, multiply, append, array, matrix, ones, zeros, eye
from math import floor
from vin_ml.models import linear_regression

plt.ion()

# data bounds
LOW = -1000
HIGH = 1000

MIN_SLOPE = -1
MAX_SLOPE = 1

# delta from line to be predicted (noise)
DELTA = 300
# number of data points
N = 50
# regularization parameter
L = 10

TRAINING_RATIO = 80 / 100
TRAINING_SIZE = floor(TRAINING_RATIO * N)

X = matrix([])
Y = matrix([])
x_vals = np.arange(LOW, HIGH, 0.1)
title = ''

# return slope, intercept
def generate_random_line():
	return np.random.uniform(MIN_SLOPE, MAX_SLOPE), np.random.uniform(LOW + HIGH / 2, HIGH - HIGH / 2)

def generate_random_data():
	global title
	# y = m * x + c
	(m, c) = generate_random_line()
	title = 'Linear and Plynomial Regression\nSample Data generated around y = (%f) + (%f) * x' % (c, m)
	plt.title(title)
	count = 0
	coords = np.empty((0, 2), int)
	while count < N:
		coord = np.random.randint(low = LOW, high = HIGH, size = 2)
		# normalize toward line
		x = coord[0]
		y = m * x + c
		coord[1] = y - DELTA + (coord[1] - LOW) / (HIGH - LOW) * 2 * DELTA
		coords = append(coords, [coord], axis = 0)
		count += 1
	X = append(ones(shape = (N, 1)), array([coords[:, 0]]).transpose(), axis = 1)
	Y = array([coords[:, 1]]).transpose()
	plot_points(X, Y, 'red', 'x')
	return X, Y

def plot_points(X, Y, color, marker):
	for i, x in enumerate(X):
		plt.scatter(x.item(1), Y.item(i), color=color, marker=marker)
	plt.draw()
	plt.pause(0.01)

def plot_seperator(W, Wr):
	global x_vals
	deg = len(W) - 1
	y_vals = zeros(len(x_vals))
	yr_vals = zeros(len(x_vals))
	while deg >= 0:
		y_vals += W.item(deg) * np.power(x_vals, deg)
		yr_vals += Wr.item(deg) * np.power(x_vals, deg)
		deg -= 1
	axes = plt.gca()
	if len(axes.lines): axes.lines[-1].remove()
	if len(axes.lines): axes.lines[-1].remove()
	plt.plot(x_vals, y_vals, color = 'blue', label='Y=W\'X')
	plt.plot(x_vals, yr_vals, color = 'green', label='regularized Y=W\'X')
	plt.pause(0.01)
	plt.legend()

def plot_error_graph(E_arr_in, Er_arr_in, E_arr_out, Er_arr_out):
	plt.figure()
	plt.title('Cost vs Degree Graph')
	plt.xlabel('Degree')
	plt.ylabel('Cost')
	Xs = np.arange(1, len(E_arr_in) + 1)
	plt.plot(Xs, E_arr_in, 'blue', marker='+', label='Training Error')
	plt.plot(Xs, Er_arr_in, 'green', marker='+', label='Training Regularized Error')
	plt.plot(Xs, E_arr_out, 'red', marker='+', label='Testing Error')
	plt.plot(Xs, Er_arr_out, 'brown', marker='+', label='Testing Regularized Error')
	plt.legend()
	plt.pause(0.01)

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

# update Matrix X, add next higher order term, permute data if required
def update_X(X, Y):
	initial_col = array(X[:, 1])
	curr_last_col = array(X[:, -1])
	new_col = multiply(initial_col, curr_last_col)
	X = append(X, array([new_col]).transpose(), axis = 1)
	# X, Y = permute(X, Y)
	return X, Y

def print_training_data(X, Y, W, Wr, E, Er):
	print('Training Data')
	print('X = ', X, '\n')
	print('Y = ', Y, '\n')
	print('W = ', W, '\n')
	print('Wr = ', Wr, '\n')
	print('Ein = ', E, '\n')
	print('Erout = ', Er, '\n')
	print('h(x) = ', end = '')
	for i, w in enumerate(W):
		print('(%.3g) * x ^ %d' % (w, i), end = '')
		if (i != len(W) - 1): print(' + ' , end = '')
		else: print('\n')
	print('hr(x) = ', end = '')
	for i, w in enumerate(Wr):
		print('(%.3g) * x ^ %d' % (w, i), end = '')
		if (i != len(Wr) - 1): print(' + ' , end = '')
		else: print('\n')

def print_testing_data(E, Er):
	print('Testing Error\n')
	print('Eout = ', E, '\n')
	print('Erout = ', Er, '\n')

def train(lin_reg, lin_reg_l, X, Y):
	global E_arr_in, Er_arr_in
	W = lin_reg.fit(X, Y)
	Wr = lin_reg_l.fit(X, Y)
	E = lin_reg.loss(X, Y)
	Er = lin_reg_l.loss(X, Y)
	E_arr_in = append(E_arr_in, E)
	Er_arr_in = append(Er_arr_in, Er)
	print_training_data(X, Y, W, Wr, E, Er)
	plot_points(X, Y, 'red', 'x')
	plot_seperator(W, Wr)

def test(lin_reg, lin_reg_l, X, Y):
	global E_arr_out, Er_arr_out
	E = lin_reg.loss(X, Y)
	Er = lin_reg_l.loss(X, Y)
	E_arr_out = append(E_arr_out, E)
	Er_arr_out = append(Er_arr_out, Er)
	print_testing_data(E, Er)
	plot_points(X, Y, 'purple', '+')

def setup_plot():
	plt.clf()
	plt.xlim(LOW, HIGH)
	plt.ylim(LOW, HIGH)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title(title)

def action(lin_reg, lin_reg_l, X, Y):
	setup_plot()
	Xin = X[:TRAINING_SIZE, :]
	Xout = X[TRAINING_SIZE:, :]
	Yin = Y[:TRAINING_SIZE, :]
	Yout = Y[TRAINING_SIZE:, :]
	train(lin_reg, lin_reg_l, Xin, Yin)
	test(lin_reg, lin_reg_l, Xout, Yout)
	return update_X(X, Y)

lin_reg = linear_regression.model()
# regularized model
lin_reg_l = linear_regression.model(regularization=L)
setup_plot()
X, Y = generate_random_data()
deg = 1
E_arr_in = np.array([])
Er_arr_in = np.array([])
E_arr_out = np.array([])
Er_arr_out = np.array([])
while True:
	if input('Want to Plot Curve of degree %d [Y/n] ' % deg) not in ['Y', 'y', '']:
		break
	X, Y = action(lin_reg, lin_reg_l, X, Y)
	deg += 1
plot_error_graph(E_arr_in, Er_arr_in, E_arr_out, Er_arr_out)
plt.pause(10000)