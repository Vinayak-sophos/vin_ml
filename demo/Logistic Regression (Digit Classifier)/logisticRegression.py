# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, multiply, divide
from math import log, exp, inf, floor, ceil
import os
import shutil
import cv2
from vin_ml.models import logistic_regression
from vin_ml.utils import sigmoid

plt.ion()

# list of dirctories to save images for analysis
directories = ['input/1', 'input/4', 'output/trained/1', 'output/trained/4', 'output/tested/1', 'output/tested/4']
for directory in directories:
	if not os.path.exists(directory):
		os.makedirs(directory)
	else:
		shutil.rmtree(directory)
		os.makedirs(directory)

# find features from given digit features file 'test.txt'
# type depending on which python script to use to extract features
# Types
# 1 --> Directy use 256 image points to train
# 2 --> Average Brightness and Average Boundaries
# 3 --> Average Brightness and Average Symmetry
FEATURE_TYPE = 2
PLOT_COST = True
PLOT_DATA = True
NORMALIZE_FEATURE = False
PERMUTE_DATA = True
TRAINING_RATIO = 80 / 100

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

def plot_training_data(X, Y):
	global to_draw
	if not to_draw:
		return
	training_figure = plt.figure()
	setup_plot('Logistic Regression Training Set', 'Average Brightness', 'Average Boundaries' if FEATURE_TYPE == 2 else 'Average Symmetry', -1, 0, floor((X[:,2]).min()) - 0.2, ceil((X[:,2]).max()) + 0.2)
	done_p, done_n = False, False
	for ind in range(len(X)):
		points = plt.scatter(X[ind][1], X[ind][2],
			color = 'red' if Y[ind] == 1 else 'blue',
			marker = '+' if Y[ind] == 1 else 'x',
			label='y=+1' if not done_p and Y[ind] == 1 else 'y=-1' if not done_n and Y[ind] == -1 else '')
		if Y[ind] == 1: done_p = True
		if Y[ind] == -1: done_n = True
	plt.draw()
	plt.pause(1e-40)
	plt.legend()
	return training_figure

def plot_testing_data(X, Y):
	global to_draw
	if not to_draw:
		return
	testing_figure = plt.figure()
	setup_plot('Logistic Regression Testing Set', 'Average Brightness', 'Average Boundaries' if FEATURE_TYPE == 2 else 'Average Symmetry', -1, 0, floor((X[:,2]).min()) - 0.2, ceil((X[:,2]).max()) + 0.2)
	done_p, done_n = False, False
	for ind in range(len(X)):
		plt.scatter(X[ind][1], X[ind][2],
			color = 'red' if Y[ind] == 1 else 'blue',
			marker = '+' if Y[ind] == 1 else 'x',
			label='y=+1' if not done_p and Y[ind] == 1 else 'y=-1' if not done_n and Y[ind] == -1 else '')
		if Y[ind] == 1: done_p = True
		if Y[ind] == -1: done_n = True
	plt.draw()
	plt.pause(1e-40)
	plt.legend()
	return testing_figure

def plot_separator(W, figure):
	global to_draw
	if not to_draw:
		return
	plt.figure(figure.number)
	axes = plt.gca()
	x_vals = np.arange(-1, 0, 0.01)
	for line in axes.lines:
		line.remove()
	intercept = -W.item(0) / W.item(2)
	slope = -W.item(1) / W.item(2)
	plt.plot(x_vals, slope * x_vals + intercept, color='brown', label='Predicted Classifier')
	plt.draw()
	plt.pause(1e-40)
	plt.legend()

def plot_sigmoid(X, Y, W):
	plt.figure()
	setup_plot('Sigmoid Function', 'X', 'Probability(y=+1|x)')
	x_vals = dot(X, W).transpose()[0]
	y_vals = sigmoid(x_vals)
	done, done_p, done_n = False, False, False
	for ind, (x, y) in enumerate(zip(x_vals, y_vals)):
		plt.plot(x, y, color='yellow', marker='.', markersize=5, label='Sigmoid Value' if not done else '')
		if (Y.item(ind) == 1): plt.plot(x, 1, color='red', marker='+', label='y=+1' if not done_p else '')
		else: plt.plot(x, 0, color='purple', marker='x', label='y=-1' if not done_n else '')
		done = True
		if Y.item(ind) == 1: done_p = True
		else: done_n = True
	plt.draw()
	plt.pause(1e-40)
	plt.legend()

def write_img(x, path):
	if FEATURE_TYPE != 1:
		return
	img = np.array([floor((ele + 1) / 2 * 255) for ele in x]).reshape(16, 16)
	cv2.imwrite(path, img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256, 256))
	cv2.imwrite(path, img)

def get_data():
	global NORMALIZE_FEATURE, PERMUTE_DATA
	X = np.array([])
	Y = np.array([])
	num = 0
	data = open("features.txt", "r")
	for line in data.readlines():
		y = [1 if int(line[0]) == 1 else -1]
		x = np.append([1], [float(ele) for ele in line[2:].split()])
		if line[0] == '1': write_img(x[1:], 'input/1/%d_%d.jpg' % (num + 1, int(line[0])))
		else: write_img(x[1:], 'input/4/%d_%d.jpg' % (num + 1, int(line[0])))
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
	training_figure = plot_training_data(X, Y)
	W = log_reg.fit(X, Y)
	plot_separator(log_reg.W, training_figure)
	y_pred = log_reg.predict(X)
	print("Training Cost: %f" % log_reg.loss(X, Y))
	print("Training Accuracy: %f%%" % log_reg.evaluate(X, Y))
	print("Training Error: %f%%" % (100 - log_reg.evaluate(X, Y)))
	print()
	for i, x in enumerate(X):
		if y_pred.item(i) == 1: write_img(x[1:], 'output/trained/1/%d_%d.jpg' % (i + 1, 1 if Y.item(i) == 1 else 4))
		else: write_img(x[1:], 'output/trained/4/%d_%d.jpg' % (i + 1, 1 if Y.item(i) == 1 else 4))

def test(log_reg, X, Y):
	testing_figure = plot_testing_data(X, Y)
	plot_separator(log_reg.W, testing_figure)
	y_pred = log_reg.predict(X)
	print("Testing Cost: %f" % log_reg.loss(X, Y))
	print("Testing Accuracy: %f%%" % log_reg.evaluate(X, Y))
	print("Testing Error: %f%%" % (100 - log_reg.evaluate(X, Y)))
	for i, x in enumerate(X):
		if y_pred.item(i) == 1: write_img(x[1:], 'output/tested/1/%d_%d.jpg' % (i + 1, Y.item(i)))
		else: write_img(x[1:], 'output/tested/4/%d_%d.jpg' % (i + 1, Y.item(i)))
	print()

log_reg = logistic_regression.model()
X, Y = get_data()
to_draw = ((len(X[0]) == 3) & PLOT_DATA)
TRAINING_SIZE = floor(TRAINING_RATIO * len(X))
Xin = X[:TRAINING_SIZE, :]
Xout = X[TRAINING_SIZE:, :]
Yin = Y[:TRAINING_SIZE, :]
Yout = Y[TRAINING_SIZE:, :]
train(log_reg, Xin, Yin)
test(log_reg, Xout, Yout)
plot_sigmoid(X, Y, log_reg.W)
plt.pause(10000)