# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

# write your config file change 'config' in next line to name of your config file
# sample cofig file is attached with this file
import config as cf
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import numpy as np
from vin_ml.models import NN

plt.ion()
# np.random.seed(5)

epochs = 25
bias = True
# W = [np.array([[0.1, 0.8], [0.4, 0.6]]), np.array([[0.3, 0.9]])]

plt.ylabel('Error')
plt.xlabel('Epoch')
plt.xlim((0, epochs))
plt.ylim((0, cf.max_error))

# list of dirctories to save images for analysis
directories = ['train/input', 'train/output/correct', 'train/output/incorrect', 'test/input', 'test/output/correct', 'test/output/incorrect']
for directory in directories:
	if not os.path.exists(directory):
		os.makedirs(directory)
	else:
		shutil.rmtree(directory)
		os.makedirs(directory)

def plot_error(epoch, train_error, test_error, plt):
	plt.scatter(x=epoch, y=train_error, color='blue', marker='.', label='Training Error')
	plt.scatter(x=epoch, y=test_error, color='red', marker='.', label='Testing Error')
	if epoch == 0:
		plt.legend()
		plt.title('Error Plot Neural Networks')
	plt.show()
	plt.pause(1e-40)

def mse(Y, y):
	return np.sum((Y - y) * (Y - y)) / 2

def print_images(NN, X_train, Y_train, X_test, Y_test):
	cnt = 1
	for X, Y in zip(X_train, Y_train):
		y = NN.forward(X)
		X = np.array([int((ele + 1) / 2 * 255) for ele in X]).reshape(16, 16)
		cv2.imwrite("train/input/img_%d_%d.jpg" % (np.argmax(Y), cnt), X)
		if np.argmax(Y) == np.argmax(y):
			cv2.imwrite("train/output/correct/img_%d_%d.jpg" % (np.argmax(y), cnt), X)
		else:
			cv2.imwrite("train/output/incorrect/img_%d_%d.jpg" % (np.argmax(y), cnt), X)
		cnt += 1
	cnt = 1
	for X, Y in zip(X_test, Y_test):
		y = NN.forward(X)
		X = np.array([int((ele + 1) / 2 * 255) for ele in X]).reshape(16, 16)
		cv2.imwrite("test/input/img_%d_%d.jpg" % (np.argmax(Y), cnt), X)
		if np.argmax(Y) == np.argmax(y):
			cv2.imwrite("test/output/correct/img_%d_%d.jpg" % (np.argmax(y), cnt), X)
		else:
			cv2.imwrite("test/output/incorrect/img_%d_%d.jpg" % (np.argmax(y), cnt), X)
		cnt += 1

def train(NN, X_train, Y_train, X_test, Y_test):
	plt.plot(5, 0.5)
	gradient = 0
	error = 0
	for epoch in range(epochs):
		error = 0
		for X, Y in zip(X_train, Y_train):
			y = NN.forward(X)
			NN.backward(Y, y)
			error += mse(Y, y)
		error /= len(X_train)
		_, test_loss = test(NN, X_test, Y_test)
		plot_error(epoch, error, test_loss, plt)
	print("Training\nAccuracy: %f Loss: %f " % test(NN, X_train, Y_train))
	print_images(NN, X_train, Y_train, X_test, Y_test)

def test(NN, X_test, Y_test):
	loss = 0
	correct = 0
	for X, Y in zip(X_test, Y_test):
		y = NN.forward(X)
		loss += mse(Y, y)
		if cf.configuration == 1:
			idx = np.argmax(y.flatten())
			y = np.zeros((10, 1))
			y[idx, 0] = 1
		else:
			y = (y >= 0.5)
		if np.array_equal(Y, y):
			correct += 1
		else:
			pass
	return correct / len(X_test) * 100, loss / len(X_test)

if __name__ == '__main__':
	NN = NN.model(cf.number_of_layers, cf.lr, cf.layers, cf.activations, bias=bias)
	X_train, Y_train, X_test, Y_test = cf.load_data()
	train(NN, X_train, Y_train, X_test, Y_test)
	accuracy, loss = test(NN, X_test, Y_test)
	print("Testing")
	print("\tAccuracy: %f%% Loss %f" % (accuracy, loss))
	plt.pause(10000)