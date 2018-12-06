# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

from math import log2
import numpy as np
import pydotplus as pdp
from sklearn.datasets import load_iris
from sklearn import tree
import collections
import cv2
from vin_ml.models import id3

TRAIN_FILE = "train2.csv"
TEST_FILE = "train2.csv"
# whether to ignore first id column in decision tree classifier
IS_ID_COLUMN_PRESENT = 1
DELIMETER = ','
# Noise in data set (csv file)
NOISE = [' ', ',', '\t', '\n', '\"']
# 2D data to be trained, each row is signle data instance (with its feature values)
DATA = []
# list of feature names
FEATURES = []
# 2D array, each row is query for decision tree
QUERIES = []
# dictionary with keys as feature names and value as index of that feature in 2D data array
FEATURE_INDEX = {}
# dictionary with keys as feature names and value as list of values which that feature can take
LEVELS = {}
# Target Feature Name
TARGET = ""
MOST_FREQUENT_TARGET_VALUE = ""

def print_query(query):
	print("\nQUERY-")
	for feature in query:
		print('\t', feature, ":", query[feature])

def get_data():
	global DATA, LEVELS, FEATURES, FEATURE_INDEX, TARGET, MOST_FREQUENT_TARGET_VALUE
	data = open(TRAIN_FILE, "r")
	count = {}
	line_num = 1
	maximum_times = 0
	for line in data.readlines():
		for noise in NOISE:
			if noise != DELIMETER:
				line = line.replace(noise, '')
		row = line.split(DELIMETER)[IS_ID_COLUMN_PRESENT:]
		if line_num == 1:
			FEATURES = row[:-1]
			TARGET = row[-1]
			FEATURE_INDEX = {feature:index for index, feature in enumerate(row)}
			LEVELS = {feature:[] for feature in row}
		else:
			DATA.append(row)
			if row[-1] not in LEVELS[TARGET]:
				count[row[-1]] = 0
				LEVELS[TARGET].append(row[-1])
			else:
				count[row[-1]] += 1
			for index, feature_value in enumerate(row[:-1]):
				if feature_value not in LEVELS[FEATURES[index]]:
					LEVELS[FEATURES[index]].append(feature_value)
			if count[row[FEATURE_INDEX[TARGET]]] > maximum_times:
				maximum_times = count[row[FEATURE_INDEX[TARGET]]]
				MOST_FREQUENT_TARGET_VALUE = row[FEATURE_INDEX[TARGET]]
		line_num += 1
	data = open(TEST_FILE, "r")
	line_num = 1
	features = []
	for line in data.readlines():
		for noise in NOISE:
			if noise != DELIMETER:
				line = line.replace(noise, '')
		row = line.split(DELIMETER)[IS_ID_COLUMN_PRESENT:]
		if line_num == 1:
			features = row
		else:
			query = {}
			for index, feature_value in enumerate(row):
				query[features[index]] = feature_value
			QUERIES.append(query)
		line_num += 1

classification_error_samples = 0

def process_query(tree, query):
	target_value = tree.predict(query)
	print_query(query)
	print("Predicted Target Value:", target_value, '\n')
	if target_value != query[TARGET]:
		classification_error_samples += 1
	plot_categorical_decision_tree(query)

def plot_categorical_decision_tree(query=None):
	graph = pdp.Dot(graph_type='digraph')
	DECISION_TREE.plot(graph, query=query)
	graph.write_png('tree.png')
	img = cv2.imread('tree.png')
	cv2.imshow("Decision Tree", img)
	cv2.waitKey(0)

print("\nID3")
get_data()
DECISION_TREE = id3.model(DATA, FEATURE_INDEX, TARGET, LEVELS, FEATURES, MOST_FREQUENT_TARGET_VALUE).build()
plot_categorical_decision_tree()
for query in QUERIES:
	process_query(DECISION_TREE, query)
accuracy = (len(QUERIES) - classification_error_samples) / len(QUERIES) * 100
classification_error = classification_error_samples / len(QUERIES) * 100
print("Accuracy %.2f%%" % accuracy)
print("Classification Error %.2f%%" % classification_error)