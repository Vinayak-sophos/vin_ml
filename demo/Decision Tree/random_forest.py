# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

from math import log2, inf
import numpy as np
import pydotplus as pdp
from sklearn.datasets import load_iris
from sklearn import tree
import collections
import matplotlib.pyplot as plt
import cv2
import itertools
import operator
from vin_ml.models import random_forest

plt.ion()

TRAIN_FILE = "train2.csv"
TEST_FILE = "train2.csv"
# whether to ignore first id column in decision tree classifier
IS_ID_COLUMN_PRESENT = 1
DELIMETER = ','
# Noise in data set (csv file)
NOISE = [' ', ',', '\t', '\n', '\"']
# 2D data to be trained, each row is signle data instance (with its feature values)
DATA = []
# list of DECISION TREES (each with some subset of data, or some subset of features, or both)
FOREST = []
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
PROPORTION = 0.75
# number of trees to be shown in a single row in the plot
TREES_IN_A_ROW = 2

def get_query_string(query):
	str = ""
	for feature in query:
		str += '    ' + feature + " : " + query[feature] + "\n"
	return str

def print_query(query, index):
	print("\nQUERY ON TREE %d-\n" % (index + 1))
	print(get_query_string(query))

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

# find most freuently occuring value in the array
def most_common(L):
  return max(set(L), key=list(L).count)

classification_error_samples = 0

def process_query(tree, query, index):
	global classification_error_samples
	target_value = tree.predict(query)
	print_query(query, index)
	print("Predicted Target Value : " + target_value, '\n')
	plot_categorical_decision_tree(index, query)
	return target_value

def show_forest(query=None, predicted_target=None):
	plt.figure(fig.number)
	plt.clf()
	for index in range(len(DECISION_TREE)):
		plt.subplot((len(DECISION_TREE) + 1 + (TREES_IN_A_ROW - 1)) // TREES_IN_A_ROW, TREES_IN_A_ROW, index + 1)
		plt.imshow(cv2.imread("tree%d.png" % index))
		plt.xticks([])
		plt.yticks([])
		plt.pause(1e-100)
	if query:
		plt.subplot((len(DECISION_TREE) + 1 + (TREES_IN_A_ROW - 1)) // TREES_IN_A_ROW, TREES_IN_A_ROW, len(DECISION_TREE) + 1)
		plt.text(0, 0.2, get_query_string(query), fontsize=15)
		plt.text(0, 0.1, "    Predicted Target : " + predicted_target, fontsize=15)
		plt.xticks([])
		plt.yticks([])
		plt.pause(1e-100)
	plt.tight_layout()
	plt.pause(1e-100)
	input("Press Enter to continue...")

def plot_categorical_decision_tree(index, query=None):
	graph = pdp.Dot(graph_type='digraph')
	DECISION_TREE[index].plot(graph, query=query, root=DECISION_TREE[index])
	graph.write_png('tree%d.png' % index)

# create multile subsets of randomly generated data
def convert_data_for_random_forest():
	global DATA, FEATURES
	total_features = len(FEATURES)
	number_of_trees = (total_features * (total_features - 1)) // 2
	DATA = np.array(DATA)
	for combination in itertools.combinations(range(len(FEATURES)), int(PROPORTION * len(FEATURES))):
		data = np.empty((len(DATA), 0))
		features = np.empty(0)
		SINGLE_DATA = {}
		for index in combination:
			data = np.append(data, DATA[:, index].reshape((len(DATA), 1)), axis = 1)
			features = np.append(features, FEATURES[index])
		data = np.append(data, DATA[:, -1].reshape((len(DATA), 1)), axis = 1)
		feature_index = {feature:index for index, feature in enumerate(features)}
		feature_index[TARGET] = len(features)
		most_frequent_target_value = most_common(data[:, -1].flatten())
		SINGLE_DATA["DATA"] = data
		SINGLE_DATA["TARGET"] = TARGET
		SINGLE_DATA["FEATURES"] = features
		SINGLE_DATA["FEATURE_INDEX"] = feature_index
		SINGLE_DATA["LEVELS"] = LEVELS
		SINGLE_DATA["MOST_FREQUENT_TARGET_VALUE"] = most_frequent_target_value
		FOREST.append(SINGLE_DATA)


print("\nRandom Forest")
get_data()
convert_data_for_random_forest()
fig = plt.figure("Random Forest")
DECISION_TREE = []
for SINGLE_DATA in FOREST:
	DECISION_TREE.append(random_forest.model(SINGLE_DATA["DATA"], SINGLE_DATA["FEATURE_INDEX"], SINGLE_DATA["TARGET"], SINGLE_DATA["LEVELS"], SINGLE_DATA["FEATURES"], SINGLE_DATA["MOST_FREQUENT_TARGET_VALUE"]).build())
for index in range(len(FOREST)):
	plot_categorical_decision_tree(index)
show_forest()
classification_error_samples = 0
for query in QUERIES:
	# voting of queries output
	vote = {level:0 for level in LEVELS[TARGET]}
	for index in range(len(FOREST)):
		vote[process_query(DECISION_TREE[index], query, index)] += 1
	# majority voting
	predicted_target = max(vote.items(), key=operator.itemgetter(1))[0]
	if predicted_target != query[TARGET]:
		classification_error_samples += 1
	show_forest(query, predicted_target)
accuracy = (len(QUERIES) - classification_error_samples) / len(QUERIES) * 100
classification_error = classification_error_samples / len(QUERIES) * 100
print("Accuracy %.2f%%" % accuracy)
print("Classification Error %.2f%%" % classification_error)