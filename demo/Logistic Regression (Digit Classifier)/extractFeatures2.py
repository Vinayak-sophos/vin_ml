# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import numpy as np
from math import fabs

THRESHOLD = 0.08

data = open("test.txt", "r")

arr = []
features = open("features.txt", "w")

# filter only images of 1 and 4
# features --> Average Brightness and Average Boundaries
for line in data.readlines():
	if line[0] not in {'1', '4'}:
		continue
	features.write(line[0] + " ")
	img = np.array([float(ele) for ele in line[2:].split()]).reshape(16, 16)
	avg = np.average(img)
	numberOfrowsWithBounday = 0
	boundaries = 0
	for row in img:
		for ind in range(1, len(row) - 1):
			if (fabs(row[ind] - row[ind + 1]) > THRESHOLD):
				hasBoundary = True
				boundaries += 1
		if hasBoundary:
			numberOfrowsWithBounday += 1
	features.write("%f %f\n" % (avg, boundaries / numberOfrowsWithBounday))