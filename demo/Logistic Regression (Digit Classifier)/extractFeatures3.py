# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

import numpy as np
from math import fabs, inf, floor
import cv2

THRESHOLD = 0.08

data = open("test.txt", "r")

arr = []
features = open("features.txt", "w")

c = 0

# filter only images of 1 and 4
# features --> Average Brightness and Average Symmetry
for line in data.readlines():
	if line[0] not in {'1', '4'}:
		continue
	c += 1
	features.write(line[0] + " ")
	img1 = np.array([float(ele) for ele in line[2:].split()]).reshape(16, 16)
	img2 = np.fliplr(img1)
	avg = np.average(img1)
	avg2 = np.average(np.abs(img2 - img1))
	features.write("%f %f\n" % (avg, avg2))