# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

data = open("test.txt", "r")

arr = []
features = open("features.txt", "w")

# filter only images of 1 and 4
# directly use normalized image intensities as features
for line in data.readlines():
	if line[0] not in {'1', '4'}:
		continue
	features.write(line)