# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

from vin_ml.models import perceptron
import matplotlib.pyplot as plt
import numpy as np

# randomly generate line (slpoe and intercept) to be predicted
line_slope = np.random.randint(low = 1, high = 10001)
line_slope = 0.5 + 2 * (line_slope - 1) / 10000
line_intercept = np.random.uniform(0, 1)

plt.ion()
plt.title('y=%.2fx+%.2f' % (line_slope, line_intercept))

X = np.empty((0, 2))
Y = np.empty(0)

per = perceptron.model(lr=100)
num = 0
done_p = False
done_n = False
TRAIN_SIZE = 100
TEST_SIZE = 100

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
line = plt.plot(x_vals, line_slope * x_vals + line_intercept, '--', color = 'green', label='Expected Classifier')

# data bounds
bounds = (1, 100)
plt.xlim(bounds[0], bounds[1])
plt.ylim(bounds[0], bounds[1])

# generate new random point and update W to adjust decision boundary Y = (W' . X)
while num < TRAIN_SIZE:
	x = np.random.randint(low = 1, high = 100, size = 2)
	cal = x[1] - line_slope * x[0] - line_intercept
	if not cal:
		continue
	else:
		category = 1 if cal > 0 else -1
		Y = np.append(Y, category)
		if category == 1:
			plt.scatter(x[0], x[1], color = 'red', marker = 'x', label='y=+1' if not done_p else '')
			done_p = True
		else:
			plt.scatter(x[0], x[1], color = 'blue', marker = '+', label='y=-1' if not done_n else '')
			done_n = True
		X = np.append(X, [x], axis = 0)
	W = per.fit(X, Y)
	if len(axes.lines) > 1: axes.lines[-1].remove()
	x_vals = np.array(axes.get_xlim())
	if W[1]:
		slope = -(W[0] / W[1])
		y_intercept = -(W[2] / W[1])
		line = plt.plot(x_vals, slope * x_vals + y_intercept, color = 'brown', label='Predicted Classifier')
	else:
		x_intercept = -(W[2] / W[0])
		line = plt.axvline(x = x_intercept, color = 'brown', label='Predicted Classifier')
	plt.draw()
	plt.pause(1e-40)
	plt.legend()
	num += 1

X = np.empty((0, 2))
Y = np.empty(0)

num = 0
while num < TEST_SIZE:
	x = np.random.randint(low = 1, high = 100, size = 2)
	cal = x[1] - line_slope * x[0] - line_intercept
	if not cal:
		continue
	else:
		category = 1 if cal > 0 else -1
		Y = np.append(Y, category)
		X = np.append(X, [x], axis = 0)
	num += 1

y_pred = per.predict(X)
print('Expected', Y)
print('Predicted', y_pred)
accuracy = per.evaluate(X, Y)
print('Testing Accuracy: %f%%' % accuracy)
plt.pause(10000)