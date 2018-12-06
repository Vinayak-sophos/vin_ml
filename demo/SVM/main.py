# Author: Vinayak Sachdeva
# Btech, MNIT Jaipur

# write your config file change 'config4' in next line to name of your config file
# sample cofig files are attached with this file
import config4 as cf
from vin_ml.models import svm
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
plt.xlim((-0.1, 1.1))
plt.ylim((-0.1, 1.1))

def plot_line(m, c):
	if (m == -1 and c == -1):
		return
	delta = 0.025
	x = np.arange(-0.1, 1.1, delta)
	y = m * x + c
	plt.plot(x, y, 'g--', label='Expected Boundary')
	plt.legend()
	plt.pause(1e-10)

(x_train, y_train), (x_test, y_test), (m, c) = cf.load_data()
if m != -1 and c != -1: plt.title('SVM Linear Classifier y = %.2fx + %.2f' % (m, c))
else: plt.title('SVM Non-Linear Classifier')
svm = svm.model(cf.kernal)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
accuracy = svm.evaluate(x_test, y_test)
print('Testing Accuracy: %f%%' % accuracy)
svm.plot_data(x_train, y_train, 'train')
svm.plot_data(x_test, y_test, 'test')
svm.plot_decision_boundary()
plot_line(m, c)
plt.pause(10000)