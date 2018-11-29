import pylab as pb
import numpy as np
from math import pi
from scipy . spatial . distance import cdist
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as st

# To sample from a multivariate Gaussian
# f = np. random . multivariate_normal (mu ,K);
# # To compute a distance matrix between two sets of vectors
# D = cdist (x1 ,x2)
# # To compute the exponetial of all elements in a matrix
# E = np.exp(D)

# Generate datapoints
w0 = -1.5
w1 = -0.5
sig = 0.3

xi = np.arange(-2, 2.2, 0.2)
eps = np.random.normal(0, sig, len(xi))
ti = w0*xi+w1+eps

length = ti.size
X = np.arange(-2, 2.2, 0.2)


K = np.empty((length, length))
for i in range(length):
	for j in range(length):
		K[i, j] = np.dot(X[i], X[j])


priors = st.multivariate_normal.rvs(mean=[0] * length, cov=K, size=1000)

for i in range(5):
	plt.plot(X, priors[i])


plt.show()





#pick 1 variable

rand = np.random.randint(0, len(ti))
pick = [xi[rand],ti[rand]]

print(pick)
