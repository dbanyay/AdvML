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
        K[i, j] = np.dot(X[i], X[j]) # linear kernel

# mu = np.zeros(length)
#
# # priors = np.random.multivariate_normal (mu ,K);
#
#
# priors = st.multivariate_normal.rvs(mu, K, length)
#
#
# for i in range(5):
#     plt.plot(X, priors[i])
# plt.show()

x_obs = np.array([X[1],X[2]])
y_obs = np.array([ti[1],ti[2]])


N_obs = 2
K_obs = np.empty((N_obs, N_obs))
for i in range(N_obs):
    for j in range(N_obs):
        K_obs[i,j] = np.dot(x_obs[i], x_obs[j])

K_obs_star = np.empty((N_obs, length))
for i in range(N_obs):
    for j in range(length):
        K_obs_star[i,j] = np.dot(x_obs[i], X[j])

post_mean = K_obs_star.T.dot(np.linalg.pinv(K_obs)).dot(y_obs)
post_var = K - K_obs_star.T.dot(np.linalg.pinv(K_obs)).dot(K_obs_star)

posteriors = st.multivariate_normal.rvs(mean=post_mean, cov=post_var, size = length)

# for i in range(length):
#     plt.plot(X, posteriors[i])

# plt.plot(x_obs, y_obs, 'ro')
# plt.show()
#
# plt.plot(X,ti)
# plt.show()

# use exponential kernel


sigma_f = 0.3

def exp_kernel(X,l,sigma_f):
    K_exp = np.empty((length, length))
    for i in range(length):
        for j in range(length):
            K_exp[i, j] = sigma_f ** 2 * np.exp(-(1.0 / (l ** 2)) * (X[i] - X[j]) ** 2)

    return K_exp

mu = np.zeros(length)

# priors = np.random.multivariate_normal (mu ,K);
priors_exp = st.multivariate_normal.rvs(mu, exp_kernel(X,1,0.3), length)


for i in range(5):
    plt.plot(X, priors_exp[i])
plt.show()