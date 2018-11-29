import pylab as pb
import numpy as np
from math import pi
from scipy . spatial . distance import cdist



# To sample from a multivariate Gaussian
f = np. random . multivariate_normal (mu ,K);
# To compute a distance matrix between two sets of vectors
D = cdist (x1 ,x2)
# To compute the exponetial of all elements in a matrix
E = np.exp(D)