"""
A simple implementation of Poisson regression.
"""

import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap

T = 5000   # number of datapoints
N = 12


bins = np.linspace(0, 2*np.pi, 61)

alpha = np.digitize(np.cumsum(np.random.randn(T)*0.5)%(2*np.pi), bins)-1


x = np.linspace(-np.pi, np.pi, len(bins)-1)
tmp = np.exp(-x**2)
tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T


Y = np.random.poisson(tc[alpha]*10)


imap = Isomap(n_components=2, n_neighbors = 10).fit_transform(Y)





# create data
X = .3*np.random.randn(n, p)
true_b = np.random.randn(p)
y = np.random.poisson(np.exp(np.dot(X, true_b)))

# loss function and gradient
def f(b):
    Xb = np.dot(X, b)
    exp_Xb = np.exp(Xb)
    loss = exp_Xb.sum() - np.dot(y, Xb)
    grad = np.dot(X.T, exp_Xb - y)
    return loss, grad

# hessian
def hess(b):
    return np.dot(X.T, np.exp(np.dot(X, b))[:, None]*X)

# optimize
result = minimize(f, np.zeros(p), jac=True, hess=hess, method='newton-cg')

print('True regression coeffs: {}'.format(true_b))
print('Estimated regression coeffs: {}'.format(result.x))