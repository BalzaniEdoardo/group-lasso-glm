# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-06-02 12:56:57
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import PoissonRegressor


import neurostatslib as nsl
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis, MSplineBasis

# Jax import ##########################
import jax
import jax.numpy as jnp
from functools import partial
import jaxopt
from jax import grad


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


#######################################


T = 5000   # number of datapoints
N = 12


bins = np.linspace(0, 2*np.pi, 121)

alpha = np.digitize(
    gaussian_filter1d(np.cumsum(np.random.randn(T)*0.5), 3)
    %(2*np.pi), bins
    )-1


x = np.linspace(-np.pi, np.pi, len(bins)-1)
tmp = np.roll(np.exp(-(1*x)**2), (len(bins)-1)//2)
tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T

Y = np.random.poisson(tc[alpha]*3)

spike_data = Y.T

# imap = Isomap(n_components=2, n_neighbors = 10).fit_transform(Y)

#######################################
# Getting the spike basis function
ws = 100
nb = 8

spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=ws)
sim_pts = nsl.sample_points.raised_cosine_linear(nb, ws)
B = spike_basis.gen_basis_funcs(sim_pts)

# B = B[1:]

# B = B[::-1]


#####################################################################
# Fitting the GLM
#####################################################################

spike_basis = B

n_neurons, _ = spike_data.shape
n_basis_funcs, window_size = spike_basis.shape

# Convolve spikes with basis functions. We drop the last sample, as
# those are the features that could be used to predict spikes in the
# next time bin
# Broadcasted 1d convolution operations.
# [[n x t],[w]] -> [n x (t - w + 1)]
_CORR1 = jax.vmap(partial(jnp.convolve, mode='same'), (0, None), 0)
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_CORR2 = jax.vmap(_CORR1, (None, 0), 0)

X = _CORR2(jnp.atleast_2d(spike_basis),jnp.atleast_2d(spike_data))
# X = X[:,:,:-1]
# X = np.array(X)
X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
X = X.T
X = X - X.mean(0)
X = X / X.std(0)
X = np.hstack((X, jnp.ones((len(X), 1))))

b0 = jnp.zeros(((n_neurons*n_basis_funcs)+1, n_neurons))


def loss(b, X, Y):
    Xb = jnp.dot(X, b)
    exp_Xb = jnp.exp(Xb)
    loss = jnp.sum(exp_Xb, 0) - jnp.sum(Y*Xb, 0) # + jax.scipy.special.gammaln(Y+1)
    # penalty = jnp.sqrt(jnp.sum(jnp.maximum(b**2, 0), 0))
    return jnp.mean(loss) + 0.5*jnp.sum(jnp.abs(b))
    # return jnp.mean(loss) + 0.1*jnp.sum(jnp.sqrt(jnp.maximum(0, jnp.sum(b**2.0, 1))))

# def loss(b, X, Y):
#     Xb = jnp.dot(X, b)
#     P = jax.nn.softplus(Xb)
#     Xt = Y * jnp.log(P)
#     Xt = jnp.where(jnp.isnan(Xt), jnp.zeros_like(Xt), Xt)
#     return jnp.sum(P - Xt + jax.scipy.special.gammaln(Y+1))

def loss1(b, X, y):
    Xb = np.dot(X, b)
    exp_Xb = np.exp(Xb)
    loss = np.mean(exp_Xb, 0) - np.mean(y*Xb, 0) # + jax.scipy.special.gammaln(Y+1)
    grad = np.dot(X.T, exp_Xb - y)/y.shape[0] + b
    return loss + 0.5 * 1.0*np.sum(np.power(b, 2)), grad
    # return jnp.mean(loss) + 0.1*jnp.sum(jnp.sqrt(jnp.maximum(0, jnp.sum(b**2.0, 1))))

# solver = jaxopt.GradientDescent(
#     fun=loss, maxiter=20000, acceleration=False, verbose=True, stepsize=0.0
#     )

# W, state = solver.run(b0, X=X, Y=Y)

def jax_loss(param, X, y, invlink, l2reg):
    invlink_rate = invlink(jnp.dot(X, param))
    out = -(jnp.mean(-invlink_rate + y * jnp.log(invlink_rate))) + 0.5 * l2reg * jnp.sum(param ** 2)
    return out



W = []
W_edoardo = []
W_skl = []

for i in range(n_neurons):
    print(i)

    w = minimize(loss1, b0[:,i], (X, Y[:,i]), jac=True, method='L-BFGS-B', tol = 1e-8)
    W.append(w['x'])

    func_jax = lambda param: jax_loss(param, X, Y[:, i], jnp.exp, 1.)
    grad_jax = grad(func_jax)
    w = minimize(func_jax, b0[:, i].flatten(), jac=grad_jax, method='L-BFGS-B', tol=10**-8)
    W_edoardo.append(w['x'])

    # Sklearn comparison
    model= PoissonRegressor(fit_intercept=False)
    model.fit(X, Y[:,i])
    W_skl.append(model.coef_)

W = np.array(W).T
W_skl = np.array(W_skl).T
W_edoardo = np.array(W_edoardo).T

subplot(121)
scatter(W_skl.flatten(),W.flatten())
plt.xlabel('skl')
plt.ylabel('loss1')

subplot(122)
scatter(W_skl.flatten(),W_edoardo.flatten())
plt.xlabel('skl')
plt.ylabel('jax_loss')

# W2 = W[0:-1].reshape((n_neurons, n_basis_funcs, n_neurons))
#
#
# W2_skl = W_skl[0:-1].reshape((n_neurons, n_basis_funcs, n_neurons))
#
#
# figure()
# gs = GridSpec(3, len(B))
# subplot(gs[0,0:2])
# imshow(tc,aspect='auto')
# subplot(gs[0,2:4])
# imshow(W2.mean(1), aspect='auto')
#
# for i in range(n_basis_funcs):
#     subplot(gs[1,i])
#     x = np.arange(B.shape[1])*-1
#     plot(x, B[i])
#     subplot(gs[2,i])
#     im = imshow(W2[:,i,:].T,aspect='auto',vmin=W2.min(),vmax=W2.max())
#
# colorbar(im)
#
# figure()
# gs = GridSpec(3, len(B))
# subplot(gs[0,0:2])
# imshow(tc,aspect='auto')
# subplot(gs[0,2:4])
# imshow(W2_skl.mean(1), aspect='auto')
#
# for i in range(n_basis_funcs):
#     subplot(gs[1,i])
#     x = np.arange(B.shape[1])*-1
#     plot(x, B[i])
#     subplot(gs[2,i])
#     im = imshow(W2_skl[:,i,:].T,aspect='auto',vmin=W2_skl.min(),vmax=W2_skl.max())
#
# colorbar(im)
#
#
# ##############################################
# # TEST
# ##############################################
# alpha2 = np.digitize(np.cumsum(np.random.randn(2000)*0.5)%(2*np.pi), bins)-1
# Yt = np.random.poisson(tc[alpha2]*1.5)
# Xt = _CORR2(jnp.atleast_2d(spike_basis),jnp.atleast_2d(Yt.T))
# Xt = Xt.reshape(np.prod(Xt.shape[0:2]), Xt.shape[-1])
# Xt = Xt.T
# Xt = Xt - Xt.mean(0)
# Xt = Xt / Xt.std(0)
# Xt = np.hstack((Xt, jnp.ones((len(Xt), 1))))
# Yp = jnp.exp(jnp.dot(Xt, W))
#
#
# figure()
# subplot(311)
# imshow(Yt.T, aspect='auto')
# subplot(312)
# imshow(Yp.T, aspect='auto')
# subplot(313)
# plot(Yt[:,0])
# plot(Yp[:,0])
# show()