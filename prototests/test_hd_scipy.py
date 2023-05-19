# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-17 14:47:47
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-05-17 18:57:42
import jax
import jax.numpy as jnp
import numpy as onp
import neurostatslib as nsl
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis, MSplineBasis
import matplotlib
from matplotlib.pyplot import *
import pynapple as nap
import sys
import jax.numpy as jnp
from functools import partial
import jaxopt
from scipy.optimize import minimize

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# pynapple data
data = nap.load_session("/mnt/home/gviejo/pynapple/your/path/to/A2929-200711", "neurosuite")

# data = nap.load_session("/Users/gviejo/pynapple/your/path/to/A2929-200711", "neurosuite")



spikes = data.spikes.getby_category("location")["adn"]
position = data.position
wake_ep = data.epochs["wake"]

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(
    spikes, position["ry"], 120, minmax=(0, 2 * np.pi)
)
peaks = tuning_curves.idxmax()

#####################################################################
# Parameters
#####################################################################
bin_size = 0.005
ws = 200
nb = 3

count = spikes.count(bin_size, wake_ep)
order = peaks[count.columns].sort_values().index
count = count[order]
spike_data = jnp.array(count.values.T)
n_neurons = spike_data.shape[0]

# Getting the spike basis function
spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=ws)
sim_pts = nsl.sample_points.raised_cosine_linear(nb, ws)
B = spike_basis.gen_basis_funcs(sim_pts)

# B = B[::-1]

# B = B[1][None,:]


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
X = np.array(X)
X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
X = X.T
X = X - X.mean(0)
X = X / X.std(0)



Y = np.array(spike_data).T
# Y = Y[0:X.shape[0]]
y = Y[:,0]

# Initial parameters
b0 = np.zeros((n_neurons, n_basis_funcs, n_neurons))
# b0 = b0.flatten()


# # loss function and gradient
# def loss(b):
#     Xb = np.dot(X, b)
#     exp_Xb = np.exp(Xb)
#     loss = exp_Xb.sum() - np.dot(y, Xb)
#     grad = np.dot(X.T, exp_Xb - y)
#     return loss, grad

# # hessian
# def hess(b):
#     return np.dot(X.T, np.exp(np.dot(X, b))[:, None]*X)



def loss2(b):
    tmp = b.reshape((n_neurons, n_basis_funcs, n_neurons))
    for i in range(len(tmp)):
        tmp[i,:,i] = 0.0
    tmp = tmp.reshape((n_neurons*n_basis_funcs, n_neurons))
    Xb = np.dot(X, tmp)
    exp_Xb = np.exp(Xb)

    loss = np.sum(exp_Xb, 0) - np.sum(Y*Xb, 0)

    grad = np.dot(X.T, exp_Xb - Y)
    grad = grad.reshape((n_neurons, n_basis_funcs, n_neurons))
    for i in range(len(grad)):
        grad[i,:,i] = 0.0

    return np.mean(loss), grad.flatten()

def hess2(b):
    tmp = b.reshape((n_neurons*n_basis_funcs, n_neurons))
    Xb = np.dot(X, tmp)
    exp_Xb = np.exp(Xb)
    tmp2 = np.zeros((b.shape[0], b.shape[0]))

    n = n_neurons*n_basis_funcs
    for i in range(n_neurons):
        tmp2[i*n:i*n+n,i*n:i*n+n] = np.dot(X.T, exp_Xb[:,i][:, None]*X)

    return tmp2



# Run optimization
# result = minimize(loss, b0[:,0], jac=True, hess=hess, method='newton-cg')

result = minimize(loss2, b0.flatten(), jac=True, method='newton-cg')


x = result.x.reshape((n_neurons, n_basis_funcs, n_neurons))

sys.exit()




#####################################################################
# PLOT
#####################################################################
figure()
for i in spikes:
    subplot(3, 5, i + 1, projection="polar")
    plot(tuning_curves[i])

# figure()
# subplot(311)
# plot(spike_data[0])
# plot(np.arange(ws, spike_data.shape[1]+1), pred[0])
# ax = subplot(312)
# imshow(spike_data[:,ws:], aspect='auto', cmap = 'jet')
# subplot(313, sharex = ax)
# imshow(pred, aspect='auto', cmap = 'jet')


figure()
for n in range(n_neurons):
    subplot(3,3,n+1)
    tmp = B.T @ params[0][:,:,n].T
    # tmp = tmp[::-1]
    t = np.arange(0, len(tmp))*-bin_size
    plot(t, tmp)
    axhline(0, dashes=[2, 2], color='k')
# ion()


figure()
plot(B.T)

figure()
for i in range(n_neurons):
    subplot(3,7,i+1, projection='polar')
    plot(tuning_curves[order[i]])    
for i in range(n_neurons):
    subplot(3,7,i+1+7)
    imshow(params[0][i,:,:].T)
    ylabel("Neurons")
    xlabel("Basis")
for i in range(n_neurons):
    subplot(3,7,i+1+7*2)
    imshow(params[0][:,:,i])    
    ylabel("Neurons")
    xlabel("Basis")


figure()
ax = subplot(nb+1, 1, 1)
plot(spike_data[0,ws:])
for i in range(nb):
    subplot(nb+1,1,i+2, sharex = ax)
    plot(X[0,i,:])

show()




