from neurostatslib.glm import GLMGroupLasso
from sklearn.linear_model import PoissonRegressor
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from jax import grad
import scipy.stats as sts
import jax.numpy as jnp



np.random.seed(100)
alpha = 0.1
nn, nt, ws, nb, nbi = 2, 35000, 30, 5, 0
X = np.random.normal(size=(nt, nn, nb*nn+nbi))
W_true = np.random.normal(size=(nn, nb*nn+nbi)) * 0.8
W_true[0, nb:nb*2] = 0.
mask = np.zeros((nn, nb*nn+nbi))
mask[1, nb:nb*2] = 1
mask[0, :nb] = 1
mask = jnp.asarray(mask, dtype=jnp.float32)

b_true = -3*np.ones(nn)
firing_rate = np.exp(np.einsum("ik,tik->ti", W_true, X) + b_true[None, :])
spikes = np.random.poisson(firing_rate)
# check likelihood
poiss_rand = sts.poisson(firing_rate)
mean_ll = poiss_rand.logpmf(spikes).mean()

# define a mode
model_jax = GLMGroupLasso(solver_kwargs={'tol':10**-8, 'maxiter':1000, 'jit':True},
                          inverse_link_function=jnp.exp,
                          alpha=alpha)

init_params = 0.001*np.random.normal(size=(nn, nb*nn+nbi)), np.log(np.mean(spikes,axis=0))
model_jax.fit(X, spikes, mask=mask, init_params=init_params)