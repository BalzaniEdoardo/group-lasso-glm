from neurostatslib.glm import GLM
from sklearn.linear_model import PoissonRegressor
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from jax import grad
import scipy.stats as sts
import jax.numpy as jnp
import neurostatslib as nsl
from scipy.optimize import minimize
import jax
from scipy.special import gammaln
def loss(params_vec, X, y, alpha):
    _, nn, nf = X.shape
    Ws, bs = params_vec[:nn*nf].reshape(nn,nf), params_vec[nn*nf:]
    predicted_firing_rates = jnp.exp(
        jnp.einsum("ik,tik->ti", Ws, X) + bs[None, :]
    )
    x = y * jnp.log(predicted_firing_rates)
    # this is a jax jit-friendly version of saying "put a 0 wherever
    # there's a NaN". we do this because NaNs result from 0*log(0)
    # (log(0)=-inf and any non-zero multiplied by -inf gives the expected
    # +/- inf)
    #x = jnp.where(jnp.isnan(x), jnp.zeros_like(x), x)
    # see above for derivation of this.
    score = jnp.mean(
        predicted_firing_rates - x + jax.scipy.special.gammaln(y + 1)
    )
    pen = 0.5 * alpha * jnp.mean(jnp.power(Ws, 2))
    return score + pen

def numpy_loss(param, X, y, alpha, invlink=np.exp):
    #X #= np.hstack((X[:, 0, :], np.ones((X.shape[0],1))))
    invlink_rate = invlink(np.dot(X[:,0,:], param[:-1]) + param[-1])
    out = -(
        np.mean(-invlink_rate + y[:,0] * np.log(invlink_rate) + jax.scipy.special.gammaln(y[:,0] + 1) )
    )
    # print(out)
    pen = 0.5 * alpha * np.mean(np.power(param[:-1], 2))
    return out + pen

def approx_grad(x0, dim, func, epsi):
    grad = np.zeros(shape=dim)
    for j in range(grad.shape[0]):
        if np.isscalar(x0):
            ej = epsi
        else:
            ej = np.zeros(x0.shape[0])
            ej[j] = epsi
        grad[j] = (func(x0 + ej) - func(x0 - ej)) / (2 * epsi)
    return grad

def grad_numpy_loss(param,X,y,alpha):
    X = np.hstack((X[:, 0, :], np.ones((X.shape[0],1))))
    pen = alpha * np.mean(param[:-1])
    return -np.mean(-X.T*np.exp(np.dot(X,param)) + X.T*y[:,0], axis=1) + pen

solver = 'GradientDescent'
solver_kwargs = {'ScipyMinimize':{'tol':10**-13, 'method':"L-BFGS-B",
                                'maxiter':1000, 'options':{'disp':True}},
                 'GradientDescent':{'tol':10**-8, 'maxiter':1000,'verbose':True},
                 'BFGS':{'tol':10**-8, 'maxiter':1000,'verbose':True}
                 }
data = np.load('/Users/ebalzani/Code/generalized-linear-models/tests/data.npz')
X = jnp.asarray(data['X'],dtype=jnp.float32)
y = jnp.asarray(data['y'],dtype=jnp.float32)
alpha = 1.
init_params = jnp.zeros((y.shape[1],X.shape[2])), jnp.log(jnp.mean(y,axis=0))
# model_jax = nsl.glm.GLM(solver_name=solver,
#                         inverse_link_function=jnp.exp,
#                         alpha=alpha, solver_kwargs=solver_kwargs[solver])
#
# model_jax.fit(X, y,init_params=init_params)
model_skl = PoissonRegressor(alpha=alpha,tol=10**-8,solver="lbfgs",max_iter=1000,fit_intercept=True)
model_skl.fit(X[:,0,:], y[:,0])

grad_loss = jax.grad(loss)
init_param_vec = np.zeros(X.shape[1]*X.shape[2]+X.shape[1])
init_param_vec[-X.shape[1]:] = np.log(np.mean(y, axis=0))

loss_f = lambda v:numpy_loss(v, np.asarray(X),np.asarray(y),alpha=alpha)
gloss_f = lambda v:grad_numpy_loss(v, np.asarray(X), np.asarray(y),alpha=alpha)

# opt_res = minimize(
#                 loss_f,
#                 init_param_vec,
#                 method="BFGS",
#                 jac=gloss_f,
#                 options={
#                     "maxiter": 1000,
#                     "disp": True,
#                     "ftol":10**-8
#                     # The constant 64 was found empirically to pass the test suite.
#                     # The point is that ftol is very small, but a bit larger than
#                     # machine precision for float64, which is the dtype used by lbfgs.
#
#                 }
#             )

basis_2d = nsl.basis.MSplineBasis(n_basis_funcs=12, order=4) * \
            nsl.basis.MSplineBasis(n_basis_funcs=12, order=4)

# Ws = opt_res.x[:-1]
# XX, YY, Z = nsl.visualize.eval_response(basis_2d, Ws[-basis_2d.n_basis_funcs:][None], 30)
# plt.figure()
# plt.imshow(Z[:, :, 0])


Ws = model_skl.coef_
XX, YY, Z = nsl.visualize.eval_response(basis_2d, Ws[-basis_2d.n_basis_funcs:][None], 30)
plt.figure()
plt.imshow(Z[:, :, 0])

import scipy.stats as sts
poi = sts.poisson(mu=np.exp(
        np.dot(X[:,0,:], model_skl.coef_) + model_skl.intercept_))
print(poi.logpmf(y.flatten()).mean())
prms = np.hstack([model_skl.coef_, model_skl.intercept_])
print(numpy_loss(prms,X,y,0.,np.exp))
print(loss(prms,X,y,0.))