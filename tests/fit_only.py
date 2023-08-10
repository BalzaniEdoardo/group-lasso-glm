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

@jax.jit
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
    #jax.scipy.special.gammaln(y + 1)
    score = jnp.mean(
        predicted_firing_rates - x
    )
    pen = 0.5 * alpha * jnp.mean(jnp.power(Ws, 2))
    return score + pen

@jax.jit
def loss_tupe(params_vec, X, y, alpha):

    Ws, bs = params_vec
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
    #jax.scipy.special.gammaln(y + 1)
    score = jnp.mean(
        predicted_firing_rates - x
    )
    pen = 0.5 * alpha * jnp.mean(jnp.power(Ws, 2))
    return score + pen

grad_loss = jax.grad(loss)

solver = 'BFGS'
solver_kwargs = {'ScipyMinimize':{'tol':10**-13, 'method':"L-BFGS-B",
                                'maxiter':1000, 'options':{'disp':True}},
                 'GradientDescent':{'tol':10**-8, 'maxiter':1000,'verbose':True},
                 'BFGS':{'tol':10**-8, 'maxiter':1000,'verbose':True,'jit':True}
                 }
data = np.load('/Users/ebalzani/Code/generalized-linear-models/tests/data.npz')
X = jnp.asarray(data['X'],dtype=jnp.float32)
y = jnp.asarray(data['y'],dtype=jnp.float32)
alpha = 1.
init_params = jnp.zeros((y.shape[1], X.shape[2])), jnp.log(jnp.mean(y,axis=0))
model_jax = nsl.glm.GLM(solver_name=solver,
                        inverse_link_function=jnp.exp,
                        alpha=alpha, solver_kwargs=solver_kwargs[solver])

model_jax.fit(X, y, init_params=init_params)
model_skl = PoissonRegressor(alpha=alpha,tol=10**-8,solver="lbfgs",max_iter=1000,fit_intercept=True)
model_skl.fit(X[:,0,:], y[:,0])


basis_2d = nsl.basis.MSplineBasis(n_basis_funcs=12, order=4) * \
            nsl.basis.MSplineBasis(n_basis_funcs=12, order=4)

Ws = model_jax.spike_basis_coeff_
XX, YY, Z = nsl.visualize.eval_response(basis_2d, Ws[:,-basis_2d.n_basis_funcs:], 30)
plt.figure()
plt.imshow(Z[:, :, 0])


Ws = model_skl.coef_
XX, YY, Z = nsl.visualize.eval_response(basis_2d, Ws[-basis_2d.n_basis_funcs:][None], 30)
plt.figure()
plt.imshow(Z[:, :, 0])

# import scipy.stats as sts
# poi = sts.poisson(mu=np.exp(
#         np.dot(X[:,0,:], model_skl.coef_) + model_skl.intercept_))
# print(poi.logpmf(y.flatten()).mean())
# prms = np.hstack([model_skl.coef_, model_skl.intercept_])
# print(numpy_loss(prms,X,y,0.,np.exp))
# print(loss(prms,X,y,0.))