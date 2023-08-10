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



solver = 'GradientDescent'
solver_kwargs = {'ScipyMinimize':{'tol':10**-13, 'method':"L-BFGS-B",
                                'maxiter':1000, 'options':{'disp':True}},
                 'GradientDescent':{'tol':10**-8, 'maxiter':1000,'verbose':True},
                 'BFGS':{'tol':10**-8, 'maxiter':1000,'verbose':True}
                 }
data = np.load('/Users/ebalzani/Code/generalized-linear-models/tests/data.npz')
X = jnp.asarray(data['X'],dtype=jnp.float32)
y = jnp.asarray(data['y'],dtype=jnp.float32)
alpha = 2
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
loss_f = lambda v:loss(v, np.array(X),np.array(y),alpha)
gloss_f = lambda v:grad_loss(v,np.array(X), np.array(y),alpha)
opt_res = minimize(
                loss_f,
                init_param_vec,
                method="L-BFGS-B",
                jac=gloss_f,
                bounds=np.array([(-10.,10.)]*init_param_vec.shape[0]),
                options={
                    "maxiter": 1000,
                    "disp": True,
                    "tol":10**-8
                    # The constant 64 was found empirically to pass the test suite.
                    # The point is that ftol is very small, but a bit larger than
                    # machine precision for float64, which is the dtype used by lbfgs.

                }
            )


