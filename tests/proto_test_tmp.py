
import jax
from neurostatslib.glm import GLM
from neurostatslib.basis import BSplineBasis
import numpy as np
from sklearn.linear_model import PoissonRegressor
from neurostatslib.utils import convolve_1d_basis
import jax.numpy as jnp
from jax import grad
from jaxopt import LBFGS, GradientDescent
from time import perf_counter

def score_glm(params, model,X,y):
    pred = model._predict(params, X)
    return model._score(pred,y)

def jax_loss(param, X, y, invlink):
    invlink_rate = invlink(jnp.dot(X, param))
    out = -(jnp.mean(-invlink_rate + y * jnp.log(invlink_rate)))
    return out


def numpy_loss(param, X, y, invlink=np.exp):
    invlink_rate = invlink(np.dot(X, param))
    out = -(np.mean(-invlink_rate + y * np.log(invlink_rate)))
    # print(out)
    return out

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

def grad_numpy_loss(param,X,y):
    return -np.mean(-X.T*np.exp(np.dot(X,param)) + X.T*y, axis=1)
def ridge_reg_objective(params, l2reg, X, y, ff = lambda x: jnp.mean(jnp.power(x,2))):
  residuals = jnp.dot(X, params) - y
  return ff(residuals) + 0.5 * l2reg * jnp.sum(params ** 2)
  #return jnp.mean(residuals ** 2) + 0.5 * l2reg * jnp.sum(params ** 2)


params = {'test_end_to_end_glm_fit': []}
# set seed
np.random.seed(123)


n_timepoints = 1000
n_events = 50
n_basis = 6

# mean firing rate in Hz
mean_firing = 25

# define events
events = np.zeros(n_timepoints, dtype=int)
events[np.random.choice(n_timepoints, size=n_events, replace=False)] = 1

# define model params
coeff = np.random.normal(size=n_basis)

# model matrix
for filter_size in [100]:
    basis = BSplineBasis(n_basis_funcs=n_basis, window_size=filter_size, order=3)  # window_size is going to be deprecated
    bX = basis.gen_basis_funcs(np.linspace(0, 1, filter_size))
    X = convolve_1d_basis(bX, events)[:, :, :-1]

    # set firing rate
    firing = np.exp(np.dot(np.squeeze(X).T, coeff))
    firing = firing / firing.mean() * (mean_firing*0.05)
    spikes = np.random.poisson(firing)



    # sklearn solution for unpenalized glm (alpha=0)
    model = PoissonRegressor(alpha=0,fit_intercept=True)
    model.fit(np.squeeze(X).T, spikes)

    for solver in ['GradientDescent', 'BFGS']:
        params['test_end_to_end_glm_fit'].append(
            {
                'basis': bX,
                'spikes':spikes,
                'X':X,
                'solver': solver,
                'sklearn_ML_coeff': model.coef_,
                'sklearn_ML_intercept': model.intercept_
            }
        )



modelX = jnp.array(np.hstack([np.ones((n_timepoints-filter_size,1)), np.squeeze(X).T]))
y = jnp.array(spikes)



from scipy.optimize import minimize



func_numpy = lambda param: numpy_loss(param, np.array(modelX), np.array(y), invlink=np.exp)
grad_numpy = lambda param: grad_numpy_loss(param,np.array(modelX),np.array(y))
res = minimize(func_numpy, x0=np.zeros(modelX.shape[1]), jac=grad_numpy,method='BFGS',options={'disp':True})
print('numpy opt success: ', res.success)


func_jax = lambda param: jax_loss(param, modelX, y,jnp.exp)
grad_jax = grad(func_jax)
res2 = minimize(func_jax, x0=np.zeros((modelX.shape[1],1)), jac=grad_jax,method='BFGS',options={'disp':True})
print('numpy opt success: ', res2.success)

# app grad
x = jnp.array(np.random.normal(size=modelX.shape[1]))
apgr = approx_grad(x, x.shape, func_jax, 10**-4)

model = PoissonRegressor(alpha=0,fit_intercept=False)
model.fit(modelX, y)


y = y.reshape(-1,1)

log_like = lambda param: -jnp.mean(jax.scipy.stats.poisson.logpmf(y, jnp.exp(jnp.dot(modelX, param)), loc=0))

t0 = perf_counter()
optimizer = LBFGS(func_jax, maxiter=100)
optimizer.verbose=False
opt_par, state = optimizer.run(jnp.zeros((modelX.shape[1],1)))
print(f'BFGS time delta {perf_counter()-t0}')

t0 = perf_counter()
optimizer2 = GradientDescent(func_jax, maxiter=100)
optimizer2.verbose=False
opt_par2, state2 = optimizer2.run(jnp.zeros((modelX.shape[1],1)))
print(f'Gradient descent time delta {perf_counter()-t0}')



import matplotlib.pylab as plt


plt.figure()

plt.plot(res2.x,label='scipy BFGS')
plt.plot(model.coef_,'--k',label='sklearn L-BFGS-B')
plt.plot(np.array(opt_par2).flatten(),'-',color='r',label='grad desc')
plt.plot(np.array(opt_par).flatten(),'--',color='m',label='jaxopt BFGS')



glm_model = GLM(
            spike_basis_matrix = bX,
            inverse_link_function = jax.numpy.exp,
            solver_name = 'LBFGS'
        )

Y = np.zeros((1,y.shape[0]+filter_size))
Y[:,filter_size:] = y.T
Y = jnp.array(Y)



glm_model.fit(Y, X)

par = (opt_par[1:,:][None,:,:],opt_par[0])

jax_einsum = jnp.squeeze(jnp.einsum("nbt,nbj->ntj", X, par[0]))
np_einsum = np.squeeze(np.einsum("nbt,nbj->ntj", np.array(X), np.array(par[0])))
np_dot = np.squeeze(np.dot(modelX[:,1:], np.array(par[0])))
print('diff jax einsum', np.abs(np_dot-jax_einsum ).max())
print('diff numpy einsum', np.abs(np_dot-np_einsum ).max())



jax_einsum = jnp.exp(jnp.squeeze(jnp.einsum("nbt,nbj->ntj", X, par[0])+ par[1][:,None]))



pred = glm_model._predict(par, X)
pred2 = jnp.exp(jnp.dot(modelX, opt_par)).flatten()


print(max(np.abs(jnp.squeeze(pred)-pred2)))

scr = glm_model._score(par, X, y)



vec = np.hstack(( np.array(glm_model.baseline_log_fr_), np.array(glm_model.spike_basis_coeff_).flatten()))
plt.plot(vec, label='glm class fit')
plt.legend()
