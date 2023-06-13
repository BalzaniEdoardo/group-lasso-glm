import jax
from neurostatslib.glm import GLM
from neurostatslib.basis import BSplineBasis
import numpy as np
from sklearn.linear_model import PoissonRegressor
from neurostatslib.utils import convolve_1d_basis

class TestGLM():
    #def __init__(self):
    """
    Initialize the parameter grid and generates rate of the Poisson:
        1) filter size in time points
        2) Non-linearity: for now only exp
        3) solver
    """

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
    for filter_size in [50,100,150]:
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

    def test_end_to_end_glm_fit(self, basis, spikes, X, solver, sklearn_ML_coeff, sklearn_ML_intercept):
        y = np.zeros(X.shape[2]+basis.shape[1])
        y[:len(spikes)] = spikes
        y = y.reshape(1,-1)

        model = GLM(
            spike_basis_matrix = basis,
            inverse_link_function = jax.numpy.exp,
            solver_name = solver
        )

        model.fit(y, X)
        assert(all((np.array(model.spike_basis_coeff_).flatten() - sklearn_ML_coeff) < 10**-8))
        assert (all((np.array(model.baseline_log_fr_).flatten() - sklearn_ML_intercept) < 10 ** -8))
        return





#
# nn, nt = 10, 1000
# key = jax.random.PRNGKey(123)
# key, subkey = jax.random.split(key)
# spike_data = jax.random.bernoulli(
#     subkey, jax.numpy.ones((nn, nt))*.5
# ).astype("int64")
#
# spike_basis = BSplineBasis(n_basis_funcs=6, window_size=100, order=3)
# spike_basis_matrix = spike_basis.gen_basis_funcs(np.arange(100))
# model = GLM(spike_basis_matrix)
#
# model.fit(spike_data)
# model.predict(spike_data)
# key, subkey = jax.random.split(key)
# X = model.simulate(subkey, 20, spike_data[:, :100])
