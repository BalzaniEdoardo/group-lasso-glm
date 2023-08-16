import numpy as np
from numpy.typing import NDArray
import neurostatslib as nsl
import scipy.stats as sts
import jax.numpy as jnp
import jax

def temporal_fitler(ws, inhib_a=1, excit_a=2, inhib_b=2, excit_b=2):
    """Generate coupling filter as Gamma pdf difference.

    Parameters
    ----------
    ws:
        The window size of the filter.
    inhib_a:
        The `a` constant for the gamma pdf of the inhibitory part of the filer.
    excit_a:
        The `a` constant for the gamma pdf of the excitatory part of the filer.
    inhib_b:
        The `b` constant for the gamma pdf of the inhibitory part of the filer.
    excit_b:
        The `a` constant for the gamma pdf of the excitatory part of the filer.

    Returns
    -------
    filter:
        The coupling filter.
    """
    x = np.linspace(0, 5, ws)
    gm_inhibition = sts.gamma(a=inhib_a, scale=1/inhib_b)
    gm_excitation = sts.gamma(a=excit_a, scale=1/excit_b)
    filter = gm_excitation.pdf(x) - gm_inhibition.pdf(x)
    filter = 0.8 * filter / np.linalg.norm(filter)
    return filter


def regress_filter(coupling_filter_bank, n_basis_funcs):
    """Approximate scipy.stats.gamma based filters with basis function.

    Find the ols weights for representing the filters in terms of basis functions.
    This is done to re-use the nsl.glm.simulate method.

    Parameters
    ----------
    coupling_filter_bank:
        The coupling filters. Shape (n_neurons, n_neurons, window_size)
    n_basis_funcs:
        The number of basis.

    Returns
    -------
    eval_basis:
        The basis matrix, shape (window_size, n_basis_funcs)
    weights:
        The weights for each neuron. Shape (n_neurons, n_neurons, n_basis_funcs)
    """
    n_neurons, _, ws = coupling_filter_bank.shape
    basis = nsl.basis.RaisedCosineBasisLog(n_basis_funcs=n_basis_funcs)
    eval_basis = basis.evaluate(np.linspace(0, 1, ws))
    mult = np.linalg.pinv(np.dot(eval_basis.T, eval_basis))
    weights = np.einsum("ij, jk, nmk -> nmi", mult, eval_basis.T, coupling_filter_bank)
    return eval_basis, weights


def simulate_spikes(coupling_filter_bank: NDArray,
                    duration_sim_sec: float,
                    dt_sec: float,
                    mean_fr_hz: float,
                    n_basis_approx: int = 15) -> NDArray:
    """Simulate the population spiking activity.

    Approximate the filter with a basis of Raised Cosine Log-spaced and
    simulate spike trains with the `nsl.glm.GLM.simulate` method.

    Parameters
    ----------
    coupling_filter_bank:
        The bank of coupling filters. Shape (n_neurons, n_neurons, window_size).
    duration_sim_sec:
        The duration of the simulation in seconds.
    dt_sec:
        Sampling period in seconds.
    mean_fr_hz:
        Mean firing rate of the population. Defines the log-baseline rate for
        the simulation.
    n_basis_approx:
        Number of basis used for approximating the coupling filter.

    Returns
    -------
    :
        The simulated spike counts of the population. Shape (duration_sim_sec//dt_sec, n_neurons)
    """
    n_samples = int(duration_sim_sec / dt_sec)
    n_neurons, _, ws = coupling_filter_bank.shape
    # set initial spikes to 0s
    init_spikes = np.zeros((ws, n_neurons))
    # get the basis and weights
    coupling_basis, weights = regress_filter(coupling_filter_bank, n_basis_approx)
    weights = weights.reshape(n_neurons, -1)

    # empty input
    X_input = np.zeros((n_samples, n_neurons, 0))

    intercepts = np.log(dt_sec * mean_fr_hz) * np.ones(n_neurons)
    model_glm = nsl.glm.GLM(inverse_link_function=jnp.exp)
    model_glm.spike_basis_coeff_ = weights
    model_glm.baseline_log_fr_ = intercepts
    random_key = jax.random.PRNGKey(123)
    return model_glm.simulate(random_key, n_samples, init_spikes, coupling_basis, X_input)


def define_groups(n_neurons, n_basis_funcs):
    grouping = np.zeros((n_neurons, n_neurons*n_basis_funcs))
    for neu in range(n_neurons):
        grouping[neu, neu * n_basis_funcs: (neu + 1) * n_basis_funcs] = 1
    grouping = jnp.asarray(grouping, dtype=jnp.float32)
    return grouping
