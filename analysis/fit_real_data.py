# %%
# import libs
import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np
from matlab_loader import bin_spikes, data_path, load_data
from sklearn.model_selection import GridSearchCV, KFold

import neurostatslib as nsl

from simulation_utils import define_groups
from plot_utils import plot_coupling_mask, plot_filters

from scipy.cluster.hierarchy import linkage, leaves_list


# %%
# Parameters
dt_sec = 0.001
window_size = 250
min_rate_hz = 1
n_basis_funcs = 5
basis_class_coupling = nsl.basis.RaisedCosineBasisLog
basis_class_psth = nsl.basis.RaisedCosineBasisLinear
n_splits = 5
regularizer_grid = np.logspace(-5, -1, 6)

# define basis
basis_coupling = basis_class_coupling(n_basis_funcs=n_basis_funcs)
basis_psth = basis_class_psth(n_basis_funcs=n_basis_funcs)

# eval basis
eval_basis_coupling = basis_coupling.evaluate(np.linspace(0, 1, window_size))

# %%
# Plot basis
fig, axs = plt.subplots(1, 2)
axs[0].plot(np.arange(window_size)*dt_sec, eval_basis_coupling)
axs[1].plot(np.arange(int(0.5/dt_sec))*dt_sec, basis_psth.evaluate(np.linspace(0, 1, int(0.5/dt_sec))))

# %%
# Load data
curated_units, spike_times, spatial_frequencies, orientations = load_data(data_path, min_rate_hz)
time, spikes = bin_spikes(spike_times, dt_sec, 0, 0.5)
n_trials, n_time_points, n_neurons = spikes.shape

# scale a time vector to [0,1] vector
# & evaluate the basis
x_vec = np.tile(time/0.5, n_trials)
orientation_stack = np.repeat(orientations, n_time_points)
frequency_stack = np.repeat(spatial_frequencies, n_time_points)
psth_basis = basis_psth.evaluate(x_vec)

# define the masks for trial types
unq_ori = np.unique(orientations)
unq_freq = np.unique(spatial_frequencies)
n_freq = unq_freq.shape[0]
n_ori = unq_ori.shape[0]

orientation_mask = np.zeros((x_vec.shape[0], unq_ori.shape[0]))
cnt = 0
for ori in unq_ori:
    orientation_mask[orientation_stack == ori, cnt] = 1
    cnt += 1

frequency_mask = np.zeros((x_vec.shape[0], unq_freq.shape[0]))
cnt = 0
for freq in unq_freq:
    frequency_mask[frequency_stack == freq, cnt] = 1
    cnt += 1

# create the model matrix for orientation and frequencies
model_ori = np.einsum("tk,tj->tkj", orientation_mask, psth_basis).reshape(n_trials, n_time_points, -1)
model_freq = np.einsum("tk,tj->tkj", frequency_mask, psth_basis).reshape(n_trials, n_time_points, -1)

# convolve spikes
convolved_spikes = nsl.utils.convolve_1d_trials(eval_basis_coupling, spikes)
convolved_spikes = nsl.utils.nan_pad_conv(convolved_spikes, window_size, filter_type="causal")

del frequency_mask, orientation_mask, orientation_stack, frequency_stack

# %%
# Fit all neurons
# loop over neurons to use cv and to avoid over-regularizing on low firing rate neurons
coeffs = np.zeros((n_neurons, (n_neurons + n_freq + n_ori) * n_basis_funcs))
intercepts = np.zeros((n_neurons,))
best_alphas = np.zeros((n_neurons, ))
mean_pr2_test = np.zeros((n_neurons, regularizer_grid.shape[0]))
std_pr2_test = np.zeros((n_neurons, regularizer_grid.shape[0]))
for neu in range(n_neurons):
    print(f"analyzing neuron {neu+1}/{n_neurons}")
    y, X = nsl.utils.combine_inputs(spikes[..., neu: neu+1], jnp.asarray(convolved_spikes), model_ori,
                                  model_freq,strip_left=window_size, reps=1)

    # fit model
    group_mask = define_groups(n_neurons + n_freq + n_ori, n_basis_funcs=n_basis_funcs)
    # Fit a GLM with group-Lasso
    if any("gpu" in device.__repr__().lower() for device in jax.local_devices()):
        print("Fit with GPU")
        target_device = jax.devices('gpu')[0]
        y = jax.device_put(y, target_device)
        X = jax.device_put(X, target_device)
        group_mask = jax.device_put(group_mask, target_device)
    model = nsl.glm.GLMGroupLasso(
                solver_kwargs={'tol': 10**-8, 'maxiter': 1000, 'jit':True},
                inverse_link_function=jnp.exp,alpha=0.001, score_type="pseudo-r2"
    )
    # Pick consecutive blocks, no time point shuffling for time series
    kf = KFold(n_splits=n_splits, shuffle=False)
    cls = GridSearchCV(model, param_grid={'alpha': regularizer_grid}, cv=kf, verbose=True)
    cls.fit(X, y, mask=group_mask)
    model = cls.best_estimator_
    best_alphas[neu] = cls.best_params_['alpha']
    coeffs[neu] = model.spike_basis_coeff_[0]
    intercepts[neu] = model.baseline_log_fr_[0]
    mean_pr2_test[neu] = cls.cv_results_['mean_test_score']
    std_pr2_test[neu] = cls.cv_results_['std_test_score']
    

# # save fit results and config
fhname = f"../results/FitAllNeurons_DT_{int(1000*dt_sec)}ms_NBasis_{n_basis_funcs}_WindowSize_{window_size}.npz"
np.savez(fhname,
        dt_sec = dt_sec,
        window_size = window_size,
        min_rate_hz = min_rate_hz,
        n_basis_funcs = n_basis_funcs,
        n_neurons = n_neurons,
        n_trials = n_trials,
        n_time_points = n_time_points,
        basis_class_coupling_name = basis_class_coupling.__name__,
        basis_class_psth_name = basis_class_psth.__name__,
        coeffs = coeffs,
        intercepts = intercepts,
        best_alphas = best_alphas,
        mean_pr2_test = mean_pr2_test,
        regularizer_grid = regularizer_grid
        )

# # plot the results
# weights_coupling = model.spike_basis_coeff_[:,:n_basis_funcs*n_neurons].reshape(n_neurons, n_neurons, -1)
# filter_predicted = np.einsum("tj, nmj -> nmt", eval_basis_coupling, weights_coupling)
# predicted_coupling_strength = np.linalg.norm(filter_predicted, axis=2)
# fig,_ = plot_coupling_mask(predicted_coupling_strength, title=["Group-Lasso"], cmap="Greys_r")




# %%
