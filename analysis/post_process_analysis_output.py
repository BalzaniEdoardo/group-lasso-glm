import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np
from matlab_loader import bin_spikes, data_path, load_data
from sklearn.model_selection import GridSearchCV, KFold
from plot_utils import plot_psth_by_category, plot_coupling_mask

import neurostatslib as nsl
post_process = "group-lasso"


if post_process == "group-lasso":
    fhpath = '../results/FitAllNeurons_DT_1ms_NBasis_5_WindowSize_250.npz'
    save_prefix = 'group-lasso_'
elif post_process == 'ridge':
    fhpath = '../results/JAXRidgeFitAllNeurons_DT_1ms_NBasis_5_WindowSize_250.npz'
    save_prefix = 'ridge_'
# unpack fit output
dat = np.load(fhpath, allow_pickle=True)

# %% ###################################################################################################################
# # Pre-processing: bin input to counts
# %% ###################################################################################################################
# sampling interval & threshold for rate (exclude neurons with < 1hz firing)
dt_sec = dat["dt_sec"].item()
min_rate_hz = dat["min_rate_hz"].item()

# load spike times, units and trial categories
curated_units, spike_times, spatial_frequencies, orientations = load_data(data_path, min_rate_hz)
# bin the spikes and structure it in a tensor
time, spikes = bin_spikes(spike_times, dt_sec, 0, 0.5)
n_trials, n_time_points, n_neurons = spikes.shape


# %% ###################################################################################################################
# #  Compute the predicted rate by re-instantiating the model
# %% ###################################################################################################################
# Extract Model setup parameters
window_size = dat["window_size"].item()
n_basis_funcs = dat["n_basis_funcs"].item()
basis_class_coupling_name = dat["basis_class_coupling_name"].item()
basis_class_psth_name = dat["basis_class_psth_name"].item()

# instantiate basis
basis_coupling = getattr(nsl.basis, basis_class_coupling_name)(n_basis_funcs=n_basis_funcs)
basis_psth = getattr(nsl.basis, basis_class_psth_name)(n_basis_funcs=n_basis_funcs)

# %%
# #Convolve spikes:
# eval basis-coupling
eval_basis_coupling = basis_coupling.evaluate(np.linspace(0, 1, window_size))
convolved_spikes = nsl.utils.convolve_1d_trials(eval_basis_coupling, spikes)
convolved_spikes = nsl.utils.nan_pad_conv(convolved_spikes, window_size, filter_type="causal")

# %% define a model matrix x trial category
# Assume for simplicity that frequency and orientation have independent effects on the stimulus.
# scale a time vector to [0,1] vector  (needed for raised cosines), repeat n_trial times & evaluate the basis
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

# Apply the mask to get the model matrix for orientation and frequencies
model_ori = np.einsum("tk,tj->tkj", orientation_mask, psth_basis).reshape(n_trials, n_time_points, -1)
model_freq = np.einsum("tk,tj->tkj", frequency_mask, psth_basis).reshape(n_trials, n_time_points, -1)

# %%
# # Extract GLM fit outputs
# (n_neurons, n_features)
coeffs = dat["coeffs"]
# (n_neurons, )
intercepts = dat["intercepts"]
# (n_neurons, )
best_alphas = dat["best_alphas"]
# (n_neurons, len(regularizer_grid))
mean_pr2_test = dat["mean_pr2_test"]
# the regularizer hyperparams used
regularizer_grid = dat["regularizer_grid"]

# compute the predicted rate (one neuron at the time for memory issues)
predicted_rate = np.zeros((n_trials*(n_time_points - window_size), 69))
spikes_stack = np.zeros((n_trials*(n_time_points - window_size), 69))
for neu in range(n_neurons):
    y, X = nsl.utils.combine_inputs(spikes[..., neu: neu + 1], jnp.asarray(convolved_spikes), model_ori,
                                    model_freq, strip_left=window_size, reps=1)
    model = nsl.glm.GLMGroupLasso(
        inverse_link_function=jnp.exp, score_type="pseudo-r2"
    )
    model.baseline_log_fr_ = intercepts[neu: neu + 1]
    model.spike_basis_coeff_ = coeffs[neu: neu + 1]
    predicted_rate[:, neu] = np.squeeze(model.predict(X)) / dt_sec
    spikes_stack[:,neu] = y.flatten()

predicted_rate = predicted_rate.reshape(n_trials, -1, n_neurons)
spikes_stack = spikes_stack.reshape(n_trials, -1, n_neurons)
np.savez("../results/" + save_prefix + "rates_" + "DT_1ms_NBasis_5_WindowSize_250.npz", 
    predicted_rate=np.asarray(predicted_rate), spikes=spikes_stack)
# %%
# Plot fit statistics
fig, axs = plt.subplots(1, 1)
plt.title("cross-validated pseudo-R2")
plt.hist(np.nanmax(mean_pr2_test, axis=1),density=True, color="Grey",alpha=0.3,edgecolor='k')
plt.xlabel('pseodo-$R^2$')
plt.ylabel('pdf')
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
plt.tight_layout()


# Plot global psth
rows = 7
cols = 10
fig = plt.figure(figsize=(12, 6))
plt.suptitle("PSTH over all categories")
for neu in range(69):
    ax = plt.subplot(rows, cols, neu + 1)
    ax.plot(time[window_size:], predicted_rate[:, :, neu].mean(axis=0))
    ax.plot(time[window_size:], spikes[:,250:,neu].mean(axis=0)/dt_sec, color="grey",alpha=0.5, lw=0.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if neu // cols == (rows-1):
        plt.xlabel('time[sec]', fontsize=8)
    if neu % cols == 0:
        plt.ylabel('rate[Hz]', fontsize=8)
fig.tight_layout()


fig, axs = plot_psth_by_category(time[window_size:], predicted_rate, orientations,
                      rows=7, cols=10, plot_every=6, is_angle=True)
plt.suptitle("PSTH by orientation")
fig.tight_layout()


# analyze coupling filters
weights_coupling = coeffs[:, :n_neurons*n_basis_funcs].reshape(-1, n_neurons, n_basis_funcs)
filters = np.einsum("ti,nmi->tnm", eval_basis_coupling, weights_coupling)

# Plot auto-corr
rows = 7
cols = 10
fig = plt.figure(figsize=(12, 6))
plt.suptitle("Auto-correlation filters")
for neu in range(69):
    ax = plt.subplot(rows, cols, neu + 1)
    ax.plot(time[:window_size], filters[:, neu, neu])
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if neu // cols == (rows-1):
        plt.xlabel('time[sec]', fontsize=8)

fig.tight_layout()


# plot coupling strength
coupling_strength = np.linalg.norm(filters, axis=0)
plot_coupling_mask(coupling_strength, colors=['white', 'k'],
                       cmap="Greys_r", title=["Coupling Map"], set_labels=False,
                       plot_grid=True, plot_ticks_every=5,lw=0., sort=True, high_percentile=90)