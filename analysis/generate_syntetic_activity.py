# %%
# # Import dependencies and set parameters
import numpy as np
import neurostatslib as nsl
import jax.numpy as jnp
import jax
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor
import matplotlib.pylab as plt

from simulation_utils import temporal_fitler, regress_filter
from plot_utils import plot_coupling_mask, plot_filters
from simulation_utils import simulate_spikes, define_groups

# set seed
np.random.seed(101)

# set parameters
dt_sec = 0.005
window_size = 100
n_neurons = 4
sparsity_mask = 0.75
sim_duration_sec = 3600
mean_firing_rate_Hz = 2
n_basis_funcs = 10

# filter parameters (gamma.pdf(a_excit, b_excit) - gamma.pdf(a_inhib, b_inhib)
a_inhib = 1
b_inhib = 1
a_excit = np.random.uniform(1.1, 5, size=(n_neurons,)*2)
b_excit = np.random.uniform(1.1, 5, size=(n_neurons,)*2)

# %%
# ## Extract and Plot filters
coupling_filter_bank = np.zeros((n_neurons, n_neurons, window_size))
for neu_i in range(n_neurons):
    for neu_j in range(n_neurons):
        coupling_filter_bank[neu_i, neu_j, :] = temporal_fitler(window_size,
                                                                inhib_a=a_inhib,
                                                                excit_a=a_excit[neu_i, neu_j],
                                                                inhib_b=b_inhib,
                                                                excit_b=b_excit[neu_i, neu_j]
                                                                )


# %%
# Define and Plot Connectivity Map
is_coupled = np.ones((n_neurons, n_neurons))
coupling_indices = np.arange(n_neurons**2)
# assume the auto-corr is always present
coupling_indices = coupling_indices[coupling_indices%n_neurons - coupling_indices//n_neurons != 0]
# select sparse coupling
decoupled = np.unravel_index(np.random.choice(
            coupling_indices,
            size=int(len(coupling_indices)*sparsity_mask), replace=False),
            shape=(n_neurons, n_neurons)
            )
is_coupled[decoupled] = 0
#is_coupled[np.diag_indices(n_neurons)] = 0.

# %%
# Remove uncoupled using the mask
coupling_filter_bank = coupling_filter_bank * is_coupled[..., None]

# %%
# Plot the coupling matrix & filters
fig,_ = plot_coupling_mask(is_coupled, colors=['k', 'white'], vmax=1)
fig.savefig(f"../results/ground_truth_coupling.pdf")
fig, axs = plot_filters(coupling_filter_bank, dt_sec, label='ground truth')

#%%
# Simulate spikes

spikes, rates = simulate_spikes(coupling_filter_bank, 
    sim_duration_sec + window_size * dt_sec, 
    dt_sec, mean_fr_hz=mean_firing_rate_Hz, n_basis_approx=20, device="cpu")
print(f"mean firing rate hz: {spikes.mean(axis=0)/dt_sec}")

# %%
# Prepare GLM predictors

# define groups per neuron
group_mask = define_groups(n_neurons, n_basis_funcs=n_basis_funcs)

# evaluate a basis
basis = nsl.basis.RaisedCosineBasisLog(n_basis_funcs=n_basis_funcs)
eval_basis = basis.evaluate(np.linspace(0, 1, window_size))

# convolve with spike trains
convolved_spikes = nsl.utils.convolve_1d_trials(eval_basis, spikes[None,:-1])

# process for model fitting
y, X = nsl.utils.combine_inputs(spikes[window_size:], convolved_spikes, reps=n_neurons)

# %%
# Fit a GLM with group-Lasso
if len(jax.devices('gpu')):
    target_device = jax.devices('gpu')[0]
    y = jax.device_put(y, target_device)
    X = jax.device_put(X, target_device)
    group_mask = jax.device_put(group_mask, target_device)
w_predicted = np.zeros((n_neurons, n_neurons * n_basis_funcs))
for neu in range(n_neurons):
    print(f"jax fit neu {neu}")
    # define a model
    model = nsl.glm.GLMGroupLasso(
                solver_kwargs={'tol':10**-8, 'maxiter':1000, 'jit':True},
                inverse_link_function=jnp.exp
    )
    # Pick consecutive blocks, no time point shuffling for time series
    kf = KFold(n_splits=5, shuffle=False)

    cls = GridSearchCV(model, param_grid={'alpha': np.logspace(-4.9, -4.5, 6)}, cv=kf, verbose=True)
    cls.fit(X[:, neu:neu+1, :], y[:, neu:neu+1], mask=group_mask)
    w_predicted[neu] = cls.best_estimator_.spike_basis_coeff_.flatten()
    print(f"best regularizer: {cls.best_params_}")

# %%
# plot predicted filters
w_predicted = w_predicted.reshape(n_neurons, n_neurons, -1)
filter_predicted = np.einsum("tj, nmj -> nmt", eval_basis, w_predicted)

fig, axs = plot_filters(coupling_filter_bank, dt_sec, label='ground truth')
plot_filters(filter_predicted, dt_sec, (fig, axs), color='tomato', ls='-', label='glm')
fig.savefig(f"../results/group_lasso_filters_{int(mean_firing_rate_Hz)}Hz.pdf")

predicted_coupling_strength = np.linalg.norm(filter_predicted, axis=2)
true_coupling_strength = np.linalg.norm(coupling_filter_bank, axis=2)



# %%
# sklearn fit
w_pred_skl = np.zeros((n_neurons, n_neurons, n_basis_funcs))
for neu in range(n_neurons):
    print("analyzing neu", neu)
    model_skl = PoissonRegressor(alpha=1., fit_intercept=True)
    kf = KFold(n_splits=5, shuffle=False)
    cls = GridSearchCV(model_skl, param_grid={'alpha': np.logspace(-5, -1, 6)}, cv=kf)
    cls.fit(X[:, neu], y[:, neu])
    w_pred_skl[neu] = cls.best_estimator_.coef_.reshape(n_neurons, -1)

#_, w_true = regress_filter(coupling_filter_bank, n_basis_funcs)
# filter_pred_true = np.einsum("tj, nmj -> nmt", eval_basis, w_true)
filter_pred_skl = np.einsum("tj, nmj -> nmt", eval_basis, w_pred_skl)
fig, axs = plot_filters(coupling_filter_bank, dt_sec, label='ground truth')
plot_filters(filter_pred_skl, dt_sec, (fig, axs), color='green', ls='-',label='glm-ridge')
fig.savefig(f"../results/slkearn_ridge_filters_{int(mean_firing_rate_Hz)}Hz.pdf")

plt.legend()

skl_coupling_strength = np.linalg.norm(filter_pred_skl, axis=2)

# %%
# Plot and compare filt strength 
fig,_ = plot_coupling_mask(true_coupling_strength,predicted_coupling_strength,skl_coupling_strength,title=['Ground truth',"Group-Lasso","Ridge"], cmap="Greys_r",
                           vmax=0.8, vmin=0.0)
fig.savefig(f"../results/compare_filter_strength_{int(mean_firing_rate_Hz)}Hz.pdf")

# plt.figure()
# plt.plot(y[:300, 0]/10)
# intercepts = np.log(dt_sec * 5) * np.ones(n_neurons)
# plt.plot(np.exp(np.dot(X[:300,0], w_true[0,0])+intercepts))

# %%

