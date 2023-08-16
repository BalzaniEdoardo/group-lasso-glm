# %%
# # Import dependencies and set parameters
import numpy as np
import neurostatslib as nsl
import jax.numpy as jnp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor
import matplotlib.pylab as plt

from simulation_utils import temporal_fitler, regress_filter
from plot_utils import plot_coupling_mask, plot_filters
from simulation_utils import simulate_spikes, define_groups

fit_gam = False

# set seed
np.random.seed(101)

# set parameters
dt_sec = 0.005
window_size = 100
n_neurons = 4
sparsity_mask = 0.75
sim_duration_sec = 1000
mean_firing_rate_Hz = 8

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

# %%|
# Remove uncoupled using the mask
coupling_filter_bank = coupling_filter_bank * is_coupled[..., None]

# %%
# Plot the coupling matrix & filters
plot_coupling_mask(is_coupled, colors=['k', 'white'])
fig, axs = plot_filters(coupling_filter_bank, dt_sec, label='ground truth')

#%%
# Simulate spikes
spikes, rates = simulate_spikes(coupling_filter_bank, sim_duration_sec + window_size * dt_sec, dt_sec, mean_fr_hz=mean_firing_rate_Hz, n_basis_approx=20)
print(f"Effective mean firing rate Hz: {spikes.mean(axis=0)/dt_sec}")
#
# # %%
# # Prepare GLM predictors
# n_basis_funcs = 10
#
# # define groups per neuron
# group_mask = define_groups(n_neurons, n_basis_funcs=n_basis_funcs)
#
# # evaluate a basis
# basis = nsl.basis.RaisedCosineBasisLog(n_basis_funcs=n_basis_funcs)
# eval_basis = basis.evaluate(np.linspace(0, 1, window_size))
#
# # convolve with spike trains
# convolved_spikes = nsl.utils.convolve_1d_trials(eval_basis, spikes[None,:-1])
#
# # process for model fitting
# y, X = nsl.utils.combine_inputs(spikes[window_size:], convolved_spikes, reps=n_neurons)
#
# # %%
# # # Fit a GLM with group-Lasso
#
# # define a model
# model = nsl.glm.GLMGroupLasso(
#             solver_kwargs={'tol': 10**-8, 'maxiter': 1000, 'jit': True},
#             inverse_link_function=jnp.exp
# )
# # Pick consecutive blocks, no time point shuffling for time series
# kf = KFold(n_splits=5, shuffle=False)
#
# cls = GridSearchCV(model, param_grid={'alpha': np.logspace(-5, 1, 5)}, cv=kf)
# cls.fit(X, y, mask=group_mask)
#
# # plot predicted filters
# w_predicted = cls.best_estimator_.spike_basis_coeff_.reshape(n_neurons, n_neurons, -1)
# filter_predicted = np.einsum("tj, nmj -> nmt", eval_basis, w_predicted)
#
# fig, axs = plot_filters(coupling_filter_bank, dt_sec, label='ground truth')
# plot_filters(filter_predicted, dt_sec, (fig, axs), color='tomato', ls='-', label='glm')
# print(f"best regularizer: {cls.best_params_}")
#
# filter_magnitude = np.linalg.norm(filter_predicted, axis=2)
# gt_filter_magnitude = np.linalg.norm(coupling_filter_bank, axis=2)
# plot_coupling_mask(filter_magnitude,
#                    cmap='Greys_r', title="GLM coupling strength", set_labels=False)
# plot_coupling_mask(gt_filter_magnitude,
#                    cmap='Greys_r', title="GT coupling strength", set_labels=False)
#
# # # %%
# # sklearn fit
# w_pred_skl = np.zeros((n_neurons, n_neurons, n_basis_funcs))
# for neu in range(n_neurons):
#     print("analyzing neu", neu)
#     model_skl = PoissonRegressor(alpha=1., fit_intercept=True)
#     #model_skl.fit(X[:, neu], y[:, neu])
#     kf = KFold(n_splits=5, shuffle=False)
#     cls = GridSearchCV(model_skl, param_grid={'alpha': np.logspace(-3, 1, 5)}, cv=kf)
#     cls.fit(X[:, neu], y[:, neu])
#     w_pred_skl[neu] = cls.best_estimator_.coef_.reshape(n_neurons, -1)
#
# filter_pred_skl = np.einsum("tj, nmj -> nmt", eval_basis, w_pred_skl)
# fig, axs = plot_filters(coupling_filter_bank, dt_sec, label='ground truth')
# plot_filters(filter_pred_skl, dt_sec, (fig, axs), color='green', ls='-',label='glm')
# plt.legend()
#
# # %%
# # # # Fit a PGAM
# # ## PGAM
#
# if fit_gam:
#     import sys
#     sys.path.append('/Users/ebalzani/Code/Demo_PGAM/PGAM/src/PGAM')
#     from GAM_library import general_additive_model
#     from gam_data_handlers import smooths_handler
#     import statsmodels.api as sm
#     y_numpy = np.array(y)
#
#     # Fit a gam
#     link = sm.genmod.families.links.log()
#     poissFam = sm.genmod.families.family.Poisson(link=link)
#     impulse = np.zeros(2 * window_size + 1)
#     impulse[100] = 1
#     cnt = 0
#     plt.figure(figsize=(10,8))
#     ax = np.zeros((n_neurons, n_neurons),dtype=object)
#     for k in range(n_neurons):
#         sm_handler = smooths_handler()
#         for neu in range(n_neurons):
#             sm_handler.add_smooth("neu_%d"%(neu+1), [y_numpy[:, neu]],
#                               ord=4, knots_num=n_basis_funcs - 2,
#                               perc_out_range=0., is_cyclic=[False],
#                               is_temporal_kernel=True, kernel_direction=1,
#                               kernel_length=2 * window_size + 1, penalty_type="der", time_bin=dt_sec, der=2,
#                               trial_idx=np.ones(y.shape[0]), lam=0.1)
#
#         gam_model = general_additive_model(sm_handler, sm_handler.smooths_var, y_numpy[:, k],
#                                        poissFam)
#         fit = gam_model.optim_gam(sm_handler.smooths_var,
#                             max_iter=100,
#                             tol=1e-10,
#                             conv_criteria='gcv',
#                             perform_PQL=True,
#                             use_dgcv=1.5,
#                             method='L-BFGS-B',
#                             methodInit='L-BFGS-B',
#                             compute_AIC=False,
#                             random_init=False,
#                             bounds_rho=None,
#                             gcv_sel_tol=1e-10,
#                             fit_initial_beta=False,
#                             )
#         for j in range(n_neurons):
#             ax[k, j] = plt.subplot(n_neurons,n_neurons, cnt+1)
#
#             fX, fXm,fXp = fit.smooth_compute([impulse], "neu_%d"%(j+1))
#             plt.plot(fX[101:], 'b')
#             plt.plot(fXp[101:], '--b')
#             plt.plot(fXm[101:], '--b')
#             plt.plot(filter_predicted[k, j],'r')
#             plt.plot(coupling_filter_bank[k, j],'k')
#             cnt += 1
#
#     ymin = min([ax[i,j].get_ylim()[0]
#                 for i in range(n_neurons)
#                 for j in range(n_neurons)])
#
#     ymax = max([ax[i,j].get_ylim()[1]
#                 for i in range(n_neurons)
#                 for j in range(n_neurons)])
#
#     for i in range(n_neurons):
#         for j in range(n_neurons):
#             ax[i,j].set_ylim(ymin, ymax)
#
#     plt.tight_layout()