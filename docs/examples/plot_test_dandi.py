"""
# Stream data with Dandi & pre-process with pynapple

Tutorial on streaming and analyzing data with dandi an pynapple.

## Stream the data
Import libraries.
"""

# load modules
import pynwb

from pynwb import NWBHDF5IO, TimeSeries
import sklearn.model_selection as model_selection

from dandi.dandiapi import DandiAPIClient
import pynapple as nap
import jax.numpy as jnp
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import neurostatslib as nsl
from scipy.interpolate import interp1d

import pynwb
import h5py

from matplotlib.pylab import *

# %%
# ## Stream a dataset of grid-cells

#####################################
# Dandi
#####################################

# ecephys, Buzsaki Lab (15.2 GB)
dandiset_id, filepath = "000582", "sub-10073/sub-10073_ses-17010302_behavior+ecephys.nwb"


with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)




# first, create a virtual filesystem based on the http protocol
fs=fsspec.filesystem("http")

# create a cache to save downloaded data to disk (optional)
fs = CachingFileSystem(
    fs=fs,
    cache_storage="nwb-cache",  # Local folder for the cache
)

# next, open the file
file = h5py.File(fs.open(s3_url, "rb"))
io = pynwb.NWBHDF5IO(file=file, load_namespaces=True)

# %%
# ## Pre-process with pynapple

#####################################
# Pynapple
#####################################

nwb = nap.NWBFile(io.read())
print(nwb)

# %%
# ## Extracting data

units = nwb["units"]

position = nwb["SpatialSeriesLED1"]

tc, binsxy = nap.compute_2d_tuning_curves(units, position, 15)


figure()
for i in tc.keys():
    ax = subplot(2,4,i+1)
    imshow(tc[i], origin="lower")
    ax.set_aspect("equal")
plt.tight_layout()

figure()
for i in units.keys():
    ax = subplot(2,4,i+1)
    plot(position['y'], position['x'])
    spk_pos = units[i].value_from(position)
    plot(spk_pos["y"], spk_pos["x"], 'o', color = 'red', markersize = 1, alpha = 0.5)
    ax.set_aspect("equal")
plt.tight_layout()
show()

# %%
# ## Fit a glm with coupling and positions

# bin spikes
counts = units.count(0.005, ep=position.time_support)
t_index = counts.times()

# linear interp position
position_interp = np.zeros((counts.shape[0], 2))
position_interp[:, 0] = interp1d(position.times(),
                                 position.x)(t_index)
position_interp[:, 1] = interp1d(position.times(),
                                 position.y)(t_index)

# convert to jax
counts = jnp.asarray(counts, dtype=jnp.float32)
position_interp = jnp.asarray(position_interp,
                              dtype=jnp.float32)
position_interp = (position_interp - position_interp.min(axis=0)) / (position_interp.max(axis=0) - position_interp.min(axis=0))

# # convolve counts
window_size = 100
basis = nsl.basis.RaisedCosineBasisLog(n_basis_funcs=7)

# extract convolution filter
_, eval_basis = basis.evaluate_on_grid(window_size)
conv_counts = nsl.utils.convolve_1d_trials(eval_basis, counts[None, :, :])
conv_counts = nsl.utils.nan_pad_conv(conv_counts,
                                     window_size,
                                     filter_type="causal")

# # evaluate position on 2D = basis
basis_2d = nsl.basis.RaisedCosineBasisLinear(n_basis_funcs=12) * \
            nsl.basis.RaisedCosineBasisLinear(n_basis_funcs=12)
position_basis = basis_2d.evaluate(position_interp[:, 0],
                                   position_interp[:, 1])

# %%
# ## Plot basis



# combine inputs
y, X = nsl.utils.combine_inputs(counts,
                                conv_counts,
                                position_basis[None, :, None],
                                strip_left=window_size,
                                reps=counts.shape[1])

# fit GLM
solver = 'BFGS'
solver_kwargs = {'tol': 10**-6, 'maxiter': 1000, 'jit':True}
init_params = jnp.zeros((y.shape[1], X.shape[2])), jnp.log(jnp.mean(y, axis=0))

alpha = 0.1
model_jax = nsl.glm.GLM(solver_name=solver,
                        inverse_link_function=jnp.exp,
                        alpha=alpha, solver_kwargs=solver_kwargs)
model_jax.fit(X, y, init_params=init_params)

# visualize output
_, _, Z = nsl.visualize.eval_response(basis_2d, model_jax.spike_basis_coeff_[:, -basis_2d.n_basis_funcs:], 30)
nsl.visualize.imshow_units(Z[1:-1, 1:-1], 2, 4)
firing_rate = model_jax.predict(X)

# %%
# ## Sklearn compatibility
cls = model_selection.GridSearchCV(model_jax, param_grid={'alpha':[0.1, 1., 10.]})
cls.fit(X, y)

firing_rate = model_jax.predict(X)
firing_rate_pred = nap.TsdFrame(t=t_index[window_size:], d=firing_rate)
position_interp_nap = nap.TsdFrame(t=t_index[window_size:], d=position_interp[window_size:])
tc, binsxy = nap.compute_2d_tuning_curves_continuous(firing_rate_pred, position_interp_nap, 15)

figure()
plt.suptitle("prediction coupling + position")
for i in tc.keys():
    ax = subplot(2,4,i+1)
    imshow(tc[i], origin="lower")
    ax.set_aspect("equal")
plt.tight_layout()


# %%
# ## Only position

# combine inputs
y, X = nsl.utils.combine_inputs(counts,
                                position_basis[None, :, None],
                                strip_left=window_size,
                                reps=counts.shape[1])

init_params = jnp.zeros((y.shape[1], X.shape[2])), jnp.log(jnp.mean(y, axis=0))

alpha = 0.1
model_jax_position = nsl.glm.GLM(solver_name=solver,
                        inverse_link_function=jnp.exp,
                        alpha=alpha, solver_kwargs=solver_kwargs)
model_jax_position.fit(X, y, init_params=init_params)


# %%
# ## Visually compare outputs
plt.figure()
ax = plt.subplot(121)
plt.imshow(tc[1], origin="lower")
ax.set_aspect("equal")

ax = plt.subplot(122)
plt.imshow(np.exp(Z[...,1]), origin="lower")
ax.set_aspect("equal")
plt.tight_layout()



# %%
# # plot position only

firing_rate_position = model_jax_position.predict(X)
firing_rate_pred_position = nap.TsdFrame(t=t_index[window_size:], d=firing_rate_position)
tc, binsxy = nap.compute_2d_tuning_curves_continuous(firing_rate_pred_position, position_interp_nap, 15)

figure()
plt.suptitle("prediction position")
for i in tc.keys():
    ax = subplot(2,4,i+1)
    imshow(tc[i], origin="lower")
    ax.set_aspect("equal")
plt.tight_layout()