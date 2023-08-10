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


#####################################
# Pynapple
#####################################

nwb = nap.NWBFile(io.read())

units = nwb["units"]

position = nwb["SpatialSeriesLED1"]

tc, binsxy = nap.compute_2d_tuning_curves(units, position, 15)


figure()
for i in tc.keys():
    ax = subplot(2,4,i+1)
    imshow(tc[i], origin="lower")
    ax.set_aspect("equal")
#show()

figure()
for i in units.keys():
    ax = subplot(2,4,i+1)
    plot(position['y'], position['x'])
    spk_pos = units[i].value_from(position)
    plot(spk_pos["y"], spk_pos["x"], 'o', color = 'red', markersize = 1, alpha = 0.5)
    ax.set_aspect("equal")
plt.tight_layout()
show()


#####################################
# GLM
#####################################

# create the binning
counts = units.count(0.005, ep=position.time_support)

# linear interp position
position_interp = np.zeros((counts.shape[0], 2))
position_interp[:, 0] = interp1d(position.times(), position.x)(counts.times())
position_interp[:, 1] = interp1d(position.times(), position.y)(counts.times())

# convert to jax
counts = jnp.asarray(counts, dtype=jnp.float32)
position_interp = jnp.asarray(position_interp, dtype=jnp.float32)

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
basis_2d = nsl.basis.MSplineBasis(n_basis_funcs=12, order=4) * \
            nsl.basis.MSplineBasis(n_basis_funcs=12, order=4)
position_basis = basis_2d.evaluate(position_interp[:, 0],
                                   position_interp[:, 1])

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

alpha = 1.
model_jax = nsl.glm.GLM(solver_name=solver,
                        inverse_link_function=jnp.exp,
                        alpha=alpha, solver_kwargs=solver_kwargs)
model_jax.fit(X, y, init_params=init_params)

# visualize output
_, _, Z = nsl.visualize.eval_response(basis_2d, model_jax.spike_basis_coeff_[:, -basis_2d.n_basis_funcs:], 30)
nsl.visualize.imshow_units(Z[1:-1, 1:-1], 2, 4)

# # Sklearn compatibility
# cls = model_selection.GridSearchCV(model_jax, param_grid={'alpha':[1., 10.]})
# cls.fit(X[:, :1, :], y)