import pynwb

from pynwb import NWBHDF5IO, TimeSeries
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV

from dandi.dandiapi import DandiAPIClient
import pynapple as nap
import numpy as np
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
    subplot(3,3,i+1)
    imshow(tc[i])
#show()

figure()
for i in units.keys():
    subplot(3,3,i+1)
    plot(position['x'], position['y'])
    spk_pos = units[i].value_from(position)
    plot(spk_pos["x"], spk_pos["y"], 'o', color = 'red', markersize = 1, alpha = 0.5)

show()


#####################################
# GLM
#####################################
# create the binning
t0 = position.time_support.start[0]
tend = position.time_support.end[0]
ts = np.arange(t0, tend + 0.005, 0.005)

# linear interp position
position_interp = np.zeros((ts.shape[0]-1, 2))
position_interp[:, 0] = interp1d(position.times(), position.x)(ts[:-1])
position_interp[:, 1] = interp1d(position.times(), position.y)(ts[:-1])

# bin spikes
binning = nap.IntervalSet(start=ts[:-1], end=ts[1:], time_units='s')
counts = jnp.asarray(units.count(ep=binning))
plt.close('all')
# convolve counts
window_size = 100
basis = nsl.basis.RaisedCosineBasisLog(n_basis_funcs=7)
x, eval_basis = basis.evaluate_on_grid(window_size)
plt.plot(x, eval_basis)
conv_counts = nsl.utils.convolve_1d_trials(eval_basis, counts[None, :, :])
conv_counts = nsl.utils.nan_pad_conv(conv_counts,
                                     window_size, filter_type="causal")

# evaluate position on 2D = basis
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
X = jnp.asarray(X)
y = jnp.asarray(y)

alpha = 2
model_jax = nsl.glm.GLM(solver_name="LBFGS",
                        inverse_link_function=jnp.exp,
                        alpha=alpha, solver_kwargs={'tol':10**-8,'maxiter':1000})
model_jax.fit(X[:,:1], y[:,:1])

XX, YY, Z = nsl.visualize.eval_response(basis_2d, model_jax.spike_basis_coeff_[:, -basis_2d.n_basis_funcs:], 30)

model = PoissonRegressor(alpha=alpha,tol=10**-8,solver="lbfgs",max_iter=1000,fit_intercept=True)
cls = model#GridSearchCV(model, param_grid={"alpha":[1, 0.5]})
cls.fit(X[:,0,:], y[:,0])
#model = cls.best_estimator_
XX, YY, Z = nsl.visualize.eval_response(basis_2d, model.coef_[None, -basis_2d.n_basis_funcs:], 30)

for k in range(Z.shape[2]):
    plt.figure()
    plt.imshow(Z[:, :, k])
# remove nans

# stack & fit