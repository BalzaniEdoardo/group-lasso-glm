import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import neurostatslib as nsl
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis
import matplotlib
from matplotlib.pyplot import *
import pynapple as nap
import sys
from scipy.linalg import hankel
from itertools import combinations
import pandas as pd
from sklearn.preprocessing import StandardScaler
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def offset_matrix(rate, binsize=0.01, windowsize = 0.1):
    rate = rate - rate.mean()
    rate = rate / rate.std()
    idx1 = -np.arange(0, windowsize + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, windowsize + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))

    # Build the Hankel matrix
    tmp = rate
    n_p = len(idx1)
    n_f = len(idx2)
    pad_tmp = np.pad(tmp, (n_p, n_f))
    offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]        

    return offset_tmp, time_idx

# pynapple data
data = nap.load_session("/mnt/home/gviejo/pynapple/your/path/to/A2929-200711", "neurosuite")
# data = nap.load_session("/mnt/ceph/users/gviejo/ADN/Mouse32/Mouse32-140822/", "neurosuite")

spikes = data.spikes
position = data.position
wake_ep = data.epochs["wake"]


# spikes = spikes[[0,6]]

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(
    spikes, position["ry"], 120, minmax=(0, 2 * np.pi)
)
peaks = tuning_curves.idxmax()

#####################################################################
# Parameters
#####################################################################
bin_size = 0.1
ws = 20


count = spikes.getby_category("location")['adn'].count(bin_size, wake_ep)
count = count.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=1)
tokeep = count.columns.values
peaks = peaks[tokeep]

X = []
Y = []
offset = []
pairs = list(combinations(tokeep, 2))

for p in pairs:
    mua, time_idx = offset_matrix(count[list(set(tokeep)-set([p[0]]))].sum(1), bin_size, bin_size*ws)
    X.append(np.hstack((mua, offset_matrix(count[p[0]].values, bin_size, bin_size*ws)[0])))
    Y.append(count[p[1]].values)
    a = peaks[p[0]] - peaks[p[1]]
    offset.append(np.abs(np.arctan2(np.sin(a), np.cos(a))))

offset = pd.Series(index = pairs, data = offset)

X = np.array(X)
X = np.transpose(X, (1, 2, 0))
Y = np.array(Y).T


#####################################################################
# Fitting the GLM
#####################################################################
# betas = np.random.randn(len(time_idx), 1)
betas = (jnp.zeros((X.shape[1], X.shape[2],1)), jnp.zeros((X.shape[2])))


def loss(params, X, y):    
    P = jax.nn.softplus(jnp.einsum("tdn,dnj->tn", X, params[0]) + params[1])
    x = y * jnp.log(P)
    x = jnp.where(jnp.isnan(x), jnp.zeros_like(x), x)
    return jnp.sum(P - x + jax.scipy.special.gammaln(y+1))

solver = getattr(jaxopt, "GradientDescent")(
    fun=loss,
    **dict(maxiter=1000, acceleration=False, verbose=True, stepsize=0.0)
    )


betas, state = solver.run(betas, X = X, y = Y)

Wmua = pd.DataFrame(index=time_idx, data=betas[0][0:len(time_idx),:,0], columns=pairs)
Wpai = pd.DataFrame(index=time_idx, data=betas[0][len(time_idx):,:,0], columns=pairs)

order = offset.sort_values().index.values

Wmua = Wmua[order]
Wpai = Wpai[order]

Wmua = (Wmua - Wmua.mean(0))/Wmua.std(0)
Wpai = (Wpai - Wpai.mean(0))/Wpai.std(0)

#####################################################################
# PLOT
#####################################################################

from scipy.ndimage import gaussian_filter

figure()
subplot(121)
tmp = gaussian_filter(Wmua.values.T, 1)
imshow(tmp, aspect='auto')
subplot(122)
tmp = gaussian_filter(Wpai.values.T, 1)
imshow(tmp, aspect='auto')
show()


sys.exit()



figure()
for i in spikes:
    subplot(3, 5, i + 1, projection="polar")
    plot(tuning_curves[i])

figure()
subplot(311)
plot(spike_data[0])
plot(np.arange(ws, spike_data.shape[1]+1), pred[0])
ax = subplot(312)
imshow(spike_data[:,ws:], aspect='auto', cmap = 'jet')
subplot(313, sharex = ax)
imshow(pred, aspect='auto', cmap = 'jet')


figure()
for n in range(n_neurons):
    subplot(3,3,n+1)
    tmp = B.T @ glm.spike_basis_coeff_[:,:,n].T
    # tmp = tmp[::-1]
    t = np.arange(0, len(tmp))*-bin_size
    plot(t, tmp)
    axhline(0, dashes=[2, 2], color='k')
# ion()


figure()
plot(glm.spike_basis_matrix.T)

figure()
for i in range(n_neurons):
    subplot(3,7,i+1, projection='polar')
    plot(tuning_curves[order[i]])    
for i in range(n_neurons):
    subplot(3,7,i+1+7)
    imshow(glm.spike_basis_coeff_[i,:,:].T)
    ylabel("Neurons")
    xlabel("Basis")
for i in range(n_neurons):
    subplot(3,7,i+1+7*2)
    imshow(glm.spike_basis_coeff_[:,:,i])    
    ylabel("Neurons")
    xlabel("Basis")

show()