import jax
import jax.numpy as jnp
import numpy as onp
import neurostatslib as nsl
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis, MSplineBasis
import matplotlib
from matplotlib.pyplot import *
import pynapple as nap
import sys
import jax.numpy as jnp
from functools import partial
import jaxopt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# pynapple data
#data = nap.load_session("/mnt/home/gviejo/pynapple/your/path/to/A2929-200711", "neurosuite")

data = nap.load_session("/Users/gviejo/pynapple/your/path/to/A2929-200711", "neurosuite")



spikes = data.spikes.getby_category("location")["adn"]
position = data.position
wake_ep = data.epochs["wake"]

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(
    spikes, position["ry"], 120, minmax=(0, 2 * np.pi)
)
peaks = tuning_curves.idxmax()
#####################################################################
# Parameters
#####################################################################
bin_size = 0.01
ws = 200
nb = 3


count = spikes.getby_category("location")['adn'].count(bin_size, wake_ep)
order = peaks[count.columns].sort_values().index
count = count[order]
spike_data = jnp.array(count.values.T)[[0, 4]] #, np.newaxis]
n_neurons = spike_data.shape[0]

# Getting the spike basis function
spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=ws)
sim_pts = nsl.sample_points.raised_cosine_log(nb, ws)
B = spike_basis.gen_basis_funcs(sim_pts)

B = B[::-1]

# spike_basis = MSplineBasis(n_basis_funcs=nb,window_size=ws)
# sim_pts = nsl.sample_points.raised_cosine_linear(nb, ws)
# B = spike_basis.gen_basis_funcs(sim_pts)


#####################################################################
# Fitting the GLM
#####################################################################
# glm = GLM(
#     B, 
#     solver_name="GradientDescent", 
#     solver_kwargs=dict(maxiter=1000, acceleration=False, verbose=True, stepsize=0.0)
#     )


# glm.fit(spike_data)
# pred = glm.predict(spike_data)

spike_basis = B

n_neurons, _ = spike_data.shape
n_basis_funcs, window_size = spike_basis.shape

# Convolve spikes with basis functions. We drop the last sample, as
# those are the features that could be used to predict spikes in the
# next time bin
# Broadcasted 1d convolution operations.
# [[n x t],[w]] -> [n x (t - w + 1)]
_CORR1 = jax.vmap(partial(jnp.convolve, mode='valid'), (0, None), 0)
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_CORR2 = jax.vmap(_CORR1, (None, 0), 0)

X = _CORR2(jnp.atleast_2d(spike_basis),jnp.atleast_2d(spike_data))

X = X[:,:,:-1]

# Initial parameters
init_params = (jnp.zeros((n_neurons, n_basis_funcs, n_neurons)),jnp.zeros(n_neurons))

# Loss functions
def score(predicted_firing_rates, target_spikes):
    x = target_spikes * jnp.log(predicted_firing_rates)
    # this is a jax jit-friendly version of saying "put a 0 wherever
    # there's a NaN". we do this because NaNs result from 0*log(0)
    # (log(0)=-inf and any non-zero multiplied by -inf gives the expected
    # +/- inf)
    x = jnp.where(jnp.isnan(x), jnp.zeros_like(x), x)
    # see above for derivation of this.
    return jnp.mean(predicted_firing_rates - x + jax.scipy.special.gammaln(target_spikes+1))

def loss(params, X, y):
    Ws, bs = params
    predicted_firing_rates = jax.nn.softplus(
        jnp.einsum("nbt,nbj->nt", X, Ws) + bs[:, None]
    )
    return score(predicted_firing_rates, y)


# Run optimization
solver = jaxopt.GradientDescent(
    fun=loss, maxiter=1000, acceleration=False, verbose=True, stepsize=0.0
    )

params, state = solver.run(init_params, X=X,
                           y=spike_data[:,:-ws])


sys.exit()




#####################################################################
# PLOT
#####################################################################
figure()
for i in spikes:
    subplot(3, 5, i + 1, projection="polar")
    plot(tuning_curves[i])

# figure()
# subplot(311)
# plot(spike_data[0])
# plot(np.arange(ws, spike_data.shape[1]+1), pred[0])
# ax = subplot(312)
# imshow(spike_data[:,ws:], aspect='auto', cmap = 'jet')
# subplot(313, sharex = ax)
# imshow(pred, aspect='auto', cmap = 'jet')


figure()
for n in range(n_neurons):
    subplot(3,3,n+1)
    tmp = B.T @ params[0][:,:,n].T
    # tmp = tmp[::-1]
    t = np.arange(0, len(tmp))*-bin_size
    plot(t, tmp)
    axhline(0, dashes=[2, 2], color='k')
# ion()


figure()
plot(B.T)

figure()
for i in range(n_neurons):
    subplot(3,7,i+1, projection='polar')
    plot(tuning_curves[order[i]])    
for i in range(n_neurons):
    subplot(3,7,i+1+7)
    imshow(params[0][i,:,:].T)
    ylabel("Neurons")
    xlabel("Basis")
for i in range(n_neurons):
    subplot(3,7,i+1+7*2)
    imshow(params[0][:,:,i])    
    ylabel("Neurons")
    xlabel("Basis")


figure()
ax = subplot(nb+1, 1, 1)
plot(spike_data[0,ws:])
for i in range(nb):
    subplot(nb+1,1,i+2, sharex = ax)
    plot(X[0,i,:])

show()




