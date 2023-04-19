import jax
import jax.numpy as jnp
import numpy as onp
import neurostatslib as nsl
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis
import matplotlib
from matplotlib.pyplot import *
import pynapple as nap
import sys
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# pynapple data
data = nap.load_session("/mnt/home/gviejo/pynapple/your/path/to/A2929-200711", "neurosuite")

spikes = data.spikes
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
bin_size = 0.3
ws = 100
nb = 5


count = spikes.getby_category("location")['adn'].count(bin_size, wake_ep)
order = peaks[count.columns].sort_values().index
count = count[order]
spike_data = jnp.array(count.values.T)#[0:2] #, np.newaxis]
n_neurons = spike_data.shape[0]


spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=ws)
sim_pts = nsl.sample_points.raised_cosine_log(nb, ws)
B = spike_basis.gen_basis_funcs(sim_pts)


#####################################################################
# Fitting the GLM
#####################################################################
glm = GLM(
    B, 
    solver_name="GradientDescent", 
    solver_kwargs=dict(maxiter=1000, acceleration=False, verbose=True, stepsize=0.0)
    )


glm.fit(spike_data)
pred = glm.predict(spike_data)

#####################################################################
# PLOT
#####################################################################
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