# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-05-19 13:29:18
# @Last Modified by:   gviejo
# @Last Modified time: 2023-05-28 23:55:34
import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA
from scipy.ndimage import gaussian_filter1d

import neurostatslib as nsl
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis, MSplineBasis

# Jax import ##########################
import jax
import jax.numpy as jnp
from functools import partial
import jaxopt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
#######################################


T = 5000 # number of data/time points
K = 2 # number of latent states
N = 12 # Number of neurons

#######################################
# Generate the data
#######################################
bins = np.linspace(0, 2*np.pi, 121)
alpha = np.digitize(
    gaussian_filter1d(np.cumsum(np.random.randn(T)*0.5), 3)
    %(2*np.pi), bins
    )-1
x = np.linspace(-np.pi, np.pi, len(bins)-1)
tmp = np.roll(np.exp(-(1.0*x)**2), (len(bins)-1)//2)
tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T
Y = np.random.poisson(tc[alpha]*3)


tcr = np.random.rand(120, N)*np.mean(Y.sum(0)/T)
Yr = np.random.poisson(tcr[alpha]*2)

#######################################
# Getting the spike basis function
#######################################
ws = 100
nb = 8

spike_basis = RaisedCosineBasis(n_basis_funcs=nb,window_size=ws)
sim_pts = nsl.sample_points.raised_cosine_linear(nb, ws)
B = spike_basis.gen_basis_funcs(sim_pts)

spike_basis = B

n_basis_funcs, window_size = spike_basis.shape

#######################################
# FITTING BOTH GLM
#######################################
_CORR1 = jax.vmap(partial(jnp.convolve, mode='same'), (0, None), 0)
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_CORR2 = jax.vmap(_CORR1, (None, 0), 0)

def loss(b, X, Y):
    Xb = jnp.dot(X, b) 
    exp_Xb = jnp.exp(Xb)
    loss = jnp.sum(exp_Xb, 0) - jnp.sum(Y*Xb, 0) # + jax.scipy.special.gammaln(Y+1)
    # penalty = jnp.sqrt(jnp.sum(jnp.maximum(b**2, 0), 0))        
    return jnp.mean(loss) + 0.5*jnp.sum(jnp.abs(b))
    # return jnp.mean(loss) + 0.1*jnp.sum(jnp.sqrt(jnp.maximum(0, jnp.sum(b**2.0, 1))))

Ws = []
W2s = []

for spike_data in [Y.T, Yr.T]:

    n_neurons, _ = spike_data.shape

    X = _CORR2(jnp.atleast_2d(spike_basis),jnp.atleast_2d(spike_data))
    X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
    X = X.T
    X = X - X.mean(0)
    X = X / X.std(0)
    X = np.hstack((X, jnp.ones((len(X), 1))))

    b0 = jnp.zeros(((n_neurons*n_basis_funcs)+1, n_neurons))

    solver = jaxopt.GradientDescent(
        fun=loss, maxiter=20000, acceleration=False, verbose=True, stepsize=0.0
        )

    W, state = solver.run(b0, X=X, Y=Y)
    W2 = W[0:-1].reshape((n_neurons, n_basis_funcs, n_neurons))

    Ws.append(W)
    W2s.append(W2)

############################################
# MIXING THE SPIKE DATA
############################################
init = np.array([0.5, 0.5])
trueA = np.array([[0.7, 0.3],[0.3, 0.7]])
Z = np.zeros(T*2, dtype='int')
m, n = (1, 0)
Yt = [Y[0]]
for i in range(1, T*2):
    Z[i] = np.sum(np.random.rand()>np.cumsum(trueA[Z[i-1]]))
    if Z[i] == 0:        
        Yt.append(Y[m])
        m+=1
    else:        
        Yt.append(Yr[n])
        n+=1    
    if m == T: m = 0
    if n == T: n = 0
        
Yt = np.array(Yt)

# Z = np.zeros(len(Yt))
# Z[2000:] = 1

# imap = Isomap(n_components=2, n_neighbors = 50).fit_transform(Yt)

# scatter(imap[:,0], imap[:,1], c = Z[0:len(Yt)])
# show()

# sys.exit()

############################################
# FITTING THE HMM
############################################

X = _CORR2(jnp.atleast_2d(spike_basis),jnp.atleast_2d(Yt.T))
X = X.reshape(np.prod(X.shape[0:2]), X.shape[-1])
X = X.T
X = X - X.mean(0)
X = X / X.std(0)
X = np.hstack((X, jnp.ones((len(X), 1))))

O = np.array([
    np.sum(np.exp(np.dot(X, Ws[i])), 1) for i in range(2)
    ]).T

# O = np.ones_like(O)
# O *= 0.5

scores = []
As = []
Zs = []

for _ in range(5):

    score = []

    # init = np.array([0.9, 0.1])
    # A = np.array([[0.3, 0.7],[0.4, 0.6]])
    # B = np.array([[0.8, 0.2],[0.4, 0.6]])

    init = np.random.rand(K)
    init = init/init.sum()
    A = np.random.rand(K, K)
    A = A/A.sum(1)[:,None]

    T = len(Yt)

    for i in range(100):
        
        # Forward
        alpha = np.zeros((T, K))
        scaling = np.zeros(T)
        alpha[0] = init#*B[:,Y[0]]
        scaling[0] = alpha[0].sum()
        alpha[0] = alpha[0]/scaling[0]
        for t in range(1, T):
            # alpha[t] = np.dot(alpha[t-1], A)*B[:,Y[t]]
            alpha[t] = np.dot(alpha[t-1], A)*O[t]
            scaling[t] = alpha[t].sum()
            alpha[t] = alpha[t]/scaling[t]

        # Backward    
        beta = np.zeros((T, K))
        beta[-1] = 1/scaling[-1]
        for t in np.arange(0, T-1)[::-1]:
            # beta[t] = np.dot(A, beta[t+1]*B[:,Y[t+1]])
            beta[t] = np.dot(A, beta[t+1]*O[t+1])
            beta[t] = beta[t]/scaling[t]

        # Expectation
        E = np.tile(A, (T-1, 1, 1)) * alpha[0:-1,:,None]*beta[1:,None,:]
        # E = E * np.tile(B[:,Y[1:]].T[:,None,:], (1, K, 1)) # Adding emission    
        E = E * np.tile(O[1:][:,None,:], (1, K, 1)) # Adding emission    

        G = np.zeros((T, K))
        G[0:-1] = E.sum(-1)
        G[-1] = alpha[-1]

        # Maximisation
        init = G[0]
        A = E.sum(0)/(G[0:-1].sum(0)[:,None])

        # for j, o in enumerate(np.unique(Y)):
        #     B[:,j] = G[Y == o].sum(0)/G.sum(0)

        score.append(np.sum(np.log(scaling)))

    # # Sampling the sequence
    # alpha = np.zeros((T, K))
    # scaling = np.zeros(T)
    # alpha[0] = init*O[0]
    # alpha = al
    # alpha[0] = alpha[0]/scaling[0]
    # for t in range(1, T):
    #     # alpha[t] = np.dot(alpha[t-1], A)*B[:,Y[t]]
    #     alpha[t] = np.dot(alpha[t-1], A)*O[t]
    #     scaling[t] = alpha[t].sum()
    #     alpha[t] = alpha[t]/scaling[t]


    scores.append(score)
    As.append(A)
    Zs.append(np.argmax(G, 1))
    # Bs.append(B)

scores = np.array(scores).T
As = np.array(As)
# Bs = np.array(Bs)

A = As[np.argmax(scores[-1])]
# B = Bs[np.argmax(scores[-1])]
bestZ = Zs[np.argmax(scores[-1])]


# imap = Isomap(n_components=3, n_neighbors = 50).fit_transform(Yt)
imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(Yt)
# scatter(imap[:,0], imap[:,1])
# show()



figure()
subplot(121)
scatter(imap[:,0], imap[:,1], 1, c = Z)
subplot(122)
scatter(imap[:,0], imap[:,1], 1, c = bestZ)


figure()
plot(Z)
plot(bestZ)

figure()
gs = GridSpec(3, len(B))
subplot(gs[0,0:2])
imshow(tc,aspect='auto')
subplot(gs[0,2:4])
imshow(W2s[0].mean(1), aspect='auto')
subplot(gs[0,4:6])
imshow(W2s[1].mean(1), aspect='auto')

for i in range(n_basis_funcs):
    subplot(gs[1,i])
    x = np.arange(B.shape[1])*-1
    plot(x, B[i])
    subplot(gs[2,i])
    im = imshow(W2s[0][:,i,:].T,aspect='auto',vmin=W2s[0].min(),vmax=W2s[0].max())

colorbar(im)

show()
