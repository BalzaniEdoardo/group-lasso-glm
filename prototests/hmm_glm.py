import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *
from sklearn.manifold import Isomap

sys.path.append("/home/gviejo/glmhmm/glmhmm/")
from glmhmm import hmm

Y = np.random.binomial(1, 0.3, 10)


N = 200 # number of data/time points
K = 2 # number of latent states
C = 2 # number of observation classes
true_HMM = hmm.HMM(N,0,C,K)
A_true,phi_true,pi_true = true_HMM.generate_params()

Y,Z = true_HMM.generate_data(A_true,phi_true)
Y = Y.astype('int')

scores = []
As = []


for _ in range(10):

    init = np.array([0.9, 0.1])
    A = np.array([[0.7, 0.3],[0.3, 0.7]])
    B = np.array([[0.4, 0.6], [0.8, 0.2]])

    init = np.random.rand(2)
    init = init/init.sum()
    A = np.random.rand(2,2)
    A = A/A.sum(1)[:,None]
    B = np.random.rand(2,2)
    B = B/B.sum(1)[:,None]

    # Y = np.array([0, 1, 0, 0])
    T = len(Y)

    score = []

    for i in range(100):
        
        # Forward
        alpha = np.zeros((T, K))
        alpha[0] = init*B[:,Y[0]]        
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t-1], A)*B[:,Y[t]]
        # scaling = alpha.sum(1)[:,None]
        # alpha = alpha/scaling

        # Backward    
        beta = np.zeros((T, K))
        beta[-1] = 1
        for t in np.arange(0, T-1)[::-1]:
            beta[t] = np.dot(A, beta[t+1]*B[:,Y[t+1]])
        # beta = beta/beta.sum(1)[:,None]

        # State posterior
        gamma = alpha*beta
        gamma = gamma/gamma.sum(1)[:,None]

        # Segment posterior
        E = np.tile(A, (T-1, 1, 1)) * alpha[0:-1,:,None]*beta[1:,None,:]
        E = E * np.tile(B[:,Y[1:]].T[:,None,:], (1, K, 1)) # Adding emission    
        E = E / E.sum((1,2))[:,None,None]

        # Maximisation
        init = gamma[0]
        A = E.sum(0)
        A = A/A.sum(1)[:,None]

        for j, o in enumerate(np.unique(Y)):
            B[:,j] = gamma[Y == o].sum(0)/gamma.sum(0)        

        # log
        logprob = np.sum(np.log(gamma[-1]))

        score.append(logprob)


    score = np.array(score)
    scores.append(score)
    As.append(A)

scores = np.array(scores)

figure()
plot(scores.T)
show()

# T = 5000   # number of datapoints
# N = 12


# bins = np.linspace(0, 2*np.pi, 61)

# alpha = np.digitize(np.cumsum(np.random.randn(T)*0.5)%(2*np.pi), bins)-1


# x = np.linspace(-np.pi, np.pi, len(bins)-1)
# tmp = np.exp(-x**2)
# tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T


# Y = np.random.poisson(tc[alpha]*10)


# imap = Isomap(n_components=2, n_neighbors = 10).fit_transform(Y)





# # create data
# X = .3*np.random.randn(n, p)
# true_b = np.random.randn(p)
# y = np.random.poisson(np.exp(np.dot(X, true_b)))

# # loss function and gradient
# def f(b):
#     Xb = np.dot(X, b)
#     exp_Xb = np.exp(Xb)
#     loss = exp_Xb.sum() - np.dot(y, Xb)
#     grad = np.dot(X.T, exp_Xb - y)
#     return loss, grad

# # hessian
# def hess(b):
#     return np.dot(X.T, np.exp(np.dot(X, b))[:, None]*X)

# # optimize
# result = minimize(f, np.zeros(p), jac=True, hess=hess, method='newton-cg')

# print('True regression coeffs: {}'.format(true_b))
# print('Estimated regression coeffs: {}'.format(result.x))