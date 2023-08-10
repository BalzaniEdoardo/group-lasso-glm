import matplotlib.pylab as plt
import numpy as np
import neurostatslib as nsl
from numpy.typing import NDArray

def eval_response(basis:nsl.basis.Basis, weights: NDArray, n_samples:int):
    res = basis.evaluate_on_grid(*[n_samples]*basis._n_input_dimensionality)
    return *res[:-1], np.einsum('...i, ji->...j', res[-1], weights)

def countrof_units(X, Y, Z, rows, cols):
    n_units = Z.shape[2]
    fig = plt.figure()
    for unt in range(n_units):
        ax = plt.subplot(rows, cols, unt+1)
        ax.set_aspect("equal")
        ax.contourf(X, Y, Z[:, :, unt])
        ax.set_title(f"unit {unt}")

    fig.tight_layout()

def imshow_units(Z, rows, cols):
    n_units = Z.shape[2]
    fig = plt.figure()
    for unt in range(n_units):
        ax = plt.subplot(rows, cols, unt+1)
        ax.set_aspect("equal")
        ax.imshow(Z[:, :, unt], origin="lower")
        ax.set_title(f"unit {unt}")

    fig.tight_layout()

