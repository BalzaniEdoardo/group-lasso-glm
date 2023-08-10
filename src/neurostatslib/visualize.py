import matplotlib.pylab as plt
import numpy as np
import neurostatslib as nsl
from numpy.typing import NDArray
def eval_response(basis:nsl.basis.Basis, weights: NDArray, n_samples:int):
    res = basis.evaluate_on_grid(*[n_samples]*basis._n_input_dimensionality)
    return *res[:-1], np.einsum('...i, ji->...j', res[-1], weights)