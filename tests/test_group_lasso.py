import pytest

import numpy as np
import jax.numpy as jnp

import neurostatslib.glm as glm
def numpy_mult_masked(W, factor, mask):
    """
    Numpy version of the masking multiply, which multiply the weights of each neuron by
    the scaling factor in proximal gradient.

    Parameters
    ----------
    W
    factor
    mask

    Returns
    -------

    """
    W = np.array(W)
    factor= np.array(factor)
    mask = np.array(mask)
    for neu in range(W.shape[0]):
        for gr in range(factor.shape[1]):
            W[neu, :] = W[neu, :] * (factor[neu, gr] * mask[gr] + 1 - mask[gr])
    return W

def numpy_norm_masked(W, mask):
    norms = np.zeros((W.shape[0], mask.shape[0]))
    for neu in range(W.shape[0]):
        for gr in range(mask.shape[0]):
            wmask = W[neu, np.where(mask[gr])[0]]
            norms[neu, gr] = np.linalg.norm(wmask)
    return jnp.asarray(norms, dtype=np.float32)

def test_mask_mult(create_weights):
    mask, weights = create_weights
    norms = jnp.asarray(glm._vmap_norm2_masked_2(weights, mask), dtype=jnp.float32)
    res_numpy = numpy_mult_masked(weights, norms, mask)
    res_jax = glm._multiply_masked(weights, norms, mask)
    if not np.allclose(res_numpy, res_jax):
        raise ValueError("JAX and numpy masked multiply do not agree!")

def test_norm2(create_weights):
    mask, weights = create_weights
    norm_jax = jnp.asarray(glm._vmap_norm2_masked_2(weights, mask), dtype=jnp.float32)
    norm_numpy = numpy_norm_masked(weights, mask)
    if not np.allclose(norm_jax, norm_numpy):
        raise ValueError("JAX and numpy norm 2 do not agree!")

