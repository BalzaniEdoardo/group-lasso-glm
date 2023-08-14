import pytest

import jax.numpy as jnp
import numpy as np

@pytest.fixture
def create_weights():
    np.random.seed(101)
    n_groups = 3
    n_features = 20
    n_neurons = 10
    group_id = [(0, 3), (3, 7), (7, 20)]
    mask = np.zeros((n_groups, n_features))
    for gr in range(n_groups):
        mask[gr, group_id[gr][0]:group_id[gr][1]] = 1

    mask = jnp.asarray(mask, dtype=jnp.float32)
    weights = jnp.asarray(np.random.normal(size=(n_neurons, n_features)), dtype=jnp.float32)
    return mask, weights