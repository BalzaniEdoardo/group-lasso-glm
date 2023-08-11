import jax.numpy as jnp
import neurostatslib as nsl
import sklearn.model_selection as model_selection
X,counts=jnp.array(10),jnp.array(10)
y,conv_counts,position_basis,window_size =jnp.array(10),jnp.array(10),jnp.array(10)
basis_2d = nsl.basis.Basis


# combine inputs
y, X = nsl.utils.combine_inputs(counts,
                                conv_counts,
                                position_basis[None, :, None],
                                strip_left=window_size,
                                reps=counts.shape[1])

# fit GLM
solver = 'BFGS'
solver_kwargs = {'tol': 10**-6, 'maxiter': 1000, 'jit':True}
init_params = jnp.zeros((y.shape[1], X.shape[2])), \
              jnp.log(jnp.mean(y, axis=0))

alpha = 0.1
model_jax = nsl.glm.GLM(solver_name=solver,
                        inverse_link_function=jnp.exp,
                        alpha=alpha,
                        solver_kwargs=solver_kwargs)

# %%
# ## Sklearn compatibility
cls = model_selection.GridSearchCV(model_jax, param_grid={'alpha':[0.1, 1., 10.]})
cls.fit(X, y)


