import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn

import matplotlib.pyplot as plt
from functools import partial

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import update_rmsprop, update_sgd, update_adam
from nn_functions import get_batches, loss, batched_predict
from nn_functions import top_eigenvalue

def train_nn(
    update_method='adam',
    num_epochs=15,
    step_size=0.001,
    use_adaptive_step_size=False,
    batch_size=32,
    regularization=None,
    optimizer_params=None,
):
    # Load data
    field = jnp.load('field.npy')
    field = field - field.mean()
    field = field / field.std()
    field = jnp.array(field, dtype=jnp.float32)
    nx, ny = field.shape
    xx = jnp.linspace(-1, 1, nx)
    yy = jnp.linspace(-1, 1, ny)
    xx, yy = jnp.meshgrid(xx, yy, indexing='ij')
    xx = jnp.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    ff = field.reshape(-1, 1)

    # Parameters
    params = init_network_params(layer_sizes, random.key(0))
    params = pack_params(params)

    # initialize gradients
    if regularization == 'ridge':
        loss_fn = partial(loss, lmbda=1e-4, reg_type='ridge')
    elif regularization == 'lasso':
        loss_fn = partial(loss, lmbda=5e-5, reg_type='lasso')
    else:
        loss_fn = partial(loss, lmbda=0, reg_type='lasso')
    xi, yi = next(get_batches(xx, ff, bs=batch_size))
    x0 = xi.copy()
    y0 = yi.copy()
    grads = grad(loss_fn)(params, x0, y0)

    # Initialize optimizer
    if update_method == 'adam':
        update = update_adam
        aux = (0,0)
    elif update_method == 'rmsprop':
        update = update_rmsprop
        aux = jnp.square(grads)
    elif update_method == 'sgd':
        update = update_sgd
    else:
        raise ValueError("Invalid update method. Choose 'adam', 'rmsprop', or 'sgd'.")
    
    log_train = []
    grads_norm = []
    hess_max = []
    all_hidden_activations = []

    # Initial snapshot
    all_hidden_activations.append(batched_predict(params, xx)[1])
    hess_max.append(float(top_eigenvalue(params, xx, ff)))

    for epoch in range(num_epochs):
        grads_epoch = []

        if use_adaptive_step_size:
            k = epoch // 5
            step_size = step_size * (.5 ** k)
        
        # shuffle & batch‐updates
        idxs = random.permutation(random.PRNGKey(epoch), xx.shape[0])
        for xi, yi in get_batches(xx[idxs], ff[idxs], bs=batch_size):
            params, aux, grads = update(params, xi, yi, step_size, aux, loss_fn, **optimizer_params)
            grads_epoch.append(jnp.linalg.norm(grads))

        all_hidden_activations.append(batched_predict(params, xx)[1])
        grads_norm.append(jnp.mean(jnp.array(grads_epoch)))
        hess_max.append(float(top_eigenvalue(params, xx, ff)))
        train_loss = loss(params, xx, ff)
        log_train.append(train_loss)
        print(f"Epoch {epoch} — Loss: {train_loss:.4e}  |  ‖grad‖: {grads_norm[-1]:.4e}  |  λ_max(H): {hess_max[-1]:.4e}")
    
    return params, log_train, grads_norm, hess_max, all_hidden_activations