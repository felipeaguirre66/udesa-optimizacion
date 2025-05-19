import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn

import matplotlib.pyplot as plt

from nn_functions import init_network_params, pack_params, layer_sizes
from nn_functions import update_rmsprop, update_sgd, update_adam
from nn_functions import get_batches, loss, batched_predict
from nn_functions import top_eigenvalue

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
num_epochs = 15
params = init_network_params(layer_sizes, random.key(0))
params = pack_params(params)

# optimizer
update = update_adam
# update = update_rmsprop
# update = update_sgd
step_size = 0.001

# initialize gradients
xi, yi = next(get_batches(xx, ff, bs=32))
x0 = xi.copy()
y0 = yi.copy()
grads = grad(loss)(params, x0, y0)
aux = jnp.square(grads)
aux = (0,0)

log_train = []
grads_norm = []
hess_max = []
all_hidden_activations = []

# Initial snapshot
all_hidden_activations.append(batched_predict(params, xx)[1])
hess_max.append(float(top_eigenvalue(params, xx, ff)))

for epoch in range(num_epochs):
    grads_epoch = []

    # shuffle & batch‐updates
    idxs = random.permutation(random.PRNGKey(epoch), xx.shape[0])
    for xi, yi in get_batches(xx[idxs], ff[idxs], bs=32):
        params, aux, grads = update(params, xi, yi, step_size, aux)
        grads_epoch.append(jnp.linalg.norm(grads))

    all_hidden_activations.append(batched_predict(params, xx)[1])
    grads_norm.append(jnp.mean(jnp.array(grads_epoch)))
    hess_max.append(float(top_eigenvalue(params, xx, ff)))
    train_loss = loss(params, xx, ff)
    log_train.append(train_loss)
    print(f"Epoch {epoch} — Loss: {train_loss:.4e}  |  ‖grad‖: {grads_norm[-1]:.4e}  |  λ_max(H): {hess_max[-1]:.4e}")

# Plot loss function
plt.figure()
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.semilogy(log_train)

# Plot gradient norm
plt.figure()
plt.title('Gradient Norm')
plt.xlabel('Epochs')
plt.ylabel('Gradient Norm')
plt.plot(grads_norm)

# Plot results
plt.figure()
plt.title('Original Field')
plt.imshow(ff.reshape((nx, ny)).T, origin='lower', cmap='jet')

plt.figure()
plt.title('Predicted Field')
plt.imshow(batched_predict(params, xx)[0].reshape((nx, ny)).T, origin='lower', cmap='jet')

# Especifica los epochs que quieres plotear
epochs_to_plot = [0, 5, 10, 15]
fig, axes = plt.subplots(2, 2, figsize=(12*.75, 10*.75), squeeze=False)
for ax, epoch in zip(axes.flatten(), epochs_to_plot):
    hidden_activations = all_hidden_activations[epoch]
    flat0 = hidden_activations[0].ravel()
    flat1 = hidden_activations[1].ravel()
    ax.hist(flat0, bins=50, alpha=0.5, density=True, label='Capa 1')
    ax.hist(flat1, bins=50, alpha=0.5, density=True, label='Capa 2')
    ax.set_title(f'Epoch {epoch}')
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize='small')

plt.suptitle('Hidden Activations at Different Epochs', fontsize=16)
plt.tight_layout()

# --- plot how λ_max evolves ---
plt.figure()
plt.plot(hess_max)
plt.xlabel("Epoch")
plt.ylabel("Approx. λ_max(H)")
plt.title("Top Hessian Eigenvalue over Training")

# Show all plots
plt.show()