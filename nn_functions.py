import jax.numpy as jnp
from jax import grad, jit, vmap, jvp
from jax import random
from jax import nn
from jax import hessian
from jax.numpy.linalg import eigvalsh
from functools import partial
from jax import jit, grad
import matplotlib.pyplot as plt

def pack_params(params):
    """Pack parameters into a single vector."""
    return jnp.concatenate([jnp.ravel(w) for w, _ in params] +
                             [jnp.ravel(b) for _, b in params])

layer_sizes = [2, 64, 64, 1]
def unpack_params(params):
    """Unpack parameters from a single vector."""
    weights = []
    for i in range(len(layer_sizes) - 1):
        weight_size = layer_sizes[i] * layer_sizes[i + 1]
        to_unpack, params = params[:weight_size], params[weight_size:]
        weights.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1], layer_sizes[i]))

    biases = []
    for i in range(len(layer_sizes) - 1):
        bias_size = layer_sizes[i + 1]
        to_unpack, params = params[:bias_size], params[bias_size:]
        biases.append(jnp.array(to_unpack).reshape(layer_sizes[i + 1]))

    params = [(w, b) for w, b in zip(weights, biases)]
    return params

def random_layer_params(m, n, key, scale=1e-2):
    ''' Randomly initialize weights and biases for a dense neural network layer '''
    w_key, b_key = random.split(key)
    scale = jnp.sqrt(6.0 / (m + n))
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    # return jnp.ones((n, m)), jnp.zeros((n,))

def init_network_params(sizes, key):
    ''' Initialize all layers for a fully-connected neural network with sizes "sizes" '''
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def predict(params, coord):
    params = unpack_params(params)
    hidden = coord
    hidden_activations = {}
    for i, (w, b) in enumerate(params[:-1]):
        outputs = jnp.dot(w, hidden) + b
        hidden = nn.tanh(outputs)
        hidden_activations[i] = hidden

    final_w, final_b = params[-1]
    output = jnp.dot(final_w, hidden) + final_b
    return output, hidden_activations
batched_predict = vmap(predict, in_axes=(None, 0))

# def loss(params, coord, target):
#     preds, _ = batched_predict(params, coord)
#     return jnp.mean(jnp.square(preds - target))

def loss(params, coord, target, lmbda=0.0, reg_type='ridge'):
    """
    params   : packed parameter vector
    coord    : input coords
    target   : true outputs
    lmbda    : regularization strength
    reg_type : 'ridge' or 'lasso'
    """
    # mean‐squared error
    preds, _ = batched_predict(params, coord)
    mse = jnp.mean((preds - target)**2)

    # choose penalty
    if reg_type == 'ridge':
        penalty = jnp.sum(params**2)
    elif reg_type == 'lasso':
        penalty = jnp.sum(jnp.abs(params))
    else:
        raise ValueError(f"Unknown reg_type {reg_type!r}; use 'ridge' or 'lasso'")

    return mse + lmbda * penalty

@partial(jit, static_argnames=('loss_fn',))
def update_sgd(params, x, y, step, aux, loss_fn, **args):
    grads  = grad(loss_fn)(params, x, y)
    params = params - step * grads
    return params, aux, grads

@partial(jit, static_argnames=('loss_fn',))
def update_rmsprop(params, x, y, step_size, aux, loss_fn, **args):
    beta = 0.9
    grads = grad(loss_fn)(params, x, y)
    aux = beta * aux + (1 - beta) * jnp.square(grads)
    step_size = step_size / (jnp.sqrt(aux) + 1e-8)
    params = params - step_size * grads
    return params, aux, grads

@partial(jit, static_argnames=('loss_fn',))
def update_adam(params, x, y, step_size, aux, loss_fn, **args):
    beta1 = args.get('beta1', 0.9)
    beta2 = args.get('beta2', 0.999)
    eps = args.get('eps', 1e-8)

    m, v = aux
    grads = grad(loss_fn)(params, x, y)
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * jnp.square(grads)

    m_hat = m / (1 - beta1)
    v_hat = v / (1 - beta2)

    params = params - step_size * m_hat / (jnp.sqrt(v_hat) + eps)
    return params, (m, v), grads


def get_batches(x, y, bs):
    for i in range(0, len(x), bs):
        yield x[i:i+bs], y[i:i+bs]


# --- Hessian-vector product for a fixed (x_h, y_h) ---
def hvp(params, x, y, v):
    """Compute H(p)·v where H = ∇²_p L(p, x, y)."""
    # gradient function p -> ∇_p L(p,x,y)
    grad_fn = lambda p: grad(loss)(p, x, y)
    # jvp out = (grad(p), H·v)
    _, hv = jvp(grad_fn, (params,), (v,))
    return hv

@jit
def top_eigenvalue(params, x, y, num_iters=20):
    """Power iteration to get the largest Hessian eigenvalue."""
    key = random.PRNGKey(0)
    # random init vector same shape as params
    v = random.normal(key, params.shape)
    v = v / jnp.linalg.norm(v)

    # iterate
    for _ in range(num_iters):
        Hv = hvp(params, x, y, v)
        v = Hv / (jnp.linalg.norm(Hv) + 1e-12)

    # Rayleigh quotient ≈ λ_max
    Hv = hvp(params, x, y, v)
    return jnp.vdot(v, Hv)