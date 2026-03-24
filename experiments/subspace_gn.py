"""Utilities for Gauss-Newton updates in a random low-dimensional subspace.

This module is intentionally model-agnostic and can be used from JAX/Flax
experiments where parameters are represented as pytrees.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax

PyTree = Any
Array = jax.Array


def params_to_vec(params: PyTree) -> Tuple[Array, Callable[[Array], PyTree]]:
    """Flattens a parameter pytree and returns an unflatten function."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    flat_parts = [jnp.ravel(x) for x in leaves]
    sizes = [x.size for x in leaves]
    split_idx = jnp.cumsum(jnp.array(sizes[:-1]))
    flat = jnp.concatenate(flat_parts) if flat_parts else jnp.array([], dtype=jnp.float32)

    def unflatten(vec: Array) -> PyTree:
        parts = jnp.split(vec, split_idx) if len(sizes) > 1 else [vec]
        reshaped = [jnp.reshape(p, x.shape) for p, x in zip(parts, leaves)]
        return jax.tree_util.tree_unflatten(treedef, reshaped)

    return flat, unflatten


def random_subspace(key: Array, m_dim: int, k_dim: int) -> Array:
    """Returns a random orthonormal basis P in R^{M x k}."""
    if not (0 < k_dim <= m_dim):
        raise ValueError(f"Expected 0 < k_dim <= m_dim, got {k_dim=} and {m_dim=}")
    p, _ = jnp.linalg.qr(jax.random.normal(key, (m_dim, k_dim)))
    return p


def hvp_gn(
    params: PyTree,
    forward_fn: Callable[[PyTree, Array], Array],
    x: Array,
    vec: Array,
    unflatten_vec: Callable[[Array], PyTree],
) -> Array:
    """Computes a GN Hessian-vector product H v ≈ J^T J v using JVP + VJP."""
    v_tree = unflatten_vec(vec)

    def fwd(p: PyTree) -> Array:
        return forward_fn(p, x)

    _, jvp_out = jax.jvp(fwd, (params,), (v_tree,))
    _, vjp_fn = jax.vjp(fwd, params)
    jtjv_tree = vjp_fn(jvp_out / x.shape[0])[0]

    out, _ = params_to_vec(jtjv_tree)
    return out


def mse_loss(params: PyTree, forward_fn: Callable[[PyTree, Array], Array], x: Array, y: Array) -> Array:
    pred = forward_fn(params, x)
    return jnp.mean((pred - y) ** 2)


def gn_subspace_direction(
    params: PyTree,
    forward_fn: Callable[[PyTree, Array], Array],
    x: Array,
    grad_flat: Array,
    p_subspace: Array,
    damping: float,
    unflatten_fn: Callable[[Array], PyTree],
) -> Array:
    """Returns the full-space GN direction from a provided full gradient.

    This solves in the reduced subspace:
      (P^T H P + λI) Δu = -P^T g
      Δθ = P Δu
    """
    hvp_batch = jax.vmap(
        lambda v: hvp_gn(params, forward_fn, x, v, unflatten_fn),
        in_axes=1,
        out_axes=1,
    )(p_subspace)

    h_red = p_subspace.T @ hvp_batch
    k_dim = p_subspace.shape[1]
    h_red = h_red + damping * jnp.eye(k_dim, dtype=h_red.dtype)
    g_red = p_subspace.T @ grad_flat

    delta_u = jnp.linalg.solve(h_red, -g_red)
    return p_subspace @ delta_u


def gn_subspace_update(
    params: PyTree,
    forward_fn: Callable[[PyTree, Array], Array],
    x: Array,
    y: Array,
    p_subspace: Array,
    damping: float,
) -> Tuple[Array, Array, Callable[[Array], PyTree]]:
    """Returns (delta_theta, gradient, unflatten_fn) in the full flattened space."""
    flat_params, unflatten_fn = params_to_vec(params)

    grad_tree = jax.grad(mse_loss)(params, forward_fn, x, y)
    grad_flat, _ = params_to_vec(grad_tree)

    delta_theta = gn_subspace_direction(
        params=params,
        forward_fn=forward_fn,
        x=x,
        grad_flat=grad_flat,
        p_subspace=p_subspace,
        damping=damping,
        unflatten_fn=unflatten_fn,
    )

    if delta_theta.shape[0] != flat_params.shape[0]:
        raise ValueError("Subspace and parameter dimensions do not match.")

    return delta_theta, grad_flat, unflatten_fn


@partial(jax.jit, static_argnums=(1,))
def step_pure_subspace(
    params: PyTree,
    forward_fn: Callable[[PyTree, Array], Array],
    x: Array,
    y: Array,
    p_subspace: Array,
    damping: float,
    lr: float,
) -> PyTree:
    """Applies a pure subspace Gauss-Newton step."""
    delta_theta, _, unflatten_fn = gn_subspace_update(params, forward_fn, x, y, p_subspace, damping)
    flat_params, _ = params_to_vec(params)
    return unflatten_fn(flat_params + lr * delta_theta)


def step_hybrid_complement(
    params: PyTree,
    opt_state_comp: optax.OptState,
    opt_comp: optax.GradientTransformation,
    forward_fn: Callable[[PyTree, Array], Array],
    x: Array,
    y: Array,
    p_subspace: Array,
    damping: float,
    lr: float,
) -> Tuple[PyTree, optax.OptState]:
    """Applies GN in subspace and AdamW-like optimizer in the complement."""
    delta_theta, grad_flat, unflatten_fn = gn_subspace_update(
        params, forward_fn, x, y, p_subspace, damping
    )

    complement_g = grad_flat - p_subspace @ (p_subspace.T @ grad_flat)
    complement_g_tree = unflatten_fn(complement_g)

    updates, new_opt_state_comp = opt_comp.update(complement_g_tree, opt_state_comp, params)
    zero_like = jax.tree_util.tree_map(jnp.zeros_like, params)
    update_applied = optax.apply_updates(zero_like, updates)
    adam_update_vec, _ = params_to_vec(update_applied)

    flat_params, _ = params_to_vec(params)
    new_flat = flat_params + lr * delta_theta + adam_update_vec
    return unflatten_fn(new_flat), new_opt_state_comp
