# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Muon.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""


from typing import NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from utils import prune_tree


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
    scale_eps: bool = False,
    din: Optional[jax.Array] = None,
    dout: Optional[jax.Array] = None,
    N: Optional[jax.Array] = None,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.

  Returns:
    The orthogonalized matrix.
  """
  if x.ndim != 2:
    raise ValueError(f'Input must have shape (m, n), got {x.shape}')
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError(
        'Newton-Schulz coefficients must have shape (3,) or (n, 3), '
        f'got {ns_coeffs.shape}'
    )
  if scale_eps:
    eps = eps * (din / dout) ** 0.5 / N
  def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x
  transposed = False
  if x.shape[0] > x.shape[1]:
    x = x.T
    transposed = True
  x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
  ns_coeffs = ns_coeffs.astype(x.dtype)
  if ns_coeffs.ndim == 1:
    x = jax.lax.fori_loop(
        0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x
    )
  else:
    x, _ = jax.lax.scan(
        lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs
    )
  if transposed:
    x = x.T
  return x


class MuonState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  ns_coeffs: chex.Array  # shape=(), dtype=jnp.int32.
  din: chex.Array  # shape=(), dtype=jnp.int32.
  dout: chex.Array  # shape=(), dtype=jnp.int32.


def scale_by_muon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    base_shapes = None,
    scale_eps: bool = False,
    B = None,
    N = None,
    static = False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Args:
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to denominators to improve numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  if scale_eps:
    # batch scaling
    eps = eps # eps ~ ||E[G]||_F ~ 1

  def init_fn(params):
    shapes = jax.tree.map(lambda x: jnp.array(x.shape), params)
    pruned_base_shapes = prune_tree(base_shapes, shapes)
    din = jax.tree.map(lambda x, y: x[0] / y[0], shapes, pruned_base_shapes)
    dout = jax.tree.map(lambda x, y: x[1] / y[1], shapes, pruned_base_shapes)
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    return MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        ns_coeffs=ns_coeffs_,
        din=din,
        dout=dout,
    )

  def update_fn(updates, state, params=None):
    del params
    pruned_N = prune_tree(N, updates)
    mu = otu.tree_update_moment(updates, state.mu, beta, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = otu.tree_update_moment(updates, mu, beta, 1)
    else:
      mu_hat = mu
    # mu_hat = otu.tree_bias_correction(mu_hat, beta, count_inc)
    # Apply Newton-schulz orthogonalization.
    updates = jax.tree.map(
        lambda x, din, dout, n: orthogonalize_via_newton_schulz(
            x, state.ns_coeffs, ns_steps, eps, scale_eps, din, dout, n
        ),
        mu_hat,
        state.din,
        state.dout,
        pruned_N,
    )
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, MuonState(
        count=count_inc,
        mu=mu,
        ns_coeffs=state.ns_coeffs,
        din=state.din,
        dout=state.dout,
    ) if not static else state
  return base.GradientTransformation(init_fn, update_fn)


class AdamuonState(NamedTuple):
  """State for the Adamuon algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates  # First moment (momentum)
  nu: base.Updates  # Second moment (on orthogonalized gradients)
  ns_coeffs: chex.Array  # shape=(3,) or (n, 3)
  din: chex.Array
  dout: chex.Array


def scale_by_adamuon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta1: float = 0.95,
    beta2: float = 0.999,
    eps: float = 1e-8,
    adam_eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    base_shapes = None,
    scale_eps: bool = False,
    B = None,
    N = None,
    static: bool = False,
    rms_align: bool = False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Adamuon algorithm.

  Adamuon is a variant of Muon that applies Adam-style second moment scaling
  after the Newton-Schulz orthogonalization. This combines the benefits of
  Muon's spectral norm steepest descent with Adam's adaptive learning rates.

  The update rule is:
    1. Accumulate momentum: mu = beta1 * mu + (1 - beta1) * grad
    2. Orthogonalize: ortho = NewtonSchulz(mu_hat)
    3. Track second moment on orthogonalized gradient: nu = beta2 * nu + (1 - beta2) * ortho^2
    4. Scale: update = ortho / (sqrt(nu_hat) + adam_eps)

  Args:
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta1: Decay rate for the first moment (momentum).
    beta2: Decay rate for the second moment.
    eps: Term added to denominators in Newton-Schulz to improve numerical stability.
    adam_eps: Term added to denominator in Adam update for numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    base_shapes: Base shapes for scaling.
    scale_eps: Whether to scale epsilon values.
    B: Batch size parameter.
    N: Parameter for epsilon scaling.
    static: Whether to use static state (don't update state).

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  if scale_eps:
    # batch scaling for Newton-Schulz eps
    eps = eps  # eps ~ ||E[G]||_F ~ 1

  def init_fn(params):
    shapes = jax.tree.map(lambda x: jnp.array(x.shape), params)
    pruned_base_shapes = prune_tree(base_shapes, shapes)
    din = jax.tree.map(lambda x, y: x[0] / y[0], shapes, pruned_base_shapes)
    dout = jax.tree.map(lambda x, y: x[1] / y[1], shapes, pruned_base_shapes)
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params, dtype=mu_dtype)  # Second moment
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    return AdamuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        nu=nu,
        ns_coeffs=ns_coeffs_,
        din=din,
        dout=dout,
    )

  def update_fn(updates, state, params=None):
    del params
    pruned_N = prune_tree(N, updates)
    
    # Update first moment (momentum)
    mu = otu.tree_update_moment(updates, state.mu, beta1, 1)
    count_inc = numerics.safe_increment(state.count)
    
    if nesterov:
      mu_hat = otu.tree_update_moment(updates, mu, beta1, 1)
    else:
      mu_hat = mu
    
    # Apply Newton-schulz orthogonalization
    ortho_updates = jax.tree.map(
        lambda x, din, dout, n: orthogonalize_via_newton_schulz(
            x, state.ns_coeffs, ns_steps, eps, scale_eps, din, dout, n
        ),
        mu_hat,
        state.din,
        state.dout,
        pruned_N,
    )
    
    # Update second moment on orthogonalized gradients
    nu = otu.tree_update_moment(ortho_updates, state.nu, beta2, 2)
    
    # Bias correction for second moment
    nu_hat = otu.tree_bias_correction(nu, beta2, count_inc)
    
    # Compute adam_eps (potentially scaled)
    # Apply Adam-style scaling: update = ortho / (sqrt(nu_hat) + adam_eps)
    updates = jax.tree.map(
        lambda o, v, din, dout: o / (jnp.sqrt(v) + (adam_eps * ((din * dout) ** -0.5) if scale_eps else adam_eps)),
        ortho_updates,
        nu_hat,
        state.din,
        state.dout,
    )
    if rms_align:
      updates = jax.tree.map(
          lambda o: 0.2 * o / (jnp.sqrt(jnp.sum(o**2)) + 1e-10) * (o.shape[0] * o.shape[1]) ** 0.5,
          updates,
      )
    
    mu = otu.tree_cast(mu, mu_dtype)
    nu = otu.tree_cast(nu, mu_dtype)
    
    if static:
      return updates, state
    else:
      return updates, AdamuonState(
          count=count_inc,
          mu=mu,
          nu=nu,
          ns_coeffs=state.ns_coeffs,
          din=state.din,
          dout=state.dout,
      )
  
  return base.GradientTransformation(init_fn, update_fn)