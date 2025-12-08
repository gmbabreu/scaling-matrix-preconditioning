import jax
import jax.numpy as jnp
from flax import nnx
from collections.abc import Mapping
from flax import traverse_util  # tiny helper, works without Flax too
from functools import partial
from typing import Any, Dict, Tuple
import jax.tree_util as tu
from jax.tree_util import tree_map
from jax import lax
import requests
from google.cloud import storage
import subprocess
import numpy as np
import re
import optax

def parse_opt_name_overrides(cfg) -> str:
    """Parse and apply optimizer-specific overrides embedded in `cfg.opt.name`.

    Accepts strings like:
      "soap|lr=2e-2|rel_eps=false|eps=1e-8|matrix_eps=1e-10|b1=0.9|b2=0.95|wdxD=0|tpp=20|warmup_tokens=1e7"

    Supported keys:
      - lr, eps, matrix_eps/matrx_eps, rel_eps (bool), b1, b2, wdxD,
        tpp (token_per_param), warmup_tokens

    Behavior:
      - Requires key=value pairs after '|'. Bare values are not allowed.
      - Sets fields on cfg in-place. Returns the base optimizer name.
      - For warmup_tokens: values < 1 are treated as a fraction of T; otherwise as absolute int tokens.
      - For tpp: values <= 0 will clear token_per_param (set to None).
    """
    raw = str(cfg.opt.name)
    if "|" not in raw:
        return cfg.opt.name

    parts = raw.split("|")
    base = parts[0].strip()

    def _parse_bool(s: str) -> bool:
        sl = str(s).strip().lower()
        if sl == "true":
            return True
        elif sl == "false":
            return False
        else:
            raise ValueError(f"Invalid boolean value for override: {sl}")

    def _parse_number(s: str) -> float:
        x = float(str(s).strip())
        if x < 0:
            return 2 ** x
        else:
            return x

    for item in parts[1:]:
        if item is None:
            continue
        item = str(item).strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid override '{item}' in '{raw}'. Expected key=value format.")
        key, value = item.split(":", 1)
        key = key.strip().lower()
        value = value.strip()

        try:
            if key == "mup":
                cfg.opt.mup = _parse_bool(value)
            if key == "lr":
                cfg.opt.lr *= _parse_number(value)
            elif key == "eps":
                cfg.opt.eps = _parse_number(value)
            elif key == "kron_max_rms":
                cfg.opt.kron_max_rms = _parse_number(value)
            elif key == 'block_size':
                cfg.opt.block_size = int(_parse_number(value))
            elif key == "matrix_eps":
                cfg.opt.matrx_eps = _parse_number(value)
            elif key == "readout_lr_mult":
                cfg.opt.readout_lr_mult = _parse_number(value)
            elif key == "rel_eps":
                cfg.opt.rel_eps = _parse_bool(value)
            elif key == "eigh":
                cfg.opt.eigh = _parse_bool(value)
            elif key == "b1":
                cfg.opt.b1 = _parse_number(value)
            elif key == "b2":
                cfg.opt.b2 = _parse_number(value)
            elif key == "wdxd":
                cfg.opt.wdxD = _parse_number(value)
            elif key == "tpp":
                v = _parse_number(value)
                cfg.token_per_param = v if v > 0 else None
            elif key == "warmup_tokens":
                v = _parse_number(value)
                cfg.opt.warmup_tokens = v if v < 1 else int(v)
            elif key == "hidden_norm":
                cfg.opt.hidden_norm = value
            elif key == "readout_norm":
                cfg.opt.readout_norm = value
            else:
                print(f"Warning: unknown opt override '{key}' in '{raw}', ignoring.")
        except Exception as e:
            raise ValueError(f"Failed parsing override '{item}' in '{raw}': {e}")

    cfg.opt.name = base
    return base

def prune_tree(source_tree, target_tree):
    """
    Return a copy of `target_tree` in which each leaf is replaced by the
    matching leaf from `source_tree` *if it exists*; otherwise `None`.

    The result has the *same* treedef as `target_tree`, so you can safely
    feed it to `jax.tree.map` together with `target_tree`.

    The tree is pruned when more than one optimizer is used for training.
    Each optimizer only requires partial information about the model.
    """
    lookup = {}

    def _record(path, leaf):
        lookup[tuple(path)] = leaf          # tuples are hashable

    tu.tree_map_with_path(_record, source_tree)

    def _replace(path, _):
        return lookup.get(tuple(path))      # None if missing

    pruned = tu.tree_map_with_path(_replace, target_tree)

    return pruned

def flatten_dict(d, prefix=None):
  if isinstance(d, Mapping):
    out = {}
    for k, v in d.items():
      nested_prefix = k if prefix is None else f'{prefix}/{k}'
      out |= flatten_dict(v, nested_prefix)
    return out
  else:
    return {prefix: d}

def get_scheduler(schedule, decay_frac, warmup, total_steps):
    """
    Returns a piecewise function of iteration => scale factor in [0, 1].
    """
    schedules = {
        'const': lambda _: 1.0,
        'linear': lambda t: 1.0 - t,
    }
    if schedule == 'const' or decay_frac == 0:
        base_fn = lambda _: 1.0
    else:
        base_fn = lambda t: schedules[schedule](
            jnp.maximum(0.0, (t - (1 - decay_frac)) / decay_frac)
        ).clip(min=0)
    if warmup > 0:
        return lambda t: jnp.where(
            t < warmup,
            t / warmup,
            base_fn((t - warmup)/(total_steps - warmup))
        )
    else:
        return lambda t: base_fn(t / total_steps)


@jax.jit
def tree_product(xs, ys):
    products = tu.tree_map(lambda x, y: jnp.sum(x*y), xs, ys)
    return sum(tu.tree_leaves(products))

@jax.jit
def tree_add(xs, ys, alpha=1.0):
    return tree_map(lambda x, y: x + alpha * y, xs, ys)

@jax.jit
def tree_take(t, idx):
    """Return the idx-th vector from a (order, …) pytree."""
    return tree_map(lambda a: a[idx], t)

@jax.jit
def tree_set(t, idx, value):
    """Set the idx-th vector to *value* in a (order, …) pytree."""
    return tree_map(lambda a, v: a.at[idx].set(v), t, value)

@partial(jax.jit, static_argnames=("rand_fn"))
def tree_random(params, key, rand_fn):
    keys = jax.random.split(key, len(tu.tree_leaves(params)))
    keys = tu.tree_unflatten(tu.tree_structure(params), keys)
    return tree_map(lambda p, k: rand_fn(k, p.shape, dtype=p.dtype), params, keys)

@jax.jit
def tree_normalize(vs):
    norm = jnp.sqrt(tree_product(vs, vs))
    return tree_map(lambda v: v / (norm + 1e-6), vs)

@jax.jit
def tree_orthogonalize(v, basis_vectors):
    for u in basis_vectors:
        proj = tree_product(v, u)
        v = tree_add(v, u, alpha=-proj)
    return tree_normalize(v)

# evaluation utilities
def zeros_like(pytree):
    """Create a pytree of zeros with the same structure/shapes as *pytree*."""
    return tu.tree_map(jnp.zeros_like, pytree)

def welford_update(mean, M2, sample, count):
    """One‑pass Welford update on pytrees.

    Args:
      mean:   running mean (pytree).
      M2:     running sum of squares of deviations (pytree).
      sample: new sample (same structure).
      count:  current 1‑based index (int).
    Returns:
      (new_mean, new_M2)
    """
    delta  = tree_add(sample, mean, alpha=-1)                 # g − μ
    mean   = tree_add(mean, delta, alpha=1 / count)           # μ ← μ + δ/i
    delta2 = tree_add(sample, mean, alpha=-1)                 # g − μ′
    M2     = tree_add(M2, tu.tree_map(lambda x, y: x * y, delta, delta2))
    return mean, M2


def _stack_random_vectors(params, rng: jax.random.PRNGKey, k: int, normalize=False):
    """Return *k* independent Rademacher vectors stacked on the leading axis."""
    keys = jax.random.split(rng, k)
    vecs = [tree_random(params, key, jax.random.rademacher) for key in keys]
    if normalize:
        vecs = [tree_normalize(v) for v in vecs]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *vecs)

# --- Utilities for stacked PyTrees (for storing multiple eigenvectors) ---
def get_slice(stacked_pytree, index):
    """Extracts a single PyTree slice from a stacked PyTree."""
    return tu.tree_map(lambda x: x[index], stacked_pytree)

def update_slice(stacked_pytree, index, new_slice_pytree):
    """Updates a slice in a stacked PyTree with a new PyTree."""
    return tu.tree_map(
        lambda T_leaf, new_slice_leaf: T_leaf.at[index].set(new_slice_leaf),
        stacked_pytree,
        new_slice_pytree
    )

# --- Optimized Core Functions ---

@jax.jit
def tree_orth_indexed(v, basis, k):
    def body_loop(j, current_v):
        u_j = get_slice(basis, j)
        proj = tree_product(current_v, u_j)
        current_v = tree_add(current_v, u_j, alpha=-proj)
        return current_v

    v_orthogonalized = lax.fori_loop(0, k, body_loop, v)
    return v_orthogonalized

def get_vm_region():
    METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(METADATA_URL, headers=headers)
    zone_path = response.text  # e.g. "projects/123456789/zones/us-central1-b"
    zone = zone_path.split("/")[-1]  # "us-central1-b"
    region = "-".join(zone.split("-")[:-1])  # "us-central1"
    
    return region

def bucket_exists(bucket_name):
    client = storage.Client()
    try:
        client.get_bucket(bucket_name)
        return True
    except Exception:
        return False
    
def get_flop_per_token(model_cfg):
    return lm_flops_per_token(
        model_cfg.D,
        model_cfg.D * model_cfg.mlp_expansion,
        model_cfg.N,
        int(model_cfg.D / model_cfg.dh),
        int(model_cfg.D / model_cfg.dh),
        model_cfg.L,
        model_cfg.V,
        model_cfg.swiglu,
    )
    
def lm_flops_per_token(
    hidden_dim: int,
    intermediate_dim: int,
    num_layers: int,
    num_kv_heads: int,
    num_heads: int,
    seq_len: int,
    vocab_size: int,
    glu: bool,
    num_experts: int = 1,
    num_shared_experts: int = 0,
    num_experts_per_tok: int = 1,
):
    head_dim = hidden_dim / num_heads
    mlp = 2 * (3 if glu else 2) * hidden_dim * intermediate_dim * (num_experts_per_tok + num_shared_experts)
    if num_experts > 1:
        mlp += 2 * hidden_dim * num_experts  # router layer
    qkv_proj = 2 * hidden_dim * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    dense_proj = 2 * hidden_dim * hidden_dim
    # The following are across the whole sequence
    # assume full attention map like megatron-lm
    key_query_logits = 2 * seq_len**2 * num_heads * head_dim
    mask = 3 * seq_len * seq_len * num_heads
    mask_value = 2 * seq_len * seq_len * head_dim * num_heads
    seq_flops = key_query_logits + mask + mask_value
    # so we divide by the sequence length to get the per-token flops
    attn = seq_flops / seq_len
    lm_head = 2 * hidden_dim * vocab_size
    return num_layers * (mlp + qkv_proj + dense_proj + attn) + lm_head

# bfloat16 flops
flops_for_tpu = {
    "tpu v4": 275e12,
    "tpu v5 lite": 197e12,
    "tpu v6 lite": 918e12,
}

def device_hardware_flops(device):
    kind = device.device_kind.lower()
    return flops_for_tpu.get(kind, float("inf"))

def clean_gcs_path(gcs_path):
    """Recursively deletes all objects under a given gs:// path."""
    # gsutil -m rm -r -f is the most efficient and concise command
    command = f"gsutil -m rm -r -f {gcs_path}"
    
    # Execute the command
    try:
        subprocess.check_call(command.split(" "))
        print(f"Successfully deleted all objects in {gcs_path} ✨")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to clean {gcs_path}. Check permissions/path.")
        print(f"Details:\n{e.stderr.strip()}")
    except FileNotFoundError:
        print("ERROR: 'gsutil' command not found. Ensure gcloud SDK is installed.")


# =====================
# Update-size metrics
# =====================

def _reshape_to_2d(x: jax.Array) -> jax.Array:
    """Reshape a tensor to 2D as (out, in) for spectral norm.

    - For vectors/scalars, treats as (N, 1) so spec-norm = L2 norm.
    - For higher-rank tensors, flattens all but the leading axis.
    """
    if x.ndim <= 1:
        return x.reshape((-1, 1))
    out = x.shape[0]
    in_dim = int(np.prod(x.shape[1:]))
    return x.reshape((out, in_dim))


def _power_iteration_sigma(A: jax.Array, iters: int = 7) -> jax.Array:
    """Approximate top singular value using power iteration on A^T A.

    Works on 2D matrices. Returns a scalar (float32).
    """
    A = A.astype(jnp.float32)
    v = jnp.ones((A.shape[1],), dtype=jnp.float32)

    def body(i, v):
        v = A.T @ (A @ v)
        v = v / (jnp.linalg.norm(v) + 1e-12)
        return v

    v = lax.fori_loop(0, iters, body, v)
    Av = A @ v
    return jnp.linalg.norm(Av)


def _leaf_spectral_norm(x: jax.Array, iters: int = 7) -> jax.Array:
    if x.ndim == 0:
        return jnp.abs(x.astype(jnp.float32))
    if x.ndim == 1:
        return jnp.linalg.norm(x.astype(jnp.float32))
    return _power_iteration_sigma(_reshape_to_2d(x), iters)


# =====================
# Optimizer transforms
# =====================

def norm_by_spect(
    num_power_iter: float = 1.0,
    *,
    seed: int = 0,
    eps: float = 1e-10,
    warmup_steps: int = 50,
    emb_norm: str = 'rms',
    hidden_norm: str = 'spec',
    readout_norm: str = 'spec',
) -> optax.GradientTransformation:
    """Normalize each parameter update to unit spectral norm.

    - Applies after optimizer preconditioning and before LR scaling.
    - Maintains a running estimate of the top left singular vector `u` per leaf.
      At init, `u` is standard normal; each update performs power iterations
      starting from the previous `u` and using the current update matrix.

    Scheduling (power iteration):
      - Requires `num_power_iter >= 1`. Performs `ceil(num_power_iter)` iterations every step after warmup.
      - During warmup (first `warmup_steps` updates), performs 5 iterations per update.

    Args:
      num_power_iter: Positive scalar controlling frequency of power iterations.
      seed: PRNG seed for initializing vectors.
      eps: Numerical epsilon to avoid division by zero.
    Returns:
      An Optax GradientTransformation.
    """
    assert num_power_iter >= 1, "num_power_iter must be >= 1"

    # Number of PI steps per update
    ceil_iters = int(np.ceil(float(num_power_iter)))

    def init_fn(params):
        leaves = tu.tree_leaves(params)
        n_leaves = len(leaves)
        keys = jax.random.split(jax.random.PRNGKey(seed), n_leaves)
        key_tree = tu.tree_unflatten(tu.tree_structure(params), keys)

        # Build a names tree for printing decisions (host-side only).
        def _name_from_path(path, leaf):
            return clean_param_path(path)
        names = tu.tree_map_with_path(_name_from_path, params)

        def _init_leaf(p, k, name):
            if p.ndim != 2:
                raise AssertionError(f"norm_by_spect expects 2D params, got {p.ndim}D for {name} with shape {p.shape}")
            din, dout = p.shape
            # Always track right singular vector v in R^{dout}
            v0 = jax.random.normal(k, (dout,), dtype=p.dtype)
            v0 = v0 / (jnp.linalg.norm(v0) + jnp.asarray(eps, dtype=p.dtype))
            # Robust skip for any parameter whose cleaned path contains 'embed'
            is_embed = ('embed' in str(name).lower())
            if is_embed and emb_norm == 'rms':
                # Print once at init, informing RMS normalization will be used for this leaf
                print(f"norm_by_spect: using RMS normalization for param '{name}' with shape {tuple(p.shape)} (din={din}, dout={dout})")
            is_readout = ('readout' in str(name).lower())
            if is_readout and readout_norm == 'fro':
                # Print once at init, informing RMS normalization will be used for this leaf
                print(f"norm_by_spect: using Frobenius normalization for param '{name}' with shape {tuple(p.shape)} (din={din}, dout={dout})")
            return v0, jnp.array(is_embed, dtype=jnp.bool_), jnp.array(is_readout, dtype=jnp.bool_)

        packed = tu.tree_map(_init_leaf, params, key_tree, names)
        is_pair = lambda x: isinstance(x, tuple)
        v_tree = tu.tree_map(lambda t: t[0], packed, is_leaf=is_pair)
        is_embed_tree = tu.tree_map(lambda t: t[1], packed, is_leaf=is_pair)
        is_readout_tree = tu.tree_map(lambda t: t[2], packed, is_leaf=is_pair)
        count = jnp.array(0, dtype=jnp.int32)
        return {"v": v_tree, "is_embed": is_embed_tree, "is_readout": is_readout_tree, "count": count}

    def update_fn(updates, state, params=None):
        del params  # unused
        count = state["count"]
        warm = count < jnp.int32(warmup_steps)

        def _update_leaf(g, v, is_embed, is_readout):
            # Cast to float32 for stability during PI; keep original dtype for output
            g_dtype = g.dtype
            A = g
            din, dout = A.shape
            v_curr = v
            eps_val = jnp.asarray(eps, dtype=g_dtype)

            def one_step(carry):
                v, _ = carry
                y = A @ v                 # R^{din}
                sigma = jnp.linalg.norm(y)
                z = A.T @ y               # R^{dout}
                v_new = z / (jnp.linalg.norm(z) + eps_val)
                # update v to v_new only if its norm is close to 1
                v = lax.cond(jnp.linalg.norm(v_new) > 0.5, lambda: v_new, lambda: v)
                return (v, sigma)

            # Run scheduled iterations, tracking last sigma
            init_carry = (v_curr, jnp.zeros((), dtype=g_dtype))
            iters_this = jnp.where(warm, jnp.int32(5), jnp.int32(ceil_iters))
            v_new, sigma_last = lax.fori_loop(0, iters_this, lambda i, c: one_step(c), init_carry)

            # Sigma from the last iteration
            sigma = sigma_last

            # If embedding, use RMS normalization instead
            def scale_hidden(_):
                if hidden_norm == 'rms':
                    rms = jnp.sqrt(jnp.mean((A)**2))
                    return g * (1 / din ** 0.5) / (rms + eps_val)
                elif hidden_norm == 'spec':
                    # Multiply by sqrt(dout/din) after spectral normalization
                    dim_factor = jnp.sqrt(jnp.asarray(dout, dtype=g_dtype) / jnp.asarray(din, dtype=g_dtype))
                    return g * (dim_factor / (sigma + eps_val))
                else:
                    raise ValueError(f"Invalid hidden normalization: {hidden_norm}")
            def scale_emb(_):
                rms = jnp.sqrt(jnp.mean((A)**2))
                return g / (rms + eps_val)
            def scale_readout(_):
                if readout_norm == 'rms':
                    rms = jnp.sqrt(jnp.mean((A)**2))
                    return g * (1 / din ** 0.5) / (rms + eps_val)
                elif readout_norm == 'spec':
                    # Multiply by sqrt(dout/din) after spectral normalization
                    dim_factor = jnp.sqrt(jnp.asarray(dout, dtype=g_dtype) / jnp.asarray(din, dtype=g_dtype))
                    return g * (dim_factor / (sigma + eps_val))
                else:
                    raise ValueError(f"Invalid readout normalization: {readout_norm}")
            g_scaled = lax.cond(is_embed, scale_emb, lambda _: lax.cond(is_readout, scale_readout, scale_hidden, operand=None), operand=None)

            return (g_scaled, v_new)

        packed = tu.tree_map(_update_leaf, updates, state["v"], state["is_embed"], state["is_readout"])
        is_pair = lambda x: isinstance(x, tuple)
        g_new = tu.tree_map(lambda t: t[0], packed, is_leaf=is_pair)
        v_new_tree = tu.tree_map(lambda t: t[1], packed, is_leaf=is_pair)
        new_state = {"v": v_new_tree, "is_embed": state["is_embed"], "is_readout": state["is_readout"], "count": count + jnp.int32(1)}
        return g_new, new_state

    return optax.GradientTransformation(init_fn, update_fn)


@partial(jax.jit, static_argnames=("iters",))
def update_metrics_tree(updates, iters: int = 7, inputs=None):
    """Compute per-parameter metrics on a weight-update pytree.

    Returns four pytrees (same structure):
      - rms: sqrt(mean(update**2))
      - spec: spectral norm (approx via power iteration)
      - stable_rank: ||A||_F^2 / ||A||_2^2
      - alignment: log_{din} RMS(X @ U) / RMS(U) / RMS(X), where X are per-leaf input features
    """
    rms_tree = jax.tree_util.tree_map(
        lambda u: jnp.sqrt(jnp.mean((u.astype(jnp.float32))**2)), updates)
    spec_tree = jax.tree_util.tree_map(
        lambda u: _leaf_spectral_norm(u.astype(jnp.float32), iters), updates)
    sr_tree = jax.tree_util.tree_map(
        lambda u, s: jnp.sum((u.astype(jnp.float32))**2) / (s * s + 1e-12),
        updates, spec_tree)

    def _align(u, x):
        # If no features provided or shapes incompatible, return NaN
        if x is None or getattr(u, 'ndim', 0) != 2:
            return jnp.asarray(jnp.nan)
        # Assume x is [batch, din] and u is [din, dout]
        # y = x @ u gives [batch, dout]
        din = u.shape[0]
        x = x.reshape((-1, din))
        y = x @ u
        rms_u = jnp.sqrt(jnp.mean(u * u))
        rms_x = jnp.sqrt(jnp.mean(x * x, axis=1))  # [batch]
        rms_y = jnp.sqrt(jnp.mean(y * y, axis=1))  # [batch]
        ratio = rms_y / (rms_u * rms_x + 1e-12)  # [batch]
        # log base din; guard small bases
        log_align = jnp.log(ratio + 1e-12) / jnp.log(din)  # [batch]
        align = jnp.mean(log_align)
        return align

    if inputs is None:
        align_tree = jax.tree_util.tree_map(lambda _: jnp.asarray(jnp.nan, dtype=jnp.float32), updates)
    else:
        align_tree = jax.tree_util.tree_map(_align, updates, inputs)

    return rms_tree, spec_tree, sr_tree, align_tree


# =====================
# Path sanitization
# =====================

def clean_param_path(path_tuple) -> str:
    """Convert an nnx/JAX tree path to a clean dotted name.

    - Strips brackets and quotes from stringified keys (e.g., ["readout"] -> readout)
    - Drops trailing/internal technical nodes like `kernel` and `value`
    - Collapses repeated dots
    """
    parts = []
    for part in path_tuple:
        s = str(part)
        # Remove brackets and quotes commonly present in nnx path printing
        s = s.replace("[", "").replace("]", "").replace("'", "")
        s = s.strip()
        # Remove leading/trailing dots that may sneak in
        s = s.strip('.')
        # Skip noisy internals
        if s in ("kernel", "value", ""):  # drop unwanted segments
            continue
        parts.append(s)
    name = ".".join(parts)
    name = re.sub(r"\.+", ".", name)  # collapse any accidental repeats
    return name
