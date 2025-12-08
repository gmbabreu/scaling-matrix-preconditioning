from typing import Any, NamedTuple
import functools

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from optax import tree_utils as otu
from optax._src import base, numerics

from utils import prune_tree
from optax_optim._google_shampoo import matrix_inverse_pth_root


class ScaleByShampooState(NamedTuple):
    count: jnp.ndarray
    L: Any
    R: Any
    L_inv: Any
    R_inv: Any
    mu: Any
    nu: Any
    din: Any
    dout: Any
    nb_in: Any
    nb_out: Any


@functools.partial(jax.jit, static_argnames=("p", "rel_eps"))
def _inv_root(mat: jnp.ndarray, p: float, eps: jnp.ndarray, rel_eps: bool) -> jnp.ndarray:
    return matrix_inverse_pth_root(mat, 1/p, ridge_epsilon=eps, rel_eps=rel_eps)[0]


@functools.partial(jax.jit, static_argnames=("p",))
def _inv_root_eigh(mat: jnp.ndarray, p: float, eps: jnp.ndarray) -> jnp.ndarray:
    eigval, eigvec = jnp.linalg.eigh(mat + eps * jnp.eye(mat.shape[0], dtype=mat.dtype))
    eigval = jnp.maximum(eigval, eps)
    return eigvec @ jnp.diag(eigval ** (-p)) @ eigvec.T


def scale_by_shampoo(
    *,
    b1: float = 0.9,
    b2: float = 0.999,
    freq: int = 1,
    kl: float = 0.25,
    kr: float = 0.25,
    adam_eps: float = 1e-8,
    matrix_eps: float = 1e-6,
    grafting: bool = False,
    scale_eps: bool = False,
    base_shapes=None,
    B=None,
    N=None,
    static: bool = False,
    eigh: bool = False,
    rel_eps: bool = False,
    block_size: int = 0,
    nb_in=None,
    nb_out=None,
    max_precond_dim: int = 0,
) -> base.GradientTransformation:
    """Bucketed Shampoo preconditioner (drop-in replacement).

    Matches math of the previous implementation but buckets inverse pth‑root
    recomputations across leaves with the same block shape to reduce compile size.
    """

    if rel_eps:
        assert not eigh, "EIGH is not supported with relative eps"

    if scale_eps:
        if not rel_eps:
            matrix_eps = matrix_eps / B
        adam_eps = adam_eps / (B ** 0.5)
        graft_eps = adam_eps * (B ** (kl + kr - 0.5))
    else:
        graft_eps = adam_eps

    def _compute_block_slices(m: int, n: int, bsz: int):
        if bsz <= 0 or (m <= bsz and n <= bsz):
            return [slice(0, m)], [slice(0, n)]
        def _mk(dim):
            if bsz <= 0 or dim <= bsz:
                return [slice(0, dim)]
            starts = list(range(0, dim, bsz))
            return [slice(s, min(s + bsz, dim)) for s in starts]
        return _mk(m), _mk(n)

    def _compute_block_slices_with_gate(m: int, n: int, bsz: int, left_id: bool, right_id: bool):
        rs, cs = _compute_block_slices(m, n, bsz)
        if left_id:
            rs = [slice(0, m)]
        if right_id:
            cs = [slice(0, n)]
        return rs, cs

    def _stack_blocks(x: jnp.ndarray, row_slices, col_slices, row_bsz: int, col_bsz: int) -> jnp.ndarray:
        blocks = []
        for rs in row_slices:
            for cs in col_slices:
                blk = x[rs, cs]
                br, bc = blk.shape
                pr = (0, row_bsz - br)
                pc = (0, col_bsz - bc)
                blk = jnp.pad(blk, (pr, pc))
                blocks.append(blk)
        return jnp.stack(blocks, axis=0)

    def _merge_stacked_blocks(stacked: jnp.ndarray, m: int, n: int, row_slices, col_slices, row_bsz: int, col_bsz: int) -> jnp.ndarray:
        out = jnp.zeros((m, n), dtype=stacked.dtype)
        idx = 0
        for rs in row_slices:
            for cs in col_slices:
                br, bc = rs.stop - rs.start, cs.stop - cs.start
                out = out.at[rs, cs].set(stacked[idx, :br, :bc])
                idx += 1
        return out

    def init_fn(params):
        def _has_array(x):
            return hasattr(x, "shape") and hasattr(x, "dtype") and (x.ndim == 2)

        shapes = jax.tree.map(lambda x: jnp.array(x.shape) if _has_array(x) else jnp.array((1, 1)), params)
        pruned_base_shapes = prune_tree(base_shapes, shapes) if base_shapes is not None else shapes
        din = jax.tree.map(lambda x, y: x[0] / y[0], shapes, pruned_base_shapes)
        dout = jax.tree.map(lambda x, y: x[1] / y[1], shapes, pruned_base_shapes)
        pruned_nb_in  = prune_tree(nb_in, params)  if nb_in  is not None else jax.tree.map(lambda g: jnp.asarray(1.0, dtype=g.dtype) if _has_array(g) else g, params)
        pruned_nb_out = prune_tree(nb_out, params) if nb_out is not None else jax.tree.map(lambda g: jnp.asarray(1.0, dtype=g.dtype) if _has_array(g) else g, params)
        pruned_N = prune_tree(N, params) if N is not None else jax.tree.map(lambda g: jnp.asarray(1.0, dtype=g.dtype) if _has_array(g) else g, params)

        def _init_leaf(w, din_i, dout_i, n_i, nb_in_i, nb_out_i):
            if not _has_array(w):
                return w
            m, n = w.shape
            left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
            right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
            rs, cs = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            Bn = len(rs) * len(cs)
            row_bs = (block_size if (block_size > 0 and len(rs) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cs) > 1) else n)
            if scale_eps:
                eps = matrix_eps * (din_i / dout_i) / (n_i ** 2) / (nb_in_i * nb_out_i)
            else:
                eps = matrix_eps
            L = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=w.dtype) * eps, (Bn, row_bs, row_bs))
            R = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=w.dtype) * eps, (Bn, col_bs, col_bs))
            Linv = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=w.dtype) / eps, (Bn, row_bs, row_bs))
            Rinv = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=w.dtype) / eps, (Bn, col_bs, col_bs))
            return L, R, Linv, Rinv

        packed = jax.tree.map(_init_leaf, params, din, dout, pruned_N, pruned_nb_in, pruned_nb_out)
        is_tuple = lambda x: isinstance(x, tuple)
        def _pick_or_pass(t, i):
            return t[i] if isinstance(t, tuple) and len(t) > i else t
        L     = jtu.tree_map(lambda t: _pick_or_pass(t, 0), packed, is_leaf=is_tuple)
        R     = jtu.tree_map(lambda t: _pick_or_pass(t, 1), packed, is_leaf=is_tuple)
        L_inv = jtu.tree_map(lambda t: _pick_or_pass(t, 2), packed, is_leaf=is_tuple)
        R_inv = jtu.tree_map(lambda t: _pick_or_pass(t, 3), packed, is_leaf=is_tuple)
        zeros_or_pass = lambda x: jnp.zeros_like(x) if hasattr(x, "dtype") else x
        mu_zeros = jax.tree.map(zeros_or_pass, params)
        nu_zeros = jax.tree.map(zeros_or_pass, params) if grafting else None
        return ScaleByShampooState(
            count=jnp.zeros([], jnp.int32),
            L=L,
            R=R,
            L_inv=L_inv,
            R_inv=R_inv,
            mu=mu_zeros,
            nu=nu_zeros,
            din=din,
            dout=dout,
            nb_in=pruned_nb_in,
            nb_out=pruned_nb_out,
        )

    def update_fn(updates, state: ScaleByShampooState, params=None):
        del params

        def _update_stats(g, L_i, R_i):
            if not (hasattr(g, "shape") and hasattr(g, "dtype")):
                return (L_i, R_i, None)
            m, n = g.shape
            left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
            right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
            rs, cs = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            row_bs = (block_size if (block_size > 0 and len(rs) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cs) > 1) else n)
            need_L = L_i is not None
            need_R = R_i is not None
            if not (need_L or need_R):
                meta = (m, n, row_bs, col_bs, 0, left_id, right_id, rs, cs)
                return (None, None, meta)
            g_stk = _stack_blocks(g, rs, cs, row_bs, col_bs)
            g_t = jnp.swapaxes(g_stk, -1, -2)
            L_new = (b2 * L_i + (1.0 - b2) * (g_stk @ g_t)) if need_L else None
            R_new = (b2 * R_i + (1.0 - b2) * (g_t @ g_stk)) if need_R else None
            meta = (m, n, row_bs, col_bs, len(rs) * len(cs), left_id, right_id, rs, cs)
            return (L_new, R_new, meta)

        LR_meta = jax.tree.map(_update_stats, updates, state.L, state.R)
        # MaskedNode is also a tuple (zero fields); guard by length to skip those sentinels.
        is_triple = lambda x: isinstance(x, tuple) and len(x) == 3
        L_new = jtu.tree_map(lambda t: t[0], LR_meta, is_leaf=is_triple)
        R_new = jtu.tree_map(lambda t: t[1], LR_meta, is_leaf=is_triple)
        meta  = jtu.tree_map(lambda t: t[2], LR_meta, is_leaf=is_triple)

        should_recompute = (state.count % freq) == 0
        count_inc = numerics.safe_increment(state.count)
        def _bias_correct(moment):
            bc = 1 - (b2 ** count_inc)
            def _f(t):
                return t / bc.astype(t.dtype) if hasattr(t, "dtype") else t
            return jax.tree.map(_f, moment)
        L_hat = _bias_correct(L_new)
        R_hat = _bias_correct(R_new)

        def_tree_N = prune_tree(N, updates) if N is not None else jax.tree.map(
            lambda g: jnp.asarray(1.0, dtype=g.dtype) if hasattr(g, "dtype") else g,
            updates,
        )

        def _recompute_all(_):
            struct = jtu.tree_structure(updates)
            L_hat_leaves = struct.flatten_up_to(L_hat)
            R_hat_leaves = struct.flatten_up_to(R_hat)
            din_leaves   = struct.flatten_up_to(state.din)
            dout_leaves  = struct.flatten_up_to(state.dout)
            N_leaves     = struct.flatten_up_to(def_tree_N)
            nb_in_leaves = struct.flatten_up_to(state.nb_in)
            nb_out_leaves= struct.flatten_up_to(state.nb_out)

            n_leaves = len(L_hat_leaves)

            L_buckets = {}
            L_slices = [None] * n_leaves
            for i in range(n_leaves):
                P = L_hat_leaves[i]
                if P is None:
                    continue
                Bn, d, _ = P.shape
                key = (int(d), P.dtype)
                if key not in L_buckets:
                    L_buckets[key] = {"mats": [], "eps": []}
                if scale_eps and not rel_eps:
                    eps_leaf = matrix_eps * (din_leaves[i] / dout_leaves[i]) / (N_leaves[i] ** 2) / (nb_in_leaves[i] * nb_out_leaves[i])
                else:
                    eps_leaf = matrix_eps
                eps_vec = jnp.repeat(jnp.asarray(eps_leaf, dtype=P.dtype)[None], Bn, axis=0)
                L_buckets[key]["mats"].append(P)
                L_buckets[key]["eps"].append(eps_vec)
                start = sum(x.shape[0] for x in L_buckets[key]["mats"]) - Bn
                L_slices[i] = (key[0], start, Bn)

            R_buckets = {}
            R_slices = [None] * n_leaves
            for i in range(n_leaves):
                P = R_hat_leaves[i]
                if P is None:
                    continue
                Bn, d, _ = P.shape
                key = (int(d), P.dtype)
                if key not in R_buckets:
                    R_buckets[key] = {"mats": [], "eps": []}
                if scale_eps and not rel_eps:
                    eps_leaf = matrix_eps * (din_leaves[i] / dout_leaves[i]) / (N_leaves[i] ** 2) / (nb_in_leaves[i] * nb_out_leaves[i])
                else:
                    eps_leaf = matrix_eps
                eps_vec = jnp.repeat(jnp.asarray(eps_leaf, dtype=P.dtype)[None], Bn, axis=0)
                R_buckets[key]["mats"].append(P)
                R_buckets[key]["eps"].append(eps_vec)
                start = sum(x.shape[0] for x in R_buckets[key]["mats"]) - Bn
                R_slices[i] = (key[0], start, Bn)

            Linv_leaves = [None] * n_leaves
            Rinv_leaves = [None] * n_leaves

            for key, bucket in L_buckets.items():
                mats = jnp.concatenate(bucket["mats"], axis=0) if bucket["mats"] else None
                epss = jnp.concatenate(bucket["eps"], axis=0) if bucket["eps"] else None
                if mats is None:
                    continue
                kernel = (lambda M, e: _inv_root_eigh(M, kl, e)) if eigh else (lambda M, e: _inv_root(M, kl, e, rel_eps))
                invs = jax.vmap(kernel)(mats, epss)
                for i in range(n_leaves):
                    s = L_slices[i]
                    if s is None:
                        continue
                    dim, start, cnt = s
                    if int(dim) != mats.shape[-1]:
                        continue
                    Linv_leaves[i] = invs[start:start+cnt]

            for key, bucket in R_buckets.items():
                mats = jnp.concatenate(bucket["mats"], axis=0) if bucket["mats"] else None
                epss = jnp.concatenate(bucket["eps"], axis=0) if bucket["eps"] else None
                if mats is None:
                    continue
                kernel = (lambda M, e: _inv_root_eigh(M, kr, e)) if eigh else (lambda M, e: _inv_root(M, kr, e, rel_eps))
                invs = jax.vmap(kernel)(mats, epss)
                for i in range(n_leaves):
                    s = R_slices[i]
                    if s is None:
                        continue
                    dim, start, cnt = s
                    if int(dim) != mats.shape[-1]:
                        continue
                    Rinv_leaves[i] = invs[start:start+cnt]

            Linv_tree = jtu.tree_unflatten(struct, Linv_leaves)
            Rinv_tree = jtu.tree_unflatten(struct, Rinv_leaves)
            return Linv_tree, Rinv_tree

        def _keep(_):
            return state.L_inv, state.R_inv

        L_inv, R_inv = jax.lax.cond(should_recompute, _recompute_all, _keep, operand=None)

        def _update_moment(upd, mom):
            if hasattr(upd, "dtype"):
                return (1 - b1) * upd + b1 * mom
            return mom
        mu = jax.tree.map(_update_moment, updates, state.mu)

        def _bias_correct_mu(moment):
            bc = 1 - (b1 ** count_inc)
            def _f(t):
                return t / bc.astype(t.dtype) if hasattr(t, "dtype") else t
            return jax.tree.map(_f, moment)
        mu_hat = _bias_correct_mu(mu)

        def _apply_precond(mu_i, Linv_i, Rinv_i, w):
            if not (hasattr(mu_i, "shape") and hasattr(mu_i, "dtype")):
                return mu_i
            m, n = mu_i.shape
            left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
            right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
            rs, cs = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            row_bs = (block_size if (block_size > 0 and len(rs) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cs) > 1) else n)
            mu_stk = _stack_blocks(mu_i, rs, cs, row_bs, col_bs)
            if Linv_i is None and Rinv_i is None:
                upd_stk = mu_stk
            elif Linv_i is None:
                upd_stk = mu_stk @ Rinv_i
            elif Rinv_i is None:
                upd_stk = Linv_i @ mu_stk
            else:
                upd_stk = Linv_i @ mu_stk @ Rinv_i
            return _merge_stacked_blocks(upd_stk, m, n, rs, cs, row_bs, col_bs)

        shampoo_updates = jax.tree.map(_apply_precond, mu_hat, L_inv, R_inv, updates)

        if grafting:
            nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
            nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
            if scale_eps:
                # Prune scalars to updates structure to align with masked leaves
                din_pruned  = prune_tree(state.din,  updates)
                dout_pruned = prune_tree(state.dout, updates)
                nbin_pruned = prune_tree(state.nb_in,  updates)
                nbout_pruned= prune_tree(state.nb_out, updates)
                def_tree_N = prune_tree(N, updates) if N is not None else jax.tree.map(
                    lambda g: jnp.asarray(1.0, dtype=g.dtype) if hasattr(g, "dtype") else g,
                    updates,
                )
                is_none = lambda x: x is None
                # Adam-style denom eps scaling: eps/dout/N
                adam_updates = jax.tree.map(
                    lambda m, v, din, dout, n: (
                        m if not (hasattr(m, "dtype") and hasattr(v, "dtype"))
                        else m / (jnp.sqrt(v) + adam_eps / dout / n)
                    ),
                    mu_hat, nu_hat, din_pruned, dout_pruned, def_tree_N,
                    is_leaf=is_none,
                )
                # Depth/shape-aware Shampoo LR for graft correction
                kl_active = jax.tree.map(
                    lambda x: 0 if (hasattr(x, "shape") and x.shape[0] > max_precond_dim) else kl,
                    updates,
                )
                kr_active = jax.tree.map(
                    lambda x: 0 if (hasattr(x, "shape") and x.shape[1] > max_precond_dim) else kr,
                    updates,
                )
                E_active = jax.tree.map(lambda x, y: x + y, kl_active, kr_active)
                shampoo_lr = jax.tree.map(
                    lambda n, E, din, dout, nbi, nbo: n ** (1 - 2*E) * (dout / din) ** (1 - E) * (nbi * nbo) ** -E,
                    def_tree_N, E_active, din_pruned, dout_pruned, nbin_pruned, nbout_pruned,
                    is_leaf=is_none,
                )
                final_updates = jax.tree.map(
                    lambda a, s, slr, din, dout: (
                        a if not (hasattr(a, "dtype") and hasattr(s, "dtype"))
                        else jnp.linalg.norm(a) / (jnp.linalg.norm(s) + graft_eps * ((dout / din) ** 0.5) / slr) * s
                    ),
                    adam_updates, shampoo_updates, shampoo_lr, din_pruned, dout_pruned,
                    is_leaf=is_none,
                )
            else:
                adam_updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + adam_eps), mu_hat, nu_hat)
                final_updates = jax.tree.map(
                    lambda a, s: jnp.linalg.norm(a) / (jnp.linalg.norm(s) + graft_eps) * s,
                    adam_updates, shampoo_updates
                )
        else:
            nu = state.nu
            final_updates = shampoo_updates

        new_state = ScaleByShampooState(
            count=count_inc,
            L=L_new,
            R=R_new,
            L_inv=L_inv,
            R_inv=R_inv,
            mu=mu,
            nu=nu,
            din=state.din,
            dout=state.dout,
            nb_in=state.nb_in,
            nb_out=state.nb_out,
        )
        return final_updates, (state if static else new_state)

    return base.GradientTransformation(init_fn, update_fn)
