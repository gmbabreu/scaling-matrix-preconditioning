import os
import threading
from typing import List, Sequence, Optional

import numpy as np

# Simple global recorder for SOAP debug tensors collected via jax.debug.callback.
# Usage:
#   - init(names, out_path)
#   - from jitted code: jax.debug.callback(record, g_list, m_list, u_list, step)
#   - after training: finalize()

_lock = threading.Lock()
_active: bool = False
_names: Optional[List[str]] = None
_out_path: Optional[str] = None

# Per-name lists of arrays for each step
_g: Optional[List[List[np.ndarray]]] = None
_m: Optional[List[List[np.ndarray]]] = None
_u: Optional[List[List[np.ndarray]]] = None
_v: Optional[List[List[np.ndarray]]] = None  # second moment (nu)
_ql: Optional[List[List[np.ndarray]]] = None  # left eigenvectors (basis)
_qr: Optional[List[List[np.ndarray]]] = None  # right eigenvectors (basis)


def is_active() -> bool:
    return _active


def init(names: Sequence[str], out_path: str) -> None:
    """Initialize recorder with ordered layer names and output path.

    Args:
      names: list/sequence of parameter names flattened in the same order as the
             grads/updates pytrees used by the optimizer.
      out_path: path to write an .npz archive at finalize().
    """
    global _active, _names, _out_path, _g, _m, _u, _v, _ql, _qr
    with _lock:
        _names = list(names)
        _out_path = out_path
        _g = [[] for _ in _names]
        _m = [[] for _ in _names]
        _u = [[] for _ in _names]
        _v = [[] for _ in _names]
        _ql = [[] for _ in _names]
        _qr = [[] for _ in _names]
        _active = True


def _to_np(x):
    # Convert JAX arrays to numpy. Ensure bfloat16 becomes float32 for portability.
    try:
        dtype_str = str(getattr(x, 'dtype', ''))
        if 'bfloat16' in dtype_str:
            x = x.astype('float32')
        return np.asarray(x)
    except Exception:
        return np.array(x)


def record(g_list, m_list, u_list, *rest):
    """Host callback target. Appends one step worth of tensors.

    Supports either (g, m, u, step) or (g, m, u, v, step) signatures.
    The lists must be aligned with _names: len(g_list) == len(_names).
    """
    if not _active:
        return
    with _lock:
        assert _names is not None and _g is not None and _m is not None and _u is not None and _v is not None and _ql is not None and _qr is not None

        # Parse optional lists and step. We accept any of:
        #  - (step)
        #  - (v, step)
        #  - (v, ql, qr, step)
        step = rest[-1] if len(rest) >= 1 else None
        v_list = rest[0] if len(rest) >= 2 else None
        ql_list = rest[1] if len(rest) >= 4 else None
        qr_list = rest[2] if len(rest) >= 4 else None

        assert len(g_list) == len(_names) == len(m_list) == len(u_list)
        if v_list is not None:
            assert len(v_list) == len(_names)
        if ql_list is not None:
            assert len(ql_list) == len(_names)
        if qr_list is not None:
            assert len(qr_list) == len(_names)

        for i in range(len(_names)):
            _g[i].append(_to_np(g_list[i]))
            _m[i].append(_to_np(m_list[i]))
            _u[i].append(_to_np(u_list[i]))
            if v_list is not None:
                _v[i].append(_to_np(v_list[i]))
            if ql_list is not None:
                _ql[i].append(_to_np(ql_list[i]))
            if qr_list is not None:
                _qr[i].append(_to_np(qr_list[i]))


def finalize() -> Optional[str]:
    """Writes an .npz file where each key is:
      - names: an array of layer names
      - g/<name>, m/<name>, u/<name>: arrays of shape (T, d_out, d_in)

    Returns the output path or None if inactive.
    """
    global _active, _names, _out_path, _g, _m, _u, _v, _ql, _qr
    with _lock:
        if not _active:
            return None

        assert _names is not None and _out_path is not None
        # Prepare payload
        save_dict = {}
        names_arr = np.array(_names, dtype=object)
        save_dict['names'] = names_arr
        def _safe(k: str) -> str:
            return k.replace('/', '__')
        for idx, name in enumerate(_names):
            g_steps = _g[idx]
            m_steps = _m[idx]
            u_steps = _u[idx]
            v_steps = _v[idx] if _v is not None else []
            ql_steps = _ql[idx] if _ql is not None else []
            qr_steps = _qr[idx] if _qr is not None else []
            # Stack along time axis
            g_arr = np.stack(g_steps, axis=0) if len(g_steps) > 0 else None
            m_arr = np.stack(m_steps, axis=0) if len(m_steps) > 0 else None
            u_arr = np.stack(u_steps, axis=0) if len(u_steps) > 0 else None
            v_arr = np.stack(v_steps, axis=0) if len(v_steps) > 0 else None
            ql_arr = np.stack(ql_steps, axis=0) if len(ql_steps) > 0 else None
            qr_arr = np.stack(qr_steps, axis=0) if len(qr_steps) > 0 else None
            key_base = _safe(name)
            if g_arr is not None:
                save_dict[f'g/{key_base}'] = g_arr
            if m_arr is not None:
                save_dict[f'm/{key_base}'] = m_arr
            if u_arr is not None:
                save_dict[f'u/{key_base}'] = u_arr
            if v_arr is not None:
                save_dict[f'v/{key_base}'] = v_arr
            if ql_arr is not None:
                save_dict[f'ql/{key_base}'] = ql_arr
            if qr_arr is not None:
                save_dict[f'qr/{key_base}'] = qr_arr

        # Ensure directory exists
        os.makedirs(os.path.dirname(_out_path) or '.', exist_ok=True)
        np.savez_compressed(_out_path, **save_dict)

        # Reset
        out = _out_path
        _active = False
        _names = None
        _out_path = None
        _g = _m = _u = _v = _ql = _qr = None
        return out
