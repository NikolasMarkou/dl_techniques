"""Inference for the Neural HMM: Viterbi decoding and forward-backward posteriors.

All functions use `keras.ops` (backend-agnostic) rather than `scipy` or raw NumPy,
so they work inside `tf.function` / `jit_compile` graphs and are callable from
within the model's forward pass. The recipe's numpy-only `viterbi`/`posteriors`
are replaced by Keras-op equivalents; the recipe's `decode` convenience that
materialises to NumPy after `model.predict` still works because the ops convert
to numpy automatically.

Two functions are exported:

- `viterbi`: returns the most-likely state path and its log-probability.
- `posteriors`: returns the marginal state-occupancy probabilities ``γ_t(k)``
  for every timestep.

Both are pure functions of the model's own forward outputs ``(log_pi, log_A,
log_emit)``; they take no model state and have no trainable parameters.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import keras
from keras import ops


def shift_right(x):
    """Shift a sequence right by one timestep, prepending zeros.

    For generative mode, the encoder should only see x_{<t}, so we shift
    the input right by one timestep.

    :param x: Input tensor of shape ``(B, T, D)`` or ``(B, T)``.
    :type x: keras.KerasTensor
    :return: Shifted tensor of same shape, with zeros at t=0.
    :rtype: keras.KerasTensor
    """
    # Prepend zeros along the time dimension
    zero_shape = list(x.shape)
    zero_shape[1] = 1  # time dimension
    zeros = ops.zeros(zero_shape, dtype=x.dtype)
    return ops.concatenate([zeros, x[:, :-1]], axis=1)


def _forward_logprob_static(log_pi, log_A, log_emit, mask):
    """Static-transition forward algorithm in log-space, unrolled.

    :param log_pi: ``(K,)`` initial log-probs.
    :type log_pi: keras.KerasTensor
    :param log_A: ``(K, K)`` transition log-probs.
    :type log_A: keras.KerasTensor
    :param log_emit: ``(B, T, K)`` emission log-densities.
    :type log_emit: keras.KerasTensor
    :param mask: ``(B, T)`` float padding mask (1 = valid).
    :type mask: keras.KerasTensor
    :return: Log-likelihood per sample ``(B,)``.
    :rtype: keras.KerasTensor
    """
    from scipy.special import logsumexp  # imported at function level to avoid a
        # top-level dependency; the recipe uses scipy in inference only.

    B = keras.ops.shape(log_emit)[0]
    T = keras.ops.shape(log_emit)[1]
    K = keras.ops.shape(log_emit)[2]

    # alpha_0
    alpha = log_pi[None] + log_emit[:, 0]  # (B, K)

    # Unrolled forward pass over t = 1 .. T-1
    for t in range(1, T):
        # logsumexp(alpha[:, None] + log_A, axis=1) -> (B, K)
        new_alpha = keras.ops.logsumexp(alpha[:, None] + log_A[None], axis=1) + log_emit[:, t]

        # Freeze on padding: where mask[b, t] == 0 keep the old alpha
        alpha = keras.ops.where(mask[:, t][:, None] > 0, new_alpha, alpha)

    # Log-likelihood per sample: logsumexp over the final alpha
    return keras.ops.logsumexp(alpha, axis=-1)


def _forward_logprob_dynamic(log_pi, log_A, log_emit, mask):
    """Dynamic-transition forward algorithm (B,T,K,K), unrolled.

    :param log_pi: ``(K,)`` initial log-probs.
    :type log_pi: keras.KerasTensor
    :param log_A: ``(T, K, K)`` transition log-probs per timestep.
    :type log_A: keras.KerasTensor
    :param log_emit: ``(B, T, K)`` emission log-densities.
    :type log_emit: keras.KerasTensor
    :param mask: ``(B, T)`` float padding mask (1 = valid).
    :type mask: keras.KerasTensor
    :return: Log-likelihood per sample ``(B,)``.
    :rtype: keras.KerasTensor
    """
    from scipy.special import logsumexp  # imported at function level to avoid a
        # top-level dependency; the recipe uses scipy in inference only.

    B = keras.ops.shape(log_emit)[0]
    T = keras.ops.shape(log_emit)[1]
    K = keras.ops.shape(log_emit)[2]

    # alpha_0
    alpha = log_pi[None] + log_emit[:, 0]  # (B, K)

    # Unrolled forward pass over t = 1 .. T-1
    for t in range(1, T):
        # log_A[t] is (K, K) at timestep t
        log_A_t = log_A[t]  # (K, K)
        # logsumexp(alpha[:, None] + log_A[t], axis=1) -> (B, K)
        new_alpha = keras.ops.logsumexp(alpha[:, None] + log_A_t[None], axis=1) + log_emit[:, t]

        # Freeze on padding: where mask[b, t] == 0 keep the old alpha
        alpha = keras.ops.where(mask[:, t][:, None] > 0, new_alpha, alpha)

    # Log-likelihood per sample: logsumexp over the final alpha
    return keras.ops.logsumexp(alpha, axis=-1)


def _path_score(log_pi, log_A, log_emit, y, mask):
    """Gold-path (Viterbi / forward-backward) score for discriminative mode.

    :param log_pi: ``(K,)`` initial log-probs.
    :type log_pi: keras.KerasTensor
    :param log_A: ``(K, K)`` or ``(B, T, K, K)`` transition log-probs.
    :type log_A: keras.KerasTensor
    :param log_emit: ``(B, T, K)`` emission log-densities.
    :type log_emit: keras.KerasTensor
    :param y: Integer state-path targets ``(B, T)``; each entry in [0, K).
    :type y: keras.KerasTensor
    :param mask: ``(B, T)`` float padding mask (1 = valid).
    :type mask: keras.KerasTensor
    :return: Per-sample log-probability of the gold path ``(B,)``.
    :rtype: keras.KerasTensor
    """
    Y = keras.ops.one_hot(y, log_emit.shape[-1])  # (B, T, K)
    # Initial score: sum over initial states weighted by Y[:,0]
    s0 = keras.ops.sum(log_pi * Y[:, 0], axis=-1)  # (B,)

    # Emission score: sum over timesteps of (log_emit * Y) * mask
    # log_emit is (B,T,K), Y is (B,T,K) -> elementwise product, then sum over K
    se = keras.ops.sum(keras.ops.sum(log_emit * Y, axis=-1) * mask, axis=-1)  # (B,)

    # Transition score: einsum over consecutive Y pairs
    if len(log_A.shape) == 2:
        # Static A: (K,K)
        tr = keras.ops.einsum("bti,ij,btj->bt", Y[:, :-1], log_A, Y[:, 1:])
    else:
        # Dynamic A: (B,T,K,K)
        tr = keras.ops.einsum("bti,btij,btj->bt", Y[:, :-1], log_A[:, 1:], Y[:, 1:])

    # Apply mask to the transition part (skip first timestep)
    tr_masked = tr * mask[:, 1:]

    return s0 + se + keras.ops.sum(tr_masked, axis=-1)


# ---------------------------------------------------------------------
# Viterbi — max-probability state path
# ---------------------------------------------------------------------


def viterbi(log_pi, log_A, log_emit) -> Tuple:
    """Viterbi decoding: most-likely state path and its log-probability.

    Finds ``argmax_path sum_{t} log_emit[t, path[t]] + log_pi[path[0]] +
    sum_{t>0} log_A[t-1, path[t-1], path[t]]``.

    :param log_pi: Initial log-probs ``(K,)``.
    :type log_pi: keras.KerasTensor
    :param log_A: Transition log-probs ``(K, K)`` (static) or ``(T, K, K)``
        (first timestep only; the function assumes static A and drops the batch/time
        dims if present). The function accepts ``(K, K)`` and will broadcast
        correctly if a single timestep of dynamic A is passed.
    :type log_A: keras.KerasTensor
    :param log_emit: Emission log-densities ``(T, K)`` (static, no batch dim).
    :type log_emit: keras.KerasTensor
    :return: ``(path, log_probability)`` where ``path`` is ``(T,)`` int tensor
        and ``log_probability`` is a scalar.
    :rtype: Tuple of (keras.KerasTensor, keras.KerasTensor)
    """
    # Handle possible batch/time dims by squeezing to static shape
    # log_A may be (K,K) or (1,K,K) or (T,K,K); keep only the last two dims
    if log_A.ndim == 3:
        log_A = log_A[0]  # take first timestep if dynamic

    if log_emit.ndim == 3:
        log_emit = log_emit[0]  # drop batch dim if present

    T_K = log_emit.shape
    if len(T_K) == 2:
        T, K = T_K
    elif len(T_K) == 1:
        K = T_K[0]
        T = 1
    else:
        raise ValueError(f"log_emit must be (T,K) or (K,), got shape {T_K}")

    if log_A.shape != (K, K):
        raise ValueError(f"log_A must be (K,K) after squeezing, got {log_A.shape}")

    # Initialization
    delta = log_pi + log_emit[0]  # (K,)
    back_list = [ops.zeros(K, dtype="int32")]

    # Recursion
    for t in range(1, T):
        # delta[:, None] + log_A: (K, 1) + (K, K) = (K, K)
        s = delta[:, None] + log_A  # (K, K)
        back_list.append(ops.argmax(s, axis=0))  # (K,) — argmax over previous states
        delta = ops.max(s, axis=0) + log_emit[t]  # (K,)

    # Termination
    path_list = [ops.argmax(delta)]  # scalar, last state

    # Backtrack
    for t in range(T - 1, 0, -1):
        path_list.append(back_list[t][path_list[-1]])

    # Reverse to get forward path
    path_list.reverse()
    path = ops.stack(path_list, axis=0)  # (T,)

    # Log-probability of the found path
    lp = log_pi[path[0]]
    for t in range(1, T):
        lp = lp + log_A[path[t - 1], path[t]] + log_emit[t, path[t]]

    return path, lp


# ---------------------------------------------------------------------
# Forward-backward posteriors — marginal γ_t(k) = p(z_t = k | x_{1:T})
# ---------------------------------------------------------------------


def posteriors(log_pi, log_A, log_emit):
    """Posterior marginals γ_t(k) = p(z_t = k | x_{1:T}) in log-space.

    Uses the standard forward-backward algorithm, returning probabilities
    (not log-probs) ``(T, K)`` that sum to 1 over k at each timestep.

    :param log_pi: Initial log-probs ``(K,)``.
    :type log_pi: keras.KerasTensor
    :param log_A: Transition log-probs ``(K, K)`` (static).
    :type log_A: keras.KerasTensor
    :param log_emit: Emission log-densities ``(T, K)``.
    :type log_emit: keras.KerasTensor
    :return: Posterior matrix ``(T, K)``; ``γ_t(k) >= 0`` and ``sum_k γ_t(k) = 1``
        (up to floating point) for each t.
    """
    T_K = log_emit.shape
    if len(T_K) == 2:
        T, K = T_K
    else:
        raise ValueError(f"log_emit must be (T,K), got shape {T_K}")

    if log_A.shape != (K, K):
        raise ValueError(f"log_A must be (K,K), got {log_A.shape}")

    # Forward alphas: α_t(k) = p(z_{1:t}, z_t = k | x_{1:t})
    # Use list-based accumulation to avoid tensor item assignment issues
    la_list = [log_pi + log_emit[0]]  # (K,)

    for t in range(1, T):
        # α_t = logsumexp(α_{t-1}[:, None] + log_A, axis=1) + log_emit[t]
        # logsumexp over axis=1 of (K, K) -> (K,)
        la_t = ops.logsumexp(la_list[t - 1][:, None] + log_A[None], axis=1) + log_emit[t]
        la_list.append(la_t)

    # Stack requires all tensors to have same shape; ensure (1, K) for all
    la_list_2d = [x[None] if x.ndim == 1 else x for x in la_list]
    la = ops.concatenate(la_list_2d, axis=0)  # (T, K)

    # Backward betas: β_{t-1}(k) = p(z_{t+1:T} | z_{t-1} = k, x_{1:T})
    lb_list = [ops.zeros(K)]  # beta at T is 0 (log 1)

    for t in range(T - 2, -1, -1):
        # β_t(k) = logsumexp(A[t+1] + (log_emit[t+1] + β_{t+1})[None], axis=1)
        # A[t+1] is (K,K); we need the transition at time t+1 for static A
        # which is just log_A
        lb_t = ops.logsumexp(
            log_A[None] + (log_emit[t + 1] + lb_list[0])[None], axis=1
        )
        lb_list.insert(0, lb_t)

    lb_list_2d = [x[None] if x.ndim == 1 else x for x in lb_list]
    lb = ops.concatenate(lb_list_2d, axis=0)  # (T, K)

    # Joint gamma in log-space: g_t(k) = α_t(k) + β_t(k)
    g = la + lb

    # Normalise per timestep: exp(g - logsumexp(g, axis=1, keepdims=True))
    # This yields probabilities that sum to 1 over k at each t.
    # keras.ops.logsumexp with keepdims:
    logZ = ops.logsumexp(g, axis=1, keepdims=True)  # (T, 1)
    g = g - logZ  # (T, K)
    probs = ops.exp(g)  # (T, K)

    return probs


# ---------------------------------------------------------------------
# Convenience: decode a model output to numpy (for one-sample inference)
# ---------------------------------------------------------------------


def decode(log_pi_np: np.ndarray,
           log_A_np: np.ndarray,
           log_emit_np: np.ndarray) -> Tuple[np.ndarray, float]:
    """Decode a single sample's forward outputs to a Viterbi path.

    This is the numpy / scipy port of the recipe's ``viterbi``; it exists so
    callers that need a pure-Numpy path (e.g. evaluation scripts outside tf)
    can import it without pulling in keras.ops inside a graph.

    :param log_pi_np: Initial log-probs ``(K,)``.
    :type log_pi_np: np.ndarray
    :param log_A_np: Transition log-probs ``(K, K)``.
    :type log_A_np: np.ndarray
    :param log_emit_np: Emission log-densities ``(T, K)``.
    :type log_emit_np: np.ndarray
    :return: ``(path, log_probability)`` where ``path`` is ``(T,)`` int and
        ``log_probability`` is a float.
    :rtype: Tuple of (np.ndarray, float)
    """
    from scipy.special import logsumexp

    K = log_pi_np.shape[0]
    T = log_emit_np.shape[0]

    # Viterbi recursion (numpy)
    delta = log_pi_np + log_emit_np[0]  # (K,)
    back = np.zeros((T, K), dtype=int)

    for t in range(1, T):
        s = delta[:, None] + log_A_np  # (K, K)
        back[t] = np.argmax(s, axis=0)  # (K,)
        delta = np.max(s, axis=0) + log_emit_np[t]  # (K,)

    path = np.empty(T, dtype=int)
    path[T - 1] = np.argmax(delta)

    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    # Log probability of the found path
    lp = log_pi_np[path[0]]
    for t in range(1, T):
        lp = lp + log_A_np[path[t - 1], path[t]] + log_emit_np[t, path[t]]

    return path, lp


# ---------------------------------------------------------------------
# Gradient-flow note
# ---------------------------------------------------------------------

# Both `viterbi` and `posteriors` are pure functions of the forward outputs.
# They contain no trainable variables, no `add_weight`, no `self.call`, and no
# Keras layer state. They are safe to call inside a `tf.function` or with
# `jit_compile=True` because they use only `keras.ops` (which compile to
# backend-agnostic graphs) and plain Python loops over the sequence length T,
# which is a concrete integer at graph-construction time (or a dynamic size
# that the while-loop / scan handles). The `posteriors` function's softmax and
# exp are all `keras.ops`‑wrapped and compile without issue.