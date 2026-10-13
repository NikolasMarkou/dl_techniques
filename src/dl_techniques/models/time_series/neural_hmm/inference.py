"""
Exact HMM inference in log space for the Neural HMM: the forward algorithm, the score of
a given state path, Viterbi decoding, and forward-backward state posteriors. Every
function takes the model's three outputs on one uniform, batched shape and a padding
mask, and has no weights and no state, so the training loss, the decoding helpers and
the trainer's evaluation all run the same code.

Shapes (``B`` batch, ``T`` time, ``K`` states):

- ``log_pi``: ``(B, K)``, initial-state log-potentials.
- ``log_A``: ``(B, T, K, K)``. ``log_A[:, t, i, j]`` scores the transition from state
  ``i`` at step ``t - 1`` INTO state ``j`` at step ``t``: the row axis is the source and
  the column axis the destination. The ``t = 0`` slot is never read.
- ``log_emit``: ``(B, T, K)``, the emission log-potential of each state at each step.
- ``mask``: ``(B, T)``, nonzero at valid steps. It must be a contiguous valid PREFIX
  (right padding). A padded step contributes no emission and no transition, so the
  result for a padded sequence equals the result for the truncated sequence. Interior
  gaps are not supported and are not checked.

The recursions run as ``keras.ops.while_loop`` over time, so the time axis may be unknown
at trace time (Keras retraces ``predict`` and ``evaluate`` with an unknown length from the
second sequence length on), and the forward pass can be differentiated under XLA. All
arithmetic runs in float32, or float64 when an input is float64; float16 and bfloat16
inputs are promoted, because a log-space recursion accumulates.

References:
    - Rabiner, 1989. A Tutorial on Hidden Markov Models and Selected Applications in
      Speech Recognition. Proceedings of the IEEE 77(2), pp. 257-286.
"""

from typing import Tuple

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.dtype_policy import accumulation_dtype

# ---------------------------------------------------------------------


def _prepare(log_pi, log_A, log_emit, mask):
    """Check ranks, promote to the accumulation dtype and build the validity mask.

    :param log_pi: ``(B, K)`` initial log-potentials.
    :param log_A: ``(B, T, K, K)`` transition log-potentials.
    :param log_emit: ``(B, T, K)`` emission log-potentials.
    :param mask: ``(B, T)`` padding mask, nonzero at valid steps.
    :return: ``(log_pi, log_A, log_emit, valid)`` with the three float inputs cast to
        one accumulation dtype and ``valid`` a boolean ``(B, T)`` tensor.
    :rtype: tuple
    :raises ValueError: If an input does not have the documented rank.
    """
    expected = (("log_pi", log_pi, 2), ("log_A", log_A, 4), ("log_emit", log_emit, 3), ("mask", mask, 2))
    for name, tensor, rank in expected:
        if len(tensor.shape) != rank:
            raise ValueError(f"{name} must have rank {rank}, got shape {tuple(tensor.shape)}.")
    widths = {accumulation_dtype(t.dtype) for t in (log_pi, log_A, log_emit)}
    accum = "float64" if "float64" in widths else "float32"
    valid = keras.ops.greater(keras.ops.cast(mask, accum), 0)
    return (
        keras.ops.cast(log_pi, accum),
        keras.ops.cast(log_A, accum),
        keras.ops.cast(log_emit, accum),
        valid,
    )


def _time_loop(body, carry, num_steps, reverse=False):
    """Apply ``carry = body(t, carry)`` for every transition step ``t = 1, ..., T - 1``.

    With ``reverse=True`` the steps run ``t = T - 1, ..., 1`` instead. ``num_steps`` is
    ``T`` and may be a Python int or a scalar tensor (an unknown time axis); ``T = 1``
    runs no step.

    :param body: Callable ``(t, carry) -> carry`` preserving the carry's structure.
    :param carry: Initial loop state (a tensor or a tuple of tensors).
    :param num_steps: Length of the time axis ``T``.
    :param reverse: Whether to count down instead of up.
    :return: The final carry.
    """

    # DECISION plan-2026-10-13T192257-1bb3bf1f/D-014: keras.ops.while_loop WITH
    # maximum_iterations. Do NOT use a Python loop over ``x.shape[1]``: Keras retraces
    # predict/evaluate with T = None from the second sequence length on (measured), and
    # range(None) raises. Do NOT use keras.ops.scan: its TensorArray has no fixed size,
    # so differentiating it under XLA raises "XLA compilation requires a fixed tensor
    # list size" (measured, CPU jit_compile=True); maximum_iterations sizes the
    # gradient accumulators.
    def cond(t, _):
        return keras.ops.greater_equal(t, 1) if reverse else keras.ops.less(t, num_steps)

    def step(t, state):
        return (t - 1 if reverse else t + 1), body(t, state)

    first = num_steps - 1 if reverse else 1
    _, carry = keras.ops.while_loop(cond, step, (first, carry), maximum_iterations=num_steps)
    return carry


def _write(history, t, value):
    """Return ``history`` with ``history[:, t] = value``, without a scatter.

    :param history: ``(B, T, ...)`` tensor.
    :param t: Scalar int step index.
    :param value: ``(B, ...)`` tensor of ``history``'s dtype.
    :return: The updated ``(B, T, ...)`` tensor.
    """
    steps = keras.ops.arange(keras.ops.shape(history)[1], dtype="int32")
    at_t = keras.ops.equal(steps, keras.ops.cast(t, "int32"))
    at_t = keras.ops.reshape(at_t, (1, -1) + (1,) * (len(history.shape) - 2))
    return keras.ops.where(at_t, keras.ops.expand_dims(value, 1), history)


def _select(values, index, n_states):
    """Return ``values[b, index[b]]`` per row by a one-hot select (no take_along_axis).

    :param values: ``(B, K)`` tensor.
    :param index: ``(B,)`` int tensor in ``[0, K)``.
    :param n_states: ``K``.
    :return: ``(B,)`` tensor.
    """
    chosen = keras.ops.one_hot(index, n_states) > 0
    return keras.ops.sum(keras.ops.where(chosen, values, keras.ops.zeros_like(values)), axis=-1)


def _forward_alphas(log_pi, log_A, log_emit, valid):
    """Run the forward recursion and return every step's alpha.

    :param log_pi: ``(B, K)`` accumulation-dtype initial log-potentials.
    :param log_A: ``(B, T, K, K)`` accumulation-dtype transition log-potentials.
    :param log_emit: ``(B, T, K)`` accumulation-dtype emission log-potentials.
    :param valid: ``(B, T)`` boolean validity mask.
    :return: ``(alpha_last, alphas)``: ``(B, K)`` alpha after the last valid step and
        ``(B, T, K)`` alpha at every step (frozen across padding).
    """
    # DECISION plan-2026-10-13T192257-1bb3bf1f/D-009: a padded step FREEZES alpha
    # (no emission, no transition), so a padded sequence's logZ equals the truncated
    # sequence's exactly. Do NOT zero the emission and keep the transition instead.
    alpha = log_pi + keras.ops.where(valid[:, 0, None], log_emit[:, 0], keras.ops.zeros_like(log_pi))
    alphas = _write(keras.ops.zeros_like(log_emit), 0, alpha)

    def body(t, carry):
        alpha, alphas = carry
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-003: alpha sits on the SOURCE axis
        # (``alpha[:, :, None] + log_A[:, t]``, logsumexp over axis 1). Do NOT
        # broadcast it on the destination axis (``alpha[:, None, :]``): that is the
        # measured defect of the old package (logZ [1.331, 0.750] against the true
        # [0.636, 2.153]).
        moved = keras.ops.logsumexp(alpha[:, :, None] + log_A[:, t], axis=1) + log_emit[:, t]
        alpha = keras.ops.where(valid[:, t, None], moved, alpha)
        return alpha, _write(alphas, t, alpha)

    num_steps = keras.ops.shape(log_emit)[1]
    return _time_loop(body, (alpha, alphas), num_steps)


def _empty_to_zero(values, valid):
    """Return ``values`` with the entries of sequences that have no valid step set to 0.

    :param values: ``(B,)`` tensor.
    :param valid: ``(B, T)`` boolean validity mask.
    :return: ``(B,)`` tensor.
    """
    return keras.ops.where(keras.ops.any(valid, axis=1), values, keras.ops.zeros_like(values))


def forward_log_likelihood(log_pi, log_A, log_emit, mask):
    """Return the log partition function of every sequence, by the forward algorithm.

    ``logZ = log sum_z exp(log_pi[z_0] + log_emit[0, z_0]
    + sum_{t>=1} (log_A[t, z_{t-1}, z_t] + log_emit[t, z_t]))`` over all state paths of
    the sequence's valid length. With normalized ``log_pi``/``log_A`` and emission
    log-densities this is ``log p(x)``. A sequence with no valid step has ``logZ = 0``
    (the empty path), whatever ``log_pi`` holds.

    :param log_pi: ``(B, K)`` initial log-potentials.
    :param log_A: ``(B, T, K, K)`` transition log-potentials, source row to destination
        column; ``log_A[:, 0]`` is ignored.
    :param log_emit: ``(B, T, K)`` emission log-potentials.
    :param mask: ``(B, T)`` right-padding mask, nonzero at valid steps.
    :return: ``(B,)`` log partition function, in float32 (float64 for float64 input).
    :raises ValueError: If an input does not have the documented rank.
    """
    log_pi, log_A, log_emit, valid = _prepare(log_pi, log_A, log_emit, mask)
    alpha_last, _ = _forward_alphas(log_pi, log_A, log_emit, valid)
    return _empty_to_zero(keras.ops.logsumexp(alpha_last, axis=-1), valid)


def path_log_score(log_pi, log_A, log_emit, states, mask):
    """Return the unnormalized log score of a given state path for every sequence.

    ``log_pi[z_0] + log_emit[0, z_0] + sum_{t>=1} (log_A[t, z_{t-1}, z_t]
    + log_emit[t, z_t])`` summed over valid steps only, so state values at padded steps
    are ignored and may hold anything, including ``-1``. A sequence with no valid step
    scores 0, matching :func:`forward_log_likelihood`, so ``logZ - score`` is the
    path's negative log-probability and is 0 for an empty sequence.

    :param log_pi: ``(B, K)`` initial log-potentials.
    :param log_A: ``(B, T, K, K)`` transition log-potentials; ``log_A[:, 0]`` is ignored.
    :param log_emit: ``(B, T, K)`` emission log-potentials.
    :param states: ``(B, T)`` integer states in ``[0, K)`` at valid steps.
    :param mask: ``(B, T)`` right-padding mask, nonzero at valid steps.
    :return: ``(B,)`` path score, in the accumulation dtype.
    :raises ValueError: If an input does not have the documented rank.
    """
    log_pi, log_A, log_emit, valid = _prepare(log_pi, log_A, log_emit, mask)
    n_states = log_emit.shape[-1]
    picked = keras.ops.one_hot(keras.ops.cast(states, "int32"), n_states) > 0  # (B,T,K)

    # Select with where, never by multiplying a one-hot: 0 * -inf is NaN.
    start = keras.ops.sum(keras.ops.where(picked[:, 0], log_pi, keras.ops.zeros_like(log_pi)), axis=-1)
    emit = keras.ops.sum(keras.ops.where(picked, log_emit, keras.ops.zeros_like(log_emit)), axis=-1)
    pair = keras.ops.logical_and(picked[:, :-1, :, None], picked[:, 1:, None, :])
    into = log_A[:, 1:]
    trans = keras.ops.sum(keras.ops.where(pair, into, keras.ops.zeros_like(into)), axis=(2, 3))

    # DECISION plan-2026-10-13T192257-1bb3bf1f/D-009: the start term, each emission
    # and each transition INTO a padded step are dropped, matching the forward freeze.
    emit = keras.ops.where(valid, emit, keras.ops.zeros_like(emit))
    trans = keras.ops.where(valid[:, 1:], trans, keras.ops.zeros_like(trans))
    start = keras.ops.where(valid[:, 0], start, keras.ops.zeros_like(start))
    return start + keras.ops.sum(emit, axis=1) + keras.ops.sum(trans, axis=1)


def viterbi_decode(log_pi, log_A, log_emit, mask) -> Tuple:
    """Return the highest-scoring state path of every sequence and its score.

    :param log_pi: ``(B, K)`` initial log-potentials.
    :param log_A: ``(B, T, K, K)`` transition log-potentials; ``log_A[:, 0]`` is ignored.
    :param log_emit: ``(B, T, K)`` emission log-potentials.
    :param mask: ``(B, T)`` right-padding mask, nonzero at valid steps.
    :return: ``(path, score)``: ``path`` is ``(B, T)`` int32 with ``-1`` at padded steps;
        ``score`` is ``(B,)``, equal to :func:`path_log_score` of ``path`` (0 for a
        sequence with no valid step). Ties resolve to the lowest state index.
    :raises ValueError: If an input does not have the documented rank.
    """
    log_pi, log_A, log_emit, valid = _prepare(log_pi, log_A, log_emit, mask)
    n_states = log_emit.shape[-1]
    num_steps = keras.ops.shape(log_emit)[1]
    identity = keras.ops.zeros_like(log_emit, dtype="int32") + keras.ops.arange(n_states, dtype="int32")

    delta = log_pi + keras.ops.where(valid[:, 0, None], log_emit[:, 0], keras.ops.zeros_like(log_pi))

    def forward(t, carry):
        delta, backpointers = carry
        scores = delta[:, :, None] + log_A[:, t]  # (B, K_source, K_destination)
        moved = keras.ops.max(scores, axis=1) + log_emit[:, t]
        pointer = keras.ops.cast(keras.ops.argmax(scores, axis=1), "int32")
        # D-009: a padded step freezes delta and keeps the identity backpointer.
        delta = keras.ops.where(valid[:, t, None], moved, delta)
        pointer = keras.ops.where(valid[:, t, None], pointer, identity[:, 0])
        return delta, _write(backpointers, t, pointer)

    delta, backpointers = _time_loop(forward, (delta, identity), num_steps)

    last = keras.ops.cast(keras.ops.argmax(delta, axis=-1), "int32")
    path = _write(keras.ops.zeros_like(identity[:, :, 0]), num_steps - 1, last)

    def backward(t, carry):
        state, path = carry
        previous = _select(backpointers[:, t], state, n_states)
        return previous, _write(path, t - 1, previous)

    _, path = _time_loop(backward, (last, path), num_steps, reverse=True)
    path = keras.ops.where(valid, path, keras.ops.full_like(path, -1))
    return path, _empty_to_zero(keras.ops.max(delta, axis=-1), valid)


def state_posteriors(log_pi, log_A, log_emit, mask):
    """Return the posterior state marginals ``p(z_t = k | x)`` by forward-backward.

    :param log_pi: ``(B, K)`` initial log-potentials.
    :param log_A: ``(B, T, K, K)`` transition log-potentials; ``log_A[:, 0]`` is ignored.
    :param log_emit: ``(B, T, K)`` emission log-potentials.
    :param mask: ``(B, T)`` right-padding mask, nonzero at valid steps.
    :return: ``(B, T, K)`` probabilities summing to 1 over ``K`` at every valid step,
        and exactly 0 at padded steps.
    :raises ValueError: If an input does not have the documented rank.
    """
    log_pi, log_A, log_emit, valid = _prepare(log_pi, log_A, log_emit, mask)
    alpha_last, alphas = _forward_alphas(log_pi, log_A, log_emit, valid)
    log_z = keras.ops.logsumexp(alpha_last, axis=-1)
    num_steps = keras.ops.shape(log_emit)[1]

    def body(t, carry):
        beta, betas = carry
        # beta_{t-1}[i] = logsumexp_j(log_A[t, i, j] + log_emit[t, j] + beta_t[j]): the
        # reduction runs over the DESTINATION axis (2). A padded step t freezes beta.
        moved = keras.ops.logsumexp(log_A[:, t] + (log_emit[:, t] + beta)[:, None, :], axis=2)
        beta = keras.ops.where(valid[:, t, None], moved, beta)
        return beta, _write(betas, t - 1, beta)

    beta = keras.ops.zeros_like(log_pi)
    _, betas = _time_loop(body, (beta, keras.ops.zeros_like(log_emit)), num_steps, reverse=True)

    gamma = keras.ops.exp(alphas + betas - log_z[:, None, None])
    return keras.ops.where(valid[:, :, None], gamma, keras.ops.zeros_like(gamma))
