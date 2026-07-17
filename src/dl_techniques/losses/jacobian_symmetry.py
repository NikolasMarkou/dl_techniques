"""Stochastic Jacobian-symmetry penalty (reverse-mode double-VJP, float32-forced).

A soft regularizer that pushes a denoiser's input-output Jacobian toward being
symmetric (i.e. toward being a conservative / gradient field). For a model
``D`` with output ``d = D(y)``, the local Jacobian is ``J = dd/dy``. A perfectly
conservative denoiser (one that is the gradient of a scalar potential, as the
MMSE denoiser is under Miyasawa) has a **symmetric** Jacobian ``J = Jᵀ``. This
module estimates the squared asymmetry via random probes::

    penalty = E_v[ ||J v - Jᵀ v||² ] / <normalizer>

using a single random probe ``v`` (Hutchinson-style), averaged over
``num_probes`` draws.

**Mechanism (crux).** The full explicit Jacobian is a documented TF-2.18
jit-conv dead-end (432 unrolled columns; see
``research/papers/bfunet/measure_jacobian.py``). A *single* JVP/VJP pair is a
different, far cheaper computation that composes cleanly. This module uses the
pure-reverse-mode "double-VJP" (forward-over-reverse) recipe -- it needs only
``tf.GradientTape`` (no ``tf.autodiff.ForwardAccumulator``), matching the repo's
established nested-tape idiom (``models/thera/hypernetwork.py``,
``layers/physics/lagrange_layer.py``):

- ``Jᵀv`` is one reverse pass: ``tape.gradient(D(y), y, output_gradients=v)``.
- ``Jv`` is obtained by differentiating the VJP ``(Jᵀu)`` w.r.t. its seed ``u``
  (a linear function of ``u``) in the direction ``v``: ``Jv = d/du[(Jᵀu)·v]``.
  This is the standard "JVP without forward mode" (Hessian-vector-product style)
  trick.

**Numerics (HARD constraint).** A JVP-of-VJP differentiates the network's
backward pass a second time, so any ``(var + eps)^k`` op inside the backbone
inherits the repo's known fp16/XLA silent-training-death risk one differentiation
order deeper (LESSONS: ``EnergyLayerNorm`` backward overflow; ``-1e9`` -> fp16
``-inf``). This module therefore **forces float32**: the input ``y`` is cast to
float32 at entry, the whole nested-tape computation runs in float32, and a
float32 scalar is returned -- regardless of the global mixed-precision policy.
Callers combining this with ``mixed_precision`` must additionally set
``jit_compile=False`` for the penalty path (enforced by the trainer's
fail-closed config validation).

**Inference map (HARD constraint).** Both forward passes run with
``training=False``. This is a correctness requirement, not a style choice
(decisions.md D-005): (1) the Miyasawa conservative-field property is about the
*deployable* denoiser ``D(y)``, so the penalty must regularize the inference map,
not the training-time stochastic ensemble; (2) with ``training=True`` a backbone
containing ``StochasticDepth`` / drop-path (every non-tiny bfconvunext variant has
``drop_path_rate >= 0.1``) samples INDEPENDENT masks on the ``Jᵀv`` and ``Jv``
passes, so the estimator would measure ``Jv(maskB) - Jᵀv(maskA)`` -- mask-mismatch
noise conflated with true asymmetry, and non-deterministic across identical calls;
(3) ``training=True`` also mutates ``BiasFreeBatchNorm`` running stats, which during
``test_step`` validation would silently corrupt the checkpoint's inference-time
normalization. ``training=False`` makes drop-path an identity (deterministic, single
map) and freezes BN stats, resolving all three. Weight-differentiability is
unaffected -- the ``training`` flag only changes drop_path/BN behavior, not whether
gradients flow to ``model.trainable_variables``.

This is a module of **pure functions** (mirroring
``losses/thera_jacobian_tv.py``), not a ``keras.losses.Loss`` subclass: the
penalty needs raw access to the model and its input ``y`` (it runs extra
tape-wrapped forward passes through the model itself), which a stock
per-output ``(y_true, y_pred)`` ``Loss`` cannot express.
"""

from typing import Any, Optional

import tensorflow as tf

# ---------------------------------------------------------------------
# Numerical floor for the relative normalizer. The penalty is normalized by the
# mean squared probe response so it is scale-comparable across probe draws and
# batches; this epsilon prevents a divide-by-zero when a probe maps to (near-)
# zero response (e.g. a dead/constant model).
_NORM_EPS = tf.constant(1e-12, dtype=tf.float32)


# ---------------------------------------------------------------------


def jacobian_symmetry_penalty(
    model: Any,
    y: Any,
    num_probes: int = 1,
    seed: Optional[int] = None,
) -> tf.Tensor:
    """Stochastic Jacobian-symmetry penalty ``mean(||Jv - Jᵀv||²)`` (normalized).

    Estimates how far the model's input-output Jacobian ``J = d(model(y))/dy``
    is from symmetric, via ``num_probes`` random Gaussian probes ``v`` (same
    shape as ``y``). For each probe it computes ``Jv`` (a JVP, via the
    reverse-mode double-VJP trick) and ``Jᵀv`` (a VJP), then the mean squared
    difference, normalized by the mean squared probe response so the value is
    scale-comparable. The whole computation is forced to float32.

    The returned scalar is differentiable w.r.t. ``model.trainable_variables``
    (the nested tapes compose with an outer weight-gradient tape), so it can be
    added directly to a training objective.

    Interface contract:
        - ``model``: a callable Keras model. It is invoked as
          ``model(y, training=False)`` (the DEPLOYABLE inference map -- see the
          module docstring "Inference map" note and decisions.md D-005) and must
          return a single tensor with the SAME shape as ``y`` (a denoiser: input
          image -> output image). It must be differentiable w.r.t. its input.
        - ``y``: input tensor, any shape ``(B, ...)``. Cast to float32 internally;
          the caller's dtype/policy is not mutated.
        - ``num_probes``: number of random probes to average over (>= 1).
        - ``seed``: optional int. When given, probes are drawn with
          ``tf.random.stateless_normal`` for reproducibility (probe ``i`` uses
          seed ``[seed, i]``); when ``None``, ``tf.random.normal`` is used.
        - Returns: a float32 scalar tensor >= 0. Zero iff ``J`` is symmetric in
          every probed direction. Never returns ``None``; raises ``ValueError``
          for ``num_probes < 1``.
        - Failure mode: if the model's backward pass produces ``None`` for
          ``Jᵀv`` (input not connected to output), a ``TypeError`` propagates
          from ``tf.GradientTape.gradient`` -- callers should treat that as a
          hard configuration error, not retry.

    Args:
        model: Callable Keras model mapping ``y`` to a same-shaped output.
        y: Input tensor (any shape with a leading batch axis).
        num_probes: Number of random probes to average (default 1). Must be >= 1.
        seed: Optional RNG seed for reproducible probes (default None).

    Returns:
        A float32 scalar tensor: the probe-averaged normalized squared Jacobian
        asymmetry ``mean_probes( ||Jv - Jᵀv||² / (mean(||Jv||²+||Jᵀv||²)/2 + eps) )``.

    Raises:
        ValueError: if ``num_probes < 1``.
    """
    if num_probes < 1:
        raise ValueError(f"num_probes must be >= 1, got {num_probes}")

    # HARD: force float32 for the entire second-order computation (see module
    # docstring / decisions.md D-003 fp16-XLA landmine).
    y = tf.cast(y, tf.float32)

    penalty_accum = tf.zeros((), dtype=tf.float32)
    for i in range(num_probes):
        if seed is None:
            v = tf.random.normal(tf.shape(y), dtype=tf.float32)
        else:
            v = tf.random.stateless_normal(
                tf.shape(y), seed=[seed, i], dtype=tf.float32
            )
        penalty_accum += _single_probe_asymmetry(model, y, v)

    return penalty_accum / tf.cast(num_probes, tf.float32)


# ---------------------------------------------------------------------


def _single_probe_asymmetry(model: Any, y: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
    """Normalized squared asymmetry ``||Jv - Jᵀv||²`` for one probe ``v``.

    Internal helper. ``y`` and ``v`` are assumed float32 and same-shaped.
    Uses the reverse-mode double-VJP recipe (see module docstring). Returns a
    float32 scalar; differentiable w.r.t. ``model.trainable_variables``.
    """
    # Jᵀv : single reverse pass.
    # DECISION plan-2026-07-17T112359-874b11cc/D-005: BOTH forward passes MUST use
    # training=False. Do NOT revert to training=True: on any drop_path>0 backbone
    # (every non-tiny bfconvunext variant) the two passes would sample INDEPENDENT
    # StochasticDepth masks -> the estimate becomes Jv(maskB)-Jᵀv(maskA) (mask-
    # mismatch noise, nondeterministic across identical calls) AND it would mutate
    # BiasFreeBatchNorm running_var during test_step validation. training=False
    # makes drop_path a deterministic identity and freezes BN stats, so the penalty
    # measures the DEPLOYABLE inference denoiser D(y). Weight-differentiability is
    # unaffected. See decisions.md D-005 + module docstring "Inference map" note.
    with tf.GradientTape() as t1:
        t1.watch(y)
        d = model(y, training=False)
        d = tf.cast(d, tf.float32)
    jt_v = t1.gradient(d, y, output_gradients=v)

    # Jv : differentiate the VJP (Jᵀu), which is linear in its seed u, w.r.t. u
    # in the direction v. This is the "JVP without forward mode" trick.
    with tf.GradientTape() as t3:
        with tf.GradientTape() as t2:
            t2.watch(y)
            d2 = model(y, training=False)
            d2 = tf.cast(d2, tf.float32)
        u = tf.ones_like(d2)
        t3.watch(u)
        vjp_u = t2.gradient(d2, y, output_gradients=u)
    j_v = t3.gradient(vjp_u, u, output_gradients=v)

    diff = j_v - jt_v
    numerator = tf.reduce_sum(diff * diff)
    # Symmetric, scale-comparable normalizer: average squared probe response.
    normalizer = 0.5 * (
        tf.reduce_sum(j_v * j_v) + tf.reduce_sum(jt_v * jt_v)
    ) + _NORM_EPS
    return numerator / normalizer

# ---------------------------------------------------------------------
