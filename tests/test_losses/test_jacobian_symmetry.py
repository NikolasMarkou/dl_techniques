"""Isolated numerical tests for ``jacobian_symmetry_penalty``.

These derisk the single genuine unknown of the plan (the reverse-mode double-VJP
Jacobian-symmetry estimator) BEFORE any trainer wiring. All tensors are tiny
(batch 2, 8x8x3) so the suite is fast and CPU-runnable.

Coverage:
    A. A provably SYMMETRIC linear op (1x1 conv with a symmetric channel-mix
       matrix) -> penalty ~ 0 (< 1e-5).
    B. A provably ASYMMETRIC linear op (1x1 conv, A != A^T) -> penalty > 1e-3.
    C. A tiny bias-free conv + LeakyReLU net -> gradients w.r.t. every trainable
       weight are non-None and finite.
    D. float32 + finiteness -- the returned penalty is a finite float32 scalar
       even when the model is built/run under a ``mixed_float16`` global policy.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.losses.jacobian_symmetry import jacobian_symmetry_penalty


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _channel_mix_model(matrix: np.ndarray) -> keras.Model:
    """A 1x1 conv with no bias whose per-pixel Jacobian equals ``matrix``.

    ``out[c] = sum_in in[in] * kernel[0, 0, in, c]``, so the per-pixel Jacobian
    ``d out[c] / d in[in] == kernel[in, c] == matrix[in, c]``. The full input->
    output Jacobian is block-diagonal (one ``matrix`` block per pixel), so it is
    symmetric iff ``matrix`` is symmetric.
    """
    c = matrix.shape[0]
    inp = keras.Input(shape=(8, 8, c))
    conv = keras.layers.Conv2D(c, kernel_size=1, use_bias=False)
    out = conv(inp)
    model = keras.Model(inp, out)
    # Conv2D kernel shape: (kh, kw, in_ch, out_ch) == (1, 1, c, c).
    kernel = matrix.astype("float32").reshape(1, 1, c, c)
    conv.set_weights([kernel])
    return model


def _bias_free_convnet() -> keras.Model:
    """Tiny bias-free conv + LeakyReLU net; output shape matches input."""
    inp = keras.Input(shape=(8, 8, 3))
    x = keras.layers.Conv2D(8, 3, padding="same", use_bias=False)(inp)
    x = keras.layers.LeakyReLU(negative_slope=0.2)(x)
    out = keras.layers.Conv2D(3, 3, padding="same", use_bias=False)(x)
    return keras.Model(inp, out)


def _sample_input(seed: int = 0) -> tf.Tensor:
    rng = np.random.default_rng(seed)
    return tf.constant(rng.standard_normal((2, 8, 8, 3)).astype("float32"))


# ---------------------------------------------------------------------
# Test A -- symmetric op -> ~0
# ---------------------------------------------------------------------


class TestJacobianSymmetryPenalty:
    def test_symmetric_op_is_near_zero(self):
        # A symmetric 3x3 channel-mix matrix (A == A^T).
        a = np.array(
            [[1.0, 0.3, -0.5],
             [0.3, 2.0, 0.7],
             [-0.5, 0.7, 1.5]],
            dtype="float32",
        )
        assert np.allclose(a, a.T), "test setup: matrix must be symmetric"
        model = _channel_mix_model(a)
        y = _sample_input(seed=1)

        penalty = jacobian_symmetry_penalty(model, y, num_probes=2, seed=42)

        assert float(penalty) < 1e-5, (
            f"symmetric op should have ~0 asymmetry, got {float(penalty)}"
        )

    # -----------------------------------------------------------------
    # Test B -- asymmetric op -> > 1e-3
    # -----------------------------------------------------------------

    def test_asymmetric_op_is_positive(self):
        # A clearly non-symmetric 3x3 channel-mix matrix (A != A^T).
        a = np.array(
            [[1.0, 2.0, 0.0],
             [-1.0, 1.0, 3.0],
             [0.5, -2.0, 1.0]],
            dtype="float32",
        )
        assert not np.allclose(a, a.T), "test setup: matrix must be asymmetric"
        model = _channel_mix_model(a)
        y = _sample_input(seed=2)

        penalty = jacobian_symmetry_penalty(model, y, num_probes=2, seed=7)

        assert float(penalty) > 1e-3, (
            f"asymmetric op should have clearly positive asymmetry, "
            f"got {float(penalty)}"
        )

    # -----------------------------------------------------------------
    # Test C -- gradients flow to weights (non-None, finite)
    # -----------------------------------------------------------------

    def test_gradients_flow_to_weights(self):
        model = _bias_free_convnet()
        y = _sample_input(seed=3)

        with tf.GradientTape() as tape:
            penalty = jacobian_symmetry_penalty(model, y, num_probes=1, seed=11)
        grads = tape.gradient(penalty, model.trainable_variables)

        assert len(grads) == len(model.trainable_variables)
        assert len(grads) > 0, "toy net should have trainable weights"
        for g, var in zip(grads, model.trainable_variables):
            assert g is not None, f"None gradient for {var.name}"
            assert bool(tf.reduce_all(tf.math.is_finite(g))), (
                f"non-finite gradient for {var.name}"
            )

    # -----------------------------------------------------------------
    # Test D -- float32 + finite, even under a mixed_float16 policy
    # -----------------------------------------------------------------

    def test_float32_and_finite_under_mixed_precision(self):
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            model = _bias_free_convnet()  # built under mixed_float16
            y = _sample_input(seed=4)  # float32 input

            penalty = jacobian_symmetry_penalty(model, y, num_probes=1, seed=5)

            assert penalty.dtype == tf.float32, (
                f"penalty must be float32, got {penalty.dtype}"
            )
            assert bool(tf.math.is_finite(penalty)), "penalty must be finite"
        finally:
            keras.mixed_precision.set_global_policy(original_policy)
