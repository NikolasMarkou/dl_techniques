"""
Gradient-safety regression pins for the sHGCN / Poincare-ball math.

sHGCN evaluates the hyperbolic exponential map AT THE ORIGIN on training step 1:
SHGCNLayer defaults bias_initializer='zeros' with use_bias=True and calls
exp_map_0(self.bias, c), i.e. safe_norm(v) at exactly v = 0.

These are REGRESSION PINS, not proofs of a fixed bug. A review flagged
``safe_norm``'s ``maximum(norm(x), eps)`` as producing a NaN gradient there, on
the reasoning that ``d maximum/d norm`` is 0 below eps while ``d norm/dx`` at 0
is ``0/0``. That reasoning is sound in general but does NOT describe this stack:
measured on TF 2.18, ``tf.norm``/``keras.ops.norm`` return a gradient of exactly
0 at the origin rather than NaN, so both the old and the new formulation are
finite here and the claim was withdrawn. ``safe_norm`` was left unchanged --
moving eps inside the sqrt would have raised the forward value at the origin
from 1e-7 to sqrt(1e-7) ~ 3.2e-4, a real numerical change in exchange for
fixing nothing.

What these tests are worth: they fail if anyone replaces the norm with a
hand-rolled ``sqrt(sum(square(...)))`` that has no epsilon, which is the
formulation that genuinely does NaN at the origin.
"""

import numpy as np
import keras
import pytest



class TestPoincareSafeNormGradient:
    """safe_norm must give a finite gradient AT the origin."""

    def test_gradient_at_zero_is_finite(self):
        """safe_norm must give a finite gradient at x = 0.

        Currently satisfied by both the shipped ``maximum(norm(x), eps)`` and by
        an eps-inside-sqrt formulation, because TF's norm gradient is already 0
        (not NaN) at the origin -- see the module docstring. This pins the
        property against a future hand-rolled sqrt with no epsilon.
        """
        import tensorflow as tf
        from dl_techniques.utils.geometry.poincare_math import PoincareMath

        ball = PoincareMath()
        x = tf.Variable(tf.zeros((3, 4)))

        with tf.GradientTape() as tape:
            n = ball.safe_norm(x, axis=-1, keepdims=True)
            loss = tf.reduce_sum(n)
        grad = tape.gradient(loss, x)

        assert grad is not None, "no gradient reached x"
        g = keras.ops.convert_to_numpy(grad)
        assert np.all(np.isfinite(g)), (
            f"safe_norm produced non-finite gradient at the origin: "
            f"{g[~np.isfinite(g)][:4]}")

    def test_shgcn_layer_gradient_is_finite_at_init(self):
        """sHGCN hits the origin on step 1: bias inits to zeros and is exp-mapped."""
        import tensorflow as tf
        from dl_techniques.layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer import (
            SHGCNLayer,
        )

        keras.utils.set_random_seed(0)
        layer = SHGCNLayer(units=8, use_bias=True)

        n = 6
        x = tf.constant(np.random.default_rng(0).normal(size=(n, 4)), dtype="float32")
        adj = tf.constant(np.eye(n), dtype="float32")

        with tf.GradientTape() as tape:
            out = layer([x, adj])
            loss = tf.reduce_sum(tf.square(out))
        grads = tape.gradient(loss, layer.trainable_variables)

        bad = [v.name for g, v in zip(grads, layer.trainable_variables)
               if g is not None and not np.all(np.isfinite(
                   keras.ops.convert_to_numpy(g)))]
        assert not bad, f"non-finite gradients at initialization for: {bad}"
