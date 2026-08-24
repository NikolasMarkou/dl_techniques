"""RED proof for C-24: SCUNet must accept dynamic spatial dims.

`SCUNet` is fully convolutional and its module docstring advertises "reflection
padding so arbitrary input resolutions" work. Before this suite the pad was
computed as ``int(np.ceil(h / 64) * 64 - h)`` over ``ops.shape(x)[1]``, which is
a scalar TENSOR whenever the extent is ``None`` — so the natural build,
``SCUNet()(keras.Input((None, None, 3)))``, died inside numpy with no
model-level message. The 882-line sibling suite (including a whole
``TestSCUNetPaddingBehavior`` class) never uses a ``None`` dimension.

Two traps this suite is written around:

* ``model.predict`` / an eager call retraces per concrete shape, so neither can
  see a bug that exists only on the symbolic path. The forward assertions go
  through an explicit ``tf.function`` with a fully dynamic ``TensorSpec``.
* One concrete size can be satisfied by a model that baked in a constant, so
  every dynamic trace is exercised at TWO different sizes, one of them not a
  multiple of 64.
"""

from __future__ import annotations

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.models.scunet.model import SCUNet


def _tiny() -> SCUNet:
    return SCUNet(in_nc=3, config=[1] * 7, dim=16, head_dim=8,
                  window_size=8, input_resolution=64)


class TestSymbolicBuild:
    def test_functional_build_over_none_spatial_dims(self) -> None:
        inp = keras.Input((None, None, 3))
        out = _tiny()(inp)
        model = keras.Model(inp, out)
        assert model.output_shape[1:] == (None, None, 3), model.output_shape


# fp32 graph-vs-eager reassociation bound for SCUNet's output range; derived at
# the single use site below (D-031). ~52 ulp at |y| = 16.
_GRAPH_VS_EAGER_ATOL = 1e-4


class TestDynamicTrace:
    def test_one_dynamic_trace_serves_two_different_sizes(self) -> None:
        model = _tiny()

        @tf.function(input_signature=[
            tf.TensorSpec([None, None, None, 3], tf.float32)
        ])
        def infer(x):
            return model(x, training=False)

        a = tf.convert_to_tensor(
            np.random.default_rng(0).random((1, 96, 96, 3)).astype("float32"))
        b = tf.convert_to_tensor(
            np.random.default_rng(1).random((2, 70, 100, 3)).astype("float32"))

        out_a = infer(a)
        out_b = infer(b)
        assert tuple(out_a.shape) == (1, 96, 96, 3), out_a.shape
        assert tuple(out_b.shape) == (2, 70, 100, 3), out_b.shape
        assert np.all(np.isfinite(out_b.numpy()))

        # A model that baked in a constant crop would still produce the right
        # shape for whichever size it baked; assert the CONTENT agrees with the
        # eager path at both sizes.
        #
        # DECISION plan-2026-08-22T035419-a11304c8/D-031
        # ``atol=_GRAPH_VS_EAGER_ATOL`` (1e-4), NOT the 1e-5 this pair carried
        # until 2026-08-22, and NOT a seeded model. Derivation, measured:
        #
        #   * At atol=rtol=1e-5 this assertion was RED at 1 failed / 11 passed
        #     over 12 solo runs, with ``Mismatched elements: 1 / 27648`` and
        #     ``Max absolute difference among violations: 1.168251e-05``
        #     (an earlier capture read 1.049e-05 -- the number moves because the
        #     model is a fresh unseeded draw every run).
        #   * Noise distribution, sampled over 25 freshly initialized ``_tiny()``
        #     models x both sizes = 50 graph-vs-eager comparisons: max |delta|
        #     ranged 6.2e-06 .. 1.335e-05, and the worst ratio to the OLD
        #     tolerance was 1.0137 -- i.e. 1 of 50 comparisons exceeded the bar,
        #     by 1.4%. The bound was not provably attainable: it sat inside the
        #     noise it was measuring.
        #   * Output magnitudes reach |y| ~ 15, where one fp32 ulp is 1.907e-06.
        #     The worst observed delta is therefore ~7 ulp -- ordinary fp32
        #     reassociation for a network whose per-output accumulation depth is
        #     in the thousands of MACs, and whose graph and eager paths take
        #     different kernel fusion/vectorization routes.
        #   * The new bar is 1e-4 ~= 52 ulp at |y| = 16, giving 7.5x headroom
        #     over the worst of the 50 samples (worst ratio under the new bar:
        #     0.134). It stays four to five orders of magnitude below any real
        #     defect this assertion can see: a one-pixel crop shift moves
        #     elements by O(1), and a baked-in constant crop is caught by the
        #     shape assertions above plus
        #     ``test_the_crop_is_not_a_no_op_under_a_dynamic_trace``.
        #
        # Seeding was considered and rejected. It is the right fix for a
        # DISCRETE draw that decides a verdict (see D-030, swin stochastic
        # depth); here the quantity is continuous, every model in the sample
        # lands in the same 6e-06..1.3e-05 band, and pinning one weight draw
        # would hide a mis-set bound instead of correcting it -- while costing
        # the population sampling that the unseeded init gives this numerics
        # test for free.
        np.testing.assert_allclose(
            out_a.numpy(), np.asarray(model(a, training=False)),
            atol=_GRAPH_VS_EAGER_ATOL, rtol=1e-5,
        )
        np.testing.assert_allclose(
            out_b.numpy(), np.asarray(model(b, training=False)),
            atol=_GRAPH_VS_EAGER_ATOL, rtol=1e-5,
        )

    def test_the_crop_is_not_a_no_op_under_a_dynamic_trace(self) -> None:
        """Anti-vacuity: 70x100 needs a real pad (58 and 28), so a dropped
        crop would leave 128x128 and the shape assertion above would fire.
        Pin the pad arithmetic directly at the same time."""
        assert (-70) % 64 == 58
        assert (-100) % 64 == 28
        assert (-128) % 64 == 0

    def test_dynamic_and_static_builds_agree(self) -> None:
        """The static path is plain Python ints; the dynamic path is tensors.
        They must produce the same numbers on the same input."""
        model = _tiny()
        x = np.random.default_rng(2).random((1, 70, 100, 3)).astype("float32")

        static_fn = tf.function(
            lambda t: model(t, training=False),
            input_signature=[tf.TensorSpec([1, 70, 100, 3], tf.float32)],
        )
        dynamic_fn = tf.function(
            lambda t: model(t, training=False),
            input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)],
        )
        # atol 1e-5, not 1e-6: MEASURED on CPU, the two traces differ by up to
        # 3.46e-06 on 1.19% of elements. Statically-shaped kernels take
        # different fusion paths than dynamically-shaped ones; this is fp32
        # reassociation noise, not a padding difference (a one-pixel crop shift
        # moves elements by O(1)).
        # [RE-MEASURED 2026-08-22, D-031] 25 freshly initialized models: max
        # |delta| 5.722e-06 (above the 3.46e-06 recorded here), worst ratio to
        # this tolerance 0.449, ZERO violations in 25 of 25. This pair keeps
        # atol=1e-5 deliberately: it compares two GRAPH traces of the same
        # model, whose spread is roughly half the graph-vs-eager spread above,
        # and it has never been observed RED. It is not widened to
        # _GRAPH_VS_EAGER_ATOL just for symmetry -- a tolerance is a claim about
        # a measured distribution, and these are two different distributions.
        np.testing.assert_allclose(
            static_fn(x).numpy(), dynamic_fn(x).numpy(), atol=1e-5, rtol=1e-5,
        )
