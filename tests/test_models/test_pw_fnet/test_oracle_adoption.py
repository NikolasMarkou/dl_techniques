"""
Oracle adoption for ``models/pw_fnet`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (GPU 1), ``PW_FNet(img_channels=3, width=8,
middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])`` on a
``(2, 32, 32, 3)`` batch after one real Adam step: **100** trainable weights,
**0** dead, **0** disconnected.

This package is GREEN and was repaired, so a failure here is a REGRESSION
---------------------------------------------------------------------------
``pw_fnet``'s fp16 kernel gap was repaired at iteration-2 step 18 and its
ULP-void scale guard at the same pass. If any assertion in this file goes red,
the first hypothesis is a regression in this package, not a flaw in the
instrument -- say so loudly rather than widening a tolerance.

The forward contract is the interesting one here. ``call`` returns a
**3-element list** of restored images at full / half / quarter resolution
(a deliberate multi-scale supervision contract, pinned in the model's own source
at the ``return [out_l0, out_l1, out_l2]`` comment block), and every level's
spatial size is asserted. A guard that checked only ``out[0]`` would accept a
model whose two coarse heads emitted anything at all.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.pw_fnet.model import PW_FNet

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

SIDE = 32
CHANNELS = 3

#: Measured 2026-08-21 at ``width=8, enc/dec=[1, 1], middle=1``.
GF_N_WEIGHTS = 100

#: The model's fixed multi-scale output contract: three scales, in this order.
OUTPUT_SCALES = 3


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random(
        (batch, SIDE, SIDE, CHANNELS)).astype("float32")


def _model(**overrides) -> PW_FNet:
    kwargs = dict(
        img_channels=CHANNELS, width=8, middle_blk_num=1,
        enc_blk_nums=[1, 1], dec_blk_nums=[1, 1],
    )
    kwargs.update(overrides)
    model = PW_FNet(**kwargs)
    model(_images(1), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestPWFNetGradientFlow:

    def test_no_layer_is_stochastic(self):
        model = _model()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], (
            f"a non-zero stochastic rate is live: {stochastic}"
        )

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        """A regression here is a pw_fnet regression -- see the docstring."""
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        # The U-shape, named part by part: without this the count could be
        # met by an encoder-only model with three output heads bolted on.
        for fragment in ("enc_l1_blk_", "enc_l2_blk_", "middle_blk_",
                         "dec_l1_blk_", "dec_l2_blk_",
                         "output_l0", "output_l1", "output_l2"):
            assert any(fragment in path for path in report), (
                f"no weight whose path contains {fragment!r}; the U-shape the "
                f"count above rests on is not all in the trainable set"
            )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())


class TestPWFNetKnobSensitivity:

    def test_width_changes_the_parameterisation(self):
        builders = {w: (lambda w=w: _model(width=w)) for w in (8, 16, 32)}
        assert_structural_knob_changes_weights(builders, knob="width")

    def test_middle_blk_num_changes_the_bottleneck(self):
        """A knob that reaches ONLY the bottleneck.

        ``width`` above would still pass with no bottleneck at all. This one
        would not.
        """
        builders = {n: (lambda n=n: _model(middle_blk_num=n)) for n in (1, 2, 3)}
        assert_structural_knob_changes_weights(builders, knob="middle_blk_num")

    def test_enc_blk_nums_changes_the_parameterisation(self):
        builders = {
            tuple(e): (lambda e=e: _model(enc_blk_nums=list(e),
                                          dec_blk_nums=list(e)))
            for e in ([1, 1], [2, 1], [2, 2])
        }
        assert_structural_knob_changes_weights(builders, knob="enc_blk_nums")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="width")


class TestPWFNetSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()
        batch = x.shape[0]

        def contract(out):
            assert isinstance(out, list), (
                f"PW_FNet returns a 3-element list of scales, got {type(out)}"
            )
            assert len(out) == OUTPUT_SCALES, (
                f"expected {OUTPUT_SCALES} scales, got {len(out)}"
            )
            for level, tensor in enumerate(out):
                side = SIDE // (2 ** level)
                assert tuple(tensor.shape) == (batch, side, side, CHANNELS), (
                    f"scale {level}: expected "
                    f"{(batch, side, side, CHANNELS)}, got {tuple(tensor.shape)}"
                )
                assert_finite(tensor)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
