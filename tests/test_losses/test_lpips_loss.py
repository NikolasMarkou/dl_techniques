"""Tests for LPIPSLoss.

The first call inside any test triggers a VGG16/VGG19 ImageNet weight
download via ``keras.applications``. Tests SKIP cleanly if weights cannot
be loaded (offline CI environments).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.losses import LPIPSLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vgg_or_skip():
    """Build a tiny VGG to confirm weights are available; SKIP otherwise."""
    try:
        m = keras.applications.VGG16(include_top=False, weights="imagenet")
        del m
    except Exception as exc:  # pragma: no cover — environment-dependent.
        pytest.skip(f"VGG16 weights unavailable in this environment: {exc}")


def _rand_batch(seed: int = 0, n: int = 2, h: int = 32, w: int = 32):
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.0, 1.0, size=(n, h, w, 3)).astype("float32")
    return ops.convert_to_tensor(arr)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestLPIPSLoss:

    def test_construction_defaults(self):
        loss = LPIPSLoss()
        assert loss.vgg_variant == "vgg16"
        assert loss.distance == "l1"
        assert loss.input_range == (0.0, 1.0)
        assert isinstance(loss.layer_weights, dict)
        assert len(loss.layer_weights) == 5

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError):
            LPIPSLoss(vgg_variant="resnet50")

    def test_invalid_distance_raises(self):
        with pytest.raises(ValueError):
            LPIPSLoss(distance="cosine")

    def test_invalid_input_range_raises(self):
        with pytest.raises(ValueError):
            LPIPSLoss(input_range=(1.0, 1.0))

    def test_get_from_config_roundtrip(self):
        loss = LPIPSLoss(
            layer_weights={"block1_conv1": 0.5, "block3_conv1": 0.5},
            vgg_variant="vgg16",
            input_range=(0.0, 1.0),
            distance="l2",
            loss_weight=0.7,
        )
        config = loss.get_config()
        loss2 = LPIPSLoss.from_config(config)
        assert loss2.layer_weights == loss.layer_weights
        assert loss2.vgg_variant == loss.vgg_variant
        assert loss2.input_range == loss.input_range
        assert loss2.distance == loss.distance
        assert loss2.loss_weight == pytest.approx(loss.loss_weight)

    def test_identical_inputs_yield_near_zero(self):
        _vgg_or_skip()
        loss = LPIPSLoss()
        x = _rand_batch(seed=1)
        out = loss(x, x)
        out_np = np.array(out)
        # Channel-normalized L1 of identical features → 0 (modulo fp eps).
        assert np.all(np.isfinite(out_np))
        assert float(np.mean(out_np)) < 1e-5

    def test_different_inputs_yield_positive(self):
        _vgg_or_skip()
        loss = LPIPSLoss()
        x = _rand_batch(seed=2)
        y = _rand_batch(seed=42)
        out = loss(x, y)
        out_np = np.array(out)
        assert np.all(np.isfinite(out_np))
        assert float(np.mean(out_np)) > 0.0

    def test_invalid_layer_name_raises_on_call(self):
        _vgg_or_skip()
        loss = LPIPSLoss(layer_weights={"not_a_real_layer": 1.0})
        x = _rand_batch(seed=3)
        with pytest.raises(ValueError, match="not present"):
            _ = loss(x, x)

    def test_l2_distance_finite(self):
        _vgg_or_skip()
        loss = LPIPSLoss(distance="l2")
        x = _rand_batch(seed=4)
        y = _rand_batch(seed=5)
        out = loss(x, y)
        assert np.all(np.isfinite(np.array(out)))

    def test_input_range_rescale(self):
        """Inputs in [0, 255] with input_range=(0,255) should match the
        result of pre-scaled [0,1] inputs with input_range=(0,1)."""
        _vgg_or_skip()
        loss_unit = LPIPSLoss(input_range=(0.0, 1.0))
        loss_255 = LPIPSLoss(input_range=(0.0, 255.0))
        x_unit = _rand_batch(seed=6)
        y_unit = _rand_batch(seed=7)
        x_255 = x_unit * 255.0
        y_255 = y_unit * 255.0
        out_unit = np.array(loss_unit(x_unit, y_unit))
        out_255 = np.array(loss_255(x_255, y_255))
        np.testing.assert_allclose(out_unit, out_255, atol=1e-5)

    def test_serialize_via_keras_loss_protocol(self):
        """Round-trip through keras.losses.serialize / deserialize."""
        _vgg_or_skip()
        loss = LPIPSLoss(
            layer_weights={"block1_conv1": 0.1, "block2_conv1": 0.2},
            distance="l1",
        )
        ser = keras.losses.serialize(loss)
        de = keras.losses.deserialize(ser)
        assert isinstance(de, LPIPSLoss)
        assert de.layer_weights == loss.layer_weights
        assert de.distance == loss.distance

    def test_compile_and_fit_two_steps(self, tmp_path):
        """Tiny model + fit(2 steps) — no NaNs."""
        _vgg_or_skip()
        loss = LPIPSLoss()
        # Identity-shaped tiny model: predicts the input via a 1x1 conv,
        # close to identity but learnable.
        inp = keras.layers.Input(shape=(16, 16, 3))
        out = keras.layers.Conv2D(3, kernel_size=1, padding="same",
                                  kernel_initializer="ones",
                                  use_bias=False)(inp)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss=loss)
        x = np.random.uniform(0.0, 1.0, size=(4, 16, 16, 3)).astype("float32")
        hist = model.fit(x, x, epochs=1, batch_size=2, verbose=0, steps_per_epoch=2)
        losses = hist.history["loss"]
        assert len(losses) == 1
        assert np.isfinite(losses[0])

    def test_resolution_invariance(self):
        """Loss must accept different spatial sizes (VGG is fully convolutional)."""
        _vgg_or_skip()
        loss = LPIPSLoss()
        x_small = _rand_batch(seed=8, h=32, w=32)
        y_small = _rand_batch(seed=9, h=32, w=32)
        x_big = _rand_batch(seed=10, h=64, w=64)
        y_big = _rand_batch(seed=11, h=64, w=64)
        out_small = loss(x_small, y_small)
        out_big = loss(x_big, y_big)
        assert np.all(np.isfinite(np.array(out_small)))
        assert np.all(np.isfinite(np.array(out_big)))
