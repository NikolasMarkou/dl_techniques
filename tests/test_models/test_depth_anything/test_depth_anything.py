"""Tests for `dl_techniques.models.depth_anything`.

Covers:
- Build + forward shape with `encoder_kind='real'` (small ViT) and
  `encoder_kind='placeholder'` (legacy Conv-BN-ReLU).
- DPTDecoder default activation is `'linear'`.
- Save/load round-trip equality (the SC-6 fix from
  `plan_2026-05-10_bd098beb` — D-004 save_own_variables override).
- `train_step` labeled-only smoke (1-step CPU `model.fit`).
- `train_step` semi-supervised smoke (1-step CPU `model.fit` on
  `((x_lab, x_unlab), y_lab)`).
- StrongAugmentation forward pass (D-005 keras.random + D-005-followup
  graph-mode cutmix gate).

All tests force CPU at module import time (via `CUDA_VISIBLE_DEVICES=""`)
and use 64x64 images with `vit_s` to keep total runtime well under a minute.
"""

import os

# Force CPU before keras / tensorflow are imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tempfile
from typing import Tuple

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.models.depth_anything import (
    DepthAnything,
    DPTDecoder,
)
from dl_techniques.layers.strong_augmentation import StrongAugmentation


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_image_shape() -> Tuple[int, int, int]:
    return (64, 64, 3)


@pytest.fixture()
def real_vit_small_model(small_image_shape):
    """A small real-ViT DepthAnything (CPU-friendly). Forces a build."""
    m = DepthAnything(
        encoder_kind="real",
        encoder_type="vit_s",
        image_shape=small_image_shape,
    )
    # Force a build via a single dummy forward pass.
    _ = m(keras.ops.zeros((1,) + small_image_shape))
    return m


# ---------------------------------------------------------------------
# Build & forward
# ---------------------------------------------------------------------


class TestDepthAnything:
    """Forward / topology / serialization tests for DepthAnything."""

    def test_build_real_vit_small_64(self, small_image_shape):
        m = DepthAnything(
            encoder_kind="real",
            encoder_type="vit_s",
            image_shape=small_image_shape,
        )
        x = keras.random.normal((1,) + small_image_shape)
        y = m(x)
        assert tuple(y.shape) == (1, 64, 64, 1), y.shape

    def test_build_placeholder(self, small_image_shape):
        m = DepthAnything(
            encoder_kind="placeholder",
            encoder_type="vit_s",
            image_shape=small_image_shape,
        )
        x = keras.random.normal((1,) + small_image_shape)
        y = m(x)
        assert tuple(y.shape) == (1, 64, 64, 1), y.shape

    def test_decoder_default_linear(self):
        d = DPTDecoder(dims=[16, 8, 4])
        assert d.output_activation == "linear"

    def test_save_load_roundtrip(self, real_vit_small_model, small_image_shape):
        """SC-6 / D-004: save/load equality on a wrapped sub-Model encoder."""
        m = real_vit_small_model
        x = keras.random.normal((1,) + small_image_shape)
        y1 = np.asarray(m(x))

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "da.keras")
            m.save(path)
            m2 = keras.models.load_model(path)
            y2 = np.asarray(m2(x))

        diff = float(np.max(np.abs(y1 - y2)))
        assert diff < 1e-5, f"save/load forward diff too large: {diff}"

    def test_train_step_labeled_only_smoke(
        self, small_image_shape
    ):
        """SC-7: 1-step CPU model.fit on labeled-only (x, y) input."""
        m = DepthAnything(
            encoder_kind="real",
            encoder_type="vit_s",
            image_shape=small_image_shape,
            use_feature_alignment=False,
        )
        x = np.random.randn(2, *small_image_shape).astype("float32")
        y = np.random.randn(2, 64, 64, 1).astype("float32")
        # Build once before fit (per LESSONS / prior plan recipe).
        _ = m(x)
        m.augmentation = None
        m.compile(optimizer=keras.optimizers.AdamW(1e-4))
        history = m.fit(x, y, epochs=1, steps_per_epoch=1, verbose=0)
        loss = float(history.history["loss"][0])
        assert np.isfinite(loss), f"non-finite labeled-only loss: {loss}"

    def test_train_step_semi_supervised_smoke(self, small_image_shape):
        """SC-8: 1-step CPU model.fit on ((x_lab, x_unlab), y_lab)."""
        m = DepthAnything(
            encoder_kind="real",
            encoder_type="vit_s",
            image_shape=small_image_shape,
            use_feature_alignment=True,
            enable_semi_supervised=True,
        )
        x_lab = np.random.randn(2, *small_image_shape).astype("float32")
        x_unlab = np.random.randn(2, *small_image_shape).astype("float32")
        y_lab = np.random.randn(2, 64, 64, 1).astype("float32")
        _ = m(x_lab)  # build
        m.augmentation = None

        def gen():
            yield (x_lab, x_unlab), y_lab

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                (
                    tf.TensorSpec((2,) + small_image_shape, tf.float32),
                    tf.TensorSpec((2,) + small_image_shape, tf.float32),
                ),
                tf.TensorSpec((2, 64, 64, 1), tf.float32),
            ),
        )
        m.compile(optimizer=keras.optimizers.AdamW(1e-4))
        history = m.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)
        loss = float(history.history["loss"][0])
        assert np.isfinite(loss), f"non-finite semi-sup loss: {loss}"

    def test_get_config_roundtrip(self, real_vit_small_model):
        """SC-13: from_config rebuilds without error."""
        m = real_vit_small_model
        cfg = m.get_config()
        m2 = DepthAnything.from_config(cfg)
        assert isinstance(m2, DepthAnything)


class TestStrongAugmentation:
    """Smoke tests for D-005 fix + cutmix-gate follow-up."""

    def test_strong_augmentation_forward(self):
        layer = StrongAugmentation()
        x = keras.random.normal((2, 32, 32, 3))
        y = layer(x, training=True)
        assert tuple(y.shape) == (2, 32, 32, 3), y.shape
