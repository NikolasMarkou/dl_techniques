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
    TeacherEMACallback,
    cosine_ema_schedule,
    linear_ema_schedule,
)
from dl_techniques.models.depth_anything import create_depth_anything
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

    @pytest.mark.parametrize("channels", [1, 4])
    def test_strong_augmentation_dynamic_channels(self, channels):
        """Issue #9: cutmix mask must adapt to channels != 3."""
        layer = StrongAugmentation(cutmix_prob=1.0)
        x = keras.random.uniform((2, 16, 16, channels), 0.0, 1.0)
        y = layer(x, training=True)
        assert tuple(y.shape) == (2, 16, 16, channels), y.shape
        assert bool(np.all(np.isfinite(np.asarray(y))))

    def test_strong_aug_per_sample_factors(self):
        """Issue #9: brightness/contrast factors should vary per sample.

        We disable cutmix (cutmix_prob=0) so the only source of cross-image
        variance is per-sample color jitter. Two identical input images
        through the same forward pass must come out *different*.
        """
        layer = StrongAugmentation(cutmix_prob=0.0, color_jitter_strength=0.5)
        # 2 identical images → if factors are per-batch the outputs are equal.
        base = keras.random.uniform((1, 16, 16, 3), 0.0, 1.0)
        x = keras.ops.tile(base, [2, 1, 1, 1])
        y = np.asarray(layer(x, training=True))
        diff = float(np.max(np.abs(y[0] - y[1])))
        assert diff > 1e-5, (
            f"per-sample brightness/contrast factors not active "
            f"(max-abs sample-vs-sample diff = {diff})"
        )


# ---------------------------------------------------------------------
# EMA schedules + TeacherEMACallback
# ---------------------------------------------------------------------


class TestTeacherEMA:
    """Schedules + callback step-driver."""

    def test_cosine_schedule_endpoints_and_monotonic(self):
        sched = cosine_ema_schedule(0.5, 0.99, 100)
        v0 = sched(0)
        v100 = sched(100)
        v50 = sched(50)
        assert abs(v0 - 0.5) < 1e-6, v0
        assert abs(v100 - 0.99) < 1e-6, v100
        assert 0.5 < v50 < 0.99, v50
        # Past the horizon, clamps at end.
        assert abs(sched(1000) - 0.99) < 1e-6

    def test_linear_schedule_endpoints(self):
        sched = linear_ema_schedule(0.5, 0.99, 100)
        assert abs(sched(0) - 0.5) < 1e-6
        assert abs(sched(100) - 0.99) < 1e-6
        # Mid-point exactly halfway between start and end.
        assert abs(sched(50) - 0.745) < 1e-6, sched(50)

    def test_teacher_ema_callback_step_advances_teacher(self, small_image_shape):
        """Callback drives `update_teacher_ema` -> teacher weights move."""
        m = DepthAnything(
            encoder_kind="placeholder",
            encoder_type="vit_s",
            image_shape=small_image_shape,
            use_feature_alignment=True,
        )
        _ = m(keras.ops.zeros((1,) + small_image_shape))
        m.augmentation = None
        # Force a divergence between student and teacher so EMA must move teacher.
        sw = m.encoder.get_weights()
        m.encoder.set_weights([w + 0.01 for w in sw])

        teacher_before = [w.copy() for w in m.frozen_encoder.get_weights()]

        cb = TeacherEMACallback(
            schedule=cosine_ema_schedule(0.5, 0.5, 10),  # constant 0.5
            warmup_steps=0,
        )
        cb.set_model(m)
        # Drive 3 batches manually.
        for i in range(3):
            cb.on_train_batch_end(batch=i)

        teacher_after = m.frozen_encoder.get_weights()
        total_diff = sum(
            float(np.sum(np.abs(a - b)))
            for a, b in zip(teacher_before, teacher_after)
        )
        assert total_diff > 0.0, (
            f"teacher weights did not move under EMA callback (diff={total_diff})"
        )
        assert cb.step == 3


# ---------------------------------------------------------------------
# from_pretrained_encoder
# ---------------------------------------------------------------------


class TestPretrainedEncoder:
    def test_pretrained_encoder_load_round_trip(self, small_image_shape):
        """Saving encoder + loading via from_pretrained_encoder reproduces forward."""
        m1 = create_depth_anything(
            encoder_type="vit_s",
            image_shape=small_image_shape,
            encoder_kind="placeholder",
            use_feature_alignment=True,
        )
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "enc.keras")
            m1.encoder.save(p)
            m2 = create_depth_anything(
                encoder_type="vit_s",
                image_shape=small_image_shape,
                encoder_kind="placeholder",
                use_feature_alignment=True,
            )
            x = keras.random.normal((1,) + small_image_shape, seed=0)
            e1 = np.asarray(m1.encoder(x, training=False))
            m2.from_pretrained_encoder(p)
            e2 = np.asarray(m2.encoder(x, training=False))
        diff = float(np.max(np.abs(e1 - e2)))
        assert diff < 1e-5, f"encoder forward diff too large: {diff}"
        # Teacher must be re-synced: student-vs-teacher delta == 0.
        sw = m2.encoder.get_weights()
        tw = m2.frozen_encoder.get_weights()
        max_st = max(float(np.max(np.abs(s - t))) for s, t in zip(sw, tw))
        assert max_st == 0.0, f"teacher not re-synced (max student-teacher diff={max_st})"


# ---------------------------------------------------------------------
# Pseudo-label depth helper
# ---------------------------------------------------------------------


class TestPseudoLabelDepth:
    def test_pseudo_label_shape_and_no_grad(self, small_image_shape):
        m = DepthAnything(
            encoder_kind="placeholder",
            encoder_type="vit_s",
            image_shape=small_image_shape,
            use_feature_alignment=True,
        )
        _ = m(keras.ops.zeros((1,) + small_image_shape))
        x_unlab = keras.random.normal((2,) + small_image_shape)
        # Variables present prior to test.
        with tf.GradientTape() as tape:
            pseudo = m._pseudo_label_depth(x_unlab)
        # Gradients of pseudo wrt student trainable vars should all be None
        # (stop_gradient + frozen_encoder is non-trainable).
        grads = tape.gradient(pseudo, m.encoder.trainable_variables)
        assert all(g is None for g in grads), "pseudo-label leaked gradients into student"
        assert tuple(pseudo.shape) == (2, 64, 64, 1), pseudo.shape


# ---------------------------------------------------------------------
# Multi-epoch FAL stability
# ---------------------------------------------------------------------


class TestMultiEpochFALStability:
    """SC-11: ≥3 epochs of semi-sup synthetic training; loss finite + non-explosive."""

    def test_multi_epoch_semi_supervised_stable(self, small_image_shape):
        m = DepthAnything(
            encoder_kind="placeholder",  # placeholder for CPU speed
            encoder_type="vit_s",
            image_shape=small_image_shape,
            use_feature_alignment=True,
            enable_semi_supervised=True,
        )
        _ = m(keras.ops.zeros((1,) + small_image_shape))
        m.augmentation = None
        m.compile(optimizer=keras.optimizers.AdamW(1e-4))

        # Snapshot teacher weights for movement check.
        teacher_before = [w.copy() for w in m.frozen_encoder.get_weights()]

        x_lab = np.random.randn(2, *small_image_shape).astype("float32")
        x_unlab = np.random.randn(2, *small_image_shape).astype("float32")
        y_lab = np.random.randn(2, 64, 64, 1).astype("float32")

        def gen():
            # Repeat indefinitely; steps_per_epoch caps consumption.
            while True:
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

        # EMA callback engaged so teacher actually moves.
        cb = TeacherEMACallback(
            schedule=cosine_ema_schedule(0.5, 0.99, 6),
            warmup_steps=0,
        )

        history = m.fit(
            ds,
            epochs=3,
            steps_per_epoch=2,
            callbacks=[cb],
            verbose=0,
        )
        losses = [float(v) for v in history.history["loss"]]
        assert all(np.isfinite(losses)), f"non-finite loss in history: {losses}"
        # Non-explosive: last <= 1.5 * first (loose tolerance)
        assert losses[-1] <= 1.5 * losses[0], (
            f"loss exploded: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

        teacher_after = m.frozen_encoder.get_weights()
        total = sum(
            float(np.sum(np.abs(a - b)))
            for a, b in zip(teacher_before, teacher_after)
        )
        assert total > 0.0, (
            "teacher weights did not move during multi-epoch semi-sup training"
        )


# ---------------------------------------------------------------------
# Dataset pairing
# ---------------------------------------------------------------------


class TestDatasetPairing:
    def test_pair_labeled_unlabeled_yields_paired_batches(self, tmp_path):
        from train.common.megadepth import (
            UnlabeledImageDataset,
            pair_labeled_unlabeled,
        )
        from PIL import Image

        # 4 fake RGB images on disk.
        paths = []
        for i in range(4):
            p = tmp_path / f"im_{i}.jpg"
            Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            ).save(p)
            paths.append(str(p))

        unlab = UnlabeledImageDataset(
            paths, batch_size=2, patch_size=16, is_training=False, workers=0,
        )

        class FakeLabeled(keras.utils.PyDataset):
            def __init__(self):
                super().__init__(workers=0)
            def __len__(self): return 3
            def __getitem__(self, idx):
                return (
                    np.random.randn(2, 16, 16, 3).astype("float32"),
                    np.random.randn(2, 16, 16, 2).astype("float32"),
                )

        paired = pair_labeled_unlabeled(
            FakeLabeled(), unlab, patch_size=16, batch_size=2,
        )
        n = 0
        for (x_lab, x_unlab), y_lab in paired.take(3):
            n += 1
            assert tuple(x_lab.shape) == (2, 16, 16, 3)
            assert tuple(x_unlab.shape) == (2, 16, 16, 3)
            assert tuple(y_lab.shape) == (2, 16, 16, 2)
        assert n == 3
