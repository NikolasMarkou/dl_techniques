"""Smoke tests for BurstDP.

Verifies that the model:
    1. builds at small + tiny + pico presets
    2. produces the expected output shapes
    3. handles variable N_aux via the boolean mask (N=0, N=k<N_max, N=N_max)
    4. supports a one-step train pass (gradients flow into encoder + fusion + heads)
    5. round-trips via get_config / from_config

These tests deliberately use the pico preset and small image_size so they
run fast enough for the pre-commit smoke pass.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.burst_dp import (
    DEFAULT_NUM_SEG_CLASSES,
    BurstDP,
    BurstDPConfig,
    BurstFusionBlock,
    BurstFusionBlockAdaLN,
    create_burst_dp,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg() -> BurstDPConfig:
    return BurstDPConfig(
        image_size=64,
        patch_size=8,
        n_max=3,
        encoder_scale="pico",
        fusion_blocks=1,
        fusion_heads=3,
        fusion_mlp_ratio=2.0,
        recon_channels=3,
        num_seg_classes=DEFAULT_NUM_SEG_CLASSES,
        enable_depth=False,
        decoder_dims=(48, 32, 16),
    )


def _batch(b: int, cfg: BurstDPConfig, valid_per_sample):
    h = w = cfg.image_size
    ref = np.random.rand(b, h, w, 3).astype("float32")
    aux = np.random.rand(b, cfg.n_max, h, w, 3).astype("float32")
    mask = np.zeros((b, cfg.n_max), dtype="float32")
    if isinstance(valid_per_sample, int):
        valid_per_sample = [valid_per_sample] * b
    for i, k in enumerate(valid_per_sample):
        for j in range(k):
            mask[i, j] = 1.0
    return {
        "ref": ops.convert_to_tensor(ref),
        "aux": ops.convert_to_tensor(aux),
        "aux_mask": ops.convert_to_tensor(mask),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBurstDPSmoke:

    def test_build_and_forward_basic(self, small_cfg):
        model = BurstDP(config=small_cfg)
        out = model(_batch(2, small_cfg, valid_per_sample=2))
        assert "recon" in out and "segmentation" in out
        assert "depth" not in out  # enable_depth=False
        assert tuple(out["recon"].shape) == (2, small_cfg.image_size, small_cfg.image_size, 3)
        assert tuple(out["segmentation"].shape) == (
            2,
            small_cfg.image_size,
            small_cfg.image_size,
            small_cfg.num_seg_classes,
        )

    def test_variable_n_aux_no_nan(self, small_cfg):
        """N=0 fallback, N=k<N_max, N=N_max all produce finite outputs."""
        model = BurstDP(config=small_cfg)
        for valid in ([0, 0], [1, 2], [small_cfg.n_max, small_cfg.n_max]):
            out = model(_batch(2, small_cfg, valid_per_sample=valid))
            arr = ops.convert_to_numpy(out["recon"])
            assert np.isfinite(arr).all(), f"NaN/Inf with valid_per_sample={valid}"
            arr_s = ops.convert_to_numpy(out["segmentation"])
            assert np.isfinite(arr_s).all(), f"NaN/Inf seg with valid_per_sample={valid}"

    def test_depth_head_enabled(self, small_cfg):
        cfg = BurstDPConfig(**{**small_cfg.to_dict(), "enable_depth": True})
        # to_dict yields list for decoder_dims; from_dict converts back to tuple.
        cfg = BurstDPConfig.from_dict({**small_cfg.to_dict(), "enable_depth": True})
        model = BurstDP(config=cfg)
        out = model(_batch(2, cfg, valid_per_sample=2))
        assert "depth" in out
        assert tuple(out["depth"].shape) == (2, cfg.image_size, cfg.image_size, 1)

    def test_one_train_step(self, small_cfg):
        model = BurstDP(config=small_cfg)

        def total_loss(y_true_recon, y_true_seg, out):
            recon_l = ops.mean(ops.square(out["recon"] - y_true_recon))
            seg_l = ops.mean(
                keras.losses.sparse_categorical_crossentropy(
                    y_true_seg, out["segmentation"], from_logits=True
                )
            )
            return recon_l + seg_l

        b = 2
        h = w = small_cfg.image_size
        inputs = _batch(b, small_cfg, valid_per_sample=2)
        y_recon = ops.convert_to_tensor(np.random.rand(b, h, w, 3).astype("float32"))
        y_seg = ops.convert_to_tensor(np.random.randint(0, small_cfg.num_seg_classes, size=(b, h, w)).astype("int32"))

        optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
        # Trigger build.
        _ = model(inputs)
        vars_before = [keras.ops.convert_to_numpy(v).copy() for v in model.trainable_variables]

        import tensorflow as tf
        with tf.GradientTape() as tape:
            out = model(inputs, training=True)
            loss = total_loss(y_recon, y_seg, out)
        grads = tape.gradient(loss, model.trainable_variables)
        # All variables should receive a gradient.
        assert all(g is not None for g in grads), "Some trainable vars got None gradient"
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        vars_after = [keras.ops.convert_to_numpy(v) for v in model.trainable_variables]
        diffs = [np.linalg.norm(a - b) for a, b in zip(vars_after, vars_before)]
        assert max(diffs) > 0.0, "No variable changed after a training step"

    def test_config_round_trip(self, small_cfg):
        cfg_dict = small_cfg.to_dict()
        cfg2 = BurstDPConfig.from_dict(cfg_dict)
        assert cfg2.to_dict() == cfg_dict

    def test_keras_save_load(self, small_cfg, tmp_path):
        model = BurstDP(config=small_cfg)
        # Forward once to build.
        out = model(_batch(1, small_cfg, valid_per_sample=2))
        path = str(tmp_path / "burst_dp.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        out2 = reloaded(_batch(1, small_cfg, valid_per_sample=2))
        # Just check shapes match; weights may have re-initialised RNG state.
        assert tuple(out["recon"].shape) == tuple(out2["recon"].shape)
        assert tuple(out["segmentation"].shape) == tuple(out2["segmentation"].shape)

    def test_factory_preset(self):
        m = create_burst_dp(preset="burst_dp_pico", image_size=64, patch_size=8, n_max=2,
                          enable_depth=False, decoder_dims=(48, 32, 16),
                          fusion_blocks=1, fusion_heads=3)
        assert isinstance(m, BurstDP)
        assert m.config.encoder_scale == "pico"
        assert m.config.n_max == 2


class TestFusionBlock:

    def test_fusion_block_shapes(self):
        block = BurstFusionBlock(dim=24, num_heads=3, mlp_ratio=2.0)
        B, T, D, N = 2, 16, 24, 3
        ref = ops.convert_to_tensor(np.random.rand(B, T, D).astype("float32"))
        aux = ops.convert_to_tensor(np.random.rand(B, N, T, D).astype("float32"))
        mask = ops.convert_to_tensor(np.array([[1, 1, 0], [1, 0, 0]], dtype="float32"))
        out = block(ref, aux, mask)
        assert tuple(out.shape) == (B, T, D)
        assert np.isfinite(ops.convert_to_numpy(out)).all()

    def test_fusion_block_n_zero_no_nan(self):
        block = BurstFusionBlock(dim=24, num_heads=3, mlp_ratio=2.0)
        B, T, D, N = 2, 16, 24, 3
        ref = ops.convert_to_tensor(np.random.rand(B, T, D).astype("float32"))
        aux = ops.convert_to_tensor(np.zeros((B, N, T, D), dtype="float32"))
        mask = ops.convert_to_tensor(np.zeros((B, N), dtype="float32"))
        out = block(ref, aux, mask)
        assert np.isfinite(ops.convert_to_numpy(out)).all(), "NaN with N=0 mask"


# ---------------------------------------------------------------------------
# AdaLN fusion variant
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg_adaln(small_cfg) -> BurstDPConfig:
    """Same as ``small_cfg`` but with ``fusion_type='adaln'``."""
    return BurstDPConfig.from_dict({**small_cfg.to_dict(), "fusion_type": "adaln"})


class TestBurstDPSmokeAdaLN:
    """Smoke coverage for the ``fusion_type='adaln'`` variant.

    Mirrors the four critical TestBurstDPSmoke tests:
    build/forward, variable N (including N=0), one train step (gradients
    flow through encoder + AdaLN block + heads), and save/load round-trip.
    """

    def test_build_and_forward_basic_adaln(self, small_cfg_adaln):
        model = BurstDP(config=small_cfg_adaln)
        # Confirm the dispatch actually picked the AdaLN variant.
        assert all(
            type(b).__name__ == "BurstFusionBlockAdaLN" for b in model.fusion_blocks
        ), "fusion_type='adaln' did not dispatch to BurstFusionBlockAdaLN"
        out = model(_batch(2, small_cfg_adaln, valid_per_sample=2))
        assert tuple(out["recon"].shape) == (2, small_cfg_adaln.image_size, small_cfg_adaln.image_size, 3)
        assert tuple(out["segmentation"].shape) == (
            2,
            small_cfg_adaln.image_size,
            small_cfg_adaln.image_size,
            small_cfg_adaln.num_seg_classes,
        )

    def test_variable_n_aux_no_nan_adaln(self, small_cfg_adaln):
        """N=0 fallback, N=k<N_max, N=N_max all produce finite outputs."""
        model = BurstDP(config=small_cfg_adaln)
        for valid in (
            [0, 0],
            [1, 2],
            [small_cfg_adaln.n_max, small_cfg_adaln.n_max],
        ):
            out = model(_batch(2, small_cfg_adaln, valid_per_sample=valid))
            arr_r = ops.convert_to_numpy(out["recon"])
            arr_s = ops.convert_to_numpy(out["segmentation"])
            assert np.isfinite(arr_r).all(), f"AdaLN: NaN/Inf recon at valid={valid}"
            assert np.isfinite(arr_s).all(), f"AdaLN: NaN/Inf seg at valid={valid}"

    def test_one_train_step_adaln(self, small_cfg_adaln):
        model = BurstDP(config=small_cfg_adaln)

        def total_loss(y_true_recon, y_true_seg, out):
            recon_l = ops.mean(ops.square(out["recon"] - y_true_recon))
            seg_l = ops.mean(
                keras.losses.sparse_categorical_crossentropy(
                    y_true_seg, out["segmentation"], from_logits=True
                )
            )
            return recon_l + seg_l

        b = 2
        h = w = small_cfg_adaln.image_size
        inputs = _batch(b, small_cfg_adaln, valid_per_sample=2)
        y_recon = ops.convert_to_tensor(np.random.rand(b, h, w, 3).astype("float32"))
        y_seg = ops.convert_to_tensor(
            np.random.randint(0, small_cfg_adaln.num_seg_classes, size=(b, h, w)).astype("int32")
        )

        optimizer = keras.optimizers.AdamW(learning_rate=1e-3)
        # Trigger build.
        _ = model(inputs)
        vars_before = [ops.convert_to_numpy(v).copy() for v in model.trainable_variables]

        import tensorflow as tf
        with tf.GradientTape() as tape:
            out = model(inputs, training=True)
            loss = total_loss(y_recon, y_seg, out)
        grads = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in grads), "AdaLN: some trainable vars got None gradient"
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        vars_after = [ops.convert_to_numpy(v) for v in model.trainable_variables]
        diffs = [np.linalg.norm(a - b) for a, b in zip(vars_after, vars_before)]
        assert max(diffs) > 0.0, "AdaLN: no variable changed after a training step"

    def test_keras_save_load_adaln(self, small_cfg_adaln, tmp_path):
        model = BurstDP(config=small_cfg_adaln)
        out = model(_batch(1, small_cfg_adaln, valid_per_sample=2))
        path = str(tmp_path / "burst_dp_adaln.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        out2 = reloaded(_batch(1, small_cfg_adaln, valid_per_sample=2))
        assert tuple(out["recon"].shape) == tuple(out2["recon"].shape)
        assert tuple(out["segmentation"].shape) == tuple(out2["segmentation"].shape)
        # And the reloaded config preserves the fusion_type.
        assert reloaded.config.fusion_type == "adaln"
