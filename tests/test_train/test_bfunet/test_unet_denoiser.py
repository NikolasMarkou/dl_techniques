"""Trainer-side tests for the plain U-Net baseline denoiser (train_unet_denoiser.py).

Construction-only + CPU-only (no ``.fit``): covers config validation/defaults, the
``build_model`` wiring of the ConvUNeXt-parity features into ``create_bfunet_denoiser``,
the bias-free check, the re-export contract, and CLI parsing.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_unet_denoiser.py -q
"""

import keras
import numpy as np
import pytest

from train.bfunet.train_unet_denoiser import (
    TrainingConfig,
    build_model,
    verify_bias_free,
    parse_arguments,
)

PATCH = 32
CHANNELS = 1
DL_LOGGER = "dl"


def _cfg(**kw):
    base = dict(variant="tiny", patch_size=PATCH, channels=CHANNELS, depth=3,
                blocks_per_level=2, batch_size=2)
    base.update(kw)
    return TrainingConfig(**base)


# ---------------------------------------------------------------------
# TrainingConfig validation + defaults
# ---------------------------------------------------------------------

class TestTrainingConfig:
    def test_default_variant_and_prefix(self):
        cfg = TrainingConfig()
        assert cfg.variant == "base"
        assert TrainingConfig.experiment_prefix == "unet_denoiser_"

    def test_default_block_normalization_is_batchnorm(self):
        assert TrainingConfig().block_normalization == "batchnorm"

    def test_rejects_unknown_variant(self):
        with pytest.raises(ValueError, match="variant"):
            TrainingConfig(variant="gigantic")

    def test_rejects_depth_below_3(self):
        with pytest.raises(ValueError, match="depth"):
            TrainingConfig(depth=2)

    def test_rejects_invalid_block_normalization(self):
        with pytest.raises(ValueError, match="block_normalization"):
            TrainingConfig(block_normalization="rmsnorm")


# ---------------------------------------------------------------------
# build_model: shape, bias-free, feature wiring
# ---------------------------------------------------------------------

def _is_bias_free(model):
    for layer in model._flatten_layers():
        if getattr(layer, "use_bias", False):
            return False
        if isinstance(layer, keras.layers.BatchNormalization) and getattr(layer, "center", False):
            return False
        if isinstance(layer, keras.layers.LayerNormalization) and getattr(layer, "center", False):
            return False
    return True


class TestBuildModel:
    def test_baseline_single_output_shape(self):
        m = build_model(_cfg())
        assert m.output_shape == (None, PATCH, PATCH, CHANNELS)
        assert _is_bias_free(m)

    def test_features_on_builds_and_is_bias_free(self):
        m = build_model(_cfg(
            use_gabor_stem=True, gabor_filters=8, use_laplacian_pyramid=True,
            zero_pad_channels=True, downsample_pool_type="average",
            block_normalization="layernorm", dropout_rate=0.1,
        ))
        assert m.output_shape == (None, PATCH, PATCH, CHANNELS)
        assert _is_bias_free(m)

    def test_expose_bottleneck_adds_second_output(self):
        m = build_model(_cfg(expose_bottleneck=True))
        assert isinstance(m.output_shape, list) and len(m.output_shape) == 2

    def test_non_integer_filter_multiplier_rejected(self):
        with pytest.raises(ValueError, match="filter_multiplier"):
            build_model(_cfg(filter_multiplier=2.5))

    def test_round_trip_serialization(self, tmp_path):
        m = build_model(_cfg(use_gabor_stem=True, gabor_filters=8, dropout_rate=0.1))
        p = str(tmp_path / "unet.keras")
        m.save(p)
        m2 = keras.models.load_model(p)
        x = np.zeros((1, PATCH, PATCH, CHANNELS), np.float32)
        assert np.allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), atol=1e-5)


# ---------------------------------------------------------------------
# verify_bias_free + re-export contract + CLI
# ---------------------------------------------------------------------

class TestVerifyAndContract:
    def test_verify_bias_free_passes_on_bias_free_model(self, caplog):
        m = build_model(_cfg(use_gabor_stem=True, gabor_filters=8))
        with caplog.at_level("INFO", logger=DL_LOGGER):
            verify_bias_free(m)
        assert "carry an additive term" not in caplog.text

    def test_reexport_contract(self):
        import train.bfunet.train_unet_denoiser as mod
        for name in ("BFUnetTrainingConfig", "add_common_arguments", "common",
                     "reject_self_iterate_with_nonadditive", "_homogeneity_probe",
                     "create_dataset", "make_curriculum_noise_fn"):
            assert hasattr(mod, name), name

    def test_parse_arguments_importable_and_wired(self, monkeypatch):
        argv = ["prog", "--variant", "small", "--no-residual-blocks",
                "--block-normalization", "layernorm", "--kernel-size", "5"]
        monkeypatch.setattr("sys.argv", argv)
        args = parse_arguments()
        assert args.variant == "small"
        assert args.no_residual_blocks is True
        assert args.block_normalization == "layernorm"
        assert args.kernel_size == 5
