"""Trainer-side tests for the bias-free flat-CNN baseline denoiser (train_bfcnn_denoiser.py).

Construction-only + CPU-only (no ``.fit``): covers config validation/defaults, the
``build_model`` dispatch into ``create_bfcnn_variant`` / ``create_bfcnn_denoiser``, the
bias-free check, the re-export contract, and CLI parsing. Mirrors
``test_unet_denoiser.py``.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_bfcnn_denoiser.py -q
"""

import keras
import numpy as np
import pytest

from train.bfunet.train_bfcnn_denoiser import (
    BFCNNTrainingConfig,
    build_model,
    verify_bias_free,
    parse_arguments,
)

PATCH = 32
CHANNELS = 1
DL_LOGGER = "dl"


def _cfg(**kw):
    base = dict(variant="tiny", patch_size=PATCH, channels=CHANNELS, batch_size=2)
    base.update(kw)
    return BFCNNTrainingConfig(**base)


# ---------------------------------------------------------------------
# BFCNNTrainingConfig validation + defaults
# ---------------------------------------------------------------------

class TestTrainingConfig:
    def test_default_variant_and_prefix(self):
        cfg = BFCNNTrainingConfig()
        assert cfg.variant == "tiny"
        assert BFCNNTrainingConfig.experiment_prefix == "bfcnn_denoiser_"

    def test_custom_variant_accepted(self):
        cfg = BFCNNTrainingConfig(variant="custom")
        assert cfg.variant == "custom"

    def test_rejects_unknown_variant(self):
        with pytest.raises(ValueError, match="variant"):
            BFCNNTrainingConfig(variant="bogus")


# ---------------------------------------------------------------------
# build_model: shape, bias-free
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
    def test_named_variant_output_shape(self):
        m = build_model(_cfg(variant="tiny"))
        assert m.output_shape == (None, PATCH, PATCH, CHANNELS)
        assert _is_bias_free(m)

    def test_custom_variant_output_shape(self):
        m = build_model(_cfg(variant="custom", num_blocks=2, filters=8,
                             initial_kernel_size=5, kernel_size=3))
        assert m.output_shape == (None, PATCH, PATCH, CHANNELS)
        assert _is_bias_free(m)

    def test_round_trip_serialization(self, tmp_path):
        m = build_model(_cfg(variant="custom", num_blocks=2, filters=8))
        p = str(tmp_path / "bfcnn.keras")
        m.save(p)
        m2 = keras.models.load_model(p)
        x = np.zeros((1, PATCH, PATCH, CHANNELS), np.float32)
        assert np.allclose(
            m(x, training=False), m2(x, training=False), atol=1e-5
        )


# ---------------------------------------------------------------------
# verify_bias_free + re-export contract + CLI
# ---------------------------------------------------------------------

class TestVerifyAndContract:
    def test_verify_bias_free_passes_on_bias_free_model(self, caplog):
        m = build_model(_cfg(variant="custom", num_blocks=2, filters=8))
        with caplog.at_level("INFO", logger=DL_LOGGER):
            verify_bias_free(m)
        assert "carry an additive term" not in caplog.text

    def test_reexport_contract(self):
        import train.bfunet.train_bfcnn_denoiser as mod
        for name in ("BFUnetTrainingConfig", "add_common_arguments", "common",
                     "reject_self_iterate_with_nonadditive", "_homogeneity_probe",
                     "create_dataset", "make_curriculum_noise_fn"):
            assert hasattr(mod, name), name

    def test_parse_arguments_importable_and_wired(self, monkeypatch):
        argv = ["prog", "--variant", "custom", "--num-blocks", "6",
                "--filters", "48", "--kernel-size", "5"]
        monkeypatch.setattr("sys.argv", argv)
        args = parse_arguments()
        assert args.variant == "custom"
        assert args.num_blocks == 6
        assert args.filters == 48
        assert args.kernel_size == 5
