"""Trainer-side tests for the ConvUNeXt denoiser audit supporting-fix set.

Covers Step 10d of plan_2026-07-01_8054f023 (the fixes that are NOT the block
norm option itself):

- ``TrainingConfig.block_normalization`` validates ('layernorm'|'batchnorm') and
  raises on an invalid value; ``build_model`` threads it into the ConvNeXt blocks'
  ``normalization_type`` (SC-adjacent wiring).
- ``verify_bias_free`` (Step 5 homogeneity probe, D-005-aware): returns None and
  emits a homogeneity WARNING on a NON-homogeneous model (GRN stem + GELU) but NOT
  on a fully-homogeneous gabor + batchnorm + LeakyReLU model (SC5).
- The dead ``parallel_reads`` field is gone (Step 8).
- The self-iterate + clip Miyasawa WARNING fires at config-validation time (Step 7).
- ``--block-normalization`` parses through the importable ``parse_arguments`` (Step 4).

All model work is tiny + CPU-only; ``training=False`` for the homogeneity probe.
Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_convunext_supporting_fixes.py -q
"""

import logging

import keras
import numpy as np
import pytest

from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
)
from train.bfunet.train_convunext_denoiser import (
    TrainingConfig,
    build_model,
    verify_bias_free,
    parse_arguments,
)

PATCH = 32
CHANNELS = 1

# The logger emitting the probe / self-iterate warnings is named "dl"
# (dl_techniques.utils.logger). caplog captures it via root propagation.
DL_LOGGER = "dl"
HOMOG_WARN_SUBSTR = "degree-1 homogeneous"
CLIP_WARN_SUBSTR = "Miyasawa residual=score identity"


# ---------------------------------------------------------------------
# TrainingConfig.block_normalization validation + build_model wiring
# ---------------------------------------------------------------------

class TestBlockNormalizationConfigWiring:
    """Step 4 / 10d: config validation + build_model threading."""

    def test_config_default_is_layernorm(self) -> None:
        assert TrainingConfig().block_normalization == "layernorm"

    def test_config_accepts_batchnorm(self) -> None:
        cfg = TrainingConfig(block_normalization="batchnorm")
        assert cfg.block_normalization == "batchnorm"

    def test_config_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="block_normalization"):
            TrainingConfig(block_normalization="rmsnorm")

    def test_build_model_threads_batchnorm_into_blocks(self) -> None:
        """Construction-only: block_normalization='batchnorm' reaches the ConvNeXt
        blocks (get_config normalization_type == 'batchnorm'). No fit."""
        cfg = TrainingConfig(
            variant="tiny",
            convnext_version="v1",
            use_gabor_stem=False,
            patch_size=PATCH,
            channels=CHANNELS,
            batch_size=2,
            self_iterate_pool_size=4,
            block_normalization="batchnorm",
        )
        model = build_model(cfg)
        blocks = [
            l for l in model._flatten_layers()
            if l.__class__.__name__ in ("ConvNextV1Block", "ConvNextV2Block")
        ]
        assert len(blocks) > 0, "no ConvNeXt blocks found in the built model"
        for blk in blocks:
            assert blk.get_config()["normalization_type"] == "batchnorm"

    def test_build_model_default_blocks_are_layernorm(self) -> None:
        cfg = TrainingConfig(
            variant="tiny",
            convnext_version="v1",
            use_gabor_stem=False,
            patch_size=PATCH,
            channels=CHANNELS,
            batch_size=2,
            self_iterate_pool_size=4,
        )
        model = build_model(cfg)
        blocks = [
            l for l in model._flatten_layers()
            if l.__class__.__name__ in ("ConvNextV1Block", "ConvNextV2Block")
        ]
        assert blocks and all(
            b.get_config()["normalization_type"] == "layernorm" for b in blocks
        )


# ---------------------------------------------------------------------
# verify_bias_free homogeneity probe (Step 5 / SC5, D-005-aware)
# ---------------------------------------------------------------------

class TestVerifyBiasFreeHomogeneityProbe:
    """SC5: log-only probe warns on a non-homogeneous model, not on a homogeneous
    one, and NEVER raises (returns None)."""

    def _non_homogeneous_model(self) -> keras.Model:
        # Standard GRN stem (degree-0, dominates) + GELU blocks -> non-homogeneous
        # even at init (the stem GRN break is NOT masked by LayerScale gamma). D-005.
        return create_convunext_denoiser(
            input_shape=(PATCH, PATCH, CHANNELS),
            depth=3,
            initial_filters=16,
            blocks_per_level=1,
            convnext_version="v1",
            filter_multiplier=2,
            use_gabor_stem=False,
            block_activation="gelu",
            final_activation="linear",
            drop_path_rate=0.0,
        )

    def _homogeneous_model(self) -> keras.Model:
        # Gabor stem (GRN-free) + LeakyReLU + batchnorm + linear final: every
        # component is homogeneous at inference at any LayerScale gamma. D-005(b).
        return create_convunext_denoiser(
            input_shape=(PATCH, PATCH, CHANNELS),
            depth=3,
            initial_filters=16,
            blocks_per_level=1,
            convnext_version="v1",
            filter_multiplier=2,
            use_gabor_stem=True,
            block_activation=keras.layers.LeakyReLU(negative_slope=0.1),
            block_normalization="batchnorm",
            final_activation="linear",
            drop_path_rate=0.0,
        )

    def test_returns_none_and_warns_on_non_homogeneous(self, caplog) -> None:
        model = self._non_homogeneous_model()
        with caplog.at_level(logging.WARNING, logger=DL_LOGGER):
            result = verify_bias_free(model)
        assert result is None  # log-only, never raises / returns
        homog_warnings = [
            r for r in caplog.records if HOMOG_WARN_SUBSTR in r.getMessage()
        ]
        assert homog_warnings, (
            "expected a homogeneity WARNING on the GRN-stem GELU model; got: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    def test_returns_none_and_no_homog_warning_on_homogeneous(self, caplog) -> None:
        model = self._homogeneous_model()
        with caplog.at_level(logging.WARNING, logger=DL_LOGGER):
            result = verify_bias_free(model)
        assert result is None
        homog_warnings = [
            r for r in caplog.records if HOMOG_WARN_SUBSTR in r.getMessage()
        ]
        assert not homog_warnings, (
            "gabor + batchnorm + LeakyReLU model should NOT trigger a homogeneity "
            f"WARNING; got: {[r.getMessage() for r in homog_warnings]}"
        )


# ---------------------------------------------------------------------
# Dead parallel_reads field removed (Step 8)
# ---------------------------------------------------------------------

class TestParallelReadsRemoved:
    def test_no_parallel_reads_attribute(self) -> None:
        assert not hasattr(TrainingConfig(), "parallel_reads"), (
            "dead parallel_reads field should have been removed (Step 8)"
        )


# ---------------------------------------------------------------------
# Self-iterate + clip Miyasawa warning (Step 7)
# ---------------------------------------------------------------------

class TestSelfIterateClipWarning:
    def test_warns_when_self_iterate_additive(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=DL_LOGGER):
            TrainingConfig(
                self_iterate=True,
                noise_type="additive",
                batch_size=4,
                self_iterate_pool_size=8,
            )
        clip_warnings = [
            r for r in caplog.records if CLIP_WARN_SUBSTR in r.getMessage()
        ]
        assert clip_warnings, (
            "expected a clip/Miyasawa WARNING when self_iterate=True + additive; "
            f"got: {[r.getMessage() for r in caplog.records]}"
        )

    def test_no_clip_warning_when_self_iterate_off(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=DL_LOGGER):
            TrainingConfig(self_iterate=False)
        clip_warnings = [
            r for r in caplog.records if CLIP_WARN_SUBSTR in r.getMessage()
        ]
        assert not clip_warnings


# ---------------------------------------------------------------------
# argparse maps --block-normalization (Step 4)
# ---------------------------------------------------------------------

class TestBlockNormalizationArgparse:
    def test_argparse_parses_batchnorm(self, monkeypatch) -> None:
        argv = [
            "train_convunext_denoiser",
            "--smoke",
            "--block-normalization", "batchnorm",
        ]
        monkeypatch.setattr("sys.argv", argv)
        args = parse_arguments()
        assert args.block_normalization == "batchnorm"

    def test_argparse_default_is_layernorm(self, monkeypatch) -> None:
        argv = ["train_convunext_denoiser", "--smoke"]
        monkeypatch.setattr("sys.argv", argv)
        args = parse_arguments()
        assert args.block_normalization == "layernorm"


if __name__ == "__main__":
    pytest.main([__file__])
