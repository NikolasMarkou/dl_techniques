"""Unit tests for ExperimentConfig + build_norm_kwargs.

Covers:
- ``build_norm_kwargs`` dispatches correctly per variant.
- ``param_matched`` mode actually drops the per-feature scale weight from
  RMSNorm / ZeroCenteredRMSNorm (the headline correctness claim for the
  parameter-matched comparison).
- BandRMS variants emit ``band_regularizer=None`` by default (silent default
  L2 disabled).
- ``ExperimentConfig`` round-trips ``to_dict`` and ``norm_kwargs``.
"""
from __future__ import annotations

import keras
import numpy as np
import pytest

from train.rms_variants_train.config import (
    NORM_VARIANTS,
    ExperimentConfig,
    build_norm_kwargs,
)
from dl_techniques.layers.norms.factory import create_normalization_layer


# ---------------------------------------------------------------------
# NORM_VARIANTS shape
# ---------------------------------------------------------------------


def test_norm_variants_tuple_shape() -> None:
    assert len(NORM_VARIANTS) == 4
    assert NORM_VARIANTS[0] == "rms_norm"  # baseline must be first
    assert set(NORM_VARIANTS) == {
        "rms_norm",
        "band_rms",
        "zero_centered_rms_norm",
        "zero_centered_band_rms_norm",
    }


# ---------------------------------------------------------------------
# build_norm_kwargs per variant
# ---------------------------------------------------------------------


def test_build_kwargs_rms_norm_default() -> None:
    kw = build_norm_kwargs("rms_norm")
    assert kw["epsilon"] == 1e-6
    assert kw["use_scale"] is True
    assert "max_band_width" not in kw  # not applicable to rms_norm


def test_build_kwargs_zero_centered_rms_norm_default() -> None:
    kw = build_norm_kwargs("zero_centered_rms_norm")
    assert kw["use_scale"] is True


def test_build_kwargs_band_rms_default() -> None:
    kw = build_norm_kwargs("band_rms")
    assert kw["max_band_width"] == 0.1
    assert "band_regularizer" in kw
    assert kw["band_regularizer"] is None  # default-L2 disabled


def test_build_kwargs_zero_centered_band_rms_norm_default() -> None:
    kw = build_norm_kwargs("zero_centered_band_rms_norm")
    assert kw["max_band_width"] == 0.1
    assert kw["band_regularizer"] is None


def test_build_kwargs_band_with_explicit_regularizer() -> None:
    kw = build_norm_kwargs("band_rms", band_regularizer_l2=1e-4)
    assert kw["band_regularizer"] is not None
    assert isinstance(kw["band_regularizer"], keras.regularizers.L2)


# ---------------------------------------------------------------------
# param_matched mode (the headline correctness claim)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["rms_norm", "zero_centered_rms_norm"])
def test_param_matched_drops_scale_for_d_param_variants(variant: str) -> None:
    """``use_scale=False`` must drop the per-feature scale weight entirely."""
    kw = build_norm_kwargs(variant, use_scale=False)
    layer = create_normalization_layer(variant, **kw)
    _ = layer(keras.ops.zeros((2, 64)))
    trainable_param_count = sum(
        int(np.prod(w.shape)) for w in layer.trainable_weights
    )
    assert trainable_param_count == 0, (
        f"{variant} with use_scale=False should have 0 trainable params, "
        f"got {trainable_param_count}"
    )


@pytest.mark.parametrize("variant", ["band_rms", "zero_centered_band_rms_norm"])
def test_band_variants_always_have_one_param(variant: str) -> None:
    """BandRMS variants have exactly 1 scalar regardless of use_scale flag."""
    for use_scale in (True, False):
        kw = build_norm_kwargs(variant, use_scale=use_scale)
        layer = create_normalization_layer(variant, **kw)
        _ = layer(keras.ops.zeros((2, 64)))
        trainable_param_count = sum(
            int(np.prod(w.shape)) for w in layer.trainable_weights
        )
        assert trainable_param_count == 1, (
            f"{variant} (use_scale={use_scale}) should have 1 trainable param, "
            f"got {trainable_param_count}"
        )


@pytest.mark.parametrize("variant", ["rms_norm", "zero_centered_rms_norm"])
def test_oob_mode_d_param_variants_have_d_params(variant: str) -> None:
    """OOB mode (``use_scale=True``) gives d-many params."""
    d = 128
    kw = build_norm_kwargs(variant, use_scale=True)
    layer = create_normalization_layer(variant, **kw)
    _ = layer(keras.ops.zeros((2, d)))
    trainable_param_count = sum(
        int(np.prod(w.shape)) for w in layer.trainable_weights
    )
    assert trainable_param_count == d


# ---------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------


def test_experiment_config_defaults_roundtrip() -> None:
    cfg = ExperimentConfig(experiment_name="e5", norm_type="rms_norm")
    d = cfg.to_dict()
    assert d["norm_type"] == "rms_norm"
    assert d["seed"] == 0
    assert d["mode"] == "oob"


def test_experiment_config_norm_kwargs_oob() -> None:
    cfg = ExperimentConfig(
        experiment_name="e5",
        norm_type="rms_norm",
        mode="oob",
        use_scale=True,
    )
    kw = cfg.norm_kwargs()
    assert kw["use_scale"] is True


def test_experiment_config_norm_kwargs_param_matched_drops_scale() -> None:
    cfg = ExperimentConfig(
        experiment_name="e5",
        norm_type="rms_norm",
        mode="param_matched",
        use_scale=True,  # mode overrides this
    )
    kw = cfg.norm_kwargs()
    assert kw["use_scale"] is False


def test_experiment_config_norm_kwargs_param_matched_doesnt_touch_band() -> None:
    cfg = ExperimentConfig(
        experiment_name="e5",
        norm_type="band_rms",
        mode="param_matched",
    )
    kw = cfg.norm_kwargs()
    # Band variants have 1 scalar always; mode doesn't affect their config.
    assert "use_scale" not in kw
    assert kw["max_band_width"] == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
