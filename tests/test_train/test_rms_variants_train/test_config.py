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
    # Phase 3 extends the 7-variant Phase 2 set to 8 variants.
    assert len(NORM_VARIANTS) == 8
    assert NORM_VARIANTS[0] == "rms_norm"  # baseline must be first
    # First 4 entries preserved verbatim (RESULTS.md Phase 1 invariant).
    assert NORM_VARIANTS[:4] == (
        "rms_norm",
        "band_rms",
        "zero_centered_rms_norm",
        "zero_centered_band_rms_norm",
    )
    # Phase 2 + Phase 3 extensions: the 4 new variants must all be present.
    assert set(NORM_VARIANTS[4:]) == {
        "adaptive_band_rms",
        "band_logit_norm",
        "dynamic_tanh",
        "zero_centered_adaptive_band_rms_norm",
    }
    # Phase 3 invariant: the 8th entry MUST be at position 7 (0-indexed).
    assert NORM_VARIANTS[7] == "zero_centered_adaptive_band_rms_norm"


def test_build_kwargs_zero_centered_adaptive_band_rms_norm_default() -> None:
    # SC-2: dispatcher returns kwargs compatible with the factory entry.
    kw = build_norm_kwargs(
        "zero_centered_adaptive_band_rms_norm",
        epsilon=1e-6,
        max_band_width=0.1,
    )
    assert kw == {
        "epsilon": 1e-6,
        "max_band_width": 0.1,
        "band_regularizer": None,
    }


def test_build_kwargs_zero_centered_adaptive_band_rms_norm_with_l2() -> None:
    # Mirror the explicit-regularizer test for adaptive_band_rms.
    kw = build_norm_kwargs(
        "zero_centered_adaptive_band_rms_norm",
        band_regularizer_l2=1e-4,
    )
    assert kw["band_regularizer"] is not None
    # band_regularizer should be a keras L2 regularizer instance.
    assert hasattr(kw["band_regularizer"], "l2") or hasattr(
        kw["band_regularizer"], "get_config"
    )


def test_build_kwargs_zero_centered_adaptive_band_rms_norm_factory_roundtrip() -> None:
    # Round-trip: dispatcher kwargs MUST be accepted by the factory and
    # construct a ZeroCenteredAdaptiveBandRMS instance (Phase 3 SC-2 deep).
    from dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm import (
        ZeroCenteredAdaptiveBandRMS,
    )
    kw = build_norm_kwargs(
        "zero_centered_adaptive_band_rms_norm", epsilon=1e-6, max_band_width=0.1
    )
    layer = create_normalization_layer(
        "zero_centered_adaptive_band_rms_norm", **kw
    )
    assert isinstance(layer, ZeroCenteredAdaptiveBandRMS)


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


# ---------------------------------------------------------------------
# Phase 2 — kwargs for the 3 new variants
# ---------------------------------------------------------------------


def test_build_kwargs_adaptive_band_rms_default() -> None:
    kw = build_norm_kwargs("adaptive_band_rms")
    assert kw["max_band_width"] == 0.1
    assert kw["epsilon"] == 1e-6
    # No per-feature scale toggle on this variant.
    assert "use_scale" not in kw
    # Default L2 silently disabled (matches band_rms knob).
    assert "band_regularizer" in kw
    assert kw["band_regularizer"] is None


def test_build_kwargs_adaptive_band_rms_with_explicit_regularizer() -> None:
    kw = build_norm_kwargs("adaptive_band_rms", band_regularizer_l2=1e-4)
    assert kw["band_regularizer"] is not None
    assert isinstance(kw["band_regularizer"], keras.regularizers.L2)


def test_build_kwargs_band_logit_norm_default() -> None:
    kw = build_norm_kwargs("band_logit_norm")
    assert kw["max_band_width"] == 0.1
    assert kw["epsilon"] == 1e-6
    # band_logit_norm exposes no use_scale and no band_regularizer.
    assert "use_scale" not in kw
    assert "band_regularizer" not in kw


def test_build_kwargs_dynamic_tanh_default() -> None:
    kw = build_norm_kwargs("dynamic_tanh")
    # DyT does NOT accept epsilon — the factory strips it explicitly.
    assert "epsilon" not in kw
    assert "use_scale" not in kw
    assert kw["axis"] == -1
    assert kw["alpha_init_value"] == 0.5


@pytest.mark.parametrize(
    "variant",
    ["adaptive_band_rms", "band_logit_norm", "dynamic_tanh"],
)
def test_new_variants_construct_via_factory(variant: str) -> None:
    """Round-trip: kwargs from ``build_norm_kwargs`` build a valid layer."""
    kw = build_norm_kwargs(variant)
    layer = create_normalization_layer(variant, **kw)
    out = layer(keras.ops.zeros((2, 64)))
    assert tuple(out.shape) == (2, 64)


@pytest.mark.parametrize(
    "variant",
    ["adaptive_band_rms", "band_logit_norm", "dynamic_tanh"],
)
def test_new_variants_param_matched_mode_is_noop(variant: str) -> None:
    """
    None of the 3 new variants expose ``use_scale``. The sweep driver skips
    param_matched cells for them (Step 5). Here we assert that even if a
    caller forces ``mode="param_matched"``, the resulting kwargs do not
    include a ``use_scale`` key.
    """
    cfg = ExperimentConfig(
        experiment_name="e5",
        norm_type=variant,
        mode="param_matched",
    )
    kw = cfg.norm_kwargs()
    assert "use_scale" not in kw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
