"""Regression tests for ``train.video_jepa.train_video_jepa``.

Mirrors ``tests/test_train/test_lewm/test_train_lewm.py``. Covers:

- ``_validate_args`` rejects odd ``embed_dim``, mismatched
  ``embed_dim`` vs ``num_heads * dim_head``, indivisible ``img_size``,
  and ``batch_size < 2``.
- ``--smoke`` preset overrides defaults but not user-provided flags.
- End-to-end 1-step synthetic fit + reload-from-disk round-trip
  (integration-marked).

Runtime target: <60s on CPU. Uses ultra-tiny dims (img=16, patch=8,
embed=8, heads=2, dim_head=4, depth=1, batch=2, T=2, num_proj=4).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402

from train.video_jepa.train_video_jepa import (  # noqa: E402
    _validate_args,
    _build_config,
    _SMOKE_OVERRIDES,
    parse_args,
)
from dl_techniques.models.video_jepa.model import VideoJEPA  # noqa: E402
from dl_techniques.datasets.synthetic_drone_video import (  # noqa: E402
    synthetic_drone_video_dataset,
)


def _tiny_args(**overrides) -> argparse.Namespace:
    """Build the minimal namespace _validate_args / _build_config expect."""
    base = dict(
        img_size=16,
        patch_size=8,
        img_channels=3,
        embed_dim=8,
        T=2,
        predictor_num_heads=2,
        predictor_dim_head=4,
        predictor_mlp_dim=16,
        predictor_depth=1,
        predictor_shifts=[1, 2],
        encoder_clifford_depth=1,
        encoder_shifts=[1, 2],
        sigreg_knots=5,
        sigreg_num_proj=4,
        sigreg_weight=0.09,
        dropout=0.0,
        mask_prediction_enabled=False,
        mask_ratio=0.0,
        lambda_next_frame=1.0,
        lambda_mask=1.0,
        batch_size=2,
        predict_horizons=[1],
        ema_momentum=0.996,
        ema_schedule="none",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------

def test_validate_rejects_indivisible_img_size() -> None:
    args = _tiny_args(img_size=15, patch_size=8)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        _validate_args(args)


def test_validate_rejects_odd_embed_dim() -> None:
    # embed_dim=7 is odd → sine2D PE rejects (D // 2 must equal D/2).
    # Also break the heads*dim_head invariant to keep the test isolated
    # to the embed-dim parity branch by adjusting heads/dim_head so the
    # parity check fires first.
    args = _tiny_args(embed_dim=7, predictor_num_heads=7, predictor_dim_head=1)
    with pytest.raises(ValueError, match="even"):
        _validate_args(args)


def test_validate_rejects_embed_dim_vs_heads_mismatch() -> None:
    # embed_dim=8 is even but != 4 * 4.
    args = _tiny_args(embed_dim=8, predictor_num_heads=4, predictor_dim_head=4)
    with pytest.raises(ValueError, match="num_heads"):
        _validate_args(args)


def test_validate_rejects_batch_below_two() -> None:
    args = _tiny_args(batch_size=1)
    with pytest.raises(ValueError, match="batch_size"):
        _validate_args(args)


def test_validate_accepts_consistent_config() -> None:
    args = _tiny_args()
    _validate_args(args)  # no raise


# ---------------------------------------------------------------------
# _validate_args — multi-horizon (plan_2026-05-23_0b664700/D-001)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_horizons,expected_msg",
    [
        ([0, 1], "positive"),       # zero is invalid
        ([1, 1], "unique"),          # duplicates
        ([2, 1], "sorted"),          # unsorted descending
        ([], "non-empty"),           # empty list
    ],
)
def test_validate_rejects_bad_predict_horizons(
    bad_horizons: list, expected_msg: str
) -> None:
    """SC1: --predict-horizons {0,1}, duplicates, unsorted, empty all reject."""
    # Use T=16 so the max(h)<T check doesn't fire first.
    args = _tiny_args(T=16, predict_horizons=bad_horizons)
    with pytest.raises(ValueError, match=expected_msg):
        _validate_args(args)


def test_validate_rejects_horizons_exceeding_T() -> None:
    """SC2: max(predict_horizons) must be strictly < T."""
    args = _tiny_args(T=4, predict_horizons=[1, 100])
    with pytest.raises(ValueError, match="strictly less than"):
        _validate_args(args)


# ---------------------------------------------------------------------
# _validate_args — EMA target encoder (plan_2026-05-23_15151c75/D-001)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("bad_m", [-0.01, 1.0, 1.5])
def test_validate_rejects_bad_ema_momentum(bad_m: float) -> None:
    """--ema-momentum must be in [0.0, 1.0); reject negative and >= 1.0."""
    args = _tiny_args(ema_momentum=bad_m)
    with pytest.raises(ValueError, match="ema-momentum"):
        _validate_args(args)


def test_validate_rejects_bad_ema_schedule() -> None:
    """--ema-schedule must be one of {none, cosine}."""
    args = _tiny_args(ema_schedule="linear")
    with pytest.raises(ValueError, match="ema-schedule"):
        _validate_args(args)


def test_validate_accepts_default_ema_args() -> None:
    """Default ema_momentum=0.996, ema_schedule='none' passes validation."""
    args = _tiny_args(ema_momentum=0.996, ema_schedule="none")
    _validate_args(args)  # no raise
    args = _tiny_args(ema_momentum=0.99, ema_schedule="cosine")
    _validate_args(args)  # no raise


# ---------------------------------------------------------------------
# --smoke preset
# ---------------------------------------------------------------------

def test_smoke_preset_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """--smoke alone should apply every override key."""
    monkeypatch.setattr(sys, "argv", ["prog", "--smoke"])
    args = parse_args()
    for key, expected in _SMOKE_OVERRIDES.items():
        assert getattr(args, key) == expected, (
            f"--smoke did not apply override for {key}: "
            f"got {getattr(args, key)!r}, expected {expected!r}"
        )


def test_smoke_preset_does_not_override_user_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-provided flags must win over --smoke overrides."""
    monkeypatch.setattr(sys, "argv", ["prog", "--smoke", "--batch-size", "8"])
    args = parse_args()
    assert args.batch_size == 8, "User --batch-size should beat --smoke"


# ---------------------------------------------------------------------
# Multi-horizon forward (plan_2026-05-23_0b664700/D-001)
# ---------------------------------------------------------------------

def test_multi_horizon_forward_produces_per_horizon_trackers() -> None:
    """SC3: T=24 horizons=[1,4,15] → 3 per-horizon trackers + combined."""
    keras.utils.set_random_seed(0)
    args = _tiny_args(T=24, predict_horizons=[1, 4, 15])
    cfg = _build_config(args)
    model = VideoJEPA(config=cfg)
    pixels = np.random.RandomState(0).randn(
        2, 24, args.img_size, args.img_size, args.img_channels
    ).astype("float32")
    out = model({"pixels": pixels}, training=True)
    assert out.shape == (2, 24, 2, 2, args.embed_dim)
    metric_names = [m.name for m in model.metrics]
    for h in (1, 4, 15):
        assert f"next_frame_loss_h{h}" in metric_names, (
            f"missing per-horizon tracker next_frame_loss_h{h} in "
            f"{metric_names!r}"
        )
    assert "next_frame_loss" in metric_names, (
        f"combined tracker missing: {metric_names!r}"
    )
    # At least 3 add_loss calls from per-horizon heads (+ sigreg).
    assert len(model.losses) >= 4, (
        f"expected >=4 add_loss entries (3 horizons + sigreg), "
        f"got {len(model.losses)}"
    )


# ---------------------------------------------------------------------
# End-to-end synthetic fit + reload round-trip
# ---------------------------------------------------------------------

@pytest.mark.integration
def test_end_to_end_fit_and_reload(tmp_path: Path) -> None:
    """1-step fit on synthetic data; save; reload; forward-pass round-trip."""
    keras.utils.set_random_seed(0)
    args = _tiny_args()
    cfg = _build_config(args)
    model = VideoJEPA(config=cfg)
    ds = synthetic_drone_video_dataset(
        batch_size=args.batch_size,
        num_batches=2,
        T=args.T,
        img_size=args.img_size,
        img_channels=args.img_channels,
        seed=0,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss=None,
        jit_compile=False,
    )
    model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)

    final_path = tmp_path / "tiny_video_jepa.keras"
    model.save(str(final_path))
    assert final_path.exists()

    sample = next(iter(ds))[0]
    y_orig = keras.ops.convert_to_numpy(model(sample, training=False))
    reloaded = keras.models.load_model(str(final_path))
    y_reload = keras.ops.convert_to_numpy(reloaded(sample, training=False))
    max_diff = float(np.max(np.abs(y_orig - y_reload)))
    assert max_diff < 1e-4, f"Reload round-trip diff too large: {max_diff:.2e}"


@pytest.mark.integration
def test_multi_horizon_fit_and_reload_preserves_trackers(tmp_path: Path) -> None:
    """SC4: 1-step fit + reload preserves all per-horizon trackers; finite."""
    keras.utils.set_random_seed(0)
    # Tiny dims, but T big enough for multiple horizons.
    args = _tiny_args(T=8, predict_horizons=[1, 3, 6])
    cfg = _build_config(args)
    model = VideoJEPA(config=cfg)
    ds = synthetic_drone_video_dataset(
        batch_size=args.batch_size,
        num_batches=2,
        T=args.T,
        img_size=args.img_size,
        img_channels=args.img_channels,
        seed=0,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss=None,
        jit_compile=False,
    )
    history = model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)

    # All per-horizon losses logged + finite.
    for h in (1, 3, 6):
        key = f"next_frame_loss_h{h}"
        assert key in history.history, (
            f"missing {key} in history.history keys: "
            f"{list(history.history)!r}"
        )
        val = history.history[key][-1]
        assert np.isfinite(val), f"{key} not finite: {val}"
    assert "next_frame_loss" in history.history
    assert np.isfinite(history.history["next_frame_loss"][-1])

    final_path = tmp_path / "mh_video_jepa.keras"
    model.save(str(final_path))
    reloaded = keras.models.load_model(str(final_path))
    reload_metric_names = [m.name for m in reloaded.metrics]
    for h in (1, 3, 6):
        assert f"next_frame_loss_h{h}" in reload_metric_names, (
            f"reloaded model missing tracker next_frame_loss_h{h}: "
            f"{reload_metric_names!r}"
        )

    # Forward parity after reload.
    sample = next(iter(ds))[0]
    y_orig = keras.ops.convert_to_numpy(model(sample, training=False))
    y_reload = keras.ops.convert_to_numpy(reloaded(sample, training=False))
    max_diff = float(np.max(np.abs(y_orig - y_reload)))
    assert max_diff < 1e-4, f"Multi-horizon reload diff: {max_diff:.2e}"


@pytest.mark.integration
def test_end_to_end_fit_and_reload_with_masking(tmp_path: Path) -> None:
    """DECISION plan_2026-05-24_ca745a6c/D-001: trainer-side reload-check
    must succeed when `mask_prediction_enabled=True`. This is the exact
    code path the production trainer's runtime reload-check executes — it
    used to fail with `max|delta|≈8.30` because tube-mask substitution ran
    under `training=False` with an unseeded RNG (the misdiagnosed
    'hires reload bug'). Locks in F1 root-cause fix at trainer scope."""
    keras.utils.set_random_seed(0)
    args = _tiny_args(mask_prediction_enabled=True, mask_ratio=0.6)
    cfg = _build_config(args)
    model = VideoJEPA(config=cfg)
    ds = synthetic_drone_video_dataset(
        batch_size=args.batch_size,
        num_batches=2,
        T=args.T,
        img_size=args.img_size,
        img_channels=args.img_channels,
        seed=0,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss=None,
        jit_compile=False,
    )
    model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)

    final_path = tmp_path / "tiny_video_jepa_mask.keras"
    model.save(str(final_path))
    assert final_path.exists()

    sample = next(iter(ds))[0]
    y_orig = keras.ops.convert_to_numpy(model(sample, training=False))
    reloaded = keras.models.load_model(str(final_path))
    y_reload = keras.ops.convert_to_numpy(reloaded(sample, training=False))
    max_diff = float(np.max(np.abs(y_orig - y_reload)))
    assert max_diff < 1e-4, (
        f"Reload round-trip diff too large under masking: {max_diff:.2e} "
        f"(D-001 regression — mask substitution leaked to inference?)"
    )
