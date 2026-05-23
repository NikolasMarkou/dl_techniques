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
