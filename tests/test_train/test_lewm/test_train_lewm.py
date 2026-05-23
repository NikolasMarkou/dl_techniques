"""Regression tests for ``train.lewm.train_lewm``.

Covers:
- ``_build_model`` argument validation (img_size % patch_size,
  embed_dim vs ViT.SCALE_CONFIGS[scale]).
- End-to-end 1-epoch synthetic fit + reload-from-disk round-trip.
- Rollout S!=1 guard from DECISION plan_2026-05-23_692fd5e5/D-001.

Runtime target: <60s on CPU. Uses ultra-tiny dims (img=28, patch=14,
depth=1, num_proj=4, batch=1, steps=1).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from train.lewm.train_lewm import _build_model  # noqa: E402
from dl_techniques.models.lewm.model import LeWM  # noqa: E402
from dl_techniques.models.lewm.config import LeWMConfig  # noqa: E402
from dl_techniques.datasets.pusht_hdf5 import synthetic_lewm_dataset  # noqa: E402


def _tiny_args(**overrides) -> argparse.Namespace:
    """Build the minimal namespace _build_model expects."""
    base = dict(
        img_size=28,
        patch_size=14,
        encoder_scale="tiny",
        embed_dim=192,         # tiny -> 192
        history_size=2,
        num_preds=1,
        depth=1,
        heads=2,
        dim_head=16,
        mlp_dim=32,
        dropout=0.0,
        action_dim=2,
        smoothed_dim=4,
        mlp_scale=2,
        sigreg_weight=0.09,
        sigreg_knots=5,
        sigreg_num_proj=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------

def test_build_model_rejects_indivisible_img_size() -> None:
    args = _tiny_args(img_size=27, patch_size=14)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        _build_model(args)


def test_build_model_rejects_unknown_encoder_scale() -> None:
    args = _tiny_args(encoder_scale="ginormous")
    with pytest.raises(ValueError, match="encoder_scale"):
        _build_model(args)


def test_build_model_rejects_mismatched_embed_dim() -> None:
    args = _tiny_args(encoder_scale="tiny", embed_dim=256)
    with pytest.raises(ValueError, match="embed_dim"):
        _build_model(args)


def test_build_model_accepts_consistent_config() -> None:
    args = _tiny_args()
    model = _build_model(args)
    assert isinstance(model, LeWM)
    assert model.config.embed_dim == 192


# ---------------------------------------------------------------------
# Rollout S!=1 guard (DECISION D-001)
# ---------------------------------------------------------------------

def test_rollout_rejects_s_greater_than_one() -> None:
    cfg = LeWMConfig(
        img_size=28, patch_size=14, encoder_scale="tiny", embed_dim=192,
        history_size=2, num_preds=1, depth=1, heads=2, dim_head=16,
        mlp_dim=32, sigreg_knots=5, sigreg_num_proj=4,
    )
    model = LeWM(config=cfg)
    HS = cfg.history_size
    H = W = cfg.img_size
    pixels_history = np.zeros((1, 2, HS, H, W, 3), dtype=np.float32)  # S=2
    action_sequence = np.zeros((1, 2, HS + 1, cfg.action_dim), dtype=np.float32)
    with pytest.raises(ValueError, match="S must equal 1"):
        model.rollout(pixels_history, action_sequence)


# ---------------------------------------------------------------------
# End-to-end synthetic fit + reload round-trip
# ---------------------------------------------------------------------

@pytest.mark.integration
def test_end_to_end_fit_and_reload(tmp_path: Path) -> None:
    """1-epoch fit on synthetic data; save; reload; forward-pass round-trip."""
    keras.utils.set_random_seed(0)
    args = _tiny_args()
    model = _build_model(args)
    ds = synthetic_lewm_dataset(
        num_episodes=4,
        img_size=args.img_size,
        action_dim=args.action_dim,
        batch_size=1,
        history_size=args.history_size,
        num_preds=args.num_preds,
        seed=0,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
        loss=None,
        jit_compile=False,
    )
    model.fit(ds, epochs=1, steps_per_epoch=1, verbose=0)

    final_path = tmp_path / "tiny_lewm.keras"
    model.save(str(final_path))
    assert final_path.exists()

    sample = next(iter(ds))[0]
    y_orig = keras.ops.convert_to_numpy(model(sample, training=False))
    reloaded = keras.models.load_model(str(final_path))
    y_reload = keras.ops.convert_to_numpy(reloaded(sample, training=False))
    max_diff = float(np.max(np.abs(y_orig - y_reload)))
    assert max_diff < 1e-4, f"Reload round-trip diff too large: {max_diff:.2e}"
