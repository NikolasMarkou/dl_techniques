"""Tests for E6 — 4-layer causal-LM × Wikipedia 10k.

Pattern-3 mirror of the gpt2/pretrain trainer at 4L/d=192 scale. All tests
exercise the SYNTHETIC SMOKE path (``--synthetic-smoke`` / ``synthetic_smoke=
True``) to avoid the HF Wikipedia download cost. Real-data path is covered
by step-9 sweep smoke (subprocess level).

Pre-Mortem Scenario 1 (plan_2026-05-18_6776f8ba): each individual test must
complete in < 60s CPU; the full module must complete in < 5 min CPU. NaN in
the loss after 1 step → step-8 fails Scenario 5 (harness broken).
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

# Force CPU before any TF import is triggered transitively.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

from train.rms_variants_train.experiments import e6_clm_wiki as e6
from train.rms_variants_train.config import ExperimentConfig, NORM_VARIANTS


# ---------------------------------------------------------------------
# Module-level constants (kept tiny for CPU smoke budget)
# ---------------------------------------------------------------------

_SMOKE_KWARGS = dict(
    max_seq_len=32,
    d_model=32,
    num_heads=2,
    num_layers=2,
    intermediate_size=64,
    dropout_rate=0.0,
    warmup_ratio=0.0,
    max_train_articles=16,
    max_val_articles=4,
    steps_per_epoch_override=1,
    synthetic_smoke=True,
)


def _make_cfg(tmp_dir: str, norm_type: str = "rms_norm", mode: str = "oob") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="e6",
        norm_type=norm_type,
        seed=0,
        mode=mode,
        dataset="wikipedia_10k",
        epochs=1,
        batch_size=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_epochs=0,
        mixed_precision=False,
        max_band_width=0.1,
        out_dir=tmp_dir,
        extras={"regime": "default"},
    )


# ---------------------------------------------------------------------
# Import / structural sanity
# ---------------------------------------------------------------------


class TestE6Structure:
    def test_module_importable(self):
        assert hasattr(e6, "TinyCLM")
        assert hasattr(e6, "run")
        assert hasattr(e6, "main")
        assert hasattr(e6, "_REGIME_MAP")
        assert hasattr(e6, "WIKI_ARTICLES_DEFAULT")
        assert e6.WIKI_ARTICLES_DEFAULT == 10_000

    def test_d004_anchor_present(self):
        """D-004 anchor must be at the norm construction site."""
        path = e6.__file__
        with open(path, "r") as f:
            src = f.read()
        # Anchor uses the full plan-id-prefixed form.
        assert "DECISION plan_2026-05-18_6776f8ba/D-004" in src, (
            "D-004 anchor missing from e6_clm_wiki.py — required by plan.md step 8"
        )

    def test_regime_map_keys_and_shape(self):
        """_REGIME_MAP must be 5-tuple per plan step 6 invariant."""
        assert set(e6._REGIME_MAP.keys()) == {"default", "mp_fp16", "lr_extreme", "wd_zero"}
        for name, tup in e6._REGIME_MAP.items():
            assert len(tup) == 5, f"regime {name!r} not 5-tuple: {tup}"

    def test_tinyclm_constructs_with_each_norm(self):
        """Build-only smoke: every NORM_VARIANT instantiates without raising."""
        from train.rms_variants_train.config import build_norm_kwargs
        for norm_type in NORM_VARIANTS:
            kw = build_norm_kwargs(norm_type)
            m = e6.TinyCLM(
                vocab_size=128, max_seq_len=16,
                d_model=16, num_heads=2, num_layers=1,
                intermediate_size=32,
                norm_type=norm_type, norm_kwargs=kw,
                dropout_rate=0.0,
                name=f"build_only_{norm_type}",
            )
            x = np.zeros((1, 15), dtype=np.int32)
            out = m(x, training=False)
            assert isinstance(out, dict) and "logits" in out
            assert out["logits"].shape == (1, 15, 128), (
                f"norm={norm_type}: unexpected logits shape {out['logits'].shape}"
            )


# ---------------------------------------------------------------------
# 1-step fit smoke (CPU, synthetic data)
# ---------------------------------------------------------------------


class TestE6OneStepSmoke:
    def test_synthetic_smoke_rms_norm_non_nan(self):
        """The acceptance gate from plan.md step 8: 1-step CPU smoke produces
        a non-NaN loss with rms_norm."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, norm_type="rms_norm")
            result = e6.run(cfg, **_SMOKE_KWARGS)
        assert "final_val_loss" in result
        # Non-NaN loss is the gate. PPL may be huge after 1 step — that's fine.
        assert math.isfinite(result["final_val_loss"]), (
            f"E6 1-step smoke produced non-finite val_loss={result['final_val_loss']}"
        )
        assert result["wall_s"] < 60.0, (
            f"E6 1-step smoke wall-clock {result['wall_s']:.1f}s exceeds the "
            "60s/test budget (Pre-Mortem Scenario 1)."
        )

    def test_synthetic_smoke_zero_centered_rms_non_nan(self):
        """Coverage for the zero-centered variant (key F2 mechanism family)."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, norm_type="zero_centered_rms_norm")
            result = e6.run(cfg, **_SMOKE_KWARGS)
        assert math.isfinite(result["final_val_loss"])

    def test_results_csv_emitted(self):
        """run() must emit results.csv with the headline columns."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            _ = e6.run(cfg, **_SMOKE_KWARGS)
            csv_path = os.path.join(tmp, "results.csv")
            assert os.path.exists(csv_path)
            with open(csv_path, "r") as f:
                header = f.readline().strip().split(",")
                row = f.readline().strip().split(",")
            for col in ("experiment", "norm_type", "mode", "seed", "regime",
                        "final_loss", "final_val_loss", "final_val_perplexity",
                        "generalization_gap"):
                assert col in header, f"results.csv missing column {col}: {header}"
            assert row[header.index("experiment")] == "e6"

    def test_history_csv_emitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp)
            _ = e6.run(cfg, **_SMOKE_KWARGS)
            assert os.path.exists(os.path.join(tmp, "history.csv"))
