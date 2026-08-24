"""Checkpoint-selection DIRECTION for ``train.common.create_callbacks``.

F-10. Until this module landed, ``create_callbacks`` derived the selection mode
as ``'max' if 'accuracy' in monitor else 'min'``. Every maximize metric whose
name lacks the substring ``accuracy`` -- ``val_psnr``, ``val_iou``,
``val_box_iou``, ``val_map``, ``val_f1``, ``val_dice``, ``val_auc``,
``val_ssim``, ``val_precision``, ``val_recall`` -- therefore got ``mode='min'``,
so ``EarlyStopping(restore_best_weights=True)`` RESTORED and ``ModelCheckpoint``
SAVED the **worst** epoch, silently.

Three arms, and the third is what makes the other two readable:

1. maximize metrics without ``accuracy`` must resolve to ``max``;
2. minimize metrics must still resolve to ``min``;
3. **anti-vacuity** -- the metrics the OLD substring heuristic already got right
   (``val_accuracy`` -> max, ``val_loss`` -> min) must be UNCHANGED. That arm
   passes both before and after the fix on purpose: if it ever goes red, the
   registry is changing more than the defect.

The assertions are on the callbacks ``create_callbacks`` actually returns, not
on ``resolve_monitor_mode`` alone, so a fix that computes the right direction
and fails to wire it into the callbacks does not pass. ``ModelCheckpoint`` keeps
no ``mode`` attribute, so it is pinned on ``monitor_op`` (``np.greater`` /
``np.less``) -- the thing that actually decides which epoch is written.
"""

import numpy as np
import keras
import pytest

from train.common.callbacks import (
    create_callbacks,
    resolve_monitor_mode,
    _MAXIMIZE_METRIC_TOKENS,
    _MINIMIZE_METRIC_TOKENS,
)


# Metrics the repo really emits (or really could): `val_psnr` is
# `src/train/darkir/train_darkir.py`'s monitor, `val_box_iou` is
# `src/train/sam3/train_sam3.py`'s SELECTION_METRIC.
MAXIMIZE_MONITORS = [
    "val_psnr",
    "val_ssim",
    "val_iou",
    "val_box_iou",
    "val_map",
    "val_f1",
    "val_dice",
    "val_auc",
    "val_precision",
    "val_recall",
    # darkir's `--side-loss` branch builds the key from the model's first
    # OUTPUT LAYER name: `f"val_{model.output_names[0]}_psnr"`, and that layer
    # is `layers.Add(name="final_residual")`.
    "val_final_residual_psnr",
]

MINIMIZE_MONITORS = [
    "val_loss",
    "loss",
    "val_total_loss",
    "val_reconstruction_loss",
    "val_mae",
    "val_absrel",
    # A composite name: minimize tokens are tested FIRST, so a "dice loss" is
    # a loss, not a Dice coefficient.
    "val_dice_loss",
]

# The old heuristic's own correct answers. This list must not change meaning.
OLD_HEURISTIC_WAS_RIGHT = [
    ("val_accuracy", "max"),
    ("val_capsule_accuracy", "max"),
    ("val_top5_accuracy", "max"),
    ("val_loss", "min"),
    ("loss", "min"),
    ("val_total_loss", "min"),
]


def _selection_callbacks(monitor, **kwargs):
    """Return the (EarlyStopping, ModelCheckpoint) pair for `monitor`."""
    callbacks, _results_dir = create_callbacks(
        model_name="direction_probe",
        results_dir_prefix="test_common_callbacks",
        monitor=monitor,
        include_analyzer=False,
        **kwargs,
    )
    early = [c for c in callbacks if isinstance(c, keras.callbacks.EarlyStopping)]
    ckpt = [c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint)]
    assert len(early) == 1 and len(ckpt) == 1
    return early[0], ckpt[0]


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    """`create_callbacks` makedirs its run dir -- keep it out of `results/`."""
    monkeypatch.chdir(tmp_path)


class TestMaximizeMetricsSelectTheBestEpoch:
    """A maximize metric whose name lacks 'accuracy' must select on `max`."""

    @pytest.mark.parametrize("monitor", MAXIMIZE_MONITORS)
    def test_early_stopping_mode_is_max(self, monitor):
        early, _ckpt = _selection_callbacks(monitor)
        assert early.mode == "max", (
            f"ASSERT-EARLYSTOPPING-MODE-MAX: monitor={monitor!r} is a metric to "
            f"MAXIMIZE but EarlyStopping got mode={early.mode!r}. With "
            f"restore_best_weights=True that RESTORES the worst epoch."
        )

    @pytest.mark.parametrize("monitor", MAXIMIZE_MONITORS)
    def test_model_checkpoint_keeps_the_larger_value(self, monitor):
        _early, ckpt = _selection_callbacks(monitor)
        assert ckpt.monitor_op is np.greater, (
            f"ASSERT-CHECKPOINT-OP-GREATER: monitor={monitor!r} is a metric to "
            f"MAXIMIZE but ModelCheckpoint compares with {ckpt.monitor_op!r}, "
            f"so best_model.keras holds the WORST epoch."
        )
        assert ckpt.best == -np.inf

    @pytest.mark.parametrize("monitor", MAXIMIZE_MONITORS)
    def test_a_higher_value_is_an_improvement(self, monitor):
        """Behavioural arm: the callback must PREFER the larger number."""
        early, _ckpt = _selection_callbacks(monitor)
        early._set_monitor_op()
        assert early._is_improvement(0.9, 0.5), (
            f"ASSERT-HIGHER-IS-BETTER: for {monitor!r}, 0.9 must improve on 0.5."
        )
        assert not early._is_improvement(0.5, 0.9)


class TestMinimizeMetricsAreUnaffected:
    """A minimize metric must still select on `min`."""

    @pytest.mark.parametrize("monitor", MINIMIZE_MONITORS)
    def test_early_stopping_mode_is_min(self, monitor):
        early, _ckpt = _selection_callbacks(monitor)
        assert early.mode == "min", (
            f"ASSERT-EARLYSTOPPING-MODE-MIN: monitor={monitor!r} is a metric to "
            f"MINIMIZE but EarlyStopping got mode={early.mode!r}."
        )

    @pytest.mark.parametrize("monitor", MINIMIZE_MONITORS)
    def test_model_checkpoint_keeps_the_smaller_value(self, monitor):
        _early, ckpt = _selection_callbacks(monitor)
        assert ckpt.monitor_op is np.less, (
            f"ASSERT-CHECKPOINT-OP-LESS: monitor={monitor!r} is a metric to "
            f"MINIMIZE but ModelCheckpoint compares with {ckpt.monitor_op!r}."
        )
        assert ckpt.best == np.inf


class TestAntiVacuity:
    """The old heuristic's CORRECT answers must be preserved exactly.

    This class is expected to pass both before and against the fix. It is the
    control that proves the registry is not moving directions it should leave
    alone -- a green run here plus a red run in the two classes above is what
    localizes the defect.
    """

    @pytest.mark.parametrize("monitor,expected", OLD_HEURISTIC_WAS_RIGHT)
    def test_direction_is_unchanged(self, monitor, expected):
        old = "max" if "accuracy" in monitor else "min"
        assert old == expected, "test data error: this list is the OLD answers"

        early, ckpt = _selection_callbacks(monitor)
        assert early.mode == expected, (
            f"ASSERT-UNCHANGED-DIRECTION: {monitor!r} was already selected "
            f"correctly as {expected!r}; the fix changed it to {early.mode!r}."
        )
        expected_op = np.greater if expected == "max" else np.less
        assert ckpt.monitor_op is expected_op


class TestExplicitOverrideAndFallback:
    """`monitor_mode=` wins; an unknown name warns instead of guessing."""

    def test_explicit_max_overrides_a_minimize_name(self):
        early, ckpt = _selection_callbacks("val_loss", monitor_mode="max")
        assert early.mode == "max"
        assert ckpt.monitor_op is np.greater

    def test_explicit_min_overrides_a_maximize_name(self):
        early, ckpt = _selection_callbacks("val_psnr", monitor_mode="min")
        assert early.mode == "min"
        assert ckpt.monitor_op is np.less

    def test_a_typo_in_the_override_raises(self):
        """Keras would only WARN and silently fall back to 'auto'."""
        with pytest.raises(ValueError, match="monitor_mode"):
            resolve_monitor_mode("val_psnr", mode="maximize")

    def test_an_unrecognized_metric_warns_and_falls_back_to_min(self, caplog):
        with caplog.at_level("WARNING"):
            mode = resolve_monitor_mode("val_some_bespoke_thing")
        assert mode == "min", (
            "The fallback must reproduce the old heuristic's answer, so an "
            "unrecognized name is never silently flipped."
        )
        assert "val_some_bespoke_thing" in caplog.text
        assert "monitor_mode='max'" in caplog.text


class TestRegistryHygiene:
    """The two token sets must not disagree with each other."""

    def test_no_token_claims_both_directions(self):
        overlap = _MAXIMIZE_METRIC_TOKENS & _MINIMIZE_METRIC_TOKENS
        assert not overlap, f"tokens in both registries: {sorted(overlap)}"

    def test_val_prefix_is_not_itself_a_token(self):
        """`val` must be stripped, not matched."""
        assert "val" not in _MAXIMIZE_METRIC_TOKENS
        assert "val" not in _MINIMIZE_METRIC_TOKENS
        assert resolve_monitor_mode("val_psnr") == resolve_monitor_mode("psnr")
