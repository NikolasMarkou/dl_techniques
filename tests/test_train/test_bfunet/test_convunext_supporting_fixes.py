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

from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import (
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

    def test_config_default_is_batchnorm(self) -> None:
        assert TrainingConfig().block_normalization == "batchnorm"

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

    def test_build_model_default_blocks_are_batchnorm(self) -> None:
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
            b.get_config()["normalization_type"] == "batchnorm" for b in blocks
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

    def test_argparse_default_is_batchnorm(self, monkeypatch) -> None:
        argv = ["train_convunext_denoiser", "--smoke"]
        monkeypatch.setattr("sys.argv", argv)
        args = parse_arguments()
        assert args.block_normalization == "batchnorm"


if __name__ == "__main__":
    pytest.main([__file__])


# ---------------------------------------------------------------------
# Step-6 fixes of plan-2026-09-19T224205-49c8bf80 (D-015): dashboard ticks and best-epoch
# marker, clipped and seeded grid, seeded fixed batch, one test pass when best == last,
# the model summary out of run.log, and an import that opens no TF context
# ---------------------------------------------------------------------

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import tensorflow as tf

from train.bfunet import common

REPO_SRC = Path(__file__).resolve().parents[3] / "src"


@pytest.mark.parametrize("module", ["train.bfunet.common", "train.bfunet.train_convunext_denoiser"])
def test_importing_the_bfunet_trainers_opens_no_tensorflow_context(module) -> None:
    """``setup_gpu`` can only enable memory growth while the TF context is unopened.

    ``tf.config.set_visible_devices`` raises once the context exists (CPU-only machines too),
    so a clean call after the import proves the import ran no eager op. It used to raise:
    ``jacobian_symmetry`` built a ``tf.constant`` at import and the run log carried the
    "Physical devices cannot be modified after being initialized" ERROR.
    """
    code = f"""
        import tensorflow as tf
        import {module}
        tf.config.set_visible_devices([], "GPU")
        print("NO_CONTEXT_OPENED")
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], cwd=REPO_SRC, capture_output=True,
        text=True, env={**os.environ, "MPLBACKEND": "Agg"}, timeout=600,
    )
    assert "NO_CONTEXT_OPENED" in result.stdout, result.stderr[-1500:]


def _history(epochs: List[int], val_loss: List[float]) -> Dict[str, List[float]]:
    n = len(epochs)
    return {
        "epoch": list(epochs), "loss": [0.05] * n, "val_loss": list(val_loss), "mae": [0.1] * n,
        "val_mae": [0.1] * n, "psnr": [20.0] * n, "val_psnr": [22.0 + i for i in range(n)],
        "sigma_max": [0.1] * n, "lr": [1e-3] * n,
    }


def _render_capturing_figure(history, tmp_path, **kwargs) -> List[plt.Axes]:
    """Render the dashboard and return its axes as they stood when the PNG was saved."""
    seen: List[List[plt.Axes]] = []
    real_savefig = plt.savefig

    def spy(path, *args, **kw):
        seen.append(list(plt.gcf().axes))
        return real_savefig(path, *args, **kw)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plt, "savefig", spy)
        common.render_training_dashboard(
            history, tmp_path / "dash.png", sigma_min=0.0, val_sigma_max=0.25, **kwargs)
    return seen[0]


class TestDashboardTicksAndBestMarker:
    def test_a_two_epoch_history_has_integer_epoch_ticks_on_every_panel(self, tmp_path) -> None:
        axes = _render_capturing_figure(_history([1, 2], [0.5, 0.2]), tmp_path)
        assert len(axes) >= 8
        for ax in axes:
            low, high = ax.get_xlim()
            visible = [t for t in ax.get_xticks() if low <= t <= high]
            assert visible and all(float(t).is_integer() for t in visible), (ax.get_title(), visible)

    @staticmethod
    def _stars(ax) -> List[tuple]:
        return [(list(line.get_xdata()), list(line.get_ydata()))
                for line in ax.get_lines() if line.get_marker() == "*"]

    def test_the_best_epoch_gets_a_dashed_line_and_a_star_on_the_val_point(self, tmp_path) -> None:
        history = _history([1, 2, 3], [0.5, 0.2, 0.3])
        assert common.best_epoch_of_history(history) == 2
        axes = _render_capturing_figure(history, tmp_path, best_epoch=2)
        by_title = {ax.get_title(): ax for ax in axes}
        mse = by_title["MSE per epoch (RAW - train task is MOVING)"]
        assert self._stars(mse) == [([2], [0.2])]
        dashed = [line for line in mse.get_lines()
                  if line.get_linestyle() == "--" and list(line.get_xdata()) == [2, 2]]
        assert dashed, "no dashed vertical line at the best epoch"
        assert self._stars(by_title["PSNR per epoch (RAW) vs noisy-input PSNR"]) == [([2], [23.0])]

    def test_no_best_epoch_argument_draws_no_marker(self, tmp_path) -> None:
        axes = _render_capturing_figure(_history([1, 2, 3], [0.5, 0.2, 0.3]), tmp_path)
        assert all(self._stars(ax) == [] for ax in axes)

    def test_the_best_epoch_ignores_the_untrained_baseline_and_non_finite_values(self) -> None:
        history = _history([0, 1, 2, 3], [0.01, float("nan"), 0.4, 0.3])
        assert common.best_epoch_of_history(history) == 3
        assert common.best_epoch_of_history(_history([0], [0.01])) is None


class _Scale(keras.layers.Layer):
    """3x - 1: on a [0,1] input the output leaves [0,1] on both sides (the unclipped-PSNR trap)."""

    def call(self, x):
        return 3.0 * x - 1.0


def _grid_callback(tmp_path, clean, model, seed, name="viz") -> common.DenoisingVisualizationCallback:
    cb = common.DenoisingVisualizationCallback(
        clean_batch=clean, sigma_max_var=tf.Variable(0.1), out_dir=tmp_path / name,
        max_samples=clean.shape[0], noise_regimes=[("low", 0.1)], multi_pass_k=3, noise_seed=seed,
    )
    cb.set_model(model)
    return cb


def _render_grid(cb, epoch: int) -> SimpleNamespace:
    """Draw one grid and return its row labels and the first image of each row, read off the live figure."""
    seen = []
    real_savefig = plt.savefig

    def spy(path, *args, **kw):
        axes = plt.gcf().axes
        n_cols = cb.max_samples
        rows = [axes[r * n_cols:(r + 1) * n_cols] for r in range(len(axes) // n_cols)]
        seen.append(SimpleNamespace(
            labels=[row[0].get_ylabel() for row in rows],
            images=[np.stack([np.asarray(a.images[0].get_array()) for a in row]) for row in rows],
        ))
        return real_savefig(path, *args, **kw)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plt, "savefig", spy)
        cb._save_grid(epoch)
    return seen[0]


@pytest.fixture(scope="module")
def grid_fixture():
    clean = tf.constant(np.random.default_rng(3).uniform(0.0, 1.0, size=(2, 8, 8, 1)).astype("float32"))
    model = keras.Sequential([keras.Input(shape=(8, 8, 1)), _Scale()])
    return SimpleNamespace(clean=clean, model=model)


def _psnr(pred: np.ndarray, clean: np.ndarray) -> float:
    return 20.0 * float(np.log10(1.0 / np.sqrt(np.mean((pred - clean) ** 2))))


class TestGridLabelsAndSeededNoise:
    def test_the_pass_one_label_is_the_clipped_psnr_the_run_log_prints(self, tmp_path, grid_fixture) -> None:
        clean = grid_fixture.clean
        noise = tf.random.stateless_normal(tf.shape(clean), seed=[7, 0])
        noisy = tf.clip_by_value(clean + noise * 0.1, 0.0, 1.0).numpy()
        raw = 3.0 * noisy - 1.0
        clipped_psnr = _psnr(np.clip(raw, 0.0, 1.0), clean.numpy())
        unclipped_psnr = _psnr(raw, clean.numpy())
        assert f"{unclipped_psnr:.1f}" != f"{clipped_psnr:.1f}", "fixture must separate the two"
        grid = _render_grid(_grid_callback(tmp_path, clean, grid_fixture.model, seed=7), 1)
        label = next(t for t in grid.labels if t.startswith("Denoised low\n(PSNR"))
        assert label == f"Denoised low\n(PSNR {clipped_psnr:.1f} dB)"
        assert f"{common.multi_pass_psnr(grid_fixture.model, clean, noisy, 3)[0]:.1f}" in label

    def test_the_noisy_rows_are_identical_across_epochs_and_across_same_seed_runs(
            self, tmp_path, grid_fixture) -> None:
        first = _grid_callback(tmp_path, grid_fixture.clean, grid_fixture.model, seed=7, name="a")
        again = _grid_callback(tmp_path, grid_fixture.clean, grid_fixture.model, seed=7, name="b")
        noisy_rows = [_render_grid(cb, epoch).images[1] for cb, epoch in ((first, 1), (first, 2), (again, 1))]
        assert np.array_equal(noisy_rows[0], noisy_rows[1])
        assert np.array_equal(noisy_rows[0], noisy_rows[2])

    def test_a_different_seed_gives_a_different_noisy_row(self, tmp_path, grid_fixture) -> None:
        one = _grid_callback(tmp_path, grid_fixture.clean, grid_fixture.model, seed=7, name="a")
        other = _grid_callback(tmp_path, grid_fixture.clean, grid_fixture.model, seed=8, name="b")
        assert not np.array_equal(_render_grid(one, 1).images[1], _render_grid(other, 1).images[1])


def _write_pngs(directory: Path, count: int) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    paths = []
    for index in range(count):
        path = directory / f"img_{index}.png"
        path.write_bytes(tf.io.encode_png(rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)).numpy())
        paths.append(str(path))
    return paths


class TestFixedVizBatchIsSeeded:
    def test_one_seed_gives_the_same_batch_twice_and_another_seed_a_different_one(self, tmp_path) -> None:
        paths = _write_pngs(tmp_path / "val", 4)
        config = TrainingConfig(patch_size=16)
        first = common.build_fixed_val_batch(paths, config, n=4, seed=11)
        again = common.build_fixed_val_batch(paths, config, n=4, seed=11)
        other = common.build_fixed_val_batch(paths, config, n=4, seed=12)
        assert first.shape == (4, 16, 16, 3)
        assert np.array_equal(first.numpy(), again.numpy())
        assert not np.array_equal(first.numpy(), other.numpy())


class TestARenderThatRaisesIsRecordedNotRaised:
    def test_a_failing_dashboard_lands_in_failed_and_training_goes_on(self, tmp_path) -> None:
        cb = common.DenoisingVisualizationCallback(
            clean_batch=None, sigma_max_var=tf.Variable(0.1), out_dir=tmp_path, noise_seed=1)

        def boom(*args, **kwargs):
            raise RuntimeError("scripted dashboard failure")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(common, "render_training_dashboard", boom)
            cb.on_epoch_end(0, {"loss": 0.1, "val_loss": 0.2})
        assert cb.failed == {"training_dashboard.png": "RuntimeError: scripted dashboard failure"}
        block = cb.visualizations_block()
        assert block["failed"] == cb.failed and block["files"] == [] and block["seconds"] > 0
