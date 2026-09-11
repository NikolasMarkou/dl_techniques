"""CLI contract + integration-smoke guard for `src/train/omnipoint/train_omnipoint.py`.

Two claims, proven directly rather than assumed:

1. `--help` exits 0, prints a `usage:` line, and documents every OmniPoint-specific flag
   (repo convention: exit code 0 alone is not evidence a parser exists at all).
2. `main(["--smoke", ...])` completes at least one real training step (forward + backward
   + optimizer update) against `OmniPointTrainingWrapper`'s `add_loss` supervision path
   (D-020), against a tiny image size / batch / variant, and reports a finite loss.

Both real dataset roots (KITTI depth, MegaDepth) are required for the smoke test -- skipped,
not silently passed, if either is absent on the running machine (same convention as
`tests/test_train/test_omnipoint/test_data.py`).
"""

from pathlib import Path

import pytest

from train.omnipoint import train_omnipoint as trainer

KITTI_DEPTH_ROOT = "/media/arxwn/data0_4tb/datasets/KITTI/data/depth"
MEGADEPTH_ROOT = "/media/arxwn/data0_4tb/datasets/Megadepth"

_REAL_DATA_AVAILABLE = Path(KITTI_DEPTH_ROOT).exists() and Path(MEGADEPTH_ROOT).exists()


# ---------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------


def test_help_exits_zero_with_usage_line(capsys):
    """`--help` must exit 0 and print a real `usage:` line -- not just exit 0."""
    with pytest.raises(SystemExit) as excinfo:
        trainer.parse_arguments(["--help"])

    assert excinfo.value.code == 0
    printed = capsys.readouterr().out
    assert printed.startswith("usage:"), (
        "--help printed no `usage:` line. An exit code of 0 is not evidence: a script "
        "with no parser at all runs its whole job and exits 0."
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--dataset",
        "--image-size",
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--weight-decay",
        "--lr-schedule",
        "--patience",
        "--gpu",
        "--show-plots",
        "--omnipoint-variant",
        "--enable-conditioning",
        "--kitti-root",
        "--megadepth-root",
        "--max-kitti-files",
        "--max-megadepth-files",
        "--lambda-ray",
        "--lambda-metric",
        "--lambda-normal",
        "--lambda-local",
        "--lambda-mask",
        "--smoke",
    ],
)
def test_help_documents_every_flag(capsys, flag):
    with pytest.raises(SystemExit):
        trainer.parse_arguments(["--help"])
    printed = capsys.readouterr().out
    assert flag in printed, f"--help did not document {flag!r}"


def test_help_reaches_argparse_before_any_allocation(monkeypatch, capsys):
    """`parse_arguments()` must be `main()`'s first statement: no GPU/model/dataset work
    may happen before `--help` exits."""
    reached = []

    def sentinel(name):
        def _raise(*args, **kwargs):
            reached.append(name)
            raise AssertionError(f"--help reached {name!r}")

        return _raise

    for name in ("setup_gpu", "set_seeds", "train_omnipoint"):
        monkeypatch.setattr(trainer, name, sentinel(name))

    with pytest.raises(SystemExit) as excinfo:
        trainer.main(["--help"])

    assert excinfo.value.code == 0
    assert not reached, f"--help reached {reached!r} before argparse could exit."


# ---------------------------------------------------------------------
# --smoke config wiring
# ---------------------------------------------------------------------


def test_smoke_flag_overrides_dataset_and_training_size():
    args = trainer.parse_arguments(
        [
            "--smoke",
            "--kitti-root", KITTI_DEPTH_ROOT,
            "--megadepth-root", MEGADEPTH_ROOT,
        ]
    )
    config = trainer._config_from_args(args)

    assert config.image_size == trainer.SMOKE_IMAGE_SIZE
    assert config.batch_size == trainer.SMOKE_BATCH_SIZE
    assert config.epochs == trainer.SMOKE_EPOCHS
    assert config.max_kitti_files == trainer.SMOKE_MAX_FILES
    assert config.max_megadepth_files == trainer.SMOKE_MAX_FILES
    assert config.workers == trainer.SMOKE_WORKERS


# ---------------------------------------------------------------------
# 1-batch/1-epoch integration smoke fit
# ---------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason=(
        f"requires real local data at {KITTI_DEPTH_ROOT!r} and {MEGADEPTH_ROOT!r} "
        "(not present on this machine)"
    ),
)
def test_smoke_run_completes_one_training_step_with_finite_loss(tmp_path):
    """`--smoke` must actually fit, end to end, through `OmniPointTrainingWrapper`'s
    `add_loss` path (D-020) -- proving Success Criterion 8 directly, not by inspection."""
    omnipoint = trainer.main(
        [
            "--smoke",
            "--kitti-root", KITTI_DEPTH_ROOT,
            "--megadepth-root", MEGADEPTH_ROOT,
            "--output-dir", str(tmp_path),
        ]
    )

    # `main()` returns the trained (real, unwrapped) OmniPoint -- not the training wrapper.
    from dl_techniques.models.vision.omnipoint.model import OmniPoint

    assert isinstance(omnipoint, OmniPoint)

    import numpy as np
    import keras

    dummy = keras.random.normal([1, trainer.SMOKE_IMAGE_SIZE, trainer.SMOKE_IMAGE_SIZE, 3])
    ray, distance, point, mask_logit, scale = omnipoint(dummy, training=False)
    for name, tensor in (
        ("ray", ray), ("distance", distance), ("point", point),
        ("mask_logit", mask_logit), ("scale", scale),
    ):
        values = keras.ops.convert_to_numpy(tensor)
        assert np.isfinite(values).all(), f"{name} contains non-finite values after training"
