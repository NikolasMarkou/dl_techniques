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

import keras
import numpy as np
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


@pytest.mark.integration
@pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason=(
        f"requires real local data at {KITTI_DEPTH_ROOT!r} and {MEGADEPTH_ROOT!r} "
        "(not present on this machine)"
    ),
)
def test_smoke_run_completes_one_training_step_with_conditioning_enabled(tmp_path):
    """`--smoke --enable-conditioning` must also fit end to end (decisions.md D-028) --
    proving the CLI path that exercises `OmniPoint`'s conditioning branch still runs,
    not just the default `enable_conditioning=False` path already covered above."""
    omnipoint = trainer.main(
        [
            "--smoke",
            "--enable-conditioning",
            "--kitti-root", KITTI_DEPTH_ROOT,
            "--megadepth-root", MEGADEPTH_ROOT,
            "--output-dir", str(tmp_path),
        ]
    )

    from dl_techniques.models.vision.omnipoint.model import OmniPoint

    assert isinstance(omnipoint, OmniPoint)
    assert omnipoint.enable_conditioning is True

    import numpy as np

    dummy = keras.random.normal([1, trainer.SMOKE_IMAGE_SIZE, trainer.SMOKE_IMAGE_SIZE, 3])
    ray, distance, point, mask_logit, scale = omnipoint(dummy, training=False)
    for name, tensor in (
        ("ray", ray), ("distance", distance), ("point", point),
        ("mask_logit", mask_logit), ("scale", scale),
    ):
        values = keras.ops.convert_to_numpy(tensor)
        assert np.isfinite(values).all(), f"{name} contains non-finite values after training"


# ---------------------------------------------------------------------
# `--enable-conditioning` gradient-flow proof (decisions.md D-025 CRITICAL #4 / D-028)
# ---------------------------------------------------------------------


class TestEnableConditioningGradientFlow:
    """Reproduces the adversarial review's exact gradient-flow measurement directly
    against `OmniPointTrainingWrapper` (not just `OmniPoint` in isolation), using a
    real batch shape from `CombinedOmniPointDataset` -- synthetic data, so this runs
    fast and does not require real local KITTI/MegaDepth mirrors.

    Before D-028: `intrinsics_conv1/{kernel,bias}`, `intrinsics_conv2/kernel` and both
    `*_state/present_embedding` vectors were measured to have EXACTLY zero gradient
    whenever `enable_conditioning=True` (decisions.md D-025 CRITICAL #4) -- the one
    mechanism by which camera geometry was supposed to enter the model was never
    trained. After D-028, the intrinsics half is wired (via `data.py`'s own `gt_ray`,
    reused rather than re-derived) and MUST show live gradient; the sparse-depth half
    remains genuinely unwired (no sparse-depth data exists in this pipeline) and MUST
    still show exactly zero gradient -- a documented, intentional limitation now,
    proven by name rather than merely asserted in a docstring.
    """

    def _build_wrapper_and_batch(self, tmp_path):
        from tests.test_train.test_omnipoint.test_data import (
            _make_synthetic_kitti_tree,
            _make_synthetic_megadepth_tree,
        )
        from train.omnipoint.data import CombinedOmniPointDataset
        from dl_techniques.losses.omnipoint_losses import OmniPointCombinedLoss
        from dl_techniques.models.vision.omnipoint.model import create_omnipoint

        image_size = 28  # 2x OmniPoint's ViT-L/14 patch size -- smallest valid grid
        kitti_root, kitti_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=4)
        mega_rgb, mega_depth, _ = _make_synthetic_megadepth_tree(tmp_path, num_pairs=4)
        ds = CombinedOmniPointDataset(
            kitti_pairs, mega_rgb, mega_depth,
            batch_size=2, patch_size=image_size, is_training=False, workers=1,
        )
        rgb, y_true = ds[0]

        omnipoint = create_omnipoint(
            "omnipoint_base", image_shape=(image_size, image_size, 3),
            enable_conditioning=True,
        )
        wrapper = trainer.OmniPointTrainingWrapper(omnipoint, OmniPointCombinedLoss())
        return wrapper, (rgb, y_true)

    def test_intrinsics_conditioning_weights_now_receive_nonzero_gradient(self, tmp_path):
        from tests.test_models.gradient_flow_oracle import gradient_report

        wrapper, batch = self._build_wrapper_and_batch(tmp_path)

        # The wrapper supervises via `self.add_loss(...)` inside `call()` (D-020), not a
        # returned tensor -- so the loss for this gradient probe is the SAME quantity
        # `fit()` actually trains against, read from `wrapper.losses` after the forward
        # pass the oracle's own tape already ran.
        report = gradient_report(
            wrapper, batch, loss_fn=lambda outputs: keras.ops.sum(wrapper.losses),
        )

        def _matching(substr):
            matches = {p: v for p, v in report.items() if substr in p}
            assert matches, (
                f"no weight path contains {substr!r} -- rename drifted, update this test"
            )
            return matches

        # THE FIX (D-028): intrinsics conditioning is now wired and trains.
        for substr in (
            "intrinsics_conv1/kernel", "intrinsics_conv1/bias",
            "intrinsics_conv2/kernel", "intrinsics_conv2/bias",
            "intrinsics_state/present_embedding",
        ):
            for path, value in _matching(substr).items():
                assert value is not None and np.isfinite(value) and value > 0.0, (
                    f"{path}: expected a live, finite, nonzero gradient now that "
                    f"intrinsics conditioning is wired (D-028), got {value}"
                )

        # STILL UNWIRED, ON PURPOSE (no sparse-depth data exists in this pipeline):
        # the sparse-depth conv KERNELS and the depth present-embedding never see a
        # nonzero input/flag, so they must remain exactly dead. (Their bias terms are
        # NOT asserted here: a `linear`-activation conv's bias trains even on an
        # all-zero input, and a `relu`-activation conv's bias sits at the 0-gradient
        # kink of relu(0) -- both are incidental to the all-zero depth branch, not
        # evidence the modality itself is wired.)
        for substr in (
            "depth_conv1/kernel", "depth_conv2/kernel", "depth_state/present_embedding",
        ):
            for path, value in _matching(substr).items():
                assert value == 0.0, (
                    f"{path}: expected EXACTLY zero gradient (sparse-depth conditioning "
                    f"is deliberately unwired -- no sparse-depth data exists in this "
                    f"pipeline, see decisions.md D-028), got {value}"
                )
