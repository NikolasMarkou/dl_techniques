"""``train.convnext.common`` fits conformal prediction sets on the whole TEST split.

The model is selected on a separate split (the ImageNet holdout, or the validation slice
of train), so the test split never chose it and the record says so. This is the opposite
of the ResNet, ViT and CliffordNet runs, which calibrate on the val set that picked their
checkpoint. One fake-ImageNet run (3 classes, 6 test images) and one MNIST/CIFAR-shaped
run (fake loader, 64 test images after --max-samples) are checked for: the files, the calibration set being the
whole test split, the flag in the record, the exported model, and the off switch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.common.classification_data as classification_data  # noqa: E402
import train.convnext.common as common  # noqa: E402
from tests.imagenet_fake_tree import (  # noqa: E402
    VAL_CLASSES,
    build_fake_imagenet,
    use_the_fake_tree_class_count,
)
from tests.test_train.test_convnext.test_train_convnext import (  # noqa: E402
    N_TEST,
    _fake_loader,
)
MAX_SAMPLES = 64
from train.common.conformal import (  # noqa: E402
    CAVEAT_NOT_SELECTED_ON_CALIBRATION,
    CONFORMAL_JSON_NAME,
    CONFORMAL_MODEL_NAME,
    load_conformal_model,
)

ALPHA = 0.3


def _imagenet_config(root, out: Path, name: str, **overrides):
    settings = dict(
        dataset="imagenet", imagenet_root=root, variant="cifar10", strides=2, image_size=32,
        holdout_per_class=1, eval_samples=4, epochs=1, batch_size=4, learning_rate=1e-3,
        seed=3, output_dir=str(out), experiment_name=name, conformal_alpha=ALPHA,
        model_analysis=False,
    )
    settings.update(overrides)
    return common.TrainingConfig(**settings)


@pytest.fixture(scope="module")
def imagenet_runs(tmp_path_factory):
    tree = build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")
    out = tmp_path_factory.mktemp("convnext_conformal")
    seen = SimpleNamespace(datasets=[], kwargs=[])
    real = common.run_conformal

    def spy(model, dataset, *args, **kwargs):
        seen.datasets.append(sum(len(y) for _, y in dataset))
        seen.kwargs.append(kwargs)
        return real(model, dataset, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        use_the_fake_tree_class_count(mp)
        mp.setattr(common, "run_conformal", spy)
        on = common.train(_imagenet_config(tree, out, "on"))
        off = common.train(_imagenet_config(tree, out, "off", conformal_alpha=None))
    return SimpleNamespace(out=out, on=on, off=off, seen=seen)


def test_an_imagenet_run_writes_the_record_and_the_exported_model(imagenet_runs) -> None:
    run_dir = imagenet_runs.out / "on"
    record = json.loads((run_dir / CONFORMAL_JSON_NAME).read_text())
    assert record["alpha"] == ALPHA and record["score"] == "aps_deterministic"
    assert (run_dir / CONFORMAL_MODEL_NAME).is_file()


def test_the_calibration_set_is_the_whole_test_split(imagenet_runs) -> None:
    """6 official val images: not the 3 selection images, not the 4 figure images."""
    assert imagenet_runs.seen.datasets == [len(VAL_CLASSES)]
    record = json.loads((imagenet_runs.out / "on" / CONFORMAL_JSON_NAME).read_text())
    assert record["n_calibration"] == len(VAL_CLASSES)


def test_the_record_says_the_model_was_not_chosen_on_the_calibration_data(imagenet_runs) -> None:
    assert imagenet_runs.seen.kwargs[0]["selection_overlaps_calibration"] is False
    record = json.loads((imagenet_runs.out / "on" / CONFORMAL_JSON_NAME).read_text())
    assert record["model_selected_on_calibration_set"] is False
    assert record["caveat"] == CAVEAT_NOT_SELECTED_ON_CALIBRATION
    assert "holdout" in record["weights"] or "held out" in record["weights"]


def test_the_exported_model_returns_logits_probs_and_a_mask(imagenet_runs) -> None:
    model = load_conformal_model(imagenet_runs.out / "on" / CONFORMAL_MODEL_NAME)
    logits, probs, in_set = model(np.random.default_rng(0).random((5, 32, 32, 3), dtype=np.float32))
    assert tuple(logits.shape) == tuple(probs.shape) == tuple(in_set.shape) == (5, 3)
    assert set(np.unique(keras.ops.convert_to_numpy(in_set))) <= {0.0, 1.0}


def test_the_summary_carries_the_record(imagenet_runs) -> None:
    on_disk = json.loads((imagenet_runs.out / "on" / "results_summary.json").read_text())
    record = json.loads((imagenet_runs.out / "on" / CONFORMAL_JSON_NAME).read_text())
    assert on_disk["conformal"] == record
    assert any("conformal.json" in n and "WHOLE test split" in n for n in on_disk["notes"])


def test_a_run_without_conformal_alpha_leaves_no_files_and_a_null_summary_block(imagenet_runs) -> None:
    run_dir = imagenet_runs.out / "off"
    assert not (run_dir / CONFORMAL_JSON_NAME).exists()
    assert not (run_dir / CONFORMAL_MODEL_NAME).exists()
    summary = json.loads((run_dir / "results_summary.json").read_text())
    assert summary["conformal"] is None
    assert any("--conformal-alpha" in n for n in summary["notes"])
    assert len(imagenet_runs.seen.datasets) == 1


def test_an_array_dataset_calibrates_on_its_test_split_too(tmp_path) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(classification_data, "_load_raw", _fake_loader(10))
        mp.setattr(common, "run_model_analysis", lambda *a, **k: None)
        config = common.TrainingConfig(
            variant="cifar10", epochs=1, batch_size=8, learning_rate=1e-3, max_samples=MAX_SAMPLES,
            seed=5, output_dir=str(tmp_path), experiment_name="cifar", conformal_alpha=0.1,
            model_analysis=False,
        )
        summary = common.train(config)
    record = json.loads((tmp_path / "cifar" / CONFORMAL_JSON_NAME).read_text())
    # --max-samples caps the test split as well; the record covers all of what is left.
    assert record["n_calibration"] == summary["n_test"] == MAX_SAMPLES < N_TEST
    assert record["model_selected_on_calibration_set"] is False
    assert "validation_split=0.1" in record["weights"]


# ---------------------------------------------------------------------
# The step is off unless asked for
# ---------------------------------------------------------------------


def _config_from(*argv):
    args = common.parse_arguments(list(argv), "v1")
    return common.config_from_args(args, "v1")


def test_the_step_is_off_by_default(tmp_path) -> None:
    assert common.TrainingConfig(output_dir=str(tmp_path)).conformal_alpha is None
    assert _config_from().conformal_alpha is None


def test_one_flag_turns_it_on_and_a_bare_flag_uses_the_default_alpha() -> None:
    assert _config_from("--conformal-alpha").conformal_alpha == 0.1
    assert _config_from("--conformal-alpha", "0.05").conformal_alpha == 0.05


def test_there_is_no_off_switch_because_off_is_the_default(capsys) -> None:
    with pytest.raises(SystemExit):
        common.parse_arguments(["--no-conformal"], "v1")
    assert "unrecognized arguments" in capsys.readouterr().err
