"""``train_vit --dataset imagenet`` fits conformal prediction sets on all of val after the run.

A 3-class fake tree (``tests/imagenet_fake_tree.py``: 12 train images, 6 val). One image per
class is held out of train for selection, so the 6 official val images are a clean test split.
The CIFAR datasets get no conformal step.
"""

import json

import keras
import numpy as np
import pytest

import train.vit.train_vit as train_vit
from tests.imagenet_fake_tree import build_fake_imagenet, use_the_fake_tree_class_count
from train.common.conformal import CONFORMAL_JSON_NAME, CONFORMAL_MODEL_NAME, load_conformal_model


def _config(root, out, name, **kw):
    return train_vit.TrainingConfig(
        dataset="imagenet", imagenet_root=root, image_size=32, patch_size=8,
        batch_size=4, model_variant="vit_pico", num_classes=3, epochs=1,
        warmup_epochs=0, augment_data=False, output_dir=str(out), experiment_name=name,
        holdout_per_class=1, **kw,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")
    out = tmp_path_factory.mktemp("vit_conformal")
    mp = pytest.MonkeyPatch()
    use_the_fake_tree_class_count(mp)
    mp.setattr(train_vit, "setup_gpu", lambda *a, **k: None)
    # The checkpoint, analyzer and plotting callbacks are not under test and cost minutes.
    mp.setattr(train_vit, "create_callbacks", lambda *a, **k: ([], None))
    try:
        train_vit.train_vit(_config(root, out, "with", conformal_alpha=0.3))
        train_vit.train_vit(_config(root, out, "without", conformal_alpha=None))
    finally:
        mp.undo()
    return out / "with", out / "without"


def test_the_record_covers_all_of_val(runs):
    record = json.loads((runs[0] / CONFORMAL_JSON_NAME).read_text())
    assert record["n_calibration"] == 6
    assert record["alpha"] == 0.3 and record["model"] == "vit_vit_pico"
    assert record["model_selected_on_calibration_set"] is False


def test_the_exported_model_emits_logits_probs_and_the_set_mask(runs):
    model = load_conformal_model(runs[0] / CONFORMAL_MODEL_NAME)
    logits, probs, in_set = model(np.zeros((2, 32, 32, 3), "float32"))
    assert logits.shape == probs.shape == in_set.shape == (2, 3)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(probs).sum(axis=1), 1.0, atol=1e-5)


def test_no_conformal_leaves_no_files(runs):
    assert not (runs[1] / CONFORMAL_JSON_NAME).exists()
    assert not (runs[1] / CONFORMAL_MODEL_NAME).exists()
