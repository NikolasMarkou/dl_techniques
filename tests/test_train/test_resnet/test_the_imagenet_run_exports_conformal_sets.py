"""``train_resnet`` fits conformal prediction sets on all of val after an ImageNet run.

A 3-class fake tree (``tests/imagenet_fake_tree.py``: 12 train images, 6 val). The run
uses deep supervision, so the conformal step must reduce the model to its primary
output, and val labels must be the plain integers, not the per-output label tuples the
fit sees.
"""

import json

import keras
import numpy as np
import pytest

import train.resnet.train_resnet as train_resnet
from tests.imagenet_fake_tree import build_fake_imagenet, use_the_fake_tree_class_count
from train.common.conformal import CONFORMAL_JSON_NAME, CONFORMAL_MODEL_NAME, load_conformal_model


def _config(root, out, name, **kw):
    return train_resnet.TrainingConfig(
        imagenet_root=root, image_size=32, batch_size=4, model_variant="resnet18",
        num_classes=3, enable_deep_supervision=True, epochs=1, warmup_epochs=0,
        augment_data=False, label_smoothing=0.0, output_dir=str(out),
        experiment_name=name, holdout_per_class=1, **kw,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = build_fake_imagenet(tmp_path_factory.mktemp("tree") / "imagenet2012")
    out = tmp_path_factory.mktemp("resnet_conformal")
    mp = pytest.MonkeyPatch()
    use_the_fake_tree_class_count(mp)
    mp.setattr(train_resnet, "setup_gpu", lambda *a, **k: None)
    # The checkpoint, analyzer and plotting callbacks are not under test and cost minutes.
    mp.setattr(train_resnet, "create_callbacks", lambda *a, **k: ([], None))
    # The trainer reports top-5; the fake tree has 3 classes, so top-k needs k <= 3.
    real_topk = train_resnet.PrimaryOutputTopKAccuracy
    mp.setattr(train_resnet, "PrimaryOutputTopKAccuracy",
               lambda *a, **k: real_topk(*a, **{**k, "k": 2}))
    try:
        train_resnet.train_resnet_imagenet(_config(root, out, "with", conformal_alpha=0.3))
        train_resnet.train_resnet_imagenet(_config(root, out, "without", conformal_alpha=None))
    finally:
        mp.undo()
    return out / "with", out / "without"


def test_the_record_covers_all_of_val_with_plain_labels(runs):
    record = json.loads((runs[0] / CONFORMAL_JSON_NAME).read_text())
    assert record["n_calibration"] == 6
    assert record["alpha"] == 0.3 and record["model"] == "resnet_resnet18"
    assert 0.0 <= record["accuracy"] <= 1.0


def test_the_exported_model_has_one_logits_output_and_the_set_mask(runs):
    model = load_conformal_model(runs[0] / CONFORMAL_MODEL_NAME)
    logits, probs, in_set = model(np.zeros((2, 32, 32, 3), "float32"))
    assert logits.shape == (2, 3) and probs.shape == (2, 3) and in_set.shape == (2, 3)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(probs).sum(axis=1), 1.0, atol=1e-5)


def test_the_primary_inference_model_is_still_saved(runs):
    assert (runs[0] / "inference_model.keras").is_file()


def test_no_conformal_leaves_no_files(runs):
    assert not (runs[1] / CONFORMAL_JSON_NAME).exists()
    assert not (runs[1] / CONFORMAL_MODEL_NAME).exists()
