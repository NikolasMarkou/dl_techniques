"""FFTNet is one class with ``include_top`` and a ``task_type`` (GUIDE section 1).

Until plan-2026-10-14T062616-81938e81 step 10.1 ``FFTNet`` was an encoder only
(D-025 deleted its dead ``include_top``) and ``create_fftnet_with_head`` /
``create_fftnet_classifier`` wrapped it in a functional ``keras.Model`` per
task. Now the class builds the task head in ``__init__`` through the vision
head factory and applies it in ``call``; ``create_fftnet`` only delegates to
``from_variant``; the two wrappers are gone. ``include_top=False`` is the
encoder, unchanged.
"""

import os

import keras
import numpy as np
import pytest

import dl_techniques.models.vision.fftnet as fftnet_package
from dl_techniques.models.vision.fftnet.model import FFTNet, create_fftnet

IMAGE, PATCH, CLASSES = 32, 16, 4
TASKS = ("classification", "detection", "segmentation")


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch, IMAGE, IMAGE, 3)).astype("float32")


def _fftnet(**kwargs) -> FFTNet:
    kwargs.setdefault("num_classes", CLASSES)
    return create_fftnet("tiny", image_size=IMAGE, patch_size=PATCH, **kwargs)


def _numpy(out):
    return {k: np.asarray(v) for k, v in out.items()} if isinstance(out, dict) \
        else np.asarray(out)


EXPECTED = {
    "classification": (2, CLASSES),
    "detection": {"classifications": (2, 2, 2, 9 * CLASSES),
                  "regressions": (2, 2, 2, 36)},
    "segmentation": (2, IMAGE, IMAGE, CLASSES),
}


def _shapes(out):
    return {k: v.shape for k, v in out.items()} if isinstance(out, dict) else out.shape


@pytest.mark.parametrize("task", TASKS)
def test_each_task_is_one_class_and_round_trips_by_value(task, tmp_path,
                                                         golden_reference_device):
    x = _images()
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(0)
        model = _fftnet(task_type=task)
        assert type(model) is FFTNet and model.include_top
        out = _numpy(model(x, training=False))
        assert _shapes(out) == EXPECTED[task]
        path = os.path.join(tmp_path, f"fftnet_{task}.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        assert type(loaded) is FFTNet and loaded.task_type == task
        again = _numpy(loaded(x, training=False))
    if isinstance(out, dict):
        for key in out:
            np.testing.assert_array_equal(again[key], out[key])
    else:
        np.testing.assert_array_equal(again, out)


def test_the_head_is_built_in_init_and_named_for_its_task():
    assert FFTNet.TASK_TYPES == TASKS
    for task in TASKS:
        model = _fftnet(task_type=task)
        assert model.head is not None and model.head.name == f"{task}_head"


def test_include_top_false_is_the_encoder_with_no_head():
    encoder = _fftnet(include_top=False)
    out = encoder(_images(), training=False)
    assert encoder.head is None
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "last_hidden_state": (2, 5, 384), "cls_token": (2, 384),
        "patch_features": (2, 4, 384)}
    headed = _fftnet()
    headed(_images(), training=False)
    head_weights = sum(int(np.prod(w.shape)) for w in headed.head.weights)
    assert headed.count_params() == encoder.count_params() + head_weights
    # The README's encoder count (tiny, 32 px, patch 16).
    assert encoder.count_params() == 5_336_832


def test_the_classifier_is_a_linear_dense_on_the_cls_token(golden_reference_device):
    """Logits are ``cls @ W + b`` of the encoder's CLS token: no activation
    (a softmax or sigmoid would break the equality; the sign of a logit is a
    draw, not evidence, at this init scale). CPU golden device: the GPU runs
    the matmul in TF32."""
    x = _images()
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(0)
        model = _fftnet()
        logits = np.asarray(model(x, training=False))
        encoder = _fftnet(include_top=False)
        encoder(x, training=False)
        # Same weights: copy the headed model's encoder weights by path.
        by_path = {w.path.split("/", 1)[1]: w for w in model.weights}
        for w in encoder.weights:
            w.assign(by_path[w.path.split("/", 1)[1]])
        cls = np.asarray(encoder(x, training=False)["cls_token"])
        kernel, bias = (np.asarray(v) for v in model.head.weights)
    np.testing.assert_allclose(logits, cls @ kernel + bias, rtol=1e-5, atol=1e-6)


def test_create_fftnet_only_delegates_and_the_wrappers_are_gone():
    a = create_fftnet("tiny", image_size=IMAGE, patch_size=PATCH, num_classes=3)
    b = FFTNet.from_variant("tiny", image_size=IMAGE, patch_size=PATCH, num_classes=3)
    assert a.get_config() | {"name": None} == b.get_config() | {"name": None}
    for gone in ("create_fftnet_with_head", "create_fftnet_classifier"):
        assert not hasattr(fftnet_package, gone)
        assert not hasattr(fftnet_package.model, gone)
    assert fftnet_package.__all__ == ["FFTNet", "create_fftnet"]


@pytest.mark.parametrize("pretrained", [True, "weights.keras"])
def test_pretrained_is_refused(pretrained):
    with pytest.raises(NotImplementedError, match="No pretrained FFTNet"):
        create_fftnet("tiny", pretrained=pretrained)


@pytest.mark.parametrize("bad,match", [
    ({"task_type": "depth"}, "task_type"),
    ({"num_classes": 0}, "num_classes"),
    ({"include_top": False, "head_config_overrides": {"dropout_rate": 0.0}},
     "head_config_overrides"),
])
def test_bad_head_arguments_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        _fftnet(**bad)
