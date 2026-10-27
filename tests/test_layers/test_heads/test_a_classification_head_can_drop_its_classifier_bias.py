"""``ClassificationHead`` can build its classifier ``Dense`` without a bias.

The head's classifier always had a bias, so a model whose reference head is a
bias-free linear layer (RADConvNet with ``use_bias=False``) could not use the
shared head. ``classifier_use_bias`` is the knob: ``True`` (the default) keeps
every existing head byte-identical, ``False`` builds the classifier with no
bias weight.

``bias_initializer`` and ``bias_regularizer`` are read only by a ``Dense`` that
has a bias. With no bias anywhere in the head (``classifier_use_bias=False``
and no hidden block with a bias) a value for either is refused and the head
stores ``None``, the rule of ``test_a_vision_head_stores_no_argument_it_never_reads.py``.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision.classification import ClassificationHead
from dl_techniques.layers.heads.vision.config import build_model_vision_head

HIDDEN_DIM = 16
SHAPE = (2, 4, 4, HIDDEN_DIM)
X = np.random.default_rng(5).normal(size=SHAPE).astype("float32")


def _paths(head: keras.layers.Layer) -> list:
    """Weight paths below the head: its own auto name carries a process counter."""
    return [w.path.split("/", 1)[1] for w in head.weights]


def _round_trip(head: ClassificationHead) -> ClassificationHead:
    restored = ClassificationHead.from_config(head.get_config())
    assert restored.get_config() == head.get_config()
    return restored


def _bias_free(**kwargs: Any) -> ClassificationHead:
    kwargs.setdefault("block_dims", ())
    return ClassificationHead(
        num_classes=3, hidden_dim=HIDDEN_DIM, classifier_use_bias=False, **kwargs)


# ---------------------------------------------------------------------
# the classifier has no bias
# ---------------------------------------------------------------------

def test_the_classifier_bias_can_be_dropped() -> None:
    head = _bias_free()
    head(X, training=False)
    assert head.classifier.use_bias is False
    assert head.classifier.bias is None
    assert _paths(head) == ["classifier/kernel"]
    assert tuple(head.classifier.kernel.shape) == (HIDDEN_DIM, 3)


def test_the_logits_without_a_bias_are_the_pooled_vector_times_the_kernel(
        golden_reference_device: str) -> None:
    with keras.device(golden_reference_device):
        head = _bias_free()
        out = head(X, training=False)
        pooled = keras.ops.mean(X, axis=(1, 2))
        expected = keras.ops.matmul(pooled, head.classifier.kernel)
        np.testing.assert_allclose(
            np.asarray(out["logits"]), np.asarray(expected), rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(out["probabilities"]),
            np.asarray(keras.ops.softmax(out["logits"], axis=-1)), rtol=0, atol=0)


def test_the_hidden_blocks_keep_their_bias_when_only_the_classifier_drops_it() -> None:
    head = ClassificationHead(
        num_classes=3, hidden_dim=HIDDEN_DIM, block_dims=(8,),
        classifier_use_bias=False)
    head(X, training=False)
    assert head.dense_blocks[0].dense.bias is not None
    assert head.classifier.bias is None


# ---------------------------------------------------------------------
# the default is unchanged
# ---------------------------------------------------------------------

def test_the_default_classifier_keeps_its_bias() -> None:
    head = ClassificationHead(num_classes=3, hidden_dim=HIDDEN_DIM, block_dims=())
    head(X, training=False)
    assert head.classifier_use_bias is True
    assert head.classifier.use_bias is True
    assert _paths(head) == ["classifier/kernel", "classifier/bias"]
    assert head.get_config()["classifier_use_bias"] is True


@pytest.mark.parametrize("kwargs", [
    {},
    {"block_dims": ()},
    {"block_dims": (8, 4), "normalization_type": "batch_norm",
     "block_use_bias": False},
    {"block_dims": (8,), "kernel_initializer": "he_normal",
     "bias_initializer": "ones", "bias_regularizer": "l1"},
], ids=["default", "no_blocks", "batch_norm_blocks", "initializers"])
def test_an_explicit_true_is_the_omitted_default_down_to_the_last_bit(
        kwargs: Dict[str, Any], golden_reference_device: str) -> None:
    def make(**extra: Any) -> ClassificationHead:
        keras.utils.set_random_seed(1234)
        return ClassificationHead(
            num_classes=3, hidden_dim=HIDDEN_DIM, **kwargs, **extra)

    with keras.device(golden_reference_device):
        omitted, explicit = make(), make(classifier_use_bias=True)
        out_omitted = omitted(X, training=False)
        out_explicit = explicit(X, training=False)
    assert _paths(omitted) == _paths(explicit)
    for a, b in zip(omitted.weights, explicit.weights):
        assert tuple(a.shape) == tuple(b.shape)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=a.path)
    for key in ("logits", "probabilities"):
        np.testing.assert_array_equal(
            np.asarray(out_omitted[key]), np.asarray(out_explicit[key]))


# ---------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------

@pytest.mark.parametrize("flag", [True, False])
def test_the_flag_is_in_the_config_and_survives_a_round_trip(flag: bool) -> None:
    head = ClassificationHead(
        num_classes=3, hidden_dim=HIDDEN_DIM, block_dims=(),
        classifier_use_bias=flag)
    assert head.get_config()["classifier_use_bias"] is flag
    restored = _round_trip(head)
    assert restored.classifier_use_bias is flag
    restored(X, training=False)
    assert (restored.classifier.bias is None) is (not flag)


def test_a_bias_free_head_survives_a_keras_save_and_load(tmp_path) -> None:
    head = _bias_free()
    inputs = keras.Input(shape=SHAPE[1:])
    model = keras.Model(inputs, head(inputs)["logits"])
    path = tmp_path / "bias_free_head.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    assert sorted(w.path for w in restored.weights) == sorted(
        w.path for w in model.weights)
    assert not any(w.path.endswith("/bias") for w in restored.weights)
    np.testing.assert_array_equal(
        np.asarray(restored(X, training=False)),
        np.asarray(model(X, training=False)))


# ---------------------------------------------------------------------
# a bias argument no layer reads is refused and stored as None
# ---------------------------------------------------------------------

@pytest.mark.parametrize("unread", [
    {"bias_initializer": "ones"},
    {"bias_regularizer": "l2"},
], ids=["bias_initializer", "bias_regularizer"])
def test_a_bias_argument_with_no_bias_anywhere_is_refused(
        unread: Dict[str, Any]) -> None:
    (name,) = unread
    with pytest.raises(ValueError, match=name):
        _bias_free(**unread)


@pytest.mark.parametrize("unread", [
    {"bias_initializer": "ones"},
    {"bias_regularizer": "l2"},
], ids=["bias_initializer", "bias_regularizer"])
def test_bias_free_blocks_with_a_bias_free_classifier_refuse_them_too(
        unread: Dict[str, Any]) -> None:
    (name,) = unread
    with pytest.raises(ValueError, match=name):
        _bias_free(block_dims=(8,), block_use_bias=False, **unread)


def test_a_head_with_no_bias_anywhere_stores_none_and_round_trips() -> None:
    head = _bias_free()
    assert head.bias_initializer is None and head.bias_regularizer is None
    config = head.get_config()
    assert config["bias_initializer"] is None
    assert config["bias_regularizer"] is None
    restored = _round_trip(head)
    assert restored.bias_initializer is None and restored.bias_regularizer is None
    restored(X, training=False)
    assert _paths(restored) == ["classifier/kernel"]


def test_an_explicit_zeros_initializer_is_the_default_not_a_value() -> None:
    head = _bias_free(bias_initializer="zeros")
    assert head.bias_initializer is None


@pytest.mark.parametrize("make", [
    lambda: ClassificationHead(
        num_classes=3, hidden_dim=HIDDEN_DIM, block_dims=(),
        bias_initializer="ones", bias_regularizer="l2"),
    lambda: ClassificationHead(
        num_classes=3, hidden_dim=HIDDEN_DIM, block_dims=(8,),
        classifier_use_bias=False, bias_initializer="ones", bias_regularizer="l2"),
], ids=["classifier_bias", "block_bias"])
def test_a_bias_argument_is_kept_while_some_dense_has_a_bias(make) -> None:
    head = make()
    assert head.bias_initializer is not None and head.bias_regularizer is not None
    assert _round_trip(head).bias_regularizer is not None


# ---------------------------------------------------------------------
# the knob reaches the head through the model-side factory
# ---------------------------------------------------------------------

def test_the_model_head_config_carries_the_flag_to_the_head() -> None:
    head = build_model_vision_head(
        "classification",
        {"num_classes": 3, "name": "classification_head", "block_dims": (),
         "classifier_use_bias": False, "use_ffn": False,
         "hidden_dim": HIDDEN_DIM})
    head(X, training=False)
    assert isinstance(head, ClassificationHead)
    assert head.classifier_use_bias is False and head.classifier.bias is None


def test_head_config_overrides_can_carry_the_flag_to_the_head() -> None:
    head = build_model_vision_head(
        "classification",
        {"num_classes": 3, "name": "classification_head", "block_dims": (),
         "use_ffn": False, "hidden_dim": HIDDEN_DIM},
        {"classifier_use_bias": False})
    head(X, training=False)
    assert head.classifier_use_bias is False and head.classifier.bias is None
