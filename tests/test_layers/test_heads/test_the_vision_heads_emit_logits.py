"""The vision ``ClassificationHead`` emits real logits, and their softmax beside them.

Guard for plan-2026-10-14T062616-81938e81 step 1 (decisions D-001, D-004).

Before that step the head ended in ``Dense(num_classes, activation='softmax')``
and returned the SAME tensor under ``'logits'`` and ``'probabilities'``. MEASURED
at the parent commit: ``out['logits'] is out['probabilities']`` was ``True`` and
every row summed to 1.0. A ``from_logits=True`` loss on that output trained only
because Keras 3.8 detects the softmax provenance and silently undoes it; any
custom loss that applies its own ``log_softmax`` double-softmaxed.

The contract pinned here mirrors the NLP ``TextClassificationHead``
(``layers/heads/nlp/factory.py``): a linear classifier, then
``probabilities = keras.ops.softmax(logits, axis=-1)`` in the compute dtype, with
no float32 cast.

Every tolerance is derived, never pasted: :func:`_softmax_atol` scales
``tests.numerics.reassociation_atol`` (a float32 random-walk rounding bound over
the softmax's ``num_classes``-long reduction) by the ratio of the compute dtype's
machine epsilon to float32's. Under ``mixed_float16`` that is 8192x the float32
bound, because the head's softmax runs in float16.

The file also pins :func:`merge_head_config_overrides`, the helper every model
that owns a vision head merges its ``head_config_overrides`` through.
"""

from typing import Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.heads.vision.factory import (
    RESERVED_HEAD_OVERRIDE_KEYS,
    ClassificationHead,
    merge_head_config_overrides,
)
from tests.numerics import reassociation_atol

NUM_CLASSES = 7
HIDDEN_DIM = 16

# Non-square spatial input, batch != classes != channels, so a transposed or
# mis-reduced axis cannot pass by coincidence.
FEATURE_SHAPE = (3, 5, 9, 12)
SEQUENCE_SHAPE = (3, 6, 12)

# Two pooling paths that reach the classifier: 4D feature maps through global
# average pooling, 3D sequences through the CLS token.
INPUT_VARIANTS = {
    "avg_4d": ({"pooling_type": "avg"}, FEATURE_SHAPE),
    "cls_token_3d": ({"pooling_type": "cls_token"}, SEQUENCE_SHAPE),
}


def _inputs(shape: Tuple[int, ...]) -> np.ndarray:
    """A non-degenerate input: unit-variance normal scaled by 3, fixed seed.

    :param shape: Input shape.
    :type shape: Tuple[int, ...]
    :return: float32 array.
    :rtype: np.ndarray
    """
    return (3.0 * np.random.default_rng(11).standard_normal(shape)).astype("float32")


def _head(**kwargs) -> ClassificationHead:
    """Build a small head. ``use_ffn=True`` on purpose: it must build nothing.

    :return: The head.
    :rtype: ClassificationHead
    """
    keras.utils.set_random_seed(3)
    config = dict(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        use_ffn=True,
        dropout_rate=0.1,
    )
    config.update(kwargs)
    return ClassificationHead(**config)


def _numpy(out: Dict[str, keras.KerasTensor]) -> Dict[str, np.ndarray]:
    return {k: keras.ops.convert_to_numpy(v) for k, v in out.items()}


def _softmax64(logits: np.ndarray) -> np.ndarray:
    """Reference softmax in float64 over the last axis.

    :param logits: Logits in any float dtype.
    :type logits: np.ndarray
    :return: float64 probabilities.
    :rtype: np.ndarray
    """
    z = logits.astype(np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _softmax_atol(dtype: np.dtype) -> float:
    """Derived absolute bound for a ``num_classes``-wide softmax in ``dtype``.

    ``reassociation_atol([C], 1, 1.0)`` is the float32 rounding bound of one
    pass over a ``C``-long reduction with outputs bounded by 1. The model scales
    linearly with the unit roundoff, so the bound for another dtype is that
    value times ``eps(dtype) / eps(float32)``.

    :param dtype: The compute dtype the softmax ran in.
    :type dtype: np.dtype
    :return: Absolute tolerance.
    :rtype: float
    """
    ratio = float(np.finfo(dtype).eps) / float(np.finfo(np.float32).eps)
    return reassociation_atol([NUM_CLASSES], 1, 1.0) * ratio


# ---------------------------------------------------------------------
# The classifier is linear and 'logits' are not normalized
# ---------------------------------------------------------------------

def test_the_classifier_has_no_activation() -> None:
    head = _head()
    assert head.classifier.activation is keras.activations.linear, (
        f"classifier.activation is {head.classifier.activation}; a "
        f"ClassificationHead classifier must be linear so 'logits' are logits."
    )


@pytest.mark.parametrize("variant", sorted(INPUT_VARIANTS))
def test_the_logits_are_not_normalized(variant: str) -> None:
    kwargs, shape = INPUT_VARIANTS[variant]
    head = _head(**kwargs)
    out = _numpy(head(_inputs(shape), training=False))
    logits = out["logits"].astype(np.float64)

    row_sums = logits.sum(axis=-1)
    # A softmax row sums to 1 within a few float32 ulps; logits do not.
    assert np.abs(row_sums - 1.0).max() > 1e3 * _softmax_atol(np.float32), (
        f"every 'logits' row sums to 1 (sums {row_sums}): the classifier is "
        f"normalizing, so 'logits' is a probability vector."
    )
    assert (logits < 0).any(), (
        f"no logit is negative (min {logits.min()}); a softmax output is never "
        f"negative, so this looks like probabilities under the 'logits' key."
    )


@pytest.mark.parametrize("variant", sorted(INPUT_VARIANTS))
def test_the_two_keys_are_different_tensors(variant: str) -> None:
    kwargs, shape = INPUT_VARIANTS[variant]
    head = _head(**kwargs)
    out = head(_inputs(shape), training=False)
    assert out["logits"] is not out["probabilities"], (
        "'logits' and 'probabilities' are the same tensor object."
    )


# ---------------------------------------------------------------------
# 'probabilities' == softmax('logits'), in every dtype policy
# ---------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(INPUT_VARIANTS))
def test_probabilities_are_the_softmax_of_the_logits(
        variant: str, dtype_policy: str
) -> None:
    kwargs, shape = INPUT_VARIANTS[variant]
    head = _head(**kwargs)
    out = _numpy(head(_inputs(shape), training=False))
    logits, probs = out["logits"], out["probabilities"]

    compute_dtype = np.dtype(keras.mixed_precision.global_policy().compute_dtype)
    assert logits.dtype == compute_dtype and probs.dtype == compute_dtype, (
        f"under {dtype_policy} expected both keys in {compute_dtype}, got "
        f"logits {logits.dtype}, probabilities {probs.dtype}."
    )
    assert np.isfinite(probs).all(), (
        f"non-finite probabilities under {dtype_policy}: {probs}"
    )

    atol = _softmax_atol(compute_dtype)
    row_sums = probs.astype(np.float64).sum(axis=-1)
    assert np.abs(row_sums - 1.0).max() <= atol, (
        f"under {dtype_policy} probability rows sum to {row_sums}, off 1 by "
        f"more than the derived {atol:.3e}."
    )
    # The reference is the float64 softmax of the head's own logits, i.e.
    # softmax(float32(logits)) evaluated without any further rounding.
    np.testing.assert_allclose(
        probs.astype(np.float64), _softmax64(logits), rtol=0, atol=atol,
        err_msg=f"'probabilities' is not softmax('logits') under {dtype_policy}",
    )


# ---------------------------------------------------------------------
# The head carries exactly the weights its call() runs
# ---------------------------------------------------------------------

def test_the_head_builds_no_norm_ffn_or_dropout() -> None:
    head = _head()  # use_ffn=True, dropout_rate=0.1: both ignored
    head(_inputs(FEATURE_SHAPE))
    present = [a for a in ("norm", "ffn", "dropout") if hasattr(head, a)]
    assert not present, (
        f"ClassificationHead built {present}; its call() applies none of them, "
        f"so they would be weights or state the forward pass never reaches."
    )


@pytest.mark.parametrize("use_attention", [False, True])
def test_every_trainable_weight_receives_a_gradient(use_attention: bool) -> None:
    # 'multi_head' attention takes a 3D sequence at width hidden_dim, so this
    # runs the CLS-token path on a (batch, seq_len, hidden_dim) input.
    head = _head(use_attention=use_attention, attention_type="multi_head",
                 pooling_type="cls_token", dropout_rate=0.0)
    x = keras.ops.convert_to_tensor(_inputs(SEQUENCE_SHAPE[:-1] + (HIDDEN_DIM,)))
    labels = keras.ops.convert_to_tensor(np.arange(SEQUENCE_SHAPE[0]) % NUM_CLASSES)
    with tf.GradientTape() as tape:
        out = head(x, training=True)
        loss = keras.ops.mean(keras.losses.sparse_categorical_crossentropy(
            labels, out["logits"], from_logits=True))
    weights = head.trainable_weights
    grads = tape.gradient(loss, weights)

    dead = [
        w.path for w, g in zip(weights, grads)
        if g is None or not np.any(keras.ops.convert_to_numpy(g))
    ]
    assert weights and not dead, (
        f"weights with no gradient from the logits: {dead}. The head carries "
        f"weights its call() never runs."
    )
    expected_owners = {"dense_block", "dense_block_1", "classifier"}
    if use_attention:
        expected_owners.add(head.attention.name)
    owners = {w.path.split("/")[1] for w in weights}
    assert owners == expected_owners, (
        f"weight-owning sublayers {sorted(owners)}, expected "
        f"{sorted(expected_owners)}."
    )


# ---------------------------------------------------------------------
# .keras round trip, by value, on both keys
# ---------------------------------------------------------------------

def test_a_model_holding_the_head_round_trips_by_value(tmp_path) -> None:
    head = _head()
    inp = keras.Input(shape=FEATURE_SHAPE[1:])
    model = keras.Model(inp, head(inp))
    x = _inputs(FEATURE_SHAPE)
    before = _numpy(model(x, training=False))

    path = tmp_path / "classification_head.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    after = _numpy(loaded(x, training=False))

    for w0, w1 in zip(model.weights, loaded.weights):
        assert w0.path == w1.path
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(w0), keras.ops.convert_to_numpy(w1))
    atol = _softmax_atol(np.float32)
    for key in ("logits", "probabilities"):
        np.testing.assert_allclose(
            after[key], before[key], rtol=0, atol=atol,
            err_msg=f"'{key}' changed across a .keras round trip",
        )
    assert np.abs(after["logits"] - after["probabilities"]).max() > atol, (
        "after reload the two keys coincide again"
    )


# ---------------------------------------------------------------------
# merge_head_config_overrides
# ---------------------------------------------------------------------

@pytest.mark.parametrize("key", RESERVED_HEAD_OVERRIDE_KEYS)
def test_the_override_merge_refuses_each_reserved_key(key: str) -> None:
    head_config = {"name": "classification_head", "num_classes": 10,
                   "hidden_dim": 16}
    before = dict(head_config)
    with pytest.raises(ValueError, match=key):
        merge_head_config_overrides(head_config, {key: "x", "hidden_dim": 32})
    assert head_config == before, "a refused merge mutated the head config"


def test_the_reserved_keys_are_exactly_the_model_owned_ones() -> None:
    assert set(RESERVED_HEAD_OVERRIDE_KEYS) == {"name", "num_classes", "task_type"}


def test_the_override_merge_applies_the_rest_without_mutating() -> None:
    head_config = {"name": "classification_head", "num_classes": 10,
                   "hidden_dim": 16, "dropout_rate": 0.1}
    before = dict(head_config)
    merged = merge_head_config_overrides(
        head_config, {"hidden_dim": 32, "use_attention": True})
    assert merged == {"name": "classification_head", "num_classes": 10,
                      "hidden_dim": 32, "dropout_rate": 0.1,
                      "use_attention": True}
    assert head_config == before, "the merge mutated its input"
    assert merged is not head_config


@pytest.mark.parametrize("overrides", [None, {}])
def test_no_overrides_returns_a_copy(overrides) -> None:
    head_config = {"name": "classification_head", "num_classes": 10}
    merged = merge_head_config_overrides(head_config, overrides)
    assert merged == head_config and merged is not head_config
