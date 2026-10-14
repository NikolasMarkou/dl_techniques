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


# ---------------------------------------------------------------------
# D-013: the head can express a reference head, and the model's knobs reach it
# ---------------------------------------------------------------------
#
# Added at plan-2026-10-14T062616-81938e81 step 3. Before it the head always
# built two norm'd DenseBlocks and gave every Dense the Keras defaults, so a
# consumer could not be GAP + FC (ResNet), GAP + LayerNorm + FC (ConvNeXt) or
# 4096-4096 without a norm (AlexNet), and its kernel_initializer and
# kernel_regularizer stopped at the backbone.

def _truncated_normal_std_factor() -> float:
    """Realized std of a unit normal truncated at +-2 sigma, from the closed form.

    ``sqrt(1 - 2 * a * phi(a) / (2 * Phi(a) - 1))`` at ``a = 2``. Keras'
    ``TruncatedNormal`` resamples outside two standard deviations, so its draws
    have this fraction of the nominal ``stddev``.

    :return: About 0.87962.
    :rtype: float
    """
    import math
    a = 2.0
    pdf = math.exp(-a * a / 2.0) / math.sqrt(2.0 * math.pi)
    mass = math.erf(a / math.sqrt(2.0))
    return math.sqrt(1.0 - 2.0 * a * pdf / mass)


def _owners(head: ClassificationHead) -> set:
    return {w.path.split("/")[1] for w in head.weights}


def test_empty_block_dims_leaves_pool_and_classifier_only() -> None:
    head = _head(block_dims=())
    out = head(_inputs(FEATURE_SHAPE))
    assert head.dense_blocks == [], f"built {len(head.dense_blocks)} blocks"
    assert _owners(head) == {"classifier"}, (
        f"block_dims=() should leave the classifier as the only weight owner, "
        f"got {sorted(_owners(head))}"
    )
    channels = FEATURE_SHAPE[-1]
    assert [tuple(w.shape) for w in head.weights] == [
        (channels, NUM_CLASSES), (NUM_CLASSES,)
    ], "the classifier does not read the pooled channels directly"
    assert tuple(out["logits"].shape) == (FEATURE_SHAPE[0], NUM_CLASSES)


def test_block_dims_sets_every_width_and_none_keeps_the_two_block_default() -> None:
    explicit = _head(block_dims=(10, 6, 4), normalization_type=None)
    explicit(_inputs(FEATURE_SHAPE))
    assert [b.units for b in explicit.dense_blocks] == [10, 6, 4]
    assert [b.name for b in explicit.dense_blocks] == [
        "dense_block", "dense_block_1", "dense_block_2"
    ]
    # normalization_type=None: no block owns a norm, so no norm weight exists.
    assert all(b.norm is None for b in explicit.dense_blocks)
    assert not any("norm" in w.path for w in explicit.weights), [
        w.path for w in explicit.weights if "norm" in w.path
    ]
    assert tuple(explicit.classifier.kernel.shape) == (4, NUM_CLASSES)

    default = _head()
    assert [b.units for b in default.dense_blocks] == [HIDDEN_DIM, HIDDEN_DIM // 2]


@pytest.mark.parametrize("bad", [(0,), (8, -1), (2.5,), (True,)])
def test_block_dims_refuses_a_width_that_is_not_a_positive_int(bad) -> None:
    with pytest.raises(ValueError, match="block_dims"):
        _head(block_dims=bad)


@pytest.mark.usefixtures("tf32_disabled")
def test_the_pooled_norm_is_a_layer_norm_of_the_pooled_vector() -> None:
    """GAP, LayerNorm, linear classifier, against a float64 numpy oracle."""
    head = _head(block_dims=(), pooled_normalization_type="layer_norm")
    x = _inputs(FEATURE_SHAPE)
    logits = keras.ops.convert_to_numpy(head(x, training=False)["logits"])
    assert _owners(head) == {"pooled_norm", "classifier"}

    norm = head.pooled_norm
    pooled = x.astype(np.float64).mean(axis=(1, 2))
    mean = pooled.mean(axis=-1, keepdims=True)
    var = pooled.var(axis=-1, keepdims=True)
    gamma = keras.ops.convert_to_numpy(norm.gamma).astype(np.float64)
    beta = keras.ops.convert_to_numpy(norm.beta).astype(np.float64)
    normed = (pooled - mean) / np.sqrt(var + norm.epsilon) * gamma + beta
    kernel = keras.ops.convert_to_numpy(head.classifier.kernel).astype(np.float64)
    bias = keras.ops.convert_to_numpy(head.classifier.bias).astype(np.float64)
    expected = normed @ kernel + bias

    height, width, channels = FEATURE_SHAPE[1:]
    atol = reassociation_atol(
        [height * width, channels, channels], 1, float(np.abs(expected).max()))
    np.testing.assert_allclose(logits, expected, rtol=0, atol=atol)


def test_the_kernel_and_bias_initializers_reach_every_dense() -> None:
    nominal = 5.0  # glorot at these fans draws near 0.2-0.3: 15-20x away
    head = _head(
        block_dims=(32,),
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=nominal),
        bias_initializer=keras.initializers.Constant(0.25),
    )
    head(_inputs(FEATURE_SHAPE))
    expected_std = nominal * _truncated_normal_std_factor()
    kernels = [w for w in head.weights if w.path.endswith("kernel")]
    biases = [w for w in head.weights if w.path.endswith("bias")]
    assert {w.path.split("/")[1] for w in kernels} == {"dense_block", "classifier"}
    for w in kernels:
        values = keras.ops.convert_to_numpy(w).ravel().astype(np.float64)
        # 6 standard errors of a sample std: se(std)/std ~ 1/sqrt(2n).
        rel = 6.0 / np.sqrt(2.0 * values.size)
        assert values.std() == pytest.approx(expected_std, rel=rel), (
            f"{w.path} drew at std {values.std():.4f}; the head's "
            f"kernel_initializer realizes {expected_std:.4f}"
        )
    for w in biases:
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(w), np.full(w.shape, 0.25, "float32"),
            err_msg=f"{w.path} did not take the head's bias_initializer")


def test_no_two_dense_kernels_replay_one_draw() -> None:
    """A seedless initializer INSTANCE replays its sample at every site."""
    head = _head(block_dims=(12,),
                 kernel_initializer=keras.initializers.RandomUniform(-1.0, 1.0))
    head(_inputs(FEATURE_SHAPE))
    block = keras.ops.convert_to_numpy(
        head.dense_blocks[0].dense.kernel).ravel()
    classifier = keras.ops.convert_to_numpy(head.classifier.kernel).ravel()
    n = min(block.size, classifier.size)
    assert not np.array_equal(block[:n], classifier[:n]), (
        "the classifier kernel is the block kernel's prefix: one initializer "
        "instance was shared instead of cloned"
    )


def test_the_regularizer_adds_one_loss_per_dense() -> None:
    l2 = 0.01
    head = _head(block_dims=(8, 4), kernel_regularizer=keras.regularizers.L2(l2))
    head(_inputs(FEATURE_SHAPE))
    kernels = [w for w in head.weights if w.path.endswith("kernel")]
    assert len(kernels) == 3
    losses = sorted(float(keras.ops.convert_to_numpy(v)) for v in head.losses)
    expected = sorted(
        l2 * float(np.sum(np.square(keras.ops.convert_to_numpy(k).astype(np.float64))))
        for k in kernels
    )
    assert len(losses) == 3, f"{len(losses)} regularization losses for 3 Dense kernels"
    np.testing.assert_allclose(losses, expected, rtol=1e-5)


def test_the_new_options_round_trip_by_value(tmp_path) -> None:
    head = _head(
        block_dims=(9,),
        normalization_type=None,
        pooled_normalization_type="layer_norm",
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        bias_initializer=keras.initializers.Constant(0.1),
        kernel_regularizer=keras.regularizers.L2(1e-3),
    )
    config = head.get_config()
    assert config["block_dims"] == [9]
    assert config["pooled_normalization_type"] == "layer_norm"
    assert config["kernel_initializer"]["config"]["stddev"] == 0.02
    assert config["bias_initializer"]["config"]["value"] == 0.1
    assert config["kernel_regularizer"]["config"]["l2"] == pytest.approx(1e-3)

    inp = keras.Input(shape=FEATURE_SHAPE[1:])
    model = keras.Model(inp, head(inp)["logits"])
    x = _inputs(FEATURE_SHAPE)
    before = keras.ops.convert_to_numpy(model(x, training=False))
    path = tmp_path / "head_d013.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    reloaded_head = loaded.get_layer(head.name)
    assert reloaded_head.block_dims == (9,)
    assert len(reloaded_head.losses) == 2, "the regularizer did not survive reload"
    for w0, w1 in zip(model.weights, loaded.weights):
        assert w0.path == w1.path
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(w0), keras.ops.convert_to_numpy(w1))
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(loaded(x, training=False)), before)


# ---------------------------------------------------------------------
# D-015: a dropout before the classifier, norm arguments, bias regularizer
# ---------------------------------------------------------------------

def test_the_classifier_dropout_is_built_only_when_asked_and_runs_in_training() -> None:
    assert not hasattr(_head(block_dims=()), "classifier_dropout")

    head = _head(block_dims=(), classifier_dropout_rate=0.5)
    x = _inputs(FEATURE_SHAPE)
    inference = keras.ops.convert_to_numpy(head(x, training=False)["logits"])
    pooled = x.astype(np.float64).mean(axis=(1, 2))
    kernel = keras.ops.convert_to_numpy(head.classifier.kernel).astype(np.float64)
    bias = keras.ops.convert_to_numpy(head.classifier.bias).astype(np.float64)
    # Inference: the dropout is the identity, so logits are pool @ W + b.
    expected = pooled @ kernel + bias
    atol = reassociation_atol(
        [FEATURE_SHAPE[1] * FEATURE_SHAPE[2], FEATURE_SHAPE[3]], 1,
        float(np.abs(expected).max()))
    np.testing.assert_allclose(inference, expected, rtol=0, atol=atol)

    # Training: the classifier reads a dropped-out vector, so the logits move.
    training = keras.ops.convert_to_numpy(head(x, training=True)["logits"])
    assert np.abs(training - inference).max() > 1e3 * atol, (
        "classifier_dropout_rate=0.5 changed nothing in training mode: the "
        "dropout is not on the classifier's input"
    )


@pytest.mark.parametrize("bad", [-0.1, 1.0])
def test_the_classifier_dropout_rate_is_range_checked(bad: float) -> None:
    with pytest.raises(ValueError, match="classifier_dropout_rate"):
        _head(classifier_dropout_rate=bad)


def test_normalization_kwargs_reach_every_head_norm() -> None:
    head = _head(block_dims=(10, 6), normalization_type="batch_norm",
                 pooled_normalization_type="layer_norm",
                 normalization_kwargs={"epsilon": 1e-3})
    head(_inputs(FEATURE_SHAPE))
    norms = [head.pooled_norm] + [b.norm for b in head.dense_blocks]
    assert [n.epsilon for n in norms] == [1e-3, 1e-3, 1e-3], (
        f"head norm epsilons {[n.epsilon for n in norms]}; the factory default "
        f"is 1e-6, so a norm did not take normalization_kwargs"
    )


def test_the_bias_regularizer_adds_one_loss_per_dense() -> None:
    l2 = 0.01
    head = _head(block_dims=(8,), bias_initializer=keras.initializers.Constant(0.3),
                 bias_regularizer=keras.regularizers.L2(l2))
    head(_inputs(FEATURE_SHAPE))
    biases = [w for w in head.weights if w.path.endswith("bias")]
    assert {w.path.split("/")[1] for w in biases} == {"dense_block", "classifier"}
    losses = sorted(float(keras.ops.convert_to_numpy(v)) for v in head.losses)
    expected = sorted(
        l2 * float(np.sum(np.square(keras.ops.convert_to_numpy(b).astype(np.float64))))
        for b in biases
    )
    assert len(losses) == 2, f"{len(losses)} bias losses for 2 Dense biases"
    np.testing.assert_allclose(losses, expected, rtol=1e-5)


def test_the_d015_options_round_trip() -> None:
    head = _head(block_dims=(9,), normalization_type="batch_norm",
                 normalization_kwargs={"epsilon": 1e-3, "momentum": 0.9},
                 classifier_dropout_rate=0.2,
                 bias_regularizer=keras.regularizers.L2(1e-3))
    config = head.get_config()
    assert config["normalization_kwargs"] == {"epsilon": 1e-3, "momentum": 0.9}
    assert config["classifier_dropout_rate"] == 0.2
    assert config["bias_regularizer"]["config"]["l2"] == pytest.approx(1e-3)
    clone = ClassificationHead.from_config(config)
    clone(_inputs(FEATURE_SHAPE))
    assert clone.classifier_dropout.rate == 0.2
    assert clone.dense_blocks[0].norm.epsilon == 1e-3
    assert clone.dense_blocks[0].norm.momentum == 0.9
    assert len(clone.losses) == 2


def _block_bias_gradient(block_use_bias: bool):
    head = _head(block_dims=(8,), normalization_type="batch_norm",
                 dropout_rate=0.0, block_use_bias=block_use_bias)
    x = keras.ops.convert_to_tensor(_inputs(FEATURE_SHAPE))
    labels = keras.ops.convert_to_tensor(np.arange(FEATURE_SHAPE[0]) % NUM_CLASSES)
    with tf.GradientTape() as tape:
        out = head(x, training=True)
        loss = keras.ops.mean(keras.losses.sparse_categorical_crossentropy(
            labels, out["logits"], from_logits=True))
    weights = head.trainable_weights
    return head, dict(zip([w.path for w in weights], tape.gradient(loss, weights)))


def test_a_bias_before_batch_norm_never_trains() -> None:
    """Why block_use_bias exists: BatchNorm subtracts the batch mean per
    feature, so a Dense bias in front of it gets a zero gradient.

    The bias gradient is the batch sum of the per-sample gradients at the
    Dense output ``z``, and BatchNorm's backward makes that sum exactly 0. In
    float32 it is 0 up to the rounding of a ``batch``-long sum, bounded by
    ``reassociation_atol([batch], 1, max|dL/dz|)``. The per-sample gradients
    themselves are far from 0, so the zero is the norm's doing.
    """
    head, _ = _block_bias_gradient(block_use_bias=True)
    block = head.dense_blocks[0]
    x = keras.ops.convert_to_tensor(_inputs(FEATURE_SHAPE))
    labels = keras.ops.convert_to_tensor(np.arange(FEATURE_SHAPE[0]) % NUM_CLASSES)
    z = block.dense(head.pooling(x))
    with tf.GradientTape() as tape:
        tape.watch(z)
        h = block.activation(block.norm(z, training=True))
        loss = keras.ops.mean(keras.losses.sparse_categorical_crossentropy(
            labels, head.classifier(h), from_logits=True))
    per_sample = keras.ops.convert_to_numpy(tape.gradient(loss, z)).astype(np.float64)
    scale = float(np.abs(per_sample).max())
    assert scale > 1e-3, f"degenerate probe: max|dL/dz| = {scale}"
    batch_sum = np.abs(per_sample.sum(axis=0)).max()
    bound = reassociation_atol([FEATURE_SHAPE[0]], 1, scale)
    assert batch_sum <= bound, (
        f"the bias gradient {batch_sum:.3e} exceeds the rounding bound "
        f"{bound:.3e}: BatchNorm is not removing the batch mean"
    )


def test_block_use_bias_false_builds_a_bias_free_hidden_dense() -> None:
    head, grads = _block_bias_gradient(block_use_bias=False)
    assert head.dense_blocks[0].dense.bias is None
    assert head.classifier.bias is not None, "the classifier keeps its bias"
    dead = [p for p, g in grads.items()
            if g is None or not np.any(keras.ops.convert_to_numpy(g))]
    assert not dead, f"weights with a dead gradient: {dead}"
    assert head.get_config()["block_use_bias"] is False
    clone = ClassificationHead.from_config(head.get_config())
    assert clone.dense_blocks[0].dense.use_bias is False
