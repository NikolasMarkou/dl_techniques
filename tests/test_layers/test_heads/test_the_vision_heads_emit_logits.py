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

Step 6 (decisions D-003, D-020) extends the contract to the sibling heads. The
``SegmentationHead`` class map and the ``InstanceSegmentationHead`` masks are
logits (their ``seg_head``/``mask_head`` convs are linear); the one squashed
output kept is ``DepthEstimationHead``'s sigmoid, a range map, pinned here as
the documented exception. ``BaseVisionHead`` no longer builds a common norm or
Dropout (no head's ``call`` applied either, measured per head), and a head
refuses the common-layer flags its ``call`` never applies:
``ClassificationHead(use_ffn=True)`` and ``EnhancementHead(use_attention=True
or use_ffn=True)`` raise ``ValueError``.

Step 7.1 (decision D-030) applies the same rule to the two depth decoders. The
DPT decoder stored ``min_depth``, ``max_depth`` and ``use_log_depth`` and never
applied them (MEASURED at c39e444a3: the DPT depth was bit-identical with
``min_depth=3, max_depth=7, use_log_depth=False``), and the progressive
decoder ignored every ``dpt_*`` value the same way. Each mode now refuses the
other's knobs (the DPT mode also refuses the base ``normalization_type`` and
``activation_type``, which only the progressive ConvBlocks read), and the DPT
convs take ``dpt_kernel_initializer`` and
``dpt_kernel_regularizer``, which DepthAnything forwards from its own
``kernel_initializer``/``kernel_regularizer`` (they never reached the decoder:
its convs read ``self.kernel_initializer`` behind a ``hasattr`` that was always
false).
"""

from typing import Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.heads.vision.factory import (
    _DPT_DECODER_DEFAULTS,
    _PROGRESSIVE_DECODER_DEFAULTS,
    _PROGRESSIVE_ONLY_BASE_KNOBS,
    PRESET_FIXED_KEYS,
    RESERVED_HEAD_OVERRIDE_KEYS,
    ClassificationHead,
    DepthEstimationHead,
    DetectionHead,
    EnhancementHead,
    HeadConfiguration,
    InstanceSegmentationHead,
    MultiTaskHead,
    SegmentationHead,
    create_vision_head,
    merge_head_config_overrides,
)
from dl_techniques.layers.heads.vision.task_types import VisionTaskType
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
    """Build a small head, with a block dropout so a common Dropout would show.

    :return: The head.
    :rtype: ClassificationHead
    """
    keras.utils.set_random_seed(3)
    config = dict(
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
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
    head = _head()  # dropout_rate=0.1 reaches the blocks only
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


# ---------------------------------------------------------------------
# Step 6 (D-003, D-020): the sibling heads emit logits, the one range map
# stays, and no head builds a layer its call() skips
# ---------------------------------------------------------------------

def _class_logit_check(values: np.ndarray, what: str) -> None:
    """A per-pixel class map is logits: some value is negative, and the
    channel rows do not all sum to 1 (a softmax row does, within rounding).

    :param values: Output, channels last.
    :type values: np.ndarray
    :param what: Name for the message.
    :type what: str
    """
    values = values.astype(np.float64)
    assert np.isfinite(values).all(), f"{what}: non-finite values"
    assert (values < 0).any(), (
        f"{what}: no value is negative (min {values.min()}); a softmax or a "
        f"sigmoid output never is, so this looks activated"
    )
    rows = values.sum(axis=-1)
    atol = reassociation_atol([values.shape[-1]], 1, 1.0)
    assert np.abs(rows - 1.0).max() > 1e3 * atol, (
        f"{what}: every channel row sums to 1 within {1e3 * atol:.2e}; the "
        f"map is normalized"
    )


def test_the_segmentation_class_map_is_logits() -> None:
    keras.utils.set_random_seed(3)
    head = SegmentationHead(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM)
    assert head.seg_head.activation is keras.activations.linear, (
        f"seg_head.activation is {head.seg_head.activation}; the class map "
        f"must be logits"
    )
    out = keras.ops.convert_to_numpy(head(_inputs((2, 6, 10, HIDDEN_DIM))))
    assert out.shape == (2, 24, 40, NUM_CLASSES)
    _class_logit_check(out, "SegmentationHead")


def test_the_instance_masks_are_logits() -> None:
    keras.utils.set_random_seed(3)
    head = InstanceSegmentationHead(num_classes=3, num_instances=4,
                                    hidden_dim=HIDDEN_DIM)
    assert head.mask_head.activation is keras.activations.linear, (
        f"mask_head.activation is {head.mask_head.activation}; the masks "
        f"must be logits"
    )
    out = head(_inputs((2, 6, 10, HIDDEN_DIM)))
    masks = keras.ops.convert_to_numpy(out["instance_masks"]).astype(np.float64)
    assert masks.shape == (2, 6, 10, 4)
    # Per-instance sigmoid masks, not a distribution over instances: the
    # check is that the values leave [0, 1] on both sides.
    assert masks.min() < 0.0 and masks.max() > 1.0, (
        f"instance_masks span [{masks.min()}, {masks.max()}], inside [0, 1]: "
        f"they look sigmoid'ed"
    )
    _class_logit_check(keras.ops.convert_to_numpy(out["classifications"]),
                       "InstanceSegmentationHead classifications")


@pytest.mark.parametrize("use_log_depth", [True, False])
def test_the_depth_head_keeps_its_sigmoid_range_map(use_log_depth: bool) -> None:
    """The documented exception: the depth map is a range map, squashed to
    [0, 1] by a sigmoid and scaled into [min_depth, max_depth]. Pinned so the
    logits rule above is not 'applied' to it by mistake."""
    keras.utils.set_random_seed(3)
    head = DepthEstimationHead(hidden_dim=HIDDEN_DIM, min_depth=0.5,
                               max_depth=20.0, use_log_depth=use_log_depth)
    assert head.depth_head.activation is keras.activations.sigmoid
    out = head(_inputs((2, 4, 6, HIDDEN_DIM)))
    confidence = keras.ops.convert_to_numpy(out["confidence"]).astype(np.float64)
    depth = keras.ops.convert_to_numpy(out["depth"]).astype(np.float64)
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0
    assert confidence.std() > 0.0, "degenerate probe: a constant depth map"
    if use_log_depth:
        expected = np.exp(confidence * (np.log(20.0) - np.log(0.5)) + np.log(0.5))
    else:
        expected = confidence * (20.0 - 0.5) + 0.5
    np.testing.assert_allclose(
        depth, expected, rtol=0,
        atol=reassociation_atol([2], 1, float(np.abs(expected).max())))


#: A non-None value for every mode-specific DepthEstimationHead knob. Pinned
#: here rather than read off the defaults dicts, so a key dropped from a dict
#: (and with it its refusal) fails the key-set check below.
_PROGRESSIVE_ONLY_VALUES = {
    "min_depth": 0.5, "max_depth": 20.0, "use_log_depth": False,
}
#: The base-class arguments only the progressive ConvBlocks read.
_PROGRESSIVE_ONLY_BASE_VALUES = {
    "normalization_type": "batch_norm", "activation_type": "relu",
}
_DPT_ONLY_VALUES = {
    "dpt_dims": [8], "dpt_use_bias": False, "dpt_activation": "relu",
    "dpt_output_activation": "linear", "dpt_upsample_factor": 2,
    "dpt_kernel_initializer": "zeros", "dpt_kernel_regularizer": "l2",
}


def test_the_depth_knob_sets_are_the_pinned_ones() -> None:
    assert set(_PROGRESSIVE_DECODER_DEFAULTS) == set(_PROGRESSIVE_ONLY_VALUES)
    assert set(_DPT_DECODER_DEFAULTS) == set(_DPT_ONLY_VALUES)
    assert set(_PROGRESSIVE_ONLY_BASE_KNOBS) == set(_PROGRESSIVE_ONLY_BASE_VALUES)


@pytest.mark.parametrize(
    "style,key,value",
    [("dpt", k, v) for k, v in _PROGRESSIVE_ONLY_VALUES.items()]
    + [("dpt", k, v) for k, v in _PROGRESSIVE_ONLY_BASE_VALUES.items()]
    + [("progressive", k, v) for k, v in _DPT_ONLY_VALUES.items()],
    ids=lambda v: str(v))
def test_each_depth_decoder_refuses_the_other_decoders_knobs(
        style: str, key: str, value) -> None:
    """A value equal to the other mode's default is refused too: the mode in
    hand never reads it, whatever it is."""
    with pytest.raises(ValueError, match=key):
        DepthEstimationHead(decoder_style=style, **{key: value})


@pytest.mark.parametrize("style", ["progressive", "dpt"])
def test_each_depth_decoder_stores_only_its_own_knobs(style: str) -> None:
    own, other = ((_PROGRESSIVE_DECODER_DEFAULTS, _DPT_DECODER_DEFAULTS)
                  if style == "progressive"
                  else (_DPT_DECODER_DEFAULTS, _PROGRESSIVE_DECODER_DEFAULTS))
    head = DepthEstimationHead(decoder_style=style)
    config = head.get_config()
    for key in other:
        assert getattr(head, key) is None and config[key] is None, key
    for key in _PROGRESSIVE_ONLY_BASE_KNOBS:
        assert (config[key] is None) == (style == "dpt"), key
    for key in own:
        assert config[key] is not None or own[key] is None, key
    # The DPT defaults are the documented ones, and a config round-trips.
    if style == "dpt":
        assert head.dpt_dims == [256, 128, 64, 32]
        assert head.dpt_upsample_factor == 16
        assert config["dpt_kernel_initializer"] == "he_normal"
    else:
        assert (head.min_depth, head.max_depth, head.use_log_depth) == (
            0.1, 100.0, True)
    rebuilt = DepthEstimationHead.from_config(config)
    assert rebuilt.get_config() == config


def test_the_dpt_activations_serialize_when_given_as_callables() -> None:
    """``get_config`` serializes both DPT activations; a raw callable in the
    config is not JSON and fails at load when unregistered."""
    import json
    head = DepthEstimationHead(
        decoder_style="dpt", dpt_dims=[4], dpt_upsample_factor=2,
        dpt_activation=keras.activations.gelu,
        dpt_output_activation=keras.activations.softplus)
    config = json.loads(json.dumps(head.get_config()))
    rebuilt = DepthEstimationHead.from_config(config)
    x = _inputs((2, 3, 5, 4))
    head(x)
    rebuilt(x)
    rebuilt.set_weights(head.get_weights())
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(rebuilt(x)["depth"]),
        keras.ops.convert_to_numpy(head(x)["depth"]))


def test_the_dpt_kernel_initializer_and_regularizer_reach_every_conv() -> None:
    """Every DPT conv kernel (the stages and the output conv) takes the
    initializer and the regularizer; a seedless initializer is cloned per
    conv, so two same-shaped kernels do not replay one draw."""
    dims = [4, 4, 4]
    head = DepthEstimationHead(
        decoder_style="dpt", dpt_dims=dims, dpt_upsample_factor=2,
        dpt_kernel_initializer=keras.initializers.Constant(0.25),
        dpt_kernel_regularizer=keras.regularizers.L2(1e-3))
    head(_inputs((2, 3, 5, 4)))
    convs = head.dpt_conv_layers + [head.dpt_output_conv]
    for conv in convs:
        kernel = keras.ops.convert_to_numpy(conv.kernel)
        assert np.all(kernel == 0.25), conv.name
    assert len(head.losses) == len(convs)
    config = head.get_config()
    assert config["dpt_kernel_initializer"]["class_name"] == "Constant"
    assert config["dpt_kernel_regularizer"]["class_name"] == "L2"

    keras.utils.set_random_seed(3)
    seeded = DepthEstimationHead(
        decoder_style="dpt", dpt_dims=dims, dpt_upsample_factor=2,
        dpt_kernel_initializer=keras.initializers.HeNormal())
    seeded(_inputs((2, 3, 5, 4)))
    k1, k2 = (keras.ops.convert_to_numpy(c.kernel)
              for c in seeded.dpt_conv_layers[1:3])
    assert k1.shape == k2.shape and not np.array_equal(k1, k2)


def test_classification_head_refuses_use_ffn() -> None:
    with pytest.raises(ValueError, match="use_ffn"):
        ClassificationHead(num_classes=NUM_CLASSES, use_ffn=True)
    assert ClassificationHead(num_classes=NUM_CLASSES).use_ffn is False
    assert ClassificationHead(num_classes=NUM_CLASSES).get_config()["use_ffn"] is False


@pytest.mark.parametrize("flag,on,off", [("use_attention", True, False),
                                          ("use_ffn", True, False),
                                          ("dropout_rate", 0.1, 0.0)])
def test_enhancement_head_refuses_the_layers_it_never_runs(flag, on, off) -> None:
    with pytest.raises(ValueError, match=flag):
        EnhancementHead(output_channels=3, hidden_dim=HIDDEN_DIM, **{flag: on})
    head = EnhancementHead(output_channels=3, hidden_dim=HIDDEN_DIM)
    assert getattr(head, flag) == off and type(getattr(head, flag)) is type(off)
    assert head.get_config()[flag] == off


PRESETS = {
    "default": HeadConfiguration.get_default_config,
    "efficient": HeadConfiguration.get_efficient_config,
}

#: Every task type create_vision_head dispatches, by the head it builds.
PRESET_TASKS = [
    VisionTaskType.DETECTION, VisionTaskType.SEGMENTATION,
    VisionTaskType.DEPTH_ESTIMATION, VisionTaskType.CLASSIFICATION,
    VisionTaskType.INSTANCE_SEGMENTATION, VisionTaskType.DENOISING,
    VisionTaskType.SUPER_RESOLUTION,
]


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("task", PRESET_TASKS, ids=lambda t: t.value)
def test_every_preset_builds_and_runs_its_head(preset: str, task) -> None:
    """A preset must never hand a head a flag it refuses, or a layer that
    cannot run on the head's 4D input. Before step 6 the default
    classification preset named 'multi_head' attention (a (B, N, D) layer)
    and raised on the 4D map its own 'avg' pooling expects, and a third
    preset raised for every task."""
    config = PRESETS[preset](task)
    for key, value in PRESET_FIXED_KEYS.get(task, {}).items():
        assert config[key] == value, (preset, task, key)
    # The presets' widths are ImageNet-sized; the input carries hidden_dim
    # channels so a 'cbam' attention sees its own width.
    config["hidden_dim"] = HIDDEN_DIM
    keras.utils.set_random_seed(3)
    head = create_vision_head(task, **config)
    for flag in ("use_attention", "use_ffn"):
        if not getattr(head, flag):
            assert not hasattr(head, flag[4:]), (
                f"{type(head).__name__} built {flag[4:]} with {flag}=False")
    head(_inputs((2, 4, 6, HIDDEN_DIM)))


def test_the_high_performance_preset_is_gone() -> None:
    assert not hasattr(HeadConfiguration, "get_high_performance_config")


def test_a_multi_task_classification_entry_builds_without_an_ffn() -> None:
    keras.utils.set_random_seed(3)
    head = MultiTaskHead(
        {"cls": {"task_type": VisionTaskType.CLASSIFICATION,
                 "num_classes": NUM_CLASSES, "block_dims": ()}},
        shared_backbone_dim=HIDDEN_DIM, use_task_specific_attention=False)
    out = head(_inputs(FEATURE_SHAPE))
    assert not hasattr(head.task_heads["cls"], "ffn")
    assert out["cls"]["logits"].shape == (FEATURE_SHAPE[0], NUM_CLASSES)


#: Every head class, built with every common flag it accepts switched on, so
#: a common norm or Dropout would be created if the base class still built
#: one. Inputs are channels-last at hidden_dim width.
ALL_HEADS = {
    "DetectionHead": lambda: DetectionHead(
        num_classes=3, num_anchors=2, hidden_dim=HIDDEN_DIM, dropout_rate=0.1,
        use_attention=True, attention_type="cbam", use_ffn=True),
    "SegmentationHead": lambda: SegmentationHead(
        num_classes=3, hidden_dim=HIDDEN_DIM, dropout_rate=0.1,
        use_attention=True, attention_type="cbam", use_ffn=True),
    "DepthEstimationHead": lambda: DepthEstimationHead(
        hidden_dim=HIDDEN_DIM, dropout_rate=0.1, use_attention=True,
        attention_type="cbam", use_ffn=True),
    "ClassificationHead": lambda: ClassificationHead(
        num_classes=3, hidden_dim=HIDDEN_DIM, dropout_rate=0.1,
        use_attention=True, attention_type="cbam"),
    "InstanceSegmentationHead": lambda: InstanceSegmentationHead(
        num_classes=3, num_instances=4, hidden_dim=HIDDEN_DIM, dropout_rate=0.1,
        use_attention=True, attention_type="cbam", use_ffn=True),
    # Refuses attention, an FFN and a dropout (none runs in its call()).
    "EnhancementHead": lambda: EnhancementHead(
        output_channels=3, hidden_dim=HIDDEN_DIM),
}


@pytest.mark.parametrize("name", sorted(ALL_HEADS))
def test_no_vision_head_owns_a_common_norm_or_dropout(name: str) -> None:
    keras.utils.set_random_seed(3)
    head = ALL_HEADS[name]()
    head(_inputs((2, 6, 10, HIDDEN_DIM)), training=True)
    present = [a for a in ("norm", "dropout") if hasattr(head, a)]
    assert not present, (
        f"{name} owns {present}; no vision head's call() applies a common "
        f"norm or Dropout"
    )
    stray = [w.path for w in head.weights if f"{head.name}_norm/" in w.path]
    assert not stray, f"{name} carries common-norm weights {stray}"


@pytest.mark.parametrize("name", sorted(ALL_HEADS))
def test_every_built_head_weight_receives_a_gradient(name: str) -> None:
    """The head carries only weights its call() runs."""
    keras.utils.set_random_seed(3)
    head = ALL_HEADS[name]()
    x = keras.ops.convert_to_tensor(_inputs((2, 6, 10, HIDDEN_DIM)))
    with tf.GradientTape() as tape:
        out = head(x, training=True)
        leaves = out.values() if isinstance(out, dict) else [out]
        loss = sum(keras.ops.mean(keras.ops.square(v)) for v in leaves)
    weights = head.trainable_weights
    grads = tape.gradient(loss, weights)
    dead = [w.path for w, g in zip(weights, grads)
            if g is None or not np.any(keras.ops.convert_to_numpy(g))]
    assert weights and not dead, f"{name}: weights with no gradient {dead}"


@pytest.mark.parametrize("name", sorted(ALL_HEADS))
def test_two_instances_of_a_head_share_every_weight_path(name: str) -> None:
    """Every sublayer is named, so a head's weight paths do not depend on how
    many layers the process built before it. Before step 6
    ``EnhancementHead``'s blocks and output conv and
    ``InstanceSegmentationHead``'s mask blocks and inner detection head took
    Keras' process-global auto suffix (``conv_block_102``), which made two
    instances of DarkIR, SCUNet and PFT-SR disagree by weight path."""
    def relative_paths():
        keras.utils.set_random_seed(3)
        head = ALL_HEADS[name]()
        head(_inputs((2, 6, 10, HIDDEN_DIM)))
        # The head's own name is the caller's to fix (every model passes one);
        # the attention and FFN derive theirs from it. Everything below it
        # must be the same in both instances.
        return sorted(w.path.replace(head.name, "<head>") for w in head.weights)
    first, second = relative_paths(), relative_paths()
    assert first == second, sorted(set(first) ^ set(second))[:6]
