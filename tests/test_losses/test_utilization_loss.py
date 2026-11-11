"""Tests for MANNUtilizationLoss and GNNUtilizationLoss."""

import keras
from keras import ops
import pytest
import numpy as np
import tensorflow as tf

from dl_techniques.losses.utilization_loss import (
    MANNUtilizationLoss,
    GNNUtilizationLoss,
)
from tests.numerics import reassociation_atol


class TestMANNUtilizationLoss:
    """Tests for MANNUtilizationLoss."""

    def test_init_defaults(self):
        loss = MANNUtilizationLoss()
        assert loss.name == "mann_utilization_loss"
        assert loss.entropy_weight == 0.01
        assert loss.write_weight == 0.01
        assert loss.variance_weight == 0.01

    def test_init_custom(self):
        loss = MANNUtilizationLoss(
            entropy_weight=0.1, write_weight=0.05, variance_weight=0.02
        )
        assert loss.entropy_weight == 0.1
        assert loss.write_weight == 0.05
        assert loss.variance_weight == 0.02

    def test_forward_pass(self):
        """y_pred is treated as the memory vectors tensor."""
        loss_fn = MANNUtilizationLoss()
        batch_size, time_steps, memory_dim = 4, 10, 32
        y_true = np.zeros((batch_size, time_steps, memory_dim), dtype=np.float32)
        memory_vectors = np.random.rand(
            batch_size, time_steps, memory_dim
        ).astype(np.float32)
        loss = loss_fn(y_true, memory_vectors)
        assert np.isfinite(float(loss))

    def test_collapsed_memory_higher_loss(self):
        """Collapsed (constant) memory should produce higher loss than diverse memory."""
        loss_fn = MANNUtilizationLoss(
            entropy_weight=1.0, write_weight=1.0, variance_weight=1.0
        )
        batch_size, time_steps, memory_dim = 4, 10, 16

        # Collapsed: near-constant memory
        collapsed = np.ones(
            (batch_size, time_steps, memory_dim), dtype=np.float32
        ) * 0.01 + np.random.rand(batch_size, time_steps, memory_dim).astype(np.float32) * 0.001

        # Diverse: varied memory patterns
        diverse = np.random.rand(
            batch_size, time_steps, memory_dim
        ).astype(np.float32) * 5.0

        dummy = np.zeros((batch_size, time_steps, memory_dim), dtype=np.float32)
        loss_collapsed = float(loss_fn(dummy, collapsed))
        loss_diverse = float(loss_fn(dummy, diverse))

        # Diverse memory should have lower (more negative) utilization loss
        assert loss_diverse < loss_collapsed

    def test_get_config_roundtrip(self):
        loss = MANNUtilizationLoss(
            entropy_weight=0.1, write_weight=0.05, variance_weight=0.02
        )
        config = loss.get_config()
        restored = MANNUtilizationLoss.from_config(config)
        assert restored.entropy_weight == 0.1
        assert restored.write_weight == 0.05
        assert restored.variance_weight == 0.02

    def test_serialization(self):
        loss = MANNUtilizationLoss(entropy_weight=0.5)
        config = keras.saving.serialize_keras_object(loss)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, MANNUtilizationLoss)
        assert restored.entropy_weight == 0.5


class TestGNNUtilizationLoss:
    """Tests for GNNUtilizationLoss."""

    def test_init_defaults(self):
        loss = GNNUtilizationLoss()
        assert loss.name == "gnn_utilization_loss"
        assert loss.diversity_weight == 0.01
        assert loss.attention_weight == 0.01
        assert loss.activation_weight == 0.01

    def test_forward_pass(self):
        """y_pred is treated as the entity embeddings tensor."""
        loss_fn = GNNUtilizationLoss()
        batch_size, num_entities, embed_dim = 4, 8, 32
        y_true = np.zeros((batch_size, num_entities, embed_dim), dtype=np.float32)
        entity_embeddings = np.random.rand(
            batch_size, num_entities, embed_dim
        ).astype(np.float32)
        loss = loss_fn(y_true, entity_embeddings)
        assert np.isfinite(float(loss))

    def test_collapsed_entities_higher_loss(self):
        """Collapsed (identical) entities should produce higher loss than diverse ones."""
        loss_fn = GNNUtilizationLoss(
            diversity_weight=1.0, activation_weight=1.0, attention_weight=1.0
        )
        batch_size, num_entities, embed_dim = 4, 8, 16

        # Collapsed: near-identical entities
        base = np.random.rand(batch_size, 1, embed_dim).astype(np.float32) * 0.01
        collapsed = np.tile(base, (1, num_entities, 1))
        collapsed += np.random.rand(batch_size, num_entities, embed_dim).astype(np.float32) * 0.001

        # Diverse: distinct entities with large norms
        diverse = np.random.rand(
            batch_size, num_entities, embed_dim
        ).astype(np.float32) * 5.0

        dummy = np.zeros((batch_size, num_entities, embed_dim), dtype=np.float32)
        loss_collapsed = float(loss_fn(dummy, collapsed))
        loss_diverse = float(loss_fn(dummy, diverse))

        assert loss_diverse < loss_collapsed

    def test_get_config_roundtrip(self):
        loss = GNNUtilizationLoss(
            diversity_weight=0.1, attention_weight=0.05, activation_weight=0.02
        )
        config = loss.get_config()
        restored = GNNUtilizationLoss.from_config(config)
        assert restored.diversity_weight == 0.1
        assert restored.attention_weight == 0.05
        assert restored.activation_weight == 0.02

    def test_serialization(self):
        loss = GNNUtilizationLoss(diversity_weight=0.5)
        config = keras.saving.serialize_keras_object(loss)
        restored = keras.saving.deserialize_keras_object(config)
        assert isinstance(restored, GNNUtilizationLoss)
        assert restored.diversity_weight == 0.5


# ---------------------------------------------------------------------
# GNNUtilizationLoss per-sample contract
# (plan-2026-11-11T223825-64d1b401, step 4)
# ---------------------------------------------------------------------

BATCH = 4
ENTITIES = 5
EMBED = 8
SCALES = np.array([0.5, 1.0, 2.0, 4.0])

# Default-reduction values recorded at HEAD b4831ec33, before this step, with the
# scalar-returning code, on these two fixtures (golden file of the plan).
HEAD_GOLDEN_PIN = -0.036291882395744324
HEAD_GOLDEN_RAGGED = -0.08312790095806122

# Reduction lengths on the compared path: the einsum over EMBED, the abs-mean over
# ENTITIES * ENTITIES, the norm sum over EMBED, the mean over ENTITIES, the variance
# over ENTITIES (a mean and a mean of squares), the final mean over EMBED.
_LENGTHS = [EMBED, ENTITIES * ENTITIES, EMBED, ENTITIES, 2 * ENTITIES, EMBED]


def _ragged_embeddings():
    """``(B, 5, 8)`` float64 with row scales 0.5, 1, 2, 4 so every row differs."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(BATCH, ENTITIES, EMBED)) * SCALES[:, None, None]


def _pin_embeddings():
    """The fixture of the premature-scalar pin (``_entity_pair``): iid N(0, 1)."""
    return np.random.default_rng(1234).normal(size=(BATCH, ENTITIES, EMBED))


def _f32(x):
    return tf.constant(np.asarray(x, dtype="float32"))


def _dummy(x):
    """``y_true`` is ignored by the loss; pass a zero tensor of the batch's shape."""
    return _f32(np.zeros(np.shape(x)))


def _rows64(x, wd=0.01, wa=0.01, wv=0.01):
    """Per-row reference in float64, independent of the loss code.

    ``wd * mean|cos * offdiag| + wa * (-mean_n ||e||) + wv * (-mean_d var_n(e))``
    """
    x = np.asarray(x, dtype="float64")
    unit = x / np.linalg.norm(x, axis=-1, keepdims=True)
    cos = np.einsum("bnd,bmd->bnm", unit, unit)
    offdiag = 1.0 - np.eye(x.shape[1])
    diversity = np.abs(cos * offdiag).mean(axis=(1, 2))
    activation = -np.linalg.norm(x, axis=-1).mean(axis=1)
    variance = -x.var(axis=1).mean(axis=-1)
    return wd * diversity + wa * activation + wv * variance


def _atol(rows):
    return reassociation_atol(_LENGTHS, 1, float(np.abs(rows).max()))


class _ScalarGNN(GNNUtilizationLoss):
    """The pre-fix behaviour: one batch scalar, reduced inside ``call``."""

    def call(self, y_true, y_pred):
        return keras.ops.mean(super().call(y_true, y_pred))


class _BatchVarianceGNN(GNNUtilizationLoss):
    """Same default value, wrong rows: the variance term replaced by its batch mean.

    The correction ``coupled - own`` sums to zero over the batch, so the default
    reduction is unchanged; only the per-row reference and the row-drop identity can
    see that the rows are no longer row-local.
    """

    def call(self, y_true, y_pred):
        own = -keras.ops.mean(keras.ops.var(y_pred, axis=1), axis=-1)
        coupled = keras.ops.mean(own) * keras.ops.ones_like(own)
        return super().call(y_true, y_pred) + self.attention_weight * (coupled - own)


def _returns_per_row(loss_obj):
    """True when ``call`` returns ``(B,)`` rows equal to the float64 reference."""
    x = _ragged_embeddings()
    out = loss_obj.call(_dummy(x), _f32(x))
    if tuple(out.shape) != (BATCH,):
        return False
    ref = _rows64(x, loss_obj.diversity_weight, loss_obj.activation_weight,
                  loss_obj.attention_weight)
    return bool(np.abs(out.numpy() - ref).max() < _atol(ref))


def _drop_a_row_identity_holds(loss_obj_factory):
    """``loss(w=[1,1,1,0]) * B/(B-1) == loss(batch[:B-1])`` at atol 1e-6."""
    x = _ragged_embeddings()
    w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
    weighted = float(loss_obj_factory()(_dummy(x), _f32(x), sample_weight=w))
    dropped = float(loss_obj_factory()(_dummy(x[:-1]), _f32(x[:-1])))
    return abs(weighted * BATCH / (BATCH - 1) - dropped) < 1e-6


class TestGNNUtilizationLossIsPerSample:
    """``call`` returns one plain per-row mean per sample; no term is a batch statistic."""

    def test_call_returns_one_value_per_sample(self):
        x = _ragged_embeddings()
        out = GNNUtilizationLoss().call(_dummy(x), _f32(x))
        assert tuple(out.shape) == (BATCH,)

    @pytest.mark.parametrize(
        "weights",
        [(0.01, 0.01, 0.01), (1.0, 1.0, 1.0), (0.3, 0.2, 0.7)],
    )
    def test_rows_equal_the_float64_reference(self, weights):
        wd, wa, wv = weights
        x = _ragged_embeddings()
        out = GNNUtilizationLoss(
            diversity_weight=wd, activation_weight=wa, attention_weight=wv
        ).call(_dummy(x), _f32(x))
        ref = _rows64(x, wd, wa, wv)
        assert np.abs(out.numpy() - ref).max() < _atol(ref)
        # The rows really differ, so a broadcast scalar could not pass this.
        assert np.ptp(ref) > 1e-3 * np.abs(ref).max()

    def test_default_value_equals_the_head_golden(self):
        """Invariant 1: the default-reduction value does not move (recorded at HEAD)."""
        for x, golden in [
            (_pin_embeddings(), HEAD_GOLDEN_PIN),
            (_ragged_embeddings(), HEAD_GOLDEN_RAGGED),
        ]:
            value = float(GNNUtilizationLoss()(_dummy(x), _f32(x)))
            assert abs(value - golden) < 1e-7, (value, golden)

    def test_default_value_is_the_mean_of_the_rows(self):
        x = _ragged_embeddings()
        value = float(GNNUtilizationLoss()(_dummy(x), _f32(x)))
        assert abs(value - _rows64(x).mean()) < _atol(_rows64(x))

    def test_reduction_none_returns_the_rows(self):
        """``reduction`` is live: 'none' was a shape-() no-op before this fix."""
        x = _ragged_embeddings()
        out = GNNUtilizationLoss(reduction="none")(_dummy(x), _f32(x))
        assert tuple(out.shape) == (BATCH,)
        ref = _rows64(x)
        assert np.abs(np.asarray(out) - ref).max() < _atol(ref)

    def test_reduction_sum_is_batch_times_the_default(self):
        x = _ragged_embeddings()
        default = float(GNNUtilizationLoss()(_dummy(x), _f32(x)))
        summed = float(GNNUtilizationLoss(reduction="sum")(_dummy(x), _f32(x)))
        assert abs(summed - BATCH * default) < 1e-6
        assert abs(summed - default) > 0.05  # it was equal to the default before

    @pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
    def test_config_round_trip_keeps_a_live_reduction(self, reduction):
        original = GNNUtilizationLoss(
            diversity_weight=0.3, attention_weight=0.2, activation_weight=0.7,
            reduction=reduction,
        )
        restored = GNNUtilizationLoss.from_config(original.get_config())
        assert restored.reduction == reduction
        x = _ragged_embeddings()
        a = np.asarray(original(_dummy(x), _f32(x)))
        b = np.asarray(restored(_dummy(x), _f32(x)))
        assert a.shape == b.shape == ((BATCH,) if reduction == "none" else ())
        assert np.allclose(a, b, atol=0.0)

    def test_sample_weight_charges_each_row_its_own_value_times_its_weight(self):
        x = _ragged_embeddings()
        w = np.array([0.5, 2.0, 0.0, 1.5], "float32")
        weighted = float(GNNUtilizationLoss()(_dummy(x), _f32(x), sample_weight=w))
        ref = _rows64(x)
        assert abs(weighted - np.sum(w * ref) / BATCH) < _atol(ref)

    def test_zeroing_a_sample_weight_equals_dropping_the_row(self):
        """NOT batch-coupled: the identity holds, and exactly (no row reads another)."""
        assert _drop_a_row_identity_holds(GNNUtilizationLoss)
        x = _ragged_embeddings()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        weighted = float(GNNUtilizationLoss()(_dummy(x), _f32(x), sample_weight=w))
        dropped = float(GNNUtilizationLoss()(_dummy(x[:-1]), _f32(x[:-1])))
        # the two sides differ only by float32 summation order, not by a real gap
        assert abs(weighted * BATCH / (BATCH - 1) - dropped) < 1e-7

    def test_a_zero_weighted_row_gets_exactly_zero_gradient(self):
        x = _ragged_embeddings()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        p = tf.Variable(_f32(x))
        with tf.GradientTape() as tape:
            value = GNNUtilizationLoss()(_dummy(x), p, sample_weight=w)
        grad = tape.gradient(value, p).numpy()
        assert np.all(grad[3] == 0.0)
        assert np.abs(grad[:3]).max() > 1e-5  # the other rows do learn

    def test_default_gradient_is_the_gradient_of_the_old_scalar(self):
        """The old scalar, rebuilt here with the original all-axes means."""
        x = _ragged_embeddings()
        p = tf.Variable(_f32(x))
        with tf.GradientTape() as tape:
            value = GNNUtilizationLoss()(_dummy(x), p)
        new = tape.gradient(value, p).numpy()

        p2 = tf.Variable(_f32(x))
        with tf.GradientTape() as tape:
            unit = tf.math.l2_normalize(p2, axis=-1)
            sim = tf.einsum("bnd,bmd->bnm", unit, unit)
            mask = 1.0 - tf.eye(ENTITIES)
            diversity = tf.reduce_mean(tf.abs(sim * mask))
            activation = -tf.reduce_mean(tf.norm(p2, axis=-1))
            variance = -tf.reduce_mean(tf.math.reduce_variance(p2, axis=1))
            old = 0.01 * diversity + 0.01 * activation + 0.01 * variance
        old_grad = tape.gradient(old, p2).numpy()
        assert np.abs(new - old_grad).max() < 1e-7

    def test_the_per_row_tests_reject_the_scalar_returning_class(self):
        """ANTI-VACUITY: the pre-fix scalar fails the shape, reference and identity arms."""
        assert _returns_per_row(GNNUtilizationLoss())
        assert not _returns_per_row(_ScalarGNN())
        assert _drop_a_row_identity_holds(GNNUtilizationLoss)
        assert not _drop_a_row_identity_holds(_ScalarGNN)
        x = _ragged_embeddings()
        assert tuple(_ScalarGNN(reduction="none")(_dummy(x), _f32(x)).shape) == ()

    def test_the_tests_reject_rows_that_read_the_rest_of_the_batch(self):
        """ANTI-VACUITY: a variance term replaced by its batch mean keeps the default value."""
        x = _ragged_embeddings()
        same = float(_BatchVarianceGNN()(_dummy(x), _f32(x)))
        assert abs(same - HEAD_GOLDEN_RAGGED) < 1e-7  # the default-value test is blind to it
        assert not _returns_per_row(_BatchVarianceGNN())
        assert not _drop_a_row_identity_holds(_BatchVarianceGNN)

    def test_a_direct_float16_call_is_finite_per_row(self):
        """The diversity mask follows the input dtype (a float32 ``eye`` raised on float16)."""
        x = _ragged_embeddings()
        out = GNNUtilizationLoss().call(
            tf.constant(x.astype("float16")), tf.constant(x.astype("float16"))
        )
        assert tuple(out.shape) == (BATCH,)
        assert out.dtype == tf.float16
        values = out.numpy().astype("float64")
        assert np.isfinite(values).all()
        assert np.abs(values - _rows64(x)).max() < 5e-3

    def test_float16_inputs_through_the_loss_object_match_float32(self):
        """Through ``__call__`` the inputs are cast to float32 first, as under mixed_float16."""
        x = _ragged_embeddings()
        half_in = x.astype("float16")
        base = float(GNNUtilizationLoss()(_dummy(x), _f32(x)))
        half = float(
            GNNUtilizationLoss()(tf.constant(half_in), tf.constant(half_in))
        )
        rounded = float(
            GNNUtilizationLoss()(_dummy(x), _f32(half_in.astype("float64")))
        )
        assert abs(half - rounded) < _atol(_rows64(x))  # only the cast, no extra error
        assert abs(half - base) < 1e-3  # input rounding

    def test_the_docstrings_say_one_value_per_sample_and_not_a_batch_statistic(self):
        class_doc = " ".join((GNNUtilizationLoss.__doc__ or "").split())
        call_doc = " ".join((GNNUtilizationLoss.call.__doc__ or "").split()).lower()
        assert "within a batch" not in class_doc
        assert "within each sample" in class_doc
        assert "one value per sample" in call_doc
        assert "scalar utilization loss" not in call_doc
