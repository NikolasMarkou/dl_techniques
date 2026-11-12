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
from tests.test_losses.keras_mask_probe import (
    assert_a_per_sample_mask_is_left_to_keras,
    assert_refuses_a_keras_mask,
    assert_refuses_per_timestep_weights,
    evaluate_through_a_masking_lstm,
    with_keras_mask,
)


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


# ---------------------------------------------------------------------
# GNNUtilizationLoss and a Keras mask on the entity axis
# (plan-2026-11-11T223825-64d1b401, D-014)
# ---------------------------------------------------------------------

# One row per sample: all entities, a gap pattern, a middle run, a single entity.
ENTITY_MASKS = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
    ],
    dtype=bool,
)


def _masked_entities(batch):
    """``(batch, 5, 8)`` ragged embeddings and a ``(batch, 5)`` entity mask (5 == entities)."""
    index = np.arange(batch) % BATCH
    return _ragged_embeddings()[index], ENTITY_MASKS[index]


def _gnn_masked_rows64(x, mask, wd=0.01, wa=0.01, wv=0.01):
    """Reference: each row is the UNMASKED float64 loss of that row's kept entities alone.

    A row with no kept entity has nothing to score and is 0.
    """
    return np.array([
        _rows64(row[keep][None], wd, wa, wv)[0] if keep.any() else 0.0
        for row, keep in zip(x, mask)
    ])


class TestGNNUtilizationLossHonorsAnEntityMask:
    """A ``(batch, entities)`` Keras mask on ``y_pred`` selects the entities of each row.

    Keras would multiply that mask into the ``(batch,)`` rows (a shape error unless
    batch == entities, then silent mis-weighting), so ``__call__`` takes it and keeps
    only the unmasked entities in the per-row means, then reduces without a mask.
    """

    @pytest.mark.parametrize("batch", [4, 5])
    def test_rows_equal_the_unmasked_loss_of_the_kept_entities(self, batch):
        x, mask = _masked_entities(batch)
        rows = np.asarray(
            GNNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(x, mask))
        )
        ref = _gnn_masked_rows64(x, mask)
        assert np.abs(rows - ref).max() < _atol(ref)
        assert np.ptp(ref) > 1e-3 * np.abs(ref).max()  # the rows differ
        value = float(GNNUtilizationLoss()(_dummy(x), with_keras_mask(x, mask)))
        assert abs(value - ref.mean()) < _atol(ref)

    def test_masked_entities_do_not_matter(self):
        x, mask = _masked_entities(BATCH)
        junk = np.where(mask[..., None], x, 50.0 * np.random.default_rng(3).normal(size=x.shape))
        a = np.asarray(GNNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(x, mask)))
        b = np.asarray(GNNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(junk, mask)))
        assert np.abs(a - b).max() < 1e-7
        poisoned = np.where(mask[..., None], x, np.nan)  # NaN padding must not leak either
        c = np.asarray(
            GNNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(poisoned, mask))
        )
        assert np.isfinite(c).all() and np.abs(a - c).max() < 1e-7

    def test_a_full_mask_equals_no_mask(self):
        x = _ragged_embeddings()
        full = np.ones(x.shape[:2], bool)
        masked = float(GNNUtilizationLoss()(_dummy(x), with_keras_mask(x, full)))
        plain = float(GNNUtilizationLoss()(_dummy(x), _f32(x)))
        assert abs(masked - plain) < 1e-7

    def test_a_row_with_no_kept_entity_is_exactly_zero_and_finite(self):
        x, mask = _masked_entities(BATCH)
        mask = mask.copy()
        mask[2] = False
        rows = np.asarray(
            GNNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(x, mask))
        )
        assert np.isfinite(rows).all() and rows[2] == 0.0

    def test_masked_entities_get_exactly_zero_gradient(self):
        """Even when the padding is NaN: the gradient is finite and exactly 0 there."""
        x, mask = _masked_entities(BATCH)
        x = np.where(mask[..., None], x, np.nan)
        p = tf.Variable(_f32(x))
        with tf.GradientTape() as tape:
            q = tf.identity(p)
            q._keras_mask = tf.constant(mask)
            value = GNNUtilizationLoss()(_dummy(x), q)
        grad = tape.gradient(value, p).numpy()
        assert np.isfinite(grad).all() and np.all(grad[~mask] == 0.0)
        assert np.abs(grad[mask]).max() > 1e-5

    def test_sample_weight_combines_with_the_mask(self):
        x, mask = _masked_entities(BATCH)
        w = np.array([0.5, 2.0, 0.0, 1.5], "float32")
        got = float(
            GNNUtilizationLoss()(_dummy(x), with_keras_mask(x, mask), sample_weight=w)
        )
        ref = _gnn_masked_rows64(x, mask)
        assert abs(got - np.sum(w * ref) / BATCH) < _atol(ref)

    @pytest.mark.parametrize("entities", [4, 6])
    def test_a_compiled_model_with_a_masked_output_matches_the_reference(self, entities):
        """The reviewer's regression: ``Masking -> LSTM(return_sequences)``, any batch size."""
        for batch in (4, 6):
            rng = np.random.default_rng(batch * 10 + entities)
            x = rng.normal(size=(batch, entities, 3)).astype("float32")
            x[:, -2:, :] = 0.0  # padded entities
            value, prediction, mask = evaluate_through_a_masking_lstm(
                GNNUtilizationLoss(), x, units=EMBED
            )
            ref = _gnn_masked_rows64(prediction, mask)
            assert abs(value - ref.mean()) < 5e-6, (batch, entities, value, ref.mean())

    def test_a_float16_output_keeps_its_mask_through_the_cast(self):
        """Mixed precision hands the loss a float16 ``y_pred``; Keras then casts it (the mask
        attribute is lost on the cast tensor), so the mask must be read before the cast."""
        x, mask = _masked_entities(BATCH)
        half = tf.constant(x.astype("float16"))
        half._keras_mask = tf.constant(mask)
        got = float(GNNUtilizationLoss()(_dummy(x), half))
        ref = _gnn_masked_rows64(x.astype("float16").astype("float64"), mask)
        assert abs(got - ref.mean()) < _atol(ref)

    def test_only_a_batch_entity_mask_is_accepted(self):
        x, _ = _masked_entities(BATCH)
        assert_refuses_a_keras_mask(GNNUtilizationLoss, _dummy(x), _f32(x), (BATCH, ENTITIES, EMBED))
        assert_a_per_sample_mask_is_left_to_keras(GNNUtilizationLoss, _dummy(x), _f32(x))

    @pytest.mark.parametrize("batch", [4, 5])
    def test_a_per_entity_sample_weight_raises_a_clear_error(self, batch):
        x, _ = _masked_entities(batch)
        assert_refuses_per_timestep_weights(
            GNNUtilizationLoss, _dummy(x), _f32(x), (batch, ENTITIES)
        )

    def test_the_docstring_states_the_mask_rule(self):
        doc = " ".join((GNNUtilizationLoss.__doc__ or "").split())
        assert "Keras mask" in doc and "entities" in doc and "ValueError" in doc


# ---------------------------------------------------------------------
# MANNUtilizationLoss per-sample contract
# (plan-2026-11-11T223825-64d1b401, step 5; decision D-005)
# ---------------------------------------------------------------------

MEMORY_TIME = 6
MEMORY_DIM = 8

# Default-reduction values recorded at HEAD b4831ec33, before this step, with the
# scalar-returning code (golden file of the plan). The pin fixture is the one of the
# premature-scalar pin (``_memory_pair``: seed 1234, row scales 0.5 / 1 / 2 / 4); the
# ragged fixture is seed 7 with the same scales.
HEAD_GOLDEN_MEMORY_PIN = -0.0907827764749527
HEAD_GOLDEN_MEMORY_RAGGED = -0.0808926522731781

# Reduction lengths on the compared path: the softmax-entropy sum over MEMORY_DIM, the
# mean over TIME, the norm sum over MEMORY_DIM, the mean over TIME, the temporal
# variance (a mean and a mean of squares over (TIME - 1) * MEMORY_DIM), and the batch
# mean of ``v_b``.
_MEMORY_LENGTHS = [
    MEMORY_DIM, MEMORY_TIME, MEMORY_DIM, MEMORY_TIME,
    2 * (MEMORY_TIME - 1) * MEMORY_DIM, BATCH,
]


def _memory(seed):
    """``(B, 6, 8)`` float32-representable states with row scales 0.5, 1, 2, 4."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(BATCH, MEMORY_TIME, MEMORY_DIM)) * SCALES[:, None, None]
    return x.astype("float32").astype("float64")


def _ragged_memory():
    return _memory(7)


def _pin_memory():
    return _memory(1234)


def _entropy_and_magnitude64(m):
    """Per-row mean over time of the softmax entropy ``H`` and of the norm."""
    m = np.asarray(m, dtype="float64")
    e = np.exp(m - m.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    entropy = -(p * np.log(p + 1e-10)).sum(axis=-1)
    return entropy.mean(axis=1), np.linalg.norm(m, axis=-1).mean(axis=1)


def _mann_rows64(m, we=0.01, ww=0.01, wv=0.01):
    """Per-row reference in float64, independent of the loss code.

    ``we * (-mean_t H) + ww * (-mean_t ||m||)
      + wv * (-log(V + 1e-10) - (v_b - V) / (V + 1e-10))`` with ``d`` the temporal
    difference, ``V = var(d)`` over ALL axes and ``v_b = mean((d_b - mean(d))**2)``.
    """
    m = np.asarray(m, dtype="float64")
    entropy, magnitude = _entropy_and_magnitude64(m)
    d = m[:, 1:] - m[:, :-1]
    batch_var = d.var()
    v_b = ((d - d.mean()) ** 2).mean(axis=(1, 2))
    variance = -np.log(batch_var + 1e-10) - (v_b - batch_var) / (batch_var + 1e-10)
    return we * (-entropy) + ww * (-magnitude) + wv * variance


def _mann_old_scalar64(m, we=0.01, ww=0.01, wv=0.01):
    """The pre-fix scalar in float64: ``-log(var(d))`` over every axis of ``d``."""
    m = np.asarray(m, dtype="float64")
    entropy, magnitude = _entropy_and_magnitude64(m)
    d = m[:, 1:] - m[:, :-1]
    return float(
        we * (-entropy.mean()) + ww * (-magnitude.mean())
        + wv * (-np.log(d.var() + 1e-10))
    )


def _mann_atol(rows):
    return reassociation_atol(_MEMORY_LENGTHS, 1, float(np.abs(rows).max()))


def _mann_per_row_terms(loss_obj, y_pred):
    """Entropy and magnitude rows through keras ops, for the wrong-implementation arms."""
    probs = keras.ops.softmax(y_pred, axis=-1)
    entropy = -keras.ops.sum(probs * keras.ops.log(probs + 1e-10), axis=-1)
    norm = keras.ops.sqrt(keras.ops.sum(keras.ops.square(y_pred), axis=-1))
    return (
        -loss_obj.entropy_weight * keras.ops.mean(entropy, axis=1)
        - loss_obj.write_weight * keras.ops.mean(norm, axis=1)
    )


class _ScalarMANN(MANNUtilizationLoss):
    """The pre-fix behaviour: one batch scalar, reduced inside ``call``."""

    def call(self, y_true, y_pred):
        return keras.ops.mean(super().call(y_true, y_pred))


class _BroadcastVarianceMANN(MANNUtilizationLoss):
    """Same default value and gradient, wrong rows: ``-log(V)`` broadcast to every row.

    This is option (a2) of D-005. The mean over rows is the old scalar, so the
    default-value pin and the default-gradient pin are blind to it; only the
    zero-weighted-row gradient and the per-row reference can see it.
    """

    def call(self, y_true, y_pred):
        d = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        variance = -keras.ops.log(keras.ops.var(d) + 1e-10)
        rows = _mann_per_row_terms(self, y_pred)
        return rows + self.variance_weight * variance * keras.ops.ones_like(rows)


class _PerRowLogVarianceMANN(MANNUtilizationLoss):
    """Option (a3) of D-005: row-local ``-log(var_b)``, a DIFFERENT objective.

    Row-local (the drop-a-row identity holds) but the default value moves, because
    ``mean_b(-log var_b) != -log(V)`` (Jensen gap). Only the default-value pin sees it.
    """

    def call(self, y_true, y_pred):
        d = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        per_row_var = keras.ops.var(d, axis=(1, 2))
        rows = _mann_per_row_terms(self, y_pred)
        return rows - self.variance_weight * keras.ops.log(per_row_var + 1e-10)


def _mann_returns_per_row(loss_obj):
    """True when ``call`` returns ``(B,)`` rows equal to the float64 reference."""
    x = _ragged_memory()
    out = loss_obj.call(_dummy(x), _f32(x))
    if tuple(out.shape) != (BATCH,):
        return False
    ref = _mann_rows64(x, loss_obj.entropy_weight, loss_obj.write_weight,
                       loss_obj.variance_weight)
    return bool(np.abs(out.numpy() - ref).max() < _mann_atol(ref))


def _mann_default_value_is_the_head_golden(loss_obj_factory):
    """The default-reduction value on both fixtures equals the HEAD golden (1e-7)."""
    for x, golden in [
        (_pin_memory(), HEAD_GOLDEN_MEMORY_PIN),
        (_ragged_memory(), HEAD_GOLDEN_MEMORY_RAGGED),
    ]:
        value = float(loss_obj_factory()(_dummy(x), _f32(x)))
        if not abs(value - golden) < 1e-7:
            return False
    return True


def _mann_zero_row_gradient(loss_obj):
    """Gradient w.r.t. ``y_pred`` of the weighted loss, rows [1, 1, 1, 0]."""
    x = _ragged_memory()
    w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
    p = tf.Variable(_f32(x))
    with tf.GradientTape() as tape:
        value = loss_obj(_dummy(x), p, sample_weight=w)
    return tape.gradient(value, p).numpy()


def _mann_zero_weighted_row_gets_exactly_zero_gradient(loss_obj):
    return bool(np.all(_mann_zero_row_gradient(loss_obj)[3] == 0.0))


def _mann_drop_a_row_identity_holds(loss_obj_factory):
    """``loss(w=[1,1,1,0]) * B/(B-1) == loss(batch[:B-1])`` at atol 1e-6."""
    x = _ragged_memory()
    w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
    weighted = float(loss_obj_factory()(_dummy(x), _f32(x), sample_weight=w))
    dropped = float(loss_obj_factory()(_dummy(x[:-1]), _f32(x[:-1])))
    return abs(weighted * BATCH / (BATCH - 1) - dropped) < 1e-6


def _follows_the_d005_rule(module_source, call_source):
    """True when the anchor is present and ``call`` builds the detached surrogate."""
    return (
        "# DECISION plan-2026-11-11T223825-64d1b401/D-005" in module_source
        and call_source.count("stop_gradient") >= 2  # the detached mean and V
        and "variance_row" in call_source
        and "ones_like" not in call_source
    )


class TestMANNUtilizationLossIsPerSample:
    """``call`` returns one value per sample; only the variance term reads the batch."""

    def test_call_returns_one_value_per_sample(self):
        x = _ragged_memory()
        out = MANNUtilizationLoss().call(_dummy(x), _f32(x))
        assert tuple(out.shape) == (BATCH,)

    @pytest.mark.parametrize(
        "weights",
        [(0.01, 0.01, 0.01), (1.0, 1.0, 1.0), (0.3, 0.2, 0.7)],
    )
    def test_rows_equal_the_float64_reference(self, weights):
        we, ww, wv = weights
        x = _ragged_memory()
        out = MANNUtilizationLoss(
            entropy_weight=we, write_weight=ww, variance_weight=wv
        ).call(_dummy(x), _f32(x))
        ref = _mann_rows64(x, we, ww, wv)
        assert np.abs(out.numpy() - ref).max() < _mann_atol(ref)
        # The rows really differ, so a broadcast scalar could not pass this.
        assert np.ptp(ref) > 1e-3 * np.abs(ref).max()

    def test_default_value_equals_the_head_golden(self):
        """Invariant 1 (D-013): within 1e-7 of the value recorded at HEAD."""
        assert _mann_default_value_is_the_head_golden(MANNUtilizationLoss)

    def test_default_value_is_the_float64_old_scalar(self):
        """The mean of the rows IS the old ``-log(V)`` scalar, evaluated independently."""
        for x in (_pin_memory(), _ragged_memory()):
            value = float(MANNUtilizationLoss()(_dummy(x), _f32(x)))
            old = _mann_old_scalar64(x)
            assert abs(value - old) < _mann_atol(_mann_rows64(x))
            assert abs(_mann_rows64(x).mean() - old) < 1e-12

    def test_reduction_none_returns_the_rows(self):
        """``reduction`` is live: 'none' was a shape-() no-op before this fix."""
        x = _ragged_memory()
        out = MANNUtilizationLoss(reduction="none")(_dummy(x), _f32(x))
        assert tuple(out.shape) == (BATCH,)
        ref = _mann_rows64(x)
        assert np.abs(np.asarray(out) - ref).max() < _mann_atol(ref)

    def test_reduction_sum_is_batch_times_the_default(self):
        x = _ragged_memory()
        default = float(MANNUtilizationLoss()(_dummy(x), _f32(x)))
        summed = float(MANNUtilizationLoss(reduction="sum")(_dummy(x), _f32(x)))
        assert abs(summed - BATCH * default) < 1e-6
        assert abs(summed - default) > 0.05  # it was equal to the default before

    @pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
    def test_config_round_trip_keeps_a_live_reduction(self, reduction):
        original = MANNUtilizationLoss(
            entropy_weight=0.3, write_weight=0.2, variance_weight=0.7,
            reduction=reduction,
        )
        restored = MANNUtilizationLoss.from_config(original.get_config())
        assert restored.reduction == reduction
        assert restored.variance_weight == 0.7
        x = _ragged_memory()
        a = np.asarray(original(_dummy(x), _f32(x)))
        b = np.asarray(restored(_dummy(x), _f32(x)))
        assert a.shape == b.shape == ((BATCH,) if reduction == "none" else ())
        assert np.allclose(a, b, atol=0.0)

    def test_sample_weight_charges_each_row_its_own_value_times_its_weight(self):
        """The documented form: full-batch ``V`` and ``mu``, per-row weights."""
        x = _ragged_memory()
        w = np.array([0.5, 2.0, 0.0, 1.5], "float32")
        weighted = float(MANNUtilizationLoss()(_dummy(x), _f32(x), sample_weight=w))
        ref = _mann_rows64(x)
        assert abs(weighted - np.sum(w * ref) / BATCH) < _mann_atol(ref)

    def test_a_zero_weighted_row_gets_exactly_zero_gradient(self):
        """The property the broadcast-constant alternative (D-005 option a2) fails."""
        grad = _mann_zero_row_gradient(MANNUtilizationLoss())
        assert np.all(grad[3] == 0.0)
        assert np.abs(grad[:3]).max() > 1e-5  # the other rows do learn

    def test_default_gradient_is_the_gradient_of_the_old_scalar(self):
        """The old scalar, rebuilt here with the original all-axes variance (atol 1e-9)."""
        x = _ragged_memory()
        p = tf.Variable(_f32(x))
        with tf.GradientTape() as tape:
            value = MANNUtilizationLoss()(_dummy(x), p)
        new = tape.gradient(value, p).numpy()

        p2 = tf.Variable(_f32(x))
        with tf.GradientTape() as tape:
            probs = tf.nn.softmax(p2, axis=-1)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
            d = p2[:, 1:, :] - p2[:, :-1, :]
            variance = tf.math.reduce_variance(d)
            norm = tf.sqrt(tf.reduce_sum(tf.square(p2), axis=-1))
            old = (
                0.01 * -tf.reduce_mean(entropy)
                + 0.01 * -tf.math.log(variance + 1e-10)
                + 0.01 * -tf.reduce_mean(norm)
            )
        old_grad = tape.gradient(old, p2).numpy()
        assert np.abs(old_grad).max() > 1e-4  # the comparison is not between zeros
        assert np.abs(new - old_grad).max() < 1e-9

    def test_zeroing_a_sample_weight_equals_dropping_the_row_without_the_variance_term(self):
        """With ``variance_weight=0`` nothing reads another row: the identity holds."""

        def make():
            return MANNUtilizationLoss(variance_weight=0.0)

        assert _mann_drop_a_row_identity_holds(make)
        x = _ragged_memory()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        weighted = float(make()(_dummy(x), _f32(x), sample_weight=w))
        dropped = float(make()(_dummy(x[:-1]), _f32(x[:-1])))
        assert abs(weighted * BATCH / (BATCH - 1) - dropped) < 1e-7

    def test_with_the_variance_term_the_drop_a_row_identity_does_not_hold(self):
        """DOCUMENTED batch coupling (D-005): a zero-weighted row still shifts the rest.

        The weighted loss is the three rows scored against the FULL-batch variance and
        mean; the dropped batch has its own, smaller-spread, variance. Both sides are
        pinned against the float64 reference, so the gap is a number, not a mood.
        """
        x = _ragged_memory()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        weighted = float(MANNUtilizationLoss()(_dummy(x), _f32(x), sample_weight=w))
        dropped = float(MANNUtilizationLoss()(_dummy(x[:-1]), _f32(x[:-1])))
        scaled = weighted * BATCH / (BATCH - 1)

        ref_rows = _mann_rows64(x)
        coupled = float(ref_rows[:-1].sum() / (BATCH - 1))  # full-batch V and mu
        independent = float(_mann_rows64(x[:-1]).mean())  # V and mu of 3 rows
        assert abs(scaled - coupled) < _mann_atol(ref_rows)
        assert abs(dropped - independent) < _mann_atol(ref_rows)
        # measured on this fixture: -0.0614277 against -0.0574182, a gap of 4.0e-3
        assert abs(scaled - (-0.06142767774753244)) < 1e-7
        assert abs(dropped - (-0.057418180421810336)) < 1e-7
        assert abs(scaled - dropped) > 3.9e-3
        assert not _mann_drop_a_row_identity_holds(MANNUtilizationLoss)

    def test_the_per_row_tests_reject_the_scalar_returning_class(self):
        """ANTI-VACUITY: the pre-fix scalar fails the shape and reference arms."""
        assert _mann_returns_per_row(MANNUtilizationLoss())
        assert not _mann_returns_per_row(_ScalarMANN())
        assert not _mann_drop_a_row_identity_holds(
            lambda: _ScalarMANN(variance_weight=0.0)
        )
        x = _ragged_memory()
        assert tuple(_ScalarMANN(reduction="none")(_dummy(x), _f32(x)).shape) == ()

    def test_the_gradient_arm_rejects_the_broadcast_constant_alternative(self):
        """ANTI-VACUITY (D-005 option a2): same default value and gradient, wrong rows.

        The broadcast ``-log(V)`` keeps the default value and the default gradient
        (so those pins pass it), but a zero-weighted row still receives a gradient
        through the constant, and its rows are not the reference rows.
        """
        assert _mann_default_value_is_the_head_golden(_BroadcastVarianceMANN)
        assert _mann_zero_weighted_row_gets_exactly_zero_gradient(MANNUtilizationLoss())
        assert not _mann_zero_weighted_row_gets_exactly_zero_gradient(
            _BroadcastVarianceMANN()
        )
        assert np.abs(_mann_zero_row_gradient(_BroadcastVarianceMANN())[3]).max() > 1e-5
        assert not _mann_returns_per_row(_BroadcastVarianceMANN())

    def test_the_value_arm_rejects_the_per_row_log_variance_alternative(self):
        """ANTI-VACUITY (D-005 option a3): row-local, but a different objective."""
        assert _mann_default_value_is_the_head_golden(MANNUtilizationLoss)
        assert not _mann_default_value_is_the_head_golden(_PerRowLogVarianceMANN)
        # it IS row-local, which is why the identity arm cannot tell it from the fix
        assert _mann_drop_a_row_identity_holds(_PerRowLogVarianceMANN)
        assert not _mann_returns_per_row(_PerRowLogVarianceMANN())

    def test_float16_inputs_through_the_loss_object_match_float32(self):
        """Through ``__call__`` the inputs are cast to float32 first, as under mixed_float16."""
        x = _ragged_memory()
        half_in = x.astype("float16")
        base = float(MANNUtilizationLoss()(_dummy(x), _f32(x)))
        half = float(
            MANNUtilizationLoss()(tf.constant(half_in), tf.constant(half_in))
        )
        rounded = float(
            MANNUtilizationLoss()(_dummy(x), _f32(half_in.astype("float64")))
        )
        assert abs(half - rounded) < _mann_atol(_mann_rows64(x))  # only the cast
        assert abs(half - base) < 1e-3  # input rounding

    def test_the_docstrings_say_one_value_per_sample_and_name_the_coupling(self):
        class_doc = " ".join((MANNUtilizationLoss.__doc__ or "").split())
        call_doc = " ".join((MANNUtilizationLoss.call.__doc__ or "").split()).lower()
        assert "one value per sample" in call_doc
        assert "scalar utilization loss" not in call_doc
        assert "batch-coupled" in class_doc
        assert "variance_weight=0" in class_doc
        # the documented direct-call limits (D-009): not fixed, so they are stated
        assert "float16" in class_doc and "time >= 2" in class_doc

    def test_the_variance_term_keeps_its_decision_anchor_and_surrogate_form(self):
        """D-005 guard: the anchor and the detached surrogate must survive a refactor.

        The rule the anchor states: the variance term is the detached first-order
        surrogate, never ``-log`` of a per-row variance (changes the objective) and
        never ``-log(V)`` broadcast to every row (leaks gradient into a zero-weighted
        row). A source that drops the anchor, the ``stop_gradient`` detachments, or
        the ``(v_b - V)`` correction fails here, and the two rejected alternatives
        are shown to fail it.
        """
        import inspect
        from dl_techniques.losses import utilization_loss

        assert _follows_the_d005_rule(
            inspect.getsource(utilization_loss),
            inspect.getsource(MANNUtilizationLoss.call),
        )
        module = inspect.getsource(utilization_loss)
        for wrong in (_BroadcastVarianceMANN, _PerRowLogVarianceMANN):
            assert not _follows_the_d005_rule(module, inspect.getsource(wrong.call))
        no_anchor = module.replace("# DECISION plan-2026-11-11T223825-64d1b401/D-005", "")
        assert not _follows_the_d005_rule(
            no_anchor, inspect.getsource(MANNUtilizationLoss.call)
        )


# ---------------------------------------------------------------------
# MANNUtilizationLoss and a Keras mask on the time axis
# (plan-2026-11-11T223825-64d1b401, D-014)
# ---------------------------------------------------------------------

#: Valid prefix length of each row of a right-padded batch (6 rows so batch == time).
PREFIX_LENGTHS = [6, 4, 3, 5, 2, 6]
#: A mask with gaps: the difference across a gap is not a temporal difference.
GAP_MASK = np.array(
    [
        [1, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 1],
    ],
    dtype=bool,
)


def _prefix_mask(lengths):
    return np.arange(MEMORY_TIME)[None, :] < np.asarray(lengths)[:, None]


def _masked_memory(batch, mask=None):
    """``(batch, 6, 8)`` states with row scales 0.5 to 4 and the matching ``(batch, 6)`` mask."""
    index = np.arange(batch) % BATCH
    x = _ragged_memory()[index]
    if mask is None:
        mask = _prefix_mask(PREFIX_LENGTHS[:batch])
    return x, mask


def _mann_masked_rows64(m, mask, we=0.01, ww=0.01, wv=0.01):
    """Float64 reference written row by row over the kept steps, independent of the loss code.

    Entropy and magnitude are means over a row's kept steps. A temporal difference
    exists only where BOTH neighbouring steps are kept; ``mu`` is the mean of all kept
    differences, ``v_b`` the mean squared deviation of the row's kept differences, and
    ``V`` the mean of ``v_b`` over the rows that have one. A row with no kept step is 0
    and a row with no kept difference has no variance term.
    """
    m = np.asarray(m, dtype="float64")
    batch, time, _ = m.shape
    e = np.exp(m - m.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    entropy = -(p * np.log(p + 1e-10)).sum(axis=-1)
    norm = np.linalg.norm(m, axis=-1)
    ent, mag, diffs = np.zeros(batch), np.zeros(batch), []
    for b in range(batch):
        kept = np.flatnonzero(mask[b])
        if len(kept):
            ent[b], mag[b] = entropy[b, kept].mean(), norm[b, kept].mean()
        starts = [t for t in range(time - 1) if mask[b, t] and mask[b, t + 1]]
        diffs.append(m[b, [t + 1 for t in starts]] - m[b, starts])
    mu = np.concatenate(diffs).mean()
    v = np.array([((d - mu) ** 2).mean() if len(d) else 0.0 for d in diffs])
    has = np.array([len(d) > 0 for d in diffs])
    batch_var = v[has].mean()
    variance = np.where(
        has, -np.log(batch_var + 1e-10) - (v - batch_var) / (batch_var + 1e-10), 0.0
    )
    return we * (-ent) + ww * (-mag) + wv * variance


class TestMANNUtilizationLossHonorsATimeMask:
    """A ``(batch, time)`` Keras mask on ``y_pred`` selects the timesteps of each row.

    This is the reviewer's regression: ``Masking`` (or ``Embedding(mask_zero=True)``) then
    an ``LSTM(return_sequences=True)`` hands the loss a masked sequence. Keras would
    multiply that mask into the ``(batch,)`` rows: a shape error unless batch == time,
    then silent mis-weighting (measured -0.0777 against the correct -0.0742). The loss
    takes the mask, keeps only the unmasked steps in its per-row means and in the
    temporal difference, and reduces without a mask.
    """

    @pytest.mark.parametrize("batch", [4, 6])
    @pytest.mark.parametrize("which", ["prefix", "gap"])
    def test_rows_equal_the_float64_reference_over_the_kept_steps(self, batch, which):
        mask = None if which == "prefix" else GAP_MASK[np.arange(batch) % 4]
        x, mask = _masked_memory(batch, mask)
        rows = np.asarray(
            MANNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(x, mask))
        )
        ref = _mann_masked_rows64(x, mask)
        assert np.abs(rows - ref).max() < _mann_atol(ref)
        assert np.ptp(ref) > 1e-3 * np.abs(ref).max()  # the rows differ
        value = float(MANNUtilizationLoss()(_dummy(x), with_keras_mask(x, mask)))
        assert abs(value - ref.mean()) < _mann_atol(ref)

    @pytest.mark.parametrize("batch", [4, 6])
    def test_a_masked_row_is_the_unmasked_loss_of_its_valid_prefix(self, batch):
        """Without the variance term nothing reads another row, so the identity is exact,
        and the right-hand side runs the UNMASKED code path."""
        x, mask = _masked_memory(batch)
        rows = np.asarray(
            MANNUtilizationLoss(variance_weight=0.0, reduction="none")(
                _dummy(x), with_keras_mask(x, mask)
            )
        )
        for b, length in enumerate(PREFIX_LENGTHS[:batch]):
            alone = MANNUtilizationLoss(variance_weight=0.0)(
                _dummy(x[b:b + 1, :length]), _f32(x[b:b + 1, :length])
            )
            assert abs(rows[b] - float(alone)) < 1e-6, (b, rows[b], float(alone))

    def test_masked_steps_do_not_matter(self):
        x, mask = _masked_memory(BATCH)
        junk = np.where(
            mask[..., None], x, 50.0 * np.random.default_rng(3).normal(size=x.shape)
        )
        a = np.asarray(MANNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(x, mask)))
        b = np.asarray(MANNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(junk, mask)))
        assert np.abs(a - b).max() < 1e-6
        poisoned = np.where(mask[..., None], x, np.nan)  # NaN padding must not leak either
        c = np.asarray(
            MANNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(poisoned, mask))
        )
        assert np.isfinite(c).all() and np.abs(a - c).max() < 1e-6

    def test_a_full_mask_equals_no_mask(self):
        x = _ragged_memory()
        full = np.ones(x.shape[:2], bool)
        masked = float(MANNUtilizationLoss()(_dummy(x), with_keras_mask(x, full)))
        plain = float(MANNUtilizationLoss()(_dummy(x), _f32(x)))
        assert abs(masked - plain) < 1e-7

    def test_short_rows_are_finite_and_the_empty_row_is_zero(self):
        """One kept step has no temporal difference; no kept step has nothing to score."""
        x, _ = _masked_memory(BATCH)
        mask = _prefix_mask([6, 1, 0, 2])
        rows = np.asarray(
            MANNUtilizationLoss(reduction="none")(_dummy(x), with_keras_mask(x, mask))
        )
        ref = _mann_masked_rows64(x, mask)
        assert np.isfinite(rows).all() and rows[2] == 0.0
        assert np.abs(rows - ref).max() < _mann_atol(ref)

    def test_masked_steps_get_exactly_zero_gradient(self):
        """Even when the padding is NaN: the gradient is finite and exactly 0 there."""
        x, mask = _masked_memory(BATCH)
        x = np.where(mask[..., None], x, np.nan)
        p = tf.Variable(_f32(x))
        with tf.GradientTape() as tape:
            q = tf.identity(p)
            q._keras_mask = tf.constant(mask)
            value = MANNUtilizationLoss()(_dummy(x), q)
        grad = tape.gradient(value, p).numpy()
        assert np.isfinite(grad).all() and np.all(grad[~mask] == 0.0)
        assert np.abs(grad[mask]).max() > 1e-5

    def test_sample_weight_combines_with_the_mask(self):
        x, mask = _masked_memory(BATCH)
        w = np.array([0.5, 2.0, 0.0, 1.5], "float32")
        got = float(
            MANNUtilizationLoss()(_dummy(x), with_keras_mask(x, mask), sample_weight=w)
        )
        ref = _mann_masked_rows64(x, mask)
        assert abs(got - np.sum(w * ref) / BATCH) < _mann_atol(ref)

    @pytest.mark.parametrize("batch", [4, 6])
    def test_a_compiled_model_with_a_masked_output_matches_the_reference(self, batch):
        """``Masking -> LSTM(return_sequences)`` through ``compile``/``evaluate``.

        At batch 4 (time 6) the unfixed loss raised ``Dimensions must be equal``; at
        batch 6 it ran and returned 0.028364 against the right 0.027974.
        """
        rng = np.random.default_rng(batch)
        x = rng.normal(size=(batch, MEMORY_TIME, 3)).astype("float32")
        x[:, -2:, :] = 0.0  # padded timesteps
        value, prediction, mask = evaluate_through_a_masking_lstm(
            MANNUtilizationLoss(), x, units=MEMORY_DIM
        )
        ref = _mann_masked_rows64(prediction, mask)
        assert abs(value - ref.mean()) < 5e-6, (batch, value, ref.mean())

    def test_a_float16_output_keeps_its_mask_through_the_cast(self):
        x, mask = _masked_memory(BATCH)
        half = tf.constant(x.astype("float16"))
        half._keras_mask = tf.constant(mask)
        got = float(MANNUtilizationLoss()(_dummy(x), half))
        ref = _mann_masked_rows64(x.astype("float16").astype("float64"), mask)
        assert abs(got - ref.mean()) < _mann_atol(ref)

    def test_only_a_batch_time_mask_is_accepted(self):
        x, _ = _masked_memory(BATCH)
        assert_refuses_a_keras_mask(
            MANNUtilizationLoss, _dummy(x), _f32(x), (BATCH, MEMORY_TIME, MEMORY_DIM)
        )
        assert_a_per_sample_mask_is_left_to_keras(MANNUtilizationLoss, _dummy(x), _f32(x))

    @pytest.mark.parametrize("batch", [4, 6])
    def test_a_per_timestep_sample_weight_raises_a_clear_error(self, batch):
        x, _ = _masked_memory(batch)
        assert_refuses_per_timestep_weights(
            MANNUtilizationLoss, _dummy(x), _f32(x), (batch, MEMORY_TIME)
        )

    def test_the_docstring_states_the_mask_rule(self):
        doc = " ".join((MANNUtilizationLoss.__doc__ or "").split())
        assert "Keras mask" in doc and "time" in doc and "ValueError" in doc
