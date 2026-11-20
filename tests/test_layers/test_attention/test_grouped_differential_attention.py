"""Tests for GroupedDifferentialAttention.

The load-bearing test is the NumPy oracle: it recomputes the layer from its own weights with an
explicit loop over signal heads (head ``i = g * h + j`` reads noise head ``j`` and value head ``j``).
It is what pins the tile-versus-repeat head ordering and the "noise context computed once" shortcut.
The mechanism tests then isolate which heads each projection can reach, through an identity output
projection so that the per-head outputs are visible.

RED record (research/2026_keras_custom_models_instructions_v2.md, house rule 2). Each mutation was
injected into ``grouped_differential_attention.py`` in turn, this file run, and the file restored.
Measured 2026-11-20, CPU, 122 tests in the file, baseline 122 passed:

======================================================  ======  =================================
injection                                               failed  first-line evidence
======================================================  ======  =================================
drop the ``(1 - lambda_init)`` scale                     13     oracle (per-head loop), with mask
drop the per-head RMSNorm                                15     oracle, gradients (head_norm scale)
noise map left unmasked                                   3     masked-token influence, fp16 twin
lambda without its second exponential                    16     oracle, gradients (lambda vectors)
no static contract re-assertion in ``call``               2     feature-width and rank tests
dropout layers run although the rate is 0                 1     layout test (call counted)
initializer instance shared across projections            3     replay oracle, spec-not-instance
values repeated element-wise instead of tiled            11     value-head reach, oracle
noise context repeated element-wise instead of tiled     13     noise-head reach, oracle
unbuilt twin dropout (control, equivalent mutant)         0     expected: Dropout has no weights
======================================================  ======  =================================
"""

import inspect
import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.factory import (
    create_attention_layer,
    validate_attention_config,
)
from dl_techniques.layers.attention.grouped_differential_attention import (
    GroupedDifferentialAttention,
)

# ---------------------------------------------------------------------

DIM = 64
SEQ = 12
BATCH = 3

# Tolerance of the float64 NumPy oracle, relative to max(1, output absmax). Measured 2026-11-20, CPU,
# float32, 8 heads G=3 and 12 heads G=5, layer_idx 0 and 3: max|layer - oracle| / max(1, absmax)
# = 4.5e-07 (float32 rounding). The smallest defect this must catch moves the output by 7.2e-04
# (lambda_q1 + 1e-3), a layer_idx off by one by 1.3e-01. 1e-5 is 22x above the floor and 72x below
# the smallest defect.
ORACLE_RTOL = 1e-5


def _config(**changes: Any) -> Dict[str, Any]:
    config = dict(dim=DIM, num_heads=8, head_dim=16, group_ratio=3)
    config.update(changes)
    return config


def _x(seed: int = 0, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(BATCH, SEQ, dim)).astype("float32")


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=-1, keepdims=True)


def _randomize(layer: GroupedDifferentialAttention, seed: int = 1) -> None:
    """Replace every weight by a nonzero draw so no term of the formula is trivially zero."""
    rng = np.random.default_rng(seed)
    layer.set_weights([
        rng.normal(scale=0.3, size=w.shape).astype("float32") for w in layer.weights
    ])


def _oracle(layer: GroupedDifferentialAttention, x: np.ndarray, layer_idx: int,
            mask: np.ndarray = None) -> np.ndarray:
    """The layer's formula, one signal head at a time, in float64."""
    h, G, d = layer.num_noise_heads, layer.group_ratio, layer.head_dim

    def dense(sub, inp):
        kernel, bias = [np.asarray(w, dtype="float64") for w in sub.weights]
        return inp @ kernel + bias

    x = x.astype("float64")
    B, L, _ = x.shape
    q1 = dense(layer.q1_dense, x).reshape(B, L, G * h, d)
    k1 = dense(layer.k1_dense, x).reshape(B, L, G * h, d)
    q2 = dense(layer.q2_dense, x).reshape(B, L, h, d)
    k2 = dense(layer.k2_dense, x).reshape(B, L, h, d)
    v = dense(layer.v_dense, x).reshape(B, L, h, d)

    lam_vecs = [np.asarray(w, dtype="float64") for w in
                (layer.lambda_q1, layer.lambda_k1, layer.lambda_q2, layer.lambda_k2)]
    lambda_init = 0.8 - 0.6 * np.exp(-0.3 * layer_idx)
    lam = (np.exp(np.sum(lam_vecs[0] * lam_vecs[1]))
           - np.exp(np.sum(lam_vecs[2] * lam_vecs[3])) + lambda_init)

    norm_scale = np.asarray(layer.head_norm.scale, dtype="float64")
    eps = layer.head_norm.epsilon

    heads = []
    for i in range(G * h):
        j = i % h
        s1 = np.einsum("bqd,bkd->bqk", q1[:, :, i], k1[:, :, i]) / np.sqrt(d)
        s2 = np.einsum("bqd,bkd->bqk", q2[:, :, j], k2[:, :, j]) / np.sqrt(d)
        if mask is not None:
            keep = mask[:, None, :] > 0
            s1 = np.where(keep, s1, -1e30)
            s2 = np.where(keep, s2, -1e30)
        out = np.einsum("bqk,bkd->bqd", _softmax(s1) - lam * _softmax(s2), v[:, :, j])
        out = out / np.sqrt(np.mean(out ** 2, axis=-1, keepdims=True) + eps) * norm_scale
        heads.append(out * (1.0 - lambda_init))
    merged = np.concatenate(heads, axis=-1)
    return dense(layer.proj, merged)


# ---------------------------------------------------------------------

class TestConstruction:

    def test_defaults_and_head_counts(self):
        layer = GroupedDifferentialAttention(dim=DIM, num_heads=8, head_dim=16)
        assert layer.group_ratio == 3
        assert layer.num_noise_heads == 2
        assert layer.num_signal_heads == 6
        assert layer.num_signal_heads + layer.num_noise_heads == layer.num_heads

    @pytest.mark.parametrize("bad", [
        dict(dim=0), dict(dim=-1), dict(num_heads=0), dict(head_dim=0),
        dict(group_ratio=0), dict(group_ratio=-2),
        dict(num_heads=7), dict(num_heads=10, group_ratio=3),
        dict(dropout_rate=-0.1), dict(dropout_rate=1.5),
        dict(attention_dropout_rate=2.0),
        dict(probability_type="routing"), dict(probability_type="Hierarchical"),
    ])
    def test_invalid_configuration_raises(self, bad):
        with pytest.raises(ValueError):
            GroupedDifferentialAttention(**_config(**bad))

    def test_divisibility_message_names_the_ratio(self):
        with pytest.raises(ValueError, match=r"group_ratio \+ 1"):
            GroupedDifferentialAttention(**_config(num_heads=9))

    def test_group_ratio_one_is_the_balanced_split(self):
        layer = GroupedDifferentialAttention(**_config(num_heads=8, group_ratio=1))
        assert (layer.num_signal_heads, layer.num_noise_heads) == (4, 4)

    def test_unbuilt_compute_output_shape(self):
        layer = GroupedDifferentialAttention(**_config())
        assert layer.compute_output_shape((None, 5, DIM)) == (None, 5, DIM)


class TestBuild:

    def test_weight_shapes_follow_the_ratio(self):
        layer = GroupedDifferentialAttention(**_config(num_heads=8, group_ratio=3, head_dim=16))
        layer(_x())
        shapes = {w.path.split("/", 1)[1]: tuple(w.shape) for w in layer.weights}
        assert shapes["q1/kernel"] == (DIM, 6 * 16)
        assert shapes["k1/kernel"] == (DIM, 6 * 16)
        assert shapes["q2/kernel"] == (DIM, 2 * 16)
        assert shapes["k2/kernel"] == (DIM, 2 * 16)
        assert shapes["v/kernel"] == (DIM, 2 * 16)
        assert shapes["proj/kernel"] == (6 * 16, DIM)
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            assert shapes[name] == (16,)
        assert shapes["head_norm/scale"] == (16,)

    def test_a_larger_ratio_gives_fewer_value_and_noise_parameters(self):
        def count(G):
            layer = GroupedDifferentialAttention(**_config(num_heads=12, group_ratio=G))
            layer(_x())
            return layer.count_params()
        # 12 heads: G=1 -> 6 noise heads, G=2 -> 4, G=3 -> 3, G=5 -> 2
        counts = [count(G) for G in (1, 2, 3, 5)]
        # q2/k2/v shrink with h; q1/k1/proj grow with G*h. Pin the exact closed form.
        for G, got in zip((1, 2, 3, 5), counts):
            h = 12 // (G + 1)
            d = 16
            expected = (
                2 * (DIM * G * h * d + G * h * d)      # q1, k1
                + 3 * (DIM * h * d + h * d)            # q2, k2, v
                + (G * h * d * DIM + DIM)              # proj
                + 4 * d + d                            # lambda vectors, head norm scale
            )
            assert got == expected

    @pytest.mark.parametrize("shape", [(2, 8), (2, 8, 8, DIM)])
    def test_rank_other_than_three_raises(self, shape):
        with pytest.raises(ValueError, match="3D"):
            GroupedDifferentialAttention(**_config()).build(shape)

    def test_wrong_feature_width_raises(self):
        with pytest.raises(ValueError, match="doesn't match"):
            GroupedDifferentialAttention(**_config()).build((2, 8, DIM + 1))

    @pytest.mark.parametrize("qk_norm_type", [None, "rms_norm"])
    def test_every_sublayer_is_built_after_build(self, qk_norm_type):
        layer = GroupedDifferentialAttention(**_config(qk_norm_type=qk_norm_type, attention_dropout_rate=0.1))
        layer.build((None, SEQ, DIM))
        subs = [layer.q1_dense, layer.k1_dense, layer.q2_dense, layer.k2_dense, layer.v_dense,
                layer.proj, layer.dropout_layer, layer.attn_prob_1, layer.attn_prob_2,
                layer.attn_dropout_1, layer.attn_dropout_2, layer.head_norm]
        if qk_norm_type:
            subs += [layer.q_norm_1, layer.k_norm_1, layer.q_norm_2, layer.k_norm_2]
        assert all(s.built for s in subs)


# ---------------------------------------------------------------------

class TestAgainstTheOracle:

    @pytest.mark.parametrize("num_heads,group_ratio", [(8, 3), (6, 2), (4, 1), (4, 3), (10, 4), (12, 5)])
    @pytest.mark.parametrize("layer_idx", [0, 3])
    def test_matches_the_per_head_loop(self, num_heads, group_ratio, layer_idx):
        layer = GroupedDifferentialAttention(**_config(num_heads=num_heads, group_ratio=group_ratio))
        x = _x()
        layer(x)
        _randomize(layer)
        got = keras.ops.convert_to_numpy(layer(x, layer_idx=layer_idx, training=False))
        want = _oracle(layer, x, layer_idx)
        np.testing.assert_allclose(got, want, rtol=0, atol=ORACLE_RTOL * max(1.0, np.abs(want).max()))

    def test_matches_the_per_head_loop_with_a_padding_mask(self):
        layer = GroupedDifferentialAttention(**_config())
        x = _x()
        layer(x)
        _randomize(layer)
        mask = np.ones((BATCH, SEQ), dtype="float32")
        mask[:, -4:] = 0.0
        got = keras.ops.convert_to_numpy(layer(x, attention_mask=mask, layer_idx=2, training=False))
        want = _oracle(layer, x, 2, mask=mask)
        np.testing.assert_allclose(got, want, rtol=0, atol=ORACLE_RTOL * max(1.0, np.abs(want).max()))

    def test_the_oracle_is_not_blind_to_the_head_ordering(self):
        """A repeat-interleave instead of a tile would pair signal head i with noise head i // G."""
        layer = GroupedDifferentialAttention(**_config())
        x = _x()
        layer(x)
        _randomize(layer)
        got = keras.ops.convert_to_numpy(layer(x, training=False))

        wrong = GroupedDifferentialAttention(**_config())
        wrong(x)
        wrong.set_weights(layer.get_weights())
        original_tile = keras.ops.tile

        def interleaved(a, repeats):
            return keras.ops.repeat(a, repeats[1], axis=1)

        keras.ops.tile = interleaved
        try:
            swapped = keras.ops.convert_to_numpy(wrong(x, training=False))
        finally:
            keras.ops.tile = original_tile
        assert np.abs(got - swapped).max() > 1e-3


class TestSharingStructure:
    """Which signal heads each projection can reach, read off an identity output projection."""

    H, G, D = 8, 3, 4          # h = 2, 6 signal heads
    DIM_ = 6 * 4

    def _layer(self):
        layer = GroupedDifferentialAttention(
            dim=self.DIM_, num_heads=self.H, head_dim=self.D, group_ratio=self.G)
        x = _x(dim=self.DIM_)
        layer(x)
        _randomize(layer)
        kernel, bias = layer.proj.weights
        kernel.assign(np.eye(self.DIM_, dtype="float32"))
        bias.assign(np.zeros(self.DIM_, dtype="float32"))
        return layer, x

    def _per_head(self, layer, x):
        y = keras.ops.convert_to_numpy(layer(x, layer_idx=1, training=False))
        return y.reshape(BATCH, SEQ, self.G * (self.H // (self.G + 1)), self.D)

    def _changed(self, sub, columns):
        layer, x = self._layer()
        before = self._per_head(layer, x)
        kernel = sub(layer).weights[0]
        noise = np.random.default_rng(5).normal(size=(self.DIM_, columns.stop - columns.start))
        value = keras.ops.convert_to_numpy(kernel).copy()
        value[:, columns] += noise.astype("float32")
        kernel.assign(value)
        after = self._per_head(layer, x)
        return {i for i in range(after.shape[2]) if np.abs(before[:, :, i] - after[:, :, i]).max() > 1e-4}

    @pytest.mark.parametrize("j", [0, 1])
    def test_a_noise_head_reaches_exactly_the_heads_with_its_offset(self, j):
        h = self.H // (self.G + 1)
        changed = self._changed(lambda l: l.q2_dense, slice(j * self.D, (j + 1) * self.D))
        assert changed == {i for i in range(self.G * h) if i % h == j}

    @pytest.mark.parametrize("j", [0, 1])
    def test_a_value_head_reaches_exactly_the_heads_with_its_offset(self, j):
        h = self.H // (self.G + 1)
        changed = self._changed(lambda l: l.v_dense, slice(j * self.D, (j + 1) * self.D))
        assert changed == {i for i in range(self.G * h) if i % h == j}

    @pytest.mark.parametrize("i", [0, 3, 5])
    def test_a_signal_key_reaches_only_its_own_head(self, i):
        changed = self._changed(lambda l: l.k1_dense, slice(i * self.D, (i + 1) * self.D))
        assert changed == {i}

    @pytest.mark.parametrize("i", [1, 4])
    def test_a_signal_query_reaches_only_its_own_head(self, i):
        changed = self._changed(lambda l: l.q1_dense, slice(i * self.D, (i + 1) * self.D))
        assert changed == {i}


class TestLambda:

    def test_layer_idx_changes_the_output_and_lambda_init_follows_the_schedule(self):
        layer = GroupedDifferentialAttention(**_config())
        x = _x()
        layer(x)
        zero_idx = keras.ops.convert_to_numpy(layer(x, layer_idx=0, training=False))
        deep = keras.ops.convert_to_numpy(layer(x, layer_idx=8, training=False))
        assert np.abs(zero_idx - deep).max() > 1e-4
        values = [float(layer.get_lambda_init(i)) for i in (0, 1, 2, 8, 100)]
        np.testing.assert_allclose(values, [0.2, 0.8 - 0.6 * np.exp(-0.3), 0.8 - 0.6 * np.exp(-0.6),
                                            0.8 - 0.6 * np.exp(-2.4), 0.8], atol=1e-6, rtol=0)

    def test_get_lambda_is_the_paper_reparameterisation(self):
        layer = GroupedDifferentialAttention(**_config())
        layer(_x())
        _randomize(layer)
        v = [keras.ops.convert_to_numpy(w).astype("float64") for w in
             (layer.lambda_q1, layer.lambda_k1, layer.lambda_q2, layer.lambda_k2)]
        want = np.exp(v[0] @ v[1]) - np.exp(v[2] @ v[3]) + (0.8 - 0.6 * np.exp(-0.9))
        assert float(layer.get_lambda(3)) == pytest.approx(want, abs=1e-5)

    def test_lambda_vectors_start_small_and_zero_mean(self):
        layer = GroupedDifferentialAttention(**_config(head_dim=64))
        layer(_x())
        stds = [float(np.std(keras.ops.convert_to_numpy(w))) for w in
                (layer.lambda_q1, layer.lambda_k1, layer.lambda_q2, layer.lambda_k2)]
        assert all(0.03 < s < 0.2 for s in stds)

    def test_the_four_lambda_vectors_draw_independently(self):
        layer = GroupedDifferentialAttention(**_config())
        layer(_x())
        vecs = [keras.ops.convert_to_numpy(w) for w in
                (layer.lambda_q1, layer.lambda_k1, layer.lambda_q2, layer.lambda_k2)]
        for a in range(4):
            for b in range(a + 1, 4):
                assert not np.allclose(vecs[a], vecs[b])


class TestForward:

    def test_shape_and_finite(self):
        layer = GroupedDifferentialAttention(**_config())
        y = layer(_x(), training=False)
        assert tuple(y.shape) == (BATCH, SEQ, DIM)
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()

    @pytest.mark.parametrize("rank", [2, 3, 4])
    def test_mask_ranks_run(self, rank):
        layer = GroupedDifferentialAttention(**_config())
        mask = {
            2: np.ones((BATCH, SEQ)),
            3: np.ones((BATCH, SEQ, SEQ)),
            4: np.ones((BATCH, 1, SEQ, SEQ)),
        }[rank].astype("float32")
        y = layer(_x(), attention_mask=mask, training=False)
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()

    def test_an_all_ones_mask_changes_nothing(self):
        layer = GroupedDifferentialAttention(**_config())
        x = _x()
        free = keras.ops.convert_to_numpy(layer(x, training=False))
        masked = keras.ops.convert_to_numpy(
            layer(x, attention_mask=np.ones((BATCH, SEQ), "float32"), training=False))
        np.testing.assert_allclose(free, masked, rtol=0, atol=1e-5)

    def test_a_masked_token_has_no_influence_on_kept_rows(self):
        layer = GroupedDifferentialAttention(**_config())
        x = _x()
        layer(x)
        _randomize(layer)
        mask = np.ones((BATCH, SEQ), "float32")
        mask[:, 5] = 0.0
        other = x.copy()
        other[:, 5] += 3.0
        a = keras.ops.convert_to_numpy(layer(x, attention_mask=mask, training=False))
        b = keras.ops.convert_to_numpy(layer(other, attention_mask=mask, training=False))
        keep = [i for i in range(SEQ) if i != 5]
        np.testing.assert_allclose(a[:, keep], b[:, keep], rtol=0, atol=1e-5)
        assert np.abs(a[:, 5] - b[:, 5]).max() > 1e-3

    def test_a_fully_masked_row_stays_finite(self):
        layer = GroupedDifferentialAttention(**_config())
        mask = np.zeros((BATCH, SEQ, SEQ), "float32")
        mask[:, 1:] = 1.0
        y = layer(_x(), attention_mask=mask, training=False)
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()

    @pytest.mark.filterwarnings("ignore:.*axis has size 1:UserWarning")
    def test_sequence_length_one_is_finite(self):
        # Keras' own Softmax warns that a size-1 axis always yields 1; the output must still be finite.
        layer = GroupedDifferentialAttention(**_config())
        y = layer(_x()[:, :1], training=False)
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()

    def test_dropout_acts_only_in_training(self):
        layer = GroupedDifferentialAttention(**_config(attention_dropout_rate=0.5, dropout_rate=0.5))
        x = _x()
        layer(x)
        a = keras.ops.convert_to_numpy(layer(x, training=False))
        b = keras.ops.convert_to_numpy(layer(x, training=False))
        c = keras.ops.convert_to_numpy(layer(x, training=True))
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
        assert np.abs(a - c).max() > 1e-3

    @pytest.mark.parametrize("probability_type", ["softmax", "sparsemax"])
    def test_probability_types_run(self, probability_type):
        layer = GroupedDifferentialAttention(**_config(probability_type=probability_type))
        y = layer(_x(), training=False)
        assert np.isfinite(keras.ops.convert_to_numpy(y)).all()

    def test_qk_norm_runs_and_changes_the_output(self):
        x = _x()
        plain = GroupedDifferentialAttention(**_config())
        normed = GroupedDifferentialAttention(**_config(qk_norm_type="rms_norm"))
        plain(x)
        normed(x)
        assert normed.q_norm_1 is not None and normed.k_norm_2 is not None
        assert np.isfinite(keras.ops.convert_to_numpy(normed(x, training=False))).all()


class TestGradients:

    def test_every_trainable_weight_receives_a_gradient(self):
        layer = GroupedDifferentialAttention(**_config(qk_norm_type="rms_norm"))
        x = tf.constant(_x())
        layer(x)
        _randomize(layer)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(x, layer_idx=2, training=True) ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        missing = [v.path for v, g in zip(layer.trainable_variables, grads) if g is None]
        assert not missing, missing
        for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
            g = grads[[v.path.split("/")[-1] for v in layer.trainable_variables].index(name)]
            assert float(tf.reduce_max(tf.abs(g))) > 0.0, name


# ---------------------------------------------------------------------

class TestMixedPrecision:

    def test_masked_forward_is_finite_under_every_policy(self, dtype_policy):
        layer = GroupedDifferentialAttention(**_config())
        x = _x()
        mask = np.ones((BATCH, SEQ, SEQ), "float32")
        mask[:, :, -3:] = 0.0
        y = layer(x, attention_mask=mask, layer_idx=4, training=False)
        assert np.isfinite(keras.ops.convert_to_numpy(y).astype("float32")).all()
        expected = {"float32": "float32", "mixed_float16": "float16", "float64": "float64"}[dtype_policy]
        assert keras.backend.standardize_dtype(y.dtype) == expected

    def test_fully_masked_row_is_finite_in_half_precision(self, mixed_float16_policy):
        layer = GroupedDifferentialAttention(**_config())
        mask = np.zeros((BATCH, SEQ, SEQ), "float32")
        mask[:, 2:] = 1.0
        y = layer(_x(), attention_mask=mask, training=False)
        assert np.isfinite(keras.ops.convert_to_numpy(y).astype("float32")).all()


# ---------------------------------------------------------------------

class TestSerialization:

    def test_get_config_covers_every_constructor_argument(self):
        layer = GroupedDifferentialAttention(**_config())
        config = layer.get_config()
        params = {n for n in inspect.signature(GroupedDifferentialAttention.__init__).parameters
                  if n not in ("self", "kwargs")}
        assert params <= set(config), params - set(config)
        assert config["group_ratio"] == 3

    def test_from_config_round_trip(self):
        layer = GroupedDifferentialAttention(**_config(
            qk_norm_type="rms_norm", attention_dropout_rate=0.1, group_ratio=1, name="gda"))
        clone = GroupedDifferentialAttention.from_config(layer.get_config())
        assert clone.get_config() == layer.get_config()

    def test_keras_file_round_trip_restores_weights_and_outputs_exactly(self):
        inputs = keras.Input((SEQ, DIM))
        outputs = GroupedDifferentialAttention(**_config(qk_norm_type="rms_norm"), name="gda")(inputs)
        model = keras.Model(inputs, outputs)
        layer = model.get_layer("gda")
        _randomize(layer, seed=7)
        x = _x(seed=3)
        before = model.predict(x, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gda.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        restored = loaded.get_layer("gda")
        for a, b in zip(layer.weights, restored.weights):
            np.testing.assert_allclose(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b),
                                       rtol=0, atol=0)
        np.testing.assert_allclose(loaded.predict(x, verbose=0), before, rtol=0, atol=1e-6)

    def test_a_functional_model_accepts_a_dynamic_sequence_length(self):
        inputs = keras.Input((None, DIM))
        model = keras.Model(inputs, GroupedDifferentialAttention(**_config())(inputs))
        for length in (4, 9):
            out = model.predict(np.zeros((2, length, DIM), "float32"), verbose=0)
            assert out.shape == (2, length, DIM) and np.isfinite(out).all()

    def test_a_training_step_runs_under_jit_compile(self):
        inputs = keras.Input((SEQ, DIM))
        model = keras.Model(inputs, GroupedDifferentialAttention(**_config())(inputs))
        model.compile(optimizer="adam", loss="mse", jit_compile=True)
        history = model.fit(_x(), _x(seed=9), epochs=1, batch_size=BATCH, verbose=0)
        assert np.isfinite(history.history["loss"]).all()


# ---------------------------------------------------------------------

class TestFactory:

    def test_factory_builds_the_layer_with_its_defaults(self):
        layer = create_attention_layer("grouped_differential", dim=DIM, num_heads=8, head_dim=16)
        assert isinstance(layer, GroupedDifferentialAttention)
        assert layer.group_ratio == 3

    def test_factory_forwards_group_ratio(self):
        layer = create_attention_layer(
            "grouped_differential", dim=DIM, num_heads=10, head_dim=16, group_ratio=4)
        assert (layer.num_signal_heads, layer.num_noise_heads) == (8, 2)

    def test_factory_rejects_indivisible_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            validate_attention_config("grouped_differential", dim=DIM, num_heads=7, head_dim=16)

    def test_factory_rejects_a_non_positive_ratio(self):
        with pytest.raises(ValueError):
            validate_attention_config(
                "grouped_differential", dim=DIM, num_heads=8, head_dim=16, group_ratio=0)

    def test_factory_does_not_require_dim_to_equal_heads_times_head_dim(self):
        validate_attention_config("grouped_differential", dim=DIM, num_heads=8, head_dim=32)


# ---------------------------------------------------------------------
# Compliance with research/2026_keras_custom_models_instructions_v2.md, section 16.1 and 16.3.
# ---------------------------------------------------------------------

from tests.test_models.initializer_replay_oracle import replayed_pairs
from tests.test_models.knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)


def _built(**changes):
    layer = GroupedDifferentialAttention(**_config(**changes))
    layer(_x(), training=False)
    return layer


def _relative(layer):
    return sorted(w.path.split("/", 1)[-1] for w in layer.weights)


class TestBuildMaterialization:
    """Section 8.3: parity of explicit and lazy build, plus its no-sub-layer sibling."""

    @pytest.mark.parametrize("config", [
        dict(), dict(qk_norm_type="rms_norm"), dict(attention_dropout_rate=0.2), dict(group_ratio=1),
    ])
    def test_explicit_build_matches_lazy_build(self, config):
        explicit = GroupedDifferentialAttention(**_config(**config), name="gda")
        explicit.build((None, SEQ, DIM))
        lazy = GroupedDifferentialAttention(**_config(**config), name="gda")
        lazy(_x(), training=False)
        assert _relative(explicit) == _relative(lazy)
        assert _relative(lazy), "no weights: the parity above would be vacuous"

    def test_no_qk_norm_config_builds_no_qk_norm(self):
        plain, normed = _built(), _built(qk_norm_type="rms_norm")
        assert plain.q_norm_1 is None and plain.k_norm_2 is None
        assert not [p for p in _relative(plain) if "norm" in p and "head_norm" not in p]
        # The twin: the same query finds them when the config asks for them.
        assert len([p for p in _relative(normed) if "q_norm" in p or "k_norm" in p]) == 4

    def test_a_zero_attention_dropout_never_runs_the_dropout_layers(self):
        """Dropout has no weights and reports built=True from construction, so the layout is read off
        the calls: created always (section 1.3), run only when the rate is positive."""
        from unittest import mock

        off, on = _built(), _built(attention_dropout_rate=0.2)
        assert off.attn_dropout_1 is not None and off.attn_dropout_2 is not None      # created always
        with mock.patch.object(off.attn_dropout_1, "call", side_effect=AssertionError("ran")), \
                mock.patch.object(off.attn_dropout_2, "call", side_effect=AssertionError("ran")):
            off(_x(), training=True)
        with mock.patch.object(on.attn_dropout_1, "call", side_effect=AssertionError("ran")):
            with pytest.raises(AssertionError, match="ran"):                           # the twin
                on(_x(), training=True)

    def test_every_sublayer_is_named(self):
        layer = _built(qk_norm_type="rms_norm", attention_dropout_rate=0.1)
        names = {l.name for l in layer._flatten_layers(include_self=False)}
        assert {"q1", "k1", "q2", "k2", "v", "proj", "dropout", "attn_dropout_1", "attn_dropout_2",
                "attn_prob_1", "attn_prob_2", "head_norm", "q_norm_1", "k_norm_1", "q_norm_2",
                "k_norm_2"} <= names


class TestStaticContractInCall:
    """Section 3.5: the contract is re-asserted in call(), because build() sees one shape."""

    def test_a_built_layer_refuses_another_feature_width(self):
        layer = _built()
        with pytest.raises(ValueError, match="Expected input"):
            layer(np.zeros((BATCH, SEQ, DIM + 8), "float32"))

    def test_a_built_layer_refuses_another_rank(self):
        layer = _built()
        with pytest.raises(ValueError):
            layer(np.zeros((BATCH, DIM), "float32"))


class TestInitializers:
    """Section 3.1 stores a spec; independent draws and honoured knobs, both ways."""

    def test_the_stored_initializer_is_a_spec_not_an_instance(self):
        layer = GroupedDifferentialAttention(**_config(kernel_initializer=keras.initializers.HeNormal()))
        assert isinstance(layer.kernel_initializer, (str, dict))
        assert isinstance(layer.get_config()["kernel_initializer"], (str, dict))

    def test_no_two_projection_kernels_replay_one_draw(self):
        for kernel_initializer in ("glorot_uniform", keras.initializers.HeNormal()):
            replayed, compared = replayed_pairs(_built(kernel_initializer=kernel_initializer))
            assert compared == 15, compared          # C(6, 2): q1 k1 q2 k2 v proj
            assert not replayed, replayed

    def test_the_replay_oracle_is_not_blind(self):
        """A SEEDED initializer replays by contract (initializers/clone.py), so the oracle must see it."""
        replayed, compared = replayed_pairs(_built(kernel_initializer=keras.initializers.GlorotUniform(seed=7)))
        assert compared == 15 and replayed

    def test_a_reloaded_config_still_replays_no_draw(self):
        clone = GroupedDifferentialAttention.from_config(_built().get_config())
        clone(_x(), training=False)
        assert replayed_pairs(clone)[0] == []

    def test_the_kernel_initializer_reaches_every_projection_and_no_lambda_vector(self):
        layer = _built(kernel_initializer=keras.initializers.Constant(0.5))
        for sub in (layer.q1_dense, layer.k1_dense, layer.q2_dense, layer.k2_dense, layer.v_dense, layer.proj):
            np.testing.assert_array_equal(keras.ops.convert_to_numpy(sub.kernel), 0.5)
        assert float(np.ptp(keras.ops.convert_to_numpy(layer.lambda_q1))) > 0.0       # the twin: scoped

    def test_the_bias_initializer_reaches_every_projection(self):
        layer = _built(bias_initializer="ones")
        for sub in (layer.q1_dense, layer.k1_dense, layer.q2_dense, layer.k2_dense, layer.v_dense, layer.proj):
            np.testing.assert_array_equal(keras.ops.convert_to_numpy(sub.bias), 1.0)
        assert float(np.abs(keras.ops.convert_to_numpy(_built().q1_dense.bias)).max()) == 0.0


class TestKnobs:
    """Section 12.5 and 13.3.2: every constructor knob varied, with the instrument for its class."""

    @staticmethod
    def _builder(**changes):
        return lambda: _built(**changes)

    def test_group_ratio_is_structural(self):
        assert_structural_knob_changes_weights(
            {G: self._builder(num_heads=12, group_ratio=G) for G in (1, 2, 3, 5)}, knob="group_ratio")

    @pytest.mark.parametrize("knob,values", [
        ("head_dim", (8, 16)), ("num_heads", (4, 8)), ("dim", (32, 64)),
    ])
    def test_the_size_knobs_are_structural(self, knob, values):
        def build(value):
            changes = {knob: value}
            layer = GroupedDifferentialAttention(**_config(**changes))
            layer(_x(dim=changes.get("dim", DIM)), training=False)
            return layer
        assert_structural_knob_changes_weights({v: (lambda v=v: build(v)) for v in values}, knob=knob)

    def test_qk_norm_type_is_structural(self):
        assert_structural_knob_changes_weights(
            {t: self._builder(qk_norm_type=t) for t in (None, "rms_norm")}, knob="qk_norm_type")

    def test_probability_type_is_a_value_knob(self):
        assert_value_knob_changes_output(
            {t: self._builder(probability_type=t) for t in ("softmax", "sparsemax")},
            _x(), knob="probability_type", atol=1e-3)

    def test_probability_config_reaches_the_probability_layer(self):
        assert_value_knob_changes_output(
            {t: self._builder(probability_type="adaptive", probability_config={"min_temp": t, "max_temp": t + 5.0})
             for t in (0.1, 2.0)}, _x(), knob="probability_config", atol=1e-3)

    def test_dropout_rates_act_in_training_and_only_there(self):
        x = _x()
        for knob in ("dropout_rate", "attention_dropout_rate"):
            off, on = _built(**{knob: 0.0}), _built(**{knob: 0.5})
            on.set_weights(off.get_weights())
            def run(layer, training):
                keras.utils.set_random_seed(3)
                return keras.ops.convert_to_numpy(layer(x, training=training))
            assert np.abs(run(off, True) - run(on, True)).max() > 1e-3, knob
            np.testing.assert_allclose(run(off, False), run(on, False), rtol=0, atol=0, err_msg=knob)

    @pytest.mark.parametrize("knob", ["kernel_regularizer", "bias_regularizer", "activity_regularizer"])
    def test_every_regularizer_adds_a_loss_term(self, knob):
        # Biases start at zero, where an L2 penalty is zero: start them at one so the term is visible.
        plain = GroupedDifferentialAttention(**_config(bias_initializer="ones"))
        plain(_x())
        regularized = GroupedDifferentialAttention(
            **_config(bias_initializer="ones", **{knob: keras.regularizers.L2(0.1)}))
        regularized(_x())
        assert len(plain.losses) == 0
        assert len(regularized.losses) > 0 and float(sum(regularized.losses)) > 0.0

    def test_the_activity_regularizer_survives_a_config_round_trip(self):
        layer = GroupedDifferentialAttention(**_config(activity_regularizer=keras.regularizers.L2(0.1)))
        clone = GroupedDifferentialAttention.from_config(layer.get_config())
        clone(_x())
        assert float(sum(clone.losses)) > 0.0

    def test_layer_idx_is_a_value_knob(self):
        x = _x()
        layer = _built()
        a, b = (keras.ops.convert_to_numpy(layer(x, layer_idx=k, training=False)) for k in (0, 6))
        assert np.abs(a - b).max() > 1e-3


class TestGradients2:
    """Section 13.2.2, per variable: non-None AND non-zero, named by path."""

    # The key bias is added to every key of one query, so it shifts every score of that query by
    # the same q . b and the softmax cancels it: its gradient is rounding noise by construction.
    # Measured 2026-11-20, CPU, float32, default config: 5.7e-10 (k1/bias) and 8.7e-10 (k2/bias)
    # against 3.2e-03 (smallest other variable). With QK-norm on, the norm sits after the
    # projection, the shift no longer cancels, and the same biases measure 1.1e-04 and 5.7e-05.
    SHIFT_INVARIANT = ("k1/bias", "k2/bias")
    NOISE_CEILING = 1e-8
    LIVE_FLOOR = 1e-5

    @staticmethod
    def _grads(layer):
        x = tf.constant(_x())
        layer(x)
        _randomize(layer)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(layer(x, training=True) ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(layer.trainable_variables) > 0
        return {v.path.split("/", 1)[1]: g for v, g in zip(layer.trainable_variables, grads)}

    def test_every_weight_receives_a_nonzero_gradient_but_the_shift_invariant_biases(self):
        for path, grad in self._grads(GroupedDifferentialAttention(**_config())).items():
            assert grad is not None, f"no gradient for {path}"
            magnitude = float(tf.reduce_max(tf.abs(grad)))
            if path in self.SHIFT_INVARIANT:
                assert magnitude < self.NOISE_CEILING, f"{path}: {magnitude}"
            else:
                assert magnitude > self.LIVE_FLOOR, f"{path}: {magnitude}"

    def test_with_qk_norm_the_key_biases_are_live(self):
        grads = self._grads(GroupedDifferentialAttention(**_config(qk_norm_type="rms_norm")))
        for path in self.SHIFT_INVARIANT:
            assert float(tf.reduce_max(tf.abs(grads[path]))) > self.LIVE_FLOOR, path


class TestDegenerateLengthsAndGraphModes:
    """Section 10.2 and 13.2.7: lengths 0 and 1 on the static path AND a [None, ...] trace."""

    @pytest.mark.parametrize("length", [0, 1])
    @pytest.mark.filterwarnings("ignore:.*axis has size 1:UserWarning")
    def test_static_degenerate_lengths_are_finite(self, length):
        layer = _built()
        y = keras.ops.convert_to_numpy(layer(np.zeros((2, length, DIM), "float32"), training=False))
        assert y.shape == (2, length, DIM) and np.isfinite(y).all()

    @pytest.mark.parametrize("length", [0, 1, SEQ])
    def test_a_none_dimension_trace_is_finite(self, length):
        layer = _built()

        @tf.function(input_signature=[tf.TensorSpec([None, None, DIM], tf.float32)])
        def traced(x):
            return layer(x, training=False)

        y = traced(tf.constant(np.random.default_rng(0).normal(size=(2, length, DIM)).astype("float32"))).numpy()
        assert y.shape == (2, length, DIM) and np.isfinite(y).all()

    def test_xla_matches_eager(self):
        """Measured 2026-11-20, CPU, float32, 8 heads G=3 (outputs absmax 2.79): max|diff| 8.3e-07,
        and 4.2e-07 for the [None, None, D] trace. Bound 1e-5 is 12x above that and ~100x below the
        1e-3 a wrong head pairing moves the output (test_the_oracle_is_not_blind_to_the_head_ordering)."""
        layer = _built()
        _randomize(layer)
        x = _x()
        eager = keras.ops.convert_to_numpy(layer(x, layer_idx=2, training=False))

        @tf.function(jit_compile=True)
        def compiled(a):
            return layer(a, layer_idx=2, training=False)

        out = compiled(tf.constant(x)).numpy()
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, eager, rtol=0, atol=1e-5)


class TestPrecisionArms:
    """Section 13.2.6: the hazard proven, realistic size, a control, and the repair not weakening."""

    LENGTH = 512

    @pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning",
                                "ignore:invalid value encountered:RuntimeWarning")
    def test_the_hazard_is_real(self):
        assert np.float16(-1e9) == -np.inf and np.isnan(np.float16(0.0) * np.float16(-1e9))

    def _masked(self, length=LENGTH, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(2, length, DIM)).astype("float32")
        mask = np.ones((2, length), "float32")
        mask[:, length // 2:] = 0.0
        return x, mask

    def test_half_precision_at_a_realistic_length_matches_the_float32_control(self):
        """Measured 2026-11-20, CPU, N=12, absmax 2.79: max|fp16 - fp32| = 4.5e-03 (0.16%).
        Bound 2e-2 keeps ~4x headroom and still fails loudly on a NaN or a collapsed output."""
        x, mask = self._masked()
        control = GroupedDifferentialAttention(**_config())
        control(x, attention_mask=mask)
        _randomize(control)
        reference = keras.ops.convert_to_numpy(control(x, attention_mask=mask, training=False))
        weights = control.get_weights()

        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            half = GroupedDifferentialAttention(**_config())
            half(x, attention_mask=mask)
            half.set_weights(weights)
            out = half(x, attention_mask=mask, training=False)
            assert keras.backend.standardize_dtype(out.dtype) == "float16"
            out = keras.ops.convert_to_numpy(out).astype("float32")
        finally:
            keras.mixed_precision.set_global_policy(previous)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, reference, rtol=0, atol=2e-2 * max(1.0, np.abs(reference).max()))

    def test_in_half_precision_a_masked_key_still_receives_no_weight(self, mixed_float16_policy):
        """The sentinel was made survivable, not ineffective: perturbing a masked token moves no kept row."""
        x, mask = self._masked()
        layer = GroupedDifferentialAttention(**_config())
        layer(x, attention_mask=mask)
        other = x.copy()
        other[:, -1] += 5.0
        a = keras.ops.convert_to_numpy(layer(x, attention_mask=mask, training=False)).astype("float32")
        b = keras.ops.convert_to_numpy(layer(other, attention_mask=mask, training=False)).astype("float32")
        kept = self.LENGTH // 2
        assert float(np.abs(a[:, :kept] - b[:, :kept]).max()) == 0.0
        assert float(np.abs(a[:, -1] - b[:, -1]).max()) > 1e-3          # the twin: the bump was real

    def test_float64_agrees_with_float32_to_rounding(self, float64_policy):
        """Measured 2026-11-20, CPU: max|f64 - f32| = 1.4e-06 at output absmax 2.79, i.e. float32 rounding."""
        x = _x()
        reference = GroupedDifferentialAttention(**_config())
        # built under float32 on purpose: the control
        keras.mixed_precision.set_global_policy("float32")
        reference(x)
        _randomize(reference)
        want = keras.ops.convert_to_numpy(reference(x, training=False))
        weights = reference.get_weights()
        keras.mixed_precision.set_global_policy("float64")

        layer = GroupedDifferentialAttention(**_config())
        x64 = keras.ops.convert_to_tensor(x.astype("float64"))
        assert keras.backend.standardize_dtype(x64.dtype) == "float64"
        layer(x64)
        layer.set_weights([w.astype("float64") for w in weights])
        got = layer(x64, training=False)
        assert keras.backend.standardize_dtype(got.dtype) == "float64"
        np.testing.assert_allclose(keras.ops.convert_to_numpy(got), want, rtol=0, atol=2e-5)


class TestTokenMixing:
    """Section 12.6: a token-mixing layer carries a non-cyclic permutation probe with an anti-vacuity arm."""

    def test_the_layer_is_permutation_equivariant_without_a_mask(self):
        layer = _built()
        _randomize(layer)
        x = _x()
        perm = np.random.default_rng(4).permutation(SEQ)
        assert not np.array_equal(perm, np.roll(np.arange(SEQ), 1))               # non-cyclic
        a = keras.ops.convert_to_numpy(layer(x[:, perm], training=False))
        b = keras.ops.convert_to_numpy(layer(x, training=False))[:, perm]
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-5)

    def test_tokens_really_mix(self):
        """The anti-vacuity arm: a layer that ignored its other tokens would pass the probe above."""
        layer = _built()
        _randomize(layer)
        x = _x()
        other = x.copy()
        other[:, 0] += 3.0
        a = keras.ops.convert_to_numpy(layer(x, training=False))
        b = keras.ops.convert_to_numpy(layer(other, training=False))
        assert float(np.abs(a[:, 1:] - b[:, 1:]).max()) > 1e-3

    def test_a_masking_mask_changes_the_output_where_an_all_ones_mask_does_not(self):
        """The twin of test_an_all_ones_mask_changes_nothing."""
        layer = _built()
        _randomize(layer)
        x = _x()
        free = keras.ops.convert_to_numpy(layer(x, training=False))
        mask = np.ones((BATCH, SEQ), "float32")
        mask[:, :4] = 0.0
        assert np.abs(free - keras.ops.convert_to_numpy(layer(x, attention_mask=mask, training=False))).max() > 1e-3


class TestRoundTripOnValues:
    """Section 7.1 and 8.4 exactly: training=False, rtol=0, values compared before the first call."""

    def test_the_keras_round_trip_is_value_exact(self):
        inputs = keras.Input(shape=(SEQ, DIM))
        outputs = GroupedDifferentialAttention(
            **_config(qk_norm_type="rms_norm", attention_dropout_rate=0.1), name="gda")(inputs)
        model = keras.Model(inputs, outputs)
        _randomize(model.get_layer("gda"), seed=11)
        saved = [keras.ops.convert_to_numpy(w).copy() for w in model.get_layer("gda").weights]
        assert saved, "no weights to compare"
        x = _x(seed=5)
        original = keras.ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gda.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        restored_layer = loaded.get_layer("gda")
        for want, w in zip(saved, restored_layer.weights):         # BEFORE the first forward on `loaded`
            np.testing.assert_allclose(want, keras.ops.convert_to_numpy(w), rtol=0, atol=0.0, err_msg=w.path)
        assert restored_layer.count_params() == model.get_layer("gda").count_params()
        np.testing.assert_allclose(
            original, keras.ops.convert_to_numpy(loaded(x, training=False)), rtol=0, atol=1e-6)
