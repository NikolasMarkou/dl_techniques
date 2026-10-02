"""Tests for ``losses/lightglue_loss.py``.

The oracle is an explicit-loop numpy transcription of the cvg/glue-factory source read on
2026-10-02 (``NLLLoss`` / ``weight_loss`` in ``models/utils/losses.py`` and
``LightGlue.loss`` / ``TokenConfidence.loss`` in ``models/matchers/lightglue.py``).
Torch is not installed, so numpy is the parity ground truth.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses import LightGlueLoss as ExportedLoss
from dl_techniques.losses.lightglue_loss import (
    LightGlueLoss,
    layer_weights,
    lightglue_confidence_loss,
    lightglue_nll,
    pack_matches,
    unpack_matches,
)
from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue

B, L, M, N = 3, 4, 6, 5


# --------------------------------------------------------------------------------------
# numpy oracle (explicit loops, glue-factory reading)
# --------------------------------------------------------------------------------------

def oracle_layer_nll(la, m0, m1, b):
    """``NLLLoss.forward`` for one sample and one layer; ``la`` is (M+1, N+1)."""
    m, n = len(m0), len(m1)
    pos_sum, num_pos = 0.0, 0
    for i in range(m):
        if m0[i] >= 0:
            pos_sum += la[i, m0[i]]
            num_pos += 1
    neg0_sum, num_neg0 = 0.0, 0
    for i in range(m):
        if m0[i] == -1:
            neg0_sum += la[i, n]
            num_neg0 += 1
    neg1_sum, num_neg1 = 0.0, 0
    for j in range(n):
        if m1[j] == -1:
            neg1_sum += la[m, j]
            num_neg1 += 1
    nll_pos = -pos_sum / max(num_pos, 1)
    nll_neg = -(neg0_sum + neg1_sum) / (max(num_neg0, 1) + max(num_neg1, 1))
    return b * nll_pos + (1 - b) * nll_neg


def oracle_nll(la, m0, m1, gamma=1.0, b=0.5):
    out = []
    for s in range(la.shape[0]):
        n_layers = la.shape[1]
        total = oracle_layer_nll(la[s, -1], m0[s], m1[s], b)
        sum_w = 1.0
        for i in range(n_layers - 1):
            w = gamma ** (n_layers - i - 1) if gamma > 0.0 else i + 1
            total += w * oracle_layer_nll(la[s, i], m0[s], m1[s], b)
            sum_w += w
        out.append(total / sum_w)
    return np.array(out)


def oracle_confidence(la, c0, c1, real0=None, real1=None):
    out = []
    for s in range(la.shape[0]):
        n_layers, mp1, np1 = la.shape[1:]
        m, n = mp1 - 1, np1 - 1
        r0 = np.arange(m) if real0 is None else np.flatnonzero(real0[s])
        r1 = np.arange(n) if real1 is None else np.flatnonzero(real1[s])
        cols = list(r1) + [n]
        rows = list(r0) + [m]
        total = 0.0
        for i in range(n_layers - 1):
            def row_arg(layer, r):
                vals = [la[s, layer, r, c] for c in cols]
                return cols[int(np.argmax(vals))]

            def col_arg(layer, c):
                vals = [la[s, layer, r, c] for r in rows]
                return rows[int(np.argmax(vals))]

            b0 = 0.0
            for r in r0:
                t = float(row_arg(n_layers - 1, r) == row_arg(i, r))
                p = np.clip(c0[s, i, r], 1e-7, 1 - 1e-7)
                b0 += -(t * np.log(p) + (1 - t) * np.log(1 - p))
            b1 = 0.0
            for c in r1:
                t = float(col_arg(n_layers - 1, c) == col_arg(i, c))
                p = np.clip(c1[s, i, c], 1e-7, 1 - 1e-7)
                b1 += -(t * np.log(p) + (1 - t) * np.log(1 - p))
            total += (b0 / max(len(r0), 1) + b1 / max(len(r1), 1)) / 2.0
        out.append(total / (n_layers - 1))
    return np.array(out)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def _labels(rng, batch, m, n, n_match=3, n_ignored=1):
    m0 = -np.ones((batch, m), "int32")
    m1 = -np.ones((batch, n), "int32")
    for s in range(batch):
        rows = rng.permutation(m)[:n_match]
        cols = rng.permutation(n)[:n_match]
        m0[s, rows] = cols
        m1[s, cols] = rows
        free0 = [i for i in range(m) if m0[s, i] == -1]
        free1 = [j for j in range(n) if m1[s, j] == -1]
        m0[s, free0[:n_ignored]] = -2
        m1[s, free1[:n_ignored]] = -2
    return m0, m1


def _la(rng, batch=B, layers=L, m=M, n=N):
    return (-rng.uniform(0.05, 3.0, (batch, layers, m + 1, n + 1))).astype("float32")


def _conf(rng, batch=B, layers=L, m=M, n=N):
    return (rng.uniform(0.05, 0.95, (batch, layers - 1, m)).astype("float32"),
            rng.uniform(0.05, 0.95, (batch, layers - 1, n)).astype("float32"))


def _np(x):
    return keras.ops.convert_to_numpy(x)


@pytest.fixture
def case():
    rng = np.random.RandomState(0)
    m0, m1 = _labels(rng, B, M, N)
    return _la(rng), m0, m1, _conf(rng)


# --------------------------------------------------------------------------------------
# NLL
# --------------------------------------------------------------------------------------

class TestNll:
    def test_matches_oracle(self, case):
        la, m0, m1, _ = case
        np.testing.assert_allclose(_np(lightglue_nll(la, m0, m1)), oracle_nll(la, m0, m1),
                                   rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("gamma", [0.5, 2.0, 0.0])
    @pytest.mark.parametrize("balancing", [0.5, 0.2])
    def test_gamma_and_balancing_match_oracle(self, case, gamma, balancing):
        la, m0, m1, _ = case
        got = _np(lightglue_nll(la, m0, m1, gamma=gamma, nll_balancing=balancing))
        np.testing.assert_allclose(got, oracle_nll(la, m0, m1, gamma, balancing),
                                   rtol=1e-5, atol=1e-6)

    def test_hand_computed_2x2(self):
        la = np.array([[[[-0.5, -2.0, -3.0],
                         [-1.5, -0.25, -0.75],
                         [-2.5, -1.0, -9.0]]]], "float32")        # (1, 1, 3, 3)
        m0 = np.array([[0, -1]], "int32")
        m1 = np.array([[0, -1]], "int32")
        # pos: -(-0.5)/1 = 0.5 ; neg: -(la[1,2] + la[2,1]) / 2 = (0.75 + 1.0)/2 = 0.875
        expected = 0.5 * 0.5 + 0.5 * 0.875
        got = float(_np(lightglue_nll(la, m0, m1))[0])
        assert got == pytest.approx(expected, abs=1e-6)

    def test_layer_weights_hand_computed(self):
        # two layers, gamma=0.5: weights [0.5, 1]; layer 0 nll 1.0, layer 1 nll 3.0
        la = np.zeros((1, 2, 2, 2), "float32")
        la[0, 0, 0, 0] = -1.0
        la[0, 1, 0, 0] = -3.0
        m0 = np.array([[0]], "int32")
        m1 = np.array([[0]], "int32")
        got = float(_np(lightglue_nll(la, m0, m1, gamma=0.5, nll_balancing=1.0))[0])
        assert got == pytest.approx((0.5 * 1.0 + 1.0 * 3.0) / 1.5, abs=1e-6)
        assert layer_weights(3, 0.0) == [1.0, 2.0, 1.0]
        assert layer_weights(3, 0.5) == [0.25, 0.5, 1.0]

    def test_all_ignored_batch_is_zero_and_finite(self, case):
        la, _, _, _ = case
        m0 = -2 * np.ones((B, M), "int32")
        m1 = -2 * np.ones((B, N), "int32")
        value = _np(lightglue_nll(la, m0, m1))
        assert np.all(np.isfinite(value)) and np.all(value == 0.0)
        x = tf.constant(la)
        with tf.GradientTape() as tape:
            tape.watch(x)
            total = keras.ops.sum(lightglue_nll(x, m0, m1))
        grad = tape.gradient(total, x).numpy()
        assert np.all(np.isfinite(grad)) and np.all(grad == 0.0)

    def test_only_one_side_supervised_is_finite(self, case):
        la, _, _, _ = case
        m0 = np.full((B, M), -1, "int32")
        m1 = np.full((B, N), -2, "int32")
        np.testing.assert_allclose(_np(lightglue_nll(la, m0, m1)), oracle_nll(la, m0, m1),
                                   rtol=1e-5, atol=1e-6)

    def test_padded_and_ignored_do_not_contribute(self, case):
        la, m0, m1, _ = case
        base = _np(lightglue_nll(la, m0, m1))
        edited = la.copy()
        for s in range(B):
            for i in np.flatnonzero(m0[s] == -2):
                edited[s, :, i, :] = np.nan            # whole ignored row
            for j in np.flatnonzero(m1[s] == -2):
                edited[s, :, :, j] = np.nan            # whole ignored column
            for i in np.flatnonzero(m0[s] == -2):
                for j in np.flatnonzero(m1[s] == -2):
                    edited[s, :, i, j] = np.nan
        # restore the entries that are supervised through the OTHER axis' dustbin
        # (an ignored row's dustbin cell is never read, so NaN there is fine too)
        value = _np(lightglue_nll(edited, m0, m1))
        assert np.all(np.isfinite(value))
        np.testing.assert_allclose(value, base, rtol=1e-6, atol=1e-7)

    def test_balancing_extremes_isolate_each_term(self, case):
        la, m0, m1, _ = case
        only_pos = _np(lightglue_nll(la, m0, m1, nll_balancing=1.0))
        only_neg = _np(lightglue_nll(la, m0, m1, nll_balancing=0.0))
        both = _np(lightglue_nll(la, m0, m1))
        np.testing.assert_allclose(both, 0.5 * only_pos + 0.5 * only_neg, rtol=1e-5)
        assert not np.allclose(only_pos, only_neg)
        # the positive-only value does not move when a dustbin cell changes
        edited = la.copy()
        edited[:, :, :M, N] -= 5.0
        edited[:, :, M, :N] -= 5.0
        np.testing.assert_allclose(_np(lightglue_nll(edited, m0, m1, nll_balancing=1.0)),
                                   only_pos, rtol=1e-6)
        assert not np.allclose(_np(lightglue_nll(edited, m0, m1, nll_balancing=0.0)),
                               only_neg)

    def test_static_shape_required(self):
        with pytest.raises(ValueError, match="static"):
            lightglue_nll(keras.KerasTensor((2, None, 4, 4)),
                          np.zeros((2, 3), "int32"), np.zeros((2, 3), "int32"))


# --------------------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------------------

class TestConfidence:
    def test_matches_oracle(self, case):
        la, _, _, (c0, c1) = case
        got = _np(lightglue_confidence_loss(la, c0, c1))
        np.testing.assert_allclose(got, oracle_confidence(la, c0, c1), rtol=1e-5, atol=1e-6)

    def test_matches_oracle_with_padding(self, case):
        la, _, _, (c0, c1) = case
        real0 = np.arange(M)[None, :] < np.array([[M], [4], [5]])
        real1 = np.arange(N)[None, :] < np.array([[N], [3], [N]])
        padded = la.copy()
        for s in range(B):                         # padded rows/cols are 0 in the model
            for k in np.flatnonzero(~real1[s]):
                padded[s, :, :, k] = 0.0
            for k in np.flatnonzero(~real0[s]):
                padded[s, :, k, :] = 0.0
        got = _np(lightglue_confidence_loss(padded, c0, c1, real0.astype("float32"),
                                            real1.astype("float32")))
        np.testing.assert_allclose(got, oracle_confidence(padded, c0, c1, real0, real1),
                                   rtol=1e-5, atol=1e-6)
        # padded confidences are irrelevant
        c0b = c0.copy()
        c0b[1, :, 4:] = 0.123
        again = _np(lightglue_confidence_loss(padded, c0b, c1, real0.astype("float32"),
                                              real1.astype("float32")))
        np.testing.assert_allclose(got, again, rtol=1e-6)

    def test_target_is_agreement_with_final_layer(self):
        # layer 0 argmax differs from final for token 0 only; prob 0.9 everywhere
        la = np.full((1, 2, 3, 3), -5.0, "float32")
        for r, c in ((0, 0), (1, 2), (2, 1)):      # final layer maxima
            la[0, 1, r, c] = -0.1
        for r, c in ((0, 1), (1, 2), (2, 0)):      # layer 0 maxima
            la[0, 0, r, c] = -0.1
        # rows: row0 final->col0, layer0->col1 (disagree); row1 both dustbin (agree)
        # cols: col0 final->row0, layer0->dustbin row (disagree); col1 final->dustbin
        #       row, layer0->row0 (disagree)
        c0 = np.full((1, 1, 2), 0.9, "float32")
        c1 = np.full((1, 1, 2), 0.9, "float32")
        got = float(_np(lightglue_confidence_loss(la, c0, c1))[0])
        np.testing.assert_allclose(got, oracle_confidence(la, c0, c1)[0], rtol=1e-6)
        row = (-np.log(0.1) - np.log(0.9)) / 2
        col = -np.log(0.1)
        assert got == pytest.approx((row + col) / 2, rel=1e-5)

    def test_single_layer_is_zero(self):
        la = np.zeros((2, 1, 4, 4), "float32")
        c = np.zeros((2, 0, 3), "float32")
        assert np.all(_np(lightglue_confidence_loss(la, c, c)) == 0.0)

    def test_saturated_confidences_are_finite(self, case):
        la, _, _, _ = case
        c0 = np.zeros((B, L - 1, M), "float32")
        c1 = np.ones((B, L - 1, N), "float32")
        assert np.all(np.isfinite(_np(lightglue_confidence_loss(la, c0, c1))))


# --------------------------------------------------------------------------------------
# the Loss object
# --------------------------------------------------------------------------------------

class TestLossObject:
    def test_export(self):
        assert ExportedLoss is LightGlueLoss

    def test_call_with_packed_labels_equals_oracle(self, case):
        la, m0, m1, _ = case
        loss = LightGlueLoss()
        got = float(_np(loss(_np(pack_matches(m0, m1)), la)))
        assert got == pytest.approx(float(oracle_nll(la, m0, m1).mean()), rel=1e-5)

    def test_unpack_inverts_pack_for_int_and_float(self, case):
        _, m0, m1, _ = case
        packed = pack_matches(m0, m1)
        for arr in (packed, keras.ops.cast(packed, "float32")):
            a, b = unpack_matches(arr, M)
            np.testing.assert_array_equal(_np(a), m0)
            np.testing.assert_array_equal(_np(b), m1)

    def test_compute_adds_scaled_confidence(self, case):
        la, m0, m1, (c0, c1) = case
        nll = oracle_nll(la, m0, m1)
        conf = oracle_confidence(la, c0, c1)
        for w in (0.0, 1.0, 2.5):
            got = _np(LightGlueLoss(confidence_weight=w).compute(la, m0, m1, c0, c1))
            np.testing.assert_allclose(got, nll + w * conf, rtol=1e-5, atol=1e-6)

    def test_compute_without_confidences_is_nll_only(self, case):
        la, m0, m1, _ = case
        np.testing.assert_allclose(_np(LightGlueLoss().compute(la, m0, m1)),
                                   oracle_nll(la, m0, m1), rtol=1e-5, atol=1e-6)

    def test_each_term_weight_is_live(self, case):
        la, m0, m1, (c0, c1) = case
        base = _np(LightGlueLoss().compute(la, m0, m1, c0, c1))
        for kw in ({"confidence_weight": 0.0}, {"nll_balancing": 1.0},
                   {"nll_balancing": 0.0}, {"gamma": 0.3}):
            other = _np(LightGlueLoss(**kw).compute(la, m0, m1, c0, c1))
            assert not np.allclose(base, other), kw

    def test_gradient_flows_to_log_assignments_not_through_target(self, case):
        la, m0, m1, (c0, c1) = case
        x = tf.constant(la)
        grads = {}
        for w in (0.0, 5.0):
            with tf.GradientTape() as tape:
                tape.watch(x)
                total = keras.ops.mean(LightGlueLoss(confidence_weight=w).compute(
                    x, m0, m1, c0, c1))
            grads[w] = tape.gradient(total, x).numpy()
        assert np.all(np.isfinite(grads[0.0])) and np.abs(grads[0.0]).sum() > 0
        # the BCE target is stop-gradient: the confidence term adds no gradient to la
        np.testing.assert_array_equal(grads[0.0], grads[5.0])
        # and it does give a gradient to the confidence predictions
        c = tf.constant(c0)
        with tf.GradientTape() as tape:
            tape.watch(c)
            total = keras.ops.mean(LightGlueLoss().compute(la, m0, m1, c, c1))
        assert np.abs(tape.gradient(total, c).numpy()).sum() > 0

    def test_sample_weight_selects_rows(self, case):
        # losses/CLAUDE.md: one value per sample, so a zero-weight row costs nothing
        la, m0, m1, _ = case
        packed = _np(pack_matches(m0, m1))
        loss = LightGlueLoss()
        weighted = float(_np(loss(packed, la, sample_weight=np.array([1.0, 1.0, 0.0], "float32"))))
        rest = float(_np(loss(packed[:2], la[:2])))
        assert weighted * 3.0 / 2.0 == pytest.approx(rest, rel=1e-5)
        assert _np(loss.call(packed, la)).shape == (B,)

    def test_validation(self):
        with pytest.raises(ValueError, match="nll_balancing"):
            LightGlueLoss(nll_balancing=1.5)
        with pytest.raises(ValueError, match="confidence_weight"):
            LightGlueLoss(confidence_weight=-1.0)

    def test_config_round_trip(self):
        loss = LightGlueLoss(gamma=0.7, nll_balancing=0.3, confidence_weight=0.5, name="lg")
        config = loss.get_config()
        assert {"gamma", "nll_balancing", "confidence_weight", "name"} <= set(config)
        again = LightGlueLoss.from_config(config)
        assert again.get_config() == config

    def test_keras_serialize_round_trip(self, case):
        la, m0, m1, _ = case
        loss = LightGlueLoss(gamma=0.7, nll_balancing=0.3, confidence_weight=0.5)
        again = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(loss))
        assert isinstance(again, LightGlueLoss)
        assert again.get_config() == loss.get_config()
        packed = _np(pack_matches(m0, m1))
        assert float(_np(again(packed, la))) == pytest.approx(float(_np(loss(packed, la))))

    def test_mixed_float16_inputs(self, case):
        la, m0, m1, (c0, c1) = case
        packed = _np(pack_matches(m0, m1))
        ref = float(_np(LightGlueLoss()(packed, la)))
        half = float(_np(LightGlueLoss()(packed, la.astype("float16"))))
        assert np.isfinite(half) and half == pytest.approx(ref, rel=2e-2)
        out = _np(LightGlueLoss().compute(la.astype("float16"), m0, m1,
                                          c0.astype("float16"), c1.astype("float16")))
        assert out.dtype == np.float32 and np.all(np.isfinite(out))

    def test_graph_mode_equals_eager(self, case):
        la, m0, m1, (c0, c1) = case
        loss = LightGlueLoss()
        eager = _np(loss.compute(la, m0, m1, c0, c1))

        @tf.function
        def fn(a, x0, x1, p0, p1):
            return loss.compute(a, x0, x1, p0, p1)

        np.testing.assert_allclose(fn(la, m0, m1, c0, c1).numpy(), eager, rtol=1e-6)

    def test_dict_arguments_are_not_part_of_the_interface(self):
        # probed on Keras 3.8: a direct call takes dicts, stock fit() does not; the
        # contract is therefore packed tensors (module docstring)
        inp = keras.Input((3,))
        model = keras.Model(inp, {"a": keras.layers.Dense(3)(inp)})
        model.compile(optimizer="sgd", loss=LightGlueLoss())
        with pytest.raises(Exception):
            model.fit(np.ones((4, 3), "float32"), {"t": np.ones((4, 3), "int32")},
                      batch_size=2, epochs=1, verbose=0)


class TestStockFit:
    def test_real_fit_step_through_compile_with_lightglue(self):
        from tests.test_models.test_lightglue.weight_loading import random_inputs

        d, m, n = 16, 8, 7
        rng = np.random.RandomState(0)
        model = LightGlue(input_dim=d, descriptor_dim=d, num_layers=2, num_heads=2)
        data = random_inputs(rng, 4, m, n, d)
        m0, m1 = _labels(rng, 4, m, n)
        model(data)
        before = [v.numpy().copy() for v in model.trainable_variables]
        model.compile(optimizer=keras.optimizers.SGD(0.1),
                      loss={"log_assignments": LightGlueLoss()}, jit_compile=False)
        hist = model.fit(data, {"log_assignments": _np(pack_matches(m0, m1))},
                         batch_size=2, epochs=1, verbose=0)
        loss = hist.history["loss"][0]
        assert np.isfinite(loss) and loss > 0
        moved = [not np.array_equal(b, v.numpy())
                 for b, v in zip(before, model.trainable_variables)]
        assert any(moved)
