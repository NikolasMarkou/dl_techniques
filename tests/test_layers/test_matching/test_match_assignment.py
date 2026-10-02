"""Tests for MatchAssignment and filter_matches.

The oracle (``tests/lightglue_reference_numpy.py``) transcribes the reference, which has NO
mask: it slices padding off first. So parity runs on UNPADDED real keypoints, and the
layer's padded-row/column masking (a deliberate extension) is tested separately as
invariance: appending padding must leave the real rows, real columns and the dustbin
unchanged.

Derived facts the sanity tests rely on (probability space, real row ``i``):
``S_ij = r_ij c_ij s0_i s1_j`` with ``r`` the row softmax (``sum_j r_ij = 1``), ``c`` the
column softmax (``c_ij <= 1``), ``s0 = sigmoid(z0)``. Hence
``sum_j exp(S_ij) <= s0_i`` and ``sum_j exp(S_ij) + exp(S_i,dust) <= s0_i + (1 - s0_i) = 1``,
and it is ``>= exp(S_i,dust) = 1 - s0_i``. The assignment is NOT normalised to one; the
tests assert the bounds, never equality.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.matching import match_assignment
from dl_techniques.layers.matching.match_assignment import MatchAssignment, filter_matches
from tests import lightglue_reference_numpy as ref
from tests.test_models.test_sam.dead_component_oracle import (
    NO_GRADIENTS_MESSAGE,
    component_response,
    fit_one_step_moved_variables,
    no_op_kill,
    outputs_stop_gradient,
    zeroed_variables,
)

# TF32 is on by default on this GPU and turns a float32 matmul into a ~1e-3 relative error,
# which a derived eps32 tolerance rightly rejects (measured: 3.6e-3 against the oracle with
# TF32 on, 5.8e-7 off). Scope it off for this module through the shared fixture.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

_EPS = {"float16": float(np.finfo(np.float16).eps), "float32": float(np.finfo(np.float32).eps),
        "float64": float(np.finfo(np.float64).eps)}
_NP = {"float16": np.float16, "bfloat16": np.float32, "float32": np.float32, "float64": np.float64}
PFX = "log_assignment.0"
D = 16


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _round(a, cdt):
    return a.astype(_NP[cdt]).astype(np.float64)


def _weights(cdt, seed=0):
    full = ref.random_weights(np.random.RandomState(seed), D, 2, 2)
    return {k: _round(v, cdt) for k, v in full.items()}


def _assign(variable, value):
    variable.assign(np.asarray(value, dtype=variable.dtype))


def _layer(weights):
    layer = MatchAssignment(dim=D)
    layer.build((None, None, D), (None, None, D))
    for name in ("final_proj", "matchability"):
        sub = getattr(layer, name)
        _assign(sub.kernel, weights[f"{PFX}.{name}.weight"].T)
        _assign(sub.bias, weights[f"{PFX}.{name}.bias"])
    return layer


def _np(t):
    return keras.ops.convert_to_numpy(t).astype(np.float64)


def _t(a, cdt):
    return keras.ops.convert_to_tensor(a.astype(_NP[cdt]))


def _descs(rng, cdt, b=2, m=5, n=7):
    return _round(rng.normal(size=(b, m, D)), cdt), _round(rng.normal(size=(b, n, D)), cdt)


def _tol(cdt, weights, d0, d1, expected):
    """Parity tolerance DERIVED from the arithmetic, per test case.

    Error sources, in the compute dtype with unit roundoff ``eps``: ``final_proj`` and
    ``matchability`` are Dense layers with fan-in D, relative error ``(D + 1) eps`` against
    the absolute sums ``|x||W| + |b|``. The similarity is an einsum in at least float32
    over those, so its error is ``e0 |md1| + |md0| e1``. ``log_softmax`` has Lipschitz
    constant 2 in the max norm; the interior uses two of them (row and column), so a sim
    error enters at most 4 times; ``log_sigmoid`` is 1-Lipschitz so ``z0`` and ``z1`` each
    enter once. A float32 slack ``64 eps32 (1 + max|S|)`` covers the softmax/sum stages;
    float64 uses a flat 1e-10.
    """
    if cdt == "float64":
        return 1e-10
    eps = _EPS[cdt]
    quarter = float(D) ** -0.25
    fw, fb = np.abs(weights[f"{PFX}.final_proj.weight"]), np.abs(weights[f"{PFX}.final_proj.bias"])
    mw, mb = np.abs(weights[f"{PFX}.matchability.weight"]), np.abs(weights[f"{PFX}.matchability.bias"])
    worst = 0.0
    for b in range(d0.shape[0]):
        a0 = (np.abs(d0[b]) @ fw.T + fb) * quarter
        a1 = (np.abs(d1[b]) @ fw.T + fb) * quarter
        e0, e1 = (D + 1) * eps * a0, (D + 1) * eps * a1
        sim_err = (e0 @ a1.T + a0 @ e1.T).max()
        z0e = ((D + 1) * eps * (np.abs(d0[b]) @ mw.T + mb)).max()
        z1e = ((D + 1) * eps * (np.abs(d1[b]) @ mw.T + mb)).max()
        worst = max(worst, 4 * sim_err + z0e + z1e)
    return worst + 64 * _EPS["float32"] * (1.0 + float(np.abs(expected).max()))


def _oracle(weights, d0, d1):
    """Per-sample oracle ``(scores (B, M+1, N+1), sim (B, M, N))`` on unpadded inputs."""
    outs = [ref.match_assignment(d0[b], d1[b], weights, PFX) for b in range(d0.shape[0])]
    return np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs])


def _err(layer, weights, d0, d1, cdt):
    scores, sim = layer(_t(d0, cdt), _t(d1, cdt))
    exp_scores, exp_sim = _oracle(weights, d0, d1)
    return (np.abs(_np(scores) - exp_scores).max(), np.abs(_np(sim) - exp_sim).max(),
            _tol(cdt, weights, d0, d1, exp_scores))


def _masks(b, n, n_real):
    mask = np.zeros((b, n), "float32")
    mask[:, :n_real] = 1.0
    return mask


# ---------------------------------------------------------------------
# oracle parity (unpadded)
# ---------------------------------------------------------------------


class TestParity:
    def test_matches_oracle(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        weights = _weights(cdt)
        d0, d1 = _descs(np.random.RandomState(1), cdt)             # M != N catches a transposed axis
        err, sim_err, tol = _err(_layer(weights), weights, d0, d1, cdt)
        assert err <= tol, f"{cdt}: scores err {err} > tol {tol}"
        assert sim_err <= tol, f"{cdt}: sim err {sim_err} > tol {tol}"

    def test_output_dtype_is_at_least_float32(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        weights = _weights(cdt)
        d0, d1 = _descs(np.random.RandomState(2), cdt)
        scores, _ = _layer(weights)(_t(d0, cdt), _t(d1, cdt))
        assert keras.backend.standardize_dtype(scores.dtype) == ("float64" if cdt == "float64" else "float32")

    def test_get_matchability_matches_oracle(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        weights = _weights(cdt)
        d0, _ = _descs(np.random.RandomState(3), cdt)
        got = _np(_layer(weights).get_matchability(_t(d0, cdt)))
        want = np.stack([ref.matchability(d0[b], weights, PFX) for b in range(d0.shape[0])])
        assert got.shape == want.shape == (2, 5)
        tol = 1e-10 if cdt == "float64" else (D + 2) * _EPS[cdt] * 2.0
        assert np.abs(got - want).max() <= tol

    def test_corner_and_dustbin_structure(self):
        weights = _weights("float32")
        d0, d1 = _descs(np.random.RandomState(4), "float32", m=4, n=6)
        scores = _np(_layer(weights)(_t(d0, "float32"), _t(d1, "float32"))[0])
        assert scores.shape == (2, 5, 7)
        np.testing.assert_array_equal(scores[:, 4, 6], 0.0)
        z0 = np.stack([ref.linear(d0[b], weights, PFX + ".matchability")[:, 0] for b in range(2)])
        np.testing.assert_allclose(scores[:, :4, 6], ref.log_sigmoid(-z0), atol=1e-5)


# ---------------------------------------------------------------------
# row sums, derived bounds
# ---------------------------------------------------------------------


class TestRowColumnMass:
    """Bounds derived in the module docstring, checked against oracle AND layer."""

    def _totals(self, scores):
        p = np.exp(scores)
        rows = p[:, :-1, :-1].sum(-1) + p[:, :-1, -1]           # sum_j over real cols + dustbin
        cols = p[:, :-1, :-1].sum(1) + p[:, -1, :-1]
        return rows, cols

    def test_bounds_hold_and_it_is_not_normalised(self):
        weights = _weights("float32", seed=5)
        d0, d1 = _descs(np.random.RandomState(5), "float32", b=3, m=6, n=9)
        exp_scores, _ = _oracle(weights, d0, d1)
        got = _np(_layer(weights)(_t(d0, "float32"), _t(d1, "float32"))[0])
        slack = 1e-5                                   # float32: a few dozen eps32 on sums of exps <= 1
        z0 = np.stack([ref.linear(d0[b], weights, PFX + ".matchability")[:, 0] for b in range(3)])
        z1 = np.stack([ref.linear(d1[b], weights, PFX + ".matchability")[:, 0] for b in range(3)])
        s0, s1 = ref.sigmoid(z0), ref.sigmoid(z1)
        for scores in (exp_scores, got):
            rows, cols = self._totals(scores)
            assert np.all(rows <= 1.0 + slack) and np.all(rows >= 1.0 - s0 - slack)
            assert np.all(cols <= 1.0 + slack) and np.all(cols >= 1.0 - s1 - slack)
            # interior mass alone is at most the keypoint's own matchability
            assert np.all(np.exp(scores)[:, :-1, :-1].sum(-1) <= s0 + slack)
            # and the totals are NOT one: asserting normalisation would be a false claim
            assert rows.max() < 1.0 - 1e-3
        np.testing.assert_allclose(got, exp_scores, rtol=0, atol=_tol("float32", weights, d0, d1, exp_scores))

    def test_bound_catches_dropped_certainties(self, monkeypatch):
        """Without the certainties the interior is r*c with row mass up to one plus dustbin: > 1."""
        monkeypatch.setattr(match_assignment, "_certainties", lambda z0, z1: 0.0 * (z0 + keras.ops.transpose(z1, (0, 2, 1))))
        weights = _weights("float32", seed=5)
        d0, d1 = _descs(np.random.RandomState(5), "float32", b=3, m=6, n=9)
        got = _np(_layer(weights)(_t(d0, "float32"), _t(d1, "float32"))[0])
        rows, _ = self._totals(got)
        assert rows.max() > 1.0 + 1e-3


# ---------------------------------------------------------------------
# padding invariance
# ---------------------------------------------------------------------


class TestPaddingInvariance:
    def test_appended_padding_leaves_real_entries_unchanged(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng = np.random.RandomState(6)
        weights = _weights(cdt)
        layer = _layer(weights)
        d0, d1 = _descs(rng, cdt, m=5, n=6)
        base = _np(layer(_t(d0, cdt), _t(d1, cdt), np.ones((2, 5), "float32"), np.ones((2, 6), "float32"))[0])
        p0, p1 = 3, 4
        d0p = np.concatenate([d0, _round(rng.normal(size=(2, p0, D)) * 6.0, cdt)], axis=1)
        d1p = np.concatenate([d1, _round(rng.normal(size=(2, p1, D)) * 6.0, cdt)], axis=1)
        padded = _np(layer(_t(d0p, cdt), _t(d1p, cdt), _masks(2, 5 + p0, 5), _masks(2, 6 + p1, 6))[0])
        real_rows = np.r_[0:5, 5 + p0]
        real_cols = np.r_[0:6, 6 + p1]
        tol = _tol(cdt, weights, d0, d1, base)
        got = padded[:, real_rows][:, :, real_cols]
        np.testing.assert_allclose(got, base, rtol=0, atol=tol)

    def test_padded_rows_and_columns_are_exact_zero_and_finite(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng = np.random.RandomState(7)
        layer = _layer(_weights(cdt))
        d0, d1 = _descs(rng, cdt, m=6, n=7)
        mask0 = np.ones((2, 6), "float32"); mask0[:, [1, 4]] = 0.0
        mask1 = np.ones((2, 7), "float32"); mask1[:, [0, 6]] = 0.0
        scores = _np(layer(_t(d0, cdt), _t(d1, cdt), mask0, mask1)[0])
        assert np.all(np.isfinite(scores))
        np.testing.assert_array_equal(scores[:, [1, 4], :], 0.0)
        np.testing.assert_array_equal(scores[:, :, [0, 6]], 0.0)
        assert np.abs(scores[:, 0, 1:6]).max() > 0.0               # real entries are not zeroed

    def test_scattered_padding_equals_oracle_on_the_gathered_real_keypoints(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng = np.random.RandomState(8)
        weights = _weights(cdt)
        layer = _layer(weights)
        d0, d1 = _descs(rng, cdt, b=1, m=7, n=8)
        mask0 = np.array([[1, 0, 1, 1, 0, 1, 1]], "float32")
        mask1 = np.array([[0, 1, 1, 1, 0, 1, 1, 0]], "float32")
        got = _np(layer(_t(d0, cdt), _t(d1, cdt), mask0, mask1)[0])[0]
        r0, r1 = np.flatnonzero(mask0[0]), np.flatnonzero(mask1[0])
        want, _ = ref.match_assignment(d0[0][r0], d1[0][r1], weights, PFX)
        rows, cols = np.r_[r0, 7], np.r_[r1, 8]
        tol = _tol(cdt, weights, d0[:, r0], d1[:, r1], want)
        np.testing.assert_allclose(got[np.ix_(rows, cols)], want, rtol=0, atol=tol)

    def test_padding_matters_without_the_mask(self):
        """Negative control: with no mask the junk keypoints DO move the real entries."""
        rng = np.random.RandomState(6)
        layer = _layer(_weights("float32"))
        d0, d1 = _descs(rng, "float32", m=5, n=6)
        base = _np(layer(_t(d0, "float32"), _t(d1, "float32"))[0])
        d1p = np.concatenate([d1, rng.normal(size=(2, 4, D)) * 6.0], axis=1)
        out = _np(layer(_t(d0, "float32"), _t(d1p, "float32"))[0])
        assert np.abs(out[:, :5, :6] - base[:, :5, :6]).max() > 1e-2

    def test_fully_padded_images_are_finite(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        rng = np.random.RandomState(9)
        layer = _layer(_weights(cdt))
        d0, d1 = _descs(rng, cdt, m=4, n=5)
        mask0 = np.ones((2, 4), "float32"); mask0[0] = 0.0
        mask1 = np.ones((2, 5), "float32"); mask1[1] = 0.0
        scores = _np(layer(_t(d0, cdt), _t(d1, cdt), mask0, mask1)[0])
        assert np.all(np.isfinite(scores))
        # sample 1: image 1 empty, real image-0 rows keep only their dustbin entry
        assert np.all(scores[1, :4, :5] == 0.0)

    def test_one_sided_masks_and_bool_masks(self):
        rng = np.random.RandomState(10)
        layer = _layer(_weights("float32"))
        d0, d1 = _descs(rng, "float32", m=4, n=6)
        m1 = _masks(2, 6, 4)
        a = _np(layer(_t(d0, "float32"), _t(d1, "float32"), None, m1)[0])
        b = _np(layer(_t(d0, "float32"), _t(d1, "float32"), None, m1 > 0)[0])
        np.testing.assert_array_equal(a, b)
        assert np.all(a[:, :, 4:6] == 0.0) and np.all(a[:, :4, :4] != 0.0)

    def test_padded_inputs_receive_exactly_zero_gradient(self):
        import tensorflow as tf
        rng = np.random.RandomState(11)
        layer = _layer(_weights("float32"))
        d0, d1 = _descs(rng, "float32", b=1, m=5, n=6)
        x0, x1 = tf.constant(d0.astype("float32")), tf.constant(d1.astype("float32"))
        m0, m1 = _masks(1, 5, 3), _masks(1, 6, 4)
        with tf.GradientTape() as tape:
            tape.watch([x0, x1])
            scores, _ = layer(x0, x1, m0, m1)
            loss = tf.reduce_sum(tf.where(tf.math.is_finite(scores), scores, 0.0)[:, :3, :4] ** 2) \
                + tf.reduce_sum(scores[:, :3, 6] ** 2)
        g0, g1 = [g.numpy() for g in tape.gradient(loss, [x0, x1])]
        assert np.abs(g0[0, 3:]).max() == 0.0 and np.abs(g1[0, 4:]).max() == 0.0
        assert np.abs(g0[0, :3]).max() > 0.0 and np.abs(g1[0, :4]).max() > 0.0


# ---------------------------------------------------------------------
# mutation guards: each defect must break the parity check the real code passes
# ---------------------------------------------------------------------


class TestMutationGuards:
    def _parity(self, m=6, n=6):
        weights = _weights("float32", seed=12)
        d0, d1 = _descs(np.random.RandomState(12), "float32", m=m, n=n)
        err, _, tol = _err(_layer(weights), weights, d0, d1, "float32")
        return err, tol

    def test_baseline_passes(self):
        err, tol = self._parity()
        assert err <= tol

    def test_dropping_the_certainties_is_caught(self, monkeypatch):
        monkeypatch.setattr(match_assignment, "_certainties",
                            lambda z0, z1: keras.ops.zeros_like(z0 + keras.ops.transpose(z1, (0, 2, 1))))
        err, tol = self._parity()
        assert err > 10 * tol, f"not caught: {err} vs {tol}"

    def test_swapping_dustbin_row_and_column_is_caught(self, monkeypatch):
        def swapped(interior, z0, z1):                  # M == N so the swap is shape-legal
            col = keras.ops.transpose(keras.ops.log_sigmoid(-z1), (0, 2, 1))[:, 0, :][:, :, None]
            row = keras.ops.transpose(keras.ops.log_sigmoid(-z0), (0, 2, 1))
            corner = keras.ops.zeros_like(col[:, :1, :])
            top = keras.ops.concatenate([interior, col], axis=2)
            return keras.ops.concatenate([top, keras.ops.concatenate([row, corner], axis=2)], axis=1)
        monkeypatch.setattr(match_assignment, "_assemble_scores", swapped)
        err, tol = self._parity()
        assert err > 10 * tol, f"not caught: {err} vs {tol}"

    def test_dustbin_sign_flip_is_caught(self, monkeypatch):
        def flipped(interior, z0, z1):
            col = keras.ops.log_sigmoid(z0)
            row = keras.ops.transpose(keras.ops.log_sigmoid(z1), (0, 2, 1))
            corner = keras.ops.zeros_like(col[:, :1, :])
            return keras.ops.concatenate(
                [keras.ops.concatenate([interior, col], axis=2), keras.ops.concatenate([row, corner], axis=2)], axis=1)
        monkeypatch.setattr(match_assignment, "_assemble_scores", flipped)
        err, tol = self._parity(m=5, n=7)
        assert err > 10 * tol

    def test_single_softmax_is_caught(self, monkeypatch):
        monkeypatch.setattr(match_assignment, "_double_log_softmax",
                            lambda logits: 2.0 * keras.ops.log_softmax(logits, axis=2))
        err, tol = self._parity(m=5, n=7)
        assert err > 10 * tol


# ---------------------------------------------------------------------
# filter_matches
# ---------------------------------------------------------------------


def _fm(scores, th, m0=None, m1=None):
    return [keras.ops.convert_to_numpy(o) for o in filter_matches(
        keras.ops.convert_to_tensor(np.asarray(scores, "float32")), th, m0, m1)]


def _fm_oracle_batched(scores, th):
    outs = [ref.filter_matches(s, th) for s in scores]
    return [np.stack([o[k] for o in outs]) for k in range(4)]


def _log_assignment(core, last_col=-5.0, last_row=-5.0):
    """Hand-built (1, M+1, N+1) log assignment from interior probabilities."""
    core = np.log(np.asarray(core, "float64"))
    m, n = core.shape
    out = np.zeros((1, m + 1, n + 1))
    out[0, :m, :n] = core
    out[0, :m, n] = last_col
    out[0, m, :n] = last_row
    return out


class TestFilterMatches:
    def test_matches_oracle_on_random_assignments(self):
        weights = _weights("float64", seed=13)
        for seed, (m, n) in enumerate([(6, 6), (5, 9), (9, 4)]):
            d0, d1 = _descs(np.random.RandomState(20 + seed), "float64", b=3, m=m, n=n)
            scores, _ = _oracle(weights, d0, d1)
            for th in (0.0, 0.01, 0.1, 0.3):
                got = _fm(scores, th)
                want = _fm_oracle_batched(scores, th)
                np.testing.assert_array_equal(got[0], want[0])
                np.testing.assert_array_equal(got[1], want[1])
                np.testing.assert_allclose(got[2], want[2], atol=1e-6)
                np.testing.assert_allclose(got[3], want[3], atol=1e-6)

    def test_mutual_miss_is_rejected(self):
        # row 0 prefers col 0 (0.6), but col 0 prefers row 1 (0.7): row 0 unmatched
        core = [[0.6, 0.1], [0.7, 0.05]]
        m0, m1, s0, s1 = _fm(_log_assignment(core), 0.1)
        # row 1 -> col 0 and col 0 -> row 1 are mutual; row 0 -> col 0 is not
        assert m0.tolist() == [[-1, 0]]
        assert m1.tolist() == [[1, -1]]
        assert s0[0, 0] == 0.0 and np.isclose(s0[0, 1], 0.7, atol=1e-6)

    def test_below_threshold_is_rejected_but_scored(self):
        core = [[0.05, 0.01], [0.02, 0.4]]
        m0, m1, s0, s1 = _fm(_log_assignment(core), 0.1)
        assert m0.tolist() == [[-1, 1]] and m1.tolist() == [[-1, 1]]
        assert np.isclose(s0[0, 0], 0.05, atol=1e-6)          # mutual, so scored, yet < threshold
        assert np.isclose(s1[0, 0], 0.05, atol=1e-6)

    def test_threshold_is_strict(self):
        core = np.array([[0.25]], "float64")
        scores = _log_assignment(core)
        assert _fm(scores, float(np.float32(0.25)))[0].tolist() == [[-1]]
        assert _fm(scores, 0.2499)[0].tolist() == [[0]]

    def test_dustbin_entries_are_ignored(self):
        core = [[0.2, 0.1]]
        assert _fm(_log_assignment(core, last_col=5.0, last_row=5.0), 0.1)[0].tolist() == [[0]]

    def test_masked_keypoints_are_never_matched(self):
        # col 0 is padded and holds the best score of both rows; it must not be chosen
        core = [[0.9, 0.3, 0.1], [0.8, 0.1, 0.4]]
        scores = _log_assignment(core)
        mask0 = np.array([[1, 1]], "float32")
        mask1 = np.array([[0, 1, 1]], "float32")
        m0, m1, s0, s1 = _fm(scores, 0.1, mask0, mask1)
        assert m1[0, 0] == -1 and s1[0, 0] == 0.0
        # real columns 1, 2: row 0 -> col 1 (0.3), row 1 -> col 2 (0.4): mutual
        assert m0.tolist() == [[1, 2]] and m1.tolist() == [[-1, 0, 1]]
        # without the mask the padded column wins row 0 and the result differs
        assert _fm(scores, 0.1)[0].tolist() != m0.tolist()

    def test_padded_rows_unmatched_and_masked_equals_unpadded(self):
        weights = _weights("float64", seed=14)
        rng = np.random.RandomState(14)
        d0, d1 = _descs(rng, "float64", b=1, m=7, n=8)
        mask0 = np.array([[1, 0, 1, 1, 0, 1, 1]], "float32")
        mask1 = np.array([[0, 1, 1, 1, 0, 1, 1, 0]], "float32")
        layer = _layer(weights)
        scores = _np(layer(_t(d0, "float64"), _t(d1, "float64"), mask0, mask1)[0])
        m0, m1, s0, s1 = _fm(scores, 0.0, mask0, mask1)
        assert np.all(m0[0][mask0[0] == 0] == -1) and np.all(m1[0][mask1[0] == 0] == -1)
        r0, r1 = np.flatnonzero(mask0[0]), np.flatnonzero(mask1[0])
        want, _ = ref.match_assignment(d0[0][r0], d1[0][r1], weights, PFX)
        w0, w1, _, _ = ref.filter_matches(want, 0.0)
        np.testing.assert_array_equal(m0[0][r0], np.where(w0 >= 0, r1[np.maximum(w0, 0)], -1))
        np.testing.assert_array_equal(m1[0][r1], np.where(w1 >= 0, r0[np.maximum(w1, 0)], -1))

    def test_empty_image_has_no_matches(self):
        scores = _log_assignment([[0.9, 0.8], [0.7, 0.6]])
        m0, m1, s0, s1 = _fm(scores, 0.0, np.array([[1, 1]], "float32"), np.zeros((1, 2), "float32"))
        assert np.all(m0 == -1) and np.all(m1 == -1) and np.all(s0 == 0.0) and np.all(s1 == 0.0)

    def test_output_dtypes_and_shapes(self):
        m0, m1, s0, s1 = _fm(_log_assignment(np.full((3, 4), 0.2)), 0.1)
        assert m0.shape == (1, 3) and m1.shape == (1, 4) and m0.dtype == np.int32


# ---------------------------------------------------------------------
# construction, config, serialization, gradient flow
# ---------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("dim", [0, -4])
    def test_bad_dim_raises(self, dim):
        with pytest.raises(ValueError):
            MatchAssignment(dim=dim)

    def test_bad_build_shape_raises(self):
        with pytest.raises(ValueError):
            MatchAssignment(dim=16).build((None, 5, 17), (None, 5, 16))
        with pytest.raises(ValueError):
            MatchAssignment(dim=16).build((5, 16), (None, 5, 16))

    def test_sublayer_names_follow_the_torch_state_dict(self):
        layer = MatchAssignment(dim=16)
        layer.build((None, None, 16), (None, None, 16))
        assert tuple(layer.final_proj.kernel.shape) == (16, 16)
        assert tuple(layer.matchability.kernel.shape) == (16, 1)
        assert {w.path.split("/")[-2] for w in layer.weights} == {"final_proj", "matchability"}

    def test_get_config_round_trip(self):
        layer = MatchAssignment(dim=24, name="ma")
        config = layer.get_config()
        assert config["dim"] == 24
        assert MatchAssignment.from_config(config).get_config() == config

    def test_compute_output_shape(self):
        assert MatchAssignment(dim=16).compute_output_shape((None, 9, 16), (None, 7, 16)) == (
            (None, 10, 8), (None, 9, 7))

    def test_symbolic_call_shapes(self):
        scores, sim = MatchAssignment(dim=D)(keras.Input((9, D)), keras.Input((7, D)),
                                             keras.Input((9,)), keras.Input((7,)))
        assert tuple(scores.shape) == (None, 10, 8) and tuple(sim.shape) == (None, 9, 7)


def _wrapper(m=6, n=5):
    d0, d1 = keras.Input((m, D)), keras.Input((n, D))
    m0, m1 = keras.Input((m,)), keras.Input((n,))
    scores, sim = MatchAssignment(dim=D)(d0, d1, m0, m1)
    return keras.Model([d0, d1, m0, m1], scores)


def _wrapper_data(rng, b=4, m=6, n=5):
    m0 = np.ones((b, m), "float32"); m0[:, -1] = 0.0
    m1 = np.ones((b, n), "float32"); m1[:, -2:] = 0.0
    return [rng.normal(size=(b, m, D)).astype("float32"), rng.normal(size=(b, n, D)).astype("float32"), m0, m1]


class TestSerialization:
    def test_save_load_round_trip_through_a_model(self, tmp_path):
        data = _wrapper_data(np.random.RandomState(15))
        model = _wrapper()
        expected = model.predict(data, verbose=0)
        path = str(tmp_path / "assign.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        np.testing.assert_array_equal(reloaded.predict(data, verbose=0), expected)


class TestGradientFlow:
    def _compiled(self):
        model = _wrapper()
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
        return model

    def _targets(self, rng):
        return rng.normal(size=(4, 7, 6)).astype("float32")

    def test_every_trainable_variable_moves(self):
        rng = np.random.RandomState(16)
        report = fit_one_step_moved_variables(self._compiled(), _wrapper_data(rng), self._targets(rng))
        assert report.total == 4
        assert report.unmoved == (), f"non-movers: {report.unmoved}"
        assert any("final_proj" in l for l in report.moved) and any("matchability" in l for l in report.moved)

    def test_stop_gradient_injection_makes_the_step_raise(self):
        rng = np.random.RandomState(17)
        model = self._compiled()
        data = _wrapper_data(rng)
        model.predict(data, verbose=0)
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                model.fit(data, self._targets(rng), epochs=1, verbose=0)

    def test_each_projection_responds_when_zeroed(self):
        model = _wrapper()
        data = _wrapper_data(np.random.RandomState(18))
        layer = next(l for l in model.layers if isinstance(l, MatchAssignment))

        def metric():
            return float(np.abs(model.predict(data, verbose=0)).sum())

        control = component_response(metric, no_op_kill, name="no-op control")
        assert not control.moved and control.delta == 0.0, control.summary()
        for name in ("final_proj", "matchability"):
            sub = getattr(layer, name)
            result = component_response(metric, lambda s=sub: zeroed_variables(s.weights), name=name, atol=1e-4)
            assert result.moved, result.summary()
