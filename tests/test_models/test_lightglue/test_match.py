"""``LightGlue.match``: the eager adaptive path (early stop, point pruning, scatter-back).

Ground truth is ``tests/lightglue_reference_numpy.forward_adaptive``, an explicit-loop
transcription of the reference ``_forward``. Random weights are tuned per fixture so that the
adaptive machinery actually fires; every fixture asserts that it did (a stop before the last
layer, or at least one pruned point) because an adaptive path that never triggers would pass
any parity test vacuously. The mutation tests inject a defect into one decision and show the
shared assertion going RED.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.keypoints.lightglue import model as lightglue_model
from dl_techniques.models.vision.keypoints.lightglue.model import (
    LightGlue, check_if_stop, confidence_threshold, get_pruning_mask)
from tests import lightglue_reference_numpy as ref

from .weight_loading import load_torch_weights, random_inputs

# TF32 matmul would turn the float32 comparison into a ~1e-3 one; run in the TF32-off regime.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

_EPS32 = float(np.finfo(np.float32).eps)
D, H, L = 32, 2, 4
M = N = 12


def _tol(expected):
    """float32 bound for ``L`` chained layers (same per-stage rate as ``test_parity._tol``)."""
    return 4 * 20 * L * _EPS32 * (1.0 + float(np.abs(expected).max()))


def _weights(seed, token_scale=1.0, token_bias=(0.0,) * (L - 1), match_scale=1.0, match_bias=0.0):
    """Torch-layout random weights with scaled / biased confidence and matchability heads."""
    w = ref.random_weights(np.random.RandomState(seed), D, H, L)
    for i in range(L - 1):
        w[f"token_confidence.{i}.token.0.weight"] = w[f"token_confidence.{i}.token.0.weight"] * token_scale
        w[f"token_confidence.{i}.token.0.bias"][:] = token_bias[i]
    for i in range(L):
        w[f"log_assignment.{i}.matchability.weight"] = w[f"log_assignment.{i}.matchability.weight"] * match_scale
        w[f"log_assignment.{i}.matchability.bias"][:] = match_bias
    return w


# name -> (weights, depth_confidence, width_confidence)
FIXTURES = {
    # confidence heads of layers 0 and 1 say "unsure" (0), layer 2 says "sure" (1): stop at i=2
    "stop": (lambda: _weights(5, token_bias=(-50.0, -50.0, 50.0)), 0.95, -1),
    # matchability only (depth disabled, no confidence head evaluated): points are pruned
    "prune": (lambda: _weights(1, 1.0, match_scale=5.0), -1, 0.5),
    # both: pruning at layer 0, then the early exit fires at layer 2
    "both": (lambda: _weights(0, 5.0, match_scale=5.0), 0.8, 0.5),
    # every point is confident and unmatchable (depth 1.0 never stops): both sides are pruned to nothing at layer 0
    "empty": (lambda: _weights(3, token_bias=(50.0,) * (L - 1), match_bias=-50.0), 1.0, 1e-6),
}


def _model(weights, depth, width, pruning_min_kpts=-1, **kw):
    model = LightGlue(input_dim=D, descriptor_dim=D, num_layers=L, num_heads=H,
                      filter_threshold=0.0, depth_confidence=depth, width_confidence=width,
                      pruning_min_kpts=pruning_min_kpts, **kw)
    model.build(None)
    load_torch_weights(model, weights)
    return model


def _data(m_pts=M, n_pts=N, seed=1):
    return random_inputs(np.random.RandomState(seed), 1, m_pts, n_pts, D)


_ORACLE = {}


def _oracle(name, pruning_min_kpts=-1):
    """Oracle result of a fixture, computed once per (fixture, threshold)."""
    key = (name, pruning_min_kpts)
    if key not in _ORACLE:
        make, depth, width = FIXTURES[name]
        data = _data()
        _ORACLE[key] = ref.forward_adaptive(
            make(), data["keypoints0"][0].astype("float64"), data["descriptors0"][0],
            data["image_size0"][0].astype("float64"),
            data["keypoints1"][0].astype("float64"), data["descriptors1"][0],
            data["image_size1"][0].astype("float64"),
            L, H, depth, width, 0.0, pruning_min_kpts)
    return _ORACLE[key]


def _match(name, pruning_min_kpts=-1):
    make, depth, width = FIXTURES[name]
    return _model(make(), depth, width, pruning_min_kpts).match(_data())


def _assert_matches_oracle(got, exp):
    """Everything the reference returns, equal to the oracle (indices exactly, scores to eps32)."""
    assert got["stop"] == exp["stop"], f"stop {got['stop']} != oracle {exp['stop']}"
    np.testing.assert_array_equal(got["prune0"][0], exp["prune0"])
    np.testing.assert_array_equal(got["prune1"][0], exp["prune1"])
    np.testing.assert_array_equal(got["matches0"][0], exp["matches0"])
    np.testing.assert_array_equal(got["matches1"][0], exp["matches1"])
    np.testing.assert_array_equal(got["matches"], exp["matches"])
    tol = _tol(np.ones(1))
    for key in ("matching_scores0", "matching_scores1"):
        np.testing.assert_allclose(got[key][0], exp[key], rtol=0, atol=tol)
    np.testing.assert_allclose(got["scores"], exp["scores"], rtol=0, atol=tol)


# ---------------------------------------------------------------------
# equals the static path when both knobs are off
# ---------------------------------------------------------------------

class TestEqualsStaticPath:

    def test_disabled_knobs_equal_the_last_static_layer(self):
        model = _model(_weights(7), -1, -1)
        data = _data()
        out = model(data)
        got = model.match(data)
        np.testing.assert_array_equal(got["matches0"], keras.ops.convert_to_numpy(out["matches0"]))
        np.testing.assert_array_equal(got["matches1"], keras.ops.convert_to_numpy(out["matches1"]))
        for key in ("matching_scores0", "matching_scores1"):
            exp = keras.ops.convert_to_numpy(out[key])
            np.testing.assert_allclose(got[key], exp, rtol=0, atol=_tol(exp))
        assert got["stop"] == L
        assert np.all(got["prune0"] == L) and np.all(got["prune1"] == L)
        assert (got["matches0"] >= 0).sum() > 0, "vacuous: no match to compare"
        assert len(got["matches"]) == (got["matches0"] >= 0).sum()

    def test_padded_variant_equals_static_call_with_masks(self):
        model = _model(_weights(7), -1, -1)
        data = _data()
        mask0 = np.zeros((1, M), "float32")
        mask1 = np.zeros((1, N), "float32")
        mask0[0, [0, 1, 2, 4, 5, 7, 8, 9, 11]] = 1      # a non-prefix real set
        mask1[0, :10] = 1
        padded = {**data, "mask0": mask0, "mask1": mask1}
        out = model(padded)
        got = model.match(padded)
        np.testing.assert_array_equal(got["matches0"], keras.ops.convert_to_numpy(out["matches0"]))
        np.testing.assert_array_equal(got["matches1"], keras.ops.convert_to_numpy(out["matches1"]))
        assert np.all(got["matches0"][0][mask0[0] == 0] == -1)
        assert np.all(got["matching_scores0"][0][mask0[0] == 0] == 0)
        assert (got["matches0"] >= 0).sum() > 0, "vacuous: no match to compare"

    def test_batch_other_than_one_is_rejected(self):
        model = _model(_weights(7), -1, -1)
        data = random_inputs(np.random.RandomState(0), 2, M, N, D)
        with pytest.raises(ValueError, match="batch size 1"):
            model.match(data)


# ---------------------------------------------------------------------
# oracle parity of the adaptive path
# ---------------------------------------------------------------------

class TestAdaptiveParity:

    def test_stop_fixture_stops_at_the_forced_layer(self):
        got = _match("stop")
        assert got["stop"] == 3 and got["stop"] < L        # i = 2: layers 0 and 1 are "unsure"
        assert np.all(got["prune0"] == L), "width pruning is off in this fixture"
        _assert_matches_oracle(got, _oracle("stop"))

    def test_prune_fixture_prunes_with_matchability_only(self):
        got = _match("prune")
        pruned0 = got["prune0"][0] < got["prune0"][0].max()
        assert pruned0.any(), "vacuous: nothing was pruned"
        _assert_matches_oracle(got, _oracle("prune"))

    def test_pruned_points_are_unmatched(self):
        got = _match("prune")
        survivors0 = got["prune0"][0] == got["prune0"][0].max()
        survivors1 = got["prune1"][0] == got["prune1"][0].max()
        assert (~survivors0).any() and (~survivors1).any(), "vacuous: nothing was pruned"
        assert np.all(got["matches0"][0][~survivors0] == -1)
        assert np.all(got["matches1"][0][~survivors1] == -1)
        assert np.all(got["matching_scores0"][0][~survivors0] == 0)
        # a reported partner is always a survivor of the other image
        partners = got["matches0"][0][got["matches0"][0] >= 0]
        assert survivors1[partners].all()

    def test_both_fixture_prunes_then_stops_early(self):
        got = _match("both")
        assert got["stop"] < L, "vacuous: the early exit never fired"
        assert (got["prune0"][0] == 1).any(), "vacuous: nothing was pruned"
        _assert_matches_oracle(got, _oracle("both"))

    def test_padded_adaptive_run_equals_the_compact_run(self):
        make, depth, width = FIXTURES["both"]
        model = _model(make(), depth, width)
        data = _data()
        k0, k1 = 10, 11
        mask0 = np.zeros((1, M), "float32")
        mask1 = np.zeros((1, N), "float32")
        mask0[0, :k0] = 1
        mask1[0, :k1] = 1
        compact = model.match({
            **data, "keypoints0": data["keypoints0"][:, :k0], "keypoints1": data["keypoints1"][:, :k1],
            "descriptors0": data["descriptors0"][:, :k0], "descriptors1": data["descriptors1"][:, :k1]})
        padded = model.match({**data, "mask0": mask0, "mask1": mask1})
        assert compact["stop"] == padded["stop"]
        np.testing.assert_array_equal(padded["matches0"][0, :k0], compact["matches0"][0])
        np.testing.assert_array_equal(padded["matches1"][0, :k1], compact["matches1"][0])
        np.testing.assert_array_equal(padded["prune0"][0, :k0], compact["prune0"][0])
        assert np.all(padded["matches0"][0, k0:] == -1)


class TestEmptyAfterPruning:

    def test_both_images_pruned_to_zero_points(self):
        got = _match("empty")
        exp = _oracle("empty")
        assert got["stop"] == exp["stop"] == 2             # reference: i + 1 of the breaking iteration
        for key in ("matches0", "matches1", "matching_scores0", "matching_scores1", "scores"):
            assert np.all(np.isfinite(got[key])), key
        assert np.all(got["matches0"] == -1) and np.all(got["matches1"] == -1)
        assert np.all(got["matching_scores0"] == 0) and np.all(got["matching_scores1"] == 0)
        assert got["matches"].shape == (0, 2)
        _assert_matches_oracle(got, exp)

    def test_an_image_without_real_keypoints(self):
        make, depth, width = FIXTURES["both"]
        model = _model(make(), depth, width)
        data = _data()
        got = model.match({**data, "mask0": np.zeros((1, M), "float32")})
        assert got["stop"] == 1
        assert np.all(got["matches0"] == -1) and np.all(got["matches1"] == -1)
        assert got["matches0"].shape == (1, M) and got["matches1"].shape == (1, N)
        assert np.all(np.isfinite(got["matching_scores1"]))


# ---------------------------------------------------------------------
# the decision rules, on hand-built vectors
# ---------------------------------------------------------------------

class TestDecisionRules:

    def test_confidence_threshold_values(self):
        assert confidence_threshold(0, 9) == pytest.approx(0.9)
        for i in range(9):
            assert confidence_threshold(i, 9) == pytest.approx(ref.confidence_threshold(i, 9))
        assert confidence_threshold(8, 9) < confidence_threshold(0, 9)

    def test_pruning_rule_on_a_hand_built_vector(self):
        # layer 1 of 4: threshold = 0.8 + 0.1 exp(-1) = 0.83679
        th = ref.confidence_threshold(1, 4)
        conf = np.array([0.99, th - 1e-3, th, th + 1e-3, 0.1, 0.99, 0.5])
        match = np.array([0.90, 0.001, 0.001, 0.001, 0.001, 0.51, 0.49])
        # width 0.5 keeps matchability > 0.5; a low confidence (<= th) is always kept
        expected = np.array([True, True, True, False, True, True, True])
        np.testing.assert_array_equal(get_pruning_mask(conf, match, 1, 4, 0.5), expected)
        np.testing.assert_array_equal(ref.get_pruning_mask_ref(conf, match, 1, 4, 0.5), expected)
        # without confidences only matchability keeps: the rule is NOT "conf <= th" by default
        np.testing.assert_array_equal(
            get_pruning_mask(None, match, 1, 4, 0.5), np.array([True, False, False, False, False, True, False]))

    def test_stop_rule_on_a_hand_built_vector(self):
        th = ref.confidence_threshold(0, 4)
        c0 = np.array([0.99, 0.99, th - 0.01, 0.99])
        c1 = np.array([0.99, 0.99, 0.99, th])               # == th is NOT below
        # 1 of 8 below -> ratio 0.875
        assert check_if_stop(c0, c1, 0, 4, 8, 0.8) is True
        assert check_if_stop(c0, c1, 0, 4, 8, 0.875) is False   # strictly greater
        assert ref.check_if_stop_ref(c0, c1, 0, 4, 8, 0.8) is True
        assert ref.check_if_stop_ref(c0, c1, 0, 4, 8, 0.875) is False
        # pruned points count as confident: the denominator is the ORIGINAL point count
        assert check_if_stop(c0, c1, 0, 4, 16, 0.9) == ref.check_if_stop_ref(c0, c1, 0, 4, 16, 0.9)
        assert check_if_stop(c0, c1, 0, 4, 16, 0.9) is True    # 1 - 1/16 = 0.9375


class TestPruningMinKpts:

    def test_no_pruning_at_or_below_the_threshold(self):
        got = _match("prune", pruning_min_kpts=M)          # 12 points are not MORE than 12
        assert np.all(got["prune0"] == 1) and np.all(got["prune1"] == 1)   # width on, nobody pruned
        no_width = _model(FIXTURES["prune"][0](), -1, -1).match(_data())
        np.testing.assert_array_equal(got["matches0"], no_width["matches0"])
        _assert_matches_oracle(got, _oracle("prune", M))

    def test_pruning_above_the_threshold(self):
        got = _match("prune", pruning_min_kpts=5)
        assert got["prune0"].max() > got["prune0"].min(), "vacuous: nothing was pruned"
        _assert_matches_oracle(got, _oracle("prune", 5))

    def test_the_gate_changes_the_outcome(self):
        a, b = _match("prune", pruning_min_kpts=-1), _match("prune", pruning_min_kpts=M)
        assert not np.array_equal(a["prune0"], b["prune0"])


# ---------------------------------------------------------------------
# model hygiene
# ---------------------------------------------------------------------

class TestHygiene:

    def test_match_leaves_every_variable_bitwise_unchanged(self):
        make, depth, width = FIXTURES["both"]
        model = _model(make(), depth, width)
        before = [np.array(keras.ops.convert_to_numpy(v)) for v in model.weights]
        model.match(_data())
        after = [np.array(keras.ops.convert_to_numpy(v)) for v in model.weights]
        assert len(before) == len(after) > 0
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_config_round_trip_includes_pruning_min_kpts(self):
        model = LightGlue(input_dim=D, descriptor_dim=D, num_layers=L, num_heads=H,
                          pruning_min_kpts=1024, depth_confidence=0.9, width_confidence=0.5)
        config = model.get_config()
        assert config["pruning_min_kpts"] == 1024
        again = LightGlue.from_config(config)
        assert again.pruning_min_kpts == 1024
        assert again.get_config() == config

    def test_default_pruning_min_kpts_is_the_cpu_setting(self):
        assert LightGlue(input_dim=D, descriptor_dim=D, num_layers=L, num_heads=H).pruning_min_kpts == -1

    @pytest.mark.parametrize("bad", [1.5, "8", None, True])
    def test_non_integer_pruning_min_kpts_is_rejected(self, bad):
        with pytest.raises(ValueError, match="pruning_min_kpts"):
            LightGlue(input_dim=D, descriptor_dim=D, num_layers=L, num_heads=H, pruning_min_kpts=bad)


# ---------------------------------------------------------------------
# mutation RED proofs: each defect must break the shared assertions above
# ---------------------------------------------------------------------

class TestMutationsAreCaught:

    def test_inverted_keep_rule_is_caught(self, monkeypatch):
        original = lightglue_model.get_pruning_mask
        monkeypatch.setattr(
            lightglue_model, "get_pruning_mask",
            lambda *a, **k: np.logical_not(original(*a, **k)))
        with pytest.raises(AssertionError):
            _assert_matches_oracle(_match("prune"), _oracle("prune"))

    def test_stop_one_layer_late_is_caught(self, monkeypatch):
        original = lightglue_model.check_if_stop
        state = {"previous": False}

        def late(*args, **kwargs):
            fired, state["previous"] = state["previous"], original(*args, **kwargs)
            return fired

        monkeypatch.setattr(lightglue_model, "check_if_stop", late)
        got = _match("stop")
        assert got["stop"] != 3
        with pytest.raises(AssertionError):
            _assert_matches_oracle(got, _oracle("stop"))

    def test_skipped_scatter_back_is_caught(self, monkeypatch):
        def no_scatter(m0, m1, s0, s1, ind0, ind1, size0, size1):
            pad = lambda a, n, fill: np.concatenate([a, np.full(n - len(a), fill, a.dtype)])
            return pad(m0, size0, -1), pad(m1, size1, -1), pad(s0, size0, 0), pad(s1, size1, 0)

        monkeypatch.setattr(lightglue_model, "_scatter_back", no_scatter)
        with pytest.raises(AssertionError):
            _assert_matches_oracle(_match("prune"), _oracle("prune"))

    def test_the_unmutated_fixtures_pass(self):
        for name in ("stop", "prune", "both", "empty"):
            _assert_matches_oracle(_match(name), _oracle(name))
