"""Value-level guards for :mod:`dl_techniques.analyzer.analyzers.weight_analyzer`.

Before this module existed, `weight_analyzer.py` was executed by exactly 3 of the
378 analyzer tests (all in `test_analyzer_docs.py`) and **none of them asserted
the value of any statistic**. That blindness is what let a NaN-producing
`_compute_weight_statistics` abort the whole of `ModelAnalyzer.analyze()` on a
freshly-initialised model.

The centrepiece is an executable CENSUS over 20 degenerate-but-legal weight
tensors covering the five measured mechanisms:

M1  zero-variance moments  -- `scipy.stats.skew`/`kurtosis` return NaN.
M2  float32 reduction OVERFLOW -- every element representable, the sum is not.
M3  float32 2nd-moment UNDERFLOW on a NON-constant tensor. This is why the guard
    may not key on "is the tensor constant".
M4  already-corrupt incoming weights (a real NaN/inf in the tensor).
M5  raises from `_compute_weight_statistics` itself -- zero-size tensors and
    float16, neither of which `except np.linalg.LinAlgError` can catch.

Three clean tensors ride along as the ANTI-VACUITY arm: if the classifier flags
those too, the predicate does not discriminate and the census proves nothing.
"""

import keras
import numpy as np
import pytest

from dl_techniques.analyzer.analyzers.weight_analyzer import WeightAnalyzer
from dl_techniques.analyzer.constants import StatusCode
from dl_techniques.analyzer import AnalysisConfig


def _analyzer() -> WeightAnalyzer:
    """Build a `WeightAnalyzer` with no models; only the pure statistics are used."""
    return WeightAnalyzer(models={}, config=AnalysisConfig(verbose=False))


_RNG = np.random.default_rng(0)

#: ``(id, tensor, mechanism)`` for every degenerate-but-legal tensor in the census.
DEGENERATE_CENSUS = [
    # M1 -- zero-variance moments
    ("m1_zeros_2d", np.zeros((6, 8), dtype=np.float32), "M1"),
    ("m1_constant_2d", np.full((6, 8), 0.3, dtype=np.float32), "M1"),
    ("m1_single_element_2d", np.full((1, 1), 1.5, dtype=np.float32), "M1"),
    ("m1_length_one_1d", np.full((1,), 1.5, dtype=np.float32), "M1"),
    ("m1_zero_bias_1d", np.zeros((12,), dtype=np.float32), "M1"),
    ("m1_ones_bias_1d", np.ones((12,), dtype=np.float32), "M1"),
    ("m1_zeros_conv4d", np.zeros((3, 3, 3, 16), dtype=np.float32), "M1"),
    ("m1_constant_conv4d", np.full((3, 3, 3, 16), 0.02, dtype=np.float32), "M1"),
    ("m1_zeros_3d", np.zeros((2, 3, 4), dtype=np.float32), "M1"),
    ("m1_zeros_float64", np.zeros((6, 8), dtype=np.float64), "M1"),
    ("m1_zeros_int32", np.zeros((6, 8), dtype=np.int32), "M1"),
    # M2 -- float32 reduction overflow on individually-representable values
    ("m2_l2_overflow", np.full((1000, 1000), 1e20, dtype=np.float32), "M2"),
    ("m2_mean_overflow", np.full((100000,), 1e35, dtype=np.float32), "M2"),
    # M3 -- float32 underflow of the 2nd moment on a NON-constant tensor
    ("m3_underflow_1e30", (_RNG.standard_normal((6, 8)) * 1e-30).astype(np.float32), "M3"),
    ("m3_underflow_subnormal", (_RNG.standard_normal((6, 8)) * 1e-45).astype(np.float32), "M3"),
    # M4 -- already-corrupt incoming weights
    ("m4_one_nan", np.concatenate(
        [np.array([np.nan], dtype=np.float32),
         _RNG.standard_normal(47).astype(np.float32)]).reshape(6, 8), "M4"),
    ("m4_all_inf", np.full((6, 8), np.inf, dtype=np.float32), "M4"),
    # M5 -- raises from `_compute_weight_statistics` itself
    ("m5_zero_size_2d", np.zeros((0, 8), dtype=np.float32), "M5"),
    ("m5_zero_size_1d", np.zeros((0,), dtype=np.float32), "M5"),
    ("m5_float16", _RNG.standard_normal((6, 8)).astype(np.float16), "M5"),
]

#: Clean tensors. Anti-vacuity: these must NOT be flagged.
CLEAN_CENSUS = [
    ("clean_healthy_2d", _RNG.standard_normal((6, 8)).astype(np.float32)),
    ("clean_all_negative_2d", -np.abs(_RNG.standard_normal((6, 8))).astype(np.float32)),
    ("clean_nearly_const_one_hot",
     np.eye(6, 8, dtype=np.float32)[:1].repeat(6, axis=0) * 0 + np.pad(
         np.array([[1.0]], dtype=np.float32), ((0, 5), (0, 7)))),
]

#: Every leaf the PCA feature vector reads (`_compute_weight_pca`).
PCA_LEAF_PATHS = [
    ("basic", "mean"), ("basic", "std"), ("basic", "median"),
    ("basic", "skewness"), ("basic", "kurtosis"),
    ("norms", "l1"), ("norms", "l2"), ("norms", "rms"),
    ("distribution", "zero_fraction"), ("distribution", "positive_fraction"),
    ("distribution", "negative_fraction"),
]


def _pca_leaves(stats):
    """Return ``{name: value}`` for exactly the leaves the PCA vector consumes."""
    out = {}
    for group, key in PCA_LEAF_PATHS:
        out[f"{group}.{key}"] = stats[group][key]
    if "spectral" in stats["norms"]:
        out["norms.spectral"] = stats["norms"]["spectral"]
    return out


class TestTheDegenerateWeightCensus:
    """All 20 degenerate tensors must be CLASSIFIED, never NaN-propagating."""

    @pytest.mark.parametrize(
        "tensor_id,tensor,mechanism",
        DEGENERATE_CENSUS,
        ids=[c[0] for c in DEGENERATE_CENSUS],
    )
    def test_a_degenerate_tensor_is_flagged_and_never_raises(
            self, tensor_id, tensor, mechanism):
        stats = _analyzer()._compute_weight_statistics(tensor)

        if tensor.size == 0:
            assert stats is None, (
                f"{tensor_id}: a zero-size tensor must be skipped (None), got {stats!r}"
            )
            return

        assert stats is not None, f"{tensor_id}: a non-empty tensor must produce statistics"
        assert "status" in stats, f"{tensor_id}: the statistics carry no `status` field"
        assert stats["status"] != StatusCode.SUCCESS.value, (
            f"{tensor_id} ({mechanism}): reported status "
            f"{stats['status']!r} -- a degenerate tensor must be FLAGGED"
        )

        leaves = _pca_leaves(stats)
        if mechanism == "M4":
            # A genuinely corrupt weight stays corrupt in the output: it must be
            # distinguishable from a merely degenerate one, so it is NOT repaired.
            assert stats["status"] == StatusCode.WEIGHT_NON_FINITE.value, (
                f"{tensor_id}: a corrupt weight must be reported as "
                f"{StatusCode.WEIGHT_NON_FINITE.value!r}, got {stats['status']!r}"
            )
        else:
            assert stats["status"] == StatusCode.WEIGHT_DEGENERATE.value, (
                f"{tensor_id} ({mechanism}): expected "
                f"{StatusCode.WEIGHT_DEGENERATE.value!r}, got {stats['status']!r}"
            )
            bad = {k: v for k, v in leaves.items() if not np.isfinite(v)}
            assert not bad, (
                f"{tensor_id} ({mechanism}): non-finite values reach the PCA "
                f"feature vector: {bad}"
            )

    @pytest.mark.parametrize(
        "tensor_id,tensor", CLEAN_CENSUS, ids=[c[0] for c in CLEAN_CENSUS])
    def test_anti_vacuity_a_clean_tensor_is_not_flagged(self, tensor_id, tensor):
        stats = _analyzer()._compute_weight_statistics(tensor)
        assert stats is not None
        assert stats["status"] == StatusCode.SUCCESS.value, (
            f"{tensor_id}: a healthy tensor was flagged {stats['status']!r}; the "
            "predicate does not discriminate, so the census proves nothing"
        )
        bad = {k: v for k, v in _pca_leaves(stats).items() if not np.isfinite(v)}
        assert not bad, f"{tensor_id}: clean tensor produced non-finite leaves {bad}"

    def test_a_corrupt_weight_stays_distinguishable_from_a_degenerate_one(self):
        """The sanitizing fix must not launder a real NaN into a plausible 0.0."""
        analyzer = _analyzer()
        degenerate = analyzer._compute_weight_statistics(
            np.zeros((6, 8), dtype=np.float32))
        corrupt = analyzer._compute_weight_statistics(
            np.full((6, 8), np.nan, dtype=np.float32))

        assert degenerate["status"] != corrupt["status"], (
            "a zeros tensor and a NaN tensor report the SAME status, so the "
            "output cannot tell a legal degenerate weight from a corrupt one"
        )
        assert degenerate["status"] == StatusCode.WEIGHT_DEGENERATE.value
        assert corrupt["status"] == StatusCode.WEIGHT_NON_FINITE.value
        assert not np.isfinite(corrupt["basic"]["mean"]), (
            "a corrupt weight's mean was repaired into a plausible number"
        )

    def test_the_guard_does_not_key_on_whether_the_tensor_is_constant(self):
        """MEASURED: `randn * 1e-30` is NOT constant yet yields `skew = nan`."""
        tensor = (_RNG.standard_normal((6, 8)) * 1e-30).astype(np.float32)
        assert tensor.size > 1 and float(np.ptp(tensor)) > 0.0, (
            "anti-vacuity: the underflow probe tensor is constant, so it would "
            "be caught by an `is_constant` predicate and prove nothing"
        )
        stats = _analyzer()._compute_weight_statistics(tensor)
        assert stats["status"] == StatusCode.WEIGHT_DEGENERATE.value
        assert np.isfinite(stats["basic"]["skewness"])

    def test_the_substituted_moments_are_zero_not_invented(self):
        """A zero-variance distribution's standardized moments are substituted 0.0."""
        stats = _analyzer()._compute_weight_statistics(
            np.full((6, 8), 0.3, dtype=np.float32))
        assert stats["basic"]["skewness"] == 0.0
        assert stats["basic"]["kurtosis"] == 0.0
        assert "basic.skewness" in stats["degenerate_fields"], (
            "the substitution is not recorded, so a reader cannot tell the 0.0 "
            f"apart from a measured 0.0 (fields: {stats.get('degenerate_fields')})"
        )

    def test_float32_overflow_is_recovered_rather_than_substituted(self):
        """M2: every element is representable, so the true l2 norm IS computable."""
        stats = _analyzer()._compute_weight_statistics(
            np.full((1000, 1000), 1e20, dtype=np.float32))
        assert np.isfinite(stats["norms"]["l2"])
        assert stats["norms"]["l2"] == pytest.approx(1e23, rel=1e-6), (
            "the float32 overflow was substituted with a placeholder instead of "
            f"recomputed at higher precision (got {stats['norms']['l2']})"
        )

    def test_a_healthy_tensors_statistics_are_bit_identical_to_the_old_path(self):
        """The repair must be value-PRESERVING for tensors that never overflowed."""
        weights = _RNG.standard_normal((6, 8)).astype(np.float32)
        stats = _analyzer()._compute_weight_statistics(weights)
        flat = weights.flatten()
        assert stats["basic"]["mean"] == float(np.mean(flat))
        assert stats["basic"]["std"] == float(np.std(flat))
        assert stats["norms"]["l1"] == float(np.sum(np.abs(weights)))
        assert stats["norms"]["l2"] == float(np.sqrt(np.sum(weights ** 2)))
        assert stats["norms"]["rms"] == float(np.sqrt(np.mean(weights ** 2)))


def _tiny_model(name: str, corrupt: bool = False, constant: bool = False):
    """Build a two-`Dense` classifier, optionally with a corrupt or constant kernel."""
    keras.utils.set_random_seed(3)
    inputs = keras.Input(shape=(6,), name=f"{name}_in")
    x = keras.layers.Dense(8, activation="relu", name=f"{name}_d1")(inputs)
    outputs = keras.layers.Dense(3, activation="softmax", name=f"{name}_out")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    if corrupt or constant:
        layer = model.get_layer(f"{name}_d1")
        weights = layer.get_weights()
        weights[0] = (np.full_like(weights[0], np.nan) if corrupt
                      else np.zeros_like(weights[0]))
        layer.set_weights(weights)
    return model


def _weights_only_analyzer(models, output_dir):
    """A `ModelAnalyzer` running the weight analysis and nothing else."""
    from dl_techniques.analyzer import ModelAnalyzer

    return ModelAnalyzer(
        models=models,
        config=AnalysisConfig(
            analyze_weights=True, analyze_calibration=False,
            analyze_information_flow=False, analyze_training_dynamics=False,
            analyze_spectral=False, save_plots=False, verbose=False,
        ),
        output_dir=str(output_dir),
    )


class TestThePcaIsNeverFatal:
    """`analyze()` must never abort because one model's weights are degenerate."""

    def test_a_corrupt_weight_does_not_abort_the_analysis(self, tmp_path, caplog):
        analyzer = _weights_only_analyzer(
            {"corrupt": _tiny_model("corrupt", corrupt=True),
             "clean_a": _tiny_model("clean_a"),
             "clean_b": _tiny_model("clean_b")},
            tmp_path / "corrupt",
        )
        with caplog.at_level("WARNING"):
            results = analyzer.analyze(analysis_types={"weights"})

        assert results.weight_pca is not None, (
            "the PCA was skipped entirely; a corrupt model must be DROPPED, not "
            "allowed to take the whole panel down with it"
        )
        assert results.weight_pca["labels"] == ["clean_a", "clean_b"], (
            "the corrupt model's row was not dropped from the PCA "
            f"(labels: {results.weight_pca['labels']})"
        )
        text = caplog.text
        assert "corrupt" in text and "corrupt_d1_w0" in text, (
            "the warning does not NAME the dropped model and the offending "
            f"weight, so a user cannot act on it. Logged: {text!r}"
        )

    def test_the_analysis_survives_when_too_few_rows_remain(self, tmp_path):
        """Fewer than two finite rows: warn and skip, never raise."""
        analyzer = _weights_only_analyzer(
            {"corrupt_a": _tiny_model("corrupt_a", corrupt=True),
             "clean": _tiny_model("clean")},
            tmp_path / "toofew",
        )
        results = analyzer.analyze(analysis_types={"weights"})
        assert results.weight_pca is None, (
            "a PCA was published from a single surviving row"
        )
        assert results.weight_stats, "the per-layer statistics were lost too"

    def test_the_pca_coordinates_are_unchanged_for_an_all_finite_case(self, tmp_path):
        """Bit-identity: dropping rows must not perturb a run with nothing to drop.

        The expected values are computed from the module's own inputs by the
        reference sklearn pipeline, NOT copied out of a previous run of the code
        under test, so this cannot pass by agreeing with a broken implementation.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        analyzer = _weights_only_analyzer(
            {"m1": _tiny_model("m1"), "m2": _tiny_model("m2"),
             "m3": _tiny_model("m3")},
            tmp_path / "finite",
        )
        results = analyzer.analyze(analysis_types={"weights"})
        got = np.asarray(results.weight_pca["components"], dtype=float)

        features = []
        for name in results.weight_pca["labels"]:
            row = []
            for layer_name in results.weight_stats_layer_order[name]:
                stats = results.weight_stats[name][layer_name]
                row.extend([stats[g][k] for g, k in PCA_LEAF_PATHS])
                row.append(stats["norms"].get("spectral", 0.0))
            features.append(row)
        expected = PCA(n_components=min(3, len(features))).fit_transform(
            StandardScaler().fit_transform(features))

        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)
