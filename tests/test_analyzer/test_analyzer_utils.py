"""`analyzer/utils.py` had NO direct test module. These are its load-bearing contracts.

Why (F-063). `smooth_curve` and `find_metric_in_history` are the two most-reused
functions in the analyzer package — every training-dynamics plot and every training
metric goes through them — and neither had a single test. That is how two real defects
survived an otherwise heavily-guarded package:

- `smooth_curve` returned `len(values) + 1` samples for an EVEN `window_size`, because
  the padding was symmetric (`w//2` each side) while a `'valid'` convolution only
  returns `len(v)` when `w` is odd. `training_dynamics_visualizer.py` plots the result
  against `range(len(train_loss))`, so an even `smoothing_window` put a 21-point curve
  on a 20-point axis.
- `find_metric_in_history`'s `exclude_prefixes=['val_']` uses `str.startswith`, which
  does not match `validation_loss`. Keras prefixes `validation_` (not `val_`) when
  `validation_data` is a `tf.data.Dataset`, so the VALIDATION curve was returned as the
  TRAINING curve and `overfitting_index` came out identically 0 for a model that was
  overfitting.

Plus the small contracts that are easy to break and hard to notice: the truncating
`find_pareto_front`, `normalize_metric`'s all-equal collapse, `truncate_model_name`'s
even/odd split, and `DataSampler`'s without-replacement draw.
"""

import numpy as np
import pytest

from dl_techniques.analyzer import constants, utils as az
from dl_techniques.analyzer.data_types import DataInput


class TestSmoothCurveAlwaysReturnsTheInputLength:
    """F-042: an even window used to return one sample too many."""

    @pytest.mark.parametrize("window", list(range(1, 13)))
    def test_the_length_is_preserved_for_every_window(self, window):
        values = np.arange(20, dtype=float)
        smoothed = az.smooth_curve(values, window)
        assert len(smoothed) == len(values), (
            f"window={window} returned {len(smoothed)} samples for {len(values)} "
            f"inputs. A plot against range(len(values)) would then have had a "
            f"length mismatch."
        )

    @pytest.mark.parametrize("window", [3, 5, 7, 9, 11])
    def test_odd_windows_are_bit_identical_to_the_original_symmetric_form(self, window):
        """The odd case was already correct; the fix must not move it.

        The F-042 fix changed the padding to `left = (w-1)//2`. For odd `w` that reduces
        to the original `w//2` on both sides, so every previously-correct window must be
        bit-for-bit unchanged. If this fails, a real consumer's published curves moved.
        """
        values = np.arange(31, dtype=float) * 0.37
        old = np.convolve(
            np.pad(values, (window // 2, window // 2), mode='edge'),
            np.ones(window) / window,
            mode='valid',
        )
        assert np.array_equal(az.smooth_curve(values, window), old)

    def test_a_short_input_is_returned_as_an_independent_copy(self):
        """The short path used to return the CALLER'S array, not a copy.

        `TrainingMetrics.smoothed_curves[model][metric]` therefore aliased the caller's
        history list: mutating one mutated the other, and the "smoothed" series was the
        raw one by reference rather than by accident of plotting.
        """
        values = np.array([1.0, 2.0, 3.0])
        result = az.smooth_curve(values, 50)
        assert result is not values
        np.testing.assert_array_equal(result, values)
        result[0] = 999.0
        assert values[0] == 1.0, "mutating the result also mutated the caller's array"

    def test_a_zero_window_is_rejected(self):
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            az.smooth_curve(np.arange(10.0), 0)

    def test_the_output_is_a_plain_float_array(self):
        out = az.smooth_curve(np.arange(10), 5)
        assert isinstance(out, np.ndarray)
        assert np.issubdtype(out.dtype, np.floating)


class TestFindMetricInHistoryHonoursExcludePrefixes:
    """F-064: `exclude_prefixes=['val_']` silently missed `validation_*`."""

    def test_the_usual_case_still_excludes_val(self):
        history = {'loss': [1, 2, 3, 4], 'val_loss': [2, 3, 4, 5]}
        assert az.find_metric_in_history(
            history, constants.LOSS_PATTERNS,
            exclude_prefixes=['val_']) == [1, 2, 3, 4]

    def test_validation_prefixed_key_is_also_excluded(self):
        """The defect. Keras emits `validation_*` when validation_data is a Dataset.

        Without the fix this returned the validation curve as the training curve, so
        `overfitting_index = val_final - train_final` was identically 0.
        """
        history = {'training_loss': [1, 2, 3, 4], 'validation_loss': [2, 3, 4, 5]}
        found = az.find_metric_in_history(
            history, constants.LOSS_PATTERNS, exclude_prefixes=['val_'])
        assert found == [1, 2, 3, 4], (
            f"got {found!r}: the validation curve was returned as the training curve"
        )

    def test_the_same_holds_for_accuracy(self):
        history = {'accuracy': [0.1, 0.2], 'validation_accuracy': [0.05, 0.1]}
        found = az.find_metric_in_history(
            history, constants.ACC_PATTERNS, exclude_prefixes=['val_'])
        assert found == [0.1, 0.2]

    def test_an_excluded_only_history_finds_nothing(self):
        history = {'validation_loss': [2, 3, 4]}
        assert az.find_metric_in_history(
            history, constants.LOSS_PATTERNS, exclude_prefixes=['val_']) is None

    def test_an_exact_match_still_beats_a_fuzzy_one(self):
        history = {'val_loss': [9], 'loss': [1]}
        assert az.find_metric_in_history(
            history, constants.LOSS_PATTERNS, exclude_prefixes=['val_']) == [1]


class TestFindParetoFrontDominance:
    """O(n^2) but only n<=len(models); the LOGIC is what matters."""

    def test_a_dominated_point_is_excluded(self):
        # costs1 maximised, costs2 maximised: (3, 3) dominates both others.
        front = az.find_pareto_front(np.array([1.0, 2.0, 3.0]),
                                     np.array([1.0, 2.0, 3.0]))
        assert front == [2]

    def test_mutually_dominating_points_are_all_kept(self):
        front = az.find_pareto_front(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
        assert front == [0, 1], "neither point dominates the other, so both are optimal"

    def test_identical_points_are_all_optimal(self):
        """Neither STRICTLY dominates, so a tie is not dominance.

        `find_pareto_front` uses `>=` for the domination test and `>` for the strictness
        test, which is what makes a duplicate point Pareto-optimal rather than dropped.
        """
        front = az.find_pareto_front(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert front == [0, 1]

    def test_the_result_is_sorted_and_deduplicated_free(self):
        front = az.find_pareto_front(np.array([3.0, 1.0, 2.0]),
                                     np.array([1.0, 3.0, 2.0]))
        assert front == sorted(front)
        assert len(front) == len(set(front))


class TestNormalizeMetric:
    def test_higher_better_maps_the_maximum_to_one(self):
        out = az.normalize_metric([1.0, 2.0, 3.0], higher_better=True)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_lower_better_inverts(self):
        out = az.normalize_metric([1.0, 2.0, 3.0], higher_better=False)
        np.testing.assert_allclose(out, [1.0, 0.5, 0.0])

    def test_an_all_equal_input_collapses_to_the_midpoint(self):
        out = az.normalize_metric([5.0, 5.0, 5.0])
        np.testing.assert_allclose(out, [0.5, 0.5, 0.5])

    def test_empty_input_is_returned_empty(self):
        assert az.normalize_metric([]).size == 0


class TestTruncateModelName:
    @pytest.mark.parametrize("length", range(1, 24))
    def test_the_result_never_exceeds_max_len(self, length):
        name = "m" * length
        assert len(az.truncate_model_name(name)) <= 12

    @pytest.mark.parametrize("length", range(1, 13))
    def test_a_short_name_is_returned_unchanged(self, length):
        name = "n" * length
        assert az.truncate_model_name(name) == name

    def test_a_tiny_max_len_degrades_to_a_prefix(self):
        # chars_to_keep would go negative; the guard must not slice from the end.
        assert az.truncate_model_name("abcdefgh", max_len=4) == "abcd"

    def test_the_filler_is_present_for_a_long_name(self):
        assert "..." in az.truncate_model_name("a" * 30)


class TestDataSamplerDrawsWithoutReplacement:
    def _data(self, n=50):
        x = np.arange(n * 3, dtype=float).reshape(n, 3)
        y = np.arange(n)
        return DataInput(x_data=x, y_data=y)

    def test_a_short_dataset_is_passed_through(self):
        data = self._data(10)
        out = az.DataSampler.sample(data, 1000)
        assert len(out.x_data) == 10
        assert len(out.y_data) == 10

    def test_the_draw_is_a_subset_of_the_input(self):
        out = az.DataSampler.sample(self._data(50), 10)
        assert len(out.x_data) == 10
        # Every sampled x row must be an actual input row — not a re-drawn value.
        for row in out.x_data:
            assert any(np.array_equal(row, original) for original in self._data(50).x_data)

    def test_x_and_y_stay_aligned(self):
        """The sampled index must be applied to BOTH arrays."""
        data = self._data(50)
        out = az.DataSampler.sample(data, 10)
        for xs, ys in zip(out.x_data, out.y_data):
            matches = np.flatnonzero(np.all(data.x_data == xs, axis=1))
            assert matches.size == 1, "a sampled row does not correspond to one input"
            assert ys == data.y_data[matches[0]], "x and y were sampled inconsistently"

    def test_a_seeded_generator_reproduces_the_same_subset(self):
        data = self._data(50)
        first = az.DataSampler.sample(data, 10, rng=az.make_rng(1234))
        second = az.DataSampler.sample(data, 10, rng=az.make_rng(1234))
        np.testing.assert_array_equal(first.x_data, second.x_data)
        np.testing.assert_array_equal(first.y_data, second.y_data)

    def test_different_seeds_give_different_subsets(self):
        data = self._data(50)
        first = az.DataSampler.sample(data, 10, rng=az.make_rng(1))
        second = az.DataSampler.sample(data, 10, rng=az.make_rng(2))
        assert not np.array_equal(first.x_data, second.x_data)

    def test_dict_inputs_are_sampled_key_by_key_with_one_index_set(self):
        # `right` is `100 + index`, so the two keys share one index set iff
        # right - 100 == left elementwise. A per-key draw would desynchronise them and
        # hand the model mismatched inputs.
        x = {'left': np.arange(50).reshape(50, 1),
             'right': np.arange(100, 150).reshape(50, 1)}
        out = az.DataSampler.sample(DataInput(x_data=x, y_data=np.arange(50)), 10)
        assert len(out.x_data['left']) == 10
        assert len(out.x_data['right']) == 10
        np.testing.assert_array_equal(out.x_data['left'].ravel(),
                                      out.x_data['right'].ravel() - 100)


class TestValidateTrainingHistory:
    def test_an_empty_history_is_an_error(self):
        report = az.validate_training_history({})
        assert report['errors']

    def test_a_healthy_history_reports_neither(self):
        report = az.validate_training_history(
            {'loss': [1.0, 0.9], 'val_loss': [1.1, 1.0]})
        assert not report['errors']
        assert not report['warnings']

    def test_a_missing_validation_loss_is_a_warning_not_an_error(self):
        report = az.validate_training_history({'loss': [1.0, 0.9]})
        assert any('validation loss' in w for w in report['warnings'])
        assert not report['errors']
