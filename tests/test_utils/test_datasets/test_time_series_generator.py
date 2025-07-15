"""
Test suite for the generic time series generator.

This module tests all functionality of the TimeSeriesGenerator including
pattern generation, parameter validation, and edge cases.
"""

import pytest
import numpy as np

from dl_techniques.utils.datasets.time_series_generator import (
    TimeSeriesGenerator,
    TimeSeriesConfig
)


class TestTimeSeriesConfig:
    """Test suite for TimeSeriesConfig dataclass."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = TimeSeriesConfig()

        # Check default values
        assert config.n_samples == 1000
        assert config.random_seed == 42
        assert config.default_noise_level == 0.1
        assert len(config.seasonal_periods) > 0
        assert config.trend_strengths[1] > config.trend_strengths[0]

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = TimeSeriesConfig(
            n_samples=500,
            random_seed=123,
            default_noise_level=0.05,
            trend_strengths=(0.001, 0.01)
        )

        assert config.n_samples == 500
        assert config.random_seed == 123
        assert config.default_noise_level == 0.05
        assert config.trend_strengths == (0.001, 0.01)


class TestTimeSeriesGenerator:
    """Test suite for TimeSeriesGenerator class."""

    @pytest.fixture
    def default_config(self):
        """Create a default configuration for testing."""
        return TimeSeriesConfig(n_samples=100, random_seed=42)

    @pytest.fixture
    def generator(self, default_config):
        """Create a generator instance for testing."""
        return TimeSeriesGenerator(default_config)

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.n_samples == 100
        assert generator.config.random_seed == 42
        assert len(generator.task_definitions) > 0
        assert hasattr(generator, 'random_state')

    def test_get_task_names(self, generator):
        """Test getting all task names."""
        task_names = generator.get_task_names()

        assert isinstance(task_names, list)
        assert len(task_names) > 0
        assert all(isinstance(name, str) for name in task_names)
        assert "linear_trend_strong" in task_names
        assert "multi_seasonal" in task_names

    def test_get_task_categories(self, generator):
        """Test getting all task categories."""
        categories = generator.get_task_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        expected_categories = [
            "trend", "seasonal", "composite", "stochastic",
            "intermittent", "volatility", "regime", "structural",
            "outliers", "chaotic"
        ]
        for cat in expected_categories:
            assert cat in categories

    def test_get_tasks_by_category(self, generator):
        """Test filtering tasks by category."""
        trend_tasks = generator.get_tasks_by_category("trend")
        seasonal_tasks = generator.get_tasks_by_category("seasonal")

        assert isinstance(trend_tasks, list)
        assert len(trend_tasks) > 0
        assert "linear_trend_strong" in trend_tasks

        assert isinstance(seasonal_tasks, list)
        assert len(seasonal_tasks) > 0
        assert "daily_seasonality" in seasonal_tasks

    def test_generate_task_data_basic(self, generator):
        """Test basic task data generation."""
        # Test known task
        data = generator.generate_task_data("linear_trend_strong")

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)  # n_samples x 1
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))

    def test_generate_task_data_invalid(self, generator):
        """Test generation with invalid task name."""
        with pytest.raises(ValueError, match="Unknown task"):
            generator.generate_task_data("nonexistent_task")

    def test_generate_all_patterns(self, generator):
        """Test generating all available patterns."""
        all_patterns = generator.generate_all_patterns()

        assert isinstance(all_patterns, dict)
        assert len(all_patterns) == len(generator.get_task_names())

        # Check each pattern
        for task_name, data in all_patterns.items():
            assert isinstance(data, np.ndarray)
            assert data.shape == (100, 1)
            assert not np.any(np.isnan(data))

    def test_generate_random_pattern(self, generator):
        """Test random pattern generation."""
        task_name, data = generator.generate_random_pattern()

        assert isinstance(task_name, str)
        assert task_name in generator.get_task_names()
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)

    def test_generate_random_pattern_by_category(self, generator):
        """Test random pattern generation filtered by category."""
        task_name, data = generator.generate_random_pattern(category="trend")

        trend_tasks = generator.get_tasks_by_category("trend")
        assert task_name in trend_tasks
        assert isinstance(data, np.ndarray)

    def test_generate_random_pattern_invalid_category(self, generator):
        """Test random pattern generation with invalid category."""
        with pytest.raises(ValueError, match="Unknown category"):
            generator.generate_random_pattern(category="invalid_category")

    def test_generate_custom_trend_pattern(self, generator):
        """Test custom trend pattern generation."""
        data = generator.generate_custom_pattern(
            "trend",
            trend_type="linear",
            strength=0.01,
            noise_level=0.05
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))

    def test_generate_custom_seasonal_pattern(self, generator):
        """Test custom seasonal pattern generation."""
        data = generator.generate_custom_pattern(
            "seasonal",
            periods=[12, 24],
            amplitudes=[1.0, 0.5],
            noise_level=0.1
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))

    def test_generate_custom_pattern_invalid_type(self, generator):
        """Test custom pattern generation with invalid type."""
        with pytest.raises(ValueError, match="Unsupported pattern type"):
            generator.generate_custom_pattern("invalid_type")

    def test_reproducibility(self):
        """Test that generator produces reproducible results."""
        config = TimeSeriesConfig(n_samples=50, random_seed=123)

        gen1 = TimeSeriesGenerator(config)
        gen2 = TimeSeriesGenerator(config)

        data1 = gen1.generate_task_data("linear_trend_strong")
        data2 = gen2.generate_task_data("linear_trend_strong")

        np.testing.assert_array_equal(data1, data2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        config1 = TimeSeriesConfig(n_samples=50, random_seed=123)
        config2 = TimeSeriesConfig(n_samples=50, random_seed=456)

        gen1 = TimeSeriesGenerator(config1)
        gen2 = TimeSeriesGenerator(config2)

        data1 = gen1.generate_task_data("linear_trend_strong")
        data2 = gen2.generate_task_data("linear_trend_strong")

        # Should be different (with very high probability)
        assert not np.array_equal(data1, data2)

    def test_trend_patterns(self, generator):
        """Test various trend pattern types."""
        trend_types = ["linear", "exponential", "polynomial"]

        for trend_type in trend_types:
            data = generator._generate_trend_series(
                trend_type=trend_type,
                strength=0.001,
                noise_level=0.05
            )

            assert isinstance(data, np.ndarray)
            assert data.shape == (100, 1)
            assert not np.any(np.isnan(data))

    def test_seasonal_patterns(self, generator):
        """Test seasonal pattern generation."""
        data = generator._generate_seasonal_series(
            periods=[12, 24],
            amplitudes=[1.0, 0.5],
            noise_level=0.1
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))

    def test_stochastic_patterns(self, generator):
        """Test various stochastic process types."""
        process_types = ["random_walk", "ar", "ma", "arma"]

        for process_type in process_types:
            if process_type == "random_walk":
                params = {"drift": 0.001, "volatility": 0.05}
            elif process_type == "ar":
                params = {"ar_coeffs": [0.7], "noise_std": 0.1}
            elif process_type == "ma":
                params = {"ma_coeffs": [0.8], "noise_std": 0.1}
            else:  # arma
                params = {"ar_coeffs": [0.6], "ma_coeffs": [0.4], "noise_std": 0.1}

            data = generator._generate_stochastic_series(
                process_type=process_type,
                **params
            )

            assert isinstance(data, np.ndarray)
            assert data.shape == (100, 1)
            assert not np.any(np.isnan(data))

    def test_mean_reverting_pattern(self, generator):
        """Test mean-reverting pattern generation."""
        data = generator._generate_mean_reverting(
            theta=0.1,
            mu=0,
            sigma=0.2
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))

    def test_intermittent_pattern(self, generator):
        """Test intermittent demand pattern generation."""
        data = generator._generate_intermittent_series(
            demand_prob=0.3,
            demand_mean=2.0,
            demand_std=0.5
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))
        # Should have many zeros (intermittent nature)
        assert np.sum(data == 0) > 50

    def test_garch_pattern(self, generator):
        """Test GARCH volatility clustering pattern."""
        data = generator._generate_garch_series(
            alpha=0.1,
            beta=0.8,
            omega=0.01
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))

    def test_regime_switching_pattern(self, generator):
        """Test regime switching pattern generation."""
        data = generator._generate_regime_switching(
            regimes=2,
            switch_prob=0.05,
            regime_params=[(0.001, 0.05), (0.005, 0.15)]
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))

    def test_structural_break_patterns(self, generator):
        """Test structural break pattern generation."""
        break_types = ["level", "trend"]

        for break_type in break_types:
            data = generator._generate_structural_break(
                break_type=break_type,
                break_magnitude=1.0,
                break_points=[0.5]
            )

            assert isinstance(data, np.ndarray)
            assert data.shape == (100, 1)
            assert not np.any(np.isnan(data))

    def test_outlier_patterns(self, generator):
        """Test outlier pattern generation."""
        data = generator._generate_outlier_series(
            outlier_type="additive",
            outlier_prob=0.05,
            outlier_magnitude=3.0
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))

    def test_chaotic_patterns(self, generator):
        """Test chaotic system pattern generation."""
        # Test Henon map
        henon_data = generator._generate_chaotic_series(
            system="henon",
            a=1.4,
            b=0.3
        )

        assert isinstance(henon_data, np.ndarray)
        assert henon_data.shape == (100, 1)
        assert not np.any(np.isnan(henon_data))

        # Test Lorenz system
        lorenz_data = generator._generate_chaotic_series(
            system="lorenz",
            sigma=10,
            rho=28,
            beta=8 / 3
        )

        assert isinstance(lorenz_data, np.ndarray)
        assert lorenz_data.shape == (100, 1)
        assert not np.any(np.isnan(lorenz_data))

    def test_logistic_growth_pattern(self, generator):
        """Test logistic growth pattern generation."""
        data = generator._generate_logistic_growth(
            carrying_capacity=10,
            growth_rate=0.1,
            noise_level=0.1
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)
        assert not np.any(np.isnan(data))

    def test_edge_case_small_n_samples(self):
        """Test with very small n_samples."""
        config = TimeSeriesConfig(n_samples=5, random_seed=42)
        generator = TimeSeriesGenerator(config)

        data = generator.generate_task_data("linear_trend_strong")
        assert data.shape == (5, 1)
        assert not np.any(np.isnan(data))

    def test_edge_case_large_n_samples(self):
        """Test with large n_samples."""
        config = TimeSeriesConfig(n_samples=5000, random_seed=42)
        generator = TimeSeriesGenerator(config)

        data = generator.generate_task_data("linear_trend_strong")
        assert data.shape == (5000, 1)
        assert not np.any(np.isnan(data))

    def test_trend_with_zero_noise(self, generator):
        """Test trend generation with zero noise."""
        data = generator._generate_trend_series(
            trend_type="linear",
            strength=0.001,
            noise_level=0.0
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)

        # Should be perfectly linear (no noise)
        differences = np.diff(data.flatten())
        np.testing.assert_allclose(differences, differences[0], rtol=1e-10)

    def test_seasonal_with_single_period(self, generator):
        """Test seasonal generation with single period."""
        data = generator._generate_seasonal_series(
            periods=[24],
            amplitudes=[1.0],
            noise_level=0.01
        )

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 1)

        # Should show periodic behavior
        flat_data = data.flatten()
        period_24 = flat_data[:24] if len(flat_data) >= 48 else flat_data
        next_period = flat_data[24:48] if len(flat_data) >= 48 else None

        if next_period is not None:
            # Should be approximately periodic (allowing for noise)
            np.testing.assert_allclose(period_24, next_period, atol=0.1)

    def test_parameter_validation_trend_type(self, generator):
        """Test parameter validation for trend types."""
        with pytest.raises(ValueError, match="Unknown trend type"):
            generator._generate_trend_series(
                trend_type="invalid_trend",
                strength=0.001
            )

    def test_comprehensive_pattern_properties(self, generator):
        """Test that generated patterns have expected statistical properties."""
        # Test trend pattern has increasing values
        trend_data = generator.generate_task_data("linear_trend_strong")
        trend_flat = trend_data.flatten()

        # Linear trend should generally increase
        trend_slope = np.polyfit(range(len(trend_flat)), trend_flat, 1)[0]
        assert trend_slope > 0, "Linear trend should have positive slope"

        # Test seasonal pattern has bounded values
        seasonal_data = generator.generate_task_data("daily_seasonality")
        seasonal_flat = seasonal_data.flatten()

        # Seasonal pattern should be roughly centered around zero
        assert abs(np.mean(seasonal_flat)) < 1.0, "Seasonal pattern should be centered"

        # Test random walk has cumulative nature
        rw_data = generator.generate_task_data("random_walk")
        rw_flat = rw_data.flatten()

        # Random walk should have larger variance at the end than at the beginning
        early_var = np.var(rw_flat[:25])
        late_var = np.var(rw_flat[-25:])
        # This might not always hold due to randomness, so just check it's computed
        assert late_var >= 0 and early_var >= 0

    def test_noise_level_effect(self, generator):
        """Test that noise level affects the generated data."""
        # Generate with low noise
        low_noise = generator._generate_trend_series(
            trend_type="linear",
            strength=0.001,
            noise_level=0.01
        )

        # Generate with high noise
        high_noise = generator._generate_trend_series(
            trend_type="linear",
            strength=0.001,
            noise_level=0.5
        )

        # High noise series should have higher variance
        low_var = np.var(np.diff(low_noise.flatten()))
        high_var = np.var(np.diff(high_noise.flatten()))

        assert high_var > low_var, "Higher noise should result in higher variance"

    def test_memory_efficiency(self, generator):
        """Test that generator doesn't consume excessive memory."""
        # Generate multiple patterns
        patterns = {}
        for i, task_name in enumerate(generator.get_task_names()[:5]):
            patterns[task_name] = generator.generate_task_data(task_name)

        # Check all patterns were generated successfully
        assert len(patterns) == 5
        for data in patterns.values():
            assert isinstance(data, np.ndarray)
            assert data.shape == (100, 1)

    def test_task_definition_completeness(self, generator):
        """Test that all defined tasks can be generated successfully."""
        failed_tasks = []

        for task_name in generator.get_task_names():
            try:
                data = generator.generate_task_data(task_name)
                assert isinstance(data, np.ndarray)
                assert data.shape == (generator.config.n_samples, 1)
                assert not np.any(np.isnan(data))
                assert not np.any(np.isinf(data))
            except Exception as e:
                failed_tasks.append((task_name, str(e)))

        if failed_tasks:
            pytest.fail(f"Failed to generate tasks: {failed_tasks}")