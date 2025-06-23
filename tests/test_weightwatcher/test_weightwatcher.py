"""
Basic pytest suite for WeightWatcher module

Run with: pytest test_weightwatcher.py -v
"""

import pytest
import numpy as np
import pandas as pd
import keras
import tempfile
import os
from unittest.mock import patch

from dl_techniques.weightwatcher import (
    WeightWatcher, analyze_model, compare_models, get_critical_layers
)
from dl_techniques.weightwatcher.metrics import (
    calculate_gini_coefficient, calculate_dominance_ratio,
    calculate_participation_ratio, fit_powerlaw,
    calculate_concentration_metrics, compute_eigenvalues
)
from dl_techniques.weightwatcher.weights_utils import (
    infer_layer_type, get_layer_weights_and_bias, get_weight_matrices
)
from dl_techniques.weightwatcher.constants import LayerType


@pytest.fixture
def simple_dense_model():
    """Create a simple dense model for testing."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


@pytest.fixture
def simple_conv_model():
    """Create a simple convolutional model for testing."""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


@pytest.fixture
def sample_eigenvalues():
    """Create sample eigenvalues for testing."""
    # Create eigenvalues with power-law-like distribution
    x = np.linspace(0.1, 10, 100)
    evals = x ** -2.5 + np.random.normal(0, 0.01, len(x))
    return np.maximum(evals, 0.001)  # Ensure positive


@pytest.fixture
def sample_weight_matrix():
    """Create sample weight matrix for testing."""
    np.random.seed(42)
    return np.random.randn(50, 30)


class TestMetricFunctions:
    """Test standalone metric functions."""

    def test_gini_coefficient_basic(self):
        """Test Gini coefficient calculation."""
        # Perfect equality
        equal_vals = np.ones(10)
        gini_equal = calculate_gini_coefficient(equal_vals)
        assert 0 <= gini_equal <= 0.1, "Equal values should have low Gini coefficient"

        # Perfect inequality
        unequal_vals = np.array([0, 0, 0, 0, 100])
        gini_unequal = calculate_gini_coefficient(unequal_vals)
        assert gini_unequal > 0.5, "Unequal values should have high Gini coefficient"

        # Edge case: single value
        single_val = np.array([1.0])
        gini_single = calculate_gini_coefficient(single_val)
        assert gini_single == 0.0, "Single value should have zero Gini coefficient"

    def test_dominance_ratio(self):
        """Test dominance ratio calculation."""
        # Normal case
        evals = np.array([10, 5, 3, 2, 1])
        dominance = calculate_dominance_ratio(evals)
        expected = 10 / (5 + 3 + 2 + 1)  # 10/11
        assert abs(dominance - expected) < 0.001

        # Edge case: single value
        single_val = np.array([5.0])
        dominance_single = calculate_dominance_ratio(single_val)
        assert dominance_single == float('inf')

    def test_participation_ratio(self):
        """Test participation ratio calculation."""
        # Uniform vector (high participation)
        uniform_vec = np.ones(10)
        pr_uniform = calculate_participation_ratio(uniform_vec)
        assert pr_uniform > 5, "Uniform vector should have high participation ratio"

        # Localized vector (low participation)
        localized_vec = np.zeros(10)
        localized_vec[0] = 1.0
        pr_localized = calculate_participation_ratio(localized_vec)
        assert pr_localized <= 2, "Localized vector should have low participation ratio"

    def test_fit_powerlaw(self, sample_eigenvalues):
        """Test power-law fitting."""
        alpha, xmin, D, sigma, num_spikes, status, warning = fit_powerlaw(sample_eigenvalues)

        assert alpha > 0, "Alpha should be positive"
        assert xmin > 0, "Xmin should be positive"
        assert 0 <= D <= 1, "KS statistic should be between 0 and 1"
        assert sigma >= 0, "Sigma should be non-negative"
        assert num_spikes >= 0, "Number of spikes should be non-negative"
        assert status in ['success', 'failed'], "Status should be success or failed"

    def test_fit_powerlaw_insufficient_data(self):
        """Test power-law fitting with insufficient data."""
        small_evals = np.array([1.0, 2.0])  # Too few eigenvalues
        alpha, xmin, D, sigma, num_spikes, status, warning = fit_powerlaw(small_evals)

        assert status == 'failed', "Should fail with insufficient data"
        assert alpha == -1, "Alpha should be -1 for failed fit"

    def test_concentration_metrics(self, sample_weight_matrix):
        """Test concentration metrics calculation."""
        metrics = calculate_concentration_metrics(sample_weight_matrix)

        required_metrics = [
            'gini_coefficient', 'dominance_ratio', 'participation_ratio',
            'concentration_score', 'critical_weights'  # This should be in the direct calculation
        ]

        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        # Check types and ranges
        assert 0 <= metrics['gini_coefficient'] <= 1
        assert metrics['dominance_ratio'] >= 0
        assert metrics['participation_ratio'] >= 0
        assert isinstance(metrics['critical_weights'], list)

        # Check critical weights structure
        if metrics['critical_weights']:
            weight_info = metrics['critical_weights'][0]
            assert len(weight_info) == 3, "Critical weight should be (i, j, contribution)"

    def test_compute_eigenvalues(self, sample_weight_matrix):
        """Test eigenvalue computation."""
        Wmats = [sample_weight_matrix]
        N, M = max(sample_weight_matrix.shape), min(sample_weight_matrix.shape)

        evals, sv_max, sv_min, rank_loss = compute_eigenvalues(Wmats, N, M, M)

        assert len(evals) > 0, "Should compute eigenvalues"
        assert sv_max >= sv_min >= 0, "Singular values should be ordered"
        assert rank_loss >= 0, "Rank loss should be non-negative"
        assert np.all(evals >= 0), "Eigenvalues should be non-negative"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_infer_layer_type(self, simple_dense_model, simple_conv_model):
        """Test layer type inference."""
        # Dense layer
        dense_layer = simple_dense_model.layers[0]
        assert infer_layer_type(dense_layer) == LayerType.DENSE

        # Conv2D layer
        conv_layer = simple_conv_model.layers[0]
        assert infer_layer_type(conv_layer) == LayerType.CONV2D

        # MaxPooling layer (should be unknown)
        pool_layer = simple_conv_model.layers[1]
        assert infer_layer_type(pool_layer) == LayerType.UNKNOWN

    def test_get_layer_weights_and_bias(self, simple_dense_model):
        """Test weight and bias extraction."""
        layer = simple_dense_model.layers[0]

        has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

        assert has_weights, "Dense layer should have weights"
        assert weights is not None, "Weights should not be None"
        assert weights.shape == (32, 64), "Weight shape should match layer configuration"
        assert has_bias, "Dense layer should have bias by default"
        assert bias is not None, "Bias should not be None"
        assert bias.shape == (64,), "Bias shape should match output units"

    def test_get_weight_matrices_dense(self, simple_dense_model):
        """Test weight matrix extraction for dense layers."""
        layer = simple_dense_model.layers[0]
        _, weights, _, _ = get_layer_weights_and_bias(layer)

        Wmats, N, M, rf = get_weight_matrices(weights, LayerType.DENSE)

        assert len(Wmats) == 1, "Dense layer should have one weight matrix"
        assert N == 64, "N should be max dimension"
        assert M == 32, "M should be min dimension"
        assert rf == 1.0, "Receptive field should be 1 for dense layers"

    def test_get_weight_matrices_conv2d(self, simple_conv_model):
        """Test weight matrix extraction for conv2d layers."""
        layer = simple_conv_model.layers[0]
        _, weights, _, _ = get_layer_weights_and_bias(layer)

        Wmats, N, M, rf = get_weight_matrices(weights, LayerType.CONV2D)

        assert len(Wmats) == 1, "Should return one reshaped matrix"
        assert rf == 9, "Receptive field should be 3*3=9"
        assert N > M, "N should be greater than M"


class TestWeightWatcherClass:
    """Test main WeightWatcher class."""

    def test_initialization(self, simple_dense_model):
        """Test WeightWatcher initialization."""
        watcher = WeightWatcher(simple_dense_model)
        assert watcher.model == simple_dense_model
        assert watcher.details is None
        assert watcher.results is None

    def test_describe(self, simple_dense_model):
        """Test model description."""
        watcher = WeightWatcher(simple_dense_model)
        details = watcher.describe()

        assert isinstance(details, pd.DataFrame)
        assert len(details) > 0, "Should describe at least one layer"

        # Check required columns
        required_cols = ['name', 'layer_type', 'N', 'M', 'num_params']
        for col in required_cols:
            assert col in details.columns, f"Missing column: {col}"

    def test_analyze_basic(self, simple_dense_model):
        """Test basic analysis."""
        watcher = WeightWatcher(simple_dense_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            analysis = watcher.analyze(
                plot=False,  # Disable plotting for faster tests
                concentration_analysis=True,
                savefig=False
            )

        assert isinstance(analysis, pd.DataFrame)
        assert len(analysis) > 0, "Should analyze at least one layer"

        # Check for key metrics
        key_metrics = ['alpha', 'entropy', 'stable_rank']
        for metric in key_metrics:
            assert metric in analysis.columns, f"Missing metric: {metric}"

    def test_analyze_with_concentration(self, simple_dense_model):
        """Test analysis with concentration metrics."""
        watcher = WeightWatcher(simple_dense_model)
        analysis = watcher.analyze(
            plot=False,
            concentration_analysis=True
        )

        # Check for concentration metrics (but not critical_weights in DataFrame)
        concentration_metrics = ['gini_coefficient', 'dominance_ratio', 'concentration_score', 'critical_weight_count']
        for metric in concentration_metrics:
            assert metric in analysis.columns, f"Missing concentration metric: {metric}"

    def test_get_summary(self, simple_dense_model):
        """Test summary generation."""
        watcher = WeightWatcher(simple_dense_model)
        watcher.analyze(plot=False, concentration_analysis=True)

        summary = watcher.get_summary()

        assert isinstance(summary, dict)
        assert len(summary) > 0, "Summary should not be empty"

        # Check for key summary metrics
        expected_metrics = ['alpha', 'entropy', 'stable_rank']
        for metric in expected_metrics:
            if metric in summary:
                # Check if it's a numeric type (including numpy types)
                assert np.isscalar(summary[metric]) and np.isreal(summary[metric]), f"{metric} should be numeric"

    def test_get_ESD(self, simple_dense_model):
        """Test eigenvalue spectrum extraction."""
        watcher = WeightWatcher(simple_dense_model)

        evals = watcher.get_ESD(layer_id=0)

        assert isinstance(evals, np.ndarray)
        assert len(evals) > 0, "Should return eigenvalues"
        assert np.all(evals >= 0), "Eigenvalues should be non-negative"

    def test_get_layer_concentration_metrics(self, simple_dense_model):
        """Test layer-specific concentration metrics."""
        watcher = WeightWatcher(simple_dense_model)

        # First run analysis to populate internal critical weights storage
        watcher.analyze(plot=False, concentration_analysis=True)

        # Now get layer concentration metrics
        metrics = watcher.get_layer_concentration_metrics(layer_id=0)

        assert isinstance(metrics, dict)
        # Should have concentration metrics
        expected_metrics = ['gini_coefficient', 'dominance_ratio', 'participation_ratio', 'concentration_score']
        for metric in expected_metrics:
            if metric in metrics:
                # Check if it's a numeric type (including numpy types)
                assert np.isscalar(metrics[metric]) and np.isreal(metrics[metric]), f"{metric} should be numeric"

        # Should also have critical_weights after analysis
        if 'critical_weights' in metrics:
            assert isinstance(metrics['critical_weights'], list)


class TestMainAnalyzerFunctions:
    """Test main analyzer interface functions."""

    def test_analyze_model(self, simple_dense_model):
        """Test analyze_model function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = analyze_model(
                simple_dense_model,
                plot=False,
                concentration_analysis=True,
                savedir=tmpdir
            )

        assert isinstance(results, dict)
        required_keys = ['analysis', 'summary', 'recommendations']
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

        assert isinstance(results['analysis'], pd.DataFrame)
        assert isinstance(results['summary'], dict)
        assert isinstance(results['recommendations'], list)

    def test_compare_models(self, simple_dense_model):
        """Test model comparison."""
        # Create a slightly modified model
        model2 = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(32,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = compare_models(
                simple_dense_model,
                model2,
                savedir=tmpdir
            )

        assert isinstance(comparison, dict)
        required_keys = ['original_analysis', 'modified_analysis', 'metric_comparison']
        for key in required_keys:
            assert key in comparison, f"Missing key: {key}"

    def test_get_critical_layers(self, simple_dense_model):
        """Test critical layer identification."""
        critical_layers = get_critical_layers(
            simple_dense_model,
            criterion='parameters',
            top_k=2
        )

        assert isinstance(critical_layers, list)
        assert len(critical_layers) <= 2, "Should return at most top_k layers"

        if len(critical_layers) > 0:
            layer_info = critical_layers[0]
            required_fields = ['layer_id', 'name', 'type', 'rank']
            for field in required_fields:
                assert field in layer_info, f"Missing field: {field}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model_error(self):
        """Test error handling for None model."""
        watcher = WeightWatcher(None)

        with pytest.raises(ValueError, match="No model provided"):
            watcher.analyze()

    def test_invalid_layer_id(self, simple_dense_model):
        """Test invalid layer ID handling."""
        watcher = WeightWatcher(simple_dense_model)

        with pytest.raises(ValueError, match="Layer ID .* out of range"):
            watcher.get_ESD(layer_id=999)

    def test_model_with_no_analyzable_layers(self):
        """Test model with no analyzable layers."""
        # Create model with only non-analyzable layers
        model = keras.Sequential([
            keras.layers.Input(shape=(10,)),
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization()
        ])

        watcher = WeightWatcher(model)
        analysis = watcher.analyze(plot=False)

        # Should return empty DataFrame
        assert isinstance(analysis, pd.DataFrame)
        # Note: This might return empty or have minimal results

    def test_small_matrix_handling(self):
        """Test handling of very small matrices."""
        small_matrix = np.random.randn(3, 2)  # Below minimum eigenvalue threshold

        # Should not crash
        metrics = calculate_concentration_metrics(small_matrix)
        assert isinstance(metrics, dict)  # Might be empty, but shouldn't crash


class TestIntegration:
    """Integration tests."""

    def test_full_analysis_pipeline(self, simple_conv_model):
        """Test complete analysis pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Full analysis
            results = analyze_model(
                simple_conv_model,
                plot=True,
                concentration_analysis=True,
                detailed_plots=True,
                savedir=tmpdir
            )

            # Check outputs were created
            assert os.path.exists(os.path.join(tmpdir, 'analysis_summary.json'))
            assert os.path.exists(os.path.join(tmpdir, 'layer_analysis.csv'))

            # Check results structure
            assert 'analysis' in results
            assert 'summary' in results
            assert len(results['analysis']) > 0
            assert len(results['summary']) > 0

    @pytest.mark.parametrize("model_fixture", ["simple_dense_model", "simple_conv_model"])
    def test_different_model_types(self, model_fixture, request):
        """Test analysis on different model types."""
        model = request.getfixturevalue(model_fixture)

        watcher = WeightWatcher(model)
        analysis = watcher.analyze(plot=False, concentration_analysis=True)

        assert len(analysis) > 0, f"Should analyze {model_fixture}"
        assert 'alpha' in analysis.columns
        assert 'concentration_score' in analysis.columns


# Utility function for running tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])