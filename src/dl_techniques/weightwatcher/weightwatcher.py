"""
Enhanced TensorFlow WeightWatcher - A diagnostic tool for analyzing neural network weight matrices

This module provides comprehensive analysis of weight matrices in Keras models,
including power-law analysis, eigenvalue spectrum analysis, and concentration metrics.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import tempfile

import numpy as np
import pandas as pd
import keras
import warnings

from .constants import (
    DEFAULT_MIN_EVALS, DEFAULT_MAX_EVALS, DEFAULT_MAX_N, DEFAULT_SAVEDIR,
    WEAK_RANK_LOSS_TOLERANCE, DEFAULT_SUMMARY_METRICS,
    LayerType, MetricNames
)

from .metrics import (
    compute_eigenvalues, calculate_matrix_entropy, fit_powerlaw,
    calculate_spectral_metrics, jensen_shannon_distance,
    calculate_concentration_metrics, calculate_glorot_normalization_factor
)

from .weights_utils import (
    infer_layer_type, get_layer_weights_and_bias,
    get_weight_matrices, plot_powerlaw_fit
)

from dl_techniques.utils.logger import logger

# Suppress scipy/numpy warnings in eigenvalue calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)


class WeightWatcher:
    """
    Enhanced WeightWatcher for comprehensive analysis of Keras neural network weight matrices.

    This class performs spectral analysis including power-law fitting, entropy calculation,
    and concentration metrics to assess training quality and model complexity.
    """

    def __init__(self, model: Optional[keras.Model] = None):
        """
        Initialize the WeightWatcher with a model.

        Args:
            model: A Keras model to analyze. Can be set later.
        """
        self.model = model
        self.details = None
        self.results = None

        logger.info(f"Keras version: {keras.__version__}")

    def set_model(self, model: keras.Model) -> None:
        """
        Set the model to analyze.

        Args:
            model: A Keras model.
        """
        self.model = model

    def describe(self,
                 model: Optional[keras.Model] = None,
                 layers: List[int] = None,
                 min_evals: int = DEFAULT_MIN_EVALS,
                 max_evals: int = DEFAULT_MAX_EVALS,
                 max_N: int = DEFAULT_MAX_N) -> pd.DataFrame:
        """
        Describe the model architecture and layer dimensions without full analysis.

        Args:
            model: Keras model to analyze. Uses the model set in constructor if None.
            layers: List of layer indices to analyze. If None, analyze all layers.
            min_evals: Minimum number of eigenvalues required for analysis.
            max_evals: Maximum number of eigenvalues to analyze.
            max_N: Maximum matrix dimension to analyze.

        Returns:
            DataFrame with layer information.
        """
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Set model in constructor or pass as argument.")

        if layers is None:
            layers = []

        details = pd.DataFrame()

        for layer_id, layer in enumerate(self.model.layers):
            # Skip if layer is filtered out
            if layers and layer_id not in layers:
                continue

            layer_name = layer.name
            layer_type = infer_layer_type(layer)

            # Get weights and bias
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

            if not has_weights or layer_type == LayerType.UNKNOWN:
                continue

            # Extract weight matrices
            Wmats, N, M, rf = get_weight_matrices(weights, layer_type)

            # Check matrix dimensions against thresholds
            if M < min_evals or M > max_evals or N > max_N:
                continue

            # Create row data
            row_data = {
                'layer_id': layer_id,
                'name': layer_name,
                'layer_type': layer_type.value,  # Convert enum to string
                'N': N,
                'M': M,
                'rf': rf,
                'Q': N / M if M > 0 else -1,
                'num_params': int(np.prod(weights.shape)),
                MetricNames.NUM_EVALS: M * rf
            }

            # Add row to details DataFrame
            details = pd.concat([details, pd.DataFrame([row_data])], ignore_index=True)

        # Set layer_id as index
        if not details.empty:
            details.set_index('layer_id', inplace=True)

        self.details = details
        return details

    def analyze(self,
                model: Optional[keras.Model] = None,
                layers: List[int] = None,
                min_evals: int = DEFAULT_MIN_EVALS,
                max_evals: int = DEFAULT_MAX_EVALS,
                max_N: int = DEFAULT_MAX_N,
                glorot_fix: bool = False,
                plot: bool = False,
                randomize: bool = False,
                concentration_analysis: bool = True,
                savefig: Union[bool, str] = DEFAULT_SAVEDIR,
                tolerance: float = WEAK_RANK_LOSS_TOLERANCE) -> pd.DataFrame:
        """
        Analyze weight matrices of the model using spectral methods.

        Args:
            model: Keras model to analyze. Uses the model set in constructor if None.
            layers: List of layer indices to analyze. If None, analyze all layers.
            min_evals: Minimum number of eigenvalues required for analysis.
            max_evals: Maximum number of eigenvalues to analyze.
            max_N: Maximum matrix dimension to analyze.
            glorot_fix: Whether to apply Glorot normalization.
            plot: Whether to create plots of the eigenvalue distributions.
            randomize: Whether to analyze randomized weight matrices.
            concentration_analysis: Whether to perform concentration analysis.
            savefig: Directory to save figures or False to disable saving.
            tolerance: Tolerance for eigenvalue calculations.

        Returns:
            DataFrame with layer analysis results.
        """
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Set model in constructor or pass as argument.")

        if layers is None:
            layers = []

        # Setup save directory if needed
        savedir = DEFAULT_SAVEDIR
        if isinstance(savefig, str):
            savedir = savefig
        elif not savefig:
            savedir = None

        if savedir and not os.path.exists(savedir):
            os.makedirs(savedir, exist_ok=True)

        # Create a basic description of the model to start
        details = self.describe(
            model=self.model,
            layers=layers,
            min_evals=min_evals,
            max_evals=max_evals,
            max_N=max_N
        )

        if details.empty:
            logger.warning("No layers found that meet the criteria for analysis.")
            return details

        # Iterate through layers to analyze
        for layer_id in details.index:
            layer = self.model.layers[layer_id]
            layer_name = layer.name
            layer_type = infer_layer_type(layer)

            logger.info(f"Analyzing layer {layer_id}: {layer_name}")

            # Get weights and bias
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

            # Skip if no weights
            if not has_weights:
                continue

            # Extract weight matrices and dimensions
            Wmats, N, M, rf = get_weight_matrices(weights, layer_type)

            # Apply Glorot normalization if requested
            if glorot_fix:
                kappa = calculate_glorot_normalization_factor(N, M, rf)
                Wmats = [W / kappa for W in Wmats]

            # Calculate number of components to analyze
            n_comp = M * rf

            # Compute eigenvalues
            evals, sv_max, sv_min, rank_loss = compute_eigenvalues(
                Wmats, N, M, n_comp, normalize=False
            )

            # Calculate weak rank loss (eigenvalues below tolerance)
            weak_rank_loss = np.sum(evals < tolerance)

            # Fit power law to eigenvalue distribution
            alpha, xmin, D, sigma, num_pl_spikes, status, warning = fit_powerlaw(evals)

            # Plot if requested
            if plot and status == 'success' and savedir:
                plot_powerlaw_fit(
                    evals, alpha, xmin, D, sigma,
                    f"{layer_id}: {layer_name}", layer_id, savedir
                )

            # Calculate matrix entropy
            entropy = calculate_matrix_entropy(np.sqrt(evals), N)

            # Calculate spectral metrics
            spectral_metrics = calculate_spectral_metrics(evals, alpha)

            # Initialize concentration metrics
            concentration_metrics = {}

            # Calculate concentration metrics if requested
            if concentration_analysis:
                # Use the first (typically largest) weight matrix for concentration analysis
                if len(Wmats) > 0:
                    concentration_metrics = calculate_concentration_metrics(Wmats[0])

            # Analyze randomized matrices if requested
            randomization_metrics = {}
            if randomize:
                # Generate randomized matrices with same dimensions
                rand_Wmats = []
                for W in Wmats:
                    W_flat = W.flatten()
                    np.random.shuffle(W_flat)
                    rand_Wmats.append(W_flat.reshape(W.shape))

                # Compute eigenvalues for randomized matrices
                rand_evals, rand_sv_max, rand_sv_min, rand_rank_loss = compute_eigenvalues(
                    rand_Wmats, N, M, n_comp, normalize=False
                )

                # Calculate distance between original and randomized eigenvalues
                rand_distance = jensen_shannon_distance(evals, rand_evals)
                ww_softrank = np.max(rand_evals) / np.max(evals) if np.max(evals) > 0 else 0

                randomization_metrics = {
                    'rand_sv_max': rand_sv_max,
                    'rand_distance': rand_distance,
                    'ww_softrank': ww_softrank
                }

            # Combine all metrics
            metrics = {
                MetricNames.HAS_ESD: True,
                MetricNames.NUM_EVALS: len(evals),
                MetricNames.SV_MAX: sv_max,
                MetricNames.SV_MIN: sv_min,
                MetricNames.RANK_LOSS: rank_loss,
                MetricNames.WEAK_RANK_LOSS: weak_rank_loss,
                MetricNames.LAMBDA_MAX: np.max(evals) if len(evals) > 0 else 0,
                MetricNames.ALPHA: alpha,
                MetricNames.XMIN: xmin,
                MetricNames.D: D,
                MetricNames.SIGMA: sigma,
                MetricNames.NUM_PL_SPIKES: num_pl_spikes,
                MetricNames.STATUS: status,
                MetricNames.WARNING: warning,
                MetricNames.ENTROPY: entropy,
                **spectral_metrics,
                **randomization_metrics
            }

            # Add concentration metrics, but handle critical_weights separately
            for key, value in concentration_metrics.items():
                if key == 'critical_weights':
                    # Store critical weights count instead of the full list
                    metrics['critical_weight_count'] = len(value) if value else 0
                    # Store the critical weights separately for later access
                    if not hasattr(self, '_critical_weights'):
                        self._critical_weights = {}
                    self._critical_weights[layer_id] = value
                else:
                    metrics[key] = value

            # Update details with metrics
            for key, value in metrics.items():
                details.at[layer_id, key] = value

        self.details = details

        # Log summary of high concentration layers
        if concentration_analysis and 'concentration_score' in details.columns:
            high_conc_layers = details[details['concentration_score'] > details['concentration_score'].quantile(0.8)]
            if not high_conc_layers.empty:
                logger.info(f"Layers with high concentration scores: {list(high_conc_layers.index)}")

        return details

    def get_summary(self, details: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Get summary metrics averaged across all analyzed layers.

        Args:
            details: DataFrame with layer analysis results.
                    Uses self.details if None.

        Returns:
            Dictionary with summary metrics.
        """
        summary = {}

        if details is None:
            details = self.details

        if details is None or details.empty:
            logger.warning("No analysis details available. Run analyze() first.")
            return summary

        # Standard metrics
        metrics_to_summarize = DEFAULT_SUMMARY_METRICS.copy()

        # Add concentration metrics if available
        concentration_metrics = ['gini_coefficient', 'dominance_ratio', 'participation_ratio', 'concentration_score']
        for metric in concentration_metrics:
            if metric in details.columns:
                metrics_to_summarize.append(metric)

        # Calculate mean for each metric
        for metric in metrics_to_summarize:
            if metric in details.columns:
                # Skip non-numeric or failed values
                valid_values = details[metric][details[metric].apply(
                    lambda x: isinstance(x, (int, float)) and not np.isnan(x) and x > -1
                )]

                if not valid_values.empty:
                    summary[metric] = float(valid_values.mean())

        # Add counts
        summary['total_layers_analyzed'] = len(details)
        if 'concentration_score' in details.columns:
            high_conc_threshold = details['concentration_score'].quantile(0.8)
            summary['high_concentration_layers'] = int(sum(details['concentration_score'] > high_conc_threshold))

        return summary

    def get_ESD(self, layer_id: int) -> np.ndarray:
        """
        Get the eigenvalue spectrum distribution for a specific layer.

        Args:
            layer_id: ID of the layer to analyze.

        Returns:
            Array of eigenvalues.
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        if layer_id < 0 or layer_id >= len(self.model.layers):
            raise ValueError(f"Layer ID {layer_id} out of range.")

        layer = self.model.layers[layer_id]
        layer_type = infer_layer_type(layer)

        # Get weights
        has_weights, weights, _, _ = get_layer_weights_and_bias(layer)

        if not has_weights:
            logger.warning(f"Layer {layer_id} ({layer.name}) has no weights.")
            return np.array([])

        # Get weight matrices and calculate eigenvalues
        Wmats, N, M, rf = get_weight_matrices(weights, layer_type)
        n_comp = M * rf

        evals, _, _, _ = compute_eigenvalues(Wmats, N, M, n_comp)

        return evals

    def get_layer_concentration_metrics(self, layer_id: int) -> Dict[str, Any]:
        """
        Get detailed concentration metrics for a specific layer.

        Args:
            layer_id: ID of the layer to analyze.

        Returns:
            Dictionary with concentration metrics.
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        if layer_id < 0 or layer_id >= len(self.model.layers):
            raise ValueError(f"Layer ID {layer_id} out of range.")

        layer = self.model.layers[layer_id]
        layer_type = infer_layer_type(layer)

        # Get weights
        has_weights, weights, _, _ = get_layer_weights_and_bias(layer)

        if not has_weights:
            logger.warning(f"Layer {layer_id} ({layer.name}) has no weights.")
            return {}

        # Get weight matrices and calculate concentration metrics
        Wmats, N, M, rf = get_weight_matrices(weights, layer_type)

        if len(Wmats) == 0:
            return {}

        metrics = calculate_concentration_metrics(Wmats[0])

        # Add critical weights if they were stored during analysis
        if hasattr(self, '_critical_weights') and layer_id in self._critical_weights:
            metrics['critical_weights'] = self._critical_weights[layer_id]

        return metrics

    def create_smoothed_model(self,
                             method: str = 'detX',
                             percent: float = 0.8,
                             save_path: Optional[str] = None) -> keras.Model:
        """
        Create a smoothed version of the model using SVD truncation.

        Args:
            method: Smoothing method ('svd', 'detX', or 'lambda_min').
            percent: Percentage of eigenvalues to keep when using 'svd' method.
            save_path: Optional path to save the smoothed model.

        Returns:
            Smoothed Keras model.
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        if self.details is None:
            logger.info("Running analysis to get layer details...")
            self.analyze(plot=False)

        # Clone the model
        smoothed_model = keras.models.clone_model(self.model)
        smoothed_model.set_weights(self.model.get_weights())

        from .metrics import smooth_matrix, compute_detX_constraint

        # Process each analyzed layer
        for layer_id in self.details.index:
            layer = smoothed_model.layers[layer_id]
            layer_type = infer_layer_type(layer)

            # Skip unsupported layer types
            if layer_type not in [LayerType.DENSE, LayerType.CONV1D, LayerType.CONV2D]:
                continue

            # Get weights
            has_weights, old_weights, has_bias, old_bias = get_layer_weights_and_bias(layer)

            if not has_weights:
                continue

            # Determine number of components to keep
            if method == 'detX':
                evals = self.get_ESD(layer_id)
                num_smooth = compute_detX_constraint(evals)
            elif method == 'lambda_min':
                if MetricNames.NUM_PL_SPIKES in self.details.columns:
                    num_smooth = int(self.details.at[layer_id, MetricNames.NUM_PL_SPIKES])
                else:
                    num_smooth = int(0.5 * self.details.at[layer_id, MetricNames.NUM_EVALS])
            else:  # 'svd' or default
                num_smooth = int(percent * self.details.at[layer_id, MetricNames.NUM_EVALS])

            logger.info(f"Layer {layer_id} ({layer.name}): keeping {num_smooth} components")

            # Apply smoothing based on layer type
            if layer_type == LayerType.DENSE:
                new_weights = smooth_matrix(old_weights, num_smooth)

                # Update layer weights
                if has_bias:
                    layer.set_weights([new_weights, old_bias])
                else:
                    layer.set_weights([new_weights])

            elif layer_type in [LayerType.CONV1D, LayerType.CONV2D]:
                # For convolutional layers, apply smoothing to reshaped weights
                if layer_type == LayerType.CONV2D:
                    kh, kw, in_c, out_c = old_weights.shape
                    # Reshape to 2D for smoothing
                    W_reshaped = old_weights.reshape(-1, out_c)
                else:  # CONV1D
                    kernel_size, input_dim, output_dim = old_weights.shape
                    W_reshaped = old_weights.reshape(-1, output_dim)

                # Apply smoothing
                W_smoothed = smooth_matrix(W_reshaped, num_smooth)

                # Reshape back to original dimensions
                new_weights = W_smoothed.reshape(old_weights.shape)

                # Update layer weights
                if has_bias:
                    layer.set_weights([new_weights, old_bias])
                else:
                    layer.set_weights([new_weights])

        # Save if requested
        if save_path:
            smoothed_model.save(save_path)
            logger.info(f"Saved smoothed model to {save_path}")

        return smoothed_model

    def compare_models(self,
                      other_model: keras.Model,
                      test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Compare this model with another model (e.g., smoothed version).

        Args:
            other_model: Model to compare against.
            test_data: Optional tuple of (x_test, y_test) for performance comparison.

        Returns:
            Dictionary with comparison results.
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        # Analyze both models
        original_summary = self.get_summary()

        # Analyze the other model
        other_watcher = WeightWatcher(other_model)
        other_details = other_watcher.analyze(plot=False)
        other_summary = other_watcher.get_summary()

        comparison = {
            'original_summary': original_summary,
            'other_summary': other_summary,
            'metric_changes': {}
        }

        # Calculate metric changes
        for metric in original_summary:
            if metric in other_summary:
                original_val = original_summary[metric]
                other_val = other_summary[metric]
                change = other_val - original_val
                percent_change = (change / original_val * 100) if original_val != 0 else float('inf')

                comparison['metric_changes'][metric] = {
                    'original': original_val,
                    'other': other_val,
                    'change': change,
                    'percent_change': percent_change
                }

        # Performance comparison if test data provided
        if test_data is not None:
            x_test, y_test = test_data

            # Evaluate original model
            original_results = self.model.evaluate(x_test, y_test, verbose=0)
            if isinstance(original_results, list):
                original_results = dict(zip(self.model.metrics_names, original_results))

            # Evaluate other model
            other_results = other_model.evaluate(x_test, y_test, verbose=0)
            if isinstance(other_results, list):
                other_results = dict(zip(other_model.metrics_names, other_results))

            comparison['performance_comparison'] = {
                'original': original_results,
                'other': other_results
            }

        return comparison