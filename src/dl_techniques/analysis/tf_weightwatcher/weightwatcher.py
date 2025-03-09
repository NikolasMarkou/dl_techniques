"""
TensorFlow WeightWatcher - A diagnostic tool for analyzing neural network weight matrices

This module provides tools to analyze weight matrices in TensorFlow/Keras models,
with a focus on power-law analysis and eigenvalue spectrum.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import warnings

# Import constants
from .constants import (
    DEFAULT_MIN_EVALS, DEFAULT_MAX_EVALS, DEFAULT_MAX_N, DEFAULT_SAVEDIR,
    WEAK_RANK_LOSS_TOLERANCE, DEFAULT_SUMMARY_METRICS,
    LayerType, MetricNames
)

# Import metrics functions
from .metrics import (
    compute_eigenvalues, calculate_matrix_entropy, fit_powerlaw,
    calculate_spectral_metrics, jensen_shannon_distance
)

# Import utility functions
from .weights_utils import (
    infer_layer_type, get_layer_weights_and_bias,
    get_weight_matrices, plot_powerlaw_fit,
    calculate_glorot_normalization_factor
)

from dl_techniques.utils.logger import logger

# Suppress scipy/numpy warnings in eigenvalue calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)


class WeightWatcher:
    """
    TensorFlow WeightWatcher - A tool for analyzing TensorFlow/Keras neural network weight matrices
    using random matrix theory and power-law analysis.
    """

    def __init__(self, model: Optional[keras.Model] = None):
        """
        Initialize the TFWeightWatcher with a model.

        Args:
            model: A Keras model to analyze. Can be set later.
        """
        self.model = model
        self.details = None
        self.results = None

        logger.info(f"TensorFlow version: {tf.__version__}")
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
                 layers: List[int] = [],
                 min_evals: int = DEFAULT_MIN_EVALS,
                 max_evals: int = DEFAULT_MAX_EVALS,
                 max_N: int = DEFAULT_MAX_N) -> pd.DataFrame:
        """
        Describe the model architecture and layer dimensions without full analysis.

        Args:
            model: Keras model to analyze. Uses the model set in constructor if None.
            layers: List of layer indices to analyze. If empty, analyze all layers.
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

        details = pd.DataFrame(columns=[])

        for layer_id, layer in enumerate(self.model.layers):
            # Skip if layer is filtered out
            if layers and layer_id not in layers:
                continue

            layer_name = layer.name
            layer_type = infer_layer_type(layer)  # Using utility function

            # Get weights and bias
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)  # Using utility function

            if not has_weights or layer_type == LayerType.UNKNOWN:
                continue

            # Extract weight matrices
            Wmats, N, M, rf = get_weight_matrices(weights, layer_type)  # Using utility function

            # Check matrix dimensions against thresholds
            if M < min_evals or M > max_evals or N > max_N:
                continue

            # Create row data
            row_data = {
                'layer_id': layer_id,
                'name': layer_name,
                'layer_type': layer_type,
                'N': N,
                'M': M,
                'rf': rf,
                'Q': N / M if M > 0 else -1,
                'num_params': np.prod(weights.shape),
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
                layers: List[int] = [],
                min_evals: int = DEFAULT_MIN_EVALS,
                max_evals: int = DEFAULT_MAX_EVALS,
                max_N: int = DEFAULT_MAX_N,
                glorot_fix: bool = False,
                plot: bool = False,
                randomize: bool = False,
                savefig: Union[bool, str] = DEFAULT_SAVEDIR,
                tolerance: float = WEAK_RANK_LOSS_TOLERANCE) -> pd.DataFrame:
        """
        Analyze weight matrices of the model using power-law and spectral methods.

        Args:
            model: Keras model to analyze. Uses the model set in constructor if None.
            layers: List of layer indices to analyze. If empty, analyze all layers.
            min_evals: Minimum number of eigenvalues required for analysis.
            max_evals: Maximum number of eigenvalues to analyze.
            max_N: Maximum matrix dimension to analyze.
            glorot_fix: Whether to apply Glorot normalization.
            plot: Whether to create plots of the eigenvalue distributions.
            randomize: Whether to analyze randomized weight matrices.
            savefig: Directory to save figures or False to disable saving.
            tolerance: Tolerance for eigenvalue calculations.

        Returns:
            DataFrame with layer analysis results.
        """
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Set model in constructor or pass as argument.")

        # Setup save directory if needed
        savedir = DEFAULT_SAVEDIR
        if isinstance(savefig, str):
            savedir = savefig
        elif not savefig:
            savedir = None

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
            layer_type = infer_layer_type(layer)  # Using utility function

            # Get weights and bias
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)  # Using utility function

            # Skip if no weights
            if not has_weights:
                continue

            # Extract weight matrices and dimensions
            Wmats, N, M, rf = get_weight_matrices(weights, layer_type)  # Using utility function

            # Apply Glorot normalization if requested
            if glorot_fix:
                # Apply Glorot normalization: std = sqrt(2 / (fan_in + fan_out))
                kappa = calculate_glorot_normalization_factor(N, M, rf)  # Using utility function
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
            alpha, xmin, D, sigma, num_pl_spikes, status, warning = fit_powerlaw(
                evals,
                xmin=None
            )

            # Plot if requested
            if plot and status == 'success':
                plot_powerlaw_fit(  # Using utility function
                    evals,
                    alpha,
                    xmin,
                    D,
                    sigma,
                    f"{layer_id}: {layer_name}",
                    layer_id,
                    savedir if savefig else DEFAULT_SAVEDIR
                )

            # Calculate matrix entropy
            entropy = calculate_matrix_entropy(np.sqrt(evals), N)

            # Calculate spectral metrics
            spectral_metrics = calculate_spectral_metrics(evals, alpha)

            # Analyze randomized matrices if requested
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
                ww_softrank = np.max(rand_evals) / np.max(evals)

                # Add randomization metrics to details
                details.at[layer_id, 'rand_sv_max'] = rand_sv_max
                details.at[layer_id, 'rand_distance'] = rand_distance
                details.at[layer_id, 'ww_softrank'] = ww_softrank

            # Add all metrics to details
            metrics = {
                MetricNames.HAS_ESD: True,
                MetricNames.NUM_EVALS: len(evals),
                MetricNames.SV_MAX: sv_max,
                MetricNames.SV_MIN: sv_min,
                MetricNames.RANK_LOSS: rank_loss,
                MetricNames.WEAK_RANK_LOSS: weak_rank_loss,
                MetricNames.LAMBDA_MAX: np.max(evals),
                MetricNames.ALPHA: alpha,
                MetricNames.XMIN: xmin,
                MetricNames.D: D,
                MetricNames.SIGMA: sigma,
                MetricNames.NUM_PL_SPIKES: num_pl_spikes,
                MetricNames.STATUS: status,
                MetricNames.WARNING: warning,
                MetricNames.ENTROPY: entropy,
                **spectral_metrics  # Unpack all spectral metrics
            }

            # Update details with metrics
            for key, value in metrics.items():
                details.at[layer_id, key] = value

        self.details = details
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

        # Calculate mean for each metric
        for metric in DEFAULT_SUMMARY_METRICS:
            if metric in details.columns:
                # Skip non-numeric or failed values
                valid_values = details[metric][details[metric].apply(
                    lambda x: isinstance(x, (int, float)) and x > 0
                )]

                if not valid_values.empty:
                    summary[metric] = valid_values.mean()

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
        layer_type = infer_layer_type(layer)  # Using utility function

        # Get weights
        has_weights, weights, _, _ = get_layer_weights_and_bias(layer)  # Using utility function

        if not has_weights:
            logger.warning(f"Layer {layer_id} ({layer.name}) has no weights.")
            return np.array([])

        # Get weight matrices and calculate eigenvalues
        Wmats, N, M, rf = get_weight_matrices(weights, layer_type)  # Using utility function
        n_comp = M * rf

        evals, _, _, _ = compute_eigenvalues(Wmats, N, M, n_comp)

        return evals