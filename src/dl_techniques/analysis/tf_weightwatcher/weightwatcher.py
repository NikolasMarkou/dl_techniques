"""
TensorFlow WeightWatcher - A diagnostic tool for analyzing neural network weight matrices

This module provides tools to analyze weight matrices in TensorFlow/Keras models,
with a focus on power-law analysis and eigenvalue spectrum.
"""

import os
import keras
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Import constants
from .constants import (
    EPSILON, EVALS_THRESH, OVER_TRAINED_THRESH, UNDER_TRAINED_THRESH,
    DEFAULT_MIN_EVALS, DEFAULT_MAX_EVALS, DEFAULT_MAX_N, DEFAULT_SAVEDIR,
    WEAK_RANK_LOSS_TOLERANCE, DEFAULT_BINS, DEFAULT_FIG_SIZE, DEFAULT_DPI,
    DEFAULT_SVD_SMOOTH_PERCENT, DEFAULT_SUMMARY_METRICS,
    LayerType, SmoothingMethod, SVDMethod, StatusCode, MetricNames
)

# Import metrics functions
from .metrics import (
    compute_eigenvalues, calculate_matrix_entropy, fit_powerlaw,
    calculate_stable_rank, calculate_spectral_metrics, jensen_shannon_distance,
    compute_detX_constraint, smooth_matrix, calculate_glorot_normalization_factor
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

    def _infer_layer_type(self, layer: keras.layers.Layer) -> LayerType:
        """
        Determine the layer type for a given Keras layer.

        Args:
            layer: Keras layer.

        Returns:
            LayerType: The type of the layer.
        """
        layer_class = layer.__class__.__name__.lower()

        if isinstance(layer, keras.layers.Dense) or 'dense' in layer_class:
            return LayerType.DENSE
        elif isinstance(layer, keras.layers.Conv1D) or 'conv1d' in layer_class:
            return LayerType.CONV1D
        elif isinstance(layer, keras.layers.Conv2D) or 'conv2d' in layer_class:
            return LayerType.CONV2D
        elif isinstance(layer, keras.layers.Conv3D) or 'conv3d' in layer_class:
            return LayerType.CONV3D
        elif isinstance(layer, keras.layers.Embedding) or 'embedding' in layer_class:
            return LayerType.EMBEDDING
        elif isinstance(layer, keras.layers.LSTM) or 'lstm' in layer_class:
            return LayerType.LSTM
        elif isinstance(layer, keras.layers.GRU) or 'gru' in layer_class:
            return LayerType.GRU
        elif isinstance(layer, keras.layers.LayerNormalization) or 'layernorm' in layer_class:
            return LayerType.NORM
        else:
            return LayerType.UNKNOWN

    def _get_layer_weights_and_bias(self, layer: keras.layers.Layer) -> Tuple[bool, np.ndarray, bool, np.ndarray]:
        """
        Extract weights and biases from a Keras layer.

        Args:
            layer: Keras layer.

        Returns:
            Tuple containing:
            - has_weights (bool): Whether layer has weights
            - weights (np.ndarray): Weight matrix if available, else None
            - has_bias (bool): Whether layer has biases
            - bias (np.ndarray): Bias vector if available, else None
        """
        has_weights, has_bias = False, False
        weights, bias = None, None

        layer_type = self._infer_layer_type(layer)

        # Get layer weights
        weights_list = layer.get_weights()

        if len(weights_list) > 0:
            if layer_type in [
                LayerType.DENSE,
                LayerType.CONV1D,
                LayerType.CONV2D,
                LayerType.CONV3D,
                LayerType.EMBEDDING
            ]:
                has_weights = True
                weights = weights_list[0]

                # Check for bias
                if hasattr(layer, 'use_bias') and layer.use_bias and len(weights_list) > 1:
                    has_bias = True
                    bias = weights_list[1]

        return has_weights, weights, has_bias, bias

    def _get_weight_matrices(self, weights: np.ndarray, layer_type: LayerType) -> Tuple[
        List[np.ndarray], int, int, int]:
        """
        Extract weight matrices from a layer's weights.

        Args:
            weights: Layer weights.
            layer_type: Type of layer.

        Returns:
            Tuple containing:
            - List of weight matrices.
            - N: Maximum dimension.
            - M: Minimum dimension.
            - rf: Receptive field size.
        """
        Wmats = []
        N, M, rf = 0, 0, 1.0

        if layer_type in [LayerType.DENSE, LayerType.EMBEDDING]:
            Wmats = [weights]
            N, M = max(weights.shape), min(weights.shape)

        elif layer_type == LayerType.CONV1D:
            # Conv1D weights shape: (kernel_size, input_dim, output_dim)
            Wmats = [weights]

            # For Conv1D, we'll use the flattened input weight dimensions
            kernel_size, input_dim, output_dim = weights.shape
            rf = kernel_size

            # Reshape to 2D matrix for eigenvalue analysis
            weights_reshaped = weights.reshape(-1, output_dim)
            N, M = max(weights_reshaped.shape), min(weights_reshaped.shape)
            Wmats = [weights_reshaped]

        elif layer_type == LayerType.CONV2D:
            # For Conv2D, extract weight matrices for each filter position
            # Conv2D weights shape: (kernel_height, kernel_width, input_channels, output_channels)
            kh, kw, in_c, out_c = weights.shape
            rf = kh * kw

            # Extract individual filter matrices
            for i in range(kh):
                for j in range(kw):
                    W = weights[i, j, :, :]

                    # Make sure larger dimension is first for consistency
                    if W.shape[0] < W.shape[1]:
                        W = W.T

                    Wmats.append(W)

            # Set N and M based on the shapes of the extracted matrices
            if len(Wmats) > 0:
                N, M = max(Wmats[0].shape), min(Wmats[0].shape)

        return Wmats, N, M, rf

    def _plot_powerlaw_fit(self,
                           evals: np.ndarray,
                           alpha: float,
                           xmin: float,
                           D: float,
                           sigma: float,
                           layer_name: str,
                           layer_id: int,
                           savedir: str) -> None:
        """
        Create and save power-law fit plots.

        Args:
            evals: Eigenvalues to plot.
            alpha: Power-law exponent.
            xmin: Minimum x value used for fit.
            D: Kolmogorov-Smirnov statistic.
            sigma: Standard error of alpha.
            layer_name: Name of the layer.
            layer_id: ID of the layer.
            savedir: Directory to save plots.
        """
        # Create save directory if it doesn't exist
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # Log-log plot
        plt.figure(figsize=DEFAULT_FIG_SIZE)

        # Plot histogram with log-log scale
        hist, bin_edges = np.histogram(evals, bins=DEFAULT_BINS)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize histogram to PDF
        hist = hist / (np.sum(hist) * (bin_edges[1] - bin_edges[0]))

        plt.loglog(bin_centers, hist, 'o', markersize=4, label='Eigenvalue distribution')

        # Add power-law fit line
        x_range = np.logspace(np.log10(xmin), np.log10(np.max(evals)), 100)
        y_fit = (alpha - 1) * (xmin ** (alpha - 1)) * x_range ** (-alpha)
        plt.loglog(x_range, y_fit, 'r-', label=f'Power-law fit: α={alpha:.3f}')

        plt.axvline(x=xmin, color='r', linestyle='--', label=f'xmin={xmin:.3f}')

        plt.title(f"Log-Log ESD for {layer_name}\nα={alpha:.3f}, D={D:.3f}, σ={sigma:.3f}")
        plt.xlabel("Eigenvalue (λ)")
        plt.ylabel("Probability density")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{savedir}/layer_{layer_id}_powerlaw.png", dpi=DEFAULT_DPI)
        plt.close()

        # Additional plot: linear scale
        plt.figure(figsize=DEFAULT_FIG_SIZE)
        plt.hist(evals, bins=DEFAULT_BINS, density=True)
        plt.axvline(x=xmin, color='r', linestyle='--', label=f'xmin={xmin:.3f}')
        plt.title(f"Linear ESD for {layer_name}")
        plt.xlabel("Eigenvalue (λ)")
        plt.ylabel("Probability density")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{savedir}/layer_{layer_id}_esd_linear.png", dpi=DEFAULT_DPI)
        plt.close()

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
            layer_type = self._infer_layer_type(layer)

            # Get weights and bias
            has_weights, weights, has_bias, bias = self._get_layer_weights_and_bias(layer)

            if not has_weights or layer_type == LayerType.UNKNOWN:
                continue

            # Extract weight matrices
            Wmats, N, M, rf = self._get_weight_matrices(weights, layer_type)

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
            layer_type = self._infer_layer_type(layer)

            # Get weights and bias
            has_weights, weights, has_bias, bias = self._get_layer_weights_and_bias(layer)

            # Skip if no weights
            if not has_weights:
                continue

            # Extract weight matrices and dimensions
            Wmats, N, M, rf = self._get_weight_matrices(weights, layer_type)

            # Apply Glorot normalization if requested
            if glorot_fix:
                # Apply Glorot normalization: std = sqrt(2 / (fan_in + fan_out))
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
            alpha, xmin, D, sigma, num_pl_spikes, status, warning = fit_powerlaw(
                evals,
                xmin=None
            )

            # Plot if requested
            if plot and status == 'success':
                self._plot_powerlaw_fit(
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
        layer_type = self._infer_layer_type(layer)

        # Get weights
        has_weights, weights, _, _ = self._get_layer_weights_and_bias(layer)

        if not has_weights:
            logger.warning(f"Layer {layer_id} ({layer.name}) has no weights.")
            return np.array([])

        # Get weight matrices and calculate eigenvalues
        Wmats, N, M, rf = self._get_weight_matrices(weights, layer_type)
        n_comp = M * rf

        evals, _, _, _ = compute_eigenvalues(Wmats, N, M, n_comp)

        return evals

