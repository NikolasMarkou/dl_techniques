"""
TensorFlow WeightWatcher - A diagnostic tool for analyzing neural network weight matrices

This module provides comprehensive analysis of weight matrices in Keras models,
including power-law analysis, eigenvalue spectrum analysis, and concentration metrics.
"""

import os
import keras
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .constants import (
    DEFAULT_MIN_EVALS, DEFAULT_MAX_EVALS, DEFAULT_MAX_N, DEFAULT_SAVEDIR,
    WEAK_RANK_LOSS_TOLERANCE, DEFAULT_SUMMARY_METRICS, HIGH_CONCENTRATION_PERCENTILE,
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

# ---------------------------------------------------------------------
# Suppress scipy/numpy warnings in eigenvalue calculations
# ---------------------------------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------

class WeightWatcher:
    """
    Enhanced WeightWatcher for comprehensive analysis of Keras neural network weight matrices.

    This class performs spectral analysis including power-law fitting, entropy calculation,
    and concentration metrics to assess training quality and model complexity. It provides
    tools for analyzing individual layers, generating summary statistics, and creating
    smoothed model variants through SVD-based techniques.

    :ivar model: The Keras model being analyzed
    :type model: Optional[keras.Model]
    :ivar details: DataFrame containing detailed analysis results for each layer
    :type details: Optional[pd.DataFrame]
    :ivar results: Legacy results storage (maintained for compatibility)
    :type results: Optional[Any]
    """

    def __init__(self, model: Optional[keras.Model] = None):
        """
        Initialize the WeightWatcher with a Keras model.

        Sets up the analyzer with an optional model. The model can be provided later
        using the set_model method. Logs the current Keras version for debugging purposes.

        :param model: A Keras model to analyze. Can be set later using set_model()
        :type model: Optional[keras.Model]
        """
        self.model = model
        self.details = None  # Will store detailed analysis results
        self.results = None  # Legacy compatibility field

        logger.info(f"Keras version: {keras.__version__}")

    def set_model(self, model: keras.Model) -> None:
        """
        Set or update the model to analyze.

        This method allows changing the model after initialization, useful for
        analyzing multiple models with the same WeightWatcher instance.

        :param model: A Keras model to analyze
        :type model: keras.Model
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

        This method provides a lightweight overview of the model structure, identifying
        analyzable layers and their basic properties such as dimensions, parameter counts,
        and matrix shapes. Useful for understanding model structure before full analysis.

        :param model: Keras model to analyze. Uses the model set in constructor if None
        :type model: Optional[keras.Model]
        :param layers: List of layer indices to analyze. If None, analyze all layers
        :type layers: Optional[List[int]]
        :param min_evals: Minimum number of eigenvalues required for a layer to be analyzable
        :type min_evals: int
        :param max_evals: Maximum number of eigenvalues to analyze per layer
        :type max_evals: int
        :param max_N: Maximum matrix dimension to analyze (filters very large layers)
        :type max_N: int
        :return: DataFrame with layer information including dimensions and basic properties
        :rtype: pd.DataFrame

        .. note::
            This method only extracts basic layer information without performing
            computationally expensive eigenvalue decompositions or spectral analysis.
        """
        # Use provided model or fall back to instance model
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Set model in constructor or pass as argument.")

        # Default to empty list if no layers specified (will analyze all valid layers)
        if layers is None:
            layers = []

        details = pd.DataFrame()

        # Iterate through all layers in the model
        for layer_id, layer in enumerate(self.model.layers):
            # Skip layers not in the specified filter list (if provided)
            if layers and layer_id not in layers:
                continue

            # Extract basic layer information
            layer_name = layer.name
            layer_type = infer_layer_type(layer)

            # Get weights and bias information
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

            # Skip layers without weights or unknown types
            if not has_weights or layer_type == LayerType.UNKNOWN:
                continue

            # Extract weight matrices and compute dimensions
            Wmats, N, M, rf = get_weight_matrices(weights, layer_type)

            # Apply dimension filters to exclude inappropriate layers
            if M < min_evals or M > max_evals or N > max_N:
                continue

            # Create row data with layer characteristics
            row_data = {
                'layer_id': layer_id,
                'name': layer_name,
                'layer_type': layer_type.value,  # Convert enum to string for storage
                'N': N,  # Input dimension
                'M': M,  # Output dimension
                'rf': rf,  # Repetition factor (for conv layers)
                'Q': N / M if M > 0 else -1,  # Aspect ratio
                'num_params': int(np.prod(weights.shape)),  # Total parameters
                MetricNames.NUM_EVALS: M * rf  # Total eigenvalues to analyze
            }

            # Add row to the details DataFrame
            details = pd.concat([details, pd.DataFrame([row_data])], ignore_index=True)

        # Set layer_id as the index for easier access
        if not details.empty:
            details.set_index('layer_id', inplace=True)

        # Store details for later use
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
        Perform comprehensive spectral analysis of the model's weight matrices.

        This is the core analysis method that computes eigenvalue distributions, fits
        power-law models, calculates entropy measures, and optionally performs concentration
        analysis. The results provide insights into training quality, model complexity,
        and potential optimization opportunities.

        :param model: Keras model to analyze. Uses the model set in constructor if None
        :type model: Optional[keras.Model]
        :param layers: List of layer indices to analyze. If None, analyze all suitable layers
        :type layers: Optional[List[int]]
        :param min_evals: Minimum number of eigenvalues required for analysis
        :type min_evals: int
        :param max_evals: Maximum number of eigenvalues to analyze
        :type max_evals: int
        :param max_N: Maximum matrix dimension to analyze
        :type max_N: int
        :param glorot_fix: Whether to apply Glorot normalization to weight matrices
        :type glorot_fix: bool
        :param plot: Whether to create plots of the eigenvalue distributions
        :type plot: bool
        :param randomize: Whether to analyze randomized weight matrices for comparison
        :type randomize: bool
        :param concentration_analysis: Whether to perform information concentration analysis
        :type concentration_analysis: bool
        :param savefig: Directory to save figures or False to disable saving
        :type savefig: Union[bool, str]
        :param tolerance: Tolerance for eigenvalue calculations and weak rank loss detection
        :type tolerance: float
        :return: DataFrame with comprehensive layer analysis results
        :rtype: pd.DataFrame

        .. note::
            The analysis includes:

            - Eigenvalue spectrum computation
            - Power-law fitting with Heavy-Tailed Random Matrix Theory
            - Matrix entropy calculation
            - Optional concentration metrics (Gini coefficient, participation ratio, etc.)
            - Optional randomization analysis for comparison
        """
        # Use provided model or fall back to instance model
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Set model in constructor or pass as argument.")

        # Default to empty list if no layers specified
        if layers is None:
            layers = []

        # Setup save directory for plots if needed
        savedir = DEFAULT_SAVEDIR
        if isinstance(savefig, str):
            savedir = savefig
        elif not savefig:
            savedir = None

        if savedir and not os.path.exists(savedir):
            os.makedirs(savedir, exist_ok=True)

        # Create basic description of the model to establish analyzable layers
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

        # Perform detailed analysis on each qualifying layer
        for layer_id in details.index:
            layer = self.model.layers[layer_id]
            layer_name = layer.name
            layer_type = infer_layer_type(layer)

            logger.info(f"Analyzing layer {layer_id}: {layer_name}")

            # Extract weight and bias information
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

            # Skip layers without weights (shouldn't happen due to describe() filtering)
            if not has_weights:
                continue

            # Extract weight matrices and compute dimensions
            Wmats, N, M, rf = get_weight_matrices(weights, layer_type)

            # Apply Glorot normalization if requested
            # This can help normalize analysis across different layer sizes
            if glorot_fix:
                kappa = calculate_glorot_normalization_factor(N, M, rf)
                Wmats = [W / kappa for W in Wmats]

            # Calculate number of eigenvalue components to analyze
            n_comp = M * rf

            # Compute eigenvalue spectrum - core of the spectral analysis
            evals, sv_max, sv_min, rank_loss = compute_eigenvalues(
                Wmats, N, M, n_comp, normalize=False
            )

            # Calculate weak rank loss (eigenvalues below tolerance threshold)
            # This indicates potential numerical instabilities or over-parameterization
            weak_rank_loss = np.sum(evals < tolerance)

            # Fit power law to eigenvalue distribution using Heavy-Tailed RMT
            alpha, xmin, D, sigma, num_pl_spikes, status, warning = fit_powerlaw(evals)

            # Create eigenvalue distribution plots if requested
            if plot and status == 'success' and savedir:
                plot_powerlaw_fit(
                    evals, alpha, xmin, D, sigma,
                    f"{layer_id}: {layer_name}", layer_id, savedir
                )

            # Calculate matrix entropy as a measure of information content
            entropy = calculate_matrix_entropy(np.sqrt(evals), N)

            # Calculate additional spectral metrics (spectral norm, condition number, etc.)
            spectral_metrics = calculate_spectral_metrics(evals, alpha)

            # Initialize concentration metrics dictionary
            concentration_metrics = {}

            # Perform concentration analysis if requested
            if concentration_analysis:
                # Use the first (typically largest) weight matrix for concentration analysis
                if len(Wmats) > 0:
                    concentration_metrics = calculate_concentration_metrics(Wmats[0])

            # Initialize randomization metrics dictionary
            randomization_metrics = {}

            # Perform randomization analysis if requested
            if randomize:
                # Generate randomized matrices with same dimensions but shuffled weights
                rand_Wmats = []
                for W in Wmats:
                    W_flat = W.flatten()
                    np.random.shuffle(W_flat)  # Randomize weight values
                    rand_Wmats.append(W_flat.reshape(W.shape))

                # Compute eigenvalues for randomized matrices
                rand_evals, rand_sv_max, rand_sv_min, rand_rank_loss = compute_eigenvalues(
                    rand_Wmats, N, M, n_comp, normalize=False
                )

                # Calculate Jensen-Shannon distance between original and randomized eigenvalues
                rand_distance = jensen_shannon_distance(evals, rand_evals)

                # Calculate soft rank measure comparing maximum eigenvalues
                ww_softrank = np.max(rand_evals) / np.max(evals) if np.max(evals) > 0 else 0

                randomization_metrics = {
                    'rand_sv_max': rand_sv_max,
                    'rand_distance': rand_distance,
                    'ww_softrank': ww_softrank
                }

            # Combine all computed metrics into a comprehensive dictionary
            metrics = {
                MetricNames.HAS_ESD: True,  # Eigenvalue Spectral Density is available
                MetricNames.NUM_EVALS: len(evals),
                MetricNames.SV_MAX: sv_max,  # Maximum singular value
                MetricNames.SV_MIN: sv_min,  # Minimum singular value
                MetricNames.RANK_LOSS: rank_loss,  # Numerical rank loss
                MetricNames.WEAK_RANK_LOSS: weak_rank_loss,  # Eigenvalues below tolerance
                MetricNames.LAMBDA_MAX: np.max(evals) if len(evals) > 0 else 0,
                MetricNames.ALPHA: alpha,  # Power-law exponent
                MetricNames.XMIN: xmin,  # Power-law minimum value
                MetricNames.D: D,  # Kolmogorov-Smirnov distance
                MetricNames.SIGMA: sigma,  # Power-law uncertainty
                MetricNames.NUM_PL_SPIKES: num_pl_spikes,  # Number of power-law spikes
                MetricNames.STATUS: status,  # Fitting status
                MetricNames.WARNING: warning,  # Any warnings from fitting
                MetricNames.ENTROPY: entropy,  # Matrix entropy
                **spectral_metrics,
                **randomization_metrics
            }

            # Add concentration metrics with special handling for critical_weights
            for key, value in concentration_metrics.items():
                if key == 'critical_weights':
                    # Store critical weights count instead of the full list for DataFrame compatibility
                    metrics['critical_weight_count'] = len(value) if value else 0
                    # Store the critical weights separately for later access
                    if not hasattr(self, '_critical_weights'):
                        self._critical_weights = {}
                    self._critical_weights[layer_id] = value
                else:
                    metrics[key] = value

            # Update details DataFrame with all computed metrics
            for key, value in metrics.items():
                details.at[layer_id, key] = value

        # Store updated details for later access
        self.details = details

        # Log summary of high concentration layers for user awareness
        if concentration_analysis and 'concentration_score' in details.columns:
            high_conc_threshold = details['concentration_score'].quantile(0.8)
            high_conc_layers = details[details['concentration_score'] > high_conc_threshold]
            if not high_conc_layers.empty:
                logger.info(f"Layers with high concentration scores: {list(high_conc_layers.index)}")

        return details

    def get_summary(self, details: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Get summary metrics averaged across all analyzed layers.

        Computes aggregate statistics across all analyzed layers to provide
        model-wide insights. The summary includes means of key spectral metrics,
        concentration measures, and layer counts.

        :param details: DataFrame with layer analysis results. Uses self.details if None
        :type details: Optional[pd.DataFrame]
        :return: Dictionary with summary metrics averaged across all layers
        :rtype: Dict[str, float]

        .. note::
            Summary metrics include:

            - Mean alpha (power-law exponent)
            - Mean entropy and spectral metrics
            - Concentration scores and participation ratios
            - Total layers analyzed and high concentration layer count
        """
        summary = {}

        # Use provided details or fall back to instance details
        if details is None:
            details = self.details

        if details is None or details.empty:
            logger.warning("No analysis details available. Run analyze() first.")
            return summary

        # Start with standard metrics defined in constants
        metrics_to_summarize = DEFAULT_SUMMARY_METRICS.copy()

        # Add concentration metrics if they were computed during analysis
        concentration_metrics = ['gini_coefficient', 'dominance_ratio', 'participation_ratio', 'concentration_score']
        for metric in concentration_metrics:
            if metric in details.columns:
                metrics_to_summarize.append(metric)

        # Calculate mean for each available metric
        for metric in metrics_to_summarize:
            if metric in details.columns:
                # Filter out non-numeric, NaN, or failed analysis values
                valid_values = details[metric][details[metric].apply(
                    lambda x: isinstance(x, (int, float)) and not np.isnan(x) and x > -1
                )]

                if not valid_values.empty:
                    summary[metric] = float(valid_values.mean())

        # Add count-based summary statistics
        summary['total_layers_analyzed'] = len(details)

        # Count high concentration layers using the defined percentile threshold
        if 'concentration_score' in details.columns:
            high_conc_threshold = details['concentration_score'].quantile(HIGH_CONCENTRATION_PERCENTILE)
            summary['high_concentration_layers'] = int(sum(details['concentration_score'] > high_conc_threshold))

        return summary

    def get_ESD(self, layer_id: int) -> np.ndarray:
        """
        Get the Eigenvalue Spectral Distribution (ESD) for a specific layer.

        Computes and returns the eigenvalue spectrum for a single layer's weight matrix,
        useful for detailed examination of individual layer properties or for custom analysis.

        :param layer_id: ID of the layer to analyze (index in model.layers)
        :type layer_id: int
        :return: Array of eigenvalues for the specified layer
        :rtype: np.ndarray
        :raises ValueError: If model is not set or layer_id is out of range

        .. note::
            This method performs eigenvalue decomposition on-demand and does not
            require prior analysis of the entire model.
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        # Validate layer ID is within valid range
        if layer_id < 0 or layer_id >= len(self.model.layers):
            raise ValueError(f"Layer ID {layer_id} out of range.")

        # Get the specified layer and determine its type
        layer = self.model.layers[layer_id]
        layer_type = infer_layer_type(layer)

        # Extract weights from the layer
        has_weights, weights, _, _ = get_layer_weights_and_bias(layer)

        if not has_weights:
            logger.warning(f"Layer {layer_id} ({layer.name}) has no weights.")
            return np.array([])

        # Get weight matrices and calculate eigenvalues
        Wmats, N, M, rf = get_weight_matrices(weights, layer_type)
        n_comp = M * rf

        # Compute eigenvalue spectrum
        evals, _, _, _ = compute_eigenvalues(Wmats, N, M, n_comp)

        return evals

    def get_layer_concentration_metrics(self, layer_id: int) -> Dict[str, Any]:
        """
        Get detailed concentration metrics for a specific layer.

        Computes comprehensive concentration analysis for a single layer, including
        Gini coefficients, participation ratios, dominance measures, and identification
        of critical weight components.

        :param layer_id: ID of the layer to analyze (index in model.layers)
        :type layer_id: int
        :return: Dictionary with detailed concentration metrics
        :rtype: Dict[str, Any]
        :raises ValueError: If model is not set or layer_id is out of range

        .. note::
            Concentration metrics help identify layers where information is highly
            concentrated in a few parameters, which can indicate brittleness or
            importance for model performance.
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        # Validate layer ID is within valid range
        if layer_id < 0 or layer_id >= len(self.model.layers):
            raise ValueError(f"Layer ID {layer_id} out of range.")

        # Get the specified layer and determine its type
        layer = self.model.layers[layer_id]
        layer_type = infer_layer_type(layer)

        # Extract weights from the layer
        has_weights, weights, _, _ = get_layer_weights_and_bias(layer)

        if not has_weights:
            logger.warning(f"Layer {layer_id} ({layer.name}) has no weights.")
            return {}

        # Get weight matrices and calculate concentration metrics
        Wmats, N, M, rf = get_weight_matrices(weights, layer_type)

        if len(Wmats) == 0:
            return {}

        # Calculate concentration metrics using the primary weight matrix
        metrics = calculate_concentration_metrics(Wmats[0])

        # Add critical weights if they were stored during previous analysis
        if hasattr(self, '_critical_weights') and layer_id in self._critical_weights:
            metrics['critical_weights'] = self._critical_weights[layer_id]

        return metrics

    def create_smoothed_model(self,
                             method: str = 'detX',
                             percent: float = 0.8,
                             save_path: Optional[str] = None) -> keras.Model:
        """
        Create a smoothed version of the model using SVD truncation techniques.

        Applies Singular Value Decomposition (SVD) based smoothing to reduce noise
        in weight matrices, potentially improving generalization. Different methods
        provide different approaches to selecting which singular values to retain.

        :param method: Smoothing method - 'svd', 'detX', or 'lambda_min'
        :type method: str
        :param percent: Percentage of singular values to keep when using 'svd' method
        :type percent: float
        :param save_path: Optional path to save the smoothed model in .keras format
        :type save_path: Optional[str]
        :return: New Keras model with smoothed weights
        :rtype: keras.Model
        :raises ValueError: If model is not set

        .. note::
            Smoothing methods:

            - 'svd': Keep a fixed percentage of singular values
            - 'detX': Use deterministic criterion based on eigenvalue gaps
            - 'lambda_min': Keep components above power-law noise floor
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        # Ensure analysis has been performed to get layer details
        if self.details is None:
            logger.info("Running analysis to get layer details...")
            self.analyze(plot=False)

        # Clone the original model to preserve its structure
        smoothed_model = keras.models.clone_model(self.model)
        smoothed_model.set_weights(self.model.get_weights())

        # Import smoothing functions (avoiding circular imports)
        from .metrics import smooth_matrix, compute_detX_constraint

        # Process each layer that was successfully analyzed
        for layer_id in self.details.index:
            layer = smoothed_model.layers[layer_id]
            layer_type = infer_layer_type(layer)

            # Skip unsupported layer types (only process Dense and Conv layers)
            if layer_type not in [LayerType.DENSE, LayerType.CONV1D, LayerType.CONV2D]:
                continue

            # Extract current weights and bias
            has_weights, old_weights, has_bias, old_bias = get_layer_weights_and_bias(layer)

            if not has_weights:
                continue

            # Determine number of components to keep based on smoothing method
            if method == 'detX':
                # Use deterministic criterion based on eigenvalue spectrum
                evals = self.get_ESD(layer_id)
                num_smooth = compute_detX_constraint(evals)
            elif method == 'lambda_min':
                # Use power-law spikes as the cutoff criterion
                if MetricNames.NUM_PL_SPIKES in self.details.columns:
                    num_smooth = int(self.details.at[layer_id, MetricNames.NUM_PL_SPIKES])
                else:
                    # Fallback to half of available eigenvalues
                    num_smooth = int(0.5 * self.details.at[layer_id, MetricNames.NUM_EVALS])
            else:  # 'svd' method or default
                # Keep a fixed percentage of singular values
                num_smooth = int(percent * self.details.at[layer_id, MetricNames.NUM_EVALS])

            logger.info(f"Layer {layer_id} ({layer.name}): keeping {num_smooth} components")

            # Apply smoothing based on layer architecture
            if layer_type == LayerType.DENSE:
                # For dense layers, directly smooth the weight matrix
                new_weights = smooth_matrix(old_weights, num_smooth)

                # Update layer weights (preserve bias if present)
                if has_bias:
                    layer.set_weights([new_weights, old_bias])
                else:
                    layer.set_weights([new_weights])

            elif layer_type in [LayerType.CONV1D, LayerType.CONV2D]:
                # For convolutional layers, reshape weights before smoothing
                if layer_type == LayerType.CONV2D:
                    kh, kw, in_c, out_c = old_weights.shape
                    # Reshape to 2D matrix: (kernel_height * kernel_width * input_channels, output_channels)
                    W_reshaped = old_weights.reshape(-1, out_c)
                else:  # CONV1D
                    kernel_size, input_dim, output_dim = old_weights.shape
                    # Reshape to 2D matrix: (kernel_size * input_dim, output_dim)
                    W_reshaped = old_weights.reshape(-1, output_dim)

                # Apply SVD-based smoothing to the reshaped matrix
                W_smoothed = smooth_matrix(W_reshaped, num_smooth)

                # Reshape back to original convolutional weight dimensions
                new_weights = W_smoothed.reshape(old_weights.shape)

                # Update layer weights (preserve bias if present)
                if has_bias:
                    layer.set_weights([new_weights, old_bias])
                else:
                    layer.set_weights([new_weights])

        # Save smoothed model if path is provided
        if save_path:
            smoothed_model.save(save_path)
            logger.info(f"Saved smoothed model to {save_path}")

        return smoothed_model

    def compare_models(self,
                      other_model: keras.Model,
                      test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Compare this model with another model using spectral analysis metrics.

        Performs comparative analysis between the current model and another model
        (e.g., smoothed, pruned, or fine-tuned version) to assess the impact of
        modifications on weight structure and optionally on performance.

        :param other_model: Model to compare against the current model
        :type other_model: keras.Model
        :param test_data: Optional tuple of (x_test, y_test) for performance comparison
        :type test_data: Optional[Tuple[np.ndarray, np.ndarray]]
        :return: Dictionary with comprehensive comparison results
        :rtype: Dict[str, Any]
        :raises ValueError: If current model is not set

        .. note::
            The comparison includes:

            - Spectral metric changes (alpha, entropy, concentration)
            - Performance metric changes (if test data provided)
            - Percentage changes for all comparable metrics
        """
        if self.model is None:
            raise ValueError("No model provided. Set model first.")

        # Get summary metrics for the original model
        original_summary = self.get_summary()

        # Analyze the comparison model using a new WeightWatcher instance
        other_watcher = WeightWatcher(other_model)
        other_details = other_watcher.analyze(plot=False)
        other_summary = other_watcher.get_summary()

        # Initialize comparison results structure
        comparison = {
            'original_summary': original_summary,
            'other_summary': other_summary,
            'metric_changes': {}
        }

        # Calculate metric changes for all comparable metrics
        for metric in original_summary:
            if metric in other_summary:
                original_val = original_summary[metric]
                other_val = other_summary[metric]
                change = other_val - original_val

                # Calculate percentage change with division by zero protection
                percent_change = (change / original_val * 100) if original_val != 0 else float('inf')

                comparison['metric_changes'][metric] = {
                    'original': original_val,
                    'other': other_val,
                    'change': change,
                    'percent_change': percent_change
                }

        # Perform performance comparison if test data is provided
        if test_data is not None:
            x_test, y_test = test_data

            # Evaluate original model on test data
            original_results = self.model.evaluate(x_test, y_test, verbose=0)
            # Handle both single metric and multiple metrics cases
            if isinstance(original_results, list):
                original_results = dict(zip(self.model.metrics_names, original_results))

            # Evaluate comparison model on test data
            other_results = other_model.evaluate(x_test, y_test, verbose=0)
            # Handle both single metric and multiple metrics cases
            if isinstance(other_results, list):
                other_results = dict(zip(other_model.metrics_names, other_results))

            # Store performance comparison results
            comparison['performance_comparison'] = {
                'original': original_results,
                'other': other_results
            }

        return comparison

# ---------------------------------------------------------------------
