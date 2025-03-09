import keras
import numpy as np

# Import constants
from .constants import (
    DEFAULT_SVD_SMOOTH_PERCENT, DEFAULT_SUMMARY_METRICS,
    LayerType, SmoothingMethod, SVDMethod, StatusCode, MetricNames
)

# Import metrics functions
from .metrics import (
    compute_eigenvalues, calculate_matrix_entropy, fit_powerlaw,
    calculate_stable_rank, calculate_spectral_metrics, jensen_shannon_distance,
    compute_detX_constraint, smooth_matrix, calculate_glorot_normalization_factor
)

from .weightwatcher import WeightWatcher

# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


class SVDSmoothing:
    """
    Applies SVD-based smoothing to neural network weight matrices.
    """

    def __init__(self, model: keras.Model, method: SmoothingMethod = SmoothingMethod.DETX,
                 percent: float = DEFAULT_SVD_SMOOTH_PERCENT):
        """
        Initialize SVD smoothing.

        Args:
            model: Keras model to smooth.
            method: Smoothing method ('svd', 'detX', or 'lambda_min').
            percent: Percentage of eigenvalues to keep when using 'svd' method.
        """
        self.model = model
        self.method = method
        self.percent = percent
        self.watcher = WeightWatcher(model)

    def smooth(self) -> keras.Model:
        """
        Apply SVD smoothing to the model.

        Returns:
            Smoothed Keras model.
        """
        logger.info(f"Applying SVD smoothing using method: {self.method}")

        # First, analyze the model to get eigenvalue information
        details = self.watcher.analyze(plot=False)

        # Iterate through layers
        for layer_id in details.index:
            layer = self.model.layers[layer_id]
            layer_type = self.watcher._infer_layer_type(layer)

            # Skip unsupported layer types
            if layer_type not in [LayerType.DENSE, LayerType.CONV1D, LayerType.CONV2D]:
                continue

            # Get weights
            has_weights, old_weights, has_bias, old_bias = self.watcher._get_layer_weights_and_bias(layer)

            if not has_weights:
                continue

            # Determine number of components to keep
            if self.method == SmoothingMethod.DETX:
                if MetricNames.NUM_PL_SPIKES in details.columns and layer_id in details.index:
                    num_smooth = int(details.at[layer_id, 'detX_num'] if 'detX_num' in details.columns else 0)
                else:
                    # Calculate DetX constraint if not already done
                    evals = self.watcher.get_ESD(layer_id)
                    num_smooth = compute_detX_constraint(evals)
            elif self.method == SmoothingMethod.LAMBDA_MIN:
                if MetricNames.NUM_PL_SPIKES in details.columns and layer_id in details.index:
                    num_smooth = int(details.at[layer_id, MetricNames.NUM_PL_SPIKES])
                else:
                    num_smooth = int(0.5 * details.at[layer_id, MetricNames.NUM_EVALS])
            else:  # 'svd' or default
                num_smooth = int(self.percent * details.at[layer_id, MetricNames.NUM_EVALS])

            logger.info(f"Layer {layer_id} ({layer.name}): keeping {num_smooth} components")

            # Apply smoothing based on layer type
            if layer_type == LayerType.DENSE:
                new_weights = smooth_matrix(old_weights, num_smooth)

                # Update layer weights
                if has_bias:
                    layer.set_weights([new_weights, old_bias])
                else:
                    layer.set_weights([new_weights])

            elif layer_type == LayerType.CONV2D:
                # Handle Conv2D weights
                kh, kw, in_c, out_c = old_weights.shape
                new_weights = np.zeros_like(old_weights)

                # Apply smoothing to each filter position
                for i in range(kh):
                    for j in range(kw):
                        W = old_weights[i, j, :, :]
                        new_weights[i, j, :, :] = smooth_matrix(W, num_smooth)

                # Update layer weights
                if has_bias:
                    layer.set_weights([new_weights, old_bias])
                else:
                    layer.set_weights([new_weights])

            elif layer_type == LayerType.CONV1D:
                # Handle Conv1D weights
                kernel_size, input_dim, output_dim = old_weights.shape
                new_weights = np.zeros_like(old_weights)

                # Reshape, apply smoothing, and reshape back
                W_reshaped = old_weights.reshape(-1, output_dim)
                W_smoothed = smooth_matrix(W_reshaped, num_smooth)
                new_weights = W_smoothed.reshape(kernel_size, input_dim, output_dim)

                # Update layer weights
                if has_bias:
                    layer.set_weights([new_weights, old_bias])
                else:
                    layer.set_weights([new_weights])

        return self.model