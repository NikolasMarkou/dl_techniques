"""
Analyze the statistical properties and structural similarity of model weights.

This analyzer provides a quantitative, data-independent assessment of a neural
network's internal state by examining its weight tensors. The core philosophy
is that the statistical distributions of weights within and across layers
can reveal insights into training health, model complexity, and architectural
similarity.

Architecture and Methodology
---------------------------
The analysis is performed in two main stages:

1.  **Per-Layer Statistical Profiling**: The analyzer iterates through each
    layer of a given model. For each weight tensor, it computes a feature
    vector comprising fundamental statistical descriptors. This captures the
    "micro-level" properties of the model's learned parameters.

2.  **Model-Level Comparison via PCA**: To compare different models, the
    per-layer feature vectors of each model are concatenated into a single,
    high-dimensional vector. This vector serves as a holistic statistical
    fingerprint for the entire model. Principal Component Analysis (PCA) is
    then applied to the collection of these model fingerprints. By projecting
    these high-dimensional vectors onto the first two principal components,
    we can visualize the models in a 2D "model space." Models that cluster
    together in this space have learned statistically similar weight
    distributions, suggesting they have converged to similar solutions or
    possess similar architectural properties.

Foundational Mathematics
------------------------
The analysis is grounded in fundamental statistical and linear algebra
concepts applied to the weight tensors of a neural network:

-   **Statistical Moments**: The analysis calculates the first four central
    moments of each layer's weight distribution: mean, standard deviation
    (variance), skewness, and kurtosis. These metrics diagnose the "health"
    of the learned weights. A near-zero mean and moderate standard deviation
    are often desirable. High skewness can indicate neuron saturation or
    dying ReLU issues, while high kurtosis points to the presence of extreme
    outlier weights, which can affect model stability.

-   **Matrix and Vector Norms**: L1, L2, and spectral norms are computed to
    quantify the overall magnitude of the weight tensors. These norms serve
    as proxies for model complexity. The spectral norm (the largest singular
    value of the weight matrix) is particularly significant as it bounds the
    Lipschitz constant of the layer, which relates directly to the model's
    robustness to adversarial perturbations and its generalization
    capabilities.

-   **Principal Component Analysis (PCA)**: This linear dimensionality
    reduction technique is used to find the principal axes of variation
    within the "model space." Each model's statistical fingerprint is treated
    as a point in a high-dimensional space. PCA identifies the directions
    (principal components) that capture the most variance among these points.
    Visualizing models along the top two components provides an intuitive map
    of their structural similarities.

References
----------
1.  Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
    MIT Press. (Provides background on weight initialization and norms).
2.  Neyshabur, B., Tomioka, R., & Srebro, N. (2015). "Norm-Based Capacity
    Control in Neural Networks." COLT.
3.  Li, H., Xu, Z., Taylor, G., & Goldstein, T. (2018). "Visualizing the
    Loss Landscape of Neural Nets." NeurIPS. (While focused on loss, it
    popularized the idea of using PCA to understand high-dimensional
    spaces in deep learning).

"""

import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput

# ---------------------------------------------------------------------

class WeightAnalyzer(BaseAnalyzer):
    """Analyzes weight distributions and statistics."""

    def requires_data(self) -> bool:
        """Weight analysis doesn't require input data."""
        return False

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze weight distributions with improved visualizations."""
        logger.info("Analyzing weight distributions...")

        # Initialize layer order tracking for robust visualization
        if not hasattr(results, 'weight_stats_layer_order'):
            results.weight_stats_layer_order = {}

        for model_name, model in self.models.items():
            results.weight_stats[model_name] = {}
            layer_names_in_order = []  # Track actual layer order

            for layer in model.layers:
                if (self.config.weight_layer_types and
                    layer.__class__.__name__ not in self.config.weight_layer_types):
                    continue

                weights = layer.get_weights()
                if not weights:
                    continue

                for idx, w in enumerate(weights):
                    if len(w.shape) < 2 and not self.config.analyze_biases:
                        continue

                    weight_name = f"{layer.name}_w{idx}"
                    stats = self._compute_weight_statistics(w)
                    results.weight_stats[model_name][weight_name] = stats

                    # Track layer order for robust visualization
                    if weight_name not in layer_names_in_order:
                        layer_names_in_order.append(weight_name)

            # Store explicit layer order for visualizers to use
            results.weight_stats_layer_order[model_name] = layer_names_in_order

        # Compute PCA if requested
        if self.config.compute_weight_pca:
            self._compute_weight_pca(results)

    def _compute_weight_statistics(self, weights: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive statistics for a weight tensor."""
        flat_weights = weights.flatten()

        stats = {
            'shape': weights.shape,
            'basic': {
                'mean': float(np.mean(flat_weights)),
                'std': float(np.std(flat_weights)),
                'median': float(np.median(flat_weights)),
                'min': float(np.min(flat_weights)),
                'max': float(np.max(flat_weights)),
                'skewness': float(scipy.stats.skew(flat_weights)),
                'kurtosis': float(scipy.stats.kurtosis(flat_weights)),
            },
            'norms': {
                'l1': float(np.sum(np.abs(weights))),
                'l2': float(np.sqrt(np.sum(weights ** 2))),
                'max': float(np.max(np.abs(weights))),
                'rms': float(np.sqrt(np.mean(weights ** 2))),
            },
            'distribution': {
                'zero_fraction': float(np.mean(np.abs(flat_weights) < 1e-6)),
                'positive_fraction': float(np.mean(flat_weights > 0)),
                'negative_fraction': float(np.mean(flat_weights < 0)),
            }
        }

        if len(weights.shape) == 2:
            try:
                stats['norms']['spectral'] = float(np.linalg.norm(weights, 2))
            except np.linalg.LinAlgError:
                stats['norms']['spectral'] = 0.0

        return stats

    def _compute_weight_pca(self, results: AnalysisResults) -> None:
        """Perform PCA analysis on concatenated weight statistics from all layers.

        Note: This analysis only works for models with the same number of analyzed layers,
        as it creates fixed-length feature vectors by concatenating statistics from each layer.
        Models with different architectures will be skipped.
        """
        model_features = []
        labels = []

        for model_name, weight_stats in results.weight_stats.items():
            if not weight_stats:
                continue

            # Extract statistical features from all layers using explicit ordering
            features = []

            # Use explicit layer order from results instead of relying on dict order
            if hasattr(results, 'weight_stats_layer_order') and model_name in results.weight_stats_layer_order:
                layer_order = results.weight_stats_layer_order[model_name]
            else:
                # Fallback to model layer order if available
                model = self.models[model_name]
                layer_order = []
                for layer in model.layers:
                    for idx in range(len(layer.get_weights())):
                        weight_name = f"{layer.name}_w{idx}"
                        if weight_name in weight_stats:
                            layer_order.append(weight_name)

            for layer_name in layer_order:
                if layer_name not in weight_stats:
                    continue

                stats = weight_stats[layer_name]

                # Create a fixed-size feature vector from statistics
                layer_features = [
                    stats['basic']['mean'],
                    stats['basic']['std'],
                    stats['basic']['median'],
                    stats['basic']['skewness'],
                    stats['basic']['kurtosis'],
                    stats['norms']['l1'],
                    stats['norms']['l2'],
                    stats['norms']['rms'],
                    stats['distribution']['zero_fraction'],
                    stats['distribution']['positive_fraction'],
                    stats['distribution']['negative_fraction']
                ]

                # Add spectral norm if available
                if 'spectral' in stats['norms']:
                    layer_features.append(stats['norms']['spectral'])
                else:
                    layer_features.append(0.0)

                features.extend(layer_features)

            # Store features if available
            if features:
                model_features.append(features)
                labels.append(model_name)

        if len(model_features) >= 2:
            # Check if all models have the same number of features before proceeding
            first_len = len(model_features[0])
            if not all(len(f) == first_len for f in model_features):
                logger.warning(
                    "Skipping weight PCA: Models have different architectures (different numbers of analyzed layers), "
                    "making direct comparison via concatenated feature vectors invalid."
                )
                return  # Exit the method

            try:
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(model_features)

                # Perform PCA
                pca = PCA(n_components=min(3, len(features_scaled)))
                pca_result = pca.fit_transform(features_scaled)

                results.weight_pca = {
                    'components': pca_result,
                    'explained_variance': pca.explained_variance_ratio_,
                    'labels': labels,
                    'feature_type': 'concatenated_weight_statistics'
                }

                logger.info(f"PCA performed on weight statistics: {len(model_features[0])} features per model")

            except np.linalg.LinAlgError as e:
                logger.warning(f"Could not perform PCA on weight statistics: {e}")