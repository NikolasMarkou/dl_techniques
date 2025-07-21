"""
Weight Analysis Module
============================================================================

Analyzes weight distributions, statistics, and health metrics.
"""

import numpy as np
import scipy.stats
import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput
from dl_techniques.utils.logger import logger


class WeightAnalyzer(BaseAnalyzer):
    """Analyzes weight distributions and statistics."""

    def requires_data(self) -> bool:
        """Weight analysis doesn't require input data."""
        return False

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Analyze weight distributions with improved visualizations."""
        logger.info("Analyzing weight distributions...")

        for model_name, model in self.models.items():
            results.weight_stats[model_name] = {}

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

            # Extract statistical features from all layers
            features = []

            # Sort layers by their appearance order in the model
            model = self.models[model_name]
            layer_order = {layer.name: i for i, layer in enumerate(model.layers)}
            sorted_stats = sorted(weight_stats.items(),
                                key=lambda x: layer_order.get(x[0].split('_w')[0], float('inf')))

            for layer_name, stats in sorted_stats:
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