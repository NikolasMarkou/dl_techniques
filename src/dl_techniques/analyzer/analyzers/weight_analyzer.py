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
            except:
                stats['norms']['spectral'] = 0.0

        return stats

    def _compute_weight_pca(self, results: AnalysisResults) -> None:
        """Perform PCA analysis specifically on final layer weights."""
        final_layer_features = []
        labels = []

        for model_name, model in self.models.items():
            # Find the final dense layer
            final_dense = None
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Dense):
                    final_dense = layer
                    break

            if final_dense:
                weights_list = final_dense.get_weights()
                if weights_list and len(weights_list) > 0:
                    weights = weights_list[0]  # Get kernel weights

                    # Validate weight shape
                    if len(weights.shape) >= 2:
                        # Flatten and take a subset if too large
                        flat_weights = weights.flatten()
                        if len(flat_weights) > 1000:
                            flat_weights = flat_weights[::len(flat_weights) // 1000]

                        # Check for finite values
                        if np.all(np.isfinite(flat_weights)):
                            final_layer_features.append(flat_weights)
                            labels.append(model_name)
                        else:
                            logger.warning(f"Skipping {model_name} in PCA due to non-finite weights")
                    else:
                        logger.warning(f"Skipping {model_name} in PCA due to invalid weight shape")

        if len(final_layer_features) >= 2:
            # Ensure all features have same length
            min_len = min(len(f) for f in final_layer_features)
            aligned_features = [f[:min_len] for f in final_layer_features]

            try:
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(aligned_features)

                # Perform PCA
                pca = PCA(n_components=min(3, len(features_scaled)))
                pca_result = pca.fit_transform(features_scaled)

                results.weight_pca = {
                    'components': pca_result,
                    'explained_variance': pca.explained_variance_ratio_,
                    'labels': labels
                }
            except Exception as e:
                logger.warning(f"Could not perform PCA on final layer weights: {e}")