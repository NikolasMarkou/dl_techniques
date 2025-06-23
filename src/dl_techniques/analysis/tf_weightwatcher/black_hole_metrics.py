"""
Black Hole Metrics for TensorFlow WeightWatcher

This module extends the TensorFlow WeightWatcher with advanced metrics
to detect "computational black holes" in neural networks - individual weights
that concentrate a disproportionate amount of Fisher Information and act as
critical control points for the entire network.

Based on research showing that large language models can have 94.3% of total
Fisher Information concentrated in just three individual weights.
"""
import os
import numpy as np
import pandas as pd
import keras
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

# Import constants
from .constants import (
    DEFAULT_FIG_SIZE, DEFAULT_DPI, EPSILON, DEFAULT_MIN_EVALS,
    LayerType, MetricNames
)

# Import utility functions
from .weights_utils import (
    infer_layer_type, get_layer_weights_and_bias,
    get_weight_matrices
)

from dl_techniques.utils.logger import logger


class BlackHoleMetrics:
    """
    Extension to TensorFlow WeightWatcher for detecting computational black holes
    in neural networks - parameters that concentrate Fisher Information and act as
    critical control points.
    """

    def __init__(self, model: keras.Model, base_watcher: Any = None):
        """
        Initialize the BlackHoleMetrics analyzer.

        Args:
            model: A Keras model to analyze.
            base_watcher: Optional WeightWatcher instance to reuse analysis.
        """
        self.model = model
        self.base_watcher = base_watcher
        self.results = None

    def calculate_gini_coefficient(self, evals: np.ndarray) -> float:
        """
        Calculate the Gini coefficient of the eigenvalue distribution.

        The Gini coefficient measures inequality in distribution (0=perfect equality, 1=perfect inequality).
        A high Gini coefficient indicates eigenvalue concentration in a few dominant values.

        Args:
            evals: Array of eigenvalues.

        Returns:
            float: Gini coefficient between 0 and 1.
        """
        if len(evals) < 2:
            return 0.0

        # Ensure all values are non-negative and sort
        sorted_evals = np.sort(np.abs(evals))
        n = len(sorted_evals)

        # Calculate Lorenz curve
        cum_evals = np.cumsum(sorted_evals)

        # Calculate Gini coefficient
        # G = 1 - 2 * area under Lorenz curve
        # = 1 - (2/n) * sum_{i=1}^n (n-i+0.5)/n * x_i/sum(x)
        # Simplified: ((n+1)/n) - (2 * sum(cum_evals)) / (n * sum(sorted_evals))
        denominator = n * sorted_evals.sum()
        if denominator < EPSILON:
            return 0.0

        return ((n + 1) / n) - (2 * np.sum(cum_evals)) / denominator

    def calculate_dominance_ratio(self, evals: np.ndarray) -> float:
        """
        Calculate the ratio of the largest eigenvalue to the sum of all other eigenvalues.

        This directly quantifies how much a single dimension dominates the spectrum.
        High values indicate potential "black hole" behavior.

        Args:
            evals: Array of eigenvalues.

        Returns:
            float: Dominance ratio (λ_max / sum(λ_others)).
        """
        if len(evals) < 2:
            return float('inf')

        lambda_max = np.max(evals)
        sum_others = np.sum(evals) - lambda_max

        if sum_others < EPSILON:
            return float('inf')

        return lambda_max / sum_others

    def calculate_participation_ratio(self, vector: np.ndarray) -> float:
        """
        Calculate the participation ratio of a vector, a measure of localization.

        PR = (sum(v_i^2))^2 / sum(v_i^4)

        A low participation ratio indicates the vector's energy is concentrated in
        a few elements (localized) - a signature of "black hole" weights.

        Args:
            vector: Eigenvector or other vector to analyze.

        Returns:
            float: Participation ratio. Lower values indicate more localization.
        """
        # Normalize the vector
        vec = vector / (np.linalg.norm(vector) + EPSILON)

        # Calculate participation ratio
        vec_sq = vec ** 2
        numerator = np.sum(vec_sq) ** 2
        denominator = np.sum(vec_sq ** 2)

        if denominator < EPSILON:
            return float('inf')

        return numerator / denominator

    def get_top_eigenvectors(self,
                             weight_matrix: np.ndarray,
                             k: int = 1,
                             method: str = 'direct') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the top k eigenvectors of a weight matrix.

        Args:
            weight_matrix: Weight matrix to analyze.
            k: Number of top eigenvectors to return.
            method: Method to use ('direct' or 'power_iteration').

        Returns:
            Tuple containing:
            - Array of eigenvalues.
            - Array of eigenvectors (each column is an eigenvector).
        """
        n, m = weight_matrix.shape
        min_dim = min(n, m)

        k = min(k, min_dim - 1)

        # Ensure matrix is square for eigendecomposition
        if n == m:
            # Square matrix - direct eigendecomposition
            try:
                if method == 'direct':
                    eigenvalues, eigenvectors = eigh(
                        weight_matrix @ weight_matrix.T,
                        eigvals=(min_dim - k, min_dim - 1)
                    )
                    # Sort by eigenvalue magnitude (descending)
                    idx = np.argsort(eigenvalues)[::-1]
                    return eigenvalues[idx], eigenvectors[:, idx]
                else:
                    # Use power iteration for very large matrices
                    return self._power_iteration(weight_matrix @ weight_matrix.T, k)
            except Exception as e:
                logger.warning(f"Eigendecomposition failed: {e}")
                return np.array([]), np.array([])
        else:
            # Non-square matrix - use SVD
            try:
                # Calculate SVD
                u, s, vh = np.linalg.svd(weight_matrix, full_matrices=False)
                # Return singular values and left singular vectors (eigenvectors of W*W^T)
                return s[:k] ** 2, u[:, :k]
            except Exception as e:
                logger.warning(f"SVD failed: {e}")
                return np.array([]), np.array([])

    def _power_iteration(self,
                         matrix: np.ndarray,
                         k: int = 1,
                         max_iter: int = 100,
                         tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the top k eigenvalues and eigenvectors using power iteration.

        This is more efficient for very large matrices when we only need a few eigenvectors.

        Args:
            matrix: Square matrix to analyze.
            k: Number of eigenvectors to compute.
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.

        Returns:
            Tuple containing:
            - Array of eigenvalues.
            - Array of eigenvectors.
        """
        n = matrix.shape[0]
        eigvals = np.zeros(k)
        eigvecs = np.zeros((n, k))

        # Start with random vectors
        Q = np.random.randn(n, k)
        Q, _ = np.linalg.qr(Q)

        for i in range(k):
            q = Q[:, i].reshape(-1, 1)

            # Deflate previously computed eigenvectors
            for j in range(i):
                q = q - eigvecs[:, j].reshape(-1, 1) @ (eigvecs[:, j].reshape(1, -1) @ q)

            # Power iteration
            for _ in range(max_iter):
                z = matrix @ q
                lambda_i = np.linalg.norm(z)
                q_new = z / (lambda_i + EPSILON)

                # Check convergence
                if np.linalg.norm(q_new - q) < tol:
                    break

                q = q_new

            # Store results
            eigvals[i] = lambda_i
            eigvecs[:, i] = q.flatten()

        return eigvals, eigvecs

    def find_super_weights(self,
                           weight_matrix: np.ndarray,
                           eigenvectors: np.ndarray,
                           eigenvalues: np.ndarray,
                           threshold: float = 0.1) -> List[Tuple[int, int, float]]:
        """
        Find the individual weights that contribute most to top eigenvectors.

        Args:
            weight_matrix: Weight matrix to analyze.
            eigenvectors: Top eigenvectors.
            eigenvalues: Corresponding eigenvalues.
            threshold: Contribution threshold for identifying super weights.

        Returns:
            List of tuples (i, j, contribution) for super weights.
        """
        if len(eigenvectors) == 0 or len(eigenvalues) == 0:
            return []

        n, m = weight_matrix.shape
        super_weights = []

        # Calculate weight importance using eigenvector components
        for k, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # Scale by eigenvalue importance
            importance = np.abs(eigvec) * np.sqrt(eigval)

            # Find indices of high-importance components
            high_importance_indices = np.where(importance > threshold * np.max(importance))[0]

            for idx in high_importance_indices:
                # For dense layers, convert flat index to 2D index
                if len(weight_matrix.shape) == 2:
                    i = idx // m if n > m else idx
                    j = idx % m if n > m else idx
                    contribution = importance[idx] * weight_matrix[i, j]
                    super_weights.append((i, j, float(contribution)))
                else:
                    # Handle convolutional layers
                    # This is a simplified approach - would need to be adapted for actual conv layers
                    super_weights.append((idx, 0, float(importance[idx])))

        # Sort by contribution magnitude (descending)
        super_weights.sort(key=lambda x: abs(x[2]), reverse=True)

        return super_weights

    def analyze_layer_black_holes(self,
                                  layer_id: int,
                                  layer: keras.layers.Layer,
                                  weight_matrices: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a layer for computational black hole properties.

        Args:
            layer_id: ID of the layer.
            layer: Keras layer.
            weight_matrices: List of weight matrices to analyze.

        Returns:
            Dictionary of metrics for the layer.
        """
        results = {}

        # Skip if no weight matrices
        if not weight_matrices:
            return results

        # Calculate metrics for each weight matrix
        all_gini = []
        all_dominance = []
        all_super_weights = []
        all_participation_ratios = []

        for w_idx, W in enumerate(weight_matrices):
            # Skip if matrix is too small
            if min(W.shape) < DEFAULT_MIN_EVALS:
                continue

            # Get top eigenvectors
            eigenvalues, eigenvectors = self.get_top_eigenvectors(W, k=3)

            if len(eigenvalues) == 0:
                continue

            # Calculate metrics
            gini = self.calculate_gini_coefficient(eigenvalues)
            dominance = self.calculate_dominance_ratio(eigenvalues)

            # Calculate participation ratios for top eigenvectors
            prs = [self.calculate_participation_ratio(eigenvectors[:, i])
                   for i in range(min(3, eigenvectors.shape[1]))]

            # Find super weights
            super_weights = self.find_super_weights(W, eigenvectors, eigenvalues)

            all_gini.append(gini)
            all_dominance.append(dominance)
            all_participation_ratios.extend(prs)
            all_super_weights.extend(super_weights)

        # Aggregate metrics
        if all_gini:
            results['gini_coefficient'] = np.mean(all_gini)
            results['max_gini_coefficient'] = np.max(all_gini)

        if all_dominance:
            results['dominance_ratio'] = np.mean(all_dominance)
            results['max_dominance_ratio'] = np.max(all_dominance)

        if all_participation_ratios:
            results['participation_ratio'] = np.mean(all_participation_ratios)
            results['min_participation_ratio'] = np.min(all_participation_ratios)

        if all_super_weights:
            # Keep only top super weights
            top_super_weights = sorted(all_super_weights, key=lambda x: abs(x[2]), reverse=True)[:10]
            results['super_weights'] = top_super_weights
            results['super_weight_count'] = len(top_super_weights)
            results['max_super_weight_contribution'] = abs(top_super_weights[0][2]) if top_super_weights else 0

        # Calculate "black hole score" - higher means more likely to be a computational black hole
        if 'gini_coefficient' in results and 'dominance_ratio' in results and 'participation_ratio' in results:
            black_hole_score = (
                    results['max_gini_coefficient'] *
                    results['max_dominance_ratio'] /
                    (results['min_participation_ratio'] + EPSILON)
            )
            results['black_hole_score'] = np.log1p(black_hole_score)  # Log to manage extreme values

        return results

    def analyze(self,
                layers: List[int] = None,
                top_k: int = 10,
                plot: bool = False,
                savefig: Union[bool, str] = 'ww-blackhole-img') -> pd.DataFrame:
        """
        Analyze the model to detect computational black holes.

        Args:
            layers: List of layer indices to analyze. If None, analyze all layers.
            top_k: Number of top contributing weights to report.
            plot: Whether to create visualizations.
            savefig: Directory to save figures or False to disable saving.

        Returns:
            DataFrame with analysis results.
        """
        logger.info("Starting computational black hole analysis")

        results = []

        # Set up savedir for plots
        savedir = None
        if plot and savefig:
            savedir = 'ww-blackhole-img' if savefig is True else savefig
            os.makedirs(savedir, exist_ok=True)

        # Iterate through layers
        for layer_id, layer in enumerate(self.model.layers):
            if layers is not None and layer_id not in layers:
                continue

            layer_name = layer.name
            layer_type = infer_layer_type(layer)

            # Skip unsupported layer types
            if layer_type not in [LayerType.DENSE, LayerType.CONV1D, LayerType.CONV2D, LayerType.CONV3D]:
                continue

            logger.info(f"Analyzing layer {layer_id}: {layer_name}")

            # Get weights
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

            if not has_weights:
                continue

            # Extract weight matrices
            Wmats, N, M, rf = get_weight_matrices(weights, layer_type)

            # Calculate black hole metrics
            black_hole_metrics = self.analyze_layer_black_holes(layer_id, layer, Wmats)

            # Skip if no metrics calculated
            if not black_hole_metrics:
                continue

            # Create result row
            row = {
                'layer_id': layer_id,
                'name': layer_name,
                'layer_type': layer_type,
                'N': N,
                'M': M,
                'param_count': np.prod(weights.shape),
                **black_hole_metrics
            }

            results.append(row)

            # Create visualizations if requested
            if plot and savedir and 'super_weights' in black_hole_metrics:
                self._plot_super_weights(
                    layer_id, layer_name, weights, black_hole_metrics['super_weights'], savedir
                )

        # Create DataFrame from results
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Set layer_id as index
            results_df.set_index('layer_id', inplace=True)

            # Identify top black hole layers
            if 'black_hole_score' in results_df.columns:
                top_layers = results_df.sort_values('black_hole_score', ascending=False).head(3)
                logger.info(f"Top 3 'Black Hole' layers:")
                for idx, row in top_layers.iterrows():
                    logger.info(f"Layer {idx} ({row['name']}): Black Hole Score = {row['black_hole_score']:.4f}")

        self.results = results_df
        return results_df

    def _plot_super_weights(self,
                            layer_id: int,
                            layer_name: str,
                            weights: np.ndarray,
                            super_weights: List[Tuple[int, int, float]],
                            savedir: str) -> None:
        """
        Create visualizations of super weights.

        Args:
            layer_id: ID of the layer.
            layer_name: Name of the layer.
            weights: Weight tensor.
            super_weights: List of super weight tuples (i, j, contribution).
            savedir: Directory to save figures.
        """
        if len(super_weights) == 0:
            return

        try:
            # For dense layers, create heatmap with super weights highlighted
            if len(weights.shape) == 2:
                plt.figure(figsize=DEFAULT_FIG_SIZE)

                # Plot weight matrix heatmap
                plt.imshow(weights, cmap='viridis', aspect='auto')
                plt.colorbar(label='Weight Value')

                # Highlight super weights
                for i, j, _ in super_weights[:10]:  # Limit to top 10 for clarity
                    plt.scatter(j, i, c='red', s=100, marker='*', edgecolors='white')

                plt.title(f"Super Weights in Layer {layer_id}: {layer_name}")
                plt.xlabel("Output Units")
                plt.ylabel("Input Units")

                plt.tight_layout()
                plt.savefig(f"{savedir}/layer_{layer_id}_superweights.png", dpi=DEFAULT_DPI)
                plt.close()

                # Plot super weight contribution distribution
                plt.figure(figsize=DEFAULT_FIG_SIZE)
                contributions = [abs(c) for _, _, c in super_weights]
                plt.bar(range(len(contributions)), contributions)
                plt.title(f"Super Weight Contributions - Layer {layer_id}: {layer_name}")
                plt.xlabel("Super Weight Rank")
                plt.ylabel("Contribution Magnitude")
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(f"{savedir}/layer_{layer_id}_contributions.png", dpi=DEFAULT_DPI)
                plt.close()

            # For conv layers, different visualization approach needed
            elif len(weights.shape) == 4:  # Conv2D
                # Just show distribution of super weight contributions for now
                plt.figure(figsize=DEFAULT_FIG_SIZE)
                contributions = [abs(c) for _, _, c in super_weights]
                plt.bar(range(len(contributions)), contributions)
                plt.title(f"Super Weight Contributions - Layer {layer_id}: {layer_name}")
                plt.xlabel("Super Weight Rank")
                plt.ylabel("Contribution Magnitude")
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(f"{savedir}/layer_{layer_id}_contributions.png", dpi=DEFAULT_DPI)
                plt.close()

        except Exception as e:
            logger.warning(f"Error creating super weight visualization: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the black hole analysis.

        Returns:
            Dictionary with summary metrics.
        """
        if self.results is None or self.results.empty:
            logger.warning("No analysis results available. Run analyze() first.")
            return {}

        summary = {}

        # Calculate aggregated metrics
        for metric in ['gini_coefficient', 'dominance_ratio', 'participation_ratio', 'black_hole_score']:
            if metric in self.results.columns:
                summary[f'mean_{metric}'] = self.results[metric].mean()
                summary[f'max_{metric}'] = self.results[metric].max()
                summary[f'min_{metric}'] = self.results[metric].min()

        # Count layers with significant black hole properties
        if 'black_hole_score' in self.results.columns:
            threshold = self.results['black_hole_score'].quantile(0.9)  # Top 10%
            summary['black_hole_layer_count'] = sum(self.results['black_hole_score'] > threshold)
            summary['black_hole_layer_percentage'] = summary['black_hole_layer_count'] / len(self.results) * 100

        return summary

    def ablate_super_weights(self, output_model_path: Optional[str] = None) -> keras.Model:
        """
        Create a copy of the model with top super weights ablated (set to zero).
        This tests the hypothesis that these weights are critical for model function.

        Args:
            output_model_path: Optional path to save the ablated model.

        Returns:
            Ablated Keras model.
        """
        if self.results is None or self.results.empty:
            logger.warning("No analysis results available. Run analyze() first.")
            return self.model

        # Clone the model
        ablated_model = keras.models.clone_model(self.model)
        ablated_model.set_weights(self.model.get_weights())

        # Get top black hole layers
        if 'black_hole_score' not in self.results.columns:
            logger.warning("No black hole scores available.")
            return ablated_model

        top_layers = self.results.sort_values('black_hole_score', ascending=False).head(3)

        for layer_id in top_layers.index:
            if 'super_weights' not in self.results.loc[layer_id]:
                continue

            logger.info(f"Ablating super weights in layer {layer_id}")

            # Get layer weights
            layer = ablated_model.layers[layer_id]
            has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

            if not has_weights:
                continue

            # Get super weights
            super_weights = self.results.loc[layer_id, 'super_weights']

            # Ablate super weights (set to zero)
            if len(weights.shape) == 2:  # Dense layer
                for i, j, _ in super_weights:
                    weights[i, j] = 0.0

            # Update layer weights
            if has_bias:
                layer.set_weights([weights, bias])
            else:
                layer.set_weights([weights])

        # Save ablated model if requested
        if output_model_path:
            ablated_model.save(output_model_path)
            logger.info(f"Saved ablated model to {output_model_path}")

        return ablated_model

    def filter_super_weight_layers(self, threshold: float = 10.0) -> Set[int]:
        """
        Return set of layer IDs that contain super weights based on black hole score.

        Args:
            threshold: Minimum black hole score to qualify.

        Returns:
            Set of layer IDs.
        """
        if self.results is None or self.results.empty or 'black_hole_score' not in self.results.columns:
            logger.warning("No analysis results available. Run analyze() first.")
            return set()

        return set(self.results[self.results['black_hole_score'] > threshold].index)