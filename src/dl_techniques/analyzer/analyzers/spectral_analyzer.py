"""
Analyze model generalization and training quality via spectral properties.

This class implements the "WeightWatcher" methodology, a data-independent
approach for assessing the quality of trained deep neural networks. It
operates by examining the spectral properties of layer weight matrices,
providing insights into phenomena like overfitting, under-training, and model
complexity without requiring access to test or training data.

Architecture and Methodology
---------------------------
The analyzer functions as an orchestrator, systematically applying spectral
analysis to each qualifying layer of a given model. Its workflow is as
follows:

1.  **Layer Identification**: It first traverses the model architecture to
    identify layers with analyzable weight tensors (e.g., Dense, Conv2D).
2.  **Weight Matrix Extraction**: For each identified layer, it uses utility
    functions to extract the weight tensor and reshape (matricize) it into a
    standard 2D matrix, `W`. This critical step adapts the heterogeneous
    tensor formats of different layer types for uniform analysis.
3.  **Spectral Computation**: It computes the eigenvalues {λ_i} of the layer's
    correlation matrix, `WW^T`. This is efficiently done by calculating the
    squared singular values of `W` via Singular Value Decomposition (SVD). The
    resulting set of eigenvalues forms the Empirical Spectral Density (ESD).
4.  **Power-Law Fitting**: The core of the analysis involves fitting the tail
    of the ESD to a truncated power-law distribution. This step tests the
    hypothesis that well-trained models exhibit heavy-tailed spectral
    distributions.
5.  **Metric Aggregation**: Results from each layer, including the power-law
    exponent (α), stable rank, and concentration metrics, are compiled into
    a comprehensive pandas DataFrame. This allows for both layer-level
    diagnostics and model-level summary statistics.
6.  **Recommendation Generation**: Based on the aggregated metrics, particularly
    the mean α, the analyzer generates actionable recommendations regarding
    the model's training state.

Foundational Mathematics: Heavy-Tailed Self-Regularization
---------------------------------------------------------
The analysis is grounded in the theory of Heavy-Tailed Self-Regularization,
which posits that Stochastic Gradient Descent (SGD) implicitly regularizes
the layers of a deep neural network, causing their weight matrix spectra to
develop characteristic heavy-tailed structures.

-   **Power-Law Exponent (α)**: The primary metric is the exponent `α` of the
    power-law fit to the ESD tail, P(λ) ~ λ^(-α). This exponent is estimated
    using a robust Maximum Likelihood Estimator. The value of `α` has been
    shown to correlate strongly with a model's generalization capabilities:
    -   `α < 2.0`: Suggests an extremely heavy-tailed spectrum, which can
        indicate overfitting or memorization of the training set.
    -   `2.0 < α < 6.0`: Typically corresponds to a well-trained model that
        has learned meaningful features and is expected to generalize well.
    -   `α > 6.0`: Indicates a less heavy-tailed spectrum, which may be a
        sign of under-training or a model that has failed to learn complex,
        hierarchical features.

-   **Concentration and Information Metrics**: In addition to `α`, other
    metrics are used to characterize the spectrum's shape:
    -   **Stable Rank**: Defined as (Σ λ_i) / max(λ_i), it measures the
        effective dimensionality of the weight matrix, providing a more robust
        metric than the discrete matrix rank.
    -   **Gini Coefficient / Dominance Ratio**: These metrics quantify the
        inequality of the eigenvalue distribution. High values indicate that
        information is concentrated in a few dominant modes, which can make
        the model brittle or sensitive to small perturbations.

References
----------
1.  Martin, C., & Mahoney, M. W. (2021). "Heavy-Tailed Universals in
    Deep Neural Networks." arXiv preprint arXiv:2106.07590.
2.  Martin, C., & Mahoney, M. W. (2019). "Predicting the Generalization
    Gap in Deep Networks with Margin Distributions." ICLR.
3.  Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). "Power-law
    distributions in empirical data." SIAM review, 51(4), 661-703.

"""

import keras
import warnings
import contextlib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput
from ..constants import (
    LayerType, MetricNames, SPECTRAL_DEFAULT_SUMMARY_METRICS,
    SPECTRAL_HIGH_CONCENTRATION_PERCENTILE, SPECTRAL_WEAK_RANK_LOSS_TOLERANCE
)
from .. import spectral_metrics
from .. import spectral_utils
from ..utils import recursively_get_layers
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class SpectralAnalyzer(BaseAnalyzer):
    """
    Performs spectral analysis of model weights (WeightWatcher).

    This analyzer computes eigenvalue distributions, fits power-law models, calculates
    entropy, and performs concentration analysis to assess training quality and complexity.
    """

    def requires_data(self) -> bool:
        """Spectral analysis is data-independent."""
        return False

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Perform comprehensive spectral analysis on all models.
        """
        if not self.config.analyze_spectral:
            return

        logger.info("Analyzing weight matrices with spectral methods (WeightWatcher)...")

        all_model_details = []
        # Initialize storage for ESDs and recommendations
        results.spectral_esds = {}
        results.spectral_rand_esds = {}
        results.spectral_recommendations = {}

        for model_name, model in self.models.items():
            logger.info(f"Starting spectral analysis for model: {model.name}")
            details_df = self._analyze_single_model(model)

            if not details_df.empty:
                details_df['model_name'] = model_name
                all_model_details.append(details_df)

                # Store ESDs and recommendations
                if hasattr(self, '_esd_cache'):
                    results.spectral_esds[model_name] = self._esd_cache
                if hasattr(self, '_rand_esd_cache') and self._rand_esd_cache:
                    results.spectral_rand_esds[model_name] = self._rand_esd_cache
                if hasattr(self, '_recommendations'):
                    results.spectral_recommendations[model_name] = self._recommendations

        if all_model_details:
            # Consolidate results into a single DataFrame
            results.spectral_analysis = pd.concat(all_model_details, ignore_index=True)

            # Compute and store summary metrics
            results.spectral_summary = self._get_summary(results.spectral_analysis)
        else:
            logger.warning("Spectral analysis did not produce any results for any model.")

    def _analyze_single_model(self, model: keras.Model) -> pd.DataFrame:
        """
        Perform spectral analysis on a single Keras model.
        """
        # Create basic description of the model to find analyzable layers
        # This now returns both the details DataFrame and the flattened list of layers.
        details, all_layers = self._describe_model(model)
        self._esd_cache: Dict[int, np.ndarray] = {}
        self._rand_esd_cache: Dict[int, np.ndarray] = {}

        if details.empty:
            logger.warning(f"No layers found in model '{model.name}' that meet the criteria for spectral analysis.")
            return details.reset_index()  # Return empty but correctly formatted DataFrame

        # Perform detailed analysis on each qualifying layer
        # Suppress runtime warnings locally (e.g. divide-by-zero in eigenvalue calculations)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self._analyze_layers(details, all_layers)

        # Generate and store recommendations for this model
        summary = self._get_summary(details)
        self._recommendations = self._generate_recommendations(details, summary)

        # Convert layer_id index to a column for robust concatenation
        return details.reset_index()

    def _analyze_layers(self, details: pd.DataFrame, all_layers: List[keras.layers.Layer]) -> None:
        """Perform spectral analysis on each qualifying layer."""
        for layer_id in details.index:
            layer = all_layers[layer_id]

            logger.debug(f"Analyzing layer {layer_id}: {layer.name}")

            # Extract weights, dimensions, and other properties
            has_weights, weights, _, _ = spectral_utils.get_layer_weights_and_bias(layer)
            if not has_weights: continue

            layer_type = spectral_utils.infer_layer_type(layer)
            Wmats, N, M, rf = spectral_utils.get_weight_matrices(weights, layer_type)

            if self.config.spectral_glorot_fix:
                kappa = spectral_metrics.calculate_glorot_normalization_factor(N, M, rf)
                Wmats = [W / kappa for W in Wmats]

            n_comp = M * rf
            evals, sv_max, sv_min, rank_loss = spectral_metrics.compute_eigenvalues(Wmats, N, M, n_comp)
            self._esd_cache[layer_id] = evals

            weak_rank_loss = np.sum(evals < SPECTRAL_WEAK_RANK_LOSS_TOLERANCE)
            alpha, xmin, D, sigma, num_pl_spikes, status, warning = spectral_metrics.fit_powerlaw(evals)
            entropy = spectral_metrics.calculate_matrix_entropy(np.sqrt(evals), N)
            spectral_mets = spectral_metrics.calculate_spectral_metrics(evals, alpha, N=N)

            # SETOL: Learning phase classification
            learning_phase = spectral_metrics.classify_learning_phase(alpha)

            # SETOL: ERG condition (Δλ_min diagnostic)
            erg_metrics = {}
            if status == "success" and xmin > 0:
                erg_metrics = spectral_metrics.compute_erg_condition(evals, xmin)

            # SETOL: Goodness-of-fit p-value (Clauset et al. 2009)
            pl_pvalue = -1.0
            if status == "success" and alpha > 1.0:
                pl_pvalue = spectral_metrics.powerlaw_goodness_of_fit(
                    evals, alpha, xmin, n_bootstraps=self.config.spectral_bootstraps)

            concentration_metrics = {}
            if self.config.spectral_concentration_analysis and Wmats:
                concentration_metrics = spectral_metrics.calculate_concentration_metrics(
                    Wmats[0], evals=evals)

            randomization_metrics = {}
            if self.config.spectral_randomize and Wmats:
                rand_Wmats = [np.random.permutation(W.flatten()).reshape(W.shape) for W in Wmats]
                rand_evals, rand_sv_max, _, _ = spectral_metrics.compute_eigenvalues(rand_Wmats, N, M, n_comp)
                rand_distance = spectral_metrics.jensen_shannon_distance(evals, rand_evals)
                ww_softrank = np.max(rand_evals) / np.max(evals) if np.max(evals) > 0 else 0
                randomization_metrics = {
                    MetricNames.RAND_SV_MAX: rand_sv_max,
                    MetricNames.RAND_DISTANCE: rand_distance,
                    MetricNames.WW_SOFTRANK: ww_softrank,
                }

                # Correlation trap detection via MP edge + Tracy-Widom threshold
                trap_result = spectral_metrics.detect_correlation_trap(rand_evals, N, M)
                randomization_metrics.update({
                    MetricNames.HAS_TRAP: trap_result['has_trap'],
                    MetricNames.NUM_RAND_SPIKES: trap_result['num_rand_spikes'],
                    MetricNames.TRAP_SEVERITY: trap_result['trap_severity'],
                    MetricNames.TRAP_SEVERITY_LABEL: trap_result['trap_severity_label'],
                    MetricNames.MP_LAMBDA_PLUS: trap_result['mp_lambda_plus'],
                    MetricNames.MP_LAMBDA_MINUS: trap_result['mp_lambda_minus'],
                    MetricNames.TRAP_THRESHOLD: trap_result['trap_threshold'],
                })

                # Store randomized eigenvalues for visualization
                self._rand_esd_cache[layer_id] = rand_evals

            metrics = {
                MetricNames.HAS_ESD: True, MetricNames.NUM_EVALS: len(evals),
                MetricNames.SV_MAX: sv_max, MetricNames.SV_MIN: sv_min,
                MetricNames.RANK_LOSS: rank_loss, MetricNames.WEAK_RANK_LOSS: weak_rank_loss,
                MetricNames.LAMBDA_MAX: np.max(evals) if len(evals) > 0 else 0,
                MetricNames.ALPHA: alpha, MetricNames.XMIN: xmin, MetricNames.D: D,
                MetricNames.SIGMA: sigma, MetricNames.NUM_PL_SPIKES: num_pl_spikes,
                MetricNames.STATUS: status, MetricNames.WARNING: warning,
                MetricNames.ENTROPY: entropy,
                'learning_phase': learning_phase,
                'pl_pvalue': pl_pvalue,
                **erg_metrics,
                **spectral_mets, **concentration_metrics, **randomization_metrics
            }

            for key, value in metrics.items():
                if key != 'critical_weights':
                    details.at[layer_id, key] = value

    def _describe_model(self, model: keras.Model) -> Tuple[pd.DataFrame, List[keras.layers.Layer]]:
        """
        Describe the model architecture to find all analyzable layers, including nested ones.

        Returns:
            A tuple containing:
            - A pandas DataFrame with metadata for each analyzable layer.
            - The flat list of all layers discovered recursively.
        """
        all_layers = recursively_get_layers(model)
        rows = []

        for layer_id, layer in enumerate(all_layers):
            layer_type = spectral_utils.infer_layer_type(layer)
            has_weights, weights, _, _ = spectral_utils.get_layer_weights_and_bias(layer)
            if not has_weights or layer_type == LayerType.UNKNOWN:
                continue

            Wmats, N, M, rf = spectral_utils.get_weight_matrices(weights, layer_type)
            if M < self.config.spectral_min_evals or M > self.config.spectral_max_evals:
                continue

            rows.append({
                'layer_id': layer_id, 'name': layer.name, 'layer_type': layer_type.value,
                'N': N, 'M': M, 'rf': rf, 'Q': N / M if M > 0 else -1,
                'num_params': int(np.prod(weights.shape)),
                MetricNames.NUM_EVALS: M * rf
            })

        details = pd.DataFrame(rows)
        if not details.empty:
            details.set_index('layer_id', inplace=True)

        return details, all_layers

    def _get_summary(self, details_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary metrics averaged across all analyzed layers for all models.
        """
        summary = {}
        if details_df is None or details_df.empty:
            return summary

        metrics_to_summarize = [m for m in SPECTRAL_DEFAULT_SUMMARY_METRICS if m in details_df.columns]

        for metric in metrics_to_summarize:
            valid_values = details_df[metric][pd.to_numeric(details_df[metric], errors='coerce').notna()]
            if not valid_values.empty:
                summary[metric] = float(valid_values.mean())

        summary['total_layers_analyzed'] = len(details_df)
        if 'concentration_score' in details_df.columns:
            score = pd.to_numeric(details_df['concentration_score'], errors='coerce').dropna()
            if not score.empty:
                high_conc_threshold = score.quantile(SPECTRAL_HIGH_CONCENTRATION_PERCENTILE)
                summary['high_concentration_layers'] = int(sum(score > high_conc_threshold))

        return summary

    def _generate_recommendations(self, analysis_df: pd.DataFrame, summary: Dict[str, float]) -> List[str]:
        """
        Generate analysis-based recommendations for model optimization.
        """
        recommendations = []
        if 'alpha' in summary:
            mean_alpha = summary['alpha']
            if mean_alpha < 2.0:
                recommendations.append("Model may be over-trained (low α). Consider early stopping or regularization.")
            elif mean_alpha > 6.0:
                recommendations.append("Model may be under-trained (high α). Consider training longer or reducing regularization.")
            else:
                recommendations.append(f"Model training quality appears good (α = {mean_alpha:.2f}).")

        # SETOL: Phase distribution analysis
        if 'learning_phase' in analysis_df.columns:
            phase_counts = analysis_df['learning_phase'].value_counts()
            total = len(analysis_df)
            for phase, count in phase_counts.items():
                pct = 100 * count / total
                if phase == 'over-regularized' and pct > 20:
                    recommendations.append(
                        f"{pct:.0f}% of layers are over-regularized (α < 2.0). "
                        "Reduce learning rate or check for correlation traps.")
                elif phase == 'under-trained' and pct > 20:
                    recommendations.append(
                        f"{pct:.0f}% of layers are under-trained (α > 6.0). "
                        "Train longer or reduce regularization.")
                elif phase == 'ideal' and pct > 50:
                    recommendations.append(
                        f"{pct:.0f}% of layers are in the ideal phase (α ≈ 2.0). Excellent training quality.")

        # SETOL: ERG condition check
        if 'erg_satisfied' in analysis_df.columns:
            erg_satisfied = analysis_df['erg_satisfied'].sum()
            total = len(analysis_df)
            if erg_satisfied > 0:
                recommendations.append(
                    f"{erg_satisfied}/{total} layers satisfy the ERG condition (ideal learning).")

        # Power-law fit quality
        if 'pl_pvalue' in analysis_df.columns:
            poor_fits = analysis_df[
                (analysis_df['pl_pvalue'] >= 0) & (analysis_df['pl_pvalue'] < 0.1)]
            if len(poor_fits) > 0:
                recommendations.append(
                    f"{len(poor_fits)} layers have poor power-law fit (p < 0.1). "
                    "Their α values may be unreliable.")

        if 'concentration_score' in summary and summary['concentration_score'] > 5.0:
            recommendations.append("High information concentration detected. Be careful with pruning/quantization.")

        if 'rank_loss' in analysis_df.columns:
            high_rank_loss = analysis_df[analysis_df['rank_loss'] > 0.1 * analysis_df['M']]
            if not high_rank_loss.empty:
                recommendations.append("Some layers show significant rank loss. Consider SVD smoothing.")

        # Correlation trap detection results
        if MetricNames.HAS_TRAP in analysis_df.columns:
            trap_layers = analysis_df[analysis_df[MetricNames.HAS_TRAP] == True]
            if len(trap_layers) > 0:
                total = len(analysis_df)
                severity_counts = trap_layers[MetricNames.TRAP_SEVERITY_LABEL].value_counts()
                severity_summary = ", ".join(
                    f"{count} {label}" for label, count in severity_counts.items()
                )
                recommendations.append(
                    f"Correlation traps detected in {len(trap_layers)}/{total} layers "
                    f"({severity_summary}). "
                    "Reduce learning rate, increase batch size, or add regularization."
                )

                # Flag critical/severe traps specifically
                critical = trap_layers[
                    trap_layers[MetricNames.TRAP_SEVERITY_LABEL].isin(['severe', 'critical'])
                ]
                if len(critical) > 0:
                    layer_names = critical['name'].tolist()[:5]
                    recommendations.append(
                        f"Severe/critical traps in: {', '.join(layer_names)}. "
                        "Consider rolling back to an earlier checkpoint."
                    )

        return recommendations