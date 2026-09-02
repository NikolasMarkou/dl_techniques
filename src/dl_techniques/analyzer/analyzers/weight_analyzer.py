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

import warnings

import numpy as np
import scipy.stats
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .base import BaseAnalyzer
from ..constants import StatusCode
from ..data_types import AnalysisResults, DataInput
from ..utils import recursively_get_layers

# ---------------------------------------------------------------------

#: Every scalar leaf of a weight-statistics dict, as ``(group, key)``. The
#: classifier walks exactly this list, so a statistic added to
#: :meth:`WeightAnalyzer._raw_statistics` must be added here too or it will be
#: allowed to carry a NaN into the PCA feature matrix unflagged.
_STAT_LEAVES: Tuple[Tuple[str, str], ...] = (
    ('basic', 'mean'), ('basic', 'std'), ('basic', 'median'),
    ('basic', 'min'), ('basic', 'max'),
    ('basic', 'skewness'), ('basic', 'kurtosis'),
    ('norms', 'l1'), ('norms', 'l2'), ('norms', 'max'), ('norms', 'rms'),
    ('norms', 'spectral'),
    ('distribution', 'zero_fraction'),
    ('distribution', 'positive_fraction'),
    ('distribution', 'negative_fraction'),
)

#: Leaves for which ``0.0`` is a defensible SUBSTITUTION when the quantity is
#: undefined rather than merely unrepresentable. See
#: :meth:`WeightAnalyzer._compute_weight_statistics` for the justification.
_SUBSTITUTABLE_LEAVES: Tuple[Tuple[str, str], ...] = (
    ('basic', 'skewness'), ('basic', 'kurtosis'), ('norms', 'spectral'),
)

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

            for layer in recursively_get_layers(model):
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
                    if stats is None:
                        logger.warning(
                            f"Skipping weight {model_name}/{weight_name}: "
                            f"the tensor is empty (shape {w.shape}), so no "
                            "statistic is defined for it."
                        )
                        continue
                    results.weight_stats[model_name][weight_name] = stats

                    # Track layer order for robust visualization
                    if weight_name not in layer_names_in_order:
                        layer_names_in_order.append(weight_name)

            # Store explicit layer order for visualizers to use
            results.weight_stats_layer_order[model_name] = layer_names_in_order

        # Compute PCA if requested
        if self.config.compute_weight_pca:
            self._compute_weight_pca(results)

    def _compute_weight_statistics(
            self, weights: np.ndarray) -> Optional[Dict[str, Any]]:
        """Compute comprehensive statistics for a weight tensor, and classify them.

        A freshly-initialised model is legal input, and its zeros-initialised
        tables make several of these statistics undefined or unrepresentable.
        Five mechanisms were measured, and all five used to put a ``NaN``/``inf``
        into the PCA feature matrix (or raise), which aborted the entire
        analysis:

        1. **zero variance** -- ``scipy.stats.skew``/``kurtosis`` return ``NaN``
           for a constant, all-zero, length-1 or length-2-equal input;
        2. **float32 reduction overflow** -- ``np.sum(w ** 2)`` overflows to
           ``inf`` even when every element is perfectly representable;
        3. **float32 underflow of the 2nd moment** -- a NON-constant tensor such
           as ``randn * 1e-30`` still yields ``skewness = NaN``, which is why the
           classification below keys on ``np.isfinite`` of the RESULT and never
           on whether the tensor is constant;
        4. **an already-corrupt tensor** -- a real ``NaN``/``inf`` in the weights;
        5. **raises** -- a zero-size tensor (``np.min``) and a ``float16`` tensor
           (``np.linalg.norm``), neither of which is a ``LinAlgError``.

        Mechanisms 2, 3 and 5 are computation artifacts: the quantity exists, the
        working precision could not hold it, so it is RECOMPUTED in ``float64``
        and the true value is published. Mechanism 1 is different -- for a
        zero-variance (Dirac) distribution the standardized third and fourth
        moments are :math:`0/0`, i.e. **undefined, not zero**. Publishing ``0.0``
        is a SUBSTITUTION CONVENTION chosen so a model's fingerprint stays
        comparable across layers; the ``status`` and ``degenerate_fields`` keys
        are what carry "this number was substituted". Mechanism 4 is never
        repaired: a corrupt weight must stay distinguishable from a merely
        degenerate one, so its statistics are published non-finite and the PCA
        drops that model's row instead.

        Args:
            weights: One weight tensor, exactly as returned by
                ``keras.layers.Layer.get_weights()``. Any numeric dtype is
                accepted; non-floating dtypes are analysed as ``float64``.

        Returns:
            The statistics dict, additionally carrying ``'status'`` (a
            :class:`~dl_techniques.analyzer.constants.StatusCode` value) and
            ``'degenerate_fields'`` (the dotted names of the leaves that were
            recomputed or substituted). ``None`` when the tensor is empty, in
            which case the caller must skip it entirely -- no statistic of an
            empty tensor is defined.
        """
        if weights.size == 0:
            return None

        # DECISION plan-2026-09-02T062406-e2aa52ef/D-001
        # Statistics are computed in the tensor's OWN dtype first and float64 is
        # used only to repair a leaf that came out non-finite. Do NOT "simplify"
        # this by upcasting unconditionally: that moves `l1`/`l2`/`rms`/`mean`/
        # `std` in the last ULPs for every ordinary float32 kernel, which shifts
        # the published PCA coordinates of models that have nothing wrong with
        # them. Non-floating dtypes (int/bool masks and counters) are the one
        # exception -- `scipy.stats.skew` raises on them -- and their float64
        # promotion is exact. See decisions.md D-001.
        if not np.issubdtype(weights.dtype, np.floating):
            weights = np.asarray(weights, dtype=np.float64)

        stats, muffled = self._raw_statistics_quietly(weights)

        # Key on `np.isfinite` of the RESULT. "Is the tensor constant" is the
        # WRONG predicate: mechanism 3 is a non-constant tensor.
        non_finite = [leaf for leaf in _STAT_LEAVES
                      if not self._leaf_is_finite(stats, leaf)]
        if not non_finite:
            # Nothing was classified, so nothing RECORDS whatever numpy or
            # scipy just complained about -- it must not be swallowed.
            self._reemit(muffled, runtime_warnings_too=True)
            stats['status'] = StatusCode.SUCCESS.value
            stats['degenerate_fields'] = []
            return stats

        # From here on the outcome IS recorded, in `status` and
        # `degenerate_fields`. A `RuntimeWarning` saying the same thing less
        # precisely is dropped; anything else still gets through.
        self._reemit(muffled, runtime_warnings_too=False)

        if not bool(np.isfinite(weights).all()):
            # Mechanism 4: the corruption is in the model, not in our arithmetic.
            stats['status'] = StatusCode.WEIGHT_NON_FINITE.value
            stats['degenerate_fields'] = [f'{g}.{k}' for g, k in non_finite]
            return stats

        repaired, _ = self._raw_statistics_quietly(
            np.asarray(weights, dtype=np.float64))
        for group, key in non_finite:
            if self._leaf_is_finite(repaired, (group, key)):
                stats[group][key] = repaired[group][key]
            elif (group, key) in _SUBSTITUTABLE_LEAVES:
                stats[group][key] = 0.0

        stats['status'] = StatusCode.WEIGHT_DEGENERATE.value
        stats['degenerate_fields'] = [f'{g}.{k}' for g, k in non_finite]
        return stats

    # DECISION plan-2026-09-02T062406-e2aa52ef/D-007
    # This pair exists so the degenerate path can be QUIET without going deaf.
    # Do NOT replace it with a blanket `warnings.simplefilter("ignore")` or a
    # module-level filter: the muffling is only justified because `status` and
    # `degenerate_fields` already carry the same information, which is true for
    # a `RuntimeWarning` on a classified-degenerate tensor and for nothing else.
    # Every other warning, and every warning at all when the statistics come
    # out clean, is re-emitted at its original location. See decisions.md D-007.
    @staticmethod
    def _raw_statistics_quietly(
            weights: np.ndarray
    ) -> Tuple[Dict[str, Any], List[warnings.WarningMessage]]:
        """Run :meth:`_raw_statistics` with its warnings captured, not printed.

        A constant or overflowing tensor makes ``scipy.stats.skew`` emit
        ``Precision loss occurred in moment calculation`` and ``np.sum(w ** 2)``
        emit ``overflow encountered in square``. Both are correct and both are
        already reported through ``status``/``degenerate_fields``, so printing
        them once per weight tensor of every ordinary model is noise the reader
        cannot act on.

        The capture is scoped to this one call. ``warnings.catch_warnings`` is
        process-global while it is open, so this must not be widened to cover
        anything else, and it is not thread-safe -- the analyzer is
        single-threaded by construction.

        Args:
            weights: A non-empty numeric weight tensor.

        Returns:
            ``(stats, muffled)`` -- the statistics dict, and every warning
            raised while computing it, for the caller to re-emit or drop.
        """
        with warnings.catch_warnings(record=True) as muffled:
            warnings.simplefilter('always')
            stats = WeightAnalyzer._raw_statistics(weights)
        return stats, list(muffled)

    @staticmethod
    def _reemit(
            muffled: List[warnings.WarningMessage],
            runtime_warnings_too: bool,
    ) -> None:
        """Re-raise captured warnings at their original location.

        Args:
            muffled: The records returned by :meth:`_raw_statistics_quietly`.
            runtime_warnings_too: When ``False``, ``RuntimeWarning`` records are
                dropped because the statistics were classified and the
                classification already reports them. Every other category is
                re-emitted regardless.
        """
        for record in muffled:
            if (not runtime_warnings_too
                    and issubclass(record.category, RuntimeWarning)):
                continue
            warnings.warn_explicit(
                record.message, record.category, record.filename, record.lineno)

    @staticmethod
    def _leaf_is_finite(stats: Dict[str, Any], leaf: Tuple[str, str]) -> bool:
        """Return whether ``stats[group][key]`` exists and is finite.

        A missing leaf counts as finite: ``norms.spectral`` is only computed for
        rank-2 tensors, and its absence is not a defect.

        Args:
            stats: A statistics dict from :meth:`_raw_statistics`.
            leaf: The ``(group, key)`` pair to inspect.

        Returns:
            ``True`` when the leaf is absent or holds a finite float.
        """
        group, key = leaf
        if key not in stats.get(group, {}):
            return True
        return bool(np.isfinite(stats[group][key]))

    @staticmethod
    def _raw_statistics(weights: np.ndarray) -> Dict[str, Any]:
        """Compute the statistics in the dtype of ``weights``, without classifying.

        This is the arithmetic half of :meth:`_compute_weight_statistics`, split
        out so the caller can run it twice -- once at native precision, once at
        ``float64`` -- and repair only the leaves that need it. It never raises
        for a non-empty numeric tensor.

        Args:
            weights: A non-empty numeric weight tensor.

        Returns:
            The statistics dict, whose leaves may be non-finite.
        """
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
            # DECISION plan-2026-09-02T062406-e2aa52ef/D-006
            # Do NOT drop this pre-check "because LAPACK handles it anyway".
            # It does -- `np.linalg.norm(x, 2)` returns `nan` for a non-finite
            # matrix and the classifier picks that up -- but on the way it
            # writes `** On entry to DLASCL parameter number 4 had an illegal
            # value` to RAW STDERR from Fortran, where no logger can filter,
            # prefix or attribute it. The published value is IDENTICAL either
            # way (`nan`); this branch only suppresses unattributable noise.
            # An O(n) `isfinite` scan is far cheaper than the SVD it replaces.
            # See decisions.md D-006.
            if not bool(np.isfinite(weights).all()):
                stats['norms']['spectral'] = float('nan')
            else:
                try:
                    stats['norms']['spectral'] = float(
                        np.linalg.norm(weights, 2))
                except (np.linalg.LinAlgError, ValueError, TypeError):
                    # MEASURED: `float16` raises `TypeError: array type float16
                    # is unsupported in linalg`, which the original
                    # `except np.linalg.LinAlgError` could not catch.
                    stats['norms']['spectral'] = float('nan')

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

            model_features, labels = self._drop_non_finite_rows(
                model_features, labels, results)
            if len(model_features) < 2:
                logger.warning(
                    "Skipping weight PCA: fewer than 2 models have fully finite "
                    f"weight statistics ({len(model_features)} left after "
                    "dropping non-finite rows). The per-layer statistics are "
                    "unaffected; read their `status` field to see why."
                )
                return

            try:
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(model_features)

                # Perform PCA
                pca = PCA(n_components=min(3, len(features_scaled)))
                pca_result = pca.fit_transform(features_scaled)

                results.weight_pca = {
                    'components': pca_result,
                    'explained_variance': self._finite_explained_variance(
                        pca.explained_variance_ratio_, labels),
                    'labels': labels,
                    'feature_type': 'concatenated_weight_statistics'
                }

                logger.info(f"PCA performed on weight statistics: {len(model_features[0])} features per model")

            except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                # MEASURED: sklearn raises a bare `ValueError` for a non-finite
                # feature matrix, and a DIFFERENT message for `inf` than for
                # `NaN`, from a different frame. Never key this on the message
                # text -- catch the type, and let the row-dropping above be what
                # prevents the condition in the first place.
                logger.warning(f"Could not perform PCA on weight statistics: {e}")

    @staticmethod
    def _finite_explained_variance(
            explained_variance_ratio: np.ndarray,
            labels: List[str],
    ) -> Optional[np.ndarray]:
        """Return the explained-variance ratios, or ``None`` when they are undefined.

        When every model's feature row is IDENTICAL -- the natural
        self-comparison smoke run, or any two copies of one architecture at one
        seed -- the standardized total variance is exactly zero and sklearn
        computes ``0 / 0``, publishing ``[nan, nan]`` WITHOUT raising. Nothing in
        the surrounding exception handling can see that, and it is fatal one step
        later: ``save_results`` uses ``json.dump(..., allow_nan=False)``, so the
        write aborts part-way and leaves a TRUNCATED, unparseable
        ``analysis_results.json`` behind.

        ``None`` is the package's established "not computed" encoding, and it is
        honest here: the fraction of variance explained is genuinely undefined
        when there is no variance to explain.

        Args:
            explained_variance_ratio: sklearn's ``PCA.explained_variance_ratio_``.
            labels: The model names in the PCA, used only for the warning.

        Returns:
            The ratios unchanged when every entry is finite, otherwise ``None``.
        """
        ratios = np.asarray(explained_variance_ratio, dtype=np.float64)
        if bool(np.isfinite(ratios).all()):
            return explained_variance_ratio

        logger.warning(
            "Weight-PCA explained variance is undefined and is reported as None: "
            "the %d models (%s) have identical weight statistics, so the total "
            "variance in the fingerprint space is exactly zero. The component "
            "coordinates are still reported.",
            len(labels), ", ".join(labels),
        )
        return None

    @staticmethod
    def _drop_non_finite_rows(
            model_features: List[List[float]],
            labels: List[str],
            results: AnalysisResults,
    ) -> Tuple[List[List[float]], List[str]]:
        """Remove any model whose feature row is not entirely finite.

        A model with a corrupt weight tensor must not be able to take the whole
        PCA -- and, before the surrounding handler was widened, the whole
        ``analyze()`` call -- down with it. The dropped models are NAMED in a
        warning together with the weight tensors responsible, because a silently
        missing point in the model-similarity panel is indistinguishable from a
        model that was never analysed.

        Imputation is deliberately NOT offered: replacing a corrupt model's
        statistics with zeros or column means would place it at a plausible
        position in the fingerprint space, which is the exact failure this
        classification exists to prevent.

        Args:
            model_features: One concatenated feature row per model.
            labels: The model names, positionally aligned with ``model_features``.
            results: The results object, read to name the offending weights.

        Returns:
            The surviving ``(model_features, labels)`` pair, in the original order.
        """
        keep_features: List[List[float]] = []
        keep_labels: List[str] = []
        dropped: List[str] = []

        for features, label in zip(model_features, labels):
            if bool(np.isfinite(np.asarray(features, dtype=np.float64)).all()):
                keep_features.append(features)
                keep_labels.append(label)
                continue

            offenders = [
                name for name, stats in results.weight_stats.get(label, {}).items()
                if stats.get('status') != StatusCode.SUCCESS.value
            ]
            dropped.append(
                f"{label} (weights: {', '.join(offenders) or 'unidentified'})")

        if dropped:
            logger.warning(
                "Dropping %d model(s) from the weight PCA because their "
                "statistics are not finite: %s. Their per-layer statistics are "
                "still reported -- read the `status` field of each weight to "
                "see which mechanism fired.",
                len(dropped), "; ".join(dropped),
            )

        return keep_features, keep_labels