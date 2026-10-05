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
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseAnalyzer
from ..data_types import AnalysisResults, DataInput
from ..constants import (
    LayerType, MetricNames, StatusCode, SPECTRAL_DEFAULT_SUMMARY_METRICS,
    SPECTRAL_HIGH_CONCENTRATION_ABSOLUTE, SPECTRAL_WEAK_RANK_LOSS_TOLERANCE,
    SPECTRAL_PVALUE_NOT_COMPUTED, SPECTRAL_TRAP_SEVERITY_MILD
)
from .. import spectral_metrics
from .. import spectral_utils
from ..utils import make_rng, recursively_get_layers
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
        results.spectral_summary_per_model = {}

        # DECISION plan-2026-10-05-analyzer-audit/F-009
        # ONE generator for the WHOLE PASS, built here and threaded down -- not
        # `make_rng(self.config.random_state)` re-evaluated per model.
        #
        # D-031's own stated invariant is "one generator per analysis pass, not one per
        # layer", with the correct reason: a generator rebuilt from the same seed hands
        # every layer the identical permutation stream. The same defect survived one
        # level up: at a fixed `random_state`, model A's layer 0 and model B's layer 0
        # drew BYTE-IDENTICAL permutations and identical bootstrap streams, because
        # each `_analyze_single_model` call constructed its own generator from the same
        # seed. In a cross-model comparison that is correlated draws presented as
        # independent evidence, and it was structurally invisible to the suite because
        # every spectral reproducibility test uses a SINGLE model.
        #
        # Passing one generator down also keeps layers within a model distinct (each
        # consumes from the same advancing stream) while making distinct models distinct
        # too. Do NOT "simplify" this back to a per-model `make_rng`.
        pass_rng = make_rng(self.config.random_state)

        for model_name, model in self.models.items():
            logger.info(f"Starting spectral analysis for model: {model.name}")
            (details_df, esds, rand_esds,
             recommendations, model_summary) = self._analyze_single_model(
                 model, rng=pass_rng)

            if not details_df.empty:
                details_df['model_name'] = model_name
                all_model_details.append(details_df)

                # DECISION plan-2026-09-01T225724-e79ad4bd/D-020
                # These artifacts arrive in the RETURN VALUE. Do NOT park them on
                # `self` and recover them with `hasattr`: the previous shape set the
                # attributes unconditionally at the top of `_analyze_single_model`, so
                # every `hasattr` guard here was true by construction and could never
                # fire. See decisions.md D-020.
                results.spectral_esds[model_name] = esds
                if rand_esds:
                    results.spectral_rand_esds[model_name] = rand_esds
                results.spectral_recommendations[model_name] = recommendations
                results.spectral_summary_per_model[model_name] = model_summary

        if all_model_details:
            # Consolidate results into a single DataFrame
            results.spectral_analysis = pd.concat(all_model_details, ignore_index=True)

            # Compute and store summary metrics. NOTE this aggregate mixes the layers of
            # every model; `results.spectral_summary_per_model` keeps them separate.
            results.spectral_summary = self._get_summary(results.spectral_analysis)
        else:
            logger.warning("Spectral analysis did not produce any results for any model.")

    def _analyze_single_model(
            self, model: keras.Model,
            rng: Optional[np.random.Generator] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, np.ndarray],
               List[str], Dict[str, float]]:
        """
        Perform spectral analysis on a single Keras model.

        Args:
            model: The Keras model to analyze.
            rng: Generator for every stochastic step of this model's layers. It MUST be
                the pass-level generator threaded down by :meth:`analyze` (F-009), not
                one built here from ``config.random_state``: a per-model generator gives
                two models byte-identical permutations at a fixed seed. ``None`` builds
                an unseeded one, which is the historical behaviour.

        Returns:
            Tuple of ``(details, esds, rand_esds, recommendations, summary)``. On a
            model with no analyzable layers ``details`` is empty and the remaining
            artifacts are empty containers.
        """
        if rng is None:
            rng = make_rng(self.config.random_state)
        # Create basic description of the model to find analyzable layers
        # This now returns both the details DataFrame and the flattened list of layers.
        details, all_layers = self._describe_model(model)
        esds: Dict[int, np.ndarray] = {}
        rand_esds: Dict[int, np.ndarray] = {}

        if details.empty:
            logger.warning(f"No layers found in model '{model.name}' that meet the criteria for spectral analysis.")
            # Return empty but correctly formatted DataFrame
            return details.reset_index(), esds, rand_esds, [], {}

        # Perform detailed analysis on each qualifying layer.
        #
        # DECISION plan-2026-09-01T225724-e79ad4bd/D-012
        # These two filters are NARROW ON PURPOSE. Do NOT collapse them back into a
        # blanket `simplefilter("ignore", category=RuntimeWarning)`: that form also
        # swallowed `overflow encountered in power`, which is exactly how a
        # `log_alpha_norm = inf` reached the published details frame unnoticed.
        # Only the two benign, expected numeric conditions in the eigenvalue path
        # are silenced; overflow stays audible. See decisions.md D-012.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered",
                category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore", message="invalid value encountered",
                category=RuntimeWarning)
            # DECISION plan-2026-09-01T225724-e79ad4bd/D-031
            # ONE generator per analysis pass, not one per layer: a per-layer
            # generator built from the same seed would hand every layer the identical
            # permutation stream, and the `spectral_n_randomizations` draws within a
            # layer must differ from each other (D-017). See decisions.md D-031.
            # F-009 extends the same invariant one level up, to the MODEL: `rng` is the
            # pass-level generator from `analyze`, so two models no longer draw
            # byte-identical permutations at a fixed seed.
            self._analyze_layers(details, all_layers, esds, rand_esds, rng=rng)

        # Generate the per-model summary and recommendations. The summary is kept (not
        # discarded after the recommendations) so callers can read a per-model figure
        # instead of only the cross-model aggregate.
        model_summary = self._get_summary(details)
        recommendations = self._generate_recommendations(details, model_summary)

        # Convert layer_id index to a column for robust concatenation
        return (details.reset_index(), esds, rand_esds, recommendations,
                model_summary)

    def _analyze_layers(
            self,
            details: pd.DataFrame,
            all_layers: List[keras.layers.Layer],
            esds: Dict[int, np.ndarray],
            rand_esds: Dict[int, np.ndarray],
            rng: np.random.Generator,
    ) -> None:
        """Perform spectral analysis on each qualifying layer.

        Args:
            details: Per-layer frame, mutated in place with the computed metrics.
            all_layers: Flattened layer list, indexed by ``details``' index.
            esds: Output mapping ``layer_id -> eigenvalue spectrum``, filled in place.
            rand_esds: Output mapping ``layer_id -> randomized spectrum``, filled in
                place; left empty unless ``config.spectral_randomize`` is set.
            rng: Generator shared by every stochastic step of this pass - the
                randomization permutations and the goodness-of-fit bootstrap.
        """
        for layer_id in details.index:
            layer = all_layers[layer_id]

            logger.debug(f"Analyzing layer {layer_id}: {layer.name}")

            # Extract weights, dimensions, and other properties
            has_weights, weights, _, _ = spectral_utils.get_layer_weights_and_bias(layer)
            if not has_weights: continue

            layer_type = spectral_utils.infer_layer_type(layer)
            Wmats, N, M, rf = spectral_utils.get_weight_matrices(weights, layer_type)

            if self.config.spectral_glorot_fix and Wmats:
                # DECISION plan-2026-09-01T225724-e79ad4bd/D-035
                # Pass the ORDERED matricized shape, never `(N, M)`: those are
                # `max`/`min` and so lose which axis is `fan_in`, which made
                # `kappa` up to 1.4015x too large for every conv whose
                # `out_c > kh*kw*in_c` (the canonical `(3,3,3,64)` first conv).
                # See decisions.md D-035.
                kappa = spectral_metrics.calculate_glorot_normalization_factor(
                    Wmats[0].shape, rf)
                Wmats = [W / kappa for W in Wmats]

            # DECISION plan-2026-09-01T225724-e79ad4bd/D-002
            # `n_comp` is M, NOT `M * rf`. `get_weight_matrices` returns ONE flattened
            # matrix per conv layer, which has exactly M singular values; `M * rf` is
            # WeightWatcher's count for ITS per-slice decomposition, which this package
            # deliberately does not use. The old spelling made `n_comp >= M` always
            # true, so the truncated `svds` branch in `compute_eigenvalues` was
            # structurally unreachable and the `sv[:n_comp]` slices were no-ops.
            # See decisions.md D-002.
            n_comp = M
            (evals, sv_max, sv_min, rank_loss,
             spectrum_truncated) = spectral_metrics.compute_eigenvalues(
                Wmats, N, M, n_comp, max_evals=self.config.spectral_max_evals)
            esds[layer_id] = evals

            # DECISION plan-2026-09-01T225724-e79ad4bd/D-027
            # The eval-count bounds this analyzer admits layers on (`_get_layer_
            # details` gates M against `spectral_min_evals` / `spectral_max_evals`)
            # are the SAME bounds the kernels must apply. Do NOT drop these keyword
            # arguments: without them the kernels fall back to `SPECTRAL_DEFAULT_
            # MIN_EVALS` (10) and `SPECTRAL_DEFAULT_MAX_EVALS` (15000), and a
            # lowered `spectral_min_evals` admits layers that `fit_powerlaw` then
            # refuses with status='failed'. See decisions.md D-027.
            alpha, xmin, D, sigma, num_pl_spikes, status, warning = spectral_metrics.fit_powerlaw(
                evals, min_evals=self.config.spectral_min_evals)

            # DECISION plan-2026-09-01T225724-e79ad4bd/D-019
            # On a truncated spectrum the small singular values were never computed,
            # so every quantity that counts or integrates over the WHOLE spectrum is
            # unknowable. Emit NaN, never 0: a `weak_rank_loss` of 0 and an entropy
            # of "1.0, perfectly flat" are the answers a complete healthy spectrum
            # gives, so the failure would be indistinguishable from success.
            # See decisions.md D-019.
            if spectrum_truncated:
                weak_rank_loss = float('nan')
                entropy = float('nan')
                matrix_rank = float('nan')
            else:
                # DECISION plan-2026-10-05-analyzer-audit/F-011
                # `weak_rank_loss` is a RELATIVE count, not an absolute one. The old
                # `np.sum(evals < 1e-6)` is an ABSOLUTE threshold on raw eigenvalues, so
                # every small-magnitude layer (a Glorot-divided conv, a low-scale init, a
                # mean-centred kernel) reported `weak_rank_loss ≈ M` — a full-rank
                # collapse — purely because of its units. Scale the tolerance by the
                # layer's own largest eigenvalue: the same relative form `rank_loss`
                # already uses in `compute_eigenvalues` (D-003). The neighbouring
                # `rank_loss` is compared against `0.1 * M` in `_generate_recommendations`,
                # so with an absolute tolerance the two rank columns sat on incomparable
                # scales. Do NOT restore the absolute form: no WeightWatcher column
                # corresponds to this quantity, so there is no parity to match.
                eval_tol = (float(np.max(evals)) * SPECTRAL_WEAK_RANK_LOSS_TOLERANCE
                            if len(evals) > 0 else 0.0)
                weak_rank_loss = float(np.sum(evals < eval_tol))
                # DECISION plan-2026-10-05-analyzer-audit/F-015
                # `evals` already holds sigma^2, so `calculate_matrix_entropy` was
                # handed `sqrt(evals)` and squared it straight back — a lossy round trip
                # plus two full-size temporaries per layer. Compute from `evals` directly.
                entropy = spectral_metrics.calculate_matrix_entropy_from_evals(evals, N)
                matrix_rank = int(len(evals) - rank_loss)
            spectral_mets = spectral_metrics.calculate_spectral_metrics(evals, alpha, N=N)

            # DECISION plan-2026-10-05-analyzer-audit/F-078
            # COMPLETING D-019. The truncated branch above NaN-ed `weak_rank_loss`,
            # `entropy` and `matrix_rank`, but three of the quantities `calculate_spectral_
            # metrics` returns INTEGRATE over the whole spectrum and were still published
            # as ordinary numbers:
            #
            #   norm             = Σ λ           the missing tail is mass the sum never saw
            #   log_norm         = log10(Σ λ)    ditto
            #   stable_rank      = (Σ λ)/max(λ)  a strict UNDER-estimate: the mass it
            #                                   integrates over is exactly what is missing
            #   log_alpha_norm   = log10 Σ λ^α   integrates the entire spectrum
            #
            # `stable_rank` is the damaging one: it is in
            # `SPECTRAL_DEFAULT_SUMMARY_METRICS`, so it was averaged into
            # `spectral_summary` and drawn on the dashboard as a real capacity-utilisation
            # number. The D-019 rationale already quoted at the branch above -- "every
            # quantity that counts or integrates over the WHOLE spectrum is unknowable" --
            # is the argument that condemns these four too.
            #
            # DELIBERATELY NOT NaN-ed, because they are KNOWN on a truncated spectrum:
            # `spectral_norm` and `log_spectral_norm` (the largest singular value is
            # exactly what a truncated SVD returns), `alpha_weighted`/`alpha_hat`/
            # `alpha_hat_normalized` (all derived from that same λ_max), `alpha_unreliable`,
            # `num_evals` and `sv_max`. See D-019 and decisions.md D-019.
            if spectrum_truncated:
                for key in (MetricNames.NORM, MetricNames.LOG_NORM,
                            MetricNames.STABLE_RANK,
                            MetricNames.LOG_ALPHA_NORM):
                    if key in spectral_mets:
                        spectral_mets[key] = float('nan')
                logger.debug(
                    f"Layer {layer_id} ({layer.name}): spectrum truncated, so the "
                    f"whole-spectrum integrals (norm, log_norm, stable_rank, "
                    f"log_alpha_norm) are NaN. Mass-based columns derived from λ_max "
                    f"(spectral_norm, alpha_weighted, alpha_hat) remain valid.")

            # SETOL: Learning phase classification
            learning_phase = spectral_metrics.classify_learning_phase(alpha)

            # SETOL: ERG condition (Δλ_min diagnostic)
            erg_metrics = {}
            if status == "success" and xmin > 0:
                erg_metrics = spectral_metrics.compute_erg_condition(evals, xmin)

            # SETOL: Goodness-of-fit p-value (Clauset et al. 2009)
            pl_pvalue = SPECTRAL_PVALUE_NOT_COMPUTED
            if status == "success" and alpha > 1.0:
                # Pass the KS distance of the fit we are actually reporting; letting the
                # test refit would re-search xmin inside the tail and test a different fit.
                pl_pvalue = spectral_metrics.powerlaw_goodness_of_fit(
                    evals, alpha, xmin, n_bootstraps=self.config.spectral_bootstraps,
                    d_observed=D, rng=rng)

            concentration_metrics = {}
            if self.config.spectral_concentration_analysis and Wmats:
                concentration_metrics = spectral_metrics.calculate_concentration_metrics(
                    Wmats[0], evals=evals)
                # DECISION plan-2026-10-05-analyzer-audit/F-078
                # `gini_coefficient`, `dominance_ratio`, `participation_ratio` and
                # `min_participation_ratio` are functions of the eigenvalue DISTRIBUTION
                # over the whole spectrum: the Gini is a Lorenz-curve quantity, the
                # dominance ratio divides by the sum of the REST, and participation
                # divides by the total. On a truncated spectrum the small eigenvalues were
                # never computed, so each is computed over a sub-distribution and is not
                # a smaller version of the truth -- it is a different quantity that looks
                # like the published one.
                #
                # `concentration_score` is derived FROM the participation ratio and
                # `critical_weight_count` from the weight matrix itself, so the count
                # stays valid; only the distribution-derived ratios are NaN-ed.
                # See decisions.md D-019.
                if spectrum_truncated:
                    for key in (MetricNames.GINI_COEFFICIENT,
                                MetricNames.DOMINANCE_RATIO,
                                MetricNames.PARTICIPATION_RATIO,
                                MetricNames.MIN_PARTICIPATION_RATIO):
                        if key in concentration_metrics:
                            concentration_metrics[key] = float('nan')
                    logger.debug(
                        f"Layer {layer_id} ({layer.name}): spectrum truncated, so the "
                        f"distribution-derived concentration ratios are NaN. The weight "
                        f"matrix itself was read in full, so concentration_score and "
                        f"critical_weight_count remain valid.")

            randomization_metrics = {}
            if self.config.spectral_randomize and Wmats:
                # DECISION plan-2026-09-01T225724-e79ad4bd/D-017
                # Randomization runs `spectral_n_randomizations` independent draws. Do
                # NOT collapse this back to one permutation per layer: a single unseeded
                # draw makes every correlation-trap verdict a coin flip. See D-017.
                #
                # DECISION plan-2026-10-05-analyzer-audit/F-033
                # The draws are still all RUN, but the published row now describes ONE
                # of them — the worst-severity draw — instead of a blend. The previous
                # shape was `has_trap` = majority vote, `trap_severity`/`trap_threshold`/
                # `num_rand_spikes`/the MP edges = MEANS, and the plotted spectrum =
                # draw #1. Four published quantities therefore described four different
                # permutations, and they could contradict each other: one trap in five
                # draws at severity 2.0 gave mean severity 0.4 -> label 'moderate' while
                # `has_trap` was False, and the overlay titled that layer "Clean" while
                # carrying a 'moderate' label.
                #
                # Every trap column below, plus the spectrum stored in `rand_esds` and
                # therefore every spike marker the overlay draws, now come from the SAME
                # draw. The randomization-only DIAGNOSTICS (rand_sv_max, rand_distance,
                # rand_sv_ratio) stay means — they are not trap verdicts, they have no
                # threshold, and averaging them is the point. See decisions.md D-017
                # and the F-033 anchor in README.md.
                n_draws = max(1, int(self.config.spectral_n_randomizations))
                draws = []

                for _ in range(n_draws):
                    rand_Wmats = [
                        rng.permutation(W.flatten()).reshape(W.shape)
                        for W in Wmats
                    ]
                    rand_evals, rand_sv_max, _, _, _ = spectral_metrics.compute_eigenvalues(
                        rand_Wmats, N, M, n_comp,
                        max_evals=self.config.spectral_max_evals)

                    trap_result = spectral_metrics.detect_correlation_trap(rand_evals, N, M)
                    draws.append({
                        'rand_evals': rand_evals,
                        'rand_sv_max': rand_sv_max,
                        'rand_distance': spectral_metrics.jensen_shannon_distance(
                            evals, rand_evals),
                        # Randomization singular-value ratio (NOT WW's mp_softrank, which
                        # is λ_plus/λ_max). Re-keyed off the former softrank name (D-006).
                        'rand_sv_ratio': (
                            np.max(rand_evals) / np.max(evals)
                            if np.max(evals) > 0 else 0
                        ),
                        'trap': trap_result,
                    })

                def _mean(key: str) -> float:
                    return float(np.nanmean([d[key] for d in draws]))

                # F-033: pick the ONE draw the published row describes. A trapping draw
                # always wins over a non-trapping one (its severity is > 0 by
                # construction); among trapping draws the most severe wins. With
                # `has_trap = any()`, this is the draw that justifies the verdict, so
                # the row can never report a verdict its own severity contradicts.
                representative = max(
                    draws,
                    key=lambda d: (bool(d['trap']['has_trap']),
                                   float(d['trap']['trap_severity'])),
                )
                trap = representative['trap']

                has_trap = any(bool(d['trap']['has_trap']) for d in draws)
                severity = float(trap['trap_severity'])
                severity_label = spectral_metrics.label_trap_severity(severity)

                # DECISION plan-2026-10-05-analyzer-audit/F-034
                # Floor the LABEL at 'mild' when a trap was detected, without touching
                # the severity NUMBER. `label_trap_severity` returns 'none' below 0.1, so
                # a genuine detection just over the threshold published `has_trap=True`
                # beside `trap_severity_label='none'` — the boolean and the band
                # contradicting each other in the same row, and the overlay printing
                # "TRAP (none)". Only the human-readable band is floored; `trap_severity`
                # stays exactly as computed, so nothing measured is laundered.
                if has_trap and severity_label == 'none':
                    severity_label = 'mild'
                    logger.debug(
                        f"Layer {layer_id}: a trap was detected with severity "
                        f"{severity:.4f} (< {SPECTRAL_TRAP_SEVERITY_MILD}), so the "
                        f"severity LABEL is floored to 'mild'; the severity NUMBER is "
                        f"reported unchanged.")

                randomization_metrics = {
                    MetricNames.RAND_SV_MAX: _mean('rand_sv_max'),
                    MetricNames.RAND_DISTANCE: _mean('rand_distance'),
                    MetricNames.RAND_SV_RATIO: _mean('rand_sv_ratio'),
                    # F-033: `any()`, not a majority vote. A trap present in ANY
                    # permutation is evidence the layer holds atypically large weight
                    # elements; requiring a majority silently discarded a real
                    # single-draw detection. The reported severity is the WORST draw's,
                    # so `has_trap` is consistent with what the row shows.
                    MetricNames.HAS_TRAP: has_trap,
                    # F-033: an int again, from ONE draw, rather than a fractional mean
                    # over draws that the overlay printed as "2.4 spike(s)".
                    MetricNames.NUM_RAND_SPIKES: int(trap['num_rand_spikes']),
                    MetricNames.TRAP_SEVERITY: severity,
                    MetricNames.TRAP_SEVERITY_LABEL: severity_label,
                    MetricNames.MP_LAMBDA_PLUS: float(trap['mp_lambda_plus']),
                    MetricNames.MP_LAMBDA_MINUS: float(trap['mp_lambda_minus']),
                    MetricNames.TRAP_THRESHOLD: float(trap['trap_threshold']),
                }

                # Store the REPRESENTATIVE randomized spectrum, so the markers the
                # overlay draws and the threshold it draws them against are the same
                # permutation's (F-033).
                rand_esds[layer_id] = representative['rand_evals']

            metrics = {
                MetricNames.HAS_ESD: True, MetricNames.NUM_EVALS: len(evals),
                MetricNames.SV_MAX: sv_max, MetricNames.SV_MIN: sv_min,
                MetricNames.RANK_LOSS: rank_loss, MetricNames.WEAK_RANK_LOSS: weak_rank_loss,
                MetricNames.SPECTRUM_TRUNCATED: spectrum_truncated,
                # DECISION plan-2026-09-01T225724-e79ad4bd/D-003
                # Effective rank = (#evals) - rank_loss, with rank_loss computed in
                # compute_eigenvalues using the tolerance max_sv * max(shape) * eps.
                # The SHAPE of that tolerance is WeightWatcher's RMT_Util.matrix_rank,
                # but the eps is taken from the SOURCE dtype (float32 for Keras weights)
                # rather than from the float64 cast WW reads it from. Do NOT restore the
                # post-cast eps to claim parity: it makes exact rank deficiency invisible.
                # This is a documented divergence - see decisions.md D-003.
                MetricNames.MATRIX_RANK: matrix_rank,
                MetricNames.LAMBDA_MAX: np.max(evals) if len(evals) > 0 else 0,
                MetricNames.ALPHA: alpha, MetricNames.XMIN: xmin, MetricNames.D: D,
                MetricNames.SIGMA: sigma, MetricNames.NUM_PL_SPIKES: num_pl_spikes,
                MetricNames.STATUS: status, MetricNames.WARNING: warning,
                MetricNames.ENTROPY: entropy,
                # DECISION plan-2026-10-05-analyzer-audit/F-078
                # `mp_softrank` is `lambda_plus / lambda_max`, and `lambda_plus` comes
                # from the MEAN of the whole spectrum. On a truncated spectrum that mean
                # is a mean over a sub-distribution, so the "theoretical MP edge" it
                # compares against is not the layer's theoretical edge at all -- the
                # column kept a plausible number describing a different question. NaN.
                # `spectral_norm` / `log_spectral_norm` / `lambda_max` are NOT NaN-ed:
                # the largest singular value is exactly what a truncated SVD still
                # returns (see `compute_eigenvalues`), so those remain valid.
                MetricNames.MP_SOFTRANK: (
                    float('nan') if spectrum_truncated
                    else spectral_metrics.calc_mp_soft_rank(evals, N, M)),
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
                # M, not M*rf — see the D-002 anchor above. This value is overwritten
                # by len(evals) in _analyze_layers; it is a placeholder label only.
                MetricNames.NUM_EVALS: M
            })

        details = pd.DataFrame(rows)
        if not details.empty:
            details.set_index('layer_id', inplace=True)

        return details, all_layers

    def _get_summary(self, details_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary metrics averaged across the SUCCESSFULLY analyzed layers.

        Layers whose power-law fit failed carry sentinel values (``alpha = -1.0``, and a
        derived ``alpha_weighted`` of ``+10.0``) rather than missing values, so they are
        numerically indistinguishable from a real fit once averaged. They are therefore
        excluded here, and their count is reported alongside the means.

        Args:
            details_df: The per-layer spectral details frame. When it carries a
                ``status`` column, only ``StatusCode.SUCCESS`` rows are aggregated.

        Returns:
            Dictionary of summary metrics. ``total_layers_analyzed`` keeps its published
            meaning (every described layer); ``successful_layers_analyzed`` and
            ``failed_layers`` partition it.
        """
        summary = {}
        if details_df is None or details_df.empty:
            return summary

        # DECISION plan-2026-09-01T225724-e79ad4bd/D-008
        # Exclude failed fits by STATUS, not by dropping the -1.0 sentinel value. Do NOT
        # "simplify" this to a `!= -1.0` value filter: -1.0 is a legitimate value for some
        # summary metrics (e.g. Q, log-norms), and alpha_weighted's failure sentinel is
        # +10.0, not -1.0, so a value filter would silently keep the worst offender. See
        # decisions.md D-008.
        if MetricNames.STATUS in details_df.columns:
            success_df = details_df[details_df[MetricNames.STATUS] == StatusCode.SUCCESS.value]
        else:
            success_df = details_df

        metrics_to_summarize = [m for m in SPECTRAL_DEFAULT_SUMMARY_METRICS if m in success_df.columns]

        for metric in metrics_to_summarize:
            valid_values = success_df[metric][pd.to_numeric(success_df[metric], errors='coerce').notna()]
            if not valid_values.empty:
                summary[metric] = float(valid_values.mean())

        summary['total_layers_analyzed'] = len(details_df)
        summary['successful_layers_analyzed'] = len(success_df)
        summary['failed_layers'] = len(details_df) - len(success_df)
        if 'concentration_score' in success_df.columns:
            score = pd.to_numeric(success_df['concentration_score'], errors='coerce').dropna()
            if not score.empty:
                # DECISION plan-2026-10-05-analyzer-audit/F-002
                # An ABSOLUTE threshold, not this layer-set's own 80th percentile.
                #
                # Counting how many values exceed their own quantile is a tautology: it
                # returns `ceil(0.2 * n)` for EVERY input, so the published
                # "high_concentration_layers" was a constant function of the layer count
                # and could not fail — it measured nothing about concentration. It was
                # also self-contradictory with `_generate_recommendations`, which warns
                # on the SAME column against the absolute constant 5.0.
                #
                # `SPECTRAL_HIGH_CONCENTRATION_ABSOLUTE` is that same 5.0: the threshold
                # the recommendation already used, so the count and the warning now agree
                # by construction. README.md documents `concentration_score` as having no
                # absolute scale for RANKING layers — that remains true, and is why this
                # counts a deliberately conservative "worst-case" set rather than
                # pretending to a precise one.
                summary['high_concentration_layers'] = int(
                    (score > SPECTRAL_HIGH_CONCENTRATION_ABSOLUTE).sum())
                summary['concentration_score_threshold'] = float(
                    SPECTRAL_HIGH_CONCENTRATION_ABSOLUTE)

        return summary

    def _generate_recommendations(self, analysis_df: pd.DataFrame, summary: Dict[str, float]) -> List[str]:
        """
        Generate analysis-based recommendations for model optimization.
        """
        recommendations = []
        if 'alpha' in summary:
            mean_alpha = summary['alpha']
            if mean_alpha < 2.0:
                recommendations.append("Model may be over-trained (α < 2, very heavy-tailed). Consider reducing the learning rate and checking for correlation traps.")
            elif mean_alpha > 6.0:
                recommendations.append("Model may be under-trained (high α). Consider training longer or reducing regularization.")
            else:
                recommendations.append(f"Model training quality appears good (α = {mean_alpha:.2f}).")

        # WeightWatcher: Phase distribution analysis (over-trained / good / under-trained)
        if 'learning_phase' in analysis_df.columns:
            phase_counts = analysis_df['learning_phase'].value_counts()
            total = len(analysis_df)
            for phase, count in phase_counts.items():
                pct = 100 * count / total
                if phase == 'over-trained' and pct > 20:
                    recommendations.append(
                        f"{pct:.0f}% of layers are over-trained (α < 2.0). "
                        "Reduce learning rate or check for correlation traps.")
                elif phase == 'under-trained' and pct > 20:
                    recommendations.append(
                        f"{pct:.0f}% of layers are under-trained (α > 6.0). "
                        "Train longer or reduce regularization.")
                elif phase == 'good' and pct > 50:
                    recommendations.append(
                        f"{pct:.0f}% of layers are in the good phase (2.0 ≤ α ≤ 6.0). Healthy training quality.")

        # ERG condition check
        if 'erg_satisfied' in analysis_df.columns:
            # DECISION plan-2026-10-05-analyzer-audit/F-003
            # Count the TRUTHS explicitly; never `Series.sum()`. `erg_satisfied` is
            # written only for rows whose fit succeeded and whose `xmin > 0`, and
            # `details.at[...] = value` enlarges a new column with NaN for every other
            # row — so ONE failed fit anywhere makes the column a bool/NaN mix and
            # `.sum()` returns NaN. `NaN > 0` is False, so the recommendation
            # SILENTLY VANISHED (and had it fired it would have printed "nan/12").
            # `== True` is the count of satisfied layers; the denominator stays every
            # layer, which is the honest reading: a layer whose ERG could not be
            # evaluated did not satisfy it.
            erg_column = analysis_df['erg_satisfied']
            erg_satisfied = int((erg_column == True).sum())  # noqa: E712 - NaN-safe
            total = len(analysis_df)
            if erg_satisfied > 0:
                recommendations.append(
                    f"{erg_satisfied}/{total} layers satisfy the ERG condition (ln det X̃ ≈ 0).")

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