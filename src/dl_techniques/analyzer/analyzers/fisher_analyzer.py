"""
Measure the diagonal model Fisher information of every model in the analysis.

The Fisher information F_i = E_x E_{y~pi_theta}[(d_i log pi_theta(y|x))^2] says how
sensitive a model's OWN output distribution is to a perturbation of parameter i. It is
objective-independent (no reward, teacher or label enters) and, through a Cauchy-Schwarz
bound, limits how large the expected gradient of any post-training objective can be on
that coordinate: |g_i| <= sqrt(E[a^2]) * sqrt(F_i). This is the quantity arXiv:2609.36262
uses to explain why post-training updates concentrate in few parameters.

Architecture
------------
The analyzer is orchestration only; every estimator lives in ``fisher_metrics``. One
``analyze()`` call:

1.  builds ONE ``np.random.Generator`` from ``config.random_state`` and threads it through
    every stochastic site (F-009: per-model reconstruction from one seed would hand every
    model the identical draw stream);
2.  draws ONE input pool and ONE ``SamplePlan`` (Fisher rows disjoint from gradient rows)
    shared by all models, so cross-model Fisher comparisons are made on the same inputs;
3.  sizes the sampling microbatch ONCE for all models, so no model is measured at a
    different budget than another (D-037);
4.  runs the models serially (parallel passes would contaminate each other's memory and
    timing), isolating each: a numeric or shape failure logs and skips that model, while
    ``AttributeError`` / ``TypeError`` / ``NameError`` / ``ImportError`` re-raise because
    they are bugs, not measurements;
5.  optionally compares an ordered list of checkpoints of one architecture (persistence).

The pass is eager and read-only: per-sample scores need concrete tensors, so it cannot be
traced or jit-compiled, and trainable variables are hashed before and after.
"""

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseAnalyzer
from ..config import AnalysisConfig
from ..constants import FisherMetricNames as FM
from ..data_types import AnalysisResults, DataInput
from ..utils import DataSampler, make_rng
from .. import fisher_metrics as fm
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


class FisherAnalyzer(BaseAnalyzer):
    """Runs the Fisher pipeline inside the ``ModelAnalyzer`` orchestration."""

    def __init__(self, models: Dict[str, keras.Model], config: AnalysisConfig,
                 signal: Optional[fm.FisherSignal] = None) -> None:
        """
        Args:
            models: Model name to model.
            config: Analysis configuration.
            signal: Callables and models the gradient arm and custom likelihoods need.
                Not part of ``AnalysisConfig`` because ``save_results`` serializes every
                public config field and a callable would lose the artifact.
        """
        super().__init__(models, config)
        self.signal = signal

    def requires_data(self) -> bool:
        """Fisher is an expectation over inputs, so it needs data."""
        return True

    # ------------------------------------------------------------------

    def analyze(self, results: AnalysisResults, data: Optional[DataInput] = None,
                cache: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Compute the Fisher outputs and publish them on ``results``.

        The prediction ``cache`` is deliberately unused: the Fisher pass needs eager
        per-sample calls under a gradient tape, not batched predictions.

        Args:
            results: The only results sink.
            data: Input pool source; its targets feed the reward / demonstrations.
            cache: Unused.

        Raises:
            ValueError: On a configuration that cannot produce a measurement (checked
                before any model runs): invalid ``fisher_*`` values, G < 2 under a
                group-centered signal, a gradient arm without a signal, or a persistence
                list naming unknown models. Also when persistence checkpoints do not
                share one variable fingerprint.
        """
        fcfg = fm.FisherConfig.from_analysis_config(self.config)
        if fcfg.compute_gradients and self.signal is None:
            raise ValueError("fisher_compute_gradients=True needs a FisherSignal "
                             "(pass fisher_signal= to ModelAnalyzer)")
        persistence_names = self._persistence_names()
        rng = make_rng(self.config.random_state)

        n_gradient = fcfg.n_gradient_prompts if fcfg.compute_gradients else 0
        pool = DataSampler.sample(data, fcfg.n_samples + n_gradient, rng=rng)
        if isinstance(pool.x_data, dict):
            logger.warning("Fisher: multi-input models have limited support; the input dict "
                           "is passed to each model as is.")
        plan = fm.SamplePlan.build(fm._n_rows(pool.x_data), fcfg.n_samples, n_gradient, rng,
                                   seed=self.config.random_state)
        self._warn_if_tf32()

        kinds, layouts, failures, batch = self._prepare_models(pool, fcfg)
        fcfg = dataclasses.replace(fcfg, batch_size=batch)
        outputs, run_failures = self._run_models(pool, plan, fcfg, rng, kinds, layouts)
        failures.update(run_failures)
        self._publish(results, outputs, failures, plan, fcfg, persistence_names)

    # ------------------------------------------------------------------

    def _persistence_names(self) -> Optional[List[str]]:
        """Validate ``fisher_persistence_indices`` against the model names."""
        names = self.config.fisher_persistence_indices
        if names is None:
            return None
        names = list(names)
        if len(names) < 2:
            raise ValueError("fisher_persistence_indices needs the initial policy and at "
                             "least one later checkpoint")
        unknown = [n for n in names if n not in self.models]
        if unknown:
            raise ValueError(f"fisher_persistence_indices names unknown models: {unknown}")
        return names

    @staticmethod
    def _warn_if_tf32() -> None:
        """Warn when TF32 matmul is on for a GPU run: it perturbs variance-sensitive scores.

        The analyzer only READS the flag. Writing it is process-global and is confined to
        ``tests/numerics.py`` by a census test.
        """
        try:
            import tensorflow as tf
            on_gpu = bool(tf.config.list_physical_devices('GPU'))
            tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
        except Exception:  # pragma: no cover - defensive, TF absent or unreadable
            return
        if on_gpu and tf32:
            logger.warning("Fisher: TF32 matmul is enabled on a visible GPU. Squared scores are "
                           "variance-sensitive; disable it for publication-grade estimates.")

    def _prepare_models(self, pool: DataInput, fcfg: fm.FisherConfig
                        ) -> Tuple[Dict[str, str], Dict[str, fm.ParamLayout], Dict[str, str], int]:
        """Infer each model's output kind and size the sampling batch ONCE for all of them.

        A budget sized per model would measure models at different batches (D-037). The
        batch only groups the sampling draws and does not change the estimator, but it
        must still be one number for the run.

        Returns:
            ``(kinds, layouts, failures, batch_size)``.
        """
        kinds: Dict[str, str] = {}
        layouts: Dict[str, fm.ParamLayout] = {}
        failures: Dict[str, str] = {}
        batches: List[int] = []
        for name, model in self.models.items():
            try:
                kind = fm.infer_output_kind(model, pool.x_data, fcfg, self.signal)
                layout = fm.canonical_layout(model)
                per_sample = fm.sampling_bytes_per_sample(model, kind, pool.x_data, fcfg)
            except (AttributeError, TypeError, NameError, ImportError):
                raise
            except Exception as e:
                logger.error(f"Fisher: {name} cannot be analyzed: {e}")
                failures[name] = str(e)
                continue
            kinds[name], layouts[name] = kind, layout
            batches.append(fm.fit_batch_size(fcfg.batch_size, per_sample, fcfg.memory_limit_mb))
        return kinds, layouts, failures, (min(batches) if batches else fcfg.batch_size)

    def _run_models(self, pool: DataInput, plan: fm.SamplePlan, fcfg: fm.FisherConfig,
                    rng: np.random.Generator, kinds: Dict[str, str],
                    layouts: Dict[str, fm.ParamLayout]
                    ) -> Tuple[Dict[str, fm.FisherOutput], Dict[str, str]]:
        """Run the arms for every model, serially, isolating each model's failure."""
        outputs: Dict[str, fm.FisherOutput] = {}
        failures: Dict[str, str] = {}
        for name, model in self.models.items():
            if name not in kinds:
                continue
            started = time.time()
            try:
                out = fm.compute_diagonal_fisher(
                    model, pool.x_data, fcfg, rng, indices=plan.fisher_indices,
                    signal=self.signal, layout=layouts[name], kind=kinds[name])
                if fcfg.compute_gradients:
                    arm = fm.compute_gradients(
                        model, pool.x_data, pool.y_data, fcfg, rng, self.signal,
                        plan.gradient_indices, layout=layouts[name], kind=kinds[name])
                    out = fm.attach_gradient_arm(out, arm, fcfg)
                if fcfg.token_level and out.output_kind == fm.KIND_SEQUENCE:
                    out = fm.compute_token_fisher(model, pool.x_data, out, fcfg, rng,
                                                  layout=layouts[name])
            except (AttributeError, TypeError, NameError, ImportError):
                # Bugs, not measurements: a swallowed AttributeError here once published
                # an empty analysis as if it had succeeded.
                raise
            except Exception as e:
                logger.error(f"Fisher: {name} failed and was skipped: {e}", exc_info=True)
                failures[name] = str(e)
                continue
            out.metrics[FM.COMPUTATION_TIME] = time.time() - started
            outputs[name] = out
        return outputs, failures

    # ------------------------------------------------------------------

    @staticmethod
    def _table(name: str, out: fm.FisherOutput) -> pd.DataFrame:
        """The per-parameter table of one model, in the canonical coordinate order."""
        n = out.n_parameters
        sizes = [stop - start for _, start, stop in out.param_slices]
        paths = pd.Categorical.from_codes(np.repeat(np.arange(len(sizes)), sizes),
                                          categories=[p for p, _, _ in out.param_slices])
        gradient = np.full(n, np.nan)
        rank = np.full(n, np.nan)
        if out.gradient_flat is not None:
            gradient = out.gradient_flat.astype(np.float64)
            order = np.argsort(-np.abs(gradient), kind='stable')
            rank[order] = np.arange(1, n + 1)
        se = (out.standard_errors.astype(np.float64) if out.standard_errors is not None
              else np.full(n, np.nan))
        return pd.DataFrame({
            'model_name': name, 'variable_path': paths, 'flat_index': np.arange(n),
            'fisher_value': out.fisher_flat.astype(np.float64), 'standard_error': se,
            'gradient_value': gradient, 'gradient_rank': rank,
        })

    def _publish(self, results: AnalysisResults, outputs: Dict[str, fm.FisherOutput],
                 failures: Dict[str, str], plan: fm.SamplePlan, fcfg: fm.FisherConfig,
                 persistence_names: Optional[List[str]]) -> None:
        """Write everything onto ``results``. Artifacts arrive by return value (D-020)."""
        results.fisher_outputs = dict(outputs)
        results.fisher_summary = {
            'models': list(outputs), 'failed_models': dict(failures),
            'n_models': len(outputs), 'n_pool': plan.n_pool,
            'sample_plan': plan.to_dict(), 'sample_plan_fingerprint': plan.fingerprint(),
            'batch_size': fcfg.batch_size, 'signal_type': fcfg.signal_type,
            'n_responses': fcfg.n_responses,
            'n_gradient_prompts': len(plan.gradient_indices),
        }
        per_model: Dict[str, Dict[str, Any]] = {}
        for name, out in outputs.items():
            per_model[name] = {**out.metrics, FM.STATUS: 'ok'}
        for name, message in failures.items():
            # A failed row says so; it never publishes 0.0 (D-036 / D-003).
            per_model[name] = {FM.STATUS: FM.ERROR, FM.ERROR: message}
        results.fisher_summary_per_model = per_model

        results.fisher_gradient_recall = {
            name: dict(out.gradient_recall) for name, out in outputs.items()
            if out.gradient_recall is not None}
        results.fisher_token_level = {
            name: {
                'group_names': out.group_names,
                'token_fisher_by_position': out.token_fisher_by_position,
                'uniform_by_position': out.uniform_by_position,
                'real_uniform_ratio': out.real_uniform_ratio,
                FM.TOKEN_FISHER_MEAN: out.token_fisher_mean,
                FM.UNIFORM_REFERENCE: out.uniform_reference,
            } for name, out in outputs.items() if out.token_fisher_mean is not None}
        results.fisher_analysis = (
            pd.concat([self._table(n, o) for n, o in outputs.items()], ignore_index=True)
            if outputs else None)

        results.fisher_persistence = None
        if persistence_names is not None:
            missing = [n for n in persistence_names if n not in outputs]
            if missing:
                logger.warning(f"Fisher persistence skipped: no output for {missing}")
            else:
                records = fm.compute_persistence(
                    outputs[persistence_names[0]],
                    {n: outputs[n] for n in persistence_names[1:]}, fcfg)
                results.fisher_persistence = pd.DataFrame(records)
