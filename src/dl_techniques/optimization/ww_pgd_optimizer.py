"""WW-PGD: WeightWatcher Projected Gradient Descent as a Keras 3 callback.

This module ports the experimental PyTorch WW-PGD spectral *tail-projection* to
Keras 3 / TensorFlow 2.18. At every epoch boundary it applies a non-differentiable
spectral projection to each rank>=2 weight matrix of a model: it SVD-decomposes the
kernel, fits a power-law to the squared singular values (``lam = S**2``), selects the
heavy tail above ``xmin``, nudges the tail eigenvalues toward an ``r^(-q)`` power-law
template via a trace-log-preserving Cayley log-space update, and blends the reshaped
matrix back into the original kernel.

Why a callback, not an optimizer
--------------------------------
The projection is *non-differentiable* and *epoch-boundary*, operating on whole
weight matrices via an eager SVD. A ``keras.optimizers.Optimizer.update_step`` is
``tf.function``-traced, per-variable, and epoch-unaware: an SVD there would run every
step per variable and fight XLA/jit_compile. A ``keras.callbacks.Callback.on_epoch_end``
runs eagerly once per epoch boundary, can call numpy SVD, and walks ``model.layers``.
See ``plans/plan_2026-06-20_73c91aad/decisions.md`` D-001.

Why native ``spectral_metrics``, not WeightWatcher
--------------------------------------------------
The original PyTorch package depends on the WeightWatcher library (PyTorch-centric,
not installed) for the per-layer ``xmin`` / ``detX_num`` diagnostics. The repo's
``analyzer/spectral_metrics.py`` natively computes the identical diagnostics for Keras
models (``fit_powerlaw``, ``compute_detX_constraint``), so no third-party dependency is
required.

Module surface
--------------
Three pieces mirror the PyTorch package split:

* :class:`WWTailConfig` (``config.py``) -- serializable scalar config (default OFF).
* :func:`ww_pgd_project` (``project.py``) -- pure stateless projection over a model.
* :class:`WWPGDProjectionCallback` (``wrapper.py``) -- epoch-boundary callback wrapper.

Invariants
----------
* OFF (``enable=False``) or warmup (``hardness == 0``) is a strict no-op: no weight
  reads, no SVD, no kernel writes.
* The Cayley update + retraction preserve the tail trace-log of the EIGENVALUE PATH
  (``sum(log(lam_new)) == sum(log(lam_tail))``) to float tolerance. NOTE: this holds for
  the pre-blend eigenvalues only. The final written kernel uses a LINEAR singular-value
  blend ``(1-be)*S + be*S_new`` (shared U, Vh), which is not geometry-preserving, so the
  ASSIGNED kernel's tail trace-log is preserved only at ``blend_eta == 1`` (or the trivial
  ``blend_eta == 0``); at the default ``blend_eta=0.5`` the assigned tail trace-log drifts
  (empirically ~+0.4..0.6 nats). The projection redistributes tail *shape*; it does not
  guarantee tail-scale preservation of the written kernel.
* Fail-soft: any per-layer numerical failure (SVD non-convergence, ``fit_powerlaw``
  returning a non-finite/<=0 ``xmin``) skips that layer and continues; one bad layer
  never aborts the others, and the projection never crashes training.

Known limitations / audit notes
-------------------------------
Recorded from the audit in ``analyses/analysis_2026-06-20_ea6a9d67``:

* Trace-log preservation is an eigenvalue-path property, not a written-kernel property
  except at ``blend_eta in {0, 1}`` (see I2 above).
* ``cayley_eta == 0`` with ``hardness > 0`` does NOT mean 'no Cayley step' -- the
  ``eta <= 0`` branch snaps the tail directly to the full ``r^(-q)`` template
  (a stronger move than a small Cayley step), then blends.
* The homotopy ``hardness`` reaches 1.0 at epoch ``warmup + ramp - 1`` (one epoch before
  ``warmup + ramp``); see :func:`_compute_hardness`.
* Frozen (``trainable=False``) kernels are skipped (``_iter_kernel_layers`` trainable
  guard) so a frozen stem is never overwritten.
* Goodness-of-fit gating is opt-in via ``WWTailConfig.max_ks_distance`` (default ``None``
  = no gating). A poor power-law fit otherwise still gets reshaped.
* SCOPE: the projection targets only the HTSR ``alpha ~= 2`` condition (the ``r^(-q)``
  template), NOT the co-equal SETOL ERG / ``det(X-tilde)=1`` condition -- it preserves
  the existing tail trace-log rather than driving it to 0. Net training-efficacy is
  UNVALIDATED (no A/B run); use as an experiment, not a proven improvement.
"""

from __future__ import annotations

import os
import csv
import keras
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.analyzer.constants import LayerType
from dl_techniques.analyzer.spectral_utils import infer_layer_type
from dl_techniques.analyzer.spectral_metrics import (
    fit_powerlaw,
    compute_detX_constraint,
)

# ---------------------------------------------------------------------

__all__: List[str] = [
    "WWTailConfig",
    "ww_pgd_project",
    "WWPGDProjectionCallback",
]

# ---------------------------------------------------------------------
# numerical guards
# ---------------------------------------------------------------------

_EPS: float = 1e-8
_SV_FLOOR: float = 1e-8
_RATIO_LO: float = 0.1
_RATIO_HI: float = 10.0
_FIT_SUCCESS: str = "success"


# ---------------------------------------------------------------------
# DECISION plan_2026-06-20_73c91aad/D-001:
# WW-PGD is ported as a serializable keras.callbacks.Callback (epoch-boundary), NOT a
# keras.optimizers.Optimizer subclass, and uses the repo-native
# analyzer.spectral_metrics (fit_powerlaw / compute_detX_constraint) instead of the
# WeightWatcher / powerlaw libraries. Do NOT reshape this into an update_step optimizer
# (tf.function-traced, per-variable, epoch-unaware -> wrong granularity, fights XLA) and
# do NOT add a weightwatcher dependency (heavy PyTorch pull, fragile Keras adapter).
# See plans/plan_2026-06-20_73c91aad/decisions.md D-001.
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WWTailConfig:
    """Scalar configuration for the WW-PGD spectral tail-projection.

    Holds only plain scalars so it round-trips cleanly through
    ``get_config`` / ``from_config``. Mirrors the PyTorch ``WWTailConfig`` (the
    ``trap_pgd`` placeholders are intentionally dropped).

    Args:
        enable: Master switch. Default ``False`` (repo opt-in convention / I1):
            when ``False`` the projection is a strict no-op.
        min_tail: Minimum number of eigenvalues in the selected heavy tail; layers
            with a smaller tail are skipped.
        q: Power-law template exponent. The target template is ``r^(-q)`` over the
            tail rank ``r``; ``q == 1`` corresponds to a target alpha of ~2.
        blend_eta: Maximum blend fraction of the reshaped matrix back into the
            original (scaled by ``hardness``).
        cayley_eta: Maximum Cayley log-space step size (scaled by ``hardness``).
        max_ks_distance: Optional KS-distance ceiling for the power-law fit. When set,
            layers whose fit_powerlaw KS distance exceeds it are skipped (poor PL fit ->
            the r^(-q) template is not meaningful). Default None -> no goodness-of-fit
            gating (behavior unchanged).
        use_detx: If ``True``, fuse the ``compute_detX_constraint`` tail count with the
            power-law tail count to pick the tail threshold.
        warmup_epochs: Number of leading epochs with ``hardness == 0`` (no projection).
        ramp_epochs: Number of epochs over which ``hardness`` ramps 0 -> 1 after warmup.
        apply_every_epochs: Callback cadence; the projection runs every Nth epoch.
        verbose: If ``True``, the callback logs a per-epoch summary at INFO.
        log_layer_stats: If ``True``, the projection captures per-layer
            ``{name, alpha_before, alpha_after, tail_size, ks_distance}`` (the
            ``alpha_after`` re-fit costs ONE extra SVD per projected layer) and
            ``ww_pgd_project`` adds a ``"layer_stats"`` key to its return dict.
            Default ``False`` -> ZERO extra SVD/refit and the OFF path is
            byte-identical (D-002). The CSV destination is NOT a config field
            (D-003); the callback takes a ``csv_path`` ctor arg instead.
    """

    def __init__(
            self,
            enable: bool = False,
            min_tail: int = 5,
            q: float = 1.0,
            blend_eta: float = 0.5,
            cayley_eta: float = 0.25,
            max_ks_distance: Optional[float] = None,
            use_detx: bool = True,
            warmup_epochs: int = 0,
            ramp_epochs: int = 5,
            apply_every_epochs: int = 1,
            verbose: bool = False,
            log_layer_stats: bool = False,
    ) -> None:
        self.enable = bool(enable)
        self.min_tail = int(min_tail)
        self.q = float(q)
        self.blend_eta = float(blend_eta)
        self.cayley_eta = float(cayley_eta)
        self.max_ks_distance = None if max_ks_distance is None else float(max_ks_distance)
        self.use_detx = bool(use_detx)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.apply_every_epochs = int(apply_every_epochs)
        self.verbose = bool(verbose)
        self.log_layer_stats = bool(log_layer_stats)

    def get_config(self) -> Dict[str, Any]:
        return {
            "enable": self.enable,
            "min_tail": self.min_tail,
            "q": self.q,
            "blend_eta": self.blend_eta,
            "cayley_eta": self.cayley_eta,
            "max_ks_distance": self.max_ks_distance,
            "use_detx": self.use_detx,
            "warmup_epochs": self.warmup_epochs,
            "ramp_epochs": self.ramp_epochs,
            "apply_every_epochs": self.apply_every_epochs,
            "verbose": self.verbose,
            "log_layer_stats": self.log_layer_stats,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WWTailConfig":
        return cls(**config)


# ---------------------------------------------------------------------
# hardness / homotopy schedule
# ---------------------------------------------------------------------

def _compute_hardness(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """Compute the homotopy hardness in ``[0, 1]`` for ``epoch``.

    ``epoch < warmup`` -> 0.0; else a linear ramp ``(epoch - warmup + 1) / max(ramp, 1)``
    clamped to ``[0, 1]``. Because of the ``+1`` in the numerator, hardness reaches 1.0 at
    ``epoch == warmup + ramp - 1`` (one epoch before ``warmup + ramp``); the explicit
    ``epoch >= warmup + ramp`` guard is therefore redundant for the boundary and only
    matters for epochs strictly beyond it. The FORMULA is unchanged from the original.
    """
    w = int(warmup_epochs)
    r = int(ramp_epochs)
    if epoch < w:
        return 0.0
    if epoch >= w + r:
        return 1.0
    h = (epoch - w + 1) / max(r, 1)
    return float(min(1.0, max(0.0, h)))


# ---------------------------------------------------------------------
# layer selection
# ---------------------------------------------------------------------

def _iter_kernel_layers(model: keras.Model) -> List[keras.layers.Layer]:
    """Return rank>=2, non-embedding kernel layers reachable from ``model``.

    Uses ``model._flatten_layers`` (recurses into nested models/sublayers) with a
    dedupe guard on object id. Embeddings are excluded (mirrors the Muon
    ``_should_use_muon`` rank>=2 / embedding-exclusion filter).
    """
    selected: List[keras.layers.Layer] = []
    seen: set = set()

    try:
        candidates = list(model._flatten_layers(include_self=False, recursive=True))
    except Exception:
        # Fallback: shallow walk if the private API is unavailable.
        candidates = list(getattr(model, "layers", []))

    for layer in candidates:
        if id(layer) in seen:
            continue
        seen.add(id(layer))

        kernel = getattr(layer, "kernel", None)
        if kernel is None:
            continue
        if len(kernel.shape) < 2:
            continue

        # Exclude embeddings (degenerate / lookup tables, not linear transforms).
        try:
            if infer_layer_type(layer) == LayerType.EMBEDDING:
                continue
        except Exception:
            pass
        if "embedding" in layer.__class__.__name__.lower():
            continue

        # DECISION plan_2026-06-20_c0f110f5/D-001: never project frozen kernels.
        # keras .assign() bypasses trainable=False, so a frozen layer (e.g. the
        # frozen Gabor depthwise stem in bfconvunext) would be silently overwritten
        # every epoch. Guard on the layer-level trainable flag (fail-open via getattr).
        if not getattr(layer, "trainable", True):
            continue

        selected.append(layer)

    return selected


# ---------------------------------------------------------------------
# per-layer projection
# ---------------------------------------------------------------------

def _shape_single_layer(
        layer: keras.layers.Layer,
        config: WWTailConfig,
        hardness: float,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Apply the WW-PGD tail-projection to one layer's kernel in place.

    Runs entirely in the eager numpy domain.

    Returns a ``(projected, stats)`` tuple:

    * ``projected`` is ``True`` if the kernel was projected and re-assigned,
      ``False`` if the layer was skipped (tail too small, failed power-law fit,
      etc.).
    * ``stats`` is ``None`` unless ``config.log_layer_stats`` is ``True`` AND the
      layer was actually projected, in which case it is a dict with keys
      ``{name, alpha_before, alpha_after, tail_size, ks_distance}``. Every skip
      path returns ``(False, None)``.

    DECISION plan_2026-06-20_5fe67f0c/D-002: ALL extra stats work (notably the
    ``alpha_after`` re-SVD/re-fit) is gated strictly behind
    ``config.log_layer_stats``. When it is False the function does ZERO extra SVD
    or power-law refit and the kernel assigned is byte-identical to the
    pre-instrumentation behavior -- the OFF path must not move the validated
    default. Do NOT compute the stats unconditionally and discard them when OFF;
    that would perturb the validated default path. See decisions.md D-002.

    Raises on hard numerical failures (caught by the per-layer try/except in
    :func:`ww_pgd_project`).
    """
    kernel = layer.kernel
    W = np.array(keras.ops.convert_to_numpy(kernel))
    orig_shape = W.shape
    orig_dtype = kernel.dtype

    # Reshape to 2D: rank-2 as-is; rank>2 conv kernels -> (-1, out_channels) (matches
    # get_weight_matrices' Conv reshape convention; out_channels last).
    if W.ndim == 2:
        W2 = W
    else:
        W2 = W.reshape(-1, orig_shape[-1])
    twod_shape = W2.shape

    # SVD (numpy, eager). full_matrices=False -> S descending.
    U, S, Vh = np.linalg.svd(W2, full_matrices=False)
    S = np.maximum(S, _SV_FLOOR)
    lam = S ** 2
    n = lam.size
    if n == 0:
        return False, None

    # Power-law fit on the eigenvalues (lam = S**2).
    alpha, optimal_xmin, ks_distance, _sigma, _num_pl, status, _warning = fit_powerlaw(lam)
    if status != _FIT_SUCCESS:
        return False, None
    xmin = float(optimal_xmin)
    if not np.isfinite(xmin) or xmin <= 0.0:
        return False, None

    # DECISION plan_2026-06-20_c0f110f5/D-002: opt-in goodness-of-fit gate. A poor
    # power-law fit (large KS distance) means the tail template is not meaningful, so
    # skip the layer (SETOL poor-fit guidance). Default max_ks_distance=None -> no gating.
    if (config.max_ks_distance is not None
            and np.isfinite(ks_distance)
            and ks_distance > config.max_ks_distance):
        return False, None

    # Tail threshold. lam is descending (numpy SVD returns descending S).
    lam_thr = xmin
    if config.use_detx:
        try:
            detx_num = int(compute_detX_constraint(lam))
        except Exception:
            detx_num = 0
        if detx_num > 0:
            k_pl = int((lam >= xmin).sum())
            if k_pl > 0:
                k_pl = int(min(max(k_pl, 1), n))
                k_detx = int(min(max(detx_num, 1), n))
                k_star = max(1, int(0.5 * (k_pl + k_detx)))
                lam_thr = max(lam_thr, float(lam[k_star - 1]))

    tail_mask = lam >= lam_thr
    tail_size = int(tail_mask.sum())
    if tail_size < config.min_tail:
        return False, None

    lam_tail = lam[tail_mask]

    # Power-law target template r^(-q), trace-log matched to the current tail.
    r = np.arange(1, tail_size + 1)
    mu = r.astype(np.float64) ** (-config.q)
    T_target = float(np.log(lam_tail).sum())
    A = np.exp((T_target - float(np.log(mu).sum())) / tail_size)
    lam_target = A * mu

    # Cayley log-space update (effective eta scaled by hardness).
    eta = hardness * config.cayley_eta
    if eta <= 0.0:
        # DECISION plan_2026-06-20_c0f110f5/D-003: eta<=0 (i.e. cayley_eta=0 at
        # hardness>0) snaps directly to the full r^(-q) template -- a STRONGER move
        # than a small Cayley step, NOT a no-op. Documented in the module docstring.
        lam_new = lam_target.copy()
    else:
        g = np.log(lam_tail + _EPS) - np.log(lam_target + _EPS)
        ratio = (1.0 - eta * g) / (1.0 + eta * g)
        ratio = np.clip(ratio, _RATIO_LO, _RATIO_HI)
        lam_new = lam_tail * ratio

    # Retract to preserve the tail trace-log (I2).
    shift = (T_target - float(np.log(lam_new + _EPS).sum())) / tail_size
    lam_new = lam_new * np.exp(shift)

    # Reconstruct the reshaped matrix from the updated tail eigenvalues.
    S_new = S.copy()
    S_new[tail_mask] = np.sqrt(np.maximum(lam_new, _SV_FLOOR))
    W2_shaped = (U * S_new) @ Vh

    # Blend (effective be scaled by hardness).
    be = hardness * config.blend_eta
    W_new2d = (1.0 - be) * W2 + be * W2_shaped

    # Reshape back to the ORIGINAL kernel shape and assign.
    W_new = W_new2d.reshape(orig_shape) if twod_shape != orig_shape else W_new2d
    layer.kernel.assign(keras.ops.cast(W_new, orig_dtype))

    # DECISION plan_2026-06-20_5fe67f0c/D-002: per-layer stats (incl. the
    # alpha_after re-SVD/refit) ONLY when log_layer_stats is ON. Placed AFTER the
    # assign so it cannot influence what was written; OFF path returns here with
    # no extra work and a byte-identical kernel. See decisions.md D-002.
    if not config.log_layer_stats:
        return True, None

    # alpha_after: ONE extra re-fit on the post-blend reshaped matrix's
    # eigenvalues (lam = S**2 of the blended W_new2d). 2D shape for SVD.
    S_after = np.linalg.svd(W_new2d, full_matrices=False, compute_uv=False)
    lam_after = np.maximum(S_after, _SV_FLOOR) ** 2
    alpha_after, _xmin_a, _ks_a, _s_a, _n_a, _status_a, _w_a = fit_powerlaw(lam_after)
    stats = {
        "name": getattr(layer, "name", "?"),
        "alpha_before": float(alpha),
        "alpha_after": float(alpha_after),
        "tail_size": int(tail_size),
        "ks_distance": float(ks_distance),
    }
    return True, stats


# ---------------------------------------------------------------------
# stateless projection over a whole model
# ---------------------------------------------------------------------

def ww_pgd_project(
        model: keras.Model,
        config: WWTailConfig,
        *,
        epoch: int,
        num_epochs: int,
        logs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply the WW-PGD spectral tail-projection to ``model`` in place.

    Pure stateless function: walks the model's rank>=2 non-embedding kernels and, for
    each layer with a sufficiently large power-law tail, runs the
    SVD -> fit -> tail-select -> template -> Cayley -> retract -> blend -> reshape ->
    assign pipeline. ``apply_every_epochs`` cadence gating is the *caller's* job (the
    callback); this function always runs the projection when invoked (modulo the
    ``enable`` / ``hardness == 0`` no-op guards).

    Args:
        model: The model whose kernels are projected (the FULL model, not a view).
        config: The :class:`WWTailConfig` controlling the projection.
        epoch: Current (0-based) epoch index, used for the hardness schedule.
        num_epochs: Total planned epochs (carried for callers; not used internally).
        logs: Optional Keras logs dict (unused; accepted for caller convenience).

    Returns:
        A summary dict ``{"hardness", "layers_projected", "layers_skipped"}``.
        When ``config.log_layer_stats`` is ``True`` an additional ``"layer_stats"``
        key (a ``List[Dict]`` of per-projected-layer
        ``{name, alpha_before, alpha_after, tail_size, ks_distance}``) is included;
        when it is ``False`` the key is OMITTED so the OFF return dict is unchanged.
    """
    del num_epochs, logs  # accepted for caller symmetry; not used here.

    if not config.enable:
        return {"hardness": 0.0, "layers_projected": 0, "layers_skipped": 0}

    hardness = _compute_hardness(epoch, config.warmup_epochs, config.ramp_epochs)
    if hardness == 0.0:
        # Cheap guard: no SVD, no weight reads.
        return {"hardness": 0.0, "layers_projected": 0, "layers_skipped": 0}

    layers = _iter_kernel_layers(model)

    projected = 0
    skipped = 0
    layer_stats: List[Dict[str, Any]] = []
    for layer in layers:
        # Per-layer fail-soft (I3): one bad layer never aborts the rest.
        try:
            did_project, stats = _shape_single_layer(layer, config, hardness)
            if did_project:
                projected += 1
                if stats is not None:
                    layer_stats.append(stats)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            logger.warning(
                f"WW-PGD: projection failed for layer "
                f"'{getattr(layer, 'name', '?')}' ({type(e).__name__}: {e}); "
                f"skipping this layer."
            )

    summary: Dict[str, Any] = {
        "hardness": float(hardness),
        "layers_projected": int(projected),
        "layers_skipped": int(skipped),
    }
    if config.log_layer_stats:
        summary["layer_stats"] = layer_stats
    return summary


# ---------------------------------------------------------------------
# epoch-boundary callback wrapper
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class WWPGDProjectionCallback(keras.callbacks.Callback):
    """Epoch-boundary callback running the WW-PGD spectral tail-projection.

    Wraps :func:`ww_pgd_project` in ``on_epoch_end``. The full model SHOULD be passed
    explicitly at construction (some trainers fit a single-output *view* that differs
    from the full model); when no explicit model is given, the callback falls back to
    Keras' ``self.model``.

    Args:
        config: The :class:`WWTailConfig`; defaults to a disabled config (no-op).
        num_epochs: Total planned epochs, forwarded to :func:`ww_pgd_project`.
        model: Optional explicit FULL model. Stored as ``self._explicit_model`` so it
            does NOT shadow Keras' ``self.model`` property. Excluded from ``get_config``.
        csv_path: Optional destination for the per-epoch per-layer alpha CSV. Only
            used when ``config.log_layer_stats`` is ``True``. DECISION
            plan_2026-06-20_5fe67f0c/D-003: this is a run-specific filesystem path
            and is intentionally NOT serialized into the config/get_config (the
            trainer re-supplies it each run, like the live model ref). Do NOT move
            this into ``WWTailConfig`` -- a serialized absolute path is a
            portability/round-trip hazard. See decisions.md D-003.
    """

    def __init__(
            self,
            config: Optional[WWTailConfig] = None,
            num_epochs: int = 1,
            model: Optional[keras.Model] = None,
            csv_path: Optional[str] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config: WWTailConfig = config or WWTailConfig()
        self._num_epochs: int = int(num_epochs)
        # Distinct attr name: do NOT shadow Keras' self.model property.
        self._explicit_model: Optional[keras.Model] = model
        self._warned_no_model: bool = False
        # D-003: run-specific path, NOT serialized.
        self._csv_path: Optional[str] = csv_path

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        cfg = self._config
        if not cfg.enable:
            return
        if (epoch + 1) % max(cfg.apply_every_epochs, 1) != 0:
            return

        m = self._explicit_model if self._explicit_model is not None else self.model
        if m is None:
            if not self._warned_no_model:
                logger.warning(
                    "WWPGDProjectionCallback: no model available (no explicit model "
                    "and self.model is None); projection is a no-op."
                )
                self._warned_no_model = True
            return

        # Whole body wrapped: a projection failure must never crash training.
        try:
            summary = ww_pgd_project(
                m, cfg, epoch=epoch, num_epochs=self._num_epochs, logs=logs
            )
            if cfg.verbose:
                logger.info(
                    f"WW-PGD epoch {epoch}: hardness={summary['hardness']:.3f}, "
                    f"projected={summary['layers_projected']}, "
                    f"skipped={summary['layers_skipped']}"
                )
            if cfg.log_layer_stats and self._csv_path and "layer_stats" in summary:
                self._append_layer_stats_csv(epoch, summary["layer_stats"])
        except Exception as e:
            logger.error(
                f"WWPGDProjectionCallback: projection raised at epoch {epoch} "
                f"({type(e).__name__}: {e}); training continues.",
                exc_info=True,
            )

    def _append_layer_stats_csv(
            self, epoch: int, layer_stats: List[Dict[str, Any]]
    ) -> None:
        """Append the per-layer alpha rows for ``epoch`` to the CSV (fail-soft).

        Header ``epoch,name,alpha_before,alpha_after,tail_size,ks_distance`` is
        written once when the file is new. Any I/O error is logged and swallowed:
        a CSV-write failure must NEVER crash training.
        """
        path = self._csv_path
        try:
            new_file = (not os.path.exists(path)) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow([
                        "epoch", "name", "alpha_before",
                        "alpha_after", "tail_size", "ks_distance",
                    ])
                for s in layer_stats:
                    writer.writerow([
                        epoch,
                        s.get("name", "?"),
                        s.get("alpha_before"),
                        s.get("alpha_after"),
                        s.get("tail_size"),
                        s.get("ks_distance"),
                    ])
        except Exception as e:
            logger.warning(
                f"WWPGDProjectionCallback: failed to write layer-stats CSV "
                f"'{path}' at epoch {epoch} ({type(e).__name__}: {e}); "
                f"training continues."
            )

    def get_config(self) -> Dict[str, Any]:
        # The live model ref is intentionally NOT serialized (re-supplied by trainer).
        return {
            "config": keras.saving.serialize_keras_object(self._config),
            "num_epochs": self._num_epochs,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WWPGDProjectionCallback":
        config = dict(config)
        cfg_obj = keras.saving.deserialize_keras_object(config.pop("config"))
        return cls(config=cfg_obj, num_epochs=config.pop("num_epochs", 1))

# ---------------------------------------------------------------------
