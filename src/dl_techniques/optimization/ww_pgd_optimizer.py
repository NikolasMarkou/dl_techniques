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
* The Cayley update + retraction preserve the tail trace-log
  (``sum(log(lam_new)) == sum(log(lam_tail))``) to float tolerance: the projection
  redistributes the tail *shape* without inflating/deflating its scale.
* Fail-soft: any per-layer numerical failure (SVD non-convergence, ``fit_powerlaw``
  returning a non-finite/<=0 ``xmin``) skips that layer and continues; one bad layer
  never aborts the others, and the projection never crashes training.
"""

from __future__ import annotations

import keras
import numpy as np
from typing import Any, Dict, List, Optional

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
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WWTailConfig":
        return cls(**config)


# ---------------------------------------------------------------------
# hardness / homotopy schedule
# ---------------------------------------------------------------------

def _compute_hardness(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """Compute the homotopy hardness in ``[0, 1]`` for ``epoch``.

    ``epoch < warmup`` -> 0.0; ``epoch >= warmup + ramp`` -> 1.0; else a linear ramp
    ``(epoch - warmup + 1) / max(ramp, 1)`` clamped to ``[0, 1]``.
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
) -> bool:
    """Apply the WW-PGD tail-projection to one layer's kernel in place.

    Runs entirely in the eager numpy domain. Returns ``True`` if the kernel was
    projected and re-assigned, ``False`` if the layer was skipped (tail too small,
    failed power-law fit, etc.). Raises on hard numerical failures (caught by the
    per-layer try/except in :func:`ww_pgd_project`).
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
        return False

    # Power-law fit on the eigenvalues (lam = S**2).
    alpha, optimal_xmin, ks_distance, _sigma, _num_pl, status, _warning = fit_powerlaw(lam)
    if status != _FIT_SUCCESS:
        return False
    xmin = float(optimal_xmin)
    if not np.isfinite(xmin) or xmin <= 0.0:
        return False

    # DECISION plan_2026-06-20_c0f110f5/D-002: opt-in goodness-of-fit gate. A poor
    # power-law fit (large KS distance) means the tail template is not meaningful, so
    # skip the layer (SETOL poor-fit guidance). Default max_ks_distance=None -> no gating.
    if (config.max_ks_distance is not None
            and np.isfinite(ks_distance)
            and ks_distance > config.max_ks_distance):
        return False

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
        return False

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
    return True


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
    for layer in layers:
        # Per-layer fail-soft (I3): one bad layer never aborts the rest.
        try:
            if _shape_single_layer(layer, config, hardness):
                projected += 1
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            logger.warning(
                f"WW-PGD: projection failed for layer "
                f"'{getattr(layer, 'name', '?')}' ({type(e).__name__}: {e}); "
                f"skipping this layer."
            )

    return {
        "hardness": float(hardness),
        "layers_projected": int(projected),
        "layers_skipped": int(skipped),
    }


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
    """

    def __init__(
            self,
            config: Optional[WWTailConfig] = None,
            num_epochs: int = 1,
            model: Optional[keras.Model] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config: WWTailConfig = config or WWTailConfig()
        self._num_epochs: int = int(num_epochs)
        # Distinct attr name: do NOT shadow Keras' self.model property.
        self._explicit_model: Optional[keras.Model] = model
        self._warned_no_model: bool = False

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
        except Exception as e:
            logger.error(
                f"WWPGDProjectionCallback: projection raised at epoch {epoch} "
                f"({type(e).__name__}: {e}); training continues.",
                exc_info=True,
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
