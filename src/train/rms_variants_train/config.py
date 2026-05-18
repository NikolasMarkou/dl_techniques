"""
Shared experiment configuration for the RMSNorm-variants study.

Defines :class:`ExperimentConfig` (single-run knobs) and :func:`build_norm_kwargs`
(per-variant kwargs dispatcher for the ``create_normalization_layer`` factory).

The variant-tuple :data:`NORM_VARIANTS` is the canonical ordering used by the
sweep driver and the report writer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------
# Norm variants under study
# ---------------------------------------------------------------------

# Canonical ordering: RMSNorm = baseline; six variants follow.
# Strings are the factory keys consumed by
# ``dl_techniques.layers.norms.factory.create_normalization_layer(<key>)``.
#
# DECISION plan_2026-05-18_74a935a2/D-001: this 7-tuple is the campaign's
# variant universe. Reordering it invalidates RESULTS.md tables (columns are
# indexed by position). Append-only on extension; never insert in the middle.
NORM_VARIANTS: Tuple[str, str, str, str, str, str, str] = (
    "rms_norm",
    "band_rms",
    "zero_centered_rms_norm",
    "zero_centered_band_rms_norm",
    "adaptive_band_rms",
    "band_logit_norm",
    "dynamic_tanh",
)


# ---------------------------------------------------------------------
# Per-variant kwarg dispatcher
# ---------------------------------------------------------------------

def build_norm_kwargs(
    norm_type: str,
    *,
    use_scale: bool = True,
    max_band_width: float = 0.1,
    epsilon: float = 1e-6,
    band_regularizer_l2: Optional[float] = None,
) -> Dict[str, Any]:
    """Produce the kwargs dict for :func:`create_normalization_layer`.

    The function quietly drops kwargs that the target variant does not accept,
    so a single ``ExperimentConfig`` can drive all four variants.

    :param norm_type: One of :data:`NORM_VARIANTS`. Other factory keys
        (``layer_norm``, ``batch_norm``, etc.) are also accepted and return
        the minimal common kwarg dict ``{'epsilon': epsilon}``.
    :param use_scale: When ``True`` (default) the per-feature learnable scale is
        enabled on RMSNorm and ZeroCenteredRMSNorm. When ``False``, those layers
        become parameter-free — this is the **param_matched** mode that pulls
        their trainable-parameter count down to 0 to match the 1-scalar count
        of BandRMS / ZeroCenteredBandRMSNorm.
    :param max_band_width: BandRMS / ZeroCenteredBandRMSNorm only — alpha for
        the ``[1-alpha, 1]`` constraint band. Default 0.1.
    :param epsilon: Numerical-stability epsilon. Aligned across the matrix at
        ``1e-6`` (the default of the three RMSNorm-family variants — BandRMS's
        in-source default of ``1e-7`` is overridden here for matrix consistency).
    :param band_regularizer_l2: BandRMS variants only — L2 coefficient on the
        scalar ``band_param``. ``None`` (default) disables the layer's silent
        in-source ``L2(1e-5)`` default to eliminate double-regularization with
        AdamW weight decay (LESSONS L91 footgun).
    """
    if norm_type in ("rms_norm", "zero_centered_rms_norm"):
        return {
            "epsilon": epsilon,
            "use_scale": use_scale,
        }
    if norm_type in ("band_rms", "zero_centered_band_rms_norm"):
        kwargs: Dict[str, Any] = {
            "epsilon": epsilon,
            "max_band_width": max_band_width,
        }
        if band_regularizer_l2 is None:
            # Explicitly disable the layer's default L2 regularizer.
            import keras  # local import — avoid cost at module load
            del keras
            kwargs["band_regularizer"] = None
        else:
            import keras
            kwargs["band_regularizer"] = keras.regularizers.L2(band_regularizer_l2)
        return kwargs
    if norm_type == "adaptive_band_rms":
        # AdaptiveBandRMS: ``use_scale`` is not accepted (no per-feature scale
        # toggle — the per-sample log-RMS-driven inner Dense replaces it).
        # ``band_regularizer`` IS accepted by the layer but the campaign
        # disables the silent in-source default (matching the band_rms knob).
        kwargs = {
            "epsilon": epsilon,
            "max_band_width": max_band_width,
        }
        if band_regularizer_l2 is None:
            kwargs["band_regularizer"] = None
        else:
            import keras
            kwargs["band_regularizer"] = keras.regularizers.L2(band_regularizer_l2)
        return kwargs
    if norm_type == "band_logit_norm":
        # BandLogitNorm: no ``use_scale``, no ``band_regularizer``. The variant
        # uses an inner LayerNormalization whose gamma/beta are the only
        # trainable weights, and exposes only ``max_band_width`` / ``epsilon``.
        return {
            "epsilon": epsilon,
            "max_band_width": max_band_width,
        }
    if norm_type == "dynamic_tanh":
        # DynamicTanh: no ``epsilon`` (the factory strips it explicitly), no
        # ``use_scale``. Expose only ``axis`` and ``alpha_init_value``. The
        # campaign uses the layer's default ``alpha_init_value=0.5``.
        return {
            "axis": -1,
            "alpha_init_value": 0.5,
        }
    # Conservative fallback for ``layer_norm`` / other factory keys.
    return {"epsilon": epsilon}


# ---------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Single-run experiment configuration.

    One instance per ``(experiment, norm_type, mode, seed)`` cell driven by
    the sweep. All fields are CLI-overridable in each trainer's argparse.
    """

    # --- Identity ----------------------------------------------------------
    experiment_name: str
    norm_type: str  # one of NORM_VARIANTS (or any factory key for ablations)
    seed: int = 0
    # OOB | param_matched. See ``build_norm_kwargs``.
    mode: str = "oob"

    # --- Model / data -----------------------------------------------------
    model_variant: str = "default"  # e.g. "vit_pico" / "resnet18" / "tiny"
    dataset: str = "synthetic"

    # --- Training ---------------------------------------------------------
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    mixed_precision: bool = False

    # --- Norm-specific (consumed by build_norm_kwargs) --------------------
    use_scale: bool = True
    max_band_width: float = 0.1
    epsilon: float = 1e-6
    band_regularizer_l2: Optional[float] = None

    # --- Probe / callback config ------------------------------------------
    enable_epoch_analyzer: bool = True
    analyzer_frequency: int = 5
    grad_norm_sample_every_n_batches: int = 10
    activation_calibration_batches: int = 2

    # --- Output -----------------------------------------------------------
    out_dir: str = "results/rms_variants_train"
    csv_filename: str = "results.csv"

    # --- Misc -------------------------------------------------------------
    # Free-form extras for trainer-specific knobs (e.g. depth_override).
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def norm_kwargs(self) -> Dict[str, Any]:
        """Resolve the per-variant kwargs for :func:`create_normalization_layer`.

        Mode logic: ``param_matched`` forces ``use_scale=False`` on
        RMSNorm-family variants to drop their per-feature scale parameter,
        making the trainable param count 0 to match BandRMS variants' 1 scalar.
        """
        effective_use_scale = self.use_scale and (self.mode == "oob")
        return build_norm_kwargs(
            self.norm_type,
            use_scale=effective_use_scale,
            max_band_width=self.max_band_width,
            epsilon=self.epsilon,
            band_regularizer_l2=self.band_regularizer_l2,
        )


__all__ = ["ExperimentConfig", "NORM_VARIANTS", "build_norm_kwargs"]
