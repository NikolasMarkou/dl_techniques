"""
RMSNorm Variants Comprehensive Study.

A multi-experiment harness comparing four normalization variants:
  - RMSNorm (baseline)
  - BandRMS
  - ZeroCenteredRMSNorm
  - ZeroCenteredBandRMSNorm

Across 5 experiments (ViT/CIFAR-10, ResNet/CIFAR-100, TinyTransformer/IMDb,
deep-residual+fp16 regression, layer-level microbench), 4 mechanistic probe
callbacks (gradient norm, weight norm trajectory, norm-layer activation stats,
norm internal stats), and dual OOB / parameter-matched modes.

Public surface
--------------
- :class:`ExperimentConfig` — dataclass capturing every knob for a single run.
- :data:`NORM_VARIANTS` — tuple of the four norm-type strings under study.
- :func:`build_norm_kwargs` — produces a kwargs dict for ``create_normalization_layer``
  conditioned on the variant and parameter-parity mode.
- :func:`set_seeds` — single-call seeder for Python / NumPy / TF / Keras RNGs.

Plan: ``plans/plan_2026-05-14_3764496e``  (see plan.md for the experiment matrix).
"""
from __future__ import annotations

from .config import ExperimentConfig, NORM_VARIANTS, build_norm_kwargs
from .seed_utils import set_seeds

__all__ = [
    "ExperimentConfig",
    "NORM_VARIANTS",
    "build_norm_kwargs",
    "set_seeds",
]
