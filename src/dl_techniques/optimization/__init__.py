"""
Optimization Module for Deep Learning Techniques.

This module provides comprehensive optimization utilities including:
- Optimizer builders with gradient clipping support
- Learning rate schedules with warmup periods
- Deep supervision weight scheduling
- The Spectrum-to-Signal Principle (SSP): diversity-first selection and fusion,
  and max-entropy-weighted group advantages

All builders support configuration-driven setup with sensible defaults
and comprehensive error handling.

Available Functions:
    - optimizer_builder: Creates optimizers (Adam, AdamW, SGD, RMSprop, Adadelta)
    - learning_rate_schedule_builder: Creates LR schedules with warmup
    - create_learning_rate_schedule: Epoch-facing cosine/exponential/constant LR
    - create_warmup_lr_schedule: Epoch-facing warmup-ratio + cosine LR
    - deep_supervision_schedule_builder: Creates deep supervision weights
    - ssp_builder: Creates a Spectrum-to-Signal strategy (see `ssp/README.md`)

Example Usage:
    >>> # Optimizer configuration
    >>> opt_config = {
    ...     "type": "adam",
    ...     "beta_1": 0.9,
    ...     "gradient_clipping_by_norm": 1.0
    ... }
    >>>
    >>> # Learning rate schedule configuration (flattened structure)
    >>> lr_config = {
    ...     "type": "cosine_decay",
    ...     "warmup_steps": 1000,
    ...     "warmup_start_lr": 1e-8,
    ...     "learning_rate": 0.001,
    ...     "decay_steps": 10000,
    ...     "alpha": 0.0001
    ... }
    >>>
    >>> # Deep supervision configuration
    >>> ds_config = {
    ...     "type": "linear_low_to_high",
    ...     "config": {}
    ... }
    >>>
    >>> # Spectrum-to-Signal configuration (master switch defaults to OFF)
    >>> ssp_config = {
    ...     "type": "ssp_v1",
    ...     "config": {"enable": True, "pass_at_k": 8, "mgpo_lambda": 2.0}
    ... }
    >>>
    >>> # Build components
    >>> lr_schedule = learning_rate_schedule_builder(lr_config)
    >>> optimizer = optimizer_builder(opt_config, lr_schedule)
    >>> ds_scheduler = deep_supervision_schedule_builder(ds_config, 5)
    >>> strategy = ssp_builder(ssp_config)
"""

from .optimizer import optimizer_builder
from .schedule import schedule_builder as learning_rate_schedule_builder
from .schedule import create_learning_rate_schedule, create_warmup_lr_schedule
from .warmup_schedule import WarmupSchedule
from .deep_supervision import schedule_builder as deep_supervision_schedule_builder
from .muon_optimizer import Muon
from .sgld_optimizer import SGLD
from .vsgd_optimizer import VSGD
from .gefen_optimizer import Gefen
from .ww_pgd_optimizer import WWTailConfig, ww_pgd_project, WWPGDProjectionCallback
from .ssp import (
    SSPStrategy,
    SSPStrategyConfig,
    ssp_builder,
    max_entropy_weight,
    mgpo_advantages,
    spectrum_profile,
    select_specialists,
    spectrum_sampling_weights,
    fuse_specialists,
    MGPOObjective,
)

__all__ = [
    "optimizer_builder",
    "learning_rate_schedule_builder",
    "create_learning_rate_schedule",
    "create_warmup_lr_schedule",
    "WarmupSchedule",
    "deep_supervision_schedule_builder",
    "Muon",
    "SGLD",
    "VSGD",
    "Gefen",
    "WWTailConfig",
    "ww_pgd_project",
    "WWPGDProjectionCallback",
    "ssp_builder",
    "SSPStrategy",
    "SSPStrategyConfig",
    "spectrum_profile",
    "select_specialists",
    "spectrum_sampling_weights",
    "fuse_specialists",
    "max_entropy_weight",
    "mgpo_advantages",
    "MGPOObjective",
]