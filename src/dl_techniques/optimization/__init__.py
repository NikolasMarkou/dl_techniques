"""
Optimization Module for Deep Learning Techniques.

This module provides comprehensive optimization utilities including:
- Optimizer builders with gradient clipping support
- Learning rate schedules with warmup periods
- Deep supervision weight scheduling

All builders support configuration-driven setup with sensible defaults
and comprehensive error handling.

Available Functions:
    - optimizer_builder: Creates optimizers (Adam, AdamW, RMSprop, Adadelta)
    - learning_rate_schedule_builder: Creates LR schedules with warmup
    - deep_supervision_schedule_builder: Creates deep supervision weights

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
    >>> # Build components
    >>> lr_schedule = learning_rate_schedule_builder(lr_config)
    >>> optimizer = optimizer_builder(opt_config, lr_schedule)
    >>> ds_scheduler = deep_supervision_schedule_builder(ds_config, 5)
"""

from .optimizer import optimizer_builder
from .schedule import schedule_builder as learning_rate_schedule_builder
from .deep_supervision import schedule_builder as deep_supervision_schedule_builder

__all__ = [
    "optimizer_builder",
    "learning_rate_schedule_builder", 
    "deep_supervision_schedule_builder"
]