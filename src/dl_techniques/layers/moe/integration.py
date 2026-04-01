"""
Integration utilities for MoE module with dl_techniques framework.

This module provides integration points between the MoE module and other
dl_techniques components, including optimizer integration, analyzer hooks,
and model conversion utilities. Updated to work with the simplified FFN-only
expert system.
"""

import keras
from dataclasses import dataclass

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder, learning_rate_schedule_builder
)

from .layer import MixtureOfExperts

# ---------------------------------------------------------------------

@dataclass
class MoETrainingConfig:
    """
    Training configuration specifically optimized for MoE models.

    Provides recommended settings for training MoE models, including optimizer
    choices, learning rate schedules, and regularization optimized for the
    FFN-only expert system.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │        MoETrainingConfig             │
        │                                      │
        │  optimizer_type ──► AdamW (default)  │
        │  base_learning_rate                  │
        │  expert_lr_multiplier                │
        │  gating_lr_multiplier                │
        │  warmup_steps / decay_steps          │
        │  aux_loss_weight / z_loss_weight     │
        └──────────────────────────────────────┘

    :param optimizer_type: Type of optimizer (``'adamw'`` recommended for MoE).
    :type optimizer_type: str
    :param base_learning_rate: Base learning rate for the model.
    :type base_learning_rate: float
    :param expert_learning_rate_multiplier: LR multiplier for expert FFN parameters.
    :type expert_learning_rate_multiplier: float
    :param gating_learning_rate_multiplier: LR multiplier for gating parameters.
    :type gating_learning_rate_multiplier: float
    :param warmup_steps: Number of warmup steps.
    :type warmup_steps: int
    :param decay_steps: Number of decay steps for learning rate schedule.
    :type decay_steps: int
    :param weight_decay: Weight decay coefficient (important for FFN experts).
    :type weight_decay: float
    :param gradient_clipping_norm: Global gradient clipping norm.
    :type gradient_clipping_norm: float
    :param aux_loss_weight: Weight for auxiliary load balancing loss.
    :type aux_loss_weight: float
    :param z_loss_weight: Weight for router z-loss.
    :type z_loss_weight: float
    :param dropout_rate: Dropout rate for FFN experts.
    :type dropout_rate: float
    :param use_mixed_precision: Whether to use mixed precision training.
    :type use_mixed_precision: bool
    """
    optimizer_type: str = 'adamw'
    base_learning_rate: float = 1e-4
    expert_learning_rate_multiplier: float = 1.0
    gating_learning_rate_multiplier: float = 1.0
    warmup_steps: int = 1000
    decay_steps: int = 10000
    weight_decay: float = 0.01
    gradient_clipping_norm: float = 1.0
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 1e-3
    dropout_rate: float = 0.1
    use_mixed_precision: bool = False

# ---------------------------------------------------------------------

class MoEOptimizerBuilder:
    """
    Builder for creating optimizers optimized for MoE training with FFN experts.

    Integrates with the dl_techniques optimization module to create optimizers
    with MoE-specific configurations, including differential learning rates for
    FFN expert parameters vs. gating parameters.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────────┐
        │          MoEOptimizerBuilder                  │
        │                                               │
        │  MoETrainingConfig ──► LR Schedule Builder    │
        │                    ──► Optimizer Builder       │
        │                    ──► LR Multipliers          │
        │                         ├─ expert params       │
        │                         └─ gating params       │
        │                                               │
        │  Output: Configured Optimizer                  │
        └───────────────────────────────────────────────┘
    """

    def __init__(self):
        """Initialize the MoE optimizer builder."""
        pass

    def build_moe_optimizer(
            self,
            model: keras.Model,
            config: MoETrainingConfig
    ) -> keras.optimizers.Optimizer:
        """
        Build an optimizer optimized for MoE training with FFN experts.

        :param model: The model containing MoE layers.
        :type model: keras.Model
        :param config: Training configuration for MoE.
        :type config: MoETrainingConfig
        :return: Configured optimizer with MoE-specific settings.
        :rtype: keras.optimizers.Optimizer
        """
        # Create learning rate schedule
        lr_schedule_config = {
            "type": "cosine_decay",
            "warmup_steps": config.warmup_steps,
            "warmup_start_lr": 1e-8,
            "learning_rate": config.base_learning_rate,
            "decay_steps": config.decay_steps,
            "alpha": 0.1
        }

        lr_schedule = learning_rate_schedule_builder(lr_schedule_config)

        # Create optimizer
        optimizer_config = {
            "type": config.optimizer_type,
            "beta_1": 0.9,
            "beta_2": 0.95,  # Higher beta_2 for MoE stability
            "gradient_clipping_by_norm": config.gradient_clipping_norm
        }

        # Add weight_decay for AdamW
        if config.optimizer_type.lower() == 'adamw':
            optimizer_config["weight_decay"] = config.weight_decay

        optimizer = optimizer_builder(optimizer_config, lr_schedule)

        # Apply learning rate multipliers for different parameter groups
        if hasattr(optimizer, 'learning_rate_multipliers'):
            self._apply_moe_learning_rate_multipliers(optimizer, model, config)
        else:
            logger.warning("Optimizer does not support learning rate multipliers")

        return optimizer

    def _apply_moe_learning_rate_multipliers(
            self,
            optimizer: keras.optimizers.Optimizer,
            model: keras.Model,
            config: MoETrainingConfig
    ) -> None:
        """Apply learning rate multipliers for MoE FFN parameters."""
        multipliers = {}

        # Find all variables in MoE layers
        for layer in model.layers:
            if isinstance(layer, MixtureOfExperts):
                # FFN expert parameters get expert multiplier
                for expert in layer.experts:
                    for var in expert.trainable_variables:
                        multipliers[var.name] = config.expert_learning_rate_multiplier

                # Gating parameters get gating multiplier
                if hasattr(layer, 'gating_network') and layer.gating_network:
                    for var in layer.gating_network.trainable_variables:
                        multipliers[var.name] = config.gating_learning_rate_multiplier

        if multipliers:
            optimizer.learning_rate_multipliers = multipliers
            logger.info(f"Applied learning rate multipliers to {len(multipliers)} MoE parameters")
        else:
            logger.warning("No MoE parameters found for learning rate multipliers")

# ---------------------------------------------------------------------
