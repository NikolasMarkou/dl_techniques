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

    This configuration provides recommended settings for training MoE models,
    including optimizer choices, learning rate schedules, and regularization
    optimized for the FFN-only expert system.

    Attributes:
        optimizer_type: Type of optimizer to use ('adamw' recommended for MoE).
        base_learning_rate: Base learning rate for the model.
        expert_learning_rate_multiplier: Learning rate multiplier for expert FFN parameters.
        gating_learning_rate_multiplier: Learning rate multiplier for gating parameters.
        warmup_steps: Number of warmup steps.
        decay_steps: Number of decay steps for learning rate schedule.
        weight_decay: Weight decay coefficient (important for FFN experts).
        gradient_clipping_norm: Global gradient clipping norm.
        aux_loss_weight: Weight for auxiliary load balancing loss.
        z_loss_weight: Weight for router z-loss.
        dropout_rate: Dropout rate for FFN experts.
        use_mixed_precision: Whether to use mixed precision training.

    Example:
        ```python
        config = MoETrainingConfig(
            base_learning_rate=1e-4,
            expert_learning_rate_multiplier=0.1,  # Lower LR for FFN experts
            warmup_steps=1000,
            aux_loss_weight=0.01
        )
        ```
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

    This class integrates with the dl_techniques optimization module to create
    optimizers with MoE-specific configurations, including different learning
    rates for FFN expert parameters vs. gating parameters.

    Example:
        ```python
        builder = MoEOptimizerBuilder()
        optimizer = builder.build_moe_optimizer(
            model=model,
            config=MoETrainingConfig(base_learning_rate=1e-4)
        )
        ```
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

        Args:
            model: The model containing MoE layers.
            config: Training configuration for MoE.

        Returns:
            Configured optimizer with MoE-specific settings.
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
