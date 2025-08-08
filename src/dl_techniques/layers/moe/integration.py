"""
Integration utilities for MoE module with dl_techniques framework.

This module provides integration points between the MoE module and other
dl_techniques components, including optimizer integration, analyzer hooks,
and model conversion utilities.
"""

from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings
from dataclasses import dataclass

import keras
import keras.ops as ops
import numpy as np

from .layer import (
    MixtureOfExperts, MoEConfig, ExpertConfig, GatingConfig,
    create_ffn_moe, create_attention_moe, create_conv_moe
)
from dl_techniques.utils.logger import logger

from dl_techniques.optimization import (
        optimizer_builder, learning_rate_schedule_builder,
    )

from dl_techniques.analyzer import ModelAnalyzer

@dataclass
class MoETrainingConfig:
    """
    Training configuration specifically optimized for MoE models.

    This configuration provides recommended settings for training MoE models,
    including optimizer choices, learning rate schedules, and regularization.

    Attributes:
        optimizer_type: Type of optimizer to use ('adamw' recommended for MoE).
        base_learning_rate: Base learning rate for the model.
        expert_learning_rate_multiplier: Learning rate multiplier for expert parameters.
        gating_learning_rate_multiplier: Learning rate multiplier for gating parameters.
        warmup_steps: Number of warmup steps.
        decay_steps: Number of decay steps for learning rate schedule.
        weight_decay: Weight decay coefficient.
        gradient_clipping_norm: Global gradient clipping norm.
        aux_loss_weight: Weight for auxiliary load balancing loss.
        z_loss_weight: Weight for router z-loss.
        dropout_rate: Dropout rate for experts.
        use_mixed_precision: Whether to use mixed precision training.

    Example:
        ```python
        config = MoETrainingConfig(
            base_learning_rate=1e-4,
            expert_learning_rate_multiplier=0.1,  # Lower LR for experts
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


class MoEOptimizerBuilder:
    """
    Builder for creating optimizers optimized for MoE training.

    This class integrates with the dl_techniques optimization module to create
    optimizers with MoE-specific configurations, including different learning
    rates for different parameter groups.

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
        if not HAS_OPTIMIZATION:
            raise ImportError("dl_techniques.optimization required for MoEOptimizerBuilder")

    def build_moe_optimizer(
            self,
            model: keras.Model,
            config: MoETrainingConfig
    ) -> keras.optimizers.Optimizer:
        """
        Build an optimizer optimized for MoE training.

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
            "weight_decay": config.weight_decay,
            "gradient_clipping_by_norm": config.gradient_clipping_norm
        }

        optimizer = optimizer_builder(optimizer_config, lr_schedule)

        # Apply learning rate multipliers for different parameter groups
        if hasattr(optimizer, 'learning_rate_multipliers'):
            self._apply_moe_learning_rate_multipliers(
                optimizer, model, config
            )

        return optimizer

    def _apply_moe_learning_rate_multipliers(
            self,
            optimizer: keras.optimizers.Optimizer,
            model: keras.Model,
            config: MoETrainingConfig
    ) -> None:
        """Apply learning rate multipliers for MoE parameters."""
        multipliers = {}

        # Find all variables in MoE layers
        for layer in model.layers:
            if isinstance(layer, MixtureOfExperts):
                # Expert parameters get expert multiplier
                for expert in layer.experts:
                    for var in expert.trainable_variables:
                        multipliers[var.name] = config.expert_learning_rate_multiplier

                # Gating parameters get gating multiplier
                for var in layer.gating_network.trainable_variables:
                    multipliers[var.name] = config.gating_learning_rate_multiplier

        optimizer.learning_rate_multipliers = multipliers
        logger.info(f"Applied learning rate multipliers to {len(multipliers)} MoE parameters")


class MoEAnalyzerIntegration:
    """
    Integration between MoE module and dl_techniques analyzer.

    This class extends the standard model analyzer to provide MoE-specific
    analysis capabilities, including expert utilization tracking and
    routing pattern analysis.

    Example:
        ```python
        integration = MoEAnalyzerIntegration()
        analyzer = integration.create_moe_analyzer(model)
        results = analyzer.analyze(data)
        ```
    """

    def __init__(self):
        """Initialize the MoE analyzer integration."""
        if not HAS_ANALYZER:
            warnings.warn("dl_techniques.analyzer not available - limited functionality")

    def create_moe_analyzer(
            self,
            model: keras.Model,
            config: Optional[Dict[str, Any]] = None
    ) -> 'ModelAnalyzer':
        """
        Create an analyzer with MoE-specific configurations.

        Args:
            model: Model to analyze.
            config: Additional analyzer configuration.

        Returns:
            Configured ModelAnalyzer with MoE extensions.
        """
        if not HAS_ANALYZER:
            raise ImportError("dl_techniques.analyzer required for MoE analysis")

        # Default MoE analyzer configuration
        default_config = {
            'analyze_weights': True,
            'analyze_training_dynamics': True,
            'moe_specific_analysis': True,
            'track_expert_utilization': True,
            'track_routing_patterns': True
        }

        if config:
            default_config.update(config)

        # Create analyzer with MoE extensions
        analyzer = ModelAnalyzer(model, default_config)

        # Add MoE-specific analysis hooks
        self._add_moe_analysis_hooks(analyzer)

        return analyzer

    def _add_moe_analysis_hooks(self, analyzer: 'ModelAnalyzer') -> None:
        """Add MoE-specific analysis hooks to the analyzer."""
        # This would add custom analysis modules for MoE
        # Implementation would depend on the analyzer's extension API
        logger.info("Added MoE-specific analysis hooks to ModelAnalyzer")


class MoEModelConverter:
    """
    Utilities for converting between different MoE formats and architectures.

    This class provides methods for converting dense models to MoE models,
    converting between different MoE configurations, and extracting/merging
    expert knowledge.

    Example:
        ```python
        converter = MoEModelConverter()
        moe_model = converter.convert_dense_to_moe(
            dense_model,
            target_layers=['transformer_block_*/ffn']
        )
        ```
    """

    def __init__(self):
        """Initialize the MoE model converter."""
        self.conversion_history = []

    def convert_dense_to_moe(
            self,
            dense_model: keras.Model,
            target_layers: List[str],
            moe_config: MoEConfig,
            preserve_weights: bool = True
    ) -> keras.Model:
        """
        Convert dense model to MoE by replacing specified layers.

        Args:
            dense_model: Original dense model.
            target_layers: List of layer name patterns to replace.
            moe_config: MoE configuration for replacement layers.
            preserve_weights: Whether to preserve original weights in experts.

        Returns:
            New model with MoE layers replacing target layers.
        """
        logger.info(f"Converting dense model to MoE: {dense_model.name}")

        # Clone the model to avoid modifying the original
        model_config = dense_model.get_config()
        new_model_config = self._replace_layers_in_config(
            model_config, target_layers, moe_config
        )

        # Create new model with MoE layers
        new_model = keras.Model.from_config(new_model_config)

        if preserve_weights:
            self._transfer_weights_to_moe(dense_model, new_model, target_layers)

        self.conversion_history.append({
            'original_model': dense_model.name,
            'converted_model': new_model.name,
            'target_layers': target_layers,
            'moe_config': moe_config.to_dict()
        })

        logger.info(f"Conversion completed: {len(target_layers)} layers converted to MoE")
        return new_model

    def _replace_layers_in_config(
            self,
            model_config: Dict[str, Any],
            target_layers: List[str],
            moe_config: MoEConfig
    ) -> Dict[str, Any]:
        """Replace layer configurations with MoE configurations."""
        # This is a simplified implementation
        # A complete version would recursively traverse the model config
        # and replace matching layer configurations

        new_config = model_config.copy()

        # Find and replace target layers
        if 'layers' in new_config:
            for i, layer_config in enumerate(new_config['layers']):
                layer_name = layer_config.get('config', {}).get('name', '')

                # Check if this layer matches any target patterns
                if any(self._layer_name_matches_pattern(layer_name, pattern)
                       for pattern in target_layers):
                    # Replace with MoE layer configuration
                    moe_layer_config = self._create_moe_layer_config(moe_config, layer_name)
                    new_config['layers'][i] = moe_layer_config

        return new_config

    def _layer_name_matches_pattern(self, layer_name: str, pattern: str) -> bool:
        """Check if layer name matches a pattern (supports wildcards)."""
        import re
        pattern_regex = pattern.replace('*', '.*')
        return bool(re.match(pattern_regex, layer_name))

    def _create_moe_layer_config(
            self,
            moe_config: MoEConfig,
            layer_name: str
    ) -> Dict[str, Any]:
        """Create Keras layer config for MoE layer."""
        return {
            'class_name': 'MixtureOfExperts',
            'config': {
                'name': f'moe_{layer_name}',
                'config': moe_config.to_dict()
            }
        }

    def _transfer_weights_to_moe(
            self,
            source_model: keras.Model,
            target_model: keras.Model,
            target_layers: List[str]
    ) -> None:
        """Transfer weights from dense layers to MoE experts."""
        logger.info("Transferring weights from dense layers to MoE experts...")

        # Find corresponding layers and transfer weights
        for source_layer in source_model.layers:
            if any(self._layer_name_matches_pattern(source_layer.name, pattern)
                   for pattern in target_layers):

                # Find corresponding MoE layer in target model
                target_layer_name = f'moe_{source_layer.name}'
                target_layer = self._find_layer_by_name(target_model, target_layer_name)

                if target_layer and isinstance(target_layer, MixtureOfExperts):
                    self._initialize_moe_experts_from_dense(source_layer, target_layer)

    def _find_layer_by_name(self, model: keras.Model, name: str) -> Optional[keras.layers.Layer]:
        """Find layer by name in model."""
        for layer in model.layers:
            if layer.name == name:
                return layer
        return None

    def _initialize_moe_experts_from_dense(
            self,
            dense_layer: keras.layers.Layer,
            moe_layer: MixtureOfExperts
    ) -> None:
        """Initialize MoE experts with weights from dense layer."""
        if not hasattr(dense_layer, 'get_weights') or not dense_layer.get_weights():
            return

        dense_weights = dense_layer.get_weights()

        # Initialize each expert with the dense layer weights
        # Add small random perturbations to encourage specialization
        for expert in moe_layer.experts:
            expert_weights = []
            for weight in dense_weights:
                # Add small random noise to break symmetry
                noise = np.random.normal(0, 0.01, weight.shape)
                perturbed_weight = weight + noise.astype(weight.dtype)
                expert_weights.append(perturbed_weight)

            if expert_weights:
                expert.set_weights(expert_weights)

        logger.info(f"Initialized {len(moe_layer.experts)} experts from dense layer {dense_layer.name}")

    def merge_expert_knowledge(
            self,
            moe_model: keras.Model,
            target_layers: List[str],
            merge_strategy: str = 'weighted_average'
    ) -> keras.Model:
        """
        Merge expert knowledge back into dense layers.

        Args:
            moe_model: Model with MoE layers.
            target_layers: MoE layer names to merge.
            merge_strategy: Strategy for merging expert weights.

        Returns:
            Model with dense layers replacing MoE layers.
        """
        logger.info(f"Merging expert knowledge using strategy: {merge_strategy}")

        # Clone model configuration
        model_config = moe_model.get_config()
        dense_config = self._replace_moe_with_dense_in_config(model_config, target_layers)

        # Create dense model
        dense_model = keras.Model.from_config(dense_config)

        # Merge expert weights
        for layer_name in target_layers:
            moe_layer = self._find_layer_by_name(moe_model, layer_name)
            dense_layer_name = layer_name.replace('moe_', '')
            dense_layer = self._find_layer_by_name(dense_model, dense_layer_name)

            if moe_layer and dense_layer:
                merged_weights = self._merge_expert_weights(
                    moe_layer, merge_strategy
                )
                if merged_weights:
                    dense_layer.set_weights(merged_weights)

        return dense_model

    def _replace_moe_with_dense_in_config(
            self,
            model_config: Dict[str, Any],
            target_layers: List[str]
    ) -> Dict[str, Any]:
        """Replace MoE layer configs with dense layer configs."""
        # Simplified implementation - would need complete layer config mapping
        return model_config

    def _merge_expert_weights(
            self,
            moe_layer: MixtureOfExperts,
            strategy: str
    ) -> Optional[List[np.ndarray]]:
        """Merge expert weights using specified strategy."""
        if not moe_layer.experts:
            return None

        expert_weights = [expert.get_weights() for expert in moe_layer.experts]

        if strategy == 'weighted_average':
            # Simple average for demonstration
            if expert_weights and expert_weights[0]:
                merged = []
                for weight_idx in range(len(expert_weights[0])):
                    weights_at_idx = [ew[weight_idx] for ew in expert_weights]
                    avg_weight = np.mean(weights_at_idx, axis=0)
                    merged.append(avg_weight)
                return merged
        elif strategy == 'best_expert':
            # Use weights from the "best" expert (first one for simplicity)
            return expert_weights[0] if expert_weights else None

        return None


def create_moe_model_zoo_entry(
        name: str,
        moe_config: MoEConfig,
        architecture_fn: Callable[[MoEConfig], keras.Model],
        description: str,
        paper_reference: Optional[str] = None,
        pretrained_weights: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a model zoo entry for an MoE architecture.

    Args:
        name: Name of the model architecture.
        moe_config: MoE configuration for the model.
        architecture_fn: Function that creates the model given the config.
        description: Description of the model architecture.
        paper_reference: Reference to the paper introducing this architecture.
        pretrained_weights: URL or path to pretrained weights.

    Returns:
        Model zoo entry dictionary.

    Example:
        ```python
        entry = create_moe_model_zoo_entry(
            name="switch_transformer_base",
            moe_config=SWITCH_TRANSFORMER_CONFIG,
            architecture_fn=create_switch_transformer,
            description="Switch Transformer with 128 experts",
            paper_reference="Switch Transformer: Scaling to Trillion Parameter Models"
        )
        ```
    """
    return {
        'name': name,
        'type': 'moe_model',
        'config': moe_config.to_dict(),
        'architecture_function': architecture_fn,
        'description': description,
        'paper_reference': paper_reference,
        'pretrained_weights': pretrained_weights,
        'tags': ['mixture_of_experts', 'sparse_model', 'scalable'],
        'creation_function': lambda: architecture_fn(moe_config)
    }


# Pre-defined model zoo entries for common MoE architectures

SWITCH_TRANSFORMER_BASE_CONFIG = MoEConfig(
    num_experts=128,
    expert_config=ExpertConfig(
        expert_type='ffn',
        hidden_dim=768,
        intermediate_size=3072,
        activation='relu'
    ),
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=1,
        aux_loss_weight=0.01
    )
)

GLAM_CONFIG = MoEConfig(
    num_experts=64,
    expert_config=ExpertConfig(
        expert_type='ffn',
        hidden_dim=1024,
        intermediate_size=4096,
        activation='gelu'
    ),
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=2,
        aux_loss_weight=0.1
    )
)

MIXTURE_OF_ATTENTION_CONFIG = MoEConfig(
    num_experts=16,
    expert_config=ExpertConfig(
        expert_type='attention',
        hidden_dim=768,
        num_heads=12,
        head_dim=64
    ),
    gating_config=GatingConfig(
        gating_type='cosine',
        top_k=2,
        aux_loss_weight=0.01
    )
)

VISION_MOE_CONFIG = MoEConfig(
    num_experts=8,
    expert_config=ExpertConfig(
        expert_type='conv2d',
        filters=256,
        kernel_size=3,
        activation='relu'
    ),
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=1,
        add_noise=False
    )
)


def get_recommended_moe_training_config(
        model_size: str = 'base',
        task_type: str = 'language_modeling'
) -> MoETrainingConfig:
    """
    Get recommended training configuration for MoE models.

    Args:
        model_size: Size of the model ('small', 'base', 'large', 'xl').
        task_type: Type of task ('language_modeling', 'classification', 'vision').

    Returns:
        Recommended training configuration.

    Example:
        ```python
        config = get_recommended_moe_training_config('large', 'language_modeling')
        optimizer = MoEOptimizerBuilder().build_moe_optimizer(model, config)
        ```
    """
    base_configs = {
        'small': MoETrainingConfig(
            base_learning_rate=1e-3,
            warmup_steps=500,
            decay_steps=5000,
            weight_decay=0.1
        ),
        'base': MoETrainingConfig(
            base_learning_rate=1e-4,
            warmup_steps=1000,
            decay_steps=10000,
            weight_decay=0.01
        ),
        'large': MoETrainingConfig(
            base_learning_rate=5e-5,
            warmup_steps=2000,
            decay_steps=20000,
            weight_decay=0.01,
            expert_learning_rate_multiplier=0.1
        ),
        'xl': MoETrainingConfig(
            base_learning_rate=1e-5,
            warmup_steps=4000,
            decay_steps=40000,
            weight_decay=0.01,
            expert_learning_rate_multiplier=0.1,
            use_mixed_precision=True
        )
    }

    config = base_configs.get(model_size, base_configs['base'])

    # Task-specific adjustments
    if task_type == 'vision':
        config.aux_loss_weight = 0.001  # Lower for vision tasks
        config.dropout_rate = 0.0
    elif task_type == 'classification':
        config.warmup_steps = config.warmup_steps // 2  # Less warmup for classification

    return config


def validate_moe_integration() -> bool:
    """
    Validate that MoE integration components are working correctly.

    Returns:
        True if all integrations are working, False otherwise.
    """
    validation_results = {
        'optimizer_integration': False,
        'analyzer_integration': False,
        'converter_functionality': False
    }

    try:
        # Test optimizer integration
        if HAS_OPTIMIZATION:
            config = MoETrainingConfig()
            builder = MoEOptimizerBuilder()
            # Create dummy model for testing
            dummy_model = keras.Sequential([
                keras.layers.Input(shape=(64,)),
                create_ffn_moe(num_experts=4, hidden_dim=64),
                keras.layers.Dense(10)
            ])
            optimizer = builder.build_moe_optimizer(dummy_model, config)
            validation_results['optimizer_integration'] = True
            logger.info("✓ MoE optimizer integration working")

        # Test analyzer integration
        if HAS_ANALYZER:
            integration = MoEAnalyzerIntegration()
            dummy_model = keras.Sequential([keras.layers.Dense(10)])
            analyzer = integration.create_moe_analyzer(dummy_model)
            validation_results['analyzer_integration'] = True
            logger.info("✓ MoE analyzer integration working")

        # Test converter functionality
        converter = MoEModelConverter()
        validation_results['converter_functionality'] = True
        logger.info("✓ MoE converter functionality working")

    except Exception as e:
        logger.error(f"MoE integration validation failed: {str(e)}")
        return False

    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())

    logger.info(f"MoE integration validation: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests