"""
Main Mixture of Experts (MoE) layer implementation.

This module provides the core MoE layer that combines expert networks and
gating mechanisms to create sparse, scalable neural network architectures.
"""

import math
import keras
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict, List

from .experts import BaseExpert, create_expert
from .config import MoEConfig, ExpertConfig, GatingConfig
from .gating import BaseGating, create_gating, compute_auxiliary_loss, compute_z_loss

from dl_techniques.utils.logger import logger


class MixtureOfExperts(keras.layers.Layer):
    """
    Mixture of Experts (MoE) layer for sparse neural networks.

    This layer implements a complete MoE mechanism that routes inputs to a subset
    of expert networks based on learned gating functions. It supports various
    expert types, routing strategies, and load balancing mechanisms.

    The MoE layer replaces dense layers (like FFN blocks) with a sparse alternative
    where only a subset of experts are activated for each input, providing
    computational efficiency and model specialization.

    Args:
        config: MoE configuration containing expert and gating settings.
        **kwargs: Additional keyword arguments for the base Layer class.

    Input shape:
        Depends on expert type:
        - FFN experts: (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
        - Attention experts: (batch_size, seq_len, hidden_dim)
        - Conv2D experts: (batch_size, height, width, channels)

    Output shape:
        Same as input shape, with last dimension potentially modified by experts.

    Example:
        ```python
        # Basic FFN MoE
        config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(expert_type='ffn', hidden_dim=768),
            gating_config=GatingConfig(gating_type='linear', top_k=2)
        )
        moe_layer = MixtureOfExperts(config)

        # Attention MoE
        config = MoEConfig(
            num_experts=12,
            expert_config=ExpertConfig(expert_type='attention', hidden_dim=768, num_heads=12),
            gating_config=GatingConfig(gating_type='cosine', top_k=1)
        )
        attention_moe = MixtureOfExperts(config)
        ```

    References:
        - Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
        - GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
        - PaLM: Scaling Language Modeling with Pathways
    """

    def __init__(self, config: MoEConfig, **kwargs: Any) -> None:
        """Initialize the MoE layer."""
        super().__init__(**kwargs)

        self.config = config
        self.num_experts = config.num_experts
        self.expert_config = config.expert_config
        self.gating_config = config.gating_config

        # Training parameters
        self.train_capacity_factor = config.train_capacity_factor
        self.eval_capacity_factor = config.eval_capacity_factor
        self.drop_tokens = config.drop_tokens
        self.use_residual_connection = config.use_residual_connection
        self.jitter_noise = config.jitter_noise

        # Expert and gating networks initialized in build()
        self.experts: List[BaseExpert] = []
        self.gating_network: Optional[BaseGating] = None

        # Capacity and routing parameters
        self._expert_capacity_train = None
        self._expert_capacity_eval = None
        self._built_input_shape = None

        # Auxiliary loss tracking
        self._auxiliary_losses = []

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the MoE layer components."""
        self._built_input_shape = input_shape

        # Create expert networks
        logger.info(f"Building {self.num_experts} {self.expert_config.expert_type} experts")
        self.experts = []

        for i in range(self.num_experts):
            expert = create_expert(
                expert_type=self.expert_config.expert_type,
                hidden_dim=self.expert_config.hidden_dim,
                output_dim=self.expert_config.output_dim,
                activation=self.expert_config.activation,
                dropout_rate=self.expert_config.dropout_rate,
                use_bias=self.expert_config.use_bias,
                kernel_initializer=self.expert_config.kernel_initializer,
                bias_initializer=self.expert_config.bias_initializer,
                kernel_regularizer=self.expert_config.kernel_regularizer,
                bias_regularizer=self.expert_config.bias_regularizer,
                # Expert-specific parameters
                intermediate_size=self.expert_config.intermediate_size,
                num_heads=self.expert_config.num_heads,
                head_dim=self.expert_config.head_dim,
                filters=self.expert_config.filters,
                kernel_size=self.expert_config.kernel_size,
                strides=self.expert_config.strides,
                padding=self.expert_config.padding,
                name=f'expert_{i}'
            )
            expert.build(input_shape)
            self.experts.append(expert)

        # Create gating network
        logger.info(f"Building {self.gating_config.gating_type} gating network")
        self.gating_network = create_gating(
            gating_type=self.gating_config.gating_type,
            num_experts=self.num_experts,
            top_k=self.gating_config.top_k,
            capacity_factor=self.gating_config.capacity_factor,
            add_noise=self.gating_config.add_noise,
            noise_std=self.gating_config.noise_std,
            temperature=self.gating_config.temperature,
            use_bias=self.gating_config.use_bias,
            embedding_dim=self.gating_config.embedding_dim,
            learnable_temperature=self.gating_config.learnable_temperature,
            num_slots=self.gating_config.num_slots,
            name='gating_network'
        )
        self.gating_network.build(input_shape)

        # Calculate expert capacities
        self._calculate_expert_capacities(input_shape)

        super().build(input_shape)

    def _calculate_expert_capacities(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Calculate expert capacities based on input shape and config."""
        # Estimate tokens per batch (approximate for dynamic shapes)
        if len(input_shape) > 2:
            # Sequence-based input (batch_size, seq_len, hidden_dim)
            approx_tokens_per_batch = 512  # Default assumption
        else:
            # Single token input (batch_size, hidden_dim)
            approx_tokens_per_batch = 32  # Default assumption

        # Base capacity per expert
        base_capacity = math.ceil(approx_tokens_per_batch / self.num_experts)

        # Apply capacity factors
        self._expert_capacity_train = max(1, int(base_capacity * self.train_capacity_factor))
        self._expert_capacity_eval = max(1, int(base_capacity * self.eval_capacity_factor))

        logger.info(f"Expert capacities - Train: {self._expert_capacity_train}, Eval: {self._expert_capacity_eval}")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the MoE layer."""
        original_shape = ops.shape(inputs)
        batch_size = original_shape[0]

        # Handle different input shapes
        if len(original_shape) > 2:
            # Sequence input: flatten to (batch_size * seq_len, hidden_dim)
            seq_len = original_shape[1]
            hidden_dim = original_shape[-1]
            inputs_flat = ops.reshape(inputs, (batch_size * seq_len, hidden_dim))
            total_tokens = batch_size * seq_len
        else:
            # Single token input: (batch_size, hidden_dim)
            hidden_dim = original_shape[-1]
            inputs_flat = inputs
            total_tokens = batch_size
            seq_len = 1

        # Get expert capacity for current phase
        expert_capacity = self._expert_capacity_train if training else self._expert_capacity_eval

        # Add jitter noise during training
        if training and self.jitter_noise > 0:
            noise = keras.random.uniform(
                shape=ops.shape(inputs_flat),
                minval=-self.jitter_noise,
                maxval=self.jitter_noise,
                dtype=inputs_flat.dtype
            )
            inputs_flat = inputs_flat + noise

        # Compute gating
        expert_weights, expert_indices, gating_info = self.gating_network(
            inputs_flat, training=training
        )

        # Handle different gating types
        if self.gating_config.gating_type == 'softmoe':
            # SoftMoE: use pre-computed expert inputs
            outputs = self._process_softmoe(
                expert_inputs=gating_info['expert_inputs'],
                expert_weights=expert_weights,
                original_shape=original_shape,
                training=training
            )
        else:
            # Traditional hard routing
            outputs = self._process_hard_routing(
                inputs_flat=inputs_flat,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                expert_capacity=expert_capacity,
                original_shape=original_shape,
                training=training
            )

        # Compute and add auxiliary losses
        if training:
            self._compute_auxiliary_losses(gating_info, total_tokens)

        return outputs

    def _process_hard_routing(
            self,
            inputs_flat: keras.KerasTensor,
            expert_weights: keras.KerasTensor,
            expert_indices: keras.KerasTensor,
            expert_capacity: int,
            original_shape: Tuple[int, ...],
            training: bool
    ) -> keras.KerasTensor:
        """Process inputs through experts using hard routing."""
        total_tokens = ops.shape(inputs_flat)[0]
        hidden_dim = ops.shape(inputs_flat)[-1]

        # Initialize output tensor
        if self.expert_config.output_dim:
            output_dim = self.expert_config.output_dim
        else:
            output_dim = hidden_dim

        outputs = ops.zeros((total_tokens, output_dim), dtype=inputs_flat.dtype)

        # Create expert assignment mask
        top_k = self.gating_config.top_k
        expert_mask = ops.zeros((total_tokens, self.num_experts), dtype='bool')

        # Process each expert
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            if top_k == 1:
                # Single expert selection
                expert_tokens_mask = ops.equal(
                    ops.argmax(expert_weights, axis=-1),
                    expert_id
                )
            else:
                # Multi-expert selection
                expert_tokens_mask = ops.any(
                    ops.equal(expert_indices, expert_id),
                    axis=-1
                )

            # Apply capacity constraints
            if self.drop_tokens and expert_capacity > 0:
                expert_token_indices = ops.where(expert_tokens_mask)
                num_expert_tokens = ops.shape(expert_token_indices)[0]

                if num_expert_tokens > expert_capacity:
                    # Randomly select tokens to keep within capacity
                    selected_indices = keras.random.shuffle(expert_token_indices)[:expert_capacity]
                    expert_tokens_mask = ops.scatter_nd_update(
                        ops.zeros_like(expert_tokens_mask),
                        selected_indices,
                        True
                    )

            # Get tokens for this expert
            expert_tokens = ops.boolean_mask(inputs_flat, expert_tokens_mask)
            expert_token_weights = ops.boolean_mask(expert_weights[:, expert_id], expert_tokens_mask)

            if ops.shape(expert_tokens)[0] > 0:
                # Process tokens through expert
                expert_output = self.experts[expert_id](expert_tokens, training=training)

                # Apply expert weights
                weighted_output = expert_output * ops.expand_dims(expert_token_weights, axis=-1)

                # Scatter back to output tensor
                expert_token_indices = ops.where(expert_tokens_mask)
                outputs = ops.scatter_nd_add(
                    outputs,
                    expert_token_indices,
                    weighted_output
                )

                # Update expert mask
                expert_mask = ops.logical_or(
                    expert_mask,
                    ops.expand_dims(expert_tokens_mask, axis=-1)
                )

        # Handle dropped tokens with residual connection
        if self.use_residual_connection and self.drop_tokens:
            unprocessed_mask = ops.logical_not(ops.any(expert_mask, axis=-1))
            unprocessed_tokens = ops.boolean_mask(inputs_flat, unprocessed_mask)

            if ops.shape(unprocessed_tokens)[0] > 0:
                # Add residual connection for unprocessed tokens
                if hidden_dim == output_dim:
                    residual_output = unprocessed_tokens
                else:
                    # Project to correct output dimension
                    residual_output = ops.zeros(
                        (ops.shape(unprocessed_tokens)[0], output_dim),
                        dtype=inputs_flat.dtype
                    )

                unprocessed_indices = ops.where(unprocessed_mask)
                outputs = ops.scatter_nd_add(
                    outputs,
                    unprocessed_indices,
                    residual_output
                )

        # Reshape back to original shape
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [output_dim]
            outputs = ops.reshape(outputs, new_shape)

        return outputs

    def _process_softmoe(
            self,
            expert_inputs: keras.KerasTensor,
            expert_weights: keras.KerasTensor,
            original_shape: Tuple[int, ...],
            training: bool
    ) -> keras.KerasTensor:
        """Process inputs through experts using SoftMoE routing."""
        batch_size = original_shape[0]
        output_dim = self.expert_config.output_dim or original_shape[-1]

        # Process each expert's soft inputs
        expert_outputs = []

        for expert_id in range(self.num_experts):
            expert_input = expert_inputs[:, expert_id, :]  # [batch, slots * hidden_dim]
            expert_output = self.experts[expert_id](expert_input, training=training)
            expert_outputs.append(expert_output)

        # Stack expert outputs
        expert_outputs_stacked = ops.stack(expert_outputs, axis=1)  # [batch, num_experts, output_dim]

        # Weight and combine expert outputs
        if len(original_shape) > 2:
            seq_len = original_shape[1]
            expert_weights = ops.reshape(
                expert_weights,
                (batch_size, seq_len, self.num_experts, 1)
            )
            expert_outputs_expanded = ops.expand_dims(
                expert_outputs_stacked, axis=1
            )  # [batch, 1, num_experts, output_dim]

            # Weighted combination
            weighted_outputs = expert_outputs_expanded * expert_weights
            outputs = ops.sum(weighted_outputs, axis=2)  # [batch, seq_len, output_dim]
        else:
            expert_weights = ops.expand_dims(expert_weights, axis=-1)
            weighted_outputs = expert_outputs_stacked * expert_weights
            outputs = ops.sum(weighted_outputs, axis=1)  # [batch, output_dim]

        return outputs

    def _compute_auxiliary_losses(
            self,
            gating_info: Dict[str, keras.KerasTensor],
            total_tokens: int
    ) -> None:
        """Compute and add auxiliary losses for load balancing."""
        self._auxiliary_losses = []

        # Standard auxiliary loss
        if self.gating_config.aux_loss_weight > 0:
            aux_loss = compute_auxiliary_loss(
                expert_weights=gating_info['expert_weights'],
                gate_probs=gating_info['raw_gate_probs'],
                num_experts=self.num_experts,
                aux_loss_weight=self.gating_config.aux_loss_weight
            )
            self._auxiliary_losses.append(aux_loss)
            self.add_loss(aux_loss)

        # Router z-loss
        if self.gating_config.z_loss_weight > 0 and 'gate_logits' in gating_info:
            z_loss = compute_z_loss(
                gate_logits=gating_info['gate_logits'],
                z_loss_weight=self.gating_config.z_loss_weight
            )
            self._auxiliary_losses.append(z_loss)
            self.add_loss(z_loss)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the MoE layer."""
        output_shape = list(input_shape)
        if self.expert_config.output_dim:
            output_shape[-1] = self.expert_config.output_dim
        return tuple(output_shape)

    def get_expert_utilization(self) -> Dict[str, Any]:
        """Get statistics about expert utilization."""
        # This would need to be computed during forward pass
        # For now, return placeholder
        return {
            'num_experts': self.num_experts,
            'expert_capacity_train': self._expert_capacity_train,
            'expert_capacity_eval': self._expert_capacity_eval,
            'routing_type': self.gating_config.gating_type,
            'top_k': self.gating_config.top_k
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MixtureOfExperts':
        """Create layer from configuration."""
        moe_config = MoEConfig.from_dict(config.pop('config'))
        return cls(config=moe_config, **config)

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._built_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# Convenience functions for creating common MoE configurations

def create_ffn_moe(
        num_experts: int = 8,
        hidden_dim: int = 768,
        intermediate_size: Optional[int] = None,
        top_k: int = 2,
        gating_type: str = 'linear',
        **kwargs: Any
) -> MixtureOfExperts:
    """
    Create a Feed-Forward Network MoE layer.

    Args:
        num_experts: Number of FFN experts.
        hidden_dim: Hidden dimension for experts.
        intermediate_size: Intermediate dimension for FFN. If None, uses 4 * hidden_dim.
        top_k: Number of experts to activate per token.
        gating_type: Type of gating mechanism ('linear', 'cosine').
        **kwargs: Additional configuration parameters.

    Returns:
        Configured MoE layer with FFN experts.

    Example:
        ```python
        moe_layer = create_ffn_moe(
            num_experts=16,
            hidden_dim=1024,
            intermediate_size=4096,
            top_k=2
        )
        ```
    """
    if intermediate_size is None:
        intermediate_size = 4 * hidden_dim

    config = MoEConfig(
        num_experts=num_experts,
        expert_config=ExpertConfig(
            expert_type='ffn',
            hidden_dim=hidden_dim,
            intermediate_size=intermediate_size,
            **kwargs.pop('expert_config', {})
        ),
        gating_config=GatingConfig(
            gating_type=gating_type,
            top_k=top_k,
            **kwargs.pop('gating_config', {})
        ),
        **kwargs
    )

    return MixtureOfExperts(config=config)


def create_attention_moe(
        num_experts: int = 8,
        hidden_dim: int = 768,
        num_heads: int = 12,
        top_k: int = 1,
        gating_type: str = 'cosine',
        **kwargs: Any
) -> MixtureOfExperts:
    """
    Create an Attention MoE layer (Mixture-of-Attention).

    Args:
        num_experts: Number of attention experts.
        hidden_dim: Hidden dimension for experts.
        num_heads: Number of attention heads per expert.
        top_k: Number of experts to activate per token.
        gating_type: Type of gating mechanism ('linear', 'cosine').
        **kwargs: Additional configuration parameters.

    Returns:
        Configured MoE layer with attention experts.

    Example:
        ```python
        moa_layer = create_attention_moe(
            num_experts=12,
            hidden_dim=768,
            num_heads=12,
            top_k=1
        )
        ```
    """
    config = MoEConfig(
        num_experts=num_experts,
        expert_config=ExpertConfig(
            expert_type='attention',
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            **kwargs.pop('expert_config', {})
        ),
        gating_config=GatingConfig(
            gating_type=gating_type,
            top_k=top_k,
            **kwargs.pop('gating_config', {})
        ),
        **kwargs
    )

    return MixtureOfExperts(config=config)


def create_conv_moe(
        num_experts: int = 6,
        filters: int = 256,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        top_k: int = 1,
        gating_type: str = 'linear',
        **kwargs: Any
) -> MixtureOfExperts:
    """
    Create a Convolutional MoE layer for vision models.

    Args:
        num_experts: Number of convolutional experts.
        filters: Number of filters for each expert.
        kernel_size: Kernel size for convolutions.
        top_k: Number of experts to activate per token.
        gating_type: Type of gating mechanism ('linear', 'cosine').
        **kwargs: Additional configuration parameters.

    Returns:
        Configured MoE layer with convolutional experts.

    Example:
        ```python
        conv_moe_layer = create_conv_moe(
            num_experts=8,
            filters=512,
            kernel_size=3,
            top_k=1
        )
        ```
    """
    config = MoEConfig(
        num_experts=num_experts,
        expert_config=ExpertConfig(
            expert_type='conv2d',
            hidden_dim=filters,
            filters=filters,
            kernel_size=kernel_size,
            **kwargs.pop('expert_config', {})
        ),
        gating_config=GatingConfig(
            gating_type=gating_type,
            top_k=top_k,
            add_noise=False,  # Usually disabled for vision
            **kwargs.pop('gating_config', {})
        ),
        **kwargs
    )

    return MixtureOfExperts(config=config)