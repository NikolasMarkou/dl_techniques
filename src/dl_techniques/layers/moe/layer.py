"""
Main Mixture of Experts (MoE) layer implementation.

This module provides the core MoE layer that combines FFN expert networks and
gating mechanisms to create sparse, scalable neural network architectures.
Refined to use the simplified FFN-only expert system and follow modern Keras 3 patterns.
"""

import math
import keras
from keras import ops
from typing import Optional, Tuple, Any, Dict, List

from .experts import FFNExpert
from .config import ExpertConfig, GatingConfig, MoEConfig
from .gating import create_gating, compute_auxiliary_loss, compute_z_loss


from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class MixtureOfExperts(keras.layers.Layer):
    """
    Mixture of Experts (MoE) layer for sparse neural networks using FFN experts.

    This layer implements a complete MoE mechanism that routes inputs to a subset
    of FFN expert networks based on learned gating functions. It supports various
    FFN types through the dl_techniques FFN factory, routing strategies, and load
    balancing mechanisms.

    The MoE layer replaces dense FFN blocks with a sparse alternative where only
    a subset of FFN experts are activated for each input, providing computational
    efficiency and model specialization.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, hidden_dim])
           ↓
    Gating Network → expert_weights, expert_indices
           ↓
    Route to FFN Experts (top_k selection)
           ↓
    Expert₁(FFN) Expert₂(FFN) ... ExpertN(FFN)
           ↓
    Weighted Combination
           ↓
    Output(shape=[batch, seq_len, output_dim])
    ```

    **Key Features**:
    - FFN-only experts using dl_techniques FFN factory
    - Multiple gating strategies (linear, cosine, softmoe)
    - Load balancing with auxiliary losses
    - Token capacity management and dropout
    - Proper Keras 3 serialization support

    Args:
        config: MoE configuration containing expert and gating settings.
        **kwargs: Additional keyword arguments for the base Layer class.

    Input shape:
        - Sequence input: (batch_size, seq_len, hidden_dim)
        - Token input: (batch_size, hidden_dim)

    Output shape:
        Same as input shape, with last dimension determined by FFN expert output_dim.

    Attributes:
        experts: List of FFN expert networks.
        gating_network: Gating mechanism for routing decisions.

    Example:
        ```python
        # Basic FFN MoE with SwiGLU experts
        config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "swiglu",
                    "output_dim": 768,
                    "ffn_expansion_factor": 4
                }
            ),
            gating_config=GatingConfig(gating_type='linear', top_k=2)
        )
        moe_layer = MixtureOfExperts(config)

        # MLP expert MoE
        config = MoEConfig(
            num_experts=16,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "mlp",
                    "hidden_dim": 2048,
                    "output_dim": 768,
                    "activation": "gelu"
                }
            )
        )
        moe_layer = MixtureOfExperts(config)
        ```

    References:
        - Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
        - GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
        - PaLM: Scaling Language Modeling with Pathways
    """

    def __init__(self, config: MoEConfig, **kwargs: Any) -> None:
        """Initialize the MoE layer following modern Keras 3 patterns."""
        super().__init__(**kwargs)

        # Validate configuration
        if not isinstance(config, MoEConfig):
            raise ValueError("config must be an instance of MoEConfig")
        if config.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {config.num_experts}")

        # Store complete configuration
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
        self.routing_dtype = config.routing_dtype

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.experts: List[FFNExpert] = []
        for i in range(self.num_experts):
            # Create a unified config dictionary for the expert to pass to the FFN factory
            expert_ffn_params = self.expert_config.ffn_config.copy()

            # Merge other ExpertConfig params. Params in ffn_config take precedence.
            if 'use_bias' not in expert_ffn_params:
                expert_ffn_params['use_bias'] = self.expert_config.use_bias
            if 'kernel_initializer' not in expert_ffn_params:
                expert_ffn_params['kernel_initializer'] = self.expert_config.kernel_initializer
            if 'bias_initializer' not in expert_ffn_params:
                expert_ffn_params['bias_initializer'] = self.expert_config.bias_initializer

            # Only add regularizers if they are explicitly set to avoid passing None
            if self.expert_config.kernel_regularizer and 'kernel_regularizer' not in expert_ffn_params:
                expert_ffn_params['kernel_regularizer'] = self.expert_config.kernel_regularizer
            if self.expert_config.bias_regularizer and 'bias_regularizer' not in expert_ffn_params:
                expert_ffn_params['bias_regularizer'] = self.expert_config.bias_regularizer

            expert = FFNExpert(
                ffn_config=expert_ffn_params,
                name=f'expert_{i}'
            )
            self.experts.append(expert)

        # Create gating network using factory
        gating_kwargs = self.gating_config.__dict__.copy()
        gating_kwargs.pop('capacity_factor', None) # Not a gating network param

        self.gating_network = create_gating(
            num_experts=self.num_experts,
            **gating_kwargs
        )

        # Capacity and routing parameters (computed in build)
        self._expert_capacity_train = None
        self._expert_capacity_eval = None
        self._built_input_shape = None

        # Auxiliary loss tracking
        self._auxiliary_losses = []

        logger.info(f"Created MoE layer with {self.num_experts} FFN experts using {self.gating_config.gating_type} gating")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the MoE layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        self._built_input_shape = input_shape

        # Validate input shape
        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be known")

        # Determine the correct input shape for the experts
        if self.gating_config.gating_type == 'softmoe':
            # For SoftMoE, experts receive flattened slots: (batch, slots * hidden_dim)
            if len(input_shape) < 3:
                 raise ValueError("SoftMoE requires at least a 3D input for sequence processing.")
            batch_dim = input_shape[0] if len(input_shape) > 1 else None
            feature_dim = self.gating_config.num_slots * input_shape[-1]
            expert_input_shape = (batch_dim, feature_dim)
        else:
            # For hard routing, experts receive tokens with the original feature dimension
            expert_input_shape = input_shape

        # BUILD all FFN experts explicitly
        logger.debug(f"Building {self.num_experts} FFN experts")
        for i, expert in enumerate(self.experts):
            expert.build(expert_input_shape)
            logger.debug(f"Built expert_{i} with FFN type: {expert.ffn_config['type']}")

        # BUILD gating network explicitly
        logger.debug(f"Building {self.gating_config.gating_type} gating network")
        self.gating_network.build(input_shape)

        # Calculate expert capacities based on input shape
        self._calculate_expert_capacities(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(f"MoE layer built with input shape: {input_shape}")

    def _calculate_expert_capacities(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Calculate expert capacities based on input shape and configuration."""
        # Estimate tokens per batch for capacity calculation
        if len(input_shape) > 2:
            # Sequence-based input (batch_size, seq_len, hidden_dim)
            num_tokens = input_shape[1] if input_shape[1] else 512
        else:
            # Single token input (batch_size, hidden_dim)
            num_tokens = 1

        # Total tokens considering a typical batch size
        approx_tokens_per_batch = 32 * num_tokens # Conservative estimate

        # Base capacity per expert
        base_capacity = math.ceil(approx_tokens_per_batch / self.num_experts)

        # Apply capacity factors
        self._expert_capacity_train = max(1, int(base_capacity * self.train_capacity_factor))
        self._expert_capacity_eval = max(1, int(base_capacity * self.eval_capacity_factor))

        logger.debug(
            f"Expert capacities calculated - Train: {self._expert_capacity_train}, "
            f"Eval: {self._expert_capacity_eval}"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the MoE layer."""
        original_shape = ops.shape(inputs)

        # Get all dimensions except the last (feature) dimension
        # This handles inputs of any rank: 2D, 3D, 4D, etc.
        *batch_dims, hidden_dim = [original_shape[i] for i in range(len(original_shape))]

        # Calculate total number of tokens (all dims except features)
        if len(batch_dims) == 0:
            # Edge case: 1D input (just features)
            batch_dims = [1]
            total_tokens = 1
            inputs_reshaped = ops.expand_dims(inputs, axis=0)
        else:
            # Product of all batch dimensions
            total_tokens = ops.prod(ops.stack(batch_dims))
            # Flatten all batch dimensions into a single token dimension
            inputs_reshaped = ops.reshape(inputs, (total_tokens, hidden_dim))

        # For SoftMoE, we need the original multi-dimensional input
        # For hard routing, we use the flattened version
        if self.gating_config.gating_type == 'softmoe':
            # SoftMoE requires at least 3D input (batch, seq, features)
            if len(original_shape) < 3:
                raise ValueError(
                    f"SoftMoE gating requires at least a 3D input (batch, seq, features), "
                    f"but got shape {original_shape}"
                )
            gating_input = inputs
            inputs_flat = inputs_reshaped
        else:
            # Hard routing uses flattened input
            gating_input = inputs_reshaped
            inputs_flat = inputs_reshaped

        # Apply jitter noise if in training
        if training and self.jitter_noise > 0:
            def add_noise():
                noise = keras.random.uniform(
                    shape=ops.shape(gating_input),
                    minval=-self.jitter_noise,
                    maxval=self.jitter_noise,
                    dtype=gating_input.dtype
                )
                return gating_input + noise

            def no_noise():
                return gating_input

            gating_input = ops.cond(training, add_noise, no_noise)

        # Compute gating decisions
        expert_weights, expert_indices, gating_info = self.gating_network(
            gating_input, training=training
        )

        # Process based on gating type
        if self.gating_config.gating_type == 'softmoe':
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
                batch_dims=batch_dims,
                hidden_dim=hidden_dim,
                total_tokens=total_tokens,
                training=training
            )

        # Compute and add auxiliary losses during training
        self._auxiliary_losses = []
        if training:
            if self.gating_config.aux_loss_weight > 0 and 'raw_gate_probs' in gating_info:
                aux_loss = compute_auxiliary_loss(
                    expert_weights=gating_info['expert_weights'],
                    gate_probs=gating_info['raw_gate_probs'],
                    num_experts=self.num_experts,
                    aux_loss_weight=self.gating_config.aux_loss_weight
                )
                self._auxiliary_losses.append(aux_loss)
                self.add_loss(aux_loss)
                logger.debug(f"Added auxiliary loss: {aux_loss}")

            if self.gating_config.z_loss_weight > 0 and 'gate_logits' in gating_info:
                z_loss = compute_z_loss(
                    gate_logits=gating_info['gate_logits'],
                    z_loss_weight=self.gating_config.z_loss_weight
                )
                self._auxiliary_losses.append(z_loss)
                self.add_loss(z_loss)
                logger.debug(f"Added z-loss: {z_loss}")

        return outputs

    def _process_hard_routing(
            self,
            inputs_flat: keras.KerasTensor,
            expert_weights: keras.KerasTensor,
            expert_indices: keras.KerasTensor,
            batch_dims: list,
            hidden_dim: int,
            total_tokens: int,
            training: bool
    ) -> keras.KerasTensor:
        """Process inputs through FFN experts using hard routing."""

        # Determine output dimension from FFN expert configuration
        ffn_config = self.expert_config.ffn_config
        output_dim = ffn_config.get('output_dim', hidden_dim)

        # Get expert capacity for current training phase
        expert_capacity = self._expert_capacity_train if training else self._expert_capacity_eval

        # Flatten expert weights if needed
        weights_shape = ops.shape(expert_weights)
        if len(weights_shape) > 2:
            # Weights have multiple batch dimensions, flatten them
            weights_flat = ops.reshape(expert_weights, (total_tokens, self.num_experts))
        else:
            weights_flat = expert_weights

        # Process each expert
        expert_outputs_list = []
        expert_masks_list = []

        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            initial_mask = ops.any(
                ops.equal(expert_indices, expert_id),
                axis=-1
            )

            # Apply capacity constraints if enabled
            if self.drop_tokens and expert_capacity > 0:
                # Get number of tokens assigned
                num_assigned = ops.sum(ops.cast(initial_mask, 'int32'))

                # Get indices of assigned tokens
                token_indices = ops.reshape(ops.where(initial_mask), (-1,))

                def apply_capacity():
                    # Select first expert_capacity tokens
                    selected_indices = token_indices[:expert_capacity]
                    updates = ops.ones((expert_capacity,), dtype='bool')
                    return ops.scatter(
                        ops.expand_dims(selected_indices, 1),
                        updates,
                        (total_tokens,)
                    )

                def no_capacity():
                    return initial_mask

                final_tokens_mask = ops.cond(
                    num_assigned > expert_capacity,
                    apply_capacity,
                    no_capacity
                )
            else:
                final_tokens_mask = initial_mask

            # Process tokens through expert
            expert_token_indices = ops.reshape(ops.where(final_tokens_mask), (-1,))
            num_expert_tokens = ops.shape(expert_token_indices)[0]

            def process_expert():
                expert_tokens = ops.take(inputs_flat, expert_token_indices, axis=0)
                expert_output = self.experts[expert_id](expert_tokens, training=training)

                # Apply weights
                expert_token_weights = ops.take(
                    weights_flat[:, expert_id],
                    expert_token_indices,
                    axis=0
                )
                weighted_output = expert_output * ops.expand_dims(expert_token_weights, axis=-1)

                # Scatter to full size
                full_output = ops.zeros((total_tokens, output_dim), dtype=inputs_flat.dtype)
                full_output = ops.scatter(
                    ops.expand_dims(expert_token_indices, 1),
                    weighted_output,
                    (total_tokens, output_dim)
                )
                return full_output

            def empty_expert():
                return ops.zeros((total_tokens, output_dim), dtype=inputs_flat.dtype)

            expert_output = ops.cond(
                num_expert_tokens > 0,
                process_expert,
                empty_expert
            )

            expert_outputs_list.append(expert_output)
            expert_masks_list.append(final_tokens_mask)

        # Sum expert outputs
        outputs = ops.sum(ops.stack(expert_outputs_list, axis=0), axis=0)

        # Handle dropped tokens with residual connection
        if self.use_residual_connection and self.drop_tokens and hidden_dim == output_dim:
            # Find unprocessed tokens
            processed_mask = ops.zeros((total_tokens,), dtype='bool')
            for mask in expert_masks_list:
                processed_mask = ops.logical_or(processed_mask, mask)

            unprocessed_mask = ops.logical_not(processed_mask)

            # Add residual for unprocessed tokens
            residual = ops.where(
                ops.expand_dims(unprocessed_mask, -1),
                inputs_flat,
                ops.zeros_like(inputs_flat)
            )
            outputs = outputs + residual[..., :output_dim]

        # Reshape back to original batch structure
        output_shape = batch_dims + [output_dim]
        outputs = ops.reshape(outputs, output_shape)

        return outputs

    def _process_softmoe(
            self,
            expert_inputs: keras.KerasTensor,
            expert_weights: keras.KerasTensor,
            original_shape: Tuple[int, ...],
            training: bool
    ) -> keras.KerasTensor:
        """Process inputs through FFN experts using SoftMoE routing."""

        # For SoftMoE, we need to handle the multi-dimensional case more carefully
        # expert_inputs shape: [batch, num_experts, slots * hidden_dim]
        # expert_weights shape: depends on input dimensions

        # Get output dimension
        ffn_config = self.expert_config.ffn_config
        output_dim = ffn_config.get('output_dim', original_shape[-1])

        # Process each expert
        expert_outputs = []
        for expert_id in range(self.num_experts):
            expert_input = expert_inputs[:, expert_id, :]
            expert_output = self.experts[expert_id](expert_input, training=training)
            expert_outputs.append(expert_output)

        # Stack expert outputs: [batch, num_experts, output_dim]
        expert_outputs_stacked = ops.stack(expert_outputs, axis=1)

        # Combine based on weights
        # The weight shape needs to match the input dimensions
        if len(original_shape) == 3:
            # 3D input: (batch, seq, features)
            batch_size = original_shape[0]
            seq_len = original_shape[1]

            # Reshape weights to [batch, seq, num_experts, 1]
            weights_reshaped = ops.reshape(
                expert_weights,
                (batch_size, seq_len, self.num_experts, 1)
            )

            # Expand expert outputs to match
            expert_outputs_expanded = ops.expand_dims(
                expert_outputs_stacked, axis=1
            )  # [batch, 1, num_experts, output_dim]

            # Weighted combination
            weighted_outputs = expert_outputs_expanded * weights_reshaped
            outputs = ops.sum(weighted_outputs, axis=2)  # [batch, seq, output_dim]
        else:
            # Handle other dimensional cases
            # For now, fall back to simpler handling
            expert_weights_expanded = ops.expand_dims(expert_weights, axis=-1)
            weighted_outputs = expert_outputs_stacked * expert_weights_expanded
            outputs = ops.sum(weighted_outputs, axis=1)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the MoE layer."""
        output_shape = list(input_shape)

        # Determine output dimension from FFN configuration
        ffn_config = self.expert_config.ffn_config
        if 'output_dim' in ffn_config:
            output_shape[-1] = ffn_config['output_dim']
        # Otherwise keep input dimension

        return tuple(output_shape)

    def get_expert_utilization(self) -> Dict[str, Any]:
        """
        Get statistics about expert utilization and configuration.

        Returns:
            Dictionary containing expert utilization statistics.
        """
        return {
            'num_experts': self.num_experts,
            'expert_type': 'ffn',
            'expert_ffn_type': self.expert_config.ffn_config.get('type', 'unknown'),
            'expert_capacity_train': self._expert_capacity_train,
            'expert_capacity_eval': self._expert_capacity_eval,
            'routing_type': self.gating_config.gating_type,
            'top_k': self.gating_config.top_k,
            'aux_loss_weight': self.gating_config.aux_loss_weight,
            'z_loss_weight': self.gating_config.z_loss_weight,
            'drop_tokens': self.drop_tokens,
            'use_residual_connection': self.use_residual_connection
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


def create_ffn_moe(
    num_experts: int,
    ffn_config: Dict[str, Any],
    top_k: int = 1,
    gating_type: str = 'linear',
    aux_loss_weight: float = 0.01,
    **kwargs: Any
) -> MixtureOfExperts:
    """
    Convenience function to create FFN-based MoE layers.

    Args:
        num_experts: Number of FFN expert networks.
        ffn_config: FFN configuration dictionary (passed to FFN factory).
        top_k: Number of experts to select per token.
        gating_type: Type of gating mechanism ('linear', 'cosine', 'softmoe').
        aux_loss_weight: Weight for auxiliary load balancing loss.
        **kwargs: Additional configuration parameters.

    Returns:
        Configured MixtureOfExperts layer with FFN experts.

    Example:
        ```python
        # SwiGLU MoE
        moe = create_ffn_moe(
            num_experts=8,
            ffn_config={
                "type": "swiglu",
                "output_dim": 768,
                "ffn_expansion_factor": 4
            },
            top_k=2,
            gating_type='linear'
        )

        # MLP MoE
        moe = create_ffn_moe(
            num_experts=16,
            ffn_config={
                "type": "mlp",
                "hidden_dim": 2048,
                "output_dim": 768,
                "activation": "gelu"
            },
            top_k=1,
            aux_loss_weight=0.02
        )
        ```
    """

    # Create expert configuration using FFN factory
    expert_config = ExpertConfig(ffn_config=ffn_config)

    # Create gating configuration
    gating_config = GatingConfig(
        gating_type=gating_type,
        top_k=top_k,
        aux_loss_weight=aux_loss_weight,
        **{k: v for k, v in kwargs.items() if k in [
            'capacity_factor', 'add_noise', 'noise_std', 'temperature',
            'use_bias', 'embedding_dim', 'learnable_temperature', 'num_slots',
            'z_loss_weight'
        ]}
    )

    # Create complete MoE configuration
    moe_config = MoEConfig(
        num_experts=num_experts,
        expert_config=expert_config,
        gating_config=gating_config,
        **{k: v for k, v in kwargs.items() if k in [
            'jitter_noise', 'drop_tokens', 'use_residual_connection',
            'train_capacity_factor', 'eval_capacity_factor', 'routing_dtype'
        ]}
    )

    return MixtureOfExperts(config=moe_config)