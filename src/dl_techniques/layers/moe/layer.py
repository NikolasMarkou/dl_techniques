"""
Main Mixture of Experts (MoE) layer implementation.

This module provides the core MoE layer that combines FFN expert networks and
gating mechanisms to create sparse, scalable neural network architectures.
Refined to use the simplified FFN-only expert system and follow modern Keras 3 patterns.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Tuple, Any, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .experts import FFNExpert
from .config import ExpertConfig, GatingConfig, MoEConfig
from .gating import create_gating, compute_auxiliary_loss, compute_z_loss

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MixtureOfExperts(keras.layers.Layer):
    """
    Mixture of Experts layer for sparse neural networks using FFN experts.

    Implements a complete MoE mechanism that routes inputs to a subset of FFN
    expert networks based on learned gating functions. The MoE layer replaces
    dense FFN blocks with a sparse alternative where only top-k experts are
    activated per token, giving ``O(k/N)`` computational cost relative to running
    all *N* experts while maintaining full model capacity.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────┐
        │              MixtureOfExperts                    │
        │                                                  │
        │  Input(batch, seq, dim)                          │
        │         │                                        │
        │         ▼                                        │
        │  ┌─────────────────┐                             │
        │  │ Gating Network  │──► weights, indices, aux    │
        │  └─────────────────┘                             │
        │         │                                        │
        │         ▼                                        │
        │  ┌─────┬─────┬─────┬─────┐                       │
        │  │FFN_0│FFN_1│ ... │FFN_N│  (top-k activated)    │
        │  └──┬──┴──┬──┴─────┴──┬──┘                       │
        │     │     │           │                          │
        │     ▼     ▼           ▼                          │
        │  Weighted Combination (weights * outputs)        │
        │         │                                        │
        │         ▼                                        │
        │  Output(batch, seq, output_dim)                  │
        │  + aux_loss + z_loss (training)                  │
        └──────────────────────────────────────────────────┘

    :param config: MoE configuration containing expert and gating settings.
    :type config: MoEConfig
    :param kwargs: Additional keyword arguments for the base Layer class.
    :type kwargs: Any
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
        self.drop_tokens = config.drop_tokens
        self.use_residual_connection = config.use_residual_connection
        self.jitter_noise = config.jitter_noise
        self.routing_dtype = config.routing_dtype

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.experts: List[FFNExpert] = []
        for i in range(self.num_experts):
            expert = FFNExpert(
                ffn_config=self.expert_config.ffn_config.copy(),
                name=f'expert_{i}'
            )
            self.experts.append(expert)

        # Create gating network using factory — only pass gating-relevant keys
        gating_kwargs = {}
        gating_type = self.gating_config.gating_type

        if gating_type == 'linear':
            for k in ('top_k', 'use_bias', 'add_noise', 'noise_std'):
                gating_kwargs[k] = getattr(self.gating_config, k)
        elif gating_type == 'cosine':
            for k in ('top_k', 'embedding_dim', 'temperature', 'learnable_temperature'):
                gating_kwargs[k] = getattr(self.gating_config, k)
        elif gating_type == 'softmoe':
            for k in ('num_slots',):
                gating_kwargs[k] = getattr(self.gating_config, k)

        self.gating_network = create_gating(
            gating_type=gating_type,
            num_experts=self.num_experts,
            **gating_kwargs
        )

        self._built_input_shape = None

        logger.info(f"Created MoE layer with {self.num_experts} FFN experts using {gating_type} gating")

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
            # For SoftMoE, experts receive individual slot vectors: (batch, hidden_dim)
            # Each slot is processed independently, not concatenated
            if len(input_shape) < 3:
                raise ValueError("SoftMoE requires at least a 3D input for sequence processing.")
            expert_input_shape = (input_shape[0], input_shape[-1])
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

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(f"MoE layer built with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the MoE layer."""
        original_shape = ops.shape(inputs)
        input_ndim = len(inputs.shape)  # Static rank — works on all backends

        # Handle different input dimensions
        if input_ndim == 2:
            # 2D input: (batch, features)
            inputs_flat = inputs
            is_sequence = False

        elif input_ndim == 3:
            # 3D input: (batch, seq_len, features)
            batch_size = original_shape[0]
            seq_len = original_shape[1]
            hidden_dim = original_shape[2]
            inputs_flat = ops.reshape(inputs, (-1, hidden_dim))
            is_sequence = True

        else:
            # Higher dimensional input: flatten all but last dimension
            hidden_dim = original_shape[-1]
            inputs_flat = ops.reshape(inputs, (-1, hidden_dim))
            is_sequence = True

        # Determine gating input based on routing type
        if self.gating_config.gating_type == 'softmoe':
            if input_ndim < 3:
                raise ValueError(
                    f"SoftMoE gating requires at least a 3D input (batch, seq, features), "
                    f"but got shape with {input_ndim} dimensions"
                )
            gating_input = inputs
        else:
            gating_input = inputs_flat

        # Apply jitter noise if training
        if training and self.jitter_noise > 0:
            noise = keras.random.uniform(
                shape=ops.shape(gating_input),
                minval=-self.jitter_noise,
                maxval=self.jitter_noise,
                dtype=gating_input.dtype
            )
            gating_input = gating_input + noise

        # Compute gating decisions
        expert_weights, expert_indices, gating_info = self.gating_network(
            gating_input, training=training
        )

        # Process based on gating type
        if self.gating_config.gating_type == 'softmoe':
            outputs = self._process_softmoe(
                inputs=inputs,
                expert_inputs=gating_info['expert_inputs'],
                phi_weights=gating_info['phi_weights'],
                training=training
            )
        else:
            outputs = self._process_hard_routing(
                inputs_flat=inputs_flat,
                expert_weights=expert_weights,
                expert_indices=expert_indices,
                original_shape=original_shape,
                input_ndim=input_ndim,
                training=training
            )

        # Handle auxiliary losses during training
        # SoftMoE always uses all experts, so standard aux/z losses don't apply
        if training and self.gating_config.gating_type != 'softmoe':
            if self.gating_config.aux_loss_weight > 0 and 'raw_gate_probs' in gating_info:
                aux_loss = compute_auxiliary_loss(
                    expert_weights=gating_info.get('expert_weights', expert_weights),
                    gate_probs=gating_info['raw_gate_probs'],
                    num_experts=self.num_experts,
                    aux_loss_weight=self.gating_config.aux_loss_weight
                )
                self.add_loss(aux_loss)

            if self.gating_config.z_loss_weight > 0 and 'gate_logits' in gating_info:
                z_loss = compute_z_loss(
                    gate_logits=gating_info['gate_logits'],
                    z_loss_weight=self.gating_config.z_loss_weight
                )
                self.add_loss(z_loss)

        return outputs

    def _process_hard_routing(
            self,
            inputs_flat: keras.KerasTensor,
            expert_weights: keras.KerasTensor,
            expert_indices: keras.KerasTensor,
            original_shape: Tuple[int, ...],
            input_ndim: int,
            training: bool
    ) -> keras.KerasTensor:
        """Process inputs through FFN experts using hard routing.

        All tokens pass through all experts, then outputs are masked by the
        routing decision.  This is computationally O(N) not O(k) but avoids
        scatter/gather ops that are problematic in graph mode.
        """
        num_tokens = ops.shape(inputs_flat)[0]

        # Determine output dimension from expert config
        ffn_config = self.expert_config.ffn_config
        output_dim = ffn_config.get('output_dim', None) or ffn_config.get('d_model', None)

        # Flatten weights to (num_tokens, num_experts)
        weights_ndim = len(expert_weights.shape)
        if weights_ndim > 2:
            weights_flat = ops.reshape(expert_weights, (-1, self.num_experts))
        else:
            weights_flat = expert_weights

        # Create expert assignment mask from indices
        indices_ndim = len(expert_indices.shape)
        if indices_ndim > 1:
            # Flatten indices to 2D: (num_tokens, top_k) or (num_tokens, num_experts)
            indices_flat = ops.reshape(expert_indices, (-1, expert_indices.shape[-1]))
        else:
            indices_flat = expert_indices

        if indices_flat.shape[-1] is not None and indices_flat.shape[-1] < self.num_experts:
            # top_k selection: create assignment from one-hot sum
            expert_one_hot = ops.one_hot(indices_flat, self.num_experts)
            expert_assignment = ops.sum(expert_one_hot, axis=-2)  # (num_tokens, num_experts)
        else:
            # All experts selected
            expert_assignment = ops.ones_like(weights_flat)

        # Process all experts and combine
        expert_outputs = []
        for expert_id in range(self.num_experts):
            expert_output = self.experts[expert_id](inputs_flat, training=training)

            # Weight and mask this expert's output
            expert_weight = weights_flat[:, expert_id:expert_id + 1]
            expert_mask = expert_assignment[:, expert_id:expert_id + 1]
            weighted_output = expert_output * expert_weight * expert_mask
            expert_outputs.append(weighted_output)

        # Sum outputs from all experts
        outputs = ops.sum(ops.stack(expert_outputs, axis=0), axis=0)

        # Apply residual connection for unrouted tokens if configured
        if self.use_residual_connection and self.drop_tokens:
            # Tokens with zero total weight get the residual input
            total_weight = ops.sum(weights_flat * expert_assignment, axis=-1, keepdims=True)
            unrouted_mask = ops.cast(total_weight < 1e-6, outputs.dtype)
            # Only apply residual if output_dim matches input_dim
            if output_dim is None or (inputs_flat.shape[-1] is not None and output_dim == inputs_flat.shape[-1]):
                outputs = outputs + unrouted_mask * inputs_flat

        # Reshape back to original structure
        if input_ndim == 2:
            return outputs
        else:
            # Infer output_dim from actual output
            actual_output_dim = ops.shape(outputs)[-1]
            new_shape = list(original_shape[:-1]) + [actual_output_dim]
            return ops.reshape(outputs, new_shape)

    def _process_softmoe(
            self,
            inputs: keras.KerasTensor,
            expert_inputs: keras.KerasTensor,
            phi_weights: keras.KerasTensor,
            training: bool
    ) -> keras.KerasTensor:
        """Process inputs through FFN experts using SoftMoE routing.

        SoftMoE creates soft slots via weighted combinations of input tokens,
        processes each slot through an expert independently, then combines
        expert outputs back to the original sequence positions using dispatch
        weights.

        expert_inputs shape: [batch, num_experts, num_slots * hidden_dim]
        phi_weights shape:   [batch, seq_len, num_experts, num_slots]
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        hidden_dim = ops.shape(inputs)[-1]
        num_slots = self.gating_config.num_slots

        # Process each expert on its slots independently
        # Each expert receives individual slot vectors, not the concatenated version
        expert_slot_outputs = []  # Will collect [batch, num_slots, output_dim] per expert

        for expert_id in range(self.num_experts):
            # Get this expert's concatenated slot input: [batch, num_slots * hidden_dim]
            expert_input = expert_inputs[:, expert_id, :]
            # Reshape to individual slots: [batch * num_slots, hidden_dim]
            expert_input = ops.reshape(expert_input, (batch_size * num_slots, hidden_dim))
            # Process through expert
            expert_output = self.experts[expert_id](expert_input, training=training)
            # Reshape back: [batch, num_slots, output_dim]
            output_dim = ops.shape(expert_output)[-1]
            expert_output = ops.reshape(expert_output, (batch_size, num_slots, output_dim))
            expert_slot_outputs.append(expert_output)

        # Stack: [batch, num_experts, num_slots, output_dim]
        all_expert_outputs = ops.stack(expert_slot_outputs, axis=1)

        # Combine expert slot outputs back to sequence positions using dispatch weights
        # phi_weights: [batch, seq_len, num_experts, num_slots]
        # all_expert_outputs: [batch, num_experts, num_slots, output_dim]

        # Expand phi_weights for matmul: [batch, seq_len, num_experts, num_slots, 1]
        phi_expanded = ops.expand_dims(phi_weights, axis=-1)

        # Expand expert outputs: [batch, 1, num_experts, num_slots, output_dim]
        outputs_expanded = ops.expand_dims(all_expert_outputs, axis=1)

        # Weighted combination: [batch, seq_len, num_experts, num_slots, output_dim]
        weighted = phi_expanded * outputs_expanded

        # Sum over experts and slots: [batch, seq_len, output_dim]
        outputs = ops.sum(weighted, axis=(2, 3))

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the MoE layer."""
        output_shape = list(input_shape)

        # Determine output dimension from FFN configuration
        ffn_config = self.expert_config.ffn_config
        if 'output_dim' in ffn_config:
            output_shape[-1] = ffn_config['output_dim']
        elif 'd_model' in ffn_config:
            output_shape[-1] = ffn_config['d_model']
        # Otherwise keep input dimension

        return tuple(output_shape)

    def get_expert_utilization(self) -> Dict[str, Any]:
        """
        Get statistics about expert utilization and configuration.

        :return: Dictionary containing expert utilization statistics.
        :rtype: Dict[str, Any]
        """
        return {
            'num_experts': self.num_experts,
            'expert_type': 'ffn',
            'expert_ffn_type': self.expert_config.ffn_config.get('type', 'unknown'),
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
        config = dict(config)  # Don't mutate caller's dict
        moe_config = MoEConfig.from_dict(config.pop('config'))
        return cls(config=moe_config, **config)

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for weight restoration."""
        return {"input_shape": self._built_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from saved configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

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

    :param num_experts: Number of FFN expert networks.
    :type num_experts: int
    :param ffn_config: FFN configuration dictionary (passed to FFN factory).
    :type ffn_config: Dict[str, Any]
    :param top_k: Number of experts to select per token.
    :type top_k: int
    :param gating_type: Type of gating mechanism (``'linear'``, ``'cosine'``, ``'softmoe'``).
    :type gating_type: str
    :param aux_loss_weight: Weight for auxiliary load balancing loss.
    :type aux_loss_weight: float
    :param kwargs: Additional configuration parameters.
    :type kwargs: Any
    :return: Configured MixtureOfExperts layer with FFN experts.
    :rtype: MixtureOfExperts
    """

    # Create expert configuration using FFN factory
    expert_config = ExpertConfig(ffn_config=ffn_config)

    # Create gating configuration
    gating_keys = {
        'capacity_factor', 'add_noise', 'noise_std', 'temperature',
        'use_bias', 'embedding_dim', 'learnable_temperature', 'num_slots',
        'z_loss_weight'
    }
    gating_config = GatingConfig(
        gating_type=gating_type,
        top_k=top_k,
        aux_loss_weight=aux_loss_weight,
        **{k: v for k, v in kwargs.items() if k in gating_keys}
    )

    # Create complete MoE configuration
    moe_keys = {
        'jitter_noise', 'drop_tokens', 'use_residual_connection',
        'train_capacity_factor', 'eval_capacity_factor', 'routing_dtype'
    }
    moe_config = MoEConfig(
        num_experts=num_experts,
        expert_config=expert_config,
        gating_config=gating_config,
        **{k: v for k, v in kwargs.items() if k in moe_keys}
    )

    return MixtureOfExperts(config=moe_config)

# ---------------------------------------------------------------------
