"""
Gating network implementations for Mixture of Experts (MoE) models.

This module provides various gating mechanisms (routers) that determine how
inputs are distributed to expert networks, including linear gating, cosine
similarity gating, and SoftMoE approaches.
"""

import keras
from abc import ABC, abstractmethod
from keras import ops, layers, initializers
from typing import Optional, Union, Tuple, Any, Dict


@keras.saving.register_keras_serializable()
class BaseGating(layers.Layer, ABC):
    """
    Abstract base class for MoE gating networks.

    This class defines the interface for all gating implementations,
    ensuring consistent behavior across different routing strategies.
    Follows modern Keras 3 patterns with proper serialization support.

    Args:
        num_experts: Number of expert networks to route to.
        name: Name for the gating layer.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
            self,
            num_experts: int,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the base gating layer."""
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        self.num_experts = num_experts

    @abstractmethod
    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Compute gating scores and routing information.

        Args:
            inputs: Input tensor to route.
            training: Whether the layer is in training mode.

        Returns:
            Tuple containing:
            - expert_weights: Weights for combining expert outputs
            - expert_indices: Indices of selected experts
            - auxiliary_info: Additional information for loss computation
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
        })
        return config


@keras.saving.register_keras_serializable()
class LinearGating(BaseGating):
    """
    Linear gating network with optional noise and top-k selection.

    This is the most common gating mechanism, using a linear transformation
    followed by softmax and top-k selection. It supports noise injection
    for improved load balancing during training.

    Args:
        num_experts: Number of expert networks.
        top_k: Number of experts to select per token. Must be <= num_experts.
        use_bias: Whether to use bias in the linear transformation.
        add_noise: Whether to add noise to gating logits during training.
        noise_std: Standard deviation of the noise.
        kernel_initializer: Weight initialization strategy for gate weights.
        bias_initializer: Bias initialization strategy.
        **kwargs: Additional keyword arguments.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.

    Output shape:
        Tuple of:
        - expert_weights: `(batch_size, ..., num_experts)`
        - expert_indices: `(batch_size, ..., top_k)`
        - auxiliary_info: Dict with routing statistics

    Example:
        ```python
        gating = LinearGating(
            num_experts=8,
            top_k=2,
            add_noise=True,
            noise_std=1.0
        )

        inputs = keras.Input(shape=(512, 768))
        weights, indices, info = gating(inputs)
        ```
    """

    def __init__(
            self,
            num_experts: int,
            top_k: int = 1,
            use_bias: bool = False,
            add_noise: bool = True,
            noise_std: float = 1.0,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            **kwargs: Any
    ) -> None:
        """Initialize the linear gating network."""
        super().__init__(num_experts=num_experts, **kwargs)

        # Validate inputs
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k must be between 1 and {num_experts}, got {top_k}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        self.top_k = top_k
        self.use_bias = use_bias
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # CREATE sublayers in __init__ (unbuilt)
        self.gate_dense = layers.Dense(
            units=num_experts,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='gate_dense'
        )

        if add_noise:
            self.noise_dense = layers.Dense(
                units=num_experts,
                use_bias=False,
                kernel_initializer='zeros',
                name='noise_dense'
            )
        else:
            self.noise_dense = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the linear gating layers."""
        # BUILD all sublayers explicitly
        self.gate_dense.build(input_shape)

        if self.noise_dense is not None:
            self.noise_dense.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the linear gating network."""
        original_shape = ops.shape(inputs)

        # Reshape to 2D for processing if needed
        if len(original_shape) > 2:
            inputs_flat = ops.reshape(inputs, (-1, original_shape[-1]))
        else:
            inputs_flat = inputs

        # Compute gating logits
        gate_logits = self.gate_dense(inputs_flat)

        # Add noise during training
        if self.add_noise and training and self.noise_dense is not None:
            noise_logits = self.noise_dense(inputs_flat)
            noise = keras.random.normal(
                shape=ops.shape(noise_logits),
                mean=0.0,
                stddev=1.0,
                dtype=inputs.dtype
            )
            # Apply softplus to ensure positive noise std
            noise_std = ops.softplus(noise_logits) * self.noise_std
            gate_logits = gate_logits + noise * noise_std

        # Top-k selection
        if self.top_k < self.num_experts:
            top_k_logits, top_k_indices = ops.top_k(gate_logits, k=self.top_k)

            # Create mask for selected experts using one_hot
            top_k_one_hot = ops.one_hot(top_k_indices, self.num_experts, dtype=gate_logits.dtype)
            mask = ops.sum(top_k_one_hot, axis=-2)

            # Apply mask to logits (set non-selected to -inf)
            masked_logits = ops.where(
                mask > 0,
                gate_logits,
                ops.full_like(gate_logits, -1e9)
            )
            expert_weights = ops.softmax(masked_logits, axis=-1)
            expert_indices = top_k_indices
        else:
            # Use all experts
            expert_weights = ops.softmax(gate_logits, axis=-1)
            expert_indices = ops.arange(self.num_experts, dtype='int32')
            expert_indices = ops.broadcast_to(
                expert_indices[None, :],
                (ops.shape(gate_logits)[0], self.num_experts)
            )

        # Reshape back to original batch structure if needed
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.num_experts]
            expert_weights = ops.reshape(expert_weights, new_shape)
            if self.top_k < self.num_experts:
                new_indices_shape = list(original_shape[:-1]) + [self.top_k]
                expert_indices = ops.reshape(expert_indices, new_indices_shape)
            else:
                expert_indices = ops.reshape(expert_indices, new_shape)

        # Prepare auxiliary information for load balancing loss
        raw_gate_probs = ops.softmax(gate_logits, axis=-1)
        auxiliary_info = {
            'gate_logits': gate_logits,
            'expert_weights': expert_weights,
            'raw_gate_probs': raw_gate_probs
        }

        return expert_weights, expert_indices, auxiliary_info

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'top_k': self.top_k,
            'use_bias': self.use_bias,
            'add_noise': self.add_noise,
            'noise_std': self.noise_std,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer)
        })
        return config


@keras.saving.register_keras_serializable()
class CosineGating(BaseGating):
    """
    Cosine similarity-based gating network.

    This gating mechanism operates in a hypersphere space, using cosine similarity
    between the input representation and learnable expert embeddings. This can
    provide better domain generalization compared to linear gating.

    Args:
        num_experts: Number of expert networks.
        embedding_dim: Dimension of expert embeddings.
        top_k: Number of experts to select per token.
        temperature: Temperature parameter for softmax scaling.
        learnable_temperature: Whether temperature is a learnable parameter.
        kernel_initializer: Weight initialization strategy.
        **kwargs: Additional keyword arguments.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.

    Output shape:
        Same as LinearGating output format.

    Example:
        ```python
        gating = CosineGating(
            num_experts=8,
            embedding_dim=256,
            top_k=1,
            temperature=0.1,
            learnable_temperature=True
        )
        ```
    """

    def __init__(
            self,
            num_experts: int,
            embedding_dim: int = 256,
            top_k: int = 1,
            temperature: float = 1.0,
            learnable_temperature: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        """Initialize the cosine gating network."""
        super().__init__(num_experts=num_experts, **kwargs)

        # Validate inputs
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k must be between 1 and {num_experts}, got {top_k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.kernel_initializer = initializers.get(kernel_initializer)

        # CREATE sublayers in __init__
        self.linear_projection = layers.Dense(
            units=embedding_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name='linear_projection'
        )

        # Weight attributes created in build()
        self.expert_embeddings = None
        self.temperature_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the cosine gating layers."""
        # CREATE weights in build()
        self.expert_embeddings = self.add_weight(
            name='expert_embeddings',
            shape=(self.embedding_dim, self.num_experts),
            initializer=self.kernel_initializer,
            trainable=True
        )

        if self.learnable_temperature:
            self.temperature_param = self.add_weight(
                name='temperature',
                shape=(),
                initializer=initializers.Constant(value=self.temperature),
                trainable=True
            )

        # BUILD sublayers explicitly
        self.linear_projection.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the cosine gating network."""
        original_shape = ops.shape(inputs)

        # Reshape to 2D for processing if needed
        if len(original_shape) > 2:
            inputs_flat = ops.reshape(inputs, (-1, original_shape[-1]))
        else:
            inputs_flat = inputs

        # Project input to embedding space
        projected_inputs = self.linear_projection(inputs_flat)

        # Normalize projected inputs and expert embeddings
        projected_inputs_norm = ops.normalize(projected_inputs, axis=-1)
        expert_embeddings_norm = ops.normalize(self.expert_embeddings, axis=0)

        # Compute cosine similarities
        cosine_similarities = ops.matmul(projected_inputs_norm, expert_embeddings_norm)

        # Apply temperature
        temperature_value = self.temperature_param if self.learnable_temperature else self.temperature
        gate_logits = cosine_similarities * temperature_value

        # Top-k selection
        if self.top_k < self.num_experts:
            top_k_logits, top_k_indices = ops.top_k(gate_logits, k=self.top_k)

            # Create mask for selected experts using one_hot
            top_k_one_hot = ops.one_hot(top_k_indices, self.num_experts, dtype=gate_logits.dtype)
            mask = ops.sum(top_k_one_hot, axis=-2)

            # Apply mask to logits
            masked_logits = ops.where(
                mask > 0,
                gate_logits,
                ops.full_like(gate_logits, -1e9)
            )
            expert_weights = ops.softmax(masked_logits, axis=-1)
            expert_indices = top_k_indices
        else:
            # Use all experts
            expert_weights = ops.softmax(gate_logits, axis=-1)
            expert_indices = ops.arange(self.num_experts, dtype='int32')
            expert_indices = ops.broadcast_to(
                expert_indices[None, :],
                (ops.shape(gate_logits)[0], self.num_experts)
            )

        # Reshape back to original batch structure if needed
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.num_experts]
            expert_weights = ops.reshape(expert_weights, new_shape)
            if self.top_k < self.num_experts:
                new_indices_shape = list(original_shape[:-1]) + [self.top_k]
                expert_indices = ops.reshape(expert_indices, new_indices_shape)
            else:
                expert_indices = ops.reshape(expert_indices, new_shape)

        # Prepare auxiliary information
        raw_gate_probs = ops.softmax(gate_logits, axis=-1)
        auxiliary_info = {
            'gate_logits': gate_logits,
            'expert_weights': expert_weights,
            'cosine_similarities': cosine_similarities,
            'raw_gate_probs': raw_gate_probs
        }

        return expert_weights, expert_indices, auxiliary_info

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'learnable_temperature': self.learnable_temperature,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


@keras.saving.register_keras_serializable()
class SoftMoEGating(BaseGating):
    """
    SoftMoE gating that creates soft input slots for experts.

    Unlike traditional hard routing, SoftMoE computes weighted combinations
    of all input tokens to create "soft slots" for each expert. This avoids
    token dropping and load balancing issues at the cost of increased computation.

    Args:
        num_experts: Number of expert networks.
        num_slots: Number of input slots per expert.
        kernel_initializer: Weight initialization strategy.
        **kwargs: Additional keyword arguments.

    Input shape:
        3D tensor with shape: `(batch_size, seq_len, hidden_dim)`.

    Output shape:
        Same format as other gating mechanisms, but includes soft slot information.

    Example:
        ```python
        gating = SoftMoEGating(
            num_experts=8,
            num_slots=4
        )
        ```
    """

    def __init__(
            self,
            num_experts: int,
            num_slots: int = 4,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        """Initialize the SoftMoE gating network."""
        super().__init__(num_experts=num_experts, **kwargs)

        # Validate inputs
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")

        self.num_slots = num_slots
        self.kernel_initializer = initializers.get(kernel_initializer)

        # CREATE sublayers in __init__
        self.phi_dense = layers.Dense(
            units=num_experts * num_slots,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            name='phi_dense'
        )

        # Weight attributes created in build()
        self.slot_embeddings = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the SoftMoE gating layers."""
        hidden_dim = input_shape[-1]
        if hidden_dim is None:
            raise ValueError("Hidden dimension must be known for SoftMoE")

        # CREATE weights
        self.slot_embeddings = self.add_weight(
            name='slot_embeddings',
            shape=(self.num_experts, self.num_slots, hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True
        )

        # BUILD sublayers explicitly
        self.phi_dense.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the SoftMoE gating network."""
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        hidden_dim = ops.shape(inputs)[-1]

        # Compute attention weights for slot assignment
        phi_logits = self.phi_dense(inputs)  # [batch, seq_len, num_experts * num_slots]
        phi_logits = ops.reshape(phi_logits, (batch_size, seq_len, self.num_experts, self.num_slots))

        # Softmax over sequence dimension for each expert-slot combination
        phi_weights = ops.softmax(phi_logits, axis=1)  # [batch, seq_len, num_experts, num_slots]

        # Compute soft input slots for each expert
        inputs_expanded = ops.expand_dims(ops.expand_dims(inputs, axis=2), axis=3)  # [b, s, 1, 1, h]
        phi_weights_expanded = ops.expand_dims(phi_weights, axis=-1)  # [b, s, e, l, 1]

        # Weighted sum to create slots
        soft_slots = ops.sum(
            inputs_expanded * phi_weights_expanded,  # Broadcasts to [b, s, e, l, h]
            axis=1
        )  # Sum over s -> [b, e, l, h]

        # Flatten slots for expert processing
        expert_inputs = ops.reshape(
            soft_slots,
            (batch_size, self.num_experts, self.num_slots * hidden_dim)
        )

        # For SoftMoE, all experts are used with equal weight
        expert_weights = ops.ones((batch_size, seq_len, self.num_experts), dtype=inputs.dtype) / self.num_experts
        expert_indices = ops.arange(self.num_experts, dtype='int32')
        expert_indices = ops.broadcast_to(
            expert_indices[None, None, :],
            (batch_size, seq_len, self.num_experts)
        )

        # Prepare auxiliary information
        auxiliary_info = {
            'phi_weights': phi_weights,
            'soft_slots': soft_slots,
            'expert_inputs': expert_inputs,
            'expert_weights': expert_weights,
            'gate_logits': phi_logits,  # For z-loss computation
            'raw_gate_probs': ops.softmax(phi_logits, axis=-1)
        }

        return expert_weights, expert_indices, auxiliary_info

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_slots': self.num_slots,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


def compute_auxiliary_loss(
        expert_weights: keras.KerasTensor,
        gate_probs: keras.KerasTensor,
        num_experts: int,
        aux_loss_weight: float = 0.01
) -> keras.KerasTensor:
    """
    Compute auxiliary load balancing loss for MoE training.

    This loss encourages uniform distribution of tokens across experts
    to prevent expert collapse and improve hardware utilization.

    Args:
        expert_weights: Expert selection weights [batch, ..., num_experts].
        gate_probs: Raw gating probabilities [batch, ..., num_experts].
        num_experts: Total number of experts.
        aux_loss_weight: Weight for the auxiliary loss.

    Returns:
        Auxiliary load balancing loss scalar.
    """
    # Determine axes for token-wise mean calculation (all but the last axis)
    num_token_axes = len(ops.shape(expert_weights)) - 1
    token_axes = list(range(num_token_axes))

    # Compute fraction of tokens dispatched to each expert
    expert_mask = ops.cast(expert_weights > 0, expert_weights.dtype)
    tokens_per_expert = ops.mean(expert_mask, axis=token_axes)  # [num_experts]

    # Compute average gate probability for each expert
    avg_gate_probs = ops.mean(gate_probs, axis=token_axes)  # [num_experts]

    # Auxiliary loss = N * sum(f_i * P_i) where f_i is fraction, P_i is avg prob
    aux_loss = num_experts * ops.sum(tokens_per_expert * avg_gate_probs)

    return aux_loss_weight * aux_loss


def compute_z_loss(
        gate_logits: keras.KerasTensor,
        z_loss_weight: float = 1e-3
) -> keras.KerasTensor:
    """
    Compute router z-loss for entropy regularization.

    The z-loss encourages the router to produce confident decisions
    by penalizing the squared logsumexp of the gate logits.

    Args:
        gate_logits: Raw gate logits [batch, seq_len, num_experts].
        z_loss_weight: Weight for the z-loss.

    Returns:
        Router z-loss scalar.
    """
    # Compute logsumexp for each token
    logsumexp = ops.logsumexp(gate_logits, axis=-1, keepdims=False)  # [batch, seq_len]

    # Z-loss is the squared mean of logsumexp
    z_loss = ops.mean(ops.square(logsumexp))

    return z_loss_weight * z_loss


def create_gating(gating_type: str, num_experts: int, **kwargs) -> BaseGating:
    """
    Factory function to create gating networks.

    Args:
        gating_type: Type of gating to create ('linear', 'cosine', 'softmoe').
        num_experts: Number of expert networks.
        **kwargs: Configuration parameters for the gating network.

    Returns:
        Configured gating network.

    Raises:
        ValueError: If gating_type is not supported.

    Example:
        ```python
        # Create linear gating
        gating = create_gating('linear', num_experts=8, top_k=2)

        # Create cosine gating
        gating = create_gating('cosine', num_experts=8, embedding_dim=256)

        # Create SoftMoE gating
        gating = create_gating('softmoe', num_experts=8, num_slots=4)
        ```
    """
    if gating_type == 'linear':
        linear_keys = ['top_k', 'use_bias', 'add_noise', 'noise_std',
                       'kernel_initializer', 'bias_initializer']
        linear_kwargs = {k: v for k, v in kwargs.items() if k in linear_keys}
        return LinearGating(num_experts=num_experts, **linear_kwargs)
    elif gating_type == 'cosine':
        cosine_keys = ['embedding_dim', 'top_k', 'temperature',
                       'learnable_temperature', 'kernel_initializer']
        cosine_kwargs = {k: v for k, v in kwargs.items() if k in cosine_keys}
        return CosineGating(num_experts=num_experts, **cosine_kwargs)
    elif gating_type == 'softmoe':
        softmoe_keys = ['num_slots', 'kernel_initializer']
        softmoe_kwargs = {k: v for k, v in kwargs.items() if k in softmoe_keys}
        return SoftMoEGating(num_experts=num_experts, **softmoe_kwargs)
    else:
        raise ValueError(
            f"Unsupported gating type: {gating_type}. "
            f"Supported types: ['linear', 'cosine', 'softmoe']"
        )