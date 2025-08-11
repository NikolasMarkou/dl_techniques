"""
Gating network implementations for Mixture of Experts (MoE) models.

This module provides various gating mechanisms (routers) that determine how
inputs are distributed to expert networks, including linear gating, cosine
similarity gating, and SoftMoE approaches.
"""


import keras
from keras import ops
from abc import ABC, abstractmethod
from keras import layers, initializers
from typing import Optional, Union, Tuple, Any, Dict


class BaseGating(layers.Layer, ABC):
    """
    Abstract base class for MoE gating networks.

    This class defines the interface for all gating implementations,
    ensuring consistent behavior across different routing strategies.

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
        self.num_experts = num_experts
        self._built_input_shape = None

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

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._built_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the gating network from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


class LinearGating(BaseGating):
    """
    Linear gating network with optional noise and top-k selection.

    This is the most common gating mechanism, using a linear transformation
    followed by softmax and top-k selection. It supports noise injection
    for improved load balancing during training.

    Args:
        num_experts: Number of expert networks.
        top_k: Number of experts to select per token.
        use_bias: Whether to use bias in the linear transformation.
        add_noise: Whether to add noise to gating logits.
        noise_std: Standard deviation of the noise.
        kernel_initializer: Weight initialization strategy.
        bias_initializer: Bias initialization strategy.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        gating = LinearGating(
            num_experts=8,
            top_k=2,
            add_noise=True,
            noise_std=1.0
        )
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

        self.top_k = top_k
        self.use_bias = use_bias
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # Sublayers initialized in build()
        self.gate_dense = None
        self.noise_dense = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the linear gating layers."""
        self._built_input_shape = input_shape

        # Main gating transformation
        self.gate_dense = layers.Dense(
            units=self.num_experts,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='gate_dense'
        )

        # Noise transformation (if enabled)
        if self.add_noise:
            self.noise_dense = layers.Dense(
                units=self.num_experts,
                use_bias=False,
                kernel_initializer='zeros',
                name='noise_dense'
            )

        # Build sublayers
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
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1] if len(ops.shape(inputs)) > 2 else 1

        # Reshape to 2D for processing
        original_shape = ops.shape(inputs)
        if len(original_shape) > 2:
            inputs_flat = ops.reshape(inputs, (-1, original_shape[-1]))
        else:
            inputs_flat = inputs

        # Compute gating logits
        gate_logits = self.gate_dense(inputs_flat)

        # Add noise during training
        if self.add_noise and training:
            noise_logits = self.noise_dense(inputs_flat)
            noise = keras.random.normal(
                shape=ops.shape(noise_logits),
                mean=0.0,
                stddev=1.0,
                dtype=inputs.dtype
            )
            # Apply softplus to ensure positive noise std
            noise_std = ops.softplus(noise_logits) * self.noise_std / (self.num_experts ** 2)
            gate_logits = gate_logits + noise * noise_std

        # Top-k selection
        if self.top_k < self.num_experts:
            top_k_logits, top_k_indices = ops.top_k(gate_logits, k=self.top_k)

            # Create mask for selected experts
            mask = ops.zeros_like(gate_logits)
            for i in range(self.top_k):
                mask = ops.scatter_update(
                    mask,
                    ops.expand_dims(top_k_indices[:, i], axis=-1),
                    1.0
                )

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

        # Reshape back to original batch structure
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.num_experts]
            expert_weights = ops.reshape(expert_weights, new_shape)
            expert_indices = ops.reshape(expert_indices, new_shape)

        # Prepare auxiliary information for load balancing loss
        auxiliary_info = {
            'gate_logits': gate_logits,
            'expert_weights': expert_weights,
            'raw_gate_probs': ops.softmax(gate_logits, axis=-1)
        }

        return expert_weights, expert_indices, auxiliary_info

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'use_bias': self.use_bias,
            'add_noise': self.add_noise,
            'noise_std': self.noise_std,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer)
        })
        return config


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
        temperature: Temperature parameter for softmax.
        learnable_temperature: Whether temperature is learnable.
        kernel_initializer: Weight initialization strategy.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        gating = CosineGating(
            num_experts=8,
            embedding_dim=256,
            top_k=1,
            temperature=0.1
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

        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.kernel_initializer = initializers.get(kernel_initializer)

        # Sublayers initialized in build()
        self.linear_projection = None
        self.expert_embeddings = None
        self.temperature_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the cosine gating layers."""
        self._built_input_shape = input_shape

        # Linear projection to embedding space
        self.linear_projection = layers.Dense(
            units=self.embedding_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name='linear_projection'
        )

        # Expert embeddings
        self.expert_embeddings = self.add_weight(
            name='expert_embeddings',
            shape=(self.embedding_dim, self.num_experts),
            initializer=self.kernel_initializer,
            trainable=True
        )

        # Temperature parameter
        if self.learnable_temperature:
            self.temperature_param = self.add_weight(
                name='temperature',
                shape=(),
                initializer=initializers.Constant(value=self.temperature),
                trainable=True
            )
        else:
            self.temperature_param = self.temperature

        # Build sublayers
        self.linear_projection.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the cosine gating network."""
        # Reshape to 2D for processing
        original_shape = ops.shape(inputs)
        if len(original_shape) > 2:
            inputs_flat = ops.reshape(inputs, (-1, original_shape[-1]))
        else:
            inputs_flat = inputs

        # Project input to embedding space
        projected_inputs = self.linear_projection(inputs_flat)

        # Normalize projected inputs
        projected_inputs_norm = ops.l2_normalize(projected_inputs, axis=-1)

        # Normalize expert embeddings
        expert_embeddings_norm = ops.l2_normalize(self.expert_embeddings, axis=0)

        # Compute cosine similarities
        cosine_similarities = ops.matmul(projected_inputs_norm, expert_embeddings_norm)

        # Apply temperature
        if self.learnable_temperature:
            gate_logits = cosine_similarities * self.temperature_param
        else:
            gate_logits = cosine_similarities * self.temperature_param

        # Top-k selection
        if self.top_k < self.num_experts:
            top_k_logits, top_k_indices = ops.top_k(gate_logits, k=self.top_k)

            # Create mask for selected experts
            mask = ops.zeros_like(gate_logits)
            for i in range(self.top_k):
                mask = ops.scatter_update(
                    mask,
                    ops.expand_dims(top_k_indices[:, i], axis=-1),
                    1.0
                )

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

        # Reshape back to original batch structure
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.num_experts]
            expert_weights = ops.reshape(expert_weights, new_shape)
            expert_indices = ops.reshape(expert_indices, new_shape)

        # Prepare auxiliary information
        auxiliary_info = {
            'gate_logits': gate_logits,
            'expert_weights': expert_weights,
            'cosine_similarities': cosine_similarities,
            'raw_gate_probs': ops.softmax(gate_logits, axis=-1)
        }

        return expert_weights, expert_indices, auxiliary_info

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
            'embedding_dim': self.embedding_dim,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'learnable_temperature': self.learnable_temperature,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


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

        self.num_slots = num_slots
        self.kernel_initializer = initializers.get(kernel_initializer)

        # Sublayers initialized in build()
        self.phi_dense = None  # For computing slot attention weights
        self.slot_embeddings = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the SoftMoE gating layers."""
        self._built_input_shape = input_shape
        hidden_dim = input_shape[-1]

        # Dense layer for computing attention weights
        self.phi_dense = layers.Dense(
            units=self.num_experts * self.num_slots,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            name='phi_dense'
        )

        # Learnable slot embeddings
        self.slot_embeddings = self.add_weight(
            name='slot_embeddings',
            shape=(self.num_experts, self.num_slots, hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True
        )

        # Build sublayers
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
        # inputs: [batch, seq_len, hidden_dim]
        # phi_weights: [batch, seq_len, num_experts, num_slots]
        inputs_expanded = ops.expand_dims(inputs, axis=2)  # [batch, seq_len, 1, hidden_dim]
        phi_weights_expanded = ops.expand_dims(phi_weights, axis=-1)  # [batch, seq_len, num_experts, num_slots, 1]

        # Weighted sum to create slots
        soft_slots = ops.sum(
            inputs_expanded * phi_weights_expanded,
            axis=1
        )  # [batch, num_experts, num_slots, hidden_dim]

        # Flatten slots for expert processing
        expert_inputs = ops.reshape(
            soft_slots,
            (batch_size, self.num_experts, self.num_slots * hidden_dim)
        )

        # For SoftMoE, all experts are used with equal weight
        expert_weights = ops.ones((batch_size, seq_len, self.num_experts)) / self.num_experts
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
            'expert_weights': expert_weights
        }

        return expert_weights, expert_indices, auxiliary_info

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
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
        expert_weights: Expert selection weights [batch, seq_len, num_experts].
        gate_probs: Raw gating probabilities [batch, seq_len, num_experts].
        num_experts: Total number of experts.
        aux_loss_weight: Weight for the auxiliary loss.

    Returns:
        Auxiliary load balancing loss scalar.

    Example:
        ```python
        aux_loss = compute_auxiliary_loss(
            expert_weights=expert_weights,
            gate_probs=gate_probs,
            num_experts=8,
            aux_loss_weight=0.01
        )
        ```
    """
    # Compute fraction of tokens dispatched to each expert
    expert_mask = ops.cast(expert_weights > 0, expert_weights.dtype)
    tokens_per_expert = ops.mean(expert_mask, axis=(0, 1))  # [num_experts]

    # Compute average gate probability for each expert
    avg_gate_probs = ops.mean(gate_probs, axis=(0, 1))  # [num_experts]

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

    Example:
        ```python
        z_loss = compute_z_loss(
            gate_logits=gate_logits,
            z_loss_weight=1e-3
        )
        ```
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
        ```
    """
    if gating_type == 'linear':
        return LinearGating(num_experts=num_experts, **kwargs)
    elif gating_type == 'cosine':
        return CosineGating(num_experts=num_experts, **kwargs)
    elif gating_type == 'softmoe':
        return SoftMoEGating(num_experts=num_experts, **kwargs)
    else:
        raise ValueError(f"Unsupported gating type: {gating_type}")