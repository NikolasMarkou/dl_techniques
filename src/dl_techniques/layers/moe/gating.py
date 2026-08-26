"""
Gating network implementations for Mixture of Experts (MoE) models.

This module provides various gating mechanisms (routers) that determine how
inputs are distributed to expert networks, including linear gating, cosine
similarity gating, and SoftMoE approaches.
"""

import keras
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms import create_normalization_layer
from ...constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------

def _mask_neg_inf(dtype: Any) -> float:
    """Return a dtype-appropriate large-negative value for softmax masking."""
    dtype_str = str(dtype)
    if 'float16' in dtype_str or 'bfloat16' in dtype_str:
        return -1e4
    return -1e9

# ---------------------------------------------------------------------

def _min_temperature(dtype: Any) -> float:
    """Return the smallest softmax temperature that is safe in ``dtype``.

    # DECISION plan-2026-08-26T100331-f3744602/D-008
    The floor is derived from ``dtype``, NOT from ``keras.backend.epsilon()``.
    Do not "simplify" this back to a single float32-scale constant. Cosine
    similarities are bounded in ``[-1, 1]``, so dividing by ``t`` yields logits
    bounded by ``1 / t``. Two dtype-dependent ceilings must be respected:

    * the finite range of the dtype (float16 tops out at ``65504``), and
    * the softmax mask sentinel ``_mask_neg_inf(dtype)`` (``-1e4`` for float16,
      ``-1e9`` for float32), which must stay an order of magnitude BELOW every
      real logit or top-k masking silently stops masking.

    The floor is therefore chosen so that ``1 / t <= |_mask_neg_inf(dtype)| / 10``,
    i.e. ``1e-3`` for float16/bfloat16 and ``1e-8`` for float32 -- with the
    float32 branch further tightened to ``keras.backend.epsilon()`` (``1e-7``),
    which is stricter, so float32 behaviour is unchanged. MEASURED at HEAD under
    ``mixed_float16``: ``temperature_param = 1e-6`` -- a no-op for the old
    ``epsilon()`` floor -- produced ``gate_logits min/max: -inf inf`` and 80/80
    NaN ``expert_weights``. See decisions.md D-008.

    :param dtype: Dtype the gate logits will be computed in.
    :type dtype: Any
    :return: Minimum safe temperature for that dtype.
    :rtype: float
    """
    sentinel_headroom = 10.0 / abs(_mask_neg_inf(dtype))
    return max(sentinel_headroom, keras.backend.epsilon())

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BaseGating(keras.layers.Layer, ABC):
    """
    Abstract base class for MoE gating networks.

    Defines the interface for all gating implementations, ensuring consistent
    behavior across different routing strategies. Each gating subclass computes
    expert selection weights, indices, and auxiliary information for load-balancing
    losses. Follows modern Keras 3 patterns with proper serialization support.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────┐
        │      BaseGating (ABC)     │
        │                           │
        │  Input ──► call() ──────► (expert_weights,
        │                            expert_indices,
        │                            auxiliary_info)
        └───────────────────────────┘

    :param num_experts: Number of expert networks to route to.
    :type num_experts: int
    :param name: Name for the gating layer.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the base Layer class.
    :type kwargs: Any
    """

    def __init__(
            self,
            num_experts: int,
            norm_type: Optional[str] = None,
            norm_config: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the base gating layer."""
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        self.num_experts = num_experts
        self.norm_type = norm_type
        self.norm_config = dict(norm_config) if norm_config else {}

        # Optional pre-gating normalization layer via factory
        if self.norm_type is not None:
            self.pre_norm = create_normalization_layer(
                self.norm_type, name='pre_gate_norm', **self.norm_config
            )
        else:
            self.pre_norm = None

    @abstractmethod
    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Compute gating scores and routing information.

        :param inputs: Input tensor to route.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Tuple of (expert_weights, expert_indices, auxiliary_info).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_experts': self.num_experts,
            'norm_type': self.norm_type,
            'norm_config': self.norm_config,
        })
        return config


@keras.saving.register_keras_serializable()
class LinearGating(BaseGating):
    """
    Linear gating network with optional noise and top-k expert selection.

    Implements the most common gating mechanism using a linear transformation
    ``W * x + b`` followed by softmax normalization and top-k selection. During
    training, Gaussian noise can be injected into the gating logits via a
    learned noise scaling network to improve load balancing across experts:
    ``logits = W_g * x + softplus(W_n * x) * N(0, noise_std)``.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────┐
        │            LinearGating                  │
        │                                          │
        │  Input ──► Dense (gate_logits)           │
        │              │                           │
        │              ├─► + Noise (training only) │
        │              ▼                           │
        │           Top-K Selection                │
        │              │                           │
        │              ▼                           │
        │           Softmax (masked)               │
        │              │                           │
        │              ▼                           │
        │  (expert_weights, expert_indices, aux)   │
        └─────────────────────────────────────────┘

    :param num_experts: Number of expert networks.
    :type num_experts: int
    :param top_k: Number of experts to select per token. Must be <= num_experts.
    :type top_k: int
    :param use_bias: Whether to use bias in the linear transformation.
    :type use_bias: bool
    :param add_noise: Whether to add noise to gating logits during training.
    :type add_noise: bool
    :param noise_std: Standard deviation of the noise.
    :type noise_std: float
    :param kernel_initializer: Weight initialization strategy for gate weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Bias initialization strategy.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            num_experts: int,
            top_k: int = 1,
            use_bias: bool = False,
            add_noise: bool = True,
            noise_std: float = 1.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
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
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # CREATE sublayers in __init__ (unbuilt)
        self.gate_dense = keras.layers.Dense(
            units=num_experts,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name='gate_dense'
        )

        if add_noise:
            self.noise_dense = keras.layers.Dense(
                units=num_experts,
                use_bias=False,
                kernel_initializer='zeros',
                name='noise_dense'
            )
        else:
            self.noise_dense = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the linear gating layers."""
        if self.built:
            return
        # BUILD all sublayers explicitly
        if self.pre_norm is not None:
            self.pre_norm.build(input_shape)

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
        original_shape = keras.ops.shape(inputs)

        # Optional pre-gating normalization (operates on full-rank input)
        if self.pre_norm is not None:
            inputs = self.pre_norm(inputs, training=training)

        # Reshape to 2D for processing if needed
        if len(original_shape) > 2:
            inputs_flat = keras.ops.reshape(inputs, (-1, original_shape[-1]))
        else:
            inputs_flat = inputs

        # Compute gating logits
        gate_logits = self.gate_dense(inputs_flat)

        # Add noise during training
        if self.add_noise and training and self.noise_dense is not None:
            noise_logits = self.noise_dense(inputs_flat)
            noise = keras.random.normal(
                shape=keras.ops.shape(noise_logits),
                mean=0.0,
                stddev=1.0,
                dtype=inputs.dtype
            )
            # Apply softplus to ensure positive noise std
            noise_std = keras.ops.softplus(noise_logits) * self.noise_std
            gate_logits = gate_logits + noise * noise_std

        # Top-k selection
        if self.top_k < self.num_experts:
            top_k_logits, top_k_indices = keras.ops.top_k(gate_logits, k=self.top_k)

            # Create mask for selected experts using one_hot
            top_k_one_hot = keras.ops.one_hot(top_k_indices, self.num_experts, dtype=gate_logits.dtype)
            mask = keras.ops.sum(top_k_one_hot, axis=-2)

            # Apply mask to logits (set non-selected to large-negative)
            neg_inf = _mask_neg_inf(gate_logits.dtype)
            masked_logits = keras.ops.where(
                mask > 0,
                gate_logits,
                keras.ops.full_like(gate_logits, neg_inf)
            )
            expert_weights = keras.ops.softmax(masked_logits, axis=-1)
            expert_indices = top_k_indices
        else:
            # Use all experts
            expert_weights = keras.ops.softmax(gate_logits, axis=-1)
            expert_indices = keras.ops.arange(self.num_experts, dtype='int32')
            expert_indices = keras.ops.broadcast_to(
                expert_indices[None, :],
                (keras.ops.shape(gate_logits)[0], self.num_experts)
            )

        # Reshape back to original batch structure if needed
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.num_experts]
            expert_weights = keras.ops.reshape(expert_weights, new_shape)
            if self.top_k < self.num_experts:
                new_indices_shape = list(original_shape[:-1]) + [self.top_k]
                expert_indices = keras.ops.reshape(expert_indices, new_indices_shape)
            else:
                expert_indices = keras.ops.reshape(expert_indices, new_shape)

        # Prepare auxiliary information for load balancing loss
        raw_gate_probs = keras.ops.softmax(gate_logits, axis=-1)
        auxiliary_info = {
            'gate_logits': gate_logits,
            'expert_weights': expert_weights,
            'raw_gate_probs': raw_gate_probs
        }

        return expert_weights, expert_indices, auxiliary_info

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[
        Tuple[Optional[int], ...],
        Tuple[Optional[int], ...],
        Dict[str, Tuple[Optional[int], ...]],
    ]:
        """Compute output shapes for (expert_weights, expert_indices, aux_info)."""
        # expert_weights: same leading dims + num_experts
        weights_shape = tuple(list(input_shape[:-1]) + [self.num_experts])
        # expert_indices: same leading dims + top_k (or num_experts)
        k = self.top_k if self.top_k < self.num_experts else self.num_experts
        indices_shape = tuple(list(input_shape[:-1]) + [k])
        return weights_shape, indices_shape, {}

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'top_k': self.top_k,
            'use_bias': self.use_bias,
            'add_noise': self.add_noise,
            'noise_std': self.noise_std,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer)
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CosineGating(BaseGating):
    """
    Cosine similarity-based gating network for hypersphere expert routing.

    Operates in a normalized embedding space, computing cosine similarity
    ``cos(theta) = (x_proj / ||x_proj||) . (e_k / ||e_k||)`` between input
    representations and learnable expert embeddings. The similarity scores
    are divided by a (optionally learnable) temperature ``tau`` before top-k
    selection and softmax normalization (standard softmax-temperature
    semantics: larger ``tau`` -> flatter distribution). This can provide
    better domain generalization compared to linear gating.

    .. note::
       Behavior changed: prior implementations multiplied by ``tau`` (acting
       as an inverse temperature). The current implementation divides, so
       checkpoints/configs with the same numeric ``temperature`` will now
       produce flatter (not sharper) routing distributions.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────┐
        │             CosineGating                   │
        │                                            │
        │  Input ──► Dense (project to embed_dim)    │
        │              │                             │
        │              ▼                             │
        │  L2-normalize ──► cosine_sim(x, E_experts) │
        │              │                             │
        │              ├──► * temperature             │
        │              ▼                             │
        │           Top-K Selection                  │
        │              │                             │
        │              ▼                             │
        │           Softmax (masked)                 │
        │              │                             │
        │              ▼                             │
        │  (expert_weights, expert_indices, aux)     │
        └────────────────────────────────────────────┘

    :param num_experts: Number of expert networks.
    :type num_experts: int
    :param embedding_dim: Dimension of expert embeddings.
    :type embedding_dim: int
    :param top_k: Number of experts to select per token.
    :type top_k: int
    :param temperature: Temperature parameter for softmax scaling.
    :type temperature: float
    :param learnable_temperature: Whether temperature is a learnable parameter.
    :type learnable_temperature: bool
    :param kernel_initializer: Weight initialization strategy.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            num_experts: int,
            embedding_dim: int = 256,
            top_k: int = 1,
            temperature: float = 1.0,
            learnable_temperature: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
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
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE sublayers in __init__
        self.linear_projection = keras.layers.Dense(
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
        if self.built:
            return
        # CREATE weights in build()
        self.expert_embeddings = self.add_weight(
            name='expert_embeddings',
            shape=(self.embedding_dim, self.num_experts),
            initializer=self.kernel_initializer,
            trainable=True
        )

        if self.learnable_temperature:
            # DECISION plan-2026-08-26T100331-f3744602/D-008
            # The constraint is the real lower bound; the ``keras.ops.maximum`` in
            # ``call`` is only the second line of defence. Do not drop it in
            # favour of the in-call clamp alone: nothing else stops an optimizer
            # from parking the variable at (or below) zero, where the gradient
            # of ``1 / t`` explodes and every subsequent step is computed from a
            # temperature the forward pass never actually used. The constraint is
            # re-created from ``self.compute_dtype`` on every ``build()``, so it
            # survives ``get_config``/``from_config`` and ``.keras`` reload
            # without being serialized itself. See decisions.md D-008.
            self.temperature_param = self.add_weight(
                name='temperature',
                shape=(),
                initializer=keras.initializers.Constant(value=self.temperature),
                constraint=ValueRangeConstraint(
                    min_value=_min_temperature(self.compute_dtype)
                ),
                trainable=True
            )

        # BUILD sublayers explicitly
        if self.pre_norm is not None:
            self.pre_norm.build(input_shape)
        self.linear_projection.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the cosine gating network."""
        # Optional pre-gating normalization (operates on full-rank input)
        if self.pre_norm is not None:
            inputs = self.pre_norm(inputs, training=training)

        original_shape = keras.ops.shape(inputs)

        # Reshape to 2D for processing if needed
        if len(original_shape) > 2:
            inputs_flat = keras.ops.reshape(inputs, (-1, original_shape[-1]))
        else:
            inputs_flat = inputs

        # Project input to embedding space
        projected_inputs = self.linear_projection(inputs_flat)

        # Normalize projected inputs and expert embeddings
        projected_inputs_norm = keras.ops.normalize(projected_inputs, axis=-1)
        expert_embeddings_norm = keras.ops.normalize(self.expert_embeddings, axis=0)

        # Compute cosine similarities
        cosine_similarities = keras.ops.matmul(projected_inputs_norm, expert_embeddings_norm)

        # Apply temperature (standard softmax-temperature semantics: divide).
        # Larger ``temperature`` -> flatter distribution. The divisor is floored
        # at ``_min_temperature(dtype)``, a bound derived from the dtype the
        # logits are actually computed in -- 1e-3 under float16/bfloat16, 1e-7
        # under float32. Because cosine similarities are bounded in [-1, 1], the
        # floor bounds |gate_logits| by 1 / floor, which keeps them finite AND an
        # order of magnitude inside the top-k mask sentinel `_mask_neg_inf`.
        # A float32-scale epsilon does NOT deliver this under mixed precision:
        # 1e-7 admits logits of ~1e7, past float16's max of 65504 (MEASURED:
        # -inf/inf logits, NaN weights). No-op for any temperature above the floor.
        temperature_value = self.temperature_param if self.learnable_temperature else self.temperature
        temperature_value = keras.ops.maximum(
            temperature_value, _min_temperature(cosine_similarities.dtype))
        gate_logits = cosine_similarities / temperature_value

        # Top-k selection
        if self.top_k < self.num_experts:
            top_k_logits, top_k_indices = keras.ops.top_k(gate_logits, k=self.top_k)

            # Create mask for selected experts using one_hot
            top_k_one_hot = keras.ops.one_hot(top_k_indices, self.num_experts, dtype=gate_logits.dtype)
            mask = keras.ops.sum(top_k_one_hot, axis=-2)

            # Apply mask to logits
            neg_inf = _mask_neg_inf(gate_logits.dtype)
            masked_logits = keras.ops.where(
                mask > 0,
                gate_logits,
                keras.ops.full_like(gate_logits, neg_inf)
            )
            expert_weights = keras.ops.softmax(masked_logits, axis=-1)
            expert_indices = top_k_indices
        else:
            # Use all experts
            expert_weights = keras.ops.softmax(gate_logits, axis=-1)
            expert_indices = keras.ops.arange(self.num_experts, dtype='int32')
            expert_indices = keras.ops.broadcast_to(
                expert_indices[None, :],
                (keras.ops.shape(gate_logits)[0], self.num_experts)
            )

        # Reshape back to original batch structure if needed
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.num_experts]
            expert_weights = keras.ops.reshape(expert_weights, new_shape)
            if self.top_k < self.num_experts:
                new_indices_shape = list(original_shape[:-1]) + [self.top_k]
                expert_indices = keras.ops.reshape(expert_indices, new_indices_shape)
            else:
                expert_indices = keras.ops.reshape(expert_indices, new_shape)

        # Prepare auxiliary information
        raw_gate_probs = keras.ops.softmax(gate_logits, axis=-1)
        auxiliary_info = {
            'gate_logits': gate_logits,
            'expert_weights': expert_weights,
            'cosine_similarities': cosine_similarities,
            'raw_gate_probs': raw_gate_probs
        }

        return expert_weights, expert_indices, auxiliary_info

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[
        Tuple[Optional[int], ...],
        Tuple[Optional[int], ...],
        Dict[str, Tuple[Optional[int], ...]],
    ]:
        """Compute output shapes for (expert_weights, expert_indices, aux_info)."""
        weights_shape = tuple(list(input_shape[:-1]) + [self.num_experts])
        k = self.top_k if self.top_k < self.num_experts else self.num_experts
        indices_shape = tuple(list(input_shape[:-1]) + [k])
        return weights_shape, indices_shape, {}

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'learnable_temperature': self.learnable_temperature,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer)
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SoftMoEGating(BaseGating):
    """
    Soft Mixture-of-Experts gating via differentiable slot assignment.

    Unlike traditional hard routing, SoftMoE computes weighted combinations
    of all input tokens to create ``num_slots`` "soft slots" per expert. Per
    Puigcerver et al. (2023), two softmaxes are computed from the same logit
    tensor ``L`` of shape ``[batch, seq, num_experts, num_slots]``:

    - **Dispatch weights** ``D = softmax(L, axis=seq)`` are used to build
      soft slots: ``slot_{e,s} = sum_t D_{t,e,s} * x_t``.
    - **Combine weights** ``C = softmax(L_reshaped_to_(e*s), axis=-1)`` are
      used to combine expert-slot outputs back to token positions:
      ``y_t = sum_{e,s} C_{t,e,s} * f_e(slot_{e,s})``.

    This avoids token dropping and load balancing issues at the cost of
    increased computation proportional to ``seq_len * num_experts * num_slots``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │            SoftMoEGating                 │
        │                                          │
        │  Input(batch, seq, dim)                  │
        │         │                                │
        │         ▼                                │
        │  Dense(num_experts * num_slots) ──► phi  │
        │         │                                │
        │         ▼                                │
        │  softmax(phi, axis=seq_len)              │
        │         │                                │
        │         ▼                                │
        │  Weighted Sum ──► soft_slots             │
        │  (batch, num_experts, slots*dim)         │
        │         │                                │
        │         ▼                                │
        │  (expert_weights, expert_indices, aux)   │
        └──────────────────────────────────────────┘

    :param num_experts: Number of expert networks.
    :type num_experts: int
    :param num_slots: Number of input slots per expert.
    :type num_slots: int
    :param kernel_initializer: Weight initialization strategy.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments.
    :type kwargs: Any
    """

    def __init__(
            self,
            num_experts: int,
            num_slots: int = 4,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        """Initialize the SoftMoE gating network."""
        super().__init__(num_experts=num_experts, **kwargs)

        # Validate inputs
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")

        self.num_slots = num_slots
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE sublayers in __init__
        self.phi_dense = keras.layers.Dense(
            units=num_experts * num_slots,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            name='phi_dense'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the SoftMoE gating layers."""
        if self.built:
            return
        hidden_dim = input_shape[-1]
        if hidden_dim is None:
            raise ValueError("Hidden dimension must be known for SoftMoE")

        # BUILD sublayers explicitly
        if self.pre_norm is not None:
            self.pre_norm.build(input_shape)
        self.phi_dense.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the SoftMoE gating network."""
        if self.pre_norm is not None:
            inputs = self.pre_norm(inputs, training=training)

        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]
        hidden_dim = keras.ops.shape(inputs)[-1]

        # Compute logits for slot assignment
        phi_logits = self.phi_dense(inputs)  # [batch, seq_len, num_experts * num_slots]
        phi_logits = keras.ops.reshape(
            phi_logits, (batch_size, seq_len, self.num_experts, self.num_slots)
        )

        # Dispatch: softmax over sequence dimension -> used to build soft slots
        dispatch_weights = keras.ops.softmax(phi_logits, axis=1)  # [b, s, e, l]

        # Combine: softmax over (experts * slots) per token -> used to combine
        # expert outputs back to token positions.
        phi_logits_flat = keras.ops.reshape(
            phi_logits, (batch_size, seq_len, self.num_experts * self.num_slots)
        )
        combine_weights_flat = keras.ops.softmax(phi_logits_flat, axis=-1)
        combine_weights = keras.ops.reshape(
            combine_weights_flat,
            (batch_size, seq_len, self.num_experts, self.num_slots),
        )

        # Compute soft input slots for each expert using dispatch weights
        inputs_expanded = keras.ops.expand_dims(keras.ops.expand_dims(inputs, axis=2), axis=3)  # [b, s, 1, 1, h]
        dispatch_expanded = keras.ops.expand_dims(dispatch_weights, axis=-1)  # [b, s, e, l, 1]

        soft_slots = keras.ops.sum(
            inputs_expanded * dispatch_expanded,  # Broadcasts to [b, s, e, l, h]
            axis=1,
        )  # Sum over seq -> [b, e, l, h]

        # Flatten slots for expert processing
        expert_inputs = keras.ops.reshape(
            soft_slots,
            (batch_size, self.num_experts, self.num_slots * hidden_dim),
        )

        # Per-token, per-expert routing weight = marginal of combine_weights over slots.
        # Shape: [batch, seq_len, num_experts].
        expert_weights = keras.ops.sum(combine_weights, axis=-1)
        expert_indices = keras.ops.arange(self.num_experts, dtype='int32')
        expert_indices = keras.ops.broadcast_to(
            expert_indices[None, None, :],
            (batch_size, seq_len, self.num_experts),
        )

        # Prepare auxiliary information
        auxiliary_info = {
            'dispatch_weights': dispatch_weights,
            'combine_weights': combine_weights,
            'soft_slots': soft_slots,
            'expert_inputs': expert_inputs,
            'expert_weights': expert_weights,
            # Raw dispatch logits, exposed for inspection and to keep this
            # gating type's aux-info contract shaped like the others'. NOT used
            # for z-loss: `MixtureOfExperts.call` skips both the auxiliary loss
            # and the z-loss entirely when `gating_type == 'softmoe'`, because
            # SoftMoE dispatches every token to every expert and so has no load
            # imbalance for those losses to regularize.
            'gate_logits': phi_logits,
            # Per-token marginal probability over experts: softmax over the
            # expert axis (2), then average out the slot axis. Shape
            # [batch, seq_len, num_experts] to match the LinearGating /
            # CosineGating aux-info contract. Same caveat as `gate_logits`: the
            # contract is matched, but compute_auxiliary_loss is not called for
            # this gating type.
            'raw_gate_probs': keras.ops.mean(keras.ops.softmax(phi_logits, axis=2), axis=-1),
        }

        return expert_weights, expert_indices, auxiliary_info

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[
        Tuple[Optional[int], ...],
        Tuple[Optional[int], ...],
        Dict[str, Tuple[Optional[int], ...]],
    ]:
        """Compute output shapes for (expert_weights, expert_indices, aux_info)."""
        # SoftMoE: expert_weights shape = (batch, seq_len, num_experts)
        weights_shape = tuple(list(input_shape[:-1]) + [self.num_experts])
        indices_shape = weights_shape  # All experts used
        return weights_shape, indices_shape, {}

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_slots': self.num_slots,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer)
        })
        return config

# ---------------------------------------------------------------------

def compute_auxiliary_loss(
        expert_weights: keras.KerasTensor,
        gate_probs: keras.KerasTensor,
        num_experts: int,
        aux_loss_weight: float = 0.01
) -> keras.KerasTensor:
    """
    Compute auxiliary load balancing loss for MoE training.

    Encourages uniform token distribution across experts via
    ``L_aux = aux_weight * N * sum(f_i * P_i)`` where ``f_i`` is the
    fraction of tokens dispatched to expert *i* and ``P_i`` is the average
    gate probability for expert *i*.

    .. important::

        **This loss scales with** ``top_k``, **and is deliberately not
        normalized.** ``f_i`` is the fraction of tokens *dispatched to* expert
        *i*, and at ``top_k = k`` every token is dispatched to ``k`` experts, so
        ``sum_i f_i = k`` rather than ``1``. The formula is the Switch
        Transformer's, which is calibrated for ``top_k = 1``.

        The consequence is that ``aux_loss_weight`` does **not** mean the same
        thing at different ``top_k``. Perfectly balanced routing does not drive
        this loss to zero -- it drives it to a floor of exactly
        ``aux_loss_weight * top_k``:

        .. code-block:: text

            MEASURED (this repo, 2026-08-26; exact uniform gate probs,
            round-robin balanced dispatch, 4096 tokens):

              aux_loss_weight = 0.01

              num_experts   top_k=1    top_k=2    top_k=4
              -----------   --------   --------   --------
                        4   0.010000   0.020000   0.040000
                        8   0.010000   0.020000   0.040000
                       16   0.010000   0.020000   0.040000
                       64   0.010000   0.020000   0.040000

              floor = aux_loss_weight * top_k, exactly, and is INDEPENDENT of
              num_experts and of the token count.

              The worst case (every token routed to the same k experts) is
              aux_loss_weight * num_experts -- 0.08 for all three columns at
              num_experts=8. So the usable dynamic range, worst/floor, is
              num_experts / top_k: raising top_k lifts the floor and shrinks the
              headroom the regularizer actually has to work with.

        A caller who wants a ``top_k``-invariant regularization strength should
        divide their intended weight by ``top_k``. Nothing here does it for
        them; see the anchored decision below.

    :param expert_weights: Expert selection weights ``[batch, ..., num_experts]``.
    :type expert_weights: keras.KerasTensor
    :param gate_probs: Raw gating probabilities ``[batch, ..., num_experts]``.
    :type gate_probs: keras.KerasTensor
    :param num_experts: Total number of experts.
    :type num_experts: int
    :param aux_loss_weight: Weight for the auxiliary loss.
    :type aux_loss_weight: float
    :return: Auxiliary load balancing loss scalar, always ``float32``.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-08-26T100331-f3744602/D-017
    # The `top_k` scaling documented above is DELIBERATE and measured, not an
    # oversight. Do NOT "fix" it by dividing `tokens_per_expert` (or the final
    # loss) by top_k to make the balanced floor top_k-invariant. Doing so
    # rescales the load-balancing regularizer for every shipped consumer --
    # models/language/qwen/qwen3.py (top_k=8) and qwen3_next.py (top_k=10) both
    # train with it -- and there is no measurement showing the published
    # Switch-Transformer calibration is wrong, only that it is a top-1 property
    # being used at top_k > 1. The relationship is pinned by
    # tests/test_layers/test_moe/test_gating.py::TestAuxiliaryLossTopKScaling.
    # See decisions.md D-017.
    #
    # DECISION plan-2026-08-26T155709-fb07cf4e/D-005
    # [CORRECTED iter-2 -- amends, does not delete, the D-017 note that stood here]
    #
    # The superseded note said: "unlike ``compute_z_loss`` (D-009), this reduction
    # needs NO float32 upcast, and one was deliberately not added", because ``f_i``
    # is a mean of a 0/1 mask and ``P_i`` a mean of softmax probabilities, so both
    # lie in [0, 1] and ``sum_i(f_i * P_i) <= max_i(f_i) * sum_i(P_i) = 1``, making
    # the loss bounded by ``num_experts`` and finite under ``mixed_float16`` at
    # every token count up to 2**20 and ``num_experts`` up to 4096 (worst case,
    # fully-imbalanced routing, ``N=512`` -> 5.121 fp16 vs 5.120 float32).
    #
    # That OVERFLOW analysis is RIGHT and still holds -- and it answered the wrong
    # question. The failure mode is not magnitude, it is the DTYPE OF THE RETURNED
    # TENSOR. `layer.py:278` hands this value to `add_loss`. Keras'
    # `_aggregate_additional_loss` (`keras/src/trainers/trainer.py:389-400`) casts
    # only NON-float losses to `floatx()`, so a float16 value passes through
    # untouched into the list that `compute_loss` reduces at `trainer.py:365`
    # (`total_loss = keras.ops.sum(losses)`) alongside the float32 compiled loss and the
    # float32 z-loss. MEASURED at HEAD c38d5f17b, one `model.fit()` step,
    # `mixed_float16`, 4 experts, linear gating, top_k=2, the shipped default
    # `aux_loss_weight=0.01`:
    #   TypeError: Cannot convert a list containing a tensor of dtype
    #   <dtype: 'float16'> to <dtype: 'float32'>
    # i.e. `model.fit()` CRASHED for every mixed-precision MoE consumer --
    # models/language/qwen/qwen3.py (top_k=8) and qwen3_next.py (top_k=10).
    #
    # INVARIANT: this function returns float32 under every global dtype policy.
    # Do NOT remove the cast below as "redundant because the value is bounded";
    # boundedness is not the property being defended. Reverting it restores the
    # `model.fit()` crash above, and NO amount of finiteness or overflow testing
    # can see it -- the value is perfectly finite, it is simply the wrong dtype
    # for a list Keras reduces without casting. The cast is a no-op under the
    # default float32 policy (MEASURED: bitwise-identical, 0.0 delta), so it costs
    # nothing there. Pinned by
    # tests/test_layers/test_moe/test_the_auxiliary_loss_survives_a_mixed_precision_fit.py.
    # See decisions.md D-005 (and D-017 of plan-2026-08-26T100331-f3744602, which
    # this amends).
    # Determine axes for token-wise mean calculation (all but the last axis)
    num_token_axes = len(keras.ops.shape(expert_weights)) - 1
    token_axes = list(range(num_token_axes))

    # Compute fraction of tokens dispatched to each expert
    expert_mask = keras.ops.cast(expert_weights > 0, expert_weights.dtype)
    tokens_per_expert = keras.ops.mean(expert_mask, axis=token_axes)  # [num_experts]

    # Compute average gate probability for each expert
    avg_gate_probs = keras.ops.mean(gate_probs, axis=token_axes)  # [num_experts]

    # Auxiliary loss = N * sum(f_i * P_i) where f_i is fraction, P_i is avg prob
    aux_loss = num_experts * keras.ops.sum(tokens_per_expert * avg_gate_probs)

    return keras.ops.cast(aux_loss_weight * aux_loss, 'float32')

# ---------------------------------------------------------------------

def compute_z_loss(
        gate_logits: keras.KerasTensor,
        z_loss_weight: float = 1e-3
) -> keras.KerasTensor:
    """
    Compute router z-loss for entropy regularization.

    Penalizes ``mean(logsumexp(logits)^2)`` to encourage confident routing
    decisions and prevent logit explosion.

    :param gate_logits: Raw gate logits ``[batch, seq_len, num_experts]``.
    :type gate_logits: keras.KerasTensor
    :param z_loss_weight: Weight for the z-loss.
    :type z_loss_weight: float
    :return: Router z-loss scalar, always ``float32``.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-08-26T100331-f3744602/D-009
    # The float32 upcast is load-bearing; do not remove it as redundant and do
    # not restructure the expression around it. `logsumexp` grows like the
    # largest logit and `square` then doubles that exponent, so under
    # `mixed_float16` (where a Dense gate emits float16) any logit past ~256
    # squares out of float16's 65504 range. MEASURED at HEAD: logits in
    # [-426.75, 435.0] gave `compute_z_loss = inf` where the float32 reference is
    # 70.707. `layer.py:262-267` feeds that value straight into `add_loss`
    # whenever `training` is truthy and `z_loss_weight > 0` -- the GatingConfig
    # DEFAULT (1e-3) -- so the `inf` poisons the whole model's loss and gradients
    # silently, with no warning and no exception. This mirrors the D-064 template
    # at `layer.py:320-337`: cast at the numerically fragile boundary, leave the
    # arithmetic alone. See decisions.md D-009.
    gate_logits = keras.ops.cast(gate_logits, 'float32')

    # Compute logsumexp for each token
    logsumexp = keras.ops.logsumexp(gate_logits, axis=-1, keepdims=False)  # [batch, seq_len]

    # Z-loss is the squared mean of logsumexp
    z_loss = keras.ops.mean(keras.ops.square(logsumexp))

    return z_loss_weight * z_loss

# ---------------------------------------------------------------------

def create_gating(gating_type: str, num_experts: int, **kwargs) -> BaseGating:
    """
    Factory function to create gating networks.

    :param gating_type: Type of gating to create (``'linear'``, ``'cosine'``, ``'softmoe'``).
    :type gating_type: str
    :param num_experts: Number of expert networks.
    :type num_experts: int
    :param kwargs: Configuration parameters for the gating network.
    :type kwargs: Any
    :return: Configured gating network.
    :rtype: BaseGating
    :raises ValueError: If gating_type is not supported.
    """
    # Shared keys forwarded to every BaseGating subclass.
    shared_keys = ['norm_type', 'norm_config']
    shared_kwargs = {k: v for k, v in kwargs.items() if k in shared_keys}

    if gating_type == 'linear':
        linear_keys = ['top_k', 'use_bias', 'add_noise', 'noise_std',
                       'kernel_initializer', 'bias_initializer']
        linear_kwargs = {k: v for k, v in kwargs.items() if k in linear_keys}
        return LinearGating(num_experts=num_experts, **shared_kwargs, **linear_kwargs)
    elif gating_type == 'cosine':
        cosine_keys = ['embedding_dim', 'top_k', 'temperature',
                       'learnable_temperature', 'kernel_initializer']
        cosine_kwargs = {k: v for k, v in kwargs.items() if k in cosine_keys}
        return CosineGating(num_experts=num_experts, **shared_kwargs, **cosine_kwargs)
    elif gating_type == 'softmoe':
        softmoe_keys = ['num_slots', 'kernel_initializer']
        softmoe_kwargs = {k: v for k, v in kwargs.items() if k in softmoe_keys}
        return SoftMoEGating(num_experts=num_experts, **shared_kwargs, **softmoe_kwargs)
    else:
        raise ValueError(
            f"Unsupported gating type: {gating_type}. "
            f"Supported types: ['linear', 'cosine', 'softmoe']"
        )

# ---------------------------------------------------------------------
