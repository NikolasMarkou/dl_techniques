"""
A Gated Delta Network, a linear-time transformer variant.

This layer provides a sophisticated, recurrent mechanism for sequence modeling,
designed to overcome the limitations of both standard quadratic attention and
simpler linear-time architectures. It fuses concepts from associative memory,
classic neural network learning rules, and modern gated state-space models
to achieve high performance on tasks requiring long-context understanding and
in-context retrieval.

Architecture and Core Concepts:

The Gated DeltaNet operates as a linear transformer, processing sequences
with a recurrent state update that has linear time complexity O(N) with
respect to sequence length. Its primary innovation lies in how it updates its
internal state matrix `S`, which represents the model's associative memory.
This update is governed by a "gated delta rule," which combines two
complementary principles:

1.  **The Delta Rule for Associative Memory:** At each timestep `t`, the model
    updates its memory state `S` based on the current key-value pair
    `(K_t, V_t)`. Unlike vanilla linear attention which uses a simple
    additive update (`S_t = S_{t-1} + K_t ⊗ V_t`), this layer employs a
    delta rule. Originating from error-correction learning (e.g., the
    Widrow-Hoff rule), the delta rule performs a more targeted and powerful
    update. It modifies the memory `S` to better associate `K_t` with `V_t`,
    effectively minimizing the "prediction error" for the current step. This
    makes the model exceptionally adept at forming and recalling precise
    key-value associations, a crucial capability for in-context learning.

2.  **Adaptive Gating Mechanism:** The delta update is modulated by two
    learned, data-dependent gates, `α` (alpha) and `β` (beta), inspired by
    the gating in LSTMs and other modern state-space models like Mamba.
    -   `α_t` acts as a "forget" or "persistence" gate, controlling how much
        of the previous memory state `S_{t-1}` is carried over to the next
        step. A value near 1 preserves memory, while a value near 0 allows
        for rapid erasure of outdated or irrelevant information.
    -   `β_t` acts as a "learning rate" or "update strength" gate, scaling
        the magnitude of the delta rule update for the current timestep.

The synergy of these two components allows for highly flexible memory
management. The gating enables the model to dynamically control the lifespan
of information, while the delta rule provides a precise mechanism for writing
new, error-corrected information into memory.

This implementation is highly configurable, allowing for different normalization
schemes for Q, K, and V, and a customizable output feed-forward network (FFN)
to replace the default output gating mechanism.

References:

This architecture builds upon a line of research aimed at creating more
efficient and powerful sequence models. The core concepts are derived from:
-   Schlag, I., et al. (2021), which introduced the use of the delta rule
    for associative memory in linear transformers (DeltaNet).
-   Recent work on state-space models like Mamba, which demonstrated the
    power of selective, gated state updates.
-   The combined "Gated DeltaNet" architecture, which explicitly merges these
    two ideas for improved performance on long-context tasks.

"""

import keras
from typing import Any, Callable, Dict, Optional, Tuple, Union
from keras import initializers, layers, ops, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .ffn.factory import create_ffn_from_config, FFNType
from .norms import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GatedDeltaNet(keras.layers.Layer):
    """
    Gated DeltaNet layer combining delta rule updates with adaptive gating mechanism.
    This layer is normally input length agnostic,
    however due to limitations of tensorflow framework
    we have to define a hard top limit named max_seq_len

    This layer implements a sophisticated linear transformer variant that combines:
    - Delta rule mechanism for targeted memory updates
    - Adaptive gating for rapid memory erasure and control
    - Configurable normalization (default: Zero-Centered RMS) for Q, K, V
    - Short convolution for position-based addressing
    - Configurable output FFN (default: Gated Linear Unit) for selective flow

    **Intent**: Provide an efficient alternative to standard attention that excels
    at in-context retrieval and long-context understanding while maintaining
    linear complexity. The gating mechanism enables flexible memory control.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, dim])
           ↓
    Q/K/V Linear Projections
           ↓
    Configurable Norm → Short Conv1D (Q, K, V) → Configurable Activation
           ↓                                             ↓
    Alpha/Beta Gating ←----------------------------------┘
           ↓
    Delta Rule Update (with gating)
           ↓
    Reshape & Project
           ↓
    Configurable Output FFN (default: GLU Gate) → Output
    ```

    Args:
        dim: Integer, the model dimension size. Must be positive.
        num_heads: Integer, number of attention heads. Must be positive.
        max_seq_len: Integer, the maximum sequence length for the `while_loop`.
        head_dim: Optional integer, dimension per head. If None, defaults to dim // num_heads.
        conv_kernel_size: Integer, kernel size for short convolution layers. Defaults to 4.
        dropout_rate: Float, dropout rate for regularization. Defaults to 0.0.
        activation: String or callable, activation function applied after convolutions. Defaults to 'silu'.
        normalization_type: NormalizationType, type of normalization to use for Q, K, V.
            Defaults to 'zero_centered_rms_norm'.
        q_norm_args: Optional dict of arguments for the Q normalization layer.
        k_norm_args: Optional dict of arguments for the K normalization layer.
        v_norm_args: Optional dict of arguments for the V normalization layer.
        ffn_type: Optional[FFNType], type of feed-forward network for the output stage.
            If None (default), uses the original gated linear unit output.
            Otherwise, replaces the output projection with a specified FFN from the factory.
        ffn_args: Optional dict of arguments for the custom FFN layer.
        intermediate_size: Optional int, intermediate size for standard FFNs if `ffn_type` is used.
            Defaults to `dim * 4` if not provided.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to False.
        kernel_initializer: String or initializer for weights. Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer for biases. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for weights.
        bias_regularizer: Optional regularizer for biases.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.

    Example:
        ```python
        # Default configuration (Zero-Centered RMSNorm, SiLU activation, GLU output)
        layer = GatedDeltaNet(dim=768, num_heads=12, max_seq_len=2048)

        # Using LayerNorm, GELU activation, and a SwiGLU FFN output
        layer_custom = GatedDeltaNet(
            dim=768,
            num_heads=12,
            max_seq_len=4096,
            activation='gelu',
            normalization_type='layer_norm',
            ffn_type='swiglu',
            intermediate_size=3072, # 768 * 4
            dropout_rate=0.1
        )

        # Usage in model
        inputs = keras.Input(shape=(None, 768))
        outputs = layer_custom(inputs)
        ```
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: Optional[int] = None,
        conv_kernel_size: int = 4,
        dropout_rate: float = 0.0,
        activation: Union[str, Callable] = "silu",
        normalization_type: NormalizationType = "zero_centered_rms_norm",
        q_norm_args: Optional[Dict[str, Any]] = None,
        k_norm_args: Optional[Dict[str, Any]] = None,
        v_norm_args: Optional[Dict[str, Any]] = None,
        ffn_type: Optional[FFNType] = None,
        ffn_args: Optional[Dict[str, Any]] = None,
        intermediate_size: Optional[int] = None,
        use_bias: bool = False,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(
            dim, num_heads, head_dim, conv_kernel_size, dropout_rate, max_seq_len
        )

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.normalization_type = normalization_type
        self.q_norm_args = q_norm_args or (
            {"epsilon": 1e-5, "use_scale": True}
            if normalization_type == "zero_centered_rms_norm"
            else {}
        )
        self.k_norm_args = k_norm_args or (
            {"epsilon": 1e-5, "use_scale": True}
            if normalization_type == "zero_centered_rms_norm"
            else {}
        )
        self.v_norm_args = v_norm_args or (
            {"epsilon": 1e-5, "use_scale": True}
            if normalization_type == "zero_centered_rms_norm"
            else {}
        )
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args or {}
        self.intermediate_size = intermediate_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Compute dimensions
        self.qk_dim = self.num_heads * self.head_dim
        self.v_dim = self.num_heads * self.head_dim * 2

        # Q/K/V projections
        self.q_proj = layers.Dense(self.qk_dim, use_bias=use_bias, name="q_proj")
        self.k_proj = layers.Dense(self.qk_dim, use_bias=use_bias, name="k_proj")
        self.v_proj = layers.Dense(self.v_dim, use_bias=use_bias, name="v_proj")

        # Configurable Normalization layers
        self.q_norm = self._create_normalization_layer("q_norm", self.q_norm_args)
        self.k_norm = self._create_normalization_layer("k_norm", self.k_norm_args)
        self.v_norm = self._create_normalization_layer("v_norm", self.v_norm_args)

        # Short convolution layers (depthwise separable)
        self.q_conv = layers.Conv1D(
            self.qk_dim, conv_kernel_size, padding="causal", groups=self.qk_dim, name="q_conv"
        )
        self.k_conv = layers.Conv1D(
            self.qk_dim, conv_kernel_size, padding="causal", groups=self.qk_dim, name="k_conv"
        )
        self.v_conv = layers.Conv1D(
            self.v_dim, conv_kernel_size, padding="causal", groups=self.v_dim, name="v_conv"
        )

        # Gating parameter projections (alpha and beta)
        self.alpha_proj = layers.Dense(self.num_heads, use_bias=use_bias, name="alpha_proj")
        self.beta_proj = layers.Dense(self.num_heads, use_bias=use_bias, name="beta_proj")

        # Configurable Output FFN
        self.use_default_ffn = self.ffn_type is None
        if self.use_default_ffn:
            self.output_proj = layers.Dense(self.dim, use_bias=use_bias, name="output_proj")
            self.output_gate_linear = layers.Dense(
                self.dim, use_bias=use_bias, name="output_gate_linear"
            )
        else:
            self.output_ffn = self._create_ffn_layer("output_ffn")

        # Configurable activation layer
        self.activation_layer = layers.Activation(self.activation, name="conv_activation")

        # Dropout for regularization
        self.dropout = (
            layers.Dropout(dropout_rate, name="dropout") if dropout_rate > 0.0 else None
        )

        logger.info(
            f"GatedDeltaNet initialized: dim={dim}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}, "
            f"max_seq_len={self.max_seq_len}, activation='{self.activation}', "
            f"norm='{self.normalization_type}', ffn='{self.ffn_type or 'default_gated'}'"
        )

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int],
        conv_kernel_size: int,
        dropout_rate: float,
        max_seq_len: int,
    ) -> None:
        """Validate layer initialization parameters."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim is None and dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}) "
                "when head_dim is None"
            )
        if conv_kernel_size <= 0:
            raise ValueError(
                f"conv_kernel_size must be positive, got {conv_kernel_size}"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def _create_normalization_layer(
        self, name: str, custom_args: Dict[str, Any]
    ) -> keras.layers.Layer:
        """Creates a normalization layer from the factory."""
        try:
            return create_normalization_layer(
                normalization_type=self.normalization_type, name=name, **custom_args
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create '{self.normalization_type}' norm layer named '{name}'. "
                f"Check parameter compatibility. Custom args: {custom_args}. Error: {e}"
            )

    def _create_ffn_layer(self, name: str) -> keras.layers.Layer:
        """Creates an FFN layer from the factory for the output stage."""
        if self.intermediate_size is None:
            self.intermediate_size = self.dim * 4  # Sensible default

        # The FFN's role is to project qk_dim -> dim.
        config = {
            "type": self.ffn_type,
            "name": name,
            "output_dim": self.dim,
            "hidden_dim": self.intermediate_size,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
        }
        config.update(self.ffn_args)
        try:
            return create_ffn_from_config(config)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create '{self.ffn_type}' FFN layer. "
                f"Check for parameter incompatibility. Custom args: {self.ffn_args}. Error: {e}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers."""
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")
        batch_size, seq_len, features = input_shape
        if features != self.dim:
            raise ValueError(f"Input feature dim ({features}) must match layer dim ({self.dim})")

        # Set common initializers/regularizers for dense layers
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.alpha_proj, self.beta_proj]:
            layer.kernel_initializer = self.kernel_initializer
            layer.bias_initializer = self.bias_initializer
            layer.kernel_regularizer = self.kernel_regularizer
            layer.bias_regularizer = self.bias_regularizer
        for layer in [self.q_conv, self.k_conv, self.v_conv]:
            layer.kernel_initializer = self.kernel_initializer
            layer.bias_initializer = self.bias_initializer

        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        q_shape = (batch_size, seq_len, self.qk_dim)
        k_shape = (batch_size, seq_len, self.qk_dim)
        v_shape = (batch_size, seq_len, self.v_dim)
        self.q_norm.build(q_shape)
        self.k_norm.build(k_shape)
        self.v_norm.build(v_shape)
        self.q_conv.build(q_shape)
        self.k_conv.build(k_shape)
        self.v_conv.build(v_shape)
        self.alpha_proj.build(input_shape)
        self.beta_proj.build(input_shape)

        self.activation_layer.build(q_shape)

        ffn_input_shape = (batch_size, seq_len, self.qk_dim)
        if self.use_default_ffn:
            self.output_proj.kernel_initializer = self.kernel_initializer
            self.output_proj.bias_initializer = self.bias_initializer
            self.output_proj.build(ffn_input_shape)
            self.output_gate_linear.kernel_initializer = self.kernel_initializer
            self.output_gate_linear.bias_initializer = self.bias_initializer
            self.output_gate_linear.build((batch_size, seq_len, self.dim))
        else:
            self.output_ffn.build(ffn_input_shape)

        super().build(input_shape)

    def delta_rule_update(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        alpha: keras.KerasTensor,
        beta: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply gated delta rule update using `keras.ops.while_loop`."""
        batch_size, seq_len, _, _ = ops.shape(q)

        i = ops.convert_to_tensor(0, dtype="int32")
        initial_state = ops.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim), dtype=q.dtype
        )
        outputs_transposed = ops.zeros(
            (seq_len, batch_size, self.num_heads, self.head_dim), dtype=q.dtype
        )

        def condition(i, state, outputs):
            return ops.less(i, seq_len)

        def body(i, state, outputs):
            q_t, k_t, v_t = q[:, i], k[:, i], v[:, i]
            alpha_t, beta_t = alpha[:, i], beta[:, i]
            v_t_1, v_t_2 = ops.split(v_t, 2, axis=-1)

            k_exp = ops.expand_dims(k_t, -1)
            v_exp = ops.expand_dims(v_t_1, -2)
            delta = ops.matmul(k_exp, v_exp)

            beta_exp = ops.expand_dims(ops.expand_dims(beta_t, -1), -1)
            alpha_exp = ops.expand_dims(ops.expand_dims(alpha_t, -1), -1)
            next_state = alpha_exp * state + beta_exp * delta

            q_exp = ops.expand_dims(q_t, -2)
            output_t = ops.squeeze(ops.matmul(q_exp, next_state), axis=-2) + v_t_2

            next_outputs = ops.scatter_update(
                outputs, ops.expand_dims([i], -1), ops.expand_dims(output_t, 0)
            )
            return i + 1, next_state, next_outputs

        _, _, final_outputs = ops.while_loop(
            cond=condition,
            body=body,
            loop_vars=(i, initial_state, outputs_transposed),
            maximum_iterations=self.max_seq_len,
        )
        return ops.transpose(final_outputs, [1, 0, 2, 3])

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the Gated DeltaNet layer."""
        batch_size, seq_len, _ = ops.shape(inputs)

        q = self.q_proj(inputs, training=training)
        k = self.k_proj(inputs, training=training)
        v = self.v_proj(inputs, training=training)

        q_norm = self.q_norm(q, training=training)
        k_norm = self.k_norm(k, training=training)
        v_norm = self.v_norm(v, training=training)

        q_conv = self.activation_layer(self.q_conv(q_norm, training=training))
        k_conv = self.activation_layer(self.k_conv(k_norm, training=training))
        v_conv = self.activation_layer(self.v_conv(v_norm, training=training))

        q_heads = ops.reshape(q_conv, (batch_size, seq_len, self.num_heads, self.head_dim))
        k_heads = ops.reshape(k_conv, (batch_size, seq_len, self.num_heads, self.head_dim))
        v_heads = ops.reshape(v_conv, (batch_size, seq_len, self.num_heads, 2 * self.head_dim))

        alpha = ops.sigmoid(self.alpha_proj(inputs, training=training))
        beta = ops.sigmoid(self.beta_proj(inputs, training=training))

        if training and self.dropout is not None:
            q_heads = self.dropout(q_heads, training=training)
            k_heads = self.dropout(k_heads, training=training)
            v_heads = self.dropout(v_heads, training=training)

        delta_output = self.delta_rule_update(q_heads, k_heads, v_heads, alpha, beta)
        delta_output = ops.reshape(delta_output, (batch_size, seq_len, self.qk_dim))

        if self.use_default_ffn:
            projected_output = self.output_proj(delta_output, training=training)
            gate = ops.sigmoid(self.output_gate_linear(projected_output, training=training))
            gated_output = gate * projected_output
        else:
            gated_output = self.output_ffn(delta_output, training=training)

        return gated_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape given input shape."""
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "max_seq_len": self.max_seq_len,
                "head_dim": self.head_dim,
                "conv_kernel_size": self.conv_kernel_size,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "normalization_type": self.normalization_type,
                "q_norm_args": self.q_norm_args,
                "k_norm_args": self.k_norm_args,
                "v_norm_args": self.v_norm_args,
                "ffn_type": self.ffn_type,
                "ffn_args": self.ffn_args,
                "intermediate_size": self.intermediate_size,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            }
        )
        return config

# ---------------------------------------------------------------------
