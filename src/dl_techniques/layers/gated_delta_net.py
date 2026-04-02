"""
A Gated Delta Network, a linear-time transformer variant.

This layer provides a recurrent mechanism for sequence modeling that combines
associative memory via the delta rule with adaptive gating for linear O(N)
complexity. The gated delta rule update is:
S_t = alpha_t * S_{t-1} + beta_t * (K_t outer V_t), where alpha controls
memory persistence and beta controls update strength. This enables precise
key-value association retrieval while allowing flexible memory management.

References:
    - Schlag, I., et al. (2021). Linear Transformers Are Secretly Fast Weight
      Programmers (DeltaNet).
    - Yang, S., et al. (2024). Gated Delta Networks: Improving Mamba2 with
      Delta Rule.
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
    Gated DeltaNet layer combining delta rule updates with adaptive gating.

    Implements a linear transformer variant that combines delta rule mechanism
    for targeted memory updates (S_t = alpha_t * S_{t-1} + beta_t * K_t outer V_t)
    with adaptive gating (alpha for persistence, beta for update strength).
    Features configurable normalization for Q/K/V, short convolution for
    position-based addressing, and a configurable output FFN. Due to TensorFlow
    framework limitations, requires a hard maximum sequence length parameter.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────────┐
        │  Input [batch, seq_len, dim]                  │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Q/K/V Linear Projections                     │
        └──────┬───────────┬───────────┬────────────────┘
               ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Q Norm   │ │ K Norm   │ │ V Norm   │
        │ Q Conv1D │ │ K Conv1D │ │ V Conv1D │
        │ Activate │ │ Activate │ │ Activate │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             ▼             ▼            ▼
        ┌───────────────────────────────────────────────┐
        │  Alpha/Beta Gating (sigmoid projections)      │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Delta Rule Update (recurrent, per timestep)  │
        │  S_t = alpha_t * S_{t-1} + beta_t * K_t⊗V_t   │
        │  out_t = Q_t @ S_t + V_t_residual             │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Reshape & Output FFN (default: GLU gate)     │
        └──────────────────┬────────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────────┐
        │  Output [batch, seq_len, dim]                 │
        └───────────────────────────────────────────────┘

    :param dim: Model dimension size. Must be positive.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param max_seq_len: Maximum sequence length for the while_loop.
    :type max_seq_len: int
    :param head_dim: Dimension per head. If None, defaults to dim // num_heads.
    :type head_dim: Optional[int]
    :param conv_kernel_size: Kernel size for short convolution layers. Defaults
        to 4.
    :type conv_kernel_size: int
    :param dropout_rate: Dropout rate for regularization. Defaults to 0.0.
    :type dropout_rate: float
    :param activation: Activation function after convolutions. Defaults to 'silu'.
    :type activation: Union[str, Callable]
    :param normalization_type: Type of normalization for Q, K, V. Defaults to
        'zero_centered_rms_norm'.
    :type normalization_type: NormalizationType
    :param q_norm_args: Optional arguments for Q normalization layer.
    :type q_norm_args: Optional[Dict[str, Any]]
    :param k_norm_args: Optional arguments for K normalization layer.
    :type k_norm_args: Optional[Dict[str, Any]]
    :param v_norm_args: Optional arguments for V normalization layer.
    :type v_norm_args: Optional[Dict[str, Any]]
    :param ffn_type: Type of FFN for output stage. If None, uses original gated
        linear unit output.
    :type ffn_type: Optional[FFNType]
    :param ffn_args: Optional arguments for the custom FFN layer.
    :type ffn_args: Optional[Dict[str, Any]]
    :param intermediate_size: Intermediate size for standard FFNs. Defaults to
        dim * 4 if not provided.
    :type intermediate_size: Optional[int]
    :param use_bias: Whether to use bias in linear layers. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class.
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
        """Validate layer initialization parameters.

        :param dim: Model dimension.
        :type dim: int
        :param num_heads: Number of heads.
        :type num_heads: int
        :param head_dim: Per-head dimension.
        :type head_dim: Optional[int]
        :param conv_kernel_size: Convolution kernel size.
        :type conv_kernel_size: int
        :param dropout_rate: Dropout rate.
        :type dropout_rate: float
        :param max_seq_len: Maximum sequence length.
        :type max_seq_len: int
        """
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
        """Create a normalization layer from the factory.

        :param name: Layer name.
        :type name: str
        :param custom_args: Custom arguments for the normalization layer.
        :type custom_args: Dict[str, Any]
        :return: Normalization layer instance.
        :rtype: keras.layers.Layer
        """
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
        """Create an FFN layer from the factory for the output stage.

        :param name: Layer name.
        :type name: str
        :return: FFN layer instance.
        :rtype: keras.layers.Layer
        """
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
        """Build the layer and all sub-layers.

        :param input_shape: Shape tuple (batch_size, sequence_length, dim).
        :type input_shape: Tuple[Optional[int], ...]
        """
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
        """Apply gated delta rule update using ``keras.ops.while_loop``.

        :param q: Query tensor of shape (batch, seq, heads, head_dim).
        :type q: keras.KerasTensor
        :param k: Key tensor of shape (batch, seq, heads, head_dim).
        :type k: keras.KerasTensor
        :param v: Value tensor of shape (batch, seq, heads, 2*head_dim).
        :type v: keras.KerasTensor
        :param alpha: Persistence gate of shape (batch, seq, heads).
        :type alpha: keras.KerasTensor
        :param beta: Update gate of shape (batch, seq, heads).
        :type beta: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch, seq, heads, head_dim).
        :rtype: keras.KerasTensor
        """
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
        """Forward pass through the Gated DeltaNet layer.

        :param inputs: Input tensor of shape (batch, seq_len, dim).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape (batch, seq_len, dim).
        :rtype: keras.KerasTensor
        """
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
        """Compute the output shape given input shape.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
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
