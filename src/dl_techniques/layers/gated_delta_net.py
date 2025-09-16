import keras
from typing import Any, Dict, Optional, Tuple, Union
from keras import initializers, layers, ops, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GatedDeltaNet(keras.layers.Layer):
    """
    Gated DeltaNet layer combining delta rule updates with adaptive gating mechanism.

    This layer implements a sophisticated linear transformer variant that combines:
    - Delta rule mechanism for targeted memory updates
    - Adaptive gating for rapid memory erasure and control
    - Zero-Centered RMS normalization for training stability
    - Short convolution for position-based addressing
    - Output gating with sigmoid activation for selective information flow

    **Intent**: Provide an efficient alternative to standard attention that excels
    at in-context retrieval and long-context understanding while maintaining
    linear complexity. The gating mechanism enables flexible memory control.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, dim])
           ↓
    Q/K/V Linear Projections
           ↓
    Zero-Centered RMSNorm → Short Conv1D (Q, K, V)
           ↓                     ↓
    Alpha/Beta Gating ←----------┘
           ↓
    Delta Rule Update (with gating: α_t * delta_update + (1-α_t) * S_{t-1})
           ↓
    Output Projection → Sigmoid Gate (⊗) → Output
    ```

    **Mathematical Operations**:
    1. **QKV Transform**: Q = Linear_q(x), K = Linear_k(x), V = Linear_v(x)
    2. **Normalization**: Q_norm = RMSNorm(Q), K_norm = RMSNorm(K), V_norm = RMSNorm(V)
    3. **Convolution**: Q_conv = Conv1D(Q_norm), K_conv = Conv1D(K_norm), V_conv = Conv1D(V_norm)
    4. **Gating Parameters**: α = sigmoid(Linear_α(x)), β = sigmoid(Linear_β(x))
    5. **Delta Rule**: S_t = α_t * (S_{t-1} + β_t * V_t * K_t^T) + (1-α_t) * S_{t-1}
    6. **Output**: output = Q_t @ S_t, projected = Linear_out(output), gated = sigmoid(Linear_gate(projected)) ⊗ projected

    The delta rule minimizes MSE between desired and predicted output at each timestep,
    making it particularly effective for associative recall and in-context retrieval tasks.

    Args:
        dim: Integer, the model dimension size. Must be positive and divisible by num_heads.
            This determines the input/output feature size and state dimension.
        num_heads: Integer, number of attention heads for multi-head processing.
            Must be positive. Each head operates independently on dim//num_heads dimensions.
        head_dim: Optional integer, dimension per head. If None, defaults to dim // num_heads.
            Allows for custom head dimensionality independent of input dimension.
        conv_kernel_size: Integer, kernel size for short convolution layers.
            Typically 4 for position-based addressing. Must be positive. Defaults to 4.
        dropout_rate: Float between 0 and 1, dropout rate applied to intermediate representations.
            Used for regularization during training. Defaults to 0.0.
        use_bias: Boolean, whether to use bias terms in linear layers.
            Modern architectures often omit bias for efficiency. Defaults to False.
        kernel_initializer: String or initializer, initialization for linear layer weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initialization for bias weights (if used).
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for linear layer weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.
        Same shape as input, preserving sequence structure.

    Example:
        ```python
        # Basic configuration
        layer = GatedDeltaNet(dim=768, num_heads=12)

        # Advanced configuration with custom parameters
        layer = GatedDeltaNet(
            dim=768,
            num_heads=12,
            head_dim=128,
            conv_kernel_size=4,
            dropout_rate=0.1,
            use_bias=False,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Usage in model
        inputs = keras.Input(shape=(seq_len, 768))
        outputs = layer(inputs)
        ```
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        conv_kernel_size: int = 4,
        dropout_rate: float = 0.0,
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
            dim, num_heads, head_dim, conv_kernel_size, dropout_rate
        )

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Compute dimensions
        self.qk_dim = self.num_heads * self.head_dim
        self.v_dim = self.num_heads * self.head_dim * 2

        # Q/K/V projections
        self.q_proj = layers.Dense(
            self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="q_proj",
        )
        self.k_proj = layers.Dense(
            self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="k_proj",
        )
        self.v_proj = layers.Dense(
            self.v_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="v_proj",
        )

        # Zero-Centered RMS Normalization layers
        self.q_norm = create_normalization_layer(
            "zero_centered_rms_norm", epsilon=1e-5, use_scale=True, name="q_norm"
        )
        self.k_norm = create_normalization_layer(
            "zero_centered_rms_norm", epsilon=1e-5, use_scale=True, name="k_norm"
        )
        self.v_norm = create_normalization_layer(
            "zero_centered_rms_norm", epsilon=1e-5, use_scale=True, name="v_norm"
        )

        # Short convolution layers (depthwise separable)
        self.q_conv = layers.Conv1D(
            filters=self.qk_dim,
            kernel_size=conv_kernel_size,
            padding="causal",
            groups=self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="q_conv",
        )
        self.k_conv = layers.Conv1D(
            filters=self.qk_dim,
            kernel_size=conv_kernel_size,
            padding="causal",
            groups=self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="k_conv",
        )
        self.v_conv = layers.Conv1D(
            filters=self.v_dim,
            kernel_size=conv_kernel_size,
            padding="causal",
            groups=self.v_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="v_conv",
        )

        # Gating parameter projections (alpha and beta)
        self.alpha_proj = layers.Dense(
            self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="alpha_proj",
        )
        self.beta_proj = layers.Dense(
            self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="beta_proj",
        )

        # Output projection layer
        self.output_proj = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_proj",
        )

        # SiLU activation for intermediate processing
        self.silu = layers.Activation("silu", name="silu")

        # Dropout for regularization
        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(dropout_rate, name="dropout")
        else:
            self.dropout = None

        # Output gate
        self.output_gate_linear = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_gate_linear",
        )

        logger.info(
            f"GatedDeltaNet initialized: dim={dim}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}, "
            f"qk_dim={self.qk_dim}, v_dim={self.v_dim}"
        )

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int],
        conv_kernel_size: int,
        dropout_rate: float,
    ) -> None:
        """Validate layer initialization parameters."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
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
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")
        batch_size, seq_len, features = input_shape
        if features != self.dim:
            raise ValueError(
                f"Input feature dimension ({features}) must match dim ({self.dim})"
            )

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
        self.output_proj.build((batch_size, seq_len, self.qk_dim))
        self.output_gate_linear.build((batch_size, seq_len, self.dim))
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
        """
        Apply gated delta rule update using `keras.ops.while_loop`.

        This implements the core gated delta rule mechanism sequentially, making it
        compatible with dynamic sequence lengths.
        S_t = α_t * S_{t-1} + (β_t * K_t * V_t^T)

        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim).
            k: Key tensor of shape (batch_size, seq_len, num_heads, head_dim).
            v: Value tensor of shape (batch_size, seq_len, num_heads, 2*head_dim).
            alpha: Gating parameter α_t of shape (batch_size, seq_len, num_heads).
            beta: Update strength β_t of shape (batch_size, seq_len, num_heads).
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, seq_len, num_heads, head_dim).
        """
        batch_size = ops.shape(q)[0]
        seq_len = ops.shape(q)[1]

        # Initial values for the loop variables
        i = ops.convert_to_tensor(0, dtype="int32")
        initial_state = ops.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim),
            dtype=q.dtype,
        )
        # Pre-allocate an output tensor in transposed layout for efficient updates
        outputs_transposed = ops.zeros(
            (seq_len, batch_size, self.num_heads, self.head_dim),
            dtype=q.dtype,
        )

        def condition(i, current_state, outputs_transposed):
            # Loop until the counter `i` reaches the sequence length
            return ops.less(i, seq_len)

        def body(i, current_state, outputs_transposed):
            # Slice inputs for the current timestep `i`
            q_t = q[:, i]  # Shape: (B, H, D)
            k_t = k[:, i]  # Shape: (B, H, D)
            v_t = v[:, i]  # Shape: (B, H, 2*D)
            alpha_t = alpha[:, i]  # Shape: (B, H)
            beta_t = beta[:, i]  # Shape: (B, H)

            # Using ops.split is robust for shape inference.
            v_t_1, v_t_2 = ops.split(v_t, indices_or_sections=2, axis=-1)

            # --- State Update Logic ---
            # Shapes: (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
            k_t_expanded = ops.expand_dims(k_t, -1)
            v_t_1_expanded = ops.expand_dims(v_t_1, -2)
            delta_update = ops.matmul(k_t_expanded, v_t_1_expanded)

            # Apply gating parameters, expanding them to broadcast correctly
            beta_t_expanded = ops.expand_dims(ops.expand_dims(beta_t, -1), -1)
            scaled_delta = beta_t_expanded * delta_update
            alpha_t_expanded = ops.expand_dims(
                ops.expand_dims(alpha_t, -1), -1
            )

            # S_t = alpha_t * S_{t-1} + (beta_t * K_t * V_t^T)
            next_state = alpha_t_expanded * current_state + scaled_delta

            # --- Output Calculation ---
            # Shapes: (B, H, 1, D) @ (B, H, D, D) -> (B, H, 1, D)
            q_t_expanded = ops.expand_dims(q_t, -2)
            output_t = ops.matmul(q_t_expanded, next_state)
            output_t = ops.squeeze(output_t, axis=-2)  # -> (B, H, D)
            output_t = output_t + v_t_2  # Add residual part of V

            # --- Accumulate Output ---
            # Update the transposed outputs tensor at index `i`.
            # `scatter_update` works on the first axis.
            indices = ops.expand_dims([i], axis=-1)  # Shape: (1, 1)
            updates = ops.expand_dims(output_t, 0)  # Shape: (1, B, H, D)

            next_outputs_transposed = ops.scatter_update(
                outputs_transposed, indices, updates
            )

            # Return the updated loop variables
            return (i + 1, next_state, next_outputs_transposed)

        # Execute the while loop
        _, _, final_outputs_transposed = ops.while_loop(
            condition,
            body,
            (i, initial_state, outputs_transposed),
        )

        # Transpose outputs back to (batch_size, seq_len, ...)
        outputs = ops.transpose(final_outputs_transposed, [1, 0, 2, 3])
        return outputs

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Gated DeltaNet layer.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, dim).
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape (batch_size, sequence_length, dim).
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Linear projections for Q, K, V
        q = self.q_proj(inputs, training=training)
        k = self.k_proj(inputs, training=training)
        v = self.v_proj(inputs, training=training)

        # Zero-centered RMS normalization
        q_norm = self.q_norm(q, training=training)
        k_norm = self.k_norm(k, training=training)
        v_norm = self.v_norm(v, training=training)

        # Short convolution for position encoding
        q_conv = self.silu(self.q_conv(q_norm, training=training))
        k_conv = self.silu(self.k_conv(k_norm, training=training))
        v_conv = self.silu(self.v_conv(v_norm, training=training))

        # Reshape to multi-head format
        q_heads = ops.reshape(
            q_conv, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        k_heads = ops.reshape(
            k_conv, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        v_heads = ops.reshape(
            v_conv, (batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        )

        # Compute gating parameters
        alpha = ops.sigmoid(self.alpha_proj(inputs, training=training))
        beta = ops.sigmoid(self.beta_proj(inputs, training=training))

        # Apply dropout if enabled
        if training and self.dropout is not None:
            q_heads = self.dropout(q_heads, training=training)
            k_heads = self.dropout(k_heads, training=training)
            v_heads = self.dropout(v_heads, training=training)

        # Apply gated delta rule update
        delta_output = self.delta_rule_update(
            q_heads, k_heads, v_heads, alpha, beta, training=training
        )

        # Reshape and project output
        delta_output = ops.reshape(
            delta_output, (batch_size, seq_len, self.qk_dim)
        )
        delta_output = self.output_proj(delta_output, training=training)

        # Apply output gating
        gate = ops.sigmoid(
            self.output_gate_linear(delta_output, training=training)
        )
        gated_output = gate * delta_output
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
                "head_dim": self.head_dim,
                "conv_kernel_size": self.conv_kernel_size,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config

# ---------------------------------------------------------------------