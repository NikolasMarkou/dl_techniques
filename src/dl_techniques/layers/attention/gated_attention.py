"""
Gated Attention Layer Implementation

This module implements a Gated Attention layer as shown in the LLM architecture diagram.
The layer features input linear projection, Q/K/V projections with Zero-Centered RMSNorm,
Partial RoPE on queries and keys, scaled dot-product attention, and output gating.
"""

import keras
from keras import layers, initializers, regularizers, ops
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GatedAttention(keras.layers.Layer):
    """
    Gated Attention layer with normalization, partial RoPE, and output gating.

    This layer implements a sophisticated attention mechanism that includes:
    - Input linear projection for feature transformation
    - Separate Q/K/V projections with Zero-Centered RMS normalization
    - Partial Rotary Position Embedding (RoPE) applied to queries and keys
    - Scaled dot-product attention mechanism
    - Output gating with sigmoid activation for selective information flow

    **Intent**: Provide a high-performance attention layer that combines modern
    techniques like RoPE and advanced normalization with gating mechanisms for
    improved training stability and model expressiveness.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, dim])
           ↓
    Linear Projection
           ↓
    Split → Q Linear → Zero-Centered RMSNorm → Partial RoPE ↘
         → K Linear → Zero-Centered RMSNorm → Partial RoPE → Attention
         → V Linear → Zero-Centered RMSNorm ---------------→ ↗
           ↓
    Attention Output → (Optional Projection) → Linear → Sigmoid → Gate (⊗) → Output
    ```

    **Mathematical Operations**:
    1. **Input Transform**: x' = Linear(x)
    2. **QKV Projections**: Q = Linear_q(x'), K = Linear_k(x'), V = Linear_v(x')
    3. **Normalization**: Q_norm = RMSNorm(Q), K_norm = RMSNorm(K), V_norm = RMSNorm(V)
    4. **RoPE**: Q_rope = RoPE(Q_norm), K_rope = RoPE(K_norm)
    5. **Attention**: A = Attention(Q_rope, K_rope, V_norm)
    6. **Projection**: A' = Linear_proj(A) if attention_dim != dim
    7. **Gating**: gate = σ(Linear_gate(A')), output = gate ⊗ A'

    Args:
        dim: Integer, the model dimension size. Must be positive and divisible by num_heads
            if head_dim is not specified. This determines the input/output feature size.
        num_heads: Integer, number of attention heads. Must be positive.
            More heads allow the model to attend to different representation subspaces.
        head_dim: Optional integer, dimension per attention head. If None, defaults to
            dim // num_heads. Allows for custom head dimensionality.
        max_seq_len: Integer, maximum sequence length for RoPE precomputation.
            Should be set to the maximum expected sequence length. Defaults to 4096.
        rope_percentage: Float between 0 and 1, fraction of head dimensions to apply RoPE to.
            Partial RoPE applies positional encoding only to a subset of dimensions.
            Defaults to 0.5 (50% of dimensions).
        dropout_rate: Float between 0 and 1, dropout rate for attention weights.
            Applied during training for regularization. Defaults to 0.0.
        use_bias: Boolean, whether to use bias terms in linear layers.
            Modern transformers often omit bias for efficiency. Defaults to False.
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

    Attributes:
        input_linear: Dense layer for input feature transformation.
        q_linear, k_linear, v_linear: Dense layers for Q/K/V projections.
        q_norm, k_norm, v_norm: Zero-Centered RMS normalization layers.
        rope: Rotary Position Embedding layer for positional encoding.
        output_proj: Optional Dense layer for dimension matching when attention_dim != dim.
        dropout: Dropout layer for attention weight regularization.
        output_gate_linear: Dense layer for computing gating weights.

    Example:
        ```python
        # Standard configuration
        attention = GatedAttention(
            dim=768,
            num_heads=12,
            max_seq_len=2048,
            dropout_rate=0.1
        )

        # Process a sequence
        inputs = keras.Input(shape=(128, 768))
        outputs = attention(inputs)  # Shape: (batch, 128, 768)

        # Custom head dimension
        attention = GatedAttention(
            dim=768,
            num_heads=8,
            head_dim=96,  # Custom head size
            rope_percentage=0.25  # Apply RoPE to 25% of dimensions
        )

        # With regularization
        attention = GatedAttention(
            dim=512,
            num_heads=8,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            dropout_rate=0.15
        )
        ```

    Raises:
        ValueError: If dim is not positive or not divisible by num_heads (when head_dim is None).
        ValueError: If num_heads is not positive.
        ValueError: If head_dim is not positive (when specified).
        ValueError: If rope_percentage is not in (0, 1].
        ValueError: If dropout_rate is not in [0, 1].
        ValueError: If max_seq_len is not positive.

    Note:
        This implementation uses Zero-Centered RMSNorm for improved training stability
        and Partial RoPE for efficient positional encoding. The gating mechanism allows
        the model to selectively control information flow through the attention output.
        When attention_dim != dim (custom head_dim case), an additional projection layer
        ensures dimensional compatibility.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            max_seq_len: int = 4096,
            rope_percentage: float = 0.5,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(dim, num_heads, head_dim, max_seq_len,
                            rope_percentage, dropout_rate)

        # Store ALL configuration parameters for serialization
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.max_seq_len = max_seq_len
        self.rope_percentage = rope_percentage
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # TRICKY POINT: When using custom head_dim, attention_dim may not equal dim
        # This requires an additional projection layer to match dimensions
        self.attention_dim = self.num_heads * self.head_dim

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Following the Golden Rule: Create in __init__, Build in build()

        # Input linear projection
        self.input_linear = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="input_linear"
        )

        # QKV projections - project to attention_dim, not dim
        self.q_linear = layers.Dense(
            self.attention_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="q_linear"
        )
        self.k_linear = layers.Dense(
            self.attention_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="k_linear"
        )
        self.v_linear = layers.Dense(
            self.attention_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="v_linear"
        )

        # Zero-Centered RMS Normalization layers
        self.q_norm = create_normalization_layer(
            'zero_centered_rms_norm',
            epsilon=1e-6,
            use_scale=True,
            name='q_norm'
        )
        self.k_norm = create_normalization_layer(
            'zero_centered_rms_norm',
            epsilon=1e-6,
            use_scale=True,
            name='k_norm'
        )
        self.v_norm = create_normalization_layer(
            'zero_centered_rms_norm',
            epsilon=1e-6,
            use_scale=True,
            name='v_norm'
        )

        # Rotary Position Embedding (Partial RoPE)
        # TRICKY POINT: RoPE doesn't have trainable parameters, just precomputed sin/cos
        self.rope = create_embedding_layer(
            'rope',
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_percentage=rope_percentage,
            name='rope'
        )

        # Dropout for attention weights (conditional creation)
        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(dropout_rate, name="attention_dropout")
        else:
            self.dropout = None

        # TRICKY POINT: Output projection needed when attention_dim != dim
        # This happens when using custom head_dim that doesn't divide evenly into dim
        if self.attention_dim != self.dim:
            self.output_proj = layers.Dense(
                self.dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="output_proj"
            )
        else:
            self.output_proj = None

        # Output gate - always projects to dim
        self.output_gate_linear = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_gate_linear"
        )

        logger.info(f"GatedAttention initialized: dim={dim}, "
                   f"num_heads={num_heads}, head_dim={self.head_dim}, "
                   f"attention_dim={self.attention_dim}")

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int],
        max_seq_len: int,
        rope_percentage: float,
        dropout_rate: float
    ) -> None:
        """
        Validate initialization parameters.

        Args:
            dim: Model dimension to validate.
            num_heads: Number of attention heads to validate.
            head_dim: Head dimension to validate (can be None).
            max_seq_len: Maximum sequence length to validate.
            rope_percentage: RoPE percentage to validate (must be > 0.0 and <= 1.0).
            dropout_rate: Dropout rate to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim is None and dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}) when head_dim is None")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: This method explicitly builds each sub-layer for robust serialization
        support, ensuring all weight variables exist before weight restoration.

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

        # Build input linear
        self.input_linear.build(input_shape)

        # Compute intermediate shapes
        linear_output_shape = (batch_size, seq_len, self.dim)
        qkv_shape = (batch_size, seq_len, self.attention_dim)

        # Build QKV projections - MUST be built explicitly
        self.q_linear.build(linear_output_shape)
        self.k_linear.build(linear_output_shape)
        self.v_linear.build(linear_output_shape)

        # Build normalization layers
        self.q_norm.build(qkv_shape)
        self.k_norm.build(qkv_shape)
        self.v_norm.build(qkv_shape)

        # Build RoPE (it expects reshaped input: [batch, seq, heads, head_dim])
        rope_input_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        self.rope.build(rope_input_shape)

        # Build dropout if used
        if self.dropout is not None:
            self.dropout.build((batch_size, self.num_heads, seq_len, seq_len))

        # TRICKY POINT: Build output projection only if it exists
        if self.output_proj is not None:
            self.output_proj.build((batch_size, seq_len, self.attention_dim))

        # Build output gate
        self.output_gate_linear.build((batch_size, seq_len, self.dim))

        # Always call parent build at the end
        super().build(input_shape)

    def scaled_dot_product_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
            k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
            v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
            attention_mask: Optional attention mask of shape [batch, seq_len]
            training: Training mode flag

        Returns:
            Attention output tensor of shape [batch, seq_len, num_heads, head_dim]
        """
        # Transpose to [batch, num_heads, seq_len, head_dim] for attention computation
        q = ops.transpose(q, axes=[0, 2, 1, 3])
        k = ops.transpose(k, axes=[0, 2, 1, 3])
        v = ops.transpose(v, axes=[0, 2, 1, 3])

        # Compute attention scores
        matmul_qk = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))

        # Scale by sqrt(head_dim) for numerical stability
        dk = ops.cast(self.head_dim, keras.backend.floatx())
        scaled_attention_logits = matmul_qk / ops.sqrt(dk)

        # TRICKY POINT: Attention mask broadcasting
        # The mask is [batch, seq_len] but needs to be [batch, num_heads, seq_len, seq_len]
        if attention_mask is not None:
            # Reshape to (batch, 1, 1, seq_len) for broadcasting
            mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
            # Create additive mask (masked positions get -inf)
            additive_mask = (1.0 - ops.cast(mask, scaled_attention_logits.dtype)) * -1e9
            scaled_attention_logits = scaled_attention_logits + additive_mask

        # Softmax over the last axis (key dimension)
        attention_weights = ops.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout during training
        if training and self.dropout is not None:
            attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        output = ops.matmul(attention_weights, v)

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        output = ops.transpose(output, axes=[0, 2, 1, 3])

        return output

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the gated attention layer.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, dim).
            attention_mask: Optional attention mask of shape (batch_size, seq_len).
            training: Boolean indicating training or inference mode.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        # Input linear projection
        x = self.input_linear(inputs, training=training)

        # Get batch and sequence dimensions dynamically
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # Generate Q, K, V projections
        q = self.q_linear(x, training=training)  # [batch, seq, attention_dim]
        k = self.k_linear(x, training=training)  # [batch, seq, attention_dim]
        v = self.v_linear(x, training=training)  # [batch, seq, attention_dim]

        # Apply Zero-Centered RMS Normalization
        q_norm = self.q_norm(q, training=training)
        k_norm = self.k_norm(k, training=training)
        v_norm = self.v_norm(v, training=training)

        # Reshape for multi-head attention and RoPE
        # [batch, seq, attention_dim] -> [batch, seq, num_heads, head_dim]
        q_reshaped = ops.reshape(q_norm, (batch_size, seq_len, self.num_heads, self.head_dim))
        k_reshaped = ops.reshape(k_norm, (batch_size, seq_len, self.num_heads, self.head_dim))
        v_reshaped = ops.reshape(v_norm, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Apply Partial RoPE to Q and K (not V)
        q_rope = self.rope(q_reshaped, training=training)
        k_rope = self.rope(k_reshaped, training=training)

        # Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(
            q_rope, k_rope, v_reshaped, attention_mask=attention_mask, training=training
        )

        # Reshape back to [batch, seq, attention_dim]
        attention_output = ops.reshape(
            attention_output, (batch_size, seq_len, self.attention_dim)
        )

        # TRICKY POINT: Project to original dim if needed
        # This is necessary when attention_dim != dim (custom head_dim case)
        if self.output_proj is not None:
            attention_output = self.output_proj(attention_output, training=training)

        # Output gating mechanism
        gate = ops.sigmoid(self.output_gate_linear(attention_output, training=training))
        gated_output = gate * attention_output

        return gated_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, same as input shape for attention layers.
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        CRITICAL: Returns ALL configuration parameters passed to __init__ for proper
        serialization and deserialization. Missing any parameter will cause
        deserialization to fail.

        Returns:
            Dictionary containing the layer configuration with all parameters
            required to recreate this layer.
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'rope_percentage': self.rope_percentage,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config