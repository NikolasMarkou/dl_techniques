"""
Differential Multi-Head Attention Implementation
===============================================

This module implements Differential Multi-Head Attention as described in the paper:
"DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context while canceling noise"

The Differential Attention mechanism employs two separate Multi-Head Attention (MHA)
layers and computes a weighted difference between them: Attention_diff = MHA1 - λ*MHA2.
This design amplifies relevant context signals while attenuating noise, resulting
in more focused attention patterns.

Key Concepts:
-----------
1. Dual Attention Process:
   - Two separate multi-head attention mechanisms process the same input
   - The first attention (MHA1) captures primary attention patterns
   - The second attention (MHA2) identifies noise and irrelevant patterns
   - The weighted difference (MHA1 - λ*MHA2) amplifies signal and reduces noise

2. Adaptive Lambda Parameter (λ):
   - Learnable parameter controlling balance between the two attention mechanisms
   - Initialized based on layer depth: 0.8 - 0.6*exp(-0.3*(layer_idx - 1))
   - Adapts during training to optimize noise cancellation effect
   - Bounded to maintain training stability [0.1, 0.9]

3. Performance Benefits:
   - Improved focus on relevant information in input sequences
   - Enhanced accuracy in long-context understanding
   - Reduction in activation outliers, enabling better quantization
   - Mitigation of hallucination in generation tasks
   - Superior performance with fewer parameters

Usage Example:
-------------
```python
# Create differential attention layer
diff_attention = DifferentialMultiHeadAttention(
    dim=512,
    num_heads=8,
    head_dim=64,
    dropout=0.1,
    attention_dropout=0.1,
    lambda_init=0.8
)

# Use in transformer block with layer index for optimal lambda computation
x_norm = keras.layers.LayerNormalization()(inputs)
attn_output = diff_attention(x_norm, layer_idx=2, training=training)
x = inputs + attn_output  # Residual connection
```
"""

import keras
from typing import Any, Dict, Optional, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DifferentialMultiHeadAttention(keras.layers.Layer):
    """
    Differential multi-head attention mechanism.

    This layer implements differential attention that uses two separate MultiHeadAttention
    layers and computes their weighted difference to amplify relevant context while
    canceling noise. The key innovation is the learnable λ parameter that balances
    the contribution of the two attention mechanisms.

    The differential attention is computed as:
        Attention_diff = MHA1(x) - λ * MHA2(x)

    Where MHA1 captures primary patterns, MHA2 identifies noise, and λ controls
    the noise cancellation strength.

    Args:
        dim: Integer, input and output dimension. Must be positive and should be
            divisible by num_heads for optimal performance.
        num_heads: Integer, number of attention heads for both attention mechanisms.
            Must be positive and should divide dim evenly.
        head_dim: Integer, dimension of each attention head. Must be positive.
            Typically computed as dim // num_heads.
        dropout: Float, output dropout rate applied after projection.
            Must be between 0 and 1. Defaults to 0.0.
        attention_dropout: Float, dropout rate applied to attention weights in
            both MHA layers. Must be between 0 and 1. Defaults to 0.0.
        lambda_init: Float, initial value for the λ parameter controlling the
            balance between attention mechanisms. Should be between 0 and 1.
            Defaults to 0.8.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional Regularizer, regularizer applied to kernel weights.
        bias_initializer: String or Initializer, initializer for bias weights.
            Defaults to 'zeros'.
        bias_regularizer: Optional Regularizer, regularizer applied to bias weights.
        activity_regularizer: Optional Regularizer, regularizer applied to layer output.
        **kwargs: Additional keyword arguments passed to Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`

    Attributes:
        attention1: First MultiHeadAttention layer capturing primary patterns.
        attention2: Second MultiHeadAttention layer identifying noise patterns.
        proj: Dense layer for final output projection.
        dropout_layer: Dropout layer applied to final output.
        lambda_param: Learnable parameter controlling attention balance.

    Example:
        ```python
        # Basic usage
        diff_attn = DifferentialMultiHeadAttention(dim=512, num_heads=8, head_dim=64)

        # Advanced configuration with regularization
        diff_attn = DifferentialMultiHeadAttention(
            dim=768,
            num_heads=12,
            head_dim=64,
            dropout=0.1,
            attention_dropout=0.05,
            lambda_init=0.7,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Use in transformer block
        inputs = keras.Input(shape=(seq_len, 768))
        x = keras.layers.LayerNormalization()(inputs)

        # Pass layer_idx for optimal lambda computation
        attn_out = diff_attn(x, layer_idx=3, training=True)
        outputs = inputs + attn_out  # Residual connection

        model = keras.Model(inputs, outputs)
        ```

    Raises:
        ValueError: If dim is not positive.
        ValueError: If num_heads is not positive.
        ValueError: If head_dim is not positive.
        ValueError: If dropout rates are not between 0 and 1.
        ValueError: If lambda_init is not between 0 and 1.

    References:
        DIFFERENTIAL TRANSFORMER: Amplifying attention to the relevant context
        while canceling noise. Paper describes the mathematical foundation and
        empirical benefits of this attention mechanism.

    Note:
        For optimal performance, use this layer within pre-normalization transformer
        architectures where LayerNormalization is applied before attention blocks.
        The layer_idx parameter should be provided during forward pass for
        layer-dependent lambda computation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        lambda_init: float = 0.8,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the differential multi-head attention layer."""
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if not (0.0 <= attention_dropout <= 1.0):
            raise ValueError(f"attention_dropout must be between 0 and 1, got {attention_dropout}")
        if not (0.0 <= lambda_init <= 1.0):
            raise ValueError(f"lambda_init must be between 0 and 1, got {lambda_init}")

        # Store configuration - ALL __init__ parameters must be stored
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout
        self.attention_dropout_rate = attention_dropout
        self.lambda_init = lambda_init

        # Store serialized initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        try:
            # First multi-head attention layer (captures primary patterns)
            self.attention1 = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.head_dim,
                dropout=self.attention_dropout_rate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                name='attention1'
            )

            # Second multi-head attention layer (identifies noise patterns)
            self.attention2 = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.head_dim,
                dropout=self.attention_dropout_rate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                name='attention2'
            )

            # Output projection layer
            self.proj = keras.layers.Dense(
                self.dim,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='proj'
            )

            # Output dropout layer
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name='dropout')

        except Exception as e:
            logger.error(f"Failed to create DifferentialMultiHeadAttention sub-layers: {e}")
            raise ValueError(
                f"Failed to create DifferentialMultiHeadAttention sub-layers. "
                f"This might be due to invalid configuration parameters. "
                f"Original error: {e}"
            )

        # Weight attributes - created in build()
        self.lambda_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create the lambda parameter weight.

        Creates the learnable lambda parameter and explicitly builds all sub-layers
        for robust serialization following modern Keras 3 patterns.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch_size, seq_len, dim), got shape: {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim != self.dim:
            raise ValueError(
                f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
            )

        # Create the layer's own weights - lambda parameter
        # Initialize lambda parameter directly with the init value
        self.lambda_param = self.add_weight(
            name="lambda_param",
            shape=(1,),
            initializer=keras.initializers.Constant(self.lambda_init),
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # CRITICAL: Explicitly build sub-layers for robust serialization
        # MultiHeadAttention.build() requires query_shape and value_shape
        # For self-attention, both are the same as input_shape
        self.attention1.build(query_shape=input_shape, value_shape=input_shape)
        self.attention2.build(query_shape=input_shape, value_shape=input_shape)
        self.proj.build(input_shape)
        self.dropout_layer.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def get_lambda(self, layer_idx: int = 0) -> keras.KerasTensor:
        """
        Compute the lambda value with layer-dependent adaptation.

        The lambda parameter is adapted based on layer depth following the paper's
        initialization strategy: λ = 0.8 - 0.6*exp(-0.3*(layer_idx - 1)).
        The learned lambda_param is then applied as a multiplicative factor.

        Args:
            layer_idx: Integer, index of the layer in the network stack (0-based).
                Used to compute layer-dependent lambda initialization.

        Returns:
            Tensor containing the computed lambda value, bounded between 0.1 and 0.9.
        """
        # Layer-dependent initialization following the paper
        # λ_init = 0.8 - 0.6*exp(-0.3*(layer_idx - 1))
        layer_factor = keras.ops.cast(layer_idx, dtype="float32")
        exp_term = keras.ops.exp(-0.3 * keras.ops.maximum(layer_factor - 1.0, 0.0))
        layer_dependent_init = 0.8 - 0.6 * exp_term

        # Apply learned lambda parameter as multiplicative factor
        # Clip to ensure training stability
        lambda_val = keras.ops.clip(
            layer_dependent_init * self.lambda_param[0],
            0.1,
            0.9
        )

        return lambda_val

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        layer_idx: int = 0,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply differential attention mechanism.

        Computes the differential attention as: Attention_diff = MHA1(x) - λ*MHA2(x)
        where MHA1 captures primary attention patterns, MHA2 identifies noise patterns,
        and λ controls the balance between them.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, dim).
            mask: Optional attention mask tensor. Can be 2D, 3D, or 4D tensor
                for different masking strategies.
            layer_idx: Integer, index of the layer in the network stack (0-based).
                Used for layer-dependent lambda computation. Defaults to 0.
            training: Optional boolean indicating whether in training mode.

        Returns:
            Output tensor of shape (batch_size, sequence_length, dim) after
            applying differential attention and output projection.
        """
        # Compute attention outputs from both attention mechanisms
        # Both use self-attention (query=key=value=inputs)
        attention_output1 = self.attention1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask,
            training=training,
            return_attention_scores=False
        )

        attention_output2 = self.attention2(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask,
            training=training,
            return_attention_scores=False
        )

        # Compute layer-dependent lambda value
        lambda_val = self.get_lambda(layer_idx)

        # Apply differential attention: MHA1 - λ*MHA2
        # This is the core innovation - subtracting weighted noise attention
        diff_attention = attention_output1 - lambda_val * attention_output2

        # Apply output projection and dropout
        output = self.proj(diff_attention, training=training)
        output = self.dropout_layer(output, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. Same as input shape for attention layers.
        """
        # Output shape is identical to input shape
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns all parameters needed to reconstruct the layer during loading.
        This must include ALL parameters passed to __init__.

        Returns:
            Dictionary containing the complete layer configuration.
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout': self.dropout_rate,
            'attention_dropout': self.attention_dropout_rate,
            'lambda_init': self.lambda_init,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
