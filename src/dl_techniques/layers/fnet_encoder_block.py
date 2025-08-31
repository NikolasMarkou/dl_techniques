"""
FNet: Fourier Transform-based Attention Replacement

This module implements the FNet architecture from "FNet: Mixing Tokens with Fourier Transforms"
(Lee-Thorp et al., 2021), which replaces self-attention with parameter-free Fourier transforms
for efficient token mixing in transformer-style architectures.

The implementation follows modern Keras 3 best practices and provides both the core
Fourier transform layer and a complete encoder block ready for use in larger models.
"""

import keras
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .attention.fnet_fourier_transform import FNetFourierTransform

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FNetEncoderBlock(keras.layers.Layer):
    """
    Complete FNet encoder block with Fourier mixing and feed-forward components.

    This layer implements a complete FNet encoder block as described in the paper,
    combining the FNet Fourier Transform with a standard position-wise feed-forward
    network. The architecture mirrors Transformer encoder blocks but replaces expensive
    self-attention with efficient Fourier-based token mixing.

    **Intent**: Provide a drop-in replacement for Transformer encoder blocks that maintains
    comparable modeling capacity while achieving significant speedup through parameter-free
    token mixing and reduced computational complexity.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, hidden_dim], mask=[batch, seq_len])
           ↓
    FNet Fourier Transform (preserves mask)
           ↓
    Residual Connection + Layer Normalization (preserves mask)
           ↓
    Position-wise Feed-Forward Network (preserves mask)
           ↓
    Residual Connection + Layer Normalization (preserves mask)
           ↓
    Output(shape=[batch, seq_len, hidden_dim], mask=[batch, seq_len])
    ```

    **Design Principles**:
    - **Masking Support**: Correctly propagates masks through all sub-layers.
    - **Efficiency**: Fourier mixing is faster than self-attention, especially for long sequences.
    - **Simplicity**: No learnable mixing parameters reduces overfitting risk.
    - **Compatibility**: Same input/output interface as standard Transformer blocks.

    Args:
        intermediate_dim: Integer, hidden size of feed-forward intermediate layer.
        dropout_rate: Float between 0 and 1, dropout probability.
        activation: Activation function for feed-forward network. Defaults to 'gelu'.
        fourier_config: Optional dictionary of FNetFourierTransform configuration.
        layer_norm_epsilon: Epsilon value for layer normalization.
        use_bias: Boolean, whether to use bias terms in feed-forward layers.
        kernel_initializer: Weight initialization for feed-forward layers.
        **kwargs: Additional Layer base class arguments.
    """

    def __init__(
            self,
            intermediate_dim: int,
            dropout_rate: float = 0.1,
            activation: Union[str, callable] = 'gelu',
            fourier_config: Optional[Dict[str, Any]] = None,
            layer_norm_epsilon: float = 1e-12,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {intermediate_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if layer_norm_epsilon <= 0:
            raise ValueError(f"layer_norm_epsilon must be positive, got {layer_norm_epsilon}")

        # Store configuration
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.fourier_config = fourier_config or {}
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.supports_masking = True

        # CREATE all sub-layers in __init__
        self.fourier_transform = FNetFourierTransform(**self.fourier_config)

        self.fourier_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name='fourier_layer_norm'
        )

        self.intermediate_dense = keras.layers.Dense(
            intermediate_dim,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            name='intermediate_dense'
        )

        self.output_dense = keras.layers.Dense(
            None,  # Units determined in build()
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            name='output_dense'
        )

        self.dropout = keras.layers.Dropout(
            rate=dropout_rate,
            name='dropout'
        )

        self.output_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name='output_layer_norm'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build encoder block and all sub-layers with proper shape inference."""
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got {len(input_shape)}D: {input_shape}")

        hidden_dim = input_shape[-1]
        if hidden_dim is None:
            raise ValueError(f"Hidden dimension must be known at build time, got {hidden_dim}")

        self.output_dense = keras.layers.Dense(
            hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name='output_dense'
        )

        # Build sub-layers in computational order
        self.fourier_transform.build(input_shape)
        self.fourier_layer_norm.build(input_shape)
        self.intermediate_dense.build(input_shape)
        intermediate_shape = tuple(input_shape[:-1]) + (self.intermediate_dim,)
        self.dropout.build(intermediate_shape)
        self.output_dense.build(intermediate_shape)
        self.output_layer_norm.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through complete FNet encoder block.
        """
        # Fourier mixing with residual connection (pass mask)
        fourier_output = self.fourier_transform(inputs, mask=mask, training=training)
        fourier_output = self.fourier_layer_norm(
            inputs + fourier_output, training=training
        )

        # Position-wise feed-forward network with residual connection
        intermediate_output = self.intermediate_dense(fourier_output, training=training)
        intermediate_output = self.dropout(intermediate_output, training=training)
        ff_output = self.output_dense(intermediate_output, training=training)

        # Final normalization (pass mask)
        return self.output_layer_norm(
            fourier_output + ff_output, training=training
        )

    def compute_mask(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> Optional[keras.KerasTensor]:
        """Propagate the input mask."""
        return mask

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Encoder block preserves input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return complete configuration for serialization."""
        config = super().get_config()
        config.update({
            'intermediate_dim': self.intermediate_dim,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'fourier_config': self.fourier_config,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
        })
        return config