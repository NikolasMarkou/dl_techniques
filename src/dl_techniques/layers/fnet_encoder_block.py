"""
FNet: Fourier Transform-based Attention Replacement with Factory Pattern

This module implements the FNet architecture from "FNet: Mixing Tokens with Fourier Transforms"
(Lee-Thorp et al., 2021), which replaces self-attention with parameter-free Fourier transforms
for efficient token mixing in transformer-style architectures.

This refined version uses the normalization and FFN factory patterns for better modularity
and consistency across the dl_techniques framework.
"""

import keras
from typing import Optional, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .attention.fnet_fourier_transform import FNetFourierTransform

from .ffn import create_ffn_layer
from .norms import create_normalization_layer

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

NormalizationType = Literal[
    'layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms',
    'global_response_norm', 'logit_norm', 'max_logit_norm', 'decoupled_max_logit',
    'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh'
]

FFNType = Literal[
    'mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp'
]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FNetEncoderBlock(keras.layers.Layer):
    """
    Complete FNet encoder block with Fourier mixing and feed-forward components using factory patterns.

    This layer implements a complete FNet encoder block as described in the paper,
    combining the FNet Fourier Transform with a configurable feed-forward network.
    The architecture mirrors Transformer encoder blocks but replaces expensive
    self-attention with efficient Fourier-based token mixing.

    **Key Improvements**:
    - Uses normalization factory for consistent normalization layer creation
    - Uses FFN factory for modular feed-forward network selection
    - Supports all dl_techniques normalization types
    - Supports all dl_techniques FFN types
    - Better configurability and experimentation support

    **Intent**: Provide a drop-in replacement for Transformer encoder blocks that maintains
    comparable modeling capacity while achieving significant speedup through parameter-free
    token mixing and reduced computational complexity.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, hidden_dim], mask=[batch, seq_len])
           ↓
    FNet Fourier Transform (preserves mask)
           ↓
    Residual Connection + Normalization (configurable type, preserves mask)
           ↓
    Feed-Forward Network (configurable type, preserves mask)
           ↓
    Residual Connection + Normalization (configurable type, preserves mask)
           ↓
    Output(shape=[batch, seq_len, hidden_dim], mask=[batch, seq_len])
    ```

    **Design Principles**:
    - **Masking Support**: Correctly propagates masks through all sub-layers.
    - **Efficiency**: Fourier mixing is faster than self-attention, especially for long sequences.
    - **Modularity**: Uses factory patterns for consistent layer creation.
    - **Flexibility**: Supports multiple normalization and FFN types.
    - **Compatibility**: Same input/output interface as standard Transformer blocks.

    Args:
        intermediate_dim: Integer, hidden size of feed-forward intermediate layer.
            Ignored if ffn_type requires different parameters (e.g., 'swiglu' uses ffn_expansion_factor).
        dropout_rate: Float between 0 and 1, dropout probability for FFN.
        fourier_config: Optional dictionary of FNetFourierTransform configuration.
        normalization_type: NormalizationType, type of normalization to use.
        normalization_kwargs: Optional dictionary of normalization-specific parameters.
        ffn_type: FFNType, type of feed-forward network to use.
        ffn_kwargs: Optional dictionary of FFN-specific parameters.
        **kwargs: Additional Layer base class arguments.

    Examples:
        # Standard FNet with layer normalization and MLP
        fnet_block = FNetEncoderBlock(intermediate_dim=2048)

        # Modern FNet with RMS normalization and SwiGLU
        fnet_block = FNetEncoderBlock(
            intermediate_dim=None,  # Not used for SwiGLU
            normalization_type='rms_norm',
            normalization_kwargs={'use_scale': True},
            ffn_type='swiglu',
            ffn_kwargs={'ffn_expansion_factor': 4}
        )

        # Efficient FNet with Band RMS and GLU
        fnet_block = FNetEncoderBlock(
            intermediate_dim=1024,
            normalization_type='band_rms',
            normalization_kwargs={'max_band_width': 0.1},
            ffn_type='glu'
        )
    """

    def __init__(
        self,
        intermediate_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
        fourier_config: Optional[Dict[str, Any]] = None,
        normalization_type: NormalizationType = 'layer_norm',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        ffn_type: FFNType = 'mlp',
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if intermediate_dim is not None and intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {intermediate_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.fourier_config = fourier_config or {}
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.ffn_type = ffn_type
        self.ffn_kwargs = ffn_kwargs or {}
        self.supports_masking = True

        # Create Fourier transform layer
        self.fourier_transform = FNetFourierTransform(**self.fourier_config)

        # Layer creation will be done in build() to ensure proper shape inference
        self.fourier_layer_norm = None
        self.ffn_layer = None
        self.output_layer_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build encoder block and all sub-layers with proper shape inference."""
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got {len(input_shape)}D: {input_shape}")

        hidden_dim = input_shape[-1]
        if hidden_dim is None:
            raise ValueError(f"Hidden dimension must be known at build time, got {hidden_dim}")

        # Build Fourier transform layer
        self.fourier_transform.build(input_shape)

        # Create normalization layers using factory
        self.fourier_layer_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name='fourier_layer_norm',
            **self.normalization_kwargs
        )

        self.output_layer_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name='output_layer_norm',
            **self.normalization_kwargs
        )

        # Create FFN layer using factory
        # Prepare FFN parameters
        ffn_params = {
            'output_dim': hidden_dim,
            'name': 'ffn',
            **self.ffn_kwargs
        }

        # Add intermediate_dim if specified and if FFN type uses it
        if self.intermediate_dim is not None:
            if self.ffn_type in ['mlp', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']:
                ffn_params['hidden_dim'] = self.intermediate_dim
            elif self.ffn_type == 'swiglu':
                # SwiGLU uses ffn_expansion_factor instead of hidden_dim
                if 'ffn_expansion_factor' not in ffn_params:
                    # Calculate expansion factor from intermediate_dim
                    ffn_params['ffn_expansion_factor'] = self.intermediate_dim // hidden_dim

        # Add dropout rate if not already specified
        if 'dropout_rate' not in ffn_params:
            ffn_params['dropout_rate'] = self.dropout_rate

        try:
            self.ffn_layer = create_ffn_layer(self.ffn_type, **ffn_params)
        except Exception as e:
            logger.error(f"Failed to create FFN layer of type '{self.ffn_type}': {e}")
            # Fallback to standard MLP
            fallback_params = {
                'hidden_dim': self.intermediate_dim or hidden_dim * 4,
                'output_dim': hidden_dim,
                'dropout_rate': self.dropout_rate,
                'name': 'ffn_fallback'
            }
            self.ffn_layer = create_ffn_layer('mlp', **fallback_params)

        # Build normalization layers
        self.fourier_layer_norm.build(input_shape)
        self.output_layer_norm.build(input_shape)

        # Build FFN layer
        self.ffn_layer.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through complete FNet encoder block.

        Args:
            inputs: Input tensor of shape [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask tensor of shape [batch, seq_len].
                Will be passed to Fourier transform if provided.
            training: Boolean indicating training mode.

        Returns:
            Output tensor of same shape as input: [batch, seq_len, hidden_dim].
        """
        # Fourier mixing with residual connection (pass mask)
        fourier_output = self.fourier_transform(
            inputs, attention_mask=attention_mask, training=training
        )
        fourier_output = self.fourier_layer_norm(
            inputs + fourier_output, training=training
        )

        # Feed-forward network with residual connection
        ffn_output = self.ffn_layer(fourier_output, training=training)

        # Final normalization with residual connection
        return self.output_layer_norm(
            fourier_output + ffn_output, training=training
        )

    def compute_mask(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None
    ) -> Optional[keras.KerasTensor]:
        """Propagate the input mask unchanged."""
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
            'fourier_config': self.fourier_config,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'ffn_type': self.ffn_type,
            'ffn_kwargs': self.ffn_kwargs,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FNetEncoderBlock':
        """Create layer from configuration dictionary."""
        return cls(**config)

# ---------------------------------------------------------------------
