"""
This module provides a `TransformerLayer`, a highly configurable and generic
implementation of the fundamental building block of Transformer-based neural networks.

This layer encapsulates the two primary sub-layers of a standard Transformer:
a configurable multi-head attention mechanism and a position-wise feed-forward network. Both
sub-layers are wrapped with residual connections and normalization, which are crucial
for enabling the training of very deep networks.

A key feature of this implementation is its flexibility. It is designed to serve as a
versatile component for research and development, allowing for easy experimentation
with different attention mechanisms, normalization techniques and feed-forward network architectures.

Architectural Design (Configurable Normalization):

This implementation supports both "post-normalization" (Post-LN) and "pre-normalization"
(Pre-LN) architectures, configurable via the `normalization_position` parameter.

**Post-Normalization (Post-LN)**: Used in the original "Attention Is All You Need" paper.
- Operational flow: `output = Norm(SubLayer(x) + x)`
- Flow: SubLayer -> Add residual -> Normalize

**Pre-Normalization (Pre-LN)**: Often more stable for training deep networks.
- Operational flow: `output = x + SubLayer(Norm(x))`
- Flow: Normalize -> SubLayer -> Add residual

The full flow through the layer depends on normalization position:

**Post-LN Flow:**
1.  **Self-Attention Block:** Attention(x) -> Add residual -> LayerNorm
2.  **Feed-Forward Block:** FFN(attn_out) -> Add residual -> LayerNorm

**Pre-LN Flow:**
1.  **Self-Attention Block:** LayerNorm(x) -> Attention -> Add residual
2.  **Feed-Forward Block:** LayerNorm(attn_out) -> FFN -> Add residual

**Attention Mechanism Options:**
The attention mechanism can be one of several configurable architectures:
- 'multi_head_attention': Standard multi-head self-attention (default)
- 'window_attention': Windowed attention for efficient processing
- 'group_query_attention': Grouped query attention for reduced parameters
- Custom attention layers can be easily added

**Feed-Forward Network Options:**
The FFN can be one of several configurable architectures:
- 'mlp': Standard MLP with intermediate expansion
- 'swiglu': SwiGLU activation with gating mechanism
- 'differential': Differential FFN with separate positive/negative pathways
- 'glu': Gated Linear Unit with sigmoid gating
- 'residual': Residual block with skip connections
- 'swin_mlp': Swin Transformer MLP variant

Configurable Components:

Major strengths of this layer include:
- `attention_type`: Choose from various attention mechanisms
- `normalization_type`: Easily swap normalization functions (layer_norm, rms_norm, etc.)
- `normalization_position`: Choose between post-LN and pre-LN architectures
- `ffn_type`: Choose from various feed-forward architectures for different use cases
- Full parameter control for both attention and FFN components

This configurability makes the layer an excellent tool for architectural experimentation
and for building custom Transformer variants.
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

from .norms.rms_norm import RMSNorm
from .norms.band_rms import BandRMS

from .ffn.mlp import MLPBlock
from .ffn.glu_ffn import GLUFFN
from .ffn.swin_mlp import SwinMLP
from .ffn.swiglu_ffn import SwiGLUFFN
from .ffn.diff_ffn import DifferentialFFN
from .ffn.residual_block import ResidualBlock

from .attention.window_attention import WindowAttention
from .attention.group_query_attention import GroupedQueryAttention

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerLayer(keras.layers.Layer):
    """
    Generic transformer layer with configurable attention, normalization and FFN.

    This layer implements a standard transformer block with:
    - Configurable multi-head attention mechanisms
    - Configurable feed-forward network
    - Residual connections
    - Configurable normalization

    Args:
        hidden_size: Integer, hidden size of the layer.
        num_heads: Integer, number of attention heads.
        intermediate_size: Integer, size of the intermediate (feed-forward) layer.
        attention_type: String, type of attention mechanism to use.
            Available options:
            - 'multi_head_attention': Standard multi-head self-attention (default)
            - 'window_attention': Windowed attention for efficient processing
            - 'group_query_attention': Grouped query attention for reduced parameters
        normalization_type: String, type of normalization to use.
            Available options:
            - 'layer_norm': Standard layer normalization (default)
            - 'batch_norm': Batch normalization
            - 'rms_norm': Root Mean Square normalization
            - 'band_rms': Band-constrained RMS normalization
        normalization_position: String, position of normalization layers.
            Available options:
            - 'post': Post-normalization (original Transformer, default)
            - 'pre': Pre-normalization (often more stable for deep networks)
        ffn_type: String, type of feed-forward network to use.
            Available options:
            - 'mlp': Standard MLP with intermediate expansion (default)
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'differential': Differential FFN with separate pathways
            - 'glu': Gated Linear Unit with sigmoid gating
            - 'residual': Residual block with skip connections
            - 'swin_mlp': Swin Transformer MLP variant
        dropout_rate: Float, dropout rate. Defaults to 0.1.
        attention_dropout_rate: Float, attention-specific dropout rate. Defaults to 0.1.
        activation: String or callable, activation function for feed-forward network.
            Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: String or initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        ffn_expansion_factor: Integer, expansion factor for SwiGLU FFN. Defaults to 4.
        ffn_multiple_of: Integer, multiple constraint for SwiGLU FFN. Defaults to 256.
        window_size: Integer, window size for window attention. Defaults to 8.
        n_kv_head: Integer, number of key-value heads for grouped query attention.
            Defaults to None (uses num_heads).
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Example:
        ```python
        # Standard multi-head attention
        inputs = keras.Input(shape=(128, 768))
        layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            attention_type='multi_head_attention',
            normalization_position='pre',
            ffn_type='swiglu'
        )
        outputs = layer(inputs)

        # Window attention
        layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            attention_type='window_attention',
            window_size=16
        )

        # Grouped query attention
        layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            attention_type='group_query_attention',
            n_kv_head=4
        )
        ```
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            attention_type: str = 'multi_head_attention',
            normalization_type: str = 'layer_norm',
            normalization_position: str = 'post',
            ffn_type: str = 'mlp',
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            activation: Union[str, callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            ffn_expansion_factor: int = 4,
            ffn_multiple_of: int = 256,
            window_size: int = 8,
            n_kv_head: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        valid_attention_types = ['multi_head_attention', 'window_attention', 'group_query_attention']
        if attention_type not in valid_attention_types:
            raise ValueError(f"attention_type must be one of {valid_attention_types}, got {attention_type}")

        valid_norm_types = ['layer_norm', 'rms_norm', 'batch_norm', 'band_rms']
        if normalization_type not in valid_norm_types:
            raise ValueError(f"normalization_type must be one of {valid_norm_types}, got {normalization_type}")

        valid_norm_positions = ['post', 'pre']
        if normalization_position not in valid_norm_positions:
            raise ValueError(f"normalization_position must be one of {valid_norm_positions}, got {normalization_position}")

        valid_ffn_types = ['mlp', 'swiglu', 'differential', 'glu', 'residual', 'swin_mlp']
        if ffn_type not in valid_ffn_types:
            raise ValueError(f"ffn_type must be one of {valid_ffn_types}, got {ffn_type}")

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        # Store configuration parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.window_size = window_size
        self.n_kv_head = n_kv_head if n_kv_head is not None else num_heads

        # Initialize layers to None - will be created in build()
        self.attention = None
        self.attention_norm = None
        self.ffn_layer = None
        self.output_norm = None
        self.dropout = None
        self.attention_dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def _create_normalization_layer(self, name: str) -> keras.layers.Layer:
        """Create a normalization layer based on the specified type.

        Args:
            name: Name for the normalization layer.

        Returns:
            A normalization layer instance.
        """
        if self.normalization_type == 'layer_norm':
            return keras.layers.LayerNormalization(
                epsilon=1e-12,
                name=name
            )
        elif self.normalization_type == 'rms_norm':
            return RMSNorm(name=name)
        elif self.normalization_type == 'band_rms':
            return BandRMS(max_band_width=0.1, name=name)
        elif self.normalization_type == 'batch_norm':
            return keras.layers.BatchNormalization(
                epsilon=1e-12,
                name=name
            )
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def _get_attention_params(self, attention_type: str, name: str) -> Dict[str, Any]:
        """Get parameters for attention layer creation based on type.

        Args:
            attention_type: Type of attention layer.
            name: Name for the layer.

        Returns:
            Dictionary of parameters for the specific attention type.
        """
        if attention_type == 'multi_head_attention':
            # Standard Keras MultiHeadAttention parameters
            return {
                'num_heads': self.num_heads,
                'key_dim': self.hidden_size // self.num_heads,
                'dropout': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'name': name
            }
        elif attention_type == 'window_attention':
            # WindowAttention parameters
            return {
                'dim': self.hidden_size,
                'window_size': self.window_size,
                'num_heads': self.num_heads,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'name': name
            }
        elif attention_type == 'group_query_attention':
            # GroupedQueryAttention parameters
            return {
                'd_model': self.hidden_size,
                'n_head': self.num_heads,
                'n_kv_head': self.n_kv_head,
                'dropout': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'name': name
            }
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def _create_attention_layer(self, name: str) -> keras.layers.Layer:
        """Create an attention layer based on the specified type.

        Args:
            name: Name for the attention layer.

        Returns:
            An attention layer instance.
        """
        # Get parameters for this attention type
        params = self._get_attention_params(self.attention_type, name)

        try:
            if self.attention_type == 'multi_head_attention':
                return keras.layers.MultiHeadAttention(**params)
            elif self.attention_type == 'window_attention':
                return WindowAttention(**params)
            elif self.attention_type == 'group_query_attention':
                return GroupedQueryAttention(**params)
            else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
        except (TypeError, ValueError) as e:
            # Log the issue and provide helpful error message
            logger.warning(f"Failed to create {self.attention_type} layer: {e}")
            logger.warning(f"Attempted parameters: {list(params.keys())}")

            # If creation fails, provide a more informative error
            raise ValueError(
                f"Failed to create {self.attention_type} layer. "
                f"This might be due to parameter incompatibility or missing dependencies. "
                f"Original error: {e}"
            )

    def _get_ffn_params(self, ffn_type: str, name: str) -> Dict[str, Any]:
        """Get parameters for FFN layer creation based on type.

        Args:
            ffn_type: Type of FFN layer.
            name: Name for the layer.

        Returns:
            Dictionary of parameters for the specific FFN type.
        """
        if ffn_type == 'mlp':
            # MLPBlock parameters (known to work except bias_regularizer)
            return {
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'name': name
            }
        elif ffn_type == 'swiglu':
            # SwiGLUFFN - be very conservative, only pass core parameters
            return {
                'd_model': self.hidden_size,
                'ffn_expansion_factor': self.ffn_expansion_factor,
                'ffn_multiple_of': self.ffn_multiple_of,
                'name': name
            }
        elif ffn_type == 'differential':
            # DifferentialFFN parameters
            return {
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'branch_activation': self.activation,
                'dropout_rate': self.dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'name': name
            }
        elif ffn_type == 'glu':
            # GLUFFN parameters
            return {
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'activation': keras.activations.get(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'name': name
            }
        elif ffn_type == 'residual':
            # ResidualBlock parameters
            return {
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'dropout_rate': self.dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'name': name
            }
        elif ffn_type == 'swin_mlp':
            # SwinMLP parameters
            return {
                'hidden_dim': self.intermediate_size,
                'out_dim': self.hidden_size,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'kernel_regularizer': self.kernel_regularizer,
                'name': name
            }
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

    def _create_ffn_layer(self, name: str) -> keras.layers.Layer:
        """Create a feed-forward network layer based on the specified type.

        Args:
            name: Name for the FFN layer.

        Returns:
            An FFN layer instance.
        """
        # Get parameters for this FFN type
        params = self._get_ffn_params(self.ffn_type, name)

        try:
            if self.ffn_type == 'mlp':
                return MLPBlock(**params)
            elif self.ffn_type == 'swiglu':
                return SwiGLUFFN(**params)
            elif self.ffn_type == 'differential':
                return DifferentialFFN(**params)
            elif self.ffn_type == 'glu':
                return GLUFFN(**params)
            elif self.ffn_type == 'residual':
                return ResidualBlock(**params)
            elif self.ffn_type == 'swin_mlp':
                return SwinMLP(**params)
            else:
                raise ValueError(f"Unknown FFN type: {self.ffn_type}")
        except (TypeError, ValueError) as e:
            # Log the issue and provide helpful error message
            logger.warning(f"Failed to create {self.ffn_type} layer: {e}")
            logger.warning(f"Attempted parameters: {list(params.keys())}")

            # If creation fails, provide a more informative error
            raise ValueError(
                f"Failed to create {self.ffn_type} layer. "
                f"This might be due to parameter incompatibility or missing dependencies. "
                f"Original error: {e}"
            )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the transformer layer components.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        # Configurable attention layer
        self.attention = self._create_attention_layer('attention')

        # Attention layer normalization
        self.attention_norm = self._create_normalization_layer('attention_norm')

        # Feed-forward network (configurable)
        self.ffn_layer = self._create_ffn_layer('ffn')

        # Output layer normalization
        self.output_norm = self._create_normalization_layer('output_norm')

        # Dropout layers
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='dropout')
        self.attention_dropout = keras.layers.Dropout(
            self.attention_dropout_rate,
            name='attention_dropout'
        )

        # Build sublayers
        # For different attention types, we need to handle the build call appropriately
        if self.attention_type == 'multi_head_attention':
            self.attention.build(query_shape=input_shape, value_shape=input_shape)
        else:
            # For custom attention layers, just pass the input shape
            self.attention.build(input_shape)

        self.attention_norm.build(input_shape)
        self.ffn_layer.build(input_shape)
        self.output_norm.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            attention_mask: Optional[Any] = None,
            training: Optional[bool] = None
    ) -> Any:
        """
        Forward pass of the transformer layer.

        Args:
            inputs: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor. Can be:
                - 2D tensor of shape (batch_size, seq_length) for padding mask
                - 3D tensor of shape (batch_size, seq_length, seq_length) for attention mask
                - 4D tensor of shape (batch_size, num_heads, seq_length, seq_length) for full mask
            training: Boolean indicating training mode

        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Handle attention mask processing
        processed_mask = None
        if attention_mask is not None:
            if len(attention_mask.shape) == 3:
                # Use 3D mask as-is
                processed_mask = attention_mask
            elif len(attention_mask.shape) == 4:
                # Use 4D mask as-is
                processed_mask = attention_mask
            else:
                # Skip 2D masks for now to avoid shape issues
                logger.warning(f"Skipping 2D attention mask of shape {attention_mask.shape}. Use 3D mask instead.")
                processed_mask = None

        if self.normalization_position == 'post':
            # Post-normalization: SubLayer(x) -> Add residual -> Normalize

            # Multi-head attention with residual connection
            if self.attention_type == 'multi_head_attention':
                # Standard Keras MultiHeadAttention call
                attention_output = self.attention(
                    query=inputs,
                    value=inputs,  # value = query for self-attention
                    key=inputs,  # key = query for self-attention
                    attention_mask=processed_mask,
                    training=training
                )
            else:
                # Custom attention layers (window_attention, group_query_attention, etc.)
                # These typically just take inputs and optional training
                attention_output = self.attention(inputs, training=training)

            attention_output = self.attention_dropout(attention_output, training=training)
            attention_output = self.attention_norm(attention_output + inputs, training=training)

            # Feed-forward network with residual connection
            ffn_output = self.ffn_layer(attention_output, training=training)

            layer_output = self.output_norm(ffn_output + attention_output, training=training)

        else:  # pre-normalization
            # Pre-normalization: Normalize -> SubLayer(x) -> Add residual

            # Multi-head attention with residual connection
            normalized_inputs = self.attention_norm(inputs, training=training)

            if self.attention_type == 'multi_head_attention':
                # Standard Keras MultiHeadAttention call
                attention_output = self.attention(
                    query=normalized_inputs,
                    value=normalized_inputs,  # value = query for self-attention
                    key=normalized_inputs,  # key = query for self-attention
                    attention_mask=processed_mask,
                    training=training
                )
            else:
                # Custom attention layers
                attention_output = self.attention(normalized_inputs, training=training)

            attention_output = self.attention_dropout(attention_output, training=training)
            attention_output = attention_output + inputs  # Add residual

            # Feed-forward network with residual connection
            normalized_attention = self.output_norm(attention_output, training=training)
            ffn_output = self.ffn_layer(normalized_attention, training=training)

            layer_output = ffn_output + attention_output  # Add residual

        return layer_output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Output shape is the same as input shape for transformer layers
        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'ffn_multiple_of': self.ffn_multiple_of,
            'window_size': self.window_size,
            'n_kv_head': self.n_kv_head,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------
