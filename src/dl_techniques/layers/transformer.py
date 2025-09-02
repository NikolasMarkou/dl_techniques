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

This configurability makes the layer an excellent tool for architectural experimentation
and for building custom Transformer variants.
"""

import keras
from keras import layers, initializers, regularizers
from typing import Optional, Union, Any, Dict, Tuple, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms.rms_norm import RMSNorm
from .norms.band_rms import BandRMS
from .stochastic_depth import StochasticDepth
from .activations.dynamic_tanh import DynamicTanh

from .ffn.mlp import MLPBlock
from .ffn.glu_ffn import GLUFFN
from .ffn.swin_mlp import SwinMLP
from .ffn.geglu_ffn import GeGLUFFN
from .ffn.swiglu_ffn import SwiGLUFFN
from .ffn.diff_ffn import DifferentialFFN
from .ffn.residual_block import ResidualBlock

from .attention.window_attention import WindowAttention
from .attention.group_query_attention import GroupedQueryAttention
from .attention.differential_attention import DifferentialMultiHeadAttention

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

NormalizationPosition = Literal['post', 'pre']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'dynamic_tanh']
AttentionType = Literal['multi_head_attention', 'window_attention', 'group_query_attention', 'differential_attention']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerLayer(keras.layers.Layer):
    """
    Generic transformer layer with configurable attention, normalization, FFN, and stochastic depth.

    This layer implements a standard transformer block with:
    - Configurable multi-head attention mechanisms
    - Configurable feed-forward network
    - Residual connections
    - Configurable normalization
    - Optional stochastic depth regularization
    - Enhanced parameter control through argument dictionaries

    Args:
        hidden_size: Integer, hidden size of the layer.
        num_heads: Integer, number of attention heads.
        intermediate_size: Integer, size of the intermediate (feed-forward) layer.
        attention_type: AttentionType, type of attention mechanism to use.
            Available options:
            - 'multi_head_attention': Standard multi-head self-attention (default)
            - 'window_attention': Windowed attention for efficient processing
            - 'group_query_attention': Grouped query attention for reduced parameters
            - 'differential_attention': Differential attention for noise cancellation
        attention_args: Optional dictionary of custom arguments for attention layer.
            These will override default parameters for the specific attention type.
        normalization_type: NormalizationType, type of normalization to use.
            Available options:
            - 'layer_norm': Standard layer normalization (default)
            - 'batch_norm': Batch normalization
            - 'rms_norm': Root Mean Square normalization
            - 'band_rms': Band-constrained RMS normalization
            - 'dynamic_tanh': Dynamic Tanh normalization (DyT) for normalization-free transformers
        normalization_position: NormalizationPosition, position of normalization layers.
            Available options:
            - 'post': Post-normalization (original Transformer, default)
            - 'pre': Pre-normalization (often more stable for deep networks)
        ffn_type: FFNType, type of feed-forward network to use.
            Available options:
            - 'mlp': Standard MLP with intermediate expansion (default)
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'differential': Differential FFN with separate pathways
            - 'glu': Gated Linear Unit with sigmoid gating
            - 'geglu': GELU-based Gated Linear Unit (GeGLU) Feed Forward Network
            - 'residual': Residual block with skip connections
            - 'swin_mlp': Swin Transformer MLP variant
        ffn_args: Optional dictionary of custom arguments for FFN layer.
            These will override default parameters for the specific FFN type.
        dropout_rate: Float, dropout rate. Defaults to 0.1.
        attention_dropout_rate: Float, attention-specific dropout rate. Defaults to 0.1.
        use_stochastic_depth: Boolean, whether to use stochastic depth regularization.
            Defaults to False.
        stochastic_depth_rate: Float, drop path rate for stochastic depth.
            Only used when use_stochastic_depth=True. Defaults to 0.1.
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
        lambda_init: Float, initial lambda value for differential attention.
            Only used when attention_type='differential_attention'. Defaults to 0.8.
        attention_norm_alpha: Float, initial alpha value for DynamicTanh in attention normalization.
            Only used when normalization_type='dynamic_tanh'. Defaults to 0.7.
        ffn_norm_alpha: Float, initial alpha value for DynamicTanh in FFN normalization.
            Only used when normalization_type='dynamic_tanh'. Defaults to 0.15.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Example:
        ```python
        # Standard multi-head attention with custom dropout
        inputs = keras.Input(shape=(128, 768))
        layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            attention_type='multi_head_attention',
            attention_args={'dropout': 0.2},  # Custom attention dropout
            normalization_position='pre',
            ffn_type='swiglu',
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1
        )
        outputs = layer(inputs)
        ```
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            attention_type: AttentionType = 'multi_head_attention',
            attention_args: Optional[Dict[str, Any]] = None,
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPosition = 'post',
            ffn_type: FFNType = 'mlp',
            ffn_args: Optional[Dict[str, Any]] = None,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            activation: Union[str, Callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            ffn_expansion_factor: int = 4,
            ffn_multiple_of: int = 256,
            window_size: int = 8,
            n_kv_head: Optional[int] = None,
            lambda_init: float = 0.8,
            attention_norm_alpha: float = 0.7,
            ffn_norm_alpha: float = 0.15,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Configuration Storage ---
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_type = attention_type
        self.attention_args = attention_args or {}
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args or {}
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.window_size = window_size
        self.n_kv_head = n_kv_head if n_kv_head is not None else num_heads
        self.lambda_init = lambda_init
        self.attention_norm_alpha = attention_norm_alpha
        self.ffn_norm_alpha = ffn_norm_alpha

        # --- Input Validation ---
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )

        # --- Create Sub-layers (unbuilt) ---
        # Per Keras best practices, all sub-layers are created in __init__
        self.attention_norm = self._create_normalization_layer('attention_norm', 'attention')
        self.output_norm = self._create_normalization_layer('output_norm', 'ffn')
        self.attention = self._create_attention_layer('attention')
        self.ffn_layer = self._create_ffn_layer('ffn')

        self.dropout = layers.Dropout(self.dropout_rate, name='dropout')
        self.attention_dropout = layers.Dropout(self.attention_dropout_rate, name='attention_dropout')

        # Stochastic depth layers (if enabled)
        self.attention_stochastic_depth = None
        self.ffn_stochastic_depth = None
        if self.use_stochastic_depth:
            self.attention_stochastic_depth = StochasticDepth(
                drop_path_rate=self.stochastic_depth_rate,
                name='attention_stochastic_depth'
            )
            self.ffn_stochastic_depth = StochasticDepth(
                drop_path_rate=self.stochastic_depth_rate,
                name='ffn_stochastic_depth'
            )

    def _create_normalization_layer(self, name: str, layer_type: str = 'attention') -> keras.layers.Layer:
        """Create a normalization layer based on the specified type."""
        if self.normalization_type == 'layer_norm':
            return layers.LayerNormalization(epsilon=1e-12, name=name)
        elif self.normalization_type == 'rms_norm':
            return RMSNorm(name=name)
        elif self.normalization_type == 'band_rms':
            return BandRMS(max_band_width=0.1, name=name)
        elif self.normalization_type == 'batch_norm':
            return layers.BatchNormalization(epsilon=1e-12, name=name)
        elif self.normalization_type == 'dynamic_tanh':
            alpha_value = self.attention_norm_alpha if layer_type == 'attention' else self.ffn_norm_alpha
            return DynamicTanh(alpha_init_value=alpha_value, name=name)
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def _get_attention_params(self, name: str) -> Dict[str, Any]:
        """Get parameters for attention layer creation, merging defaults with custom args."""
        # Define default parameters for each attention type
        if self.attention_type == 'multi_head_attention':
            default_params = {
                'num_heads': self.num_heads,
                'key_dim': self.hidden_size // self.num_heads,
                'dropout': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'bias_initializer': self.bias_initializer,
                'name': name
            }
        elif self.attention_type == 'window_attention':
            default_params = {
                'dim': self.hidden_size,
                'window_size': self.window_size,
                'num_heads': self.num_heads,
                'name': name
            }
        elif self.attention_type == 'group_query_attention':
            default_params = {
                'd_model': self.hidden_size,
                'n_head': self.num_heads,
                'n_kv_head': self.n_kv_head,
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'name': name
            }
        elif self.attention_type == 'differential_attention':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'head_dim': self.hidden_size // self.num_heads,
                'dropout': self.dropout_rate,
                'lambda_init': self.lambda_init,
                'name': name
            }
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        # Merge with custom arguments, giving priority to custom args
        return {**default_params, **self.attention_args}

    def _create_attention_layer(self, name: str) -> keras.layers.Layer:
        """Create an attention layer based on the specified type."""
        params = self._get_attention_params(name)
        try:
            if self.attention_type == 'multi_head_attention':
                return layers.MultiHeadAttention(**params)
            elif self.attention_type == 'window_attention':
                return WindowAttention(**params)
            elif self.attention_type == 'group_query_attention':
                return GroupedQueryAttention(**params)
            elif self.attention_type == 'differential_attention':
                return DifferentialMultiHeadAttention(**params)
            else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.attention_type} layer. "
                f"Check for parameter incompatibility. Custom args: {list(self.attention_args.keys())}. "
                f"Original error: {e}"
            )

    def _get_ffn_params(self, name: str) -> Dict[str, Any]:
        """Get parameters for FFN layer creation, merging defaults with custom args."""
        if self.ffn_type == 'mlp':
            default_params = {
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate,
                'name': name
            }
        elif self.ffn_type == 'swiglu':
            default_params = {
                'd_model': self.hidden_size,
                'ffn_expansion_factor': self.ffn_expansion_factor,
                'ffn_multiple_of': self.ffn_multiple_of,
                'name': name
            }
        elif self.ffn_type in ['differential', 'geglu', 'glu', 'residual', 'swin_mlp']:
            default_params = {
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate,
                'name': name
            }
        else:
            raise ValueError(f"Unknown FFN type: {self.ffn_type}")
        return {**default_params, **self.ffn_args}

    def _create_ffn_layer(self, name: str) -> keras.layers.Layer:
        """Create a feed-forward network layer based on the specified type."""
        params = self._get_ffn_params(name)
        try:
            if self.ffn_type == 'mlp':
                return MLPBlock(**params)
            elif self.ffn_type == 'swiglu':
                return SwiGLUFFN(**params)
            elif self.ffn_type == 'differential':
                params['branch_activation'] = params.pop('activation') # remap
                return DifferentialFFN(**params)
            elif self.ffn_type == 'glu':
                return GLUFFN(**params)
            elif self.ffn_type == 'geglu':
                return GeGLUFFN(**params)
            elif self.ffn_type == 'residual':
                return ResidualBlock(**params)
            elif self.ffn_type == 'swin_mlp':
                params['out_dim'] = params.pop('output_dim') # remap
                return SwinMLP(**params)
            else:
                raise ValueError(f"Unknown FFN type: {self.ffn_type}")
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.ffn_type} layer. "
                f"Check for parameter incompatibility. Custom args: {list(self.ffn_args.keys())}. "
                f"Original error: {e}"
            )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Builds all sub-layers with appropriate shapes."""
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}")
        if input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        # Build all sub-layers. Since this is a standard Transformer block
        # where the shape is preserved, we can pass `input_shape` to all.
        self.attention_norm.build(input_shape)
        self.output_norm.build(input_shape)
        self.ffn_layer.build(input_shape)

        # Handle special build signature for Keras's MultiHeadAttention
        if self.attention_type == 'multi_head_attention':
            self.attention.build(query_shape=input_shape, value_shape=input_shape)
        else:
            # Other custom attention layers expect a single input_shape
            self.attention.build(input_shape)

        # Build stochastic depth layers if they exist
        if self.attention_stochastic_depth is not None:
            self.attention_stochastic_depth.build(input_shape)
        if self.ffn_stochastic_depth is not None:
            self.ffn_stochastic_depth.build(input_shape)

        # CRITICAL: Always call super().build() at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            layer_idx: int = 0,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the transformer layer."""
        residual = inputs

        mha_attention_mask = attention_mask
        if self.attention_type == 'multi_head_attention' and mha_attention_mask is not None:
            mask_shape = mha_attention_mask.shape
            if len(mask_shape) == 2:
                # This heuristic differentiates between a (batch_size, seq_len) padding mask
                # and a (seq_len, seq_len) causal mask. It's robust because even if
                # batch_size == seq_len, a square mask is almost always a causal mask,
                # which Keras's MHA handles correctly without expansion.
                if mask_shape[0] != mask_shape[1]:
                    mha_attention_mask = mha_attention_mask[:, None, None, :]

        if self.normalization_position == 'pre':
            # --- Pre-Normalization: Normalize -> SubLayer -> StochasticDepth -> Add ---
            # 1. Attention block
            x = self.attention_norm(inputs, training=training)

            if self.attention_type == 'multi_head_attention':
                x = self.attention(query=x, value=x, key=x, attention_mask=mha_attention_mask, training=training)
            elif self.attention_type == 'differential_attention':
                x = self.attention(x, mask=attention_mask, layer_idx=layer_idx, training=training)
            else:
                x = self.attention(x, training=training) # Assume custom layers handle masks internally if needed

            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)

            attention_output = x + residual

            # 2. FFN block
            residual = attention_output
            x = self.output_norm(attention_output, training=training)
            x = self.ffn_layer(x, training=training)

            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            layer_output = x + residual
        else: # Post-normalization
            # --- Post-Normalization: SubLayer -> StochasticDepth -> Add -> Normalize ---
            # 1. Attention block
            if self.attention_type == 'multi_head_attention':
                x = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=mha_attention_mask, training=training)
            elif self.attention_type == 'differential_attention':
                x = self.attention(inputs, mask=attention_mask, layer_idx=layer_idx, training=training)
            else:
                x = self.attention(inputs, training=training)

            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)

            attention_output = self.attention_norm(x + residual, training=training)

            # 2. FFN block
            residual = attention_output
            x = self.ffn_layer(attention_output, training=training)

            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            layer_output = self.output_norm(x + residual, training=training)

        return layer_output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'attention_type': self.attention_type,
            'attention_args': self.attention_args,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'ffn_args': self.ffn_args,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'ffn_multiple_of': self.ffn_multiple_of,
            'window_size': self.window_size,
            'n_kv_head': self.n_kv_head,
            'lambda_init': self.lambda_init,
            'attention_norm_alpha': self.attention_norm_alpha,
            'ffn_norm_alpha': self.ffn_norm_alpha,
        })
        return config