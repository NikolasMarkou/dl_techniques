"""
This module provides a `TransformerLayer`, a highly configurable and generic
implementation of the fundamental building block of Transformer-based neural networks.

This layer encapsulates the two primary sub-layers of a standard Transformer:
a configurable multi-head attention mechanism and a position-wise feed-forward network. Both
sub-layers are wrapped with residual connections and normalization, which are crucial
for enabling the training of very deep networks.

A key feature of this implementation is its flexibility. It is designed to serve as a
versatile component for research and development, allowing for easy experimentation
with different attention mechanisms, normalization techniques and feed-forward network architectures,
including Mixture of Experts (MoE).

This configurability makes the layer an excellent tool for architectural experimentation
and for building custom Transformer variants.
"""

import keras
import warnings
from keras import layers, initializers, regularizers
from typing import Optional, Union, Any, Dict, Tuple, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .stochastic_depth import StochasticDepth


from .norms import create_normalization_layer
from .ffn.factory import create_ffn_from_config

from .attention.window_attention import WindowAttention
from .attention.multi_head_attention import MultiHeadAttention
from .attention.group_query_attention import GroupedQueryAttention
from .attention.differential_attention import DifferentialMultiHeadAttention

# MoE Integration
from .moe import MixtureOfExperts, MoEConfig

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

NormalizationPosition = Literal['post', 'pre']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']
AttentionType = Literal['multi_head_attention', 'window_attention', 'group_query_attention', 'differential_attention']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerLayer(keras.layers.Layer):
    """
    Generic transformer layer with configurable attention, normalization, FFN, and stochastic depth.

    This layer implements a standard transformer block with:
    - Configurable attention mechanisms
    - Configurable feed-forward network (including Mixture of Experts)
    - Residual connections
    - Configurable normalization with custom arguments support
    - Optional stochastic depth regularization
    - Enhanced parameter control through argument dictionaries

    Args:
        hidden_size: Integer, hidden size of the layer.
        num_heads: Integer, number of attention heads.
        intermediate_size: Integer, size of the intermediate (feed-forward) layer.
            Ignored if `moe_config` is provided.
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
            - 'adaptive_band_rms': Adaptive Band RMS with log-transformed scaling
            - 'logit_norm': Temperature-scaled normalization for classification
            - 'dynamic_tanh': Dynamic Tanh normalization (DyT) for normalization-free transformers
        normalization_position: NormalizationPosition, position of normalization layers.
            Available options:
            - 'post': Post-normalization (original Transformer, default)
            - 'pre': Pre-normalization (often more stable for deep networks)
        attention_norm_args: Optional dictionary of custom arguments for attention normalization layer.
            These parameters will be passed to the normalization layer factory.
            Example: For 'dynamic_tanh': {'alpha_init_value': 0.7, 'axis': [-1]}
                    For 'band_rms': {'max_band_width': 0.1, 'epsilon': 1e-7}
        ffn_norm_args: Optional dictionary of custom arguments for FFN normalization layer.
            These parameters will be passed to the normalization layer factory.
            Example: For 'dynamic_tanh': {'alpha_init_value': 0.15, 'axis': [-1]}
        ffn_type: FFNType, type of feed-forward network to use.
            Ignored if `moe_config` is provided.
            Available options:
            - 'mlp': Standard MLP with intermediate expansion (default)
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'differential': Differential FFN with separate pathways
            - 'glu': Gated Linear Unit with sigmoid gating
            - 'geglu': GELU-based Gated Linear Unit (GeGLU) Feed Forward Network
            - 'residual': Residual block with skip connections
            - 'swin_mlp': Swin Transformer MLP variant
        ffn_args: Optional dictionary of custom arguments for FFN layer.
            Ignored if `moe_config` is provided.
        moe_config: Optional[Union[MoEConfig, Dict]], configuration for a Mixture of Experts layer.
            If provided, this will replace the standard FFN block with an MoE layer.
            The `ffn_type`, `ffn_args`, and `intermediate_size` parameters will be ignored.
            Can be either an MoEConfig instance or a dictionary that will be converted to MoEConfig.
            Defaults to None.
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
        window_size: Integer, window size for window attention. Defaults to 8.
        n_kv_head: Integer, number of key-value heads for grouped query attention.
            Defaults to None (uses num_heads).
        lambda_init: Float, initial lambda value for differential attention.
            Only used when attention_type='differential_attention'. Defaults to 0.8.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`

    Example:
        ```python
        # Standard transformer layer with default normalization
        inputs = keras.Input(shape=(128, 768))
        layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            normalization_position='pre',
            ffn_type='swiglu',
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1
        )
        outputs = layer(inputs)

        # Using custom normalization arguments for DynamicTanh
        layer_with_custom_norm = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            normalization_type='dynamic_tanh',
            attention_norm_args={'alpha_init_value': 0.7, 'axis': [-1]},
            ffn_norm_args={'alpha_init_value': 0.15, 'axis': [-1]}
        )

        # Using Band RMS with custom constraints
        layer_with_band_rms = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            normalization_type='band_rms',
            attention_norm_args={'max_band_width': 0.1, 'epsilon': 1e-7},
            ffn_norm_args={'max_band_width': 0.05, 'epsilon': 1e-8}
        )

        # Using a Mixture of Experts layer with SwiGLU experts
        from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

        moe_config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "swiglu",      # Use SwiGLU for the experts
                    "output_dim": 768,     # Should match TransformerLayer's hidden_size
                    "ffn_expansion_factor": 4
                }
            ),
            gating_config=GatingConfig(top_k=2)
        )

        moe_layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,  # Ignored when moe_config is provided
            moe_config=moe_config
        )
        outputs_moe = moe_layer(inputs)
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
            attention_norm_args: Optional[Dict[str, Any]] = None,
            ffn_norm_args: Optional[Dict[str, Any]] = None,
            ffn_type: FFNType = 'mlp',
            ffn_args: Optional[Dict[str, Any]] = None,
            moe_config: Optional[Union[MoEConfig, Dict[str, Any]]] = None,
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
            window_size: int = 8,
            n_kv_head: Optional[int] = None,
            lambda_init: float = 0.8,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Input Validation (early) ---
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if intermediate_size <= 0 and moe_config is None:
            raise ValueError(
                f"intermediate_size must be positive when moe_config is None, got {intermediate_size}"
            )

        # --- Configuration Storage ---
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_type = attention_type
        self.attention_args = attention_args or {}
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_norm_args = attention_norm_args or {}
        self.ffn_norm_args = ffn_norm_args or {}
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args or {}
        self.moe_config = moe_config
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.window_size = window_size
        self.n_kv_head = n_kv_head if n_kv_head is not None else num_heads
        self.lambda_init = lambda_init

        # --- Handle MoE Configuration ---
        # Convert dict to MoEConfig if needed
        if isinstance(self.moe_config, dict):
            self.moe_config = MoEConfig.from_dict(self.moe_config)

        # Validate and adjust MoE config if provided
        if self.moe_config is not None:
            # Issue warning about ignored parameters
            if self.ffn_type != 'mlp' or self.ffn_args:
                warnings.warn(
                    "moe_config is provided, so `ffn_type`, `ffn_args`, and `intermediate_size` "
                    "parameters of TransformerLayer will be ignored. The FFN will be a "
                    "MixtureOfExperts layer configured by `moe_config`."
                )

            ffn_config = self.moe_config.expert_config.ffn_config

            # Ensure expert output_dim matches transformer's hidden_size
            if 'output_dim' in ffn_config and ffn_config['output_dim'] != self.hidden_size:
                warnings.warn(
                    f"Adjusting moe_config.expert_config.ffn_config['output_dim'] from "
                    f"{ffn_config['output_dim']} to {self.hidden_size} "
                    f"to match TransformerLayer's hidden_size for consistency."
                )
                ffn_config['output_dim'] = self.hidden_size
            elif 'output_dim' not in ffn_config:
                ffn_config['output_dim'] = self.hidden_size

            # If expert_config is for an MLP-like FFN and doesn't have its intermediate size set,
            # use TransformerLayer's intermediate_size as a sensible default.
            ffn_type = ffn_config.get('type')
            if ffn_type in ['mlp', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']:
                if 'hidden_dim' not in ffn_config:
                    ffn_config['hidden_dim'] = self.intermediate_size

        # --- Create Sub-layers (unbuilt) ---
        # Following best practices: all sub-layers created in __init__

        # Normalization layers
        self.attention_norm = self._create_normalization_layer('attention_norm', 'attention')
        self.output_norm = self._create_normalization_layer('output_norm', 'ffn')

        # Attention layer
        self.attention = self._create_attention_layer('attention')

        # Feed-forward network (or MoE)
        self.ffn_layer = self._create_ffn_layer('ffn')

        # Dropout layers
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
        """
        Create a normalization layer based on the specified type using the normalization factory.

        Args:
            name: Name for the layer.
            layer_type: Type of layer ('attention' or 'ffn') for parameter selection.

        Returns:
            Normalization layer instance.

        Raises:
            ValueError: If normalization layer creation fails due to invalid parameters.
        """
        # Select the appropriate custom arguments based on layer type
        custom_args = self.attention_norm_args if layer_type == 'attention' else self.ffn_norm_args

        try:
            # Use the factory to create the layer with custom arguments
            return create_normalization_layer(
                normalization_type=self.normalization_type,
                name=name,
                **custom_args
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.normalization_type} normalization layer for {layer_type}. "
                f"Check parameter compatibility. Custom args: {list(custom_args.keys())}. "
                f"Original error: {e}"
            )

    def _get_attention_params(self, name: str) -> Dict[str, Any]:
        """
        Get parameters for attention layer creation, merging defaults with custom args.

        Args:
            name: Name for the attention layer.

        Returns:
            Dictionary of parameters for attention layer creation.
        """
        # Define default parameters for each attention type
        if self.attention_type == 'multi_head_attention':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
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
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'num_kv_heads': self.n_kv_head,
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'name': name
            }
        elif self.attention_type == 'differential_attention':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'head_dim': self.hidden_size // self.num_heads,
                'dropout_rate': self.dropout_rate,
                'lambda_init': self.lambda_init,
                'name': name
            }
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        # Merge with custom arguments, giving priority to custom args
        return {**default_params, **self.attention_args}

    def _create_attention_layer(self, name: str) -> keras.layers.Layer:
        """
        Create an attention layer based on the specified type.

        Args:
            name: Name for the attention layer.

        Returns:
            Attention layer instance.

        Raises:
            ValueError: If attention layer creation fails due to invalid parameters.
        """
        params = self._get_attention_params(name)
        try:
            if self.attention_type == 'multi_head_attention':
                return MultiHeadAttention(**params)
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

    def _get_ffn_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration dictionary for FFN layer creation via factory.

        Args:
            name: Name for the FFN layer.

        Returns:
            Dictionary of parameters for FFN layer creation.
        """
        config = {
            'type': self.ffn_type,
            'name': name,
            'dropout_rate': self.dropout_rate
        }

        # Map TransformerLayer's generic parameters to FFN-specific ones
        if self.ffn_type == 'swiglu':
            config.update({
                'output_dim': self.hidden_size,
                'ffn_expansion_factor': 4,
                'ffn_multiple_of': 256,
            })
        elif self.ffn_type == 'differential':
            config.update({
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'branch_activation': self.activation,
            })
        elif self.ffn_type in ['mlp', 'glu', 'geglu', 'residual', 'swin_mlp']:
            config.update({
                'hidden_dim': self.intermediate_size,
                'output_dim': self.hidden_size,
                'activation': self.activation,
            })

        # User-provided args override everything.
        config.update(self.ffn_args)
        return config

    def _create_ffn_layer(self, name: str) -> keras.layers.Layer:
        """
        Create a feed-forward network layer based on the specified type or MoE config.

        Args:
            name: Name for the FFN layer.

        Returns:
            FFN or MoE layer instance.

        Raises:
            ValueError: If FFN layer creation fails due to invalid parameters.
        """
        # If MoE config is provided, create a MixtureOfExperts layer
        if self.moe_config is not None:
            return MixtureOfExperts(config=self.moe_config, name=name)

        # Otherwise, create standard FFN using the factory
        config = self._get_ffn_config(name)
        try:
            return create_ffn_from_config(config)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.ffn_type} layer. "
                f"Check for parameter incompatibility. Custom args: {list(self.ffn_args.keys())}. "
                f"Original error: {e}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all sub-layers with appropriate shapes.

        CRITICAL: For composite layers with sub-layers, explicitly build each sub-layer
        for robust serialization per Modern Keras 3 best practices.

        Args:
            input_shape: Shape tuple of input tensor.

        Raises:
            ValueError: If input shape is invalid.
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}")
        if input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        # Build normalization layers
        self.attention_norm.build(input_shape)
        self.output_norm.build(input_shape)

        # Build attention layer
        self.attention.build(input_shape)

        # Build FFN/MoE layer
        self.ffn_layer.build(input_shape)

        # Build dropout layers (no-op but for completeness)
        self.dropout.build(input_shape)
        self.attention_dropout.build(input_shape)

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
        """
        Forward pass of the transformer layer.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask: Optional attention mask tensor. Shape depends on attention type:
                - For padding mask: (batch_size, sequence_length)
                - For causal mask: (sequence_length, sequence_length)
                - For full mask: (batch_size, num_heads, sequence_length, sequence_length)
            layer_idx: Layer index for certain attention types (e.g., differential attention).
            training: Boolean flag for training mode.

        Returns:
            Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        residual = inputs

        # Process attention mask for MultiHeadAttention if needed
        mha_attention_mask = attention_mask
        if self.attention_type == 'multi_head_attention' and mha_attention_mask is not None:
            mask_shape = keras.ops.shape(mha_attention_mask)
            ndim = len(mask_shape)

            if ndim == 2:
                # Check if it's a padding mask (batch_size, seq_len) or causal mask (seq_len, seq_len)
                # If batch_size != seq_len, it's definitely a padding mask
                # If they're equal, assume it's a causal mask (more common in transformers)
                if mask_shape[0] != mask_shape[1]:
                    # Padding mask: expand to (batch_size, 1, 1, seq_len)
                    mha_attention_mask = mha_attention_mask[:, None, None, :]

        if self.normalization_position == 'pre':
            # --- Pre-Normalization: Normalize -> SubLayer -> StochasticDepth -> Add ---

            # 1. Attention block
            x = self.attention_norm(inputs, training=training)

            # Apply attention based on type
            if self.attention_type == 'multi_head_attention':
                x = self.attention(
                    x,
                    attention_mask=mha_attention_mask,
                    training=training
                )
            elif self.attention_type == 'differential_attention':
                x = self.attention(x, attention_mask=attention_mask, layer_idx=layer_idx, training=training)
            else:
                # Assume custom layers handle masks internally if needed
                x = self.attention(x, training=training)

            # Apply dropout
            x = self.attention_dropout(x, training=training)

            # Apply stochastic depth if enabled
            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)

            # Add residual
            attention_output = x + residual

            # 2. FFN block
            residual = attention_output
            x = self.output_norm(attention_output, training=training)
            x = self.ffn_layer(x, training=training)

            # Apply dropout
            x = self.dropout(x, training=training)

            # Apply stochastic depth if enabled
            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            # Add residual
            layer_output = x + residual

        else:  # Post-normalization
            # --- Post-Normalization: SubLayer -> StochasticDepth -> Add -> Normalize ---

            # 1. Attention block
            # Apply attention based on type
            if self.attention_type == 'multi_head_attention':
                x = self.attention(
                    inputs,
                    attention_mask=mha_attention_mask,
                    training=training
                )
            elif self.attention_type == 'differential_attention':
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    training=training)
            else:
                x = self.attention(inputs, training=training)

            # Apply dropout
            x = self.attention_dropout(x, training=training)

            # Apply stochastic depth if enabled
            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)

            # Add and normalize
            attention_output = self.attention_norm(x + residual, training=training)

            # 2. FFN block
            residual = attention_output
            x = self.ffn_layer(attention_output, training=training)

            # Apply dropout
            x = self.dropout(x, training=training)

            # Apply stochastic depth if enabled
            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)

            # Add and normalize
            layer_output = self.output_norm(x + residual, training=training)

        return layer_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Shape tuple of output tensor (same as input for transformer layer).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns all __init__ parameters for proper reconstruction.

        Returns:
            Configuration dictionary with all parameters.
        """
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'attention_type': self.attention_type,
            'attention_args': self.attention_args,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'attention_norm_args': self.attention_norm_args,
            'ffn_norm_args': self.ffn_norm_args,
            'ffn_type': self.ffn_type,
            'ffn_args': self.ffn_args,
            'moe_config': self.moe_config.to_dict() if self.moe_config else None,
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
            'window_size': self.window_size,
            'n_kv_head': self.n_kv_head,
            'lambda_init': self.lambda_init,
        })
        return config