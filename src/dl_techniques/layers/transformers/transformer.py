"""
Foundational building block of a Transformer network, implementing a highly
configurable and serializable encoder/decoder layer.

This layer encapsulates the two primary sub-components of a standard Transformer
architecture: a multi-head self-attention mechanism and a position-wise
feed-forward network. Each sub-component is enclosed within a residual
connection followed by layer normalization, a crucial design pattern that
enables the stable training of deep sequential models.

**Intent**: To provide a robust, production-ready, and flexible Transformer
layer that serves as a fundamental building block for a wide range of sequence
modeling tasks. It is designed to be highly configurable, allowing for easy
swapping of attention, FFN, and normalization components for architectural
research and experimentation, while strictly adhering to modern Keras 3 best
practices for serialization and composite layer construction.

**Architecture**: The layer processes an input sequence through two main blocks,
with the exact data flow determined by `normalization_position`.

**1. Pre-Normalization (`normalization_position='pre'`)**:
```
Input
  |
  +-- Norm → Attention → Dropout → StochasticDepth --+
  |                                                       |
  +----------------------- Add ---------------------------+
                                |
  +-- Norm → FFN/MoE → Dropout → StochasticDepth --+
  |                                                     |
  +----------------------- Add -------------------------+
                                |
                              Output
```

**2. Post-Normalization (`normalization_position='post'`)**:
```
Input
  |
  +-- Attention → Dropout → StochasticDepth → Add → Norm --+
                                                                |
  +-------- FFN/MoE → Dropout → StochasticDepth → Add → Norm --+
                                                                    |
                                                                  Output
```

**Mathematical Operations**:
1.  **Multi-Head Self-Attention (MHSA)**:
    -   Computes context-aware representations using scaled dot-product attention:
        `Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V`
    -   Uses multiple "heads" to attend to different representational subspaces
        in parallel, enhancing the model's ability to capture complex relationships.

2.  **Position-wise Feed-Forward Network (FFN)**:
    -   Applies a non-linear transformation independently at each sequence position.
    -   Typically a two-layer MLP: `FFN(x) = activation(x @ W₁ + b₁) @ W₂ + b₂`
    -   This component can be replaced by more advanced structures like SwiGLU or
        a Mixture of Experts (MoE) layer.

**References**:
    - Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
    - Ba, J. L., et al. (2016). Layer Normalization. *arXiv preprint*.
    - Xiong, R., et al. (2020). On Layer Normalization in the Transformer
      Architecture. *ICML*. (Analysis of Pre-LN vs. Post-LN).
"""

import keras
import warnings
from keras import layers, initializers, regularizers
from typing import Optional, Union, Any, Dict, Tuple, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..moe import MixtureOfExperts, MoEConfig
from ..stochastic_depth import StochasticDepth
from ..ffn import create_ffn_from_config, FFNType
from ..attention import create_attention_layer, AttentionType
from ..norms import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

NormalizationPositionType = Literal['post', 'pre']


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerLayer(keras.layers.Layer):
    """
    Generic transformer layer with configurable attention, FFN, and normalization.

    This layer implements a standard transformer block with residual connections,
    optional stochastic depth, and extensive configurability for its core
    components.

    Args:
        hidden_size: Integer, hidden size of the layer. Must be positive.
        num_heads: Integer, number of attention heads. Must be positive and
            a divisor of `hidden_size`.
        intermediate_size: Integer, size of the intermediate (feed-forward)
            layer. Ignored if `moe_config` is provided. Must be positive.
        attention_type: The type of attention mechanism to use. Defaults to
            'multi_head'. See `dl_techniques.layers.attention` for options.
        attention_args: Optional dictionary of custom arguments for the
            attention layer, overriding default parameters.
        normalization_type: The type of normalization to use. Defaults to
            'layer_norm'. See `dl_techniques.layers.norms` for options.
        normalization_position: Position of normalization layers. 'post' for
            the original Transformer design, or 'pre' for improved stability
            in deep models. Defaults to 'post'.
        attention_norm_args: Optional dictionary of custom arguments for the
            attention normalization layer (e.g., `{'epsilon': 1e-5}`).
        ffn_norm_args: Optional dictionary of custom arguments for the FFN
            normalization layer.
        ffn_type: The type of feed-forward network to use. Ignored if
            `moe_config` is provided. Defaults to 'mlp'. See
            `dl_techniques.layers.ffn` for options.
        ffn_args: Optional dictionary of custom arguments for the FFN layer.
            Ignored if `moe_config` is provided.
        moe_config: Optional configuration for a Mixture of Experts layer.
            If provided, replaces the standard FFN block.
        dropout_rate: Float, dropout rate for the FFN output. Defaults to 0.1.
        attention_dropout_rate: Float, dropout rate for the attention output.
            Defaults to 0.1.
        use_stochastic_depth: Boolean, whether to use stochastic depth
            regularization. Defaults to False.
        stochastic_depth_rate: Float, drop path rate for stochastic depth.
            Only used if `use_stochastic_depth` is True. Defaults to 0.1.
        activation: String or callable, activation function for the FFN.
            Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Initializer for kernel weights. Defaults to
            'glorot_uniform'.
        bias_initializer: Initializer for bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        window_size: Integer, window size for windowed attention. Defaults to 8.
        n_kv_head: Optional integer for grouped query attention, specifying the
            number of key/value heads. Defaults to `num_heads`.
        lambda_init: Float, initial lambda for differential attention.
            Defaults to 0.8.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, hidden_size)`.

    Attributes:
        attention_norm: Normalization layer for the attention block.
        output_norm: Normalization layer for the FFN block.
        attention: The attention layer instance (e.g., MultiHeadSelfAttention).
        ffn_layer: The FFN layer instance (e.g., MLPBlock or MixtureOfExperts).
        dropout: Dropout layer for the FFN output.
        attention_dropout: Dropout layer for the attention output.
        attention_stochastic_depth: StochasticDepth for the attention block.
        ffn_stochastic_depth: StochasticDepth for the FFN block.

    Raises:
        ValueError: If `hidden_size`, `num_heads`, or `intermediate_size` are
            not positive, or if `hidden_size` is not divisible by `num_heads`.
        ValueError: If sub-layer creation fails due to incompatible parameters
            provided in `attention_args`, `ffn_args`, or `*_norm_args`.

    Example:
        ```python
        # Standard transformer layer with pre-normalization and SwiGLU FFN
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

        # Using custom normalization (e.g., Band RMS) with specific arguments
        layer_with_band_rms = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            normalization_type='band_rms',
            attention_norm_args={'max_band_width': 0.1, 'epsilon': 1e-7},
            ffn_norm_args={'max_band_width': 0.05, 'epsilon': 1e-8}
        )

        # Using a Mixture of Experts layer with SwiGLU experts
        from dl_techniques.layers.moe import MoEConfig, ExpertConfig

        moe_config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "swiglu",
                    "output_dim": 768,
                    "ffn_expansion_factor": 4
                }
            )
        )
        moe_layer = TransformerLayer(
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,  # This will be ignored
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
            attention_type: AttentionType = 'multi_head',
            attention_args: Optional[Dict[str, Any]] = None,
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
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

        if self.moe_config is not None:
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
        # Per Keras best practices, all sub-layers are created in __init__.
        # They will be built with their weights in the build() method.

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
        Create a normalization layer using the component factory.

        Args:
            name: Name for the layer.
            layer_type: Type of layer ('attention' or 'ffn') to select the
                correct custom arguments.

        Returns:
            An unbuilt normalization layer instance.

        Raises:
            ValueError: If layer creation fails due to invalid parameters.
        """
        custom_args = self.attention_norm_args if layer_type == 'attention' else self.ffn_norm_args
        try:
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
        Consolidate parameters for attention layer creation.

        Args:
            name: Name for the attention layer.

        Returns:
            Dictionary of parameters for the attention layer factory.
        """
        if self.attention_type == 'multi_head':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'name': name
            }
        elif self.attention_type == 'window':
            default_params = {
                'dim': self.hidden_size,
                'window_size': self.window_size,
                'num_heads': self.num_heads,
                'name': name
            }
        elif self.attention_type == 'group_query':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'num_kv_heads': self.n_kv_head,
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'name': name
            }
        elif self.attention_type == 'differential':
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

        # User-provided args override defaults
        return {**default_params, **self.attention_args}

    def _create_attention_layer(self, name: str) -> keras.layers.Layer:
        """
        Create an attention layer using the component factory.

        Args:
            name: Name for the attention layer.

        Returns:
            An unbuilt attention layer instance.

        Raises:
            ValueError: If layer creation fails due to invalid parameters.
        """
        params = self._get_attention_params(name)
        try:
            return create_attention_layer(
                attention_type=self.attention_type,
                **params
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.attention_type} layer. "
                f"Check for parameter incompatibility. Custom args: {list(self.attention_args.keys())}. "
                f"Original error: {e}"
            )

    def _get_ffn_config(self, name: str) -> Dict[str, Any]:
        """
        Consolidate configuration for FFN layer creation.

        Args:
            name: Name for the FFN layer.

        Returns:
            Dictionary of parameters for the FFN layer factory.
        """
        config = {
            'type': self.ffn_type,
            'name': name,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer
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

        # User-provided args override everything
        config.update(self.ffn_args)
        return config

    def _create_ffn_layer(self, name: str) -> keras.layers.Layer:
        """
        Create a feed-forward network or MoE layer.

        Args:
            name: Name for the FFN layer.

        Returns:
            An unbuilt FFN or MoE layer instance.

        Raises:
            ValueError: If layer creation fails due to invalid parameters.
        """
        if self.moe_config is not None:
            return MixtureOfExperts(config=self.moe_config, name=name)

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

        CRITICAL: For composite layers with sub-layers, explicitly building each
        sub-layer in this method is essential for robust serialization. It
        ensures that all weight variables are created before Keras attempts
        to restore saved weights, preventing "Layer not built" errors.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is invalid or incompatible.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}")
        if input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        # Build all sub-layers in computational order
        self.attention_norm.build(input_shape)
        self.output_norm.build(input_shape)
        self.attention.build(input_shape)
        self.ffn_layer.build(input_shape)
        self.dropout.build(input_shape)
        self.attention_dropout.build(input_shape)
        if self.attention_stochastic_depth is not None:
            self.attention_stochastic_depth.build(input_shape)
        if self.ffn_stochastic_depth is not None:
            self.ffn_stochastic_depth.build(input_shape)

        # Always call super().build() at the end
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
            inputs: Input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            attention_mask: Optional attention mask. Its shape depends on the
                attention mechanism and masking type (e.g., padding, causal).
            layer_idx: Integer index of the current layer, primarily used by
                attention types like 'differential'.
            training: Boolean flag for training mode, passed to dropout and
                normalization layers.

        Returns:
            Output tensor of shape `(batch_size, sequence_length, hidden_size)`.
        """
        residual = inputs

        if self.normalization_position == 'pre':
            # --- Pre-Normalization: Normalize -> SubLayer -> Add ---
            # 1. Attention block
            x = self.attention_norm(inputs, training=training)
            if self.attention_type == 'differential':
                x = self.attention(x, attention_mask=attention_mask, layer_idx=layer_idx, training=training)
            else:
                x = self.attention(x, attention_mask=attention_mask, training=training)
            x = self.attention_dropout(x, training=training)
            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)
            attention_output = x + residual

            # 2. FFN block
            residual = attention_output
            x = self.output_norm(attention_output, training=training)
            x = self.ffn_layer(x, training=training)
            x = self.dropout(x, training=training)
            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)
            layer_output = x + residual
        else:
            # --- Post-Normalization: SubLayer -> Add -> Normalize ---
            # 1. Attention block
            if self.attention_type == 'differential':
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    training=training)
            else:
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    training=training)
            x = self.attention_dropout(x, training=training)
            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)
            attention_output = self.attention_norm(x + residual, training=training)

            # 2. FFN block
            residual = attention_output
            x = self.ffn_layer(attention_output, training=training)
            x = self.dropout(x, training=training)
            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)
            layer_output = self.output_norm(x + residual, training=training)

        return layer_output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Shape tuple of the output tensor, which is the same as the input.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        This method returns a dictionary containing all constructor parameters,
        allowing the layer to be fully reconstructed from its configuration.
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

# ---------------------------------------------------------------------
