"""
A self-attention mechanism for computer vision tasks,
based on the influential paper "Non-local Neural Networks".

Standard convolutional layers operate on a small, local neighborhood of pixels,
limiting their receptive field. In contrast, this Non-local Attention layer captures
long-range dependencies by computing the response at a position as a weighted sum of
features at *all* positions in the input feature map. This allows the model to
build relationships between distant pixels, which is crucial for understanding
complex scenes and objects.

It functions as a self-attention block tailored for 4D image-like tensors
(batch, height, width, channels).

Score-to-probability conversion is delegated to :class:`ProbabilityOutput`
via ``probability_type`` / ``probability_config``. Optional QK-normalization
(``qk_norm_type``) applies a normalization layer independently to the query
and key projections before computing attention scores. The optional output
spatial normalization (``output_norm_type``) is created via
:func:`create_normalization_layer` and accepts the full set of registered
normalization types.

References:
    - Wang, X., Girshick, R., Gupta, A., & He, K. (2018). "Non-local Neural
      Networks". (https://arxiv.org/abs/1711.07971)
"""

# ---------------------------------------------------------------------

import keras
from keras import ops
from typing import Any, Dict, Tuple, Optional, Literal, Union

from ..activations import ProbabilityOutput, resolve_activation_layer
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NonLocalAttention(keras.layers.Layer):
    """Non-local self-attention layer for capturing long-range spatial dependencies.

    Implements the self-attention mechanism from "Non-local Neural Networks"
    (Wang et al., 2018) that enables convolutional networks to capture global
    spatial dependencies by computing attention between all spatial positions in
    a 4D feature map. The input is first spatially pre-processed with an optional
    depthwise convolution and (optional) output spatial normalization, then projected
    into query, key, and value representations via 1x1 convolutions. The spatial
    dimensions are flattened into sequences for attention computation:
    ``score = Q K^T`` followed by ``attn = ProbabilityOutput(score)``, then
    ``out = attn @ V``. In ``'dot_product'`` mode, scores are scaled by
    ``1/sqrt(d_k)`` (matching the previous behavior of ``use_scale=True``); in
    ``'gaussian'`` mode no scaling is applied (matching the previous
    ``use_scale=False``) and the key/value channels are reduced to ``d_attn / 8``
    as in the original paper. The attended output is reshaped back to spatial
    format and projected to the desired output channels.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────┐
        │   Input [B, H, W, C]        │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ DepthwiseConv2D(kernel_size)│
        │ + OutputNorm (optional)     │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  1x1 Conv2D Projections     │
        ├─────────┬─────────┬─────────┤
        │ Q_proj  │ K_proj  │ V_proj  │
        └────┬────┴────┬────┴────┬────┘
             ▼         ▼         ▼
        ┌────────────────────────────┐
        │ Reshape (H,W) → (H*W)      │
        │ Q [B, H*W, d_attn]         │
        │ K [B, H*W, d_kv]           │
        │ V [B, H*W, d_kv]           │
        └─────────────┬──────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ (optional) q_norm / k_norm  │
        │ scores = Q K^T (/ sqrt(d_k))│
        │ attn = ProbabilityOutput(.) │
        │ out  = attn @ V             │
        │ → [B, H*W, d_kv]            │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ Reshape → [B, H, W, d_kv]   │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ Output Conv2D 1x1           │
        │ + Dropout (optional)        │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ Output [B, H, W, out_ch]    │
        └─────────────────────────────┘

    :param attention_channels: Number of channels in the attention mechanism.
        Must be positive.
    :type attention_channels: int
    :param kernel_size: Size of the depthwise convolution kernel.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param use_bias: Whether to use bias in convolution layers.
    :type use_bias: bool
    :param probability_type: Probability strategy identifier forwarded to
        :class:`ProbabilityOutput` for converting attention scores into
        probabilities. Score-level routing strategies are rejected because the
        attention probabilities must sum to 1 over the key axis.
    :type probability_type: str
    :param probability_config: Optional configuration dictionary forwarded
        to :class:`ProbabilityOutput` as its ``type_config`` argument.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied independently
        to the query and key projections before score computation, instantiated
        via :func:`create_normalization_layer`. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when ``qk_norm_type`` is set.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param output_norm_type: Type of spatial normalization applied to the
        depthwise pre-processed features, instantiated via
        :func:`create_normalization_layer`. Defaults to ``'batch_norm'`` to
        preserve the previous default behavior (``normalization='batch'``).
        Pass ``None`` to disable.
    :type output_norm_type: Optional[str]
    :param output_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when ``output_norm_type`` is set.
    :type output_norm_kwargs: Optional[Dict[str, Any]]
    :param intermediate_activation: Activation function for intermediate layers.
    :type intermediate_activation: Union[str, callable]
    :param output_activation: Activation function for the output projection.
    :type output_activation: Union[str, callable]
    :param output_channels: Number of output channels (``-1`` to match input).
    :type output_channels: int
    :param dropout_rate: Dropout rate between 0.0 and 1.0.
    :type dropout_rate: float
    :param attention_mode: Attention type (``'gaussian'`` or ``'dot_product'``).
        ``'dot_product'`` scales scores by ``1/sqrt(d_k)``; ``'gaussian'`` does
        not scale and uses reduced key/value channels.
    :type attention_mode: Literal['gaussian', 'dot_product']
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Optional regularizer for layer activity.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer parent class.
    :type kwargs: Any

    :raises ValueError: If ``attention_channels <= 0``.
    :raises ValueError: If ``dropout_rate`` not in ``[0, 1)``.
    :raises ValueError: If ``attention_mode`` not in ``['gaussian', 'dot_product']``.
    :raises ValueError: If ``probability_type`` is a score-level routing strategy
        (``'routing'``, ``'deterministic_routing'``, ``'hierarchical'``,
        ``'hierarchical_routing'``).
    """

    def __init__(
        self,
        attention_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (7, 7),
        use_bias: bool = False,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        output_norm_type: Optional[str] = "batch_norm",
        output_norm_kwargs: Optional[Dict[str, Any]] = None,
        intermediate_activation: Union[str, callable] = 'relu',
        intermediate_activation_args: Optional[Dict[str, Any]] = None,
        output_activation: Union[str, callable] = 'linear',
        output_activation_args: Optional[Dict[str, Any]] = None,
        output_channels: int = -1,
        dropout_rate: float = 0.0,
        attention_mode: Literal['gaussian', 'dot_product'] = 'gaussian',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(
            attention_channels, dropout_rate, attention_mode, probability_type
        )

        # Store ALL configuration parameters
        self.attention_channels = attention_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.use_bias = use_bias
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
        self.output_norm_type = output_norm_type
        self.output_norm_kwargs = output_norm_kwargs
        self.intermediate_activation = intermediate_activation
        self.intermediate_activation_args = intermediate_activation_args
        self.output_activation = output_activation
        self.output_activation_args = output_activation_args
        self.output_channels = output_channels
        self.dropout_rate = dropout_rate
        self.attention_mode = attention_mode

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Common convolution parameters for reuse
        self._conv_params = {
            'kernel_size': (1, 1),
            'strides': (1, 1),
            'padding': 'same',
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer
        }

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.use_bias,
            depthwise_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name='depthwise_conv'
        )

        # Intermediate activation routed through the activation factory.
        # Two instances since they are applied to differently-shaped tensors
        # (depthwise output and key projection output).
        self.depthwise_activation = resolve_activation_layer(
            self.intermediate_activation,
            name='depthwise_activation',
            **(self.intermediate_activation_args or {}),
        )
        self.key_activation = resolve_activation_layer(
            self.intermediate_activation,
            name='key_activation',
            **(self.intermediate_activation_args or {}),
        )

        # Create spatial output normalization layer if specified
        if self.output_norm_type is not None:
            self.output_norm = create_normalization_layer(
                self.output_norm_type,
                name='output_norm',
                **(self.output_norm_kwargs or {}),
            )
        else:
            self.output_norm = None

        # Create Query, Key, Value projection layers
        self.query_conv = keras.layers.Conv2D(
            filters=self.attention_channels,
            name='query_conv',
            **self._conv_params
        )

        # Adjust key/value channels based on attention mode.
        # Gaussian mode uses fewer channels as per original paper.
        self.key_value_channels = (
            self.attention_channels
            if self.attention_mode == 'dot_product'
            else self.attention_channels // 8
        )

        self.key_conv = keras.layers.Conv2D(
            filters=self.key_value_channels,
            name='key_conv',
            **self._conv_params
        )

        self.value_conv = keras.layers.Conv2D(
            filters=self.key_value_channels,
            name='value_conv',
            **self._conv_params
        )

        # Probability layer for converting attention scores to weights
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name='attn_prob',
        )

        # Optional QK-normalization layers
        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type,
                name='q_norm',
                **(self.qk_norm_kwargs or {}),
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type,
                name='k_norm',
                **(self.qk_norm_kwargs or {}),
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Attention dropout (applied to attention probabilities)
        if self.dropout_rate > 0:
            self.attn_dropout = keras.layers.Dropout(
                self.dropout_rate, name='attn_dropout'
            )
            self.dropout = keras.layers.Dropout(
                self.dropout_rate, name='dropout'
            )
        else:
            self.attn_dropout = None
            self.dropout = None

        # Note: output_conv will be created in build() since we need input channels

    def _validate_inputs(
        self,
        attention_channels: int,
        dropout_rate: float,
        attention_mode: str,
        probability_type: str,
    ) -> None:
        """Validate initialization parameters.

        :raises ValueError: If any parameter is invalid.
        """
        if attention_channels <= 0:
            raise ValueError(f"attention_channels must be positive, got {attention_channels}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if attention_mode not in ['gaussian', 'dot_product']:
            raise ValueError(f"attention_mode must be 'gaussian' or 'dot_product', got {attention_mode}")
        invalid_prob_types = {
            "routing", "deterministic_routing",
            "hierarchical", "hierarchical_routing",
        }
        if probability_type in invalid_prob_types:
            raise ValueError(
                f"Invalid probability_type '{probability_type}'. Score-level "
                f"routing strategies are not compatible with attention "
                f"probabilities that must sum to 1 over the key axis."
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for robust serialization."""
        channels = input_shape[-1]
        actual_output_channels = (
            channels if self.output_channels <= 0
            else self.output_channels
        )

        # Create output projection layer (needs input channels)
        self.output_conv = keras.layers.Conv2D(
            filters=actual_output_channels,
            name='output_conv',
            **self._conv_params
        )
        # Output activation routed through the activation factory.
        self.output_activation_layer = resolve_activation_layer(
            self.output_activation,
            name='output_activation',
            **(self.output_activation_args or {}),
        )

        # Build sub-layers in computational order for serialization robustness
        self.depthwise_conv.build(input_shape)
        self.depthwise_activation.build(input_shape)

        # Depthwise conv doesn't change shape, so output_norm uses same shape
        if self.output_norm is not None:
            self.output_norm.build(input_shape)

        # Query, Key, Value projections all use the same input shape
        self.query_conv.build(input_shape)
        self.key_conv.build(input_shape)
        self.value_conv.build(input_shape)
        key_output_shape = (
            input_shape[0], input_shape[1], input_shape[2], self.key_value_channels
        )
        self.key_activation.build(key_output_shape)

        batch_size = input_shape[0] if input_shape[0] is not None else 1
        height = input_shape[1] if input_shape[1] is not None else 32
        width = input_shape[2] if input_shape[2] is not None else 32
        seq_len = height * width if (input_shape[1] is not None and input_shape[2] is not None) else None

        q_seq_shape = (input_shape[0], seq_len, self.attention_channels)
        kv_seq_shape = (input_shape[0], seq_len, self.key_value_channels)

        if self.q_norm is not None:
            self.q_norm.build(q_seq_shape)
            self.k_norm.build(kv_seq_shape)

        # Attention scores shape: (B, N_q, N_k)
        attn_scores_shape = (input_shape[0], seq_len, seq_len)
        self.attn_prob.build(attn_scores_shape)

        if self.attn_dropout is not None:
            self.attn_dropout.build(attn_scores_shape)

        # Output conv processes the attention output
        attention_output_shape = (input_shape[0], input_shape[1], input_shape[2], self.key_value_channels)
        self.output_conv.build(attention_output_shape)
        output_conv_shape = (
            input_shape[0], input_shape[1], input_shape[2], actual_output_channels
        )
        self.output_activation_layer.build(output_conv_shape)

        if self.dropout is not None:
            self.dropout.build(attention_output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Apply non-local attention to input features.

        :param inputs: Input tensor of shape ``(batch_size, height, width, channels)``.
        :param attention_mask: Optional additive attention mask broadcastable to
            attention scores of shape ``(B, N_q, N_k)``. A value of ``0`` keeps a
            position; a large negative value masks it out.
        :param training: Whether in training mode. Affects dropout and normalization.

        :return: Tensor of shape ``(batch_size, height, width, output_channels)``
            with spatially attended features incorporating long-range dependencies.
        """
        # Apply depthwise convolution for spatial processing
        x = self.depthwise_conv(inputs, training=training)
        x = self.depthwise_activation(x, training=training)

        # Apply spatial output normalization if specified
        if self.output_norm is not None:
            x = self.output_norm(x, training=training)

        # Generate query, key, value projections
        query = self.query_conv(x, training=training)
        key = self.key_conv(x, training=training)
        key = self.key_activation(key, training=training)
        value = self.value_conv(x, training=training)

        # Reshape for attention computation: (B, H, W, C) -> (B, H*W, C)
        shape = ops.shape(query)
        batch_size, height, width = shape[0], shape[1], shape[2]

        q = ops.reshape(query, [batch_size, -1, self.attention_channels])
        k = ops.reshape(key, [batch_size, -1, self.key_value_channels])
        v = ops.reshape(value, [batch_size, -1, self.key_value_channels])

        # Optional QK-normalization
        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        # Scaled dot-product attention scores: (B, N_q, N_k)
        scores = ops.matmul(q, ops.transpose(k, axes=[0, 2, 1]))
        if self.attention_mode == 'dot_product':
            # Match previous behavior of keras.layers.Attention(use_scale=True)
            d_k = ops.cast(self.attention_channels, scores.dtype)
            scores = scores / ops.sqrt(d_k)
        # In 'gaussian' mode, no scaling (matches previous use_scale=False)

        # Optional additive attention mask
        if attention_mask is not None:
            scores = scores + ops.cast(attention_mask, scores.dtype)

        # Convert scores to attention probabilities
        attn = self.attn_prob(scores, training=training)

        # Optional dropout on attention probabilities
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Aggregate values: (B, N_q, N_k) @ (B, N_k, d_kv) -> (B, N_q, d_kv)
        attention_output = ops.matmul(attn, v)

        # Reshape back to spatial dimensions: (B, H*W, C) -> (B, H, W, C)
        attention_output = ops.reshape(
            attention_output,
            [batch_size, height, width, self.key_value_channels]
        )

        # Apply output projection + activation
        output = self.output_conv(attention_output, training=training)
        output = self.output_activation_layer(output, training=training)

        # Apply output dropout if specified
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        output_shape = list(input_shape)
        if self.output_channels > 0:
            output_shape[-1] = self.output_channels
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'attention_channels': self.attention_channels,
            'kernel_size': self.kernel_size,
            'use_bias': self.use_bias,
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
            'output_norm_type': self.output_norm_type,
            'output_norm_kwargs': self.output_norm_kwargs,
            'intermediate_activation': self.intermediate_activation,
            'intermediate_activation_args': self.intermediate_activation_args,
            'output_activation': self.output_activation,
            'output_activation_args': self.output_activation_args,
            'output_channels': self.output_channels,
            'dropout_rate': self.dropout_rate,
            'attention_mode': self.attention_mode,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
