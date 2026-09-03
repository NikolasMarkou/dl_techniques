"""Self-attention over 4D image feature maps, built by ``NonLocalAttention``.

Standard convolutions see only a local neighborhood; this layer computes the
response at each position as a weighted sum of features at every other
position, following "Non-local Neural Networks" (Wang et al., 2018). The
input first passes through a depthwise convolution and optional
normalization, giving each token a local receptive field before any global
mixing. Query, key and value are 1x1 convolutions flattened to sequences of
length H*W, so the attention matrix is quadratic in spatial resolution. In
``'dot_product'`` mode scores are scaled by ``1/sqrt(d_k)``; in
``'gaussian'`` mode they are unscaled and the query/key/value channels are
reduced to ``attention_channels // 8``, matching the paper's embedded-
Gaussian instantiation.

``output_conv`` and ``output_activation_layer`` are built lazily in
``build()`` rather than ``__init__``, the layer's one exception to eager
sub-layer creation: the default ``output_channels=-1`` (match input) cannot
resolve a filter count before the input shape is known.

References:
    - Wang et al., 2018. Non-local Neural Networks.
      (https://arxiv.org/abs/1711.07971)
"""

import keras
import numpy as np
from typing import Any, Dict, Tuple, Optional, Literal, Union

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.activations import ProbabilityOutput, resolve_activation_layer
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

from .common import compute_attention_scale, mask_dtype
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.non_local_attention")
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
    ``use_scale=False``) and the query, key, and value channels are reduced to
    ``d_attn / 8`` (clamped to ``>=1``) as in the original paper. The attended
    output is reshaped back to spatial
    format and projected to the desired output channels.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
             │
             ▼
        DepthwiseConv2D(kernel_size) -> activation
             │
             ▼
        output_norm (optional, spatial)
             │
             ▼
        1x1 Conv2D projections
             ├─► Q  [B, H, W, d_kv]
             ├─► K  [B, H, W, d_kv] -> its own activation
             └─► V  [B, H, W, d_kv]
             │
             ▼
        flatten (H, W) -> H*W tokens   [B, H*W, d_kv]
             │
             ▼
        q_norm / k_norm (optional)
             │
             ▼
        scores = Q Kᵀ  [B, H*W, H*W]
          * 1/sqrt(d_kv) only in 'dot_product' mode
             │
             ▼
        + attention_mask (additive, optional, clamped before cast)
             │
             ▼
        attn = attn_prob(scores) -> attn dropout (optional)
             │
             ▼
        out = attn @ V  [B, H*W, d_kv] -> reshape to (H, W)
             │
             ▼
        output Conv2D 1x1 -> output activation -> dropout (optional)
             │
             ▼
        output [B, H, W, output_channels]

    A single flat (H*W) x (H*W) attention map is computed; there is no
    multi-head split.

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
    :param intermediate_activation_args: Optional keyword arguments forwarded to
        the intermediate activation layer's constructor. Two independent
        instances are built from it (depthwise output and key projection output)
        because they are applied to differently-shaped tensors.
    :type intermediate_activation_args: Optional[Dict[str, Any]]
    :param output_activation: Activation function for the output projection.
    :type output_activation: Union[str, callable]
    :param output_activation_args: Optional keyword arguments forwarded to the
        output activation layer's constructor.
    :type output_activation_args: Optional[Dict[str, Any]]
    :param output_channels: Number of output channels (``-1`` to match input).
    :type output_channels: int
    :param dropout_rate: Dropout rate between 0.0 and 1.0.
    :type dropout_rate: float
    :param attention_mode: Attention type (``'gaussian'`` or ``'dot_product'``).
        Wang et al. 2018 defines four pairwise functions (gaussian, embedded
        gaussian, dot product, concatenation); only two are implemented here,
        and neither matches its name exactly. ``'gaussian'`` is the paper's
        embedded gaussian: scores come from learned theta/phi embeddings and
        are softmax-normalized. ``'dot_product'`` is a softmax-normalized
        scaled dot product, not the paper's ``1/N``-normalized variant. Both
        modes use the same default softmax :class:`ProbabilityOutput`, so the
        only functional difference is scaling: ``'dot_product'`` scales
        scores by ``1/sqrt(d_k)``; ``'gaussian'`` does not scale and uses
        reduced key/value channels.
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
        """Validate the configuration and create every eagerly-buildable sub-layer.

        Every argument is documented on the class. ``output_conv`` and
        ``output_activation_layer`` are the two exceptions: they stay ``None``
        here because their filter count is only known in :meth:`build`.
        """
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(
            attention_channels, dropout_rate, attention_mode, probability_type
        )

        # Store ALL configuration parameters
        self.attention_channels = attention_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.use_bias = use_bias
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
        self.output_norm_type = output_norm_type
        self.output_norm_kwargs = output_norm_kwargs
        self.intermediate_activation = deserialize_activation(intermediate_activation)
        self.intermediate_activation_args = intermediate_activation_args
        self.output_activation = deserialize_activation(output_activation)
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

        # DECISION plan_2026-06-14_adaddf34/D-002: reduce Q, K and V together to
        # one shared embedded dim in gaussian mode -- Q@Kᵀ needs a matched
        # contraction dim. See decisions.md.
        self.key_value_channels = (
            self.attention_channels
            if self.attention_mode == 'dot_product'
            else max(1, self.attention_channels // 8)
        )
        # Scale is precomputed here, not in call(), per the standing anchor
        # plan_2026-06-14_33b77a7a/D-002; see common.py.
        self._inv_sqrt_kv = compute_attention_scale(self.key_value_channels)

        # Create Query, Key, Value projection layers (all share the embedded dim)
        self.query_conv = keras.layers.Conv2D(
            filters=self.key_value_channels,
            name='query_conv',
            **self._conv_params
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

        # DECISION plan_2026-06-14_0c5d4a21/D-003: output_conv and
        # output_activation_layer stay None sentinels here, built in build()
        # instead, since output_channels=-1 needs the runtime input channel
        # count. See decisions.md.
        self.output_conv = None
        self.output_activation_layer = None

    def _validate_inputs(
        self,
        attention_channels: int,
        dropout_rate: float,
        attention_mode: str,
        probability_type: str,
    ) -> None:
        """Validate initialization parameters.

        :param attention_channels: Channel count of the attention mechanism.
        :type attention_channels: int
        :param dropout_rate: Dropout rate to validate.
        :type dropout_rate: float
        :param attention_mode: Attention mode name to validate.
        :type attention_mode: str
        :param probability_type: Probability strategy identifier to validate.
        :type probability_type: str

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
        """Build the layer and all sub-layers for robust serialization.

        This is also where ``output_conv`` and ``output_activation_layer`` are
        finally created: their filter count defaults to the runtime input
        channel count, which is only known here.

        :param input_shape: Shape tuple of the 4-D input ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # A second build() must be a no-op: the child .build() calls below are
        # not self-guarded and would raise on an already-built layer. See D-003.
        if self.built:
            return

        channels = input_shape[-1]
        actual_output_channels = (
            channels if self.output_channels <= 0
            else self.output_channels
        )

        # output_conv/output_activation_layer are built here, not __init__; see D-003.
        if self.output_conv is None:
            self.output_conv = keras.layers.Conv2D(
                filters=actual_output_channels,
                name='output_conv',
                **self._conv_params
            )
        if self.output_activation_layer is None:
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

        q_seq_shape = (input_shape[0], seq_len, self.key_value_channels)
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
        shape = keras.ops.shape(query)
        batch_size, height, width = shape[0], shape[1], shape[2]

        q = keras.ops.reshape(query, [batch_size, -1, self.key_value_channels])
        k = keras.ops.reshape(key, [batch_size, -1, self.key_value_channels])
        v = keras.ops.reshape(value, [batch_size, -1, self.key_value_channels])

        # Optional QK-normalization
        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        # Scaled dot-product attention scores: (B, N_q, N_k)
        scores = keras.ops.matmul(q, keras.ops.transpose(k, axes=[0, 2, 1]))
        if self.attention_mode == 'dot_product':
            # DECISION plan_2026-06-14_33b77a7a/D-003: scale by the precomputed
            # 1/sqrt(key_value_channels); never apply it in gaussian mode. See decisions.md.
            scores = scores * self._inv_sqrt_kv
        # In 'gaussian' mode, no scaling (matches previous use_scale=False)

        # Optional additive attention mask
        if attention_mask is not None:
            # DECISION plan-2026-08-27T040114-580f8b63/D-015: clamp the mask in
            # mask_dtype before casting down -- -1e9 cast to float16 is -inf. See decisions.md.
            compute_floor = float(
                np.finfo(np.dtype(self.compute_dtype)).min
            ) / 2.0
            mask = keras.ops.cast(attention_mask, mask_dtype(self.compute_dtype))
            mask = keras.ops.maximum(mask, compute_floor)
            scores = scores + keras.ops.cast(mask, scores.dtype)

        # Convert scores to attention probabilities
        attn = self.attn_prob(scores, training=training)

        # Optional dropout on attention probabilities
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Aggregate values: (B, N_q, N_k) @ (B, N_k, d_kv) -> (B, N_q, d_kv)
        attention_output = keras.ops.matmul(attn, v)

        # Reshape back to spatial dimensions: (B, H*W, C) -> (B, H, W, C)
        attention_output = keras.ops.reshape(
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
        """Compute the output shape of the layer.

        The spatial dimensions are preserved. The channel count becomes
        ``output_channels`` unless that is ``-1``, in which case the input
        channel count is kept.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        if self.output_channels > 0:
            output_shape[-1] = self.output_channels
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary holding every parameter needed to recreate this
            layer.
        :rtype: Dict[str, Any]
        """
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
            'intermediate_activation': serialize_activation(self.intermediate_activation),
            'intermediate_activation_args': self.intermediate_activation_args,
            'output_activation': serialize_activation(self.output_activation),
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
