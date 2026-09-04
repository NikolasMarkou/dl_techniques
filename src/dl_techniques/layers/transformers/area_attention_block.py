"""A transformer-style block over 2D feature maps, built on area attention.

This is the yolo12 detector's attention stage, built by
:class:`AreaAttentionBlock`. It is a bare residual pair,
``x = inputs + attn(inputs)`` then ``x = x + mlp2(mlp1(x))``, where ``attn``
is :class:`~dl_techniques.layers.attention.area_attention.AreaAttention`
(multi-head self-attention over a ``(B, H, W, C)`` map, either global or
split into ``area`` contiguous groups) and ``mlp1``/``mlp2`` are a
1x1-convolution pair that expands the channel width by ``mlp_ratio`` and
projects it back.

The block has no Pre/Post-Norm, no LayerScale and no stochastic depth on
the residual stream, and its MLP is a ``ConvBlock`` pair rather than a
`ffn/` factory type, because `'gated_mlp'`, the one 4D-capable FFN type,
carries no normalization stage and would drop the intermediate BatchNorm
the ``ConvBlock`` pair applies. Normalization type is fixed to
`'batch_norm'`; only its epsilon/momentum pair is configurable, through
`normalization_kwargs`. `use_bias` defaults to `False`.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.layers.standard_blocks import ConvBlock
from dl_techniques.layers.attention.area_attention import AreaAttention
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.transformers.area_attention_block")
class AreaAttentionBlock(keras.layers.Layer):
    """
    Area-attention transformer block over a 4D ``(batch, height, width, channels)`` map.

    Attention and a 1x1-convolution MLP, each wrapped in a plain residual add. There
    is no normalization on the residual stream, no LayerScale and no stochastic depth
    -- see the module docstring for why those are declined rather than missing.

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────┐
        │  Input [B, H, W, C]            │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  AreaAttention + Residual      │
        │  x = input + attn(input)       │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  MLP (Conv1x1 expand + shrink) │
        │  + Residual                    │
        │  x = x + mlp2(mlp1(x))         │
        └──────────────┬─────────────────┘
                       ▼
        ┌────────────────────────────────┐
        │  Output [B, H, W, dim]         │
        └────────────────────────────────┘

    :param dim: Number of feature dimensions, and the output channel count. Must be
        positive and divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param mlp_ratio: Expansion ratio for the MLP hidden width; the hidden width is
        ``int(dim * mlp_ratio)``. Defaults to 1.2.
    :type mlp_ratio: float
    :param area: Number of attention groups handed to
        :class:`~dl_techniques.layers.attention.area_attention.AreaAttention`; ``1``
        means global attention. Defaults to 1.
    :type area: int
    :param use_bias: Whether the block's convolutions -- ``mlp1``, ``mlp2`` and the
        four inside the attention sub-layer -- carry a bias term. Defaults to
        ``False``.
    :type use_bias: bool
    :param normalization_kwargs: Extra arguments forwarded to the normalization
        factory by every ``ConvBlock`` in this block and in its attention sub-layer.
        ``None`` means the factory's own defaults (``epsilon=1e-6``). Callers that
        need a specific epsilon/momentum pair -- yolo12 does -- supply it here.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param kernel_initializer: Weight initializer for every convolution. Defaults to
        ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``dim``, ``num_heads``, ``mlp_ratio`` or ``area`` is not
        positive, or if ``dim`` is not divisible by ``num_heads`` (raised by the
        attention sub-layer).

    Example:
        >>> import keras, numpy as np
        >>> block = AreaAttentionBlock(dim=64, num_heads=8, area=4)
        >>> y = block(np.zeros((2, 8, 8, 64), dtype="float32"))
        >>> y.shape
        (2, 8, 8, 64)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 1.2,
            area: int = 1,
            use_bias: bool = False,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        This block owns no weights of its own; every weight belongs to a sub-layer.
        See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        # `dim % num_heads` is not checked here; the attention sub-layer delegates
        # it to `attention.common.validate_head_divisibility`.
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")

        # Store configuration parameters.
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.area = area
        self.use_bias = use_bias
        self.normalization_kwargs = normalization_kwargs
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Sub-layers are created here, unbuilt. Reordering attn/mlp1/mlp2 below
        # would silently permute weights on a `set_weights` load.
        self.attn = AreaAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            area=self.area,
            use_bias=self.use_bias,
            normalization_kwargs=self.normalization_kwargs,
            kernel_initializer=self.kernel_initializer,
            name="attn"
        )

        # 1x1 conv expand (SiLU) then 1x1 conv project (no activation); each stage
        # is a ConvBlock (Conv2D + normalization + activation). `mlp1`
        # (mlp_hidden_dim channels) and `mlp2` (dim channels) are
        # different-shape siblings whenever `mlp_ratio != 1` (every caller in
        # this tree), and `attn` decorrelates its own internal conv siblings
        # regardless of what `kernel_initializer` object it receives -- see
        # `AreaAttention._fresh_initializer`. None of the three needs its own
        # fresh instance here.
        self.mlp1 = ConvBlock(
            filters=self.mlp_hidden_dim,
            kernel_size=1,
            activation_type="silu",
            normalization_kwargs=self.normalization_kwargs,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="mlp1"
        )

        self.mlp2 = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            activation_type="linear",
            normalization_kwargs=self.normalization_kwargs,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="mlp2"
        )

        logger.debug(
            f"AreaAttentionBlock initialized: dim={dim}, num_heads={num_heads}, "
            f"area={area}, mlp_hidden_dim={self.mlp_hidden_dim}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer explicitly, in computational order.

        :param input_shape: Shape tuple of the input tensor, ``(B, H, W, C)``.
        :type input_shape: tuple
        """
        self.attn.build(input_shape)
        self.mlp1.build(input_shape)

        # `mlp2` sees the expanded hidden width, not the block input width.
        mlp1_output_shape = self.mlp1.compute_output_shape(input_shape)
        self.mlp2.build(mlp1_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: attention residual, then MLP residual.

        :param inputs: Input tensor of shape ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional **keep** mask over spatial positions, forwarded
            verbatim to the attention sub-layer (``1 = keep``). ``None`` disables
            masking entirely -- no mask op is added to the graph.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether the block runs in training mode.
        :type training: Optional[bool]

        :return: Output tensor of shape ``(batch, height, width, dim)``.
        :rtype: keras.KerasTensor
        """
        # Attention with residual connection
        attn_out = self.attn(
            inputs, attention_mask=attention_mask, training=training
        )
        x = inputs + attn_out

        # MLP with residual connection
        mlp_out = self.mlp1(x, training=training)
        mlp_out = self.mlp2(mlp_out, training=training)
        return x + mlp_out

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: tuple

        :return: Output shape tuple, with the last axis replaced by ``dim``.
        :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the block configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "area": self.area,
            "use_bias": self.use_bias,
            "normalization_kwargs": self.normalization_kwargs,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
        })
        return config

# ---------------------------------------------------------------------
