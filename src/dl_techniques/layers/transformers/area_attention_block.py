"""
A transformer-style block over 2D feature maps, built on area attention.

The block is the yolo12 detector's attention stage. It is a *bare* residual
pair::

    x = inputs + attn(inputs)
    x = x + mlp2(mlp1(x))

where ``attn`` is
:class:`~dl_techniques.layers.attention.area_attention.AreaAttention` -- multi-head
self-attention over a ``(B, H, W, C)`` map that either runs globally (``area=1``) or
within ``area`` contiguous groups of the flattened token sequence -- and ``mlp1`` /
``mlp2`` are a 1x1-convolution pair that expands the channel width by ``mlp_ratio``
and projects it back.

This module is the relocated home of the ``AttentionBlock`` that used to live in
``dl_techniques.layers.yolo12_blocks``. The class was **renamed** on relocation:
``AttentionBlock`` is a bare, unqualified name and every sibling exported from this
package's ``__all__`` carries a prefix (``SwinTransformerBlock``, ``PFTBlock``,
``GatedLinearAttentionBlock``). See decisions.md D-006.

Two deliberate declines, both stated here because the obvious "bring it up to the
house shape" edit is a numerics change wearing a refactor's clothes:

* **No Pre/Post-Norm, no LayerScale, no StochasticDepth.** The residual shape above
  is exactly the pre-move block's. This layer lands as a *Specialized and Hybrid
  Block* (``transformers/README.md`` section "Specialized and Hybrid Blocks", the
  ``SwinConvBlock`` precedent), not as a
  :class:`~dl_techniques.layers.transformers.transformer.TransformerLayer` variant.
  Adding a normalization before the residual branches, a learned residual scale, or a
  drop-path would each change the function this block computes, which would make the
  relocation's equivalence claim unsatisfiable by construction. See decisions.md D-007.
* **``mlp1``/``mlp2`` are NOT ``create_ffn_layer('gated_mlp', ...)``.** ``'gated_mlp'``
  is the one 4D-capable FFN type in ``ffn/factory.py``, but it carries no
  normalization stage, so substituting it DROPS the intermediate BatchNorm that the
  ``mlp1``/``mlp2`` ``ConvBlock`` pair applies to the hidden activation -- a numerics
  change, not a refactor. See decisions.md D-007.

Relocation notes:
    * Normalization is **not** hardcoded here. yolo12's D-067 ``epsilon=1e-3,
      momentum=0.97`` pair keeps exactly one home,
      ``dl_techniques.layers.yolo12_blocks.YOLO12_NORM_KWARGS``, and arrives as data
      through ``normalization_kwargs``; this package sits *below* ``yolo12_blocks`` in
      the dependency order and must not import it. ``normalization_kwargs=None``
      therefore yields the normalization factory's own defaults (``epsilon=1e-6``),
      **not** yolo12's.
    * ``use_bias`` defaults to ``False``, matching the convolution convention of the
      pre-move yolo12 ``ConvBlock`` rather than
      :class:`~dl_techniques.layers.standard_blocks.ConvBlock`'s ``True``.
    * The normalization *type* is deliberately not a parameter. Both the block's own
      convolutions and the attention sub-layer use ``'batch_norm'``; the only knob any
      caller needs is the epsilon/momentum pair, which ``normalization_kwargs``
      already carries.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.standard_blocks import ConvBlock
from dl_techniques.layers.attention.area_attention import AreaAttention
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.transformers.area_attention_block")
class AreaAttentionBlock(keras.layers.Layer):
    """
    Area-attention transformer block over a 4D ``(batch, height, width, channels)`` map.

    Attention and a 1x1-convolution MLP, each wrapped in a plain residual add. There
    is no normalization on the residual stream, no LayerScale and no stochastic depth
    -- see the module docstring for why those are declined rather than missing.

    **Architecture Overview:**

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

    **Equivalence contract.** At default arguments this block reproduces the pre-move
    ``yolo12_blocks.AttentionBlock`` bit-for-bit on identical weights, provided
    ``normalization_kwargs`` carries the same normalization configuration. That claim
    is a test, not a comment:
    ``tests/test_layers/test_the_yolo12_relocation_is_equivalent.py``.

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

        # Validate inputs. `dim % num_heads` is deliberately NOT checked here -- the
        # attention sub-layer delegates it to `attention.common.validate_head_divisibility`
        # and re-checking it here would give the same condition two error messages.
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.area = area
        self.use_bias = use_bias
        self.normalization_kwargs = normalization_kwargs
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # CREATE all sub-layers in __init__ (they are unbuilt).
        #
        # The creation order -- attn, mlp1, mlp2 -- is the pre-move block's, and it is
        # the order the relocation's equivalence harness transfers weights in, by
        # ordered `set_weights`. Reordering these three statements is a silent
        # weight-permutation bug, not a cosmetic change.
        self.attn = AreaAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            area=self.area,
            use_bias=self.use_bias,
            normalization_kwargs=self.normalization_kwargs,
            kernel_initializer=self.kernel_initializer,
            name="attn"
        )

        # MLP: 1x1 conv expand (SiLU) -> 1x1 conv project (no activation). Each stage
        # is a `ConvBlock`, i.e. Conv2D + normalization + activation; the intermediate
        # normalization is load-bearing (see the module docstring's `'gated_mlp'` note).
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
