"""Sam3ViTDetBlock / Sam3ViTDetBackbone, SAM 3's plain-ViT detection trunk.

``Sam3ViTDetBlock`` is one pre-LN transformer block, either window-local or
global attention; ``Sam3ViTDetBackbone`` composes the patch-embed stem, a
tiled absolute position embedding, the block stack (mostly windowed, a few
global), and emits a single output feature map -- the multi-scale pyramid is
built downstream by the neck, not here.

Attention uses 2D axial rotary position embedding on queries and keys only.
The rotary frequency ladder is built at each block's own token grid, scaled
by the ratio of the pre-training grid to that grid, so a windowed block
(whose grid is its window) and a global block (whose grid is the whole
image) both get an angular pitch matching what they were pre-trained on.
The absolute position embedding is tiled, not interpolated, up to the
current grid, and any excess tile is cropped away rather than resized.

The MLP hidden width truncates ``dim * mlp_ratio`` rather than rounding it.
This file is channels-last throughout; the reference trunk is
channels-first.

References:
    - Li et al., 2022. Exploring Plain Vision Transformer Backbones for
      Object Detection.
    - Dosovitskiy et al., 2021. An Image is Worth 16x16 Words: Transformers
      for Image Recognition at Scale.
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding.
"""

import math
import keras
from keras import ops
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.layers.regularization.layer_scale import LayerScale
from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.embedding.axial_rope_2d import AxialRoPE2D
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# module-private helpers
# ---------------------------------------------------------------------


def _window_partition(x: Any, window_size: int) -> Any:
    """Split a spatial grid into non-overlapping square windows.

    ``(B, H, W, C) -> (B * num_windows, window_size, window_size, C)`` with the
    windows ordered row-major over the window grid.

    :param x: Channels-last spatial tensor.
    :type x: Any
    :param window_size: Side of the square window. Must divide both ``H`` and
        ``W`` exactly -- see the note in :meth:`Sam3ViTDetBlock.build`.
    :type window_size: int
    :return: The partitioned tensor.
    :rtype: Any
    """
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    grid_h = height // window_size
    grid_w = width // window_size
    x = ops.reshape(x, (-1, grid_h, window_size, grid_w, window_size, channels))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, window_size, window_size, channels))


def _window_unpartition(
        windows: Any, window_size: int, height: int, width: int
) -> Any:
    """Invert :func:`_window_partition`.

    :param windows: ``(B * num_windows, window_size, window_size, C)``.
    :type windows: Any
    :param window_size: Side of the square window.
    :type window_size: int
    :param height: Original grid height.
    :type height: int
    :param width: Original grid width.
    :type width: int
    :return: ``(B, height, width, C)``.
    :rtype: Any
    """
    channels = windows.shape[-1]
    grid_h = height // window_size
    grid_w = width // window_size
    x = ops.reshape(
        windows, (-1, grid_h, grid_w, window_size, window_size, channels)
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, height, width, channels))


# DECISION plan-2026-08-04T044628-4c240b4c/D-085: keep this attention class
# module-private and unregistered; never replace it with
# create_attention_layer('multi_head', ...).
# That factory entry has no hook to apply 2D axial RoPE between the head split and the score matmul, and expects three separate projections, not one fused qkv. See decisions.md.
class _Sam3ViTDetAttention(keras.layers.Layer):
    """Fused-qkv multi-head self-attention with 2D axial RoPE, channels-last.

    Private implementation detail of :class:`Sam3ViTDetBlock`. Consumes and
    returns a rank-4 ``(batch, height, width, dim)`` spatial tensor; the token
    axis is the row-major flattening of ``(height, width)``, which is the
    ordering :class:`AxialRoPE2D` assumes.

    **Data flow:**

    .. code-block:: text

        x (B, H, W, dim)
             │
          qkv: Dense(3 * dim)           -> split into q, k, v
             │
          split heads (B, heads, H*W, head_dim)
             │
          AxialRoPE2D(q, k)             (v is NOT rotated)
             │
          softmax(q k^T / sqrt(head_dim)) v
             │
          merge heads -> proj: Dense(dim) -> (B, H, W, dim)

    :param dim: Token width.
    :type dim: int
    :param num_heads: Number of attention heads. Must divide ``dim``, and the
        resulting head width must be a multiple of 4 for 2D axial RoPE.
    :type num_heads: int
    :param input_size: Token grid ``(H, W)`` this attention runs over. For a
        windowed block this is the WINDOW's grid, not the image's.
    :type input_size: Tuple[int, int]
    :param qkv_bias: Whether the fused ``qkv`` projection carries a bias.
    :type qkv_bias: bool
    :param use_rope: Whether to apply 2D axial RoPE to q/k.
    :type use_rope: bool
    :param rope_theta: Base of the rotary frequency ladder.
    :type rope_theta: float
    :param rope_scale_pos: Multiplier on the rotary position indices; see the
        module docstring's rotary paragraph.
    :type rope_scale_pos: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim`` is not divisible by ``num_heads`` or the head
        width is not a multiple of 4.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            input_size: Tuple[int, int],
            qkv_bias: bool = True,
            use_rope: bool = True,
            rope_theta: float = 10000.0,
            rope_scale_pos: float = 1.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        head_dim = dim // num_heads
        if use_rope and head_dim % 4 != 0:
            raise ValueError(
                f"head width ({head_dim} = {dim} // {num_heads}) must be "
                f"divisible by 4 for 2D axial RoPE"
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.qkv_bias = bool(qkv_bias)
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.rope_scale_pos = float(rope_scale_pos)

        # Derived, non-config.
        self.head_dim = head_dim
        self.num_tokens = self.input_size[0] * self.input_size[1]
        self._scale = 1.0 / math.sqrt(float(head_dim))

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.qkv = keras.layers.Dense(
            3 * self.dim, use_bias=self.qkv_bias, name="qkv"
        )
        self.proj = keras.layers.Dense(self.dim, name="proj")
        self.rope = AxialRoPE2D(
            head_dim=self.head_dim,
            feat_shape=self.input_size,
            theta=self.rope_theta,
            scale_pos=self.rope_scale_pos,
            name="rope",
        ) if self.use_rope else None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the projections and the rotary table.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not a rank-4 channels-last tensor
            whose spatial extent matches ``input_size``.
        """
        if self.built:
            return
        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"_Sam3ViTDetAttention expects a rank-4 channels-last input, "
                f"got shape {input_shape}"
            )
        if (input_shape[1], input_shape[2]) != self.input_size:
            raise ValueError(
                f"input grid {(input_shape[1], input_shape[2])} must equal the "
                f"configured input_size {self.input_size}"
            )
        self.qkv.build(input_shape)
        self.proj.build((*input_shape[:-1], self.dim))
        if self.rope is not None:
            rope_shape = (None, self.num_heads, self.num_tokens, self.head_dim)
            self.rope.build(rope_shape, rope_shape)
        super().build(input_shape)

    def _split_heads(self, x: Any) -> Any:
        """Reshape ``(B, N, dim)`` to ``(B, heads, N, head_dim)``.

        :param x: Projected tensor.
        :type x: Any
        :return: Head-split tensor.
        :rtype: Any
        """
        x = ops.reshape(x, (-1, self.num_tokens, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Run rotary multi-head self-attention over the spatial grid.

        :param inputs: ``(batch, height, width, dim)``.
        :type inputs: Any
        :param training: Unused; present for the Keras call contract.
        :type training: Optional[bool]
        :return: ``(batch, height, width, dim)``.
        :rtype: Any
        """
        height, width = self.input_size
        qkv = ops.reshape(self.qkv(inputs), (-1, self.num_tokens, 3 * self.dim))
        query, key, value = ops.split(qkv, 3, axis=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if self.rope is not None:
            # Rotate q and k only: values carry content, not position.
            query, key = self.rope(query, key)

        attn = ops.matmul(query * self._scale, ops.transpose(key, (0, 1, 3, 2)))
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, value)

        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (-1, height, width, self.dim))
        return self.proj(out)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, height, width, dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (tuple(input_shape)[0], self.input_size[0], self.input_size[1],
                self.dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "input_size": self.input_size,
            "qkv_bias": self.qkv_bias,
            "use_rope": self.use_rope,
            "rope_theta": self.rope_theta,
            "rope_scale_pos": self.rope_scale_pos,
        })
        return config


def _make_layer_scale(
        init_values: Optional[float], name: str
) -> keras.layers.Layer:
    """Build the layer-scale gain, or an explicit ``Identity`` when disabled.

    :param init_values: Constant the per-channel gain is initialized to, or
        ``None`` for no layer scale at all.
    :type init_values: Optional[float]
    :param name: Sub-layer name.
    :type name: str
    :return: A :class:`LayerScale` or a ``keras.layers.Identity``.
    :rtype: keras.layers.Layer
    """
    # DECISION plan-2026-08-04T044628-4c240b4c/D-086: at init_values=None
    # (the settled configuration) return a real Identity, never an
    # always-on LayerScale initialized to ones.
    # A trainable ones-init gain adds dim parameters the reference lacks and is identity only at step 0. Also pass constraint=None -- LayerScale's non_neg default forbids sign flips the reference permits. See decisions.md.
    if init_values is None:
        return keras.layers.Identity(name=name)
    return LayerScale(
        multiplier_type="CHANNEL",
        initializer=keras.initializers.Constant(float(init_values)),
        constraint=None,
        name=name,
    )


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam3.vitdet")
class Sam3ViTDetBlock(keras.layers.Layer):
    """One pre-LN ViT block, either window-local or global.

    Instantiated once per trunk depth (32 times at the settled config) with a
    per-block ``window_size``: a positive value makes attention window-local, a
    zero makes it global over the whole token grid.

    **Architecture Overview:**

    .. code-block:: text

        x (B, H, W, dim)
         ├──────────────────────────────── shortcut ────────┐
         ▼                                                  │
        norm1 -> [window_partition] -> attn -> [unpartition] │
         ▼                                                  │
        ls1 -> drop_path -> dropout ──────────────── + ◄─────┘
         │
         ├──────────────────────────────── shortcut ────────┐
         ▼                                                  │
        norm2 -> mlp(int(dim * mlp_ratio)) -> ls2           │
         ▼                                                  │
        drop_path -> dropout ─────────────────────── + ◄─────┘
         │
         ▼ (B, H, W, dim)

    :param dim: Token width.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param input_size: The IMAGE token grid ``(H, W)`` this block sits in. When
        ``window_size > 0`` attention runs on ``(window_size, window_size)``
        instead; ``input_size`` still describes the block's input tensor.
    :type input_size: Tuple[int, int]
    :param mlp_ratio: Hidden-width multiplier. The hidden width is
        ``int(dim * mlp_ratio)`` -- TRUNCATED, never rounded.
    :type mlp_ratio: float
    :param window_size: Side of the attention window, or ``0`` for global
        attention. Must divide both axes of ``input_size`` exactly.
    :type window_size: int
    :param qkv_bias: Whether the fused ``qkv`` projection carries a bias.
    :type qkv_bias: bool
    :param drop_path_rate: Stochastic-depth rate for both residual branches.
    :type drop_path_rate: float
    :param dropout_rate: Dropout applied inside the MLP and to both residual
        branches.
    :type dropout_rate: float
    :param norm_epsilon: LayerNorm epsilon.
    :type norm_epsilon: float
    :param activation: MLP activation.
    :type activation: str
    :param init_values: Layer-scale init constant, or ``None`` for no layer
        scale (the settled configuration).
    :type init_values: Optional[float]
    :param use_rope: Whether to apply 2D axial RoPE to q/k.
    :type use_rope: bool
    :param rope_theta: Base of the rotary frequency ladder.
    :type rope_theta: float
    :param rope_pt_size: Side of the grid the rotary embedding was pre-trained
        on. With ``use_interp_rope`` this sets
        ``scale_pos = rope_pt_size / attention_grid_side``.
    :type rope_pt_size: int
    :param use_interp_rope: Whether to scale the rotary position indices onto
        the pre-training grid's pitch. A windowed block whose window equals
        ``rope_pt_size`` gets ``scale_pos = 1.0`` either way.
    :type use_interp_rope: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``window_size`` is negative or does not divide
        ``input_size``, or if ``mlp_ratio`` is not positive.

    Example:
        >>> import numpy as np
        >>> blk = Sam3ViTDetBlock(dim=8, num_heads=2, input_size=(8, 8),
        ...                       window_size=4, rope_pt_size=4)
        >>> out = blk(np.zeros((1, 8, 8, 8), dtype="float32"))
        >>> out.shape
        (1, 8, 8, 8)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            input_size: Sequence[int],
            mlp_ratio: float = 4.0,
            window_size: int = 0,
            qkv_bias: bool = True,
            drop_path_rate: float = 0.0,
            dropout_rate: float = 0.0,
            norm_epsilon: float = 1e-5,
            activation: str = "gelu",
            init_values: Optional[float] = None,
            use_rope: bool = True,
            rope_theta: float = 10000.0,
            rope_pt_size: Optional[int] = None,
            use_interp_rope: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        input_size = (int(input_size[0]), int(input_size[1]))
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {window_size}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        # DECISION plan-2026-08-04T044628-4c240b4c/D-087: raise on
        # non-divisible window geometry; do not add the reference's
        # zero-padding branch.
        # That branch is dead code at every shipped geometry (72 % 24 == 0), and an unreachable branch is untestable. See decisions.md.
        if window_size > 0 and (
                input_size[0] % window_size or input_size[1] % window_size
        ):
            raise ValueError(
                f"window_size ({window_size}) must divide both axes of "
                f"input_size ({input_size}) exactly; this port does NOT "
                f"implement the reference's zero-padding branch, which is dead "
                f"code at every shipped geometry"
            )

        # Store ALL configuration parameters.
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.input_size = input_size
        self.mlp_ratio = float(mlp_ratio)
        self.window_size = int(window_size)
        self.qkv_bias = bool(qkv_bias)
        self.drop_path_rate = float(drop_path_rate)
        self.dropout_rate = float(dropout_rate)
        self.norm_epsilon = float(norm_epsilon)
        self.activation = deserialize_activation(activation)
        self.init_values = init_values
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.rope_pt_size = rope_pt_size
        self.use_interp_rope = bool(use_interp_rope)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-088: int() truncation,
        # never round(), matching the reference.
        # The two coincide at the shipped 1024 * 4.625 = 4736.0 ratio, so only a non-integral-ratio test can tell them apart. See decisions.md.
        self.mlp_hidden_dim = int(self.dim * self.mlp_ratio)

        attn_grid = (
            (self.window_size, self.window_size) if self.window_size > 0
            else self.input_size
        )
        pt_size = (
            self.window_size if self.window_size > 0 else attn_grid[0]
        ) if self.rope_pt_size is None else int(self.rope_pt_size)
        # DECISION plan-2026-08-04T044628-4c240b4c/D-089: scale_pos is
        # pt_size / this block's own attention grid (window for a windowed
        # block, image for a global one) -- never 1.0 for global blocks.
        # A global block at scale_pos=1.0 would triple the rotary table's angular pitch versus what it was pre-trained at, with no shape symptom. See decisions.md.
        self.rope_scale_pos = (
            float(pt_size) / float(attn_grid[0]) if self.use_interp_rope else 1.0
        )

        # Sub-layers -- created UNCONDITIONALLY, built explicitly in build().
        self.norm1 = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="norm1"
        )
        self.norm2 = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="norm2"
        )
        self.attn = _Sam3ViTDetAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            input_size=attn_grid,
            qkv_bias=self.qkv_bias,
            use_rope=self.use_rope,
            rope_theta=self.rope_theta,
            rope_scale_pos=self.rope_scale_pos,
            name="attn",
        )
        self.mlp = create_ffn_layer(
            "mlp",
            hidden_dim=self.mlp_hidden_dim,
            output_dim=self.dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            name="mlp",
        )
        self.ls1 = _make_layer_scale(self.init_values, "ls1")
        self.ls2 = _make_layer_scale(self.init_values, "ls2")
        self.drop_path = StochasticDepth(
            drop_path_rate=self.drop_path_rate, name="drop_path"
        )
        self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer explicitly.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank-4 or its spatial extent
            does not match ``input_size``.
        """
        if self.built:
            return
        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"Sam3ViTDetBlock expects a rank-4 channels-last input, got "
                f"shape {input_shape}"
            )
        if (input_shape[1], input_shape[2]) != self.input_size:
            raise ValueError(
                f"input grid {(input_shape[1], input_shape[2])} must equal the "
                f"configured input_size {self.input_size}"
            )

        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        attn_shape = (
            None, *self.attn.input_size, self.dim
        )
        self.attn.build(attn_shape)
        self.mlp.build(input_shape)
        self.ls1.build(attn_shape)
        self.ls2.build(input_shape)
        self.drop_path.build(input_shape)
        self.dropout.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Run the two residual sub-blocks.

        :param inputs: ``(batch, height, width, dim)``.
        :type inputs: Any
        :param training: Keras training flag; gates dropout and drop-path.
        :type training: Optional[bool]
        :return: ``(batch, height, width, dim)``.
        :rtype: Any
        """
        height, width = self.input_size
        shortcut = inputs
        x = self.norm1(inputs)
        if self.window_size > 0:
            x = _window_partition(x, self.window_size)
        x = self.ls1(self.attn(x, training=training))
        if self.window_size > 0:
            x = _window_unpartition(x, self.window_size, height, width)
        x = shortcut + self.dropout(
            self.drop_path(x, training=training), training=training
        )

        residual = self.ls2(self.mlp(self.norm2(x), training=training))
        return x + self.dropout(
            self.drop_path(residual, training=training), training=training
        )

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param input_shape: ``(batch, height, width, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, height, width, dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (tuple(input_shape)[0], self.input_size[0], self.input_size[1],
                self.dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "input_size": self.input_size,
            "mlp_ratio": self.mlp_ratio,
            "window_size": self.window_size,
            "qkv_bias": self.qkv_bias,
            "drop_path_rate": self.drop_path_rate,
            "dropout_rate": self.dropout_rate,
            "norm_epsilon": self.norm_epsilon,
            "activation": serialize_activation(self.activation),
            "init_values": self.init_values,
            "use_rope": self.use_rope,
            "rope_theta": self.rope_theta,
            "rope_pt_size": self.rope_pt_size,
            "use_interp_rope": self.use_interp_rope,
        })
        return config


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam3.vitdet")
class Sam3ViTDetBackbone(keras.layers.Layer):
    """SAM 3's ViTDet trunk: patch-embed stem, tiled abs-pos, block stack.

    Emits exactly ONE channels-last feature map -- the output of the LAST global
    attention block, after ``ln_post``. The multi-scale pyramid SAM 3 consumes is
    built entirely downstream by the neck.

    **Architecture Overview:**

    .. code-block:: text

        image (B, img_size, img_size, in_channels)
              │
        Conv2D(embed_dim, patch_size, stride=patch_size)   -> (B, g, g, embed_dim)
              │
            + tile_abs_pos(pos_embed)      tile-then-CROP, never interpolate
              │
            ln_pre                         ONCE, before the stack
              │
        block 0 .. block depth-1           window-local except global_att_blocks
              │
            ln_post                        only after the LAST global block
              │
              ▼ (B, g, g, embed_dim)

    :param img_size: Input image side. Must be divisible by ``patch_size``.
    :type img_size: int
    :param patch_size: Patch (and stem stride) side.
    :type patch_size: int
    :param in_channels: Input image channels.
    :type in_channels: int
    :param embed_dim: Token width.
    :type embed_dim: int
    :param depth: Number of blocks.
    :type depth: int
    :param num_heads: Attention heads per block.
    :type num_heads: int
    :param mlp_ratio: MLP hidden-width multiplier; truncated, see
        :class:`Sam3ViTDetBlock`.
    :type mlp_ratio: float
    :param window_size: Attention window side for the non-global blocks.
    :type window_size: int
    :param global_att_blocks: Block indices that attend globally. The LAST of
        them is the block whose output the trunk returns, so it is REQUIRED to
        be ``depth - 1`` -- anything earlier leaves the remaining blocks built,
        parameter-counted and optimizer-tracked but never executed. Enforced in
        ``__init__``.
    :type global_att_blocks: Sequence[int]
    :param qkv_bias: Whether the fused ``qkv`` projections carry biases.
    :type qkv_bias: bool
    :param drop_path_rate: Terminal stochastic-depth rate; the per-block rate is
        linearly interpolated from ``0`` to this value across ``depth``.
    :type drop_path_rate: float
    :param dropout_rate: Dropout rate inside every block.
    :type dropout_rate: float
    :param norm_epsilon: LayerNorm epsilon everywhere in the trunk.
    :type norm_epsilon: float
    :param activation: MLP activation.
    :type activation: str
    :param init_values: Layer-scale init constant, or ``None`` for none.
    :type init_values: Optional[float]
    :param pretrain_img_size: Image side the position embedding was pre-trained
        at; its grid is ``pretrain_img_size // patch_size``.
    :type pretrain_img_size: int
    :param pretrain_use_cls_token: Whether the pre-trained position embedding
        carries a leading class-token row. That row is STORED (so the parameter
        count matches the reference checkpoint) and then dropped, because SAM 3
        does not retain a class token.
    :type pretrain_use_cls_token: bool
    :param use_abs_pos: Whether to add the absolute position embedding at all.
    :type use_abs_pos: bool
    :param bias_patch_embed: Whether the stem convolution carries a bias.
    :type bias_patch_embed: bool
    :param ln_pre: Whether to apply a LayerNorm after stem + position embedding.
    :type ln_pre: bool
    :param ln_post: Whether to apply a LayerNorm after the last global block.
    :type ln_post: bool
    :param use_rope: Whether the blocks apply 2D axial RoPE.
    :type use_rope: bool
    :param rope_theta: Base of the rotary frequency ladder.
    :type rope_theta: float
    :param rope_pt_size: Rotary pre-training grid side. ``None`` means
        ``window_size``, which is the reference's own default.
    :type rope_pt_size: Optional[int]
    :param use_interp_rope: Whether to scale rotary position indices onto the
        pre-training pitch (relevant to the global blocks only).
    :type use_interp_rope: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``img_size`` is not divisible by ``patch_size``, if
        ``depth`` is not positive, if ``global_att_blocks`` is empty, holds an
        out-of-range index or does not name the LAST block
        (``max(global_att_blocks) != depth - 1``, which would leave the blocks
        past it built and trained but never executed), or if
        ``pretrain_img_size`` is not divisible by ``patch_size``.

    Example:
        >>> import numpy as np
        >>> trunk = Sam3ViTDetBackbone(
        ...     img_size=16, patch_size=2, embed_dim=8, depth=2, num_heads=2,
        ...     window_size=4, global_att_blocks=(1,), pretrain_img_size=8)
        >>> trunk(np.zeros((1, 16, 16, 3), dtype="float32")).shape
        (1, 8, 8, 8)
    """

    def __init__(
            self,
            img_size: int = 1008,
            patch_size: int = 14,
            in_channels: int = 3,
            embed_dim: int = 1024,
            depth: int = 32,
            num_heads: int = 16,
            mlp_ratio: float = 4.625,
            window_size: int = 24,
            global_att_blocks: Sequence[int] = (7, 15, 23, 31),
            qkv_bias: bool = True,
            drop_path_rate: float = 0.1,
            dropout_rate: float = 0.0,
            norm_epsilon: float = 1e-5,
            activation: str = "gelu",
            init_values: Optional[float] = None,
            pretrain_img_size: int = 336,
            pretrain_use_cls_token: bool = True,
            use_abs_pos: bool = True,
            bias_patch_embed: bool = False,
            ln_pre: bool = True,
            ln_post: bool = False,
            use_rope: bool = True,
            rope_theta: float = 10000.0,
            rope_pt_size: Optional[int] = None,
            use_interp_rope: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if img_size <= 0 or patch_size <= 0 or img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be a positive multiple of "
                f"patch_size ({patch_size})"
            )
        if pretrain_img_size % patch_size != 0:
            raise ValueError(
                f"pretrain_img_size ({pretrain_img_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        global_att_blocks = tuple(int(i) for i in global_att_blocks)
        if not global_att_blocks:
            raise ValueError(
                "global_att_blocks must name at least one block -- the trunk's "
                "single output feature map is the LAST global block's output"
            )
        if any(i < 0 or i >= depth for i in global_att_blocks):
            raise ValueError(
                f"global_att_blocks {global_att_blocks} must all be in "
                f"[0, depth={depth})"
            )
        # DECISION plan-2026-08-18T140459-7991552f/D-046: raise unless the
        # last block is global; never silently truncate self.blocks or
        # downgrade this to a warning.
        # call() returns at index == max(global_att_blocks), so any block after that index is built and optimized but contributes nothing -- measured max|delta| == 0.0 with no shape or warning symptom. See decisions.md.
        if max(global_att_blocks) != depth - 1:
            raise ValueError(
                f"global_att_blocks {global_att_blocks} must name the LAST "
                f"block (depth - 1 = {depth - 1}) -- the trunk returns at the "
                f"last global block, so blocks "
                f"{tuple(range(max(global_att_blocks) + 1, depth))} would be "
                f"built and trained but never executed"
            )

        # Store ALL configuration parameters.
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.window_size = int(window_size)
        self.global_att_blocks = global_att_blocks
        self.qkv_bias = bool(qkv_bias)
        self.drop_path_rate = float(drop_path_rate)
        self.dropout_rate = float(dropout_rate)
        self.norm_epsilon = float(norm_epsilon)
        self.activation = deserialize_activation(activation)
        self.init_values = init_values
        self.pretrain_img_size = int(pretrain_img_size)
        self.pretrain_use_cls_token = bool(pretrain_use_cls_token)
        self.use_abs_pos = bool(use_abs_pos)
        self.bias_patch_embed = bool(bias_patch_embed)
        self.ln_pre = bool(ln_pre)
        self.ln_post = bool(ln_post)
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.rope_pt_size = rope_pt_size
        self.use_interp_rope = bool(use_interp_rope)

        # Derived, non-config.
        self.grid_size = self.img_size // self.patch_size
        self.pretrain_grid_size = self.pretrain_img_size // self.patch_size
        self.num_pos_tokens = (
            self.pretrain_grid_size ** 2 + int(self.pretrain_use_cls_token)
        )
        self.last_global_block = max(self.global_att_blocks)

        # Sub-layers -- created UNCONDITIONALLY, built explicitly in build().
        self.patch_embed = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            use_bias=self.bias_patch_embed,
            name="patch_embed",
        )
        # `ln_pre=False` / `ln_post=False` map to a real `Identity` layer, for
        # the same reason `_make_layer_scale` does: an unconditionally built
        # sub-layer with no parameters, rather than a `call()`-time branch.
        self.norm_pre = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="ln_pre"
        ) if self.ln_pre else keras.layers.Identity(name="ln_pre")
        self.norm_post = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="ln_post"
        ) if self.ln_post else keras.layers.Identity(name="ln_post")

        # Linear stochastic-depth decay across depth, `linspace(0, rate, depth)`.
        # The helper subsumes the `depth > 1` guard this used to carry inline:
        # at `depth <= 1` it returns `[0.0] * depth` (measured, not assumed).
        per_block_drop_path = linear_drop_path_rates(
            self.depth, self.drop_path_rate
        )
        self.blocks: List[Sam3ViTDetBlock] = [
            Sam3ViTDetBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                input_size=(self.grid_size, self.grid_size),
                mlp_ratio=self.mlp_ratio,
                window_size=(
                    0 if i in self.global_att_blocks else self.window_size
                ),
                qkv_bias=self.qkv_bias,
                drop_path_rate=per_block_drop_path[i],
                dropout_rate=self.dropout_rate,
                norm_epsilon=self.norm_epsilon,
                activation=self.activation,
                init_values=self.init_values,
                use_rope=self.use_rope,
                rope_theta=self.rope_theta,
                rope_pt_size=(
                    self.window_size if self.rope_pt_size is None
                    else self.rope_pt_size
                ),
                use_interp_rope=self.use_interp_rope,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]

        # Weight created in build().
        self.pos_embed = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the position embedding and build every sub-layer.

        :param input_shape: ``(batch, img_size, img_size, in_channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank-4 or its spatial extent
            does not match ``img_size``.
        """
        if self.built:
            return
        input_shape = tuple(input_shape)
        if len(input_shape) != 4:
            raise ValueError(
                f"Sam3ViTDetBackbone expects a rank-4 channels-last image, got "
                f"shape {input_shape}"
            )
        if input_shape[1] not in (None, self.img_size) or input_shape[2] not in (
                None, self.img_size
        ):
            raise ValueError(
                f"input spatial extent {(input_shape[1], input_shape[2])} must "
                f"match img_size ({self.img_size})"
            )

        self.patch_embed.build(input_shape)
        grid_shape = (None, self.grid_size, self.grid_size, self.embed_dim)

        if self.use_abs_pos:
            # The stored table keeps the pre-training class-token row even
            # though SAM 3 discards it, so the parameter count matches the
            # reference checkpoint exactly.
            self.pos_embed = self.add_weight(
                name="pos_embed",
                shape=(1, self.num_pos_tokens, self.embed_dim),
                initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
            )

        self.norm_pre.build(grid_shape)
        for block in self.blocks:
            block.build(grid_shape)
        self.norm_post.build(grid_shape)
        logger.debug(
            "Sam3ViTDetBackbone built: grid=%dx%d embed_dim=%d depth=%d "
            "window=%d global=%s pretrain_grid=%d mlp_hidden=%d",
            self.grid_size, self.grid_size, self.embed_dim, self.depth,
            self.window_size, self.global_att_blocks, self.pretrain_grid_size,
            self.blocks[0].mlp_hidden_dim,
        )
        super().build(input_shape)

    # -----------------------------------------------------------------
    # absolute position embedding
    # -----------------------------------------------------------------

    def _abs_pos(self) -> Any:
        """Return the absolute position embedding at the current token grid.

        :return: ``(1, grid_size, grid_size, embed_dim)``.
        :rtype: Any
        """
        table = ops.convert_to_tensor(self.pos_embed)
        if self.pretrain_use_cls_token:
            table = table[:, 1:]
        side = self.pretrain_grid_size
        table = ops.reshape(table, (1, side, side, self.embed_dim))
        if side == self.grid_size:
            return table
        # DECISION plan-2026-08-04T044628-4c240b4c/D-090: literal tile-then-
        # crop, never interpolation (e.g. ops.image.resize).
        # SAM 3 selects the tiling code path; an interpolating port produces different values at every token with no shape error. See decisions.md.
        tiles_h = self.grid_size // side + 1
        tiles_w = self.grid_size // side + 1
        tiled = ops.tile(table, (1, tiles_h, tiles_w, 1))
        return tiled[:, :self.grid_size, :self.grid_size, :]

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Embed an image and run the block stack.

        :param inputs: ``(batch, img_size, img_size, in_channels)``.
        :type inputs: Any
        :param training: Keras training flag; gates dropout and drop-path.
        :type training: Optional[bool]
        :return: The LAST global block's feature map,
            ``(batch, grid_size, grid_size, embed_dim)``.
        :rtype: Any
        """
        x = self.patch_embed(inputs)
        if self.use_abs_pos:
            x = x + ops.cast(self._abs_pos(), x.dtype)
        # DECISION plan-2026-08-04T044628-4c240b4c/D-091: run norm_pre here,
        # once, before the block stack -- never move it after.
        # The blocks are pre-LN; moving this norm to the end would leave the first block reading an unnormalized signal. See decisions.md.
        x = self.norm_pre(x)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-092: return one feature
        # map (the last global block's), never a per-global-block list.
        # The neck resamples this single map to build all four pyramid scales; a list would let it silently substitute shallower features. See decisions.md.
        for index, block in enumerate(self.blocks):
            x = block(x, training=training)
            if index == self.last_global_block:
                return self.norm_post(x)
        # Unreachable: `last_global_block < depth` is enforced in `__init__`.
        raise RuntimeError(
            "block stack finished without reaching the last global block"
        )

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the trunk output shape, derived from stored config.

        Never derived from weight shapes.

        :param input_shape: ``(batch, img_size, img_size, in_channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, grid_size, grid_size, embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (tuple(input_shape)[0], self.grid_size, self.grid_size,
                self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "window_size": self.window_size,
            "global_att_blocks": self.global_att_blocks,
            "qkv_bias": self.qkv_bias,
            "drop_path_rate": self.drop_path_rate,
            "dropout_rate": self.dropout_rate,
            "norm_epsilon": self.norm_epsilon,
            "activation": serialize_activation(self.activation),
            "init_values": self.init_values,
            "pretrain_img_size": self.pretrain_img_size,
            "pretrain_use_cls_token": self.pretrain_use_cls_token,
            "use_abs_pos": self.use_abs_pos,
            "bias_patch_embed": self.bias_patch_embed,
            "ln_pre": self.ln_pre,
            "ln_post": self.ln_post,
            "use_rope": self.use_rope,
            "rope_theta": self.rope_theta,
            "rope_pt_size": self.rope_pt_size,
            "use_interp_rope": self.use_interp_rope,
        })
        return config

# ---------------------------------------------------------------------
