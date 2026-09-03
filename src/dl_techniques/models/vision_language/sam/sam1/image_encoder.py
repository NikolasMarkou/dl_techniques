"""
The ViT backbone that turns an image into the dense embedding grid every
other SAM 1 component consumes, built by :class:`ImageEncoderViT`.

It is a plain, non-hierarchical ViT: attention is windowed in every block
except a named set of global-attention indices, and a stride-1 neck
(1x1 conv, norm, 3x3 conv, norm) changes the channel count without touching
the spatial grid, which is fixed by the patch embedding alone. A windowed
attention layer with an optional learnable relative position bias
(:class:`WindowedAttentionWithRelPos`) backs each block.

A configuration with ``window_size > 0`` and no ``global_attn_indexes``
raises: it would window every block and leave the encoder with no global
receptive field. ``use_rel_pos=True`` changes the weight layout (adds
per-block ``rel_pos_h``/``rel_pos_w`` variables), so a checkpoint saved at
one setting of that flag cannot load at the other.

References:
    - Kirillov et al., 2023. Segment Anything. (https://arxiv.org/abs/2304.02643)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers
      for Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer
      using Shifted Windows. (https://arxiv.org/abs/2103.14030)
"""

import keras
from keras import layers, ops
from typing import Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.ffn.factory import assemble_ffn_config
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.sam1.image_encoder")
class WindowedAttentionWithRelPos(layers.Layer):
    """
    Multi-head self-attention over a 4D feature map, with an optional
    learnable relative position bias.

    Architecture:

    .. code-block:: text

        x [B, H, W, C]
              │
              ▼
        ┌─────────────┐
        │  Dense(3C)   │  qkv projection
        └──────┬───────┘
               ▼
        split into Q, K, V, reshape to heads
               │
               ▼
        (Q·scale) @ K^T ── + rel-pos bias (optional)
               │
               ▼
             softmax
               │
               ▼
             @ V, merge heads
               │
               ▼
        ┌─────────────┐
        │  Dense(C)    │  output projection
        └──────┬───────┘
               ▼
        out [B, H, W, C]

    :param dim: Input and output dimension of tokens.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param qkv_bias: Whether the Q/K/V projection carries a bias. Defaults
        to True.
    :type qkv_bias: bool
    :param use_rel_pos: Whether to add a learnable relative position bias to
        the attention scores. Defaults to False.
    :type use_rel_pos: bool
    :param input_size: Height and width of the input feature map. Required
        when ``use_rel_pos`` is True, to size the relative position tables.
    :type input_size: Optional[Tuple[int, int]]
    :param kwargs: Additional :class:`keras.layers.Layer` arguments.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, dim)`.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, dim)`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Store all configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        if self.use_rel_pos and input_size is None:
            raise ValueError("`input_size` must be provided if using relative positional encoding.")

        # Initialize weight attributes that will be created in build()
        self.rel_pos_h = None
        self.rel_pos_w = None

        # CREATE sub-layers in __init__
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.proj = layers.Dense(dim, name="proj")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the relative position tables (if used) and build sub-layers.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # CREATE the layer's own weights
        if self.use_rel_pos:
            self.rel_pos_h = self.add_weight(
                name='rel_pos_h',
                shape=(2 * self.input_size[0] - 1, self.head_dim),
                initializer='zeros',
                trainable=True,
            )
            self.rel_pos_w = self.add_weight(
                name='rel_pos_w',
                shape=(2 * self.input_size[1] - 1, self.head_dim),
                initializer='zeros',
                trainable=True,
            )

        # BUILD sub-layers
        self.qkv.build(input_shape)
        # The proj layer receives the same shape as input
        self.proj.build(input_shape)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Forward pass for attention.

        Args:
            x: Input tensor of shape (batch_size, height, width, dim).

        Returns:
            Output tensor of shape (batch_size, height, width, dim).
        """
        B, H, W, C = ops.shape(x)
        # Project to Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, H * W, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention
        attn = (q * self.scale) @ ops.transpose(k, (0, 1, 3, 2))

        # Add relative positional bias if enabled
        if self.use_rel_pos:
            attn = self._add_decomposed_rel_pos(attn, q, (H, W))

        attn = ops.softmax(attn, axis=-1)

        # Apply attention to values and project back
        x = attn @ v
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, H, W, C))
        x = self.proj(x)
        return x

    def _get_rel_pos(
        self,
        q_size: int,
        k_size: int,
        rel_pos: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Get relative positional embeddings.

        Args:
            q_size: Query size.
            k_size: Key size.
            rel_pos: Relative position embedding tensor.

        Returns:
            Relative positional embeddings.
        """
        max_rel_dist = 2 * max(q_size, k_size) - 1
        # Interpolate relative positional embeddings if needed.
        if ops.shape(rel_pos)[0] != max_rel_dist:
            # DECISION plan-2026-08-03T191222-1d751f81/D-007: reshape to a 4D
            # batched image (1, 1, L, C) before resize, not a 3D (1, C, L) one.
            # A 3D input reads as unbatched (h, w, c) and resizes channels, not distance. See decisions.md.
            shape = ops.shape(rel_pos)
            rel_pos_resized = ops.image.resize(
                ops.reshape(rel_pos, (1, 1, shape[0], shape[1])),
                size=(1, max_rel_dist),
                interpolation='bilinear'
            )
            rel_pos_resized = ops.reshape(rel_pos_resized, (max_rel_dist, shape[1]))
        else:
            rel_pos_resized = rel_pos

        # DECISION plan_2026-06-15_e6a0391c/D-004: cast arange to float32 before
        # scaling by the float ratio; multiplying int32 by float raises. See decisions.md.
        q_coords = ops.cast(ops.expand_dims(ops.arange(q_size), axis=1), "float32") * max(k_size / q_size, 1.0)
        k_coords = ops.cast(ops.expand_dims(ops.arange(k_size), axis=0), "float32") * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + float(k_size - 1) * max(q_size / k_size, 1.0)

        # keras.ops.gather does not exist on keras 3.8; ops.take(..., axis=0) is the row-gather.
        return ops.take(rel_pos_resized, ops.cast(relative_coords, 'int32'), axis=0)

    def _add_decomposed_rel_pos(
        self,
        attn: keras.KerasTensor,
        q: keras.KerasTensor,
        q_size: Tuple[int, int]
    ) -> keras.KerasTensor:
        """
        Calculate and add decomposed relative positional embeddings.

        Args:
            attn: Attention tensor.
            q: Query tensor.
            q_size: Query spatial dimensions (height, width).

        Returns:
            Attention tensor with relative positional bias added.
        """
        q_h, q_w = q_size
        B, nH, S, D = ops.shape(q)

        # Get relative positional embeddings for height and width
        Rh = self._get_rel_pos(q_h, q_h, self.rel_pos_h)
        Rw = self._get_rel_pos(q_w, q_w, self.rel_pos_w)

        # Reshape query for einsum operations
        r_q = ops.reshape(q, (B, nH, q_h, q_w, D))

        # DECISION plan_2026-06-15_e6a0391c/D-004: rel_w's einsum output label
        # must be 'k' like rel_h, not 'x' -- 'x' is absent from the inputs and raises. See decisions.md.
        rel_h = ops.einsum("bnhwc,hkc->bnhwk", r_q, Rh)
        rel_w = ops.einsum("bnhwc,wkc->bnhwk", r_q, Rw)

        # Add the biases to the attention scores
        attn = ops.reshape(attn, (B, nH, q_h, q_w, q_h, q_w))
        attn = attn + ops.expand_dims(rel_h, axis=-1) + ops.expand_dims(rel_w, axis=-2)
        attn = ops.reshape(attn, (B, nH, q_h * q_w, q_h * q_w))
        return attn

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "use_rel_pos": self.use_rel_pos,
            "input_size": self.input_size,
        })
        return config


@register_dl_technique("dl_techniques.models.sam1.image_encoder")
class ViTBlock(layers.Layer):
    """
    A pre-norm transformer block over a 4D feature map: windowed or global
    attention, then an FFN, each behind a residual connection.

    Architecture:

    .. code-block:: text

        x [B, H, W, C]
          │
          ├──────────────────────────────┐
          ▼                               │
        Norm1                             │
          ▼                               │
        window partition (optional)       │
          ▼                               │
        WindowedAttentionWithRelPos       │
          ▼                               │
        window unpartition (optional)     │
          ▼                               │
        Add ◄──────────────────────────────┘
          │
          ├──────────────────────────────┐
          ▼                               │
        Norm2 → FFN                       │
          ▼                               │
        Add ◄──────────────────────────────┘
          ▼
        out [B, H, W, C]

    :param dim: Embedding dimension.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: FFN hidden dimension as ``int(dim * mlp_ratio)``.
        Defaults to 4.0.
    :type mlp_ratio: float
    :param qkv_bias: Whether the QKV projection carries a bias. Defaults to
        True.
    :type qkv_bias: bool
    :param use_rel_pos: Whether attention uses a relative position bias.
        Defaults to False.
    :type use_rel_pos: bool
    :param window_size: Attention window size; 0 or less means global
        attention over the whole feature map. Defaults to 0.
    :type window_size: int
    :param input_size: Input feature map resolution ``(H, W)``. Required for
        global attention with relative positions.
    :type input_size: Optional[Tuple[int, int]]
    :param normalization_type: Normalization variant. Defaults to
        ``'layer_norm'``.
    :type normalization_type: str
    :param ffn_type: FFN variant. Defaults to ``'mlp'``.
    :type ffn_type: str
    :param activation: FFN activation. Defaults to ``'gelu'``.
    :type activation: str
    :param kwargs: Additional :class:`keras.layers.Layer` arguments.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, dim)`.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, dim)`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        ffn_type: Literal['mlp', 'swiglu', 'geglu', 'glu'] = 'mlp',
        activation: str = 'gelu',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Store all configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.input_size = input_size
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.activation = deserialize_activation(activation)

        # CREATE all sub-layers in __init__
        self.norm1 = create_normalization_layer(normalization_type, name="norm1")

        # Use custom windowed attention with relative position bias
        self.attn = WindowedAttentionWithRelPos(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size <= 0 else (window_size, window_size),
            name="attention"
        )

        self.norm2 = create_normalization_layer(normalization_type, name="norm2")

        # DECISION plan-2026-07-30T140922-8af1028f/D-022: `create_ffn_layer`
        # discards `activation` for `ffn_type='swiglu'` (SwiGLUFFN takes none);
        # do not treat that as an error, its SiLU gate is definitional. See decisions.md.
        self.ffn = create_ffn_layer(
            ffn_type,
            name="ffn",
            **assemble_ffn_config(
                ffn_type,
                {
                    "hidden_dim": int(dim * mlp_ratio),
                    "output_dim": dim,
                    "activation": activation,
                },
            ),
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Builds all sub-layers.

        This explicit build step is crucial for composite layers to ensure
        correct weight restoration during model loading.

        Args:
            input_shape: Shape tuple of the input.
        """
        self.norm1.build(input_shape)
        # The input shape to attention is the same as the block's input
        self.attn.build(input_shape)
        self.norm2.build(input_shape)
        # The input shape to the FFN is also the same
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Forward pass for the ViT block.

        Args:
            x: Input tensor of shape (batch_size, height, width, dim).

        Returns:
            Output tensor of shape (batch_size, height, width, dim).
        """
        shortcut = x
        x = self.norm1(x)

        # Apply windowing if enabled
        if self.window_size > 0:
            H, W = ops.shape(x)[1], ops.shape(x)[2]
            x, pad_hw = self._window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse windowing if enabled
        if self.window_size > 0:
            x = self._window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x

    def _window_partition(
        self,
        x: keras.KerasTensor,
        window_size: int
    ) -> Tuple[keras.KerasTensor, Tuple[int, int]]:
        """
        Partitions the input feature map into non-overlapping windows.

        Args:
            x: Input tensor.
            window_size: Size of the window.

        Returns:
            Tuple of (windowed tensor, padded dimensions).
        """
        B, H, W, C = ops.shape(x)
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        Hp, Wp = H + pad_h, W + pad_w

        x = ops.reshape(x, (B, Hp // window_size, window_size, Wp // window_size, window_size, C))
        windows = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        windows = ops.reshape(windows, (-1, window_size, window_size, C))
        return windows, (Hp, Wp)

    def _window_unpartition(
        self,
        windows: keras.KerasTensor,
        window_size: int,
        pad_hw: Tuple[int, int],
        hw: Tuple[int, int]
    ) -> keras.KerasTensor:
        """
        Merges windows back into a feature map.

        Args:
            windows: Windowed tensor.
            window_size: Size of the window.
            pad_hw: Padded dimensions (Hp, Wp).
            hw: Original dimensions (H, W).

        Returns:
            Merged feature map.
        """
        Hp, Wp = pad_hw
        H, W = hw
        num_windows_h = Hp // window_size
        num_windows_w = Wp // window_size
        B = ops.shape(windows)[0] // (num_windows_h * num_windows_w)

        x = ops.reshape(windows, (B, num_windows_h, num_windows_w, window_size, window_size, -1))
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (B, Hp, Wp, -1))

        # Remove padding if it was added
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "use_rel_pos": self.use_rel_pos,
            "window_size": self.window_size,
            "input_size": self.input_size,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "activation": serialize_activation(self.activation),
        })
        return config


@register_dl_technique("dl_techniques.models.sam1.image_encoder")
class ImageEncoderViT(keras.Model):
    """
    The ViT image encoder: patch embedding, a stack of transformer blocks,
    and a stride-1 neck that projects to the output channel count.

    Architecture:

    .. code-block:: text

        image [B, img_size, img_size, in_chans]
              │
              ▼
        ┌───────────────────┐
        │ PatchEmbedding2D   │
        └─────────┬──────────┘
                   ▼  + pos_embed
        [B, grid, grid, embed_dim]
                   │
                   ▼
        ViTBlock x depth (global at global_attn_indexes,
                           windowed elsewhere)
                   │
                   ▼
        ┌───────────────────┐
        │ neck: 1x1 conv,    │
        │ norm, 3x3 conv,    │
        │ norm (stride 1)    │
        └─────────┬──────────┘
                   ▼
        out [B, grid, grid, out_chans]

    :param img_size: Size of the input image (assumed square).
    :type img_size: int
    :param patch_size: Size of the image patches.
    :type patch_size: int
    :param in_chans: Number of input channels, e.g. 3 for RGB.
    :type in_chans: int
    :param embed_dim: Patch embedding dimension.
    :type embed_dim: int
    :param depth: Number of transformer blocks.
    :type depth: int
    :param num_heads: Number of attention heads per block.
    :type num_heads: int
    :param mlp_ratio: FFN hidden-dimension ratio.
    :type mlp_ratio: float
    :param out_chans: Number of output channels from the neck.
    :type out_chans: int
    :param qkv_bias: Whether QKV projections carry a bias.
    :type qkv_bias: bool
    :param use_rel_pos: Whether attention uses a relative position bias.
        Defaults to True, matching reference SAM. The tables are
        zero-initialized, so enabling this is numerically inert until they
        are trained.
    :type use_rel_pos: bool
    :param window_size: Attention window size. ``0`` makes every block
        global, in which case ``global_attn_indexes`` is correctly empty.
    :type window_size: int
    :param global_attn_indexes: Indices of blocks that use global attention
        instead of windowed attention. Must be non-empty whenever
        ``window_size > 0``; reference SAM uses four evenly-spaced indices,
        e.g. ``(2, 5, 8, 11)`` at ``depth=12``.
    :type global_attn_indexes: Tuple[int, ...]
    :param normalization_type: Normalization variant used in blocks.
        Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param ffn_type: FFN variant used in blocks. Defaults to ``'mlp'``.
    :type ffn_type: str
    :param activation: FFN activation. Defaults to ``'gelu'``.
    :type activation: str
    :param kwargs: Additional :class:`keras.Model` arguments.
    :raises ValueError: If ``window_size > 0`` and ``global_attn_indexes`` is
        empty -- that configuration windows every block, leaving no global
        receptive field.

    Input shape:
        4D tensor with shape: `(batch_size, img_size, img_size, in_chans)`.

    Output shape:
        4D tensor with shape: `(batch_size, img_size/patch_size, img_size/patch_size, out_chans)`.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (),
        normalization_type: Literal['layer_norm', 'rms_norm', 'batch_norm'] = 'layer_norm',
        ffn_type: Literal['mlp', 'swiglu', 'geglu', 'glu'] = 'mlp',
        activation: str = 'gelu',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # DECISION plan-2026-08-03T191222-1d751f81/D-014: refuse only when
        # window_size > 0 and global_attn_indexes is empty, not whenever window_size == 0. See decisions.md.
        # DECISION plan-2026-08-03T191222-1d751f81/D-026: also require
        # window_size < grid_size -- a window covering the whole grid is already global. See decisions.md.
        grid_size = img_size // patch_size
        if 0 < window_size < grid_size and not global_attn_indexes:
            raise ValueError(
                f"Degenerate encoder configuration: window_size={window_size} "
                f"> 0 with an empty global_attn_indexes windows every one of "
                f"the {depth} blocks into {window_size}x{window_size} tiles of "
                f"the {grid_size}x{grid_size} token grid, so the encoder never "
                f"attains a global receptive field. Supply global_attn_indexes "
                f"(reference SAM uses 4 evenly-spaced indices, e.g. "
                f"(2, 5, 8, 11) at depth=12), set window_size=0 to make every "
                f"block global, or use window_size={grid_size} so a single "
                f"window covers the whole grid exactly. (window_size > "
                f"{grid_size} is accepted too, but is NOT recommended: the "
                f"relative-position tables are sized 2*window_size-1, so such "
                f"an encoder is neither weight-compatible with a "
                f"window_size=0 one nor free of zero-pad tokens in attention.)"
            )

        # Store all configuration parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_chans = out_chans
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.activation = deserialize_activation(activation)
        self.grid_size = img_size // patch_size

        # DECISION plan_2026-06-16_6e8c78a3/D-009: PatchEmbedding2D needs
        # flatten=False here, not the default True -- this encoder adds a 4D pos_embed and runs windowed attention on the spatial layout. See decisions.md.
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            flatten=False,
            name="patch_embed"
        )
        # `pos_embed` is a weight, not a layer, so it's created in `build`
        self.pos_embed = None

        self.blocks = []
        for i in range(depth):
            # Use windowed attention unless the index is in global_attn_indexes
            block_window_size = 0 if i in global_attn_indexes else window_size
            block = ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                window_size=block_window_size,
                input_size=(self.grid_size, self.grid_size),
                normalization_type=normalization_type,
                ffn_type=ffn_type,
                activation=activation,
                name=f"block_{i}"
            )
            self.blocks.append(block)

        # Neck module using factory for normalization
        self.neck = keras.Sequential(
            [
                layers.Conv2D(
                    filters=out_chans,
                    kernel_size=1,
                    use_bias=False,
                    name="neck_conv1"
                ),
                create_normalization_layer(normalization_type, name="neck_norm1"),
                layers.Conv2D(
                    filters=out_chans,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                    name="neck_conv2"
                ),
                create_normalization_layer(normalization_type, name="neck_norm2"),
            ],
            name="neck"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the positional embedding weight and build every sub-layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]

        .. note::
           DECISION plan-2026-08-19T163559-499b6f0e/D-122: the explicit
           sub-layer builds here are required for ``load_model``, which
           builds from the saved shape and restores before any call runs.
           Without them a reloaded encoder held 1 of 65 weights. See decisions.md.
        """
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.grid_size, self.grid_size, self.embed_dim),
            initializer="zeros",
            trainable=True
        )
        token_shape = (input_shape[0], self.grid_size, self.grid_size, self.embed_dim)
        self.patch_embed.build(input_shape)
        for blk in self.blocks:
            blk.build(token_shape)
        self.neck.build(token_shape)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass for the image encoder.

        Args:
            x: Input tensor of shape (batch_size, img_size, img_size, in_chans).
            training: Boolean, whether in training mode.

        Returns:
            Output embedding of shape (batch_size, grid_size, grid_size, out_chans).
        """
        # 1. Patch and Position Embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # 2. Transformer Blocks
        for blk in self.blocks:
            x = blk(x, training=training)

        # 3. Neck
        x = self.neck(x, training=training)
        return x

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the model for serialization.

        Returns:
            Configuration dictionary.
        """
        # Start with the base config which includes name, etc.
        config = super().get_config()
        # Update with all __init__ parameters
        config.update({
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "in_chans": self.in_chans,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "out_chans": self.out_chans,
            "qkv_bias": self.qkv_bias,
            "use_rel_pos": self.use_rel_pos,
            "window_size": self.window_size,
            "global_attn_indexes": self.global_attn_indexes,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "activation": serialize_activation(self.activation),
        })
        return config

# ---------------------------------------------------------------------
