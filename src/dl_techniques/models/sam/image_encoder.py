"""
SAM Image Encoder (ViT) Implementation
=================================================

Implementation of the Vision Transformer (ViT)
based image encoder used in the Segment Anything Model (SAM). It is meticulously
crafted to follow modern Keras best practices for creating composite,
serializable, and production-ready layers and models.

**Intent**: To create a robust and production-ready ViT backbone that can
process images into high-dimensional embeddings. The implementation is designed
to be configurable, supporting different model variants (e.g., Base, Large, Huge)
while maintaining full serialization capabilities by strictly following the
"Create vs. Build" lifecycle pattern.

**Architecture**: The encoder consists of four main parts:
1.  **Patch Embedding**: Converts an input image into a sequence of patch
    embeddings using a single `Conv2D` layer.
2.  **Positional Embedding**: Adds a learnable absolute positional embedding
    to the patch embeddings to retain spatial information.
3.  **Transformer Blocks**: A series of transformer blocks that process the
    sequence of embeddings. These blocks use the configurable TransformerLayer
    from dl_techniques with windowed or global attention.
4.  **Neck**: A final feature-refining module that upsamples the feature map
    resolution using convolutional and normalization layers to produce the
    final output embedding.

**Data Flow**:
```
Input Image (B, H, W, C)
      │
      v
PatchEmbedding (Conv2D) -> (B, H/p, W/p, D)
      │
      + PositionalEmbedding (learnable weight)
      │
      v
Sequence of TransformerLayer Layers
  - Attention (Windowed or Global) with Relative Positional Bias
  - Feed-Forward Network
  - Residual Connections
  - Normalization
      │
      v
Neck (Conv2D -> LayerNorm -> Conv2D -> LayerNorm)
      │
      v
Output Embedding (B, H_emb, W_emb, D_out)
```

**Usage Example**:
```python
import keras
import numpy as np

# Instantiate the model for a 1024x1024 image
# This is the "Huge" variant configuration
encoder = ImageEncoderViT(
    img_size=1024,
    patch_size=16,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    out_chans=256,
    use_rel_pos=True,
    window_size=14,
    global_attn_indexes=(7, 15, 23, 31),
)

# Create a dummy input tensor
dummy_image = np.random.rand(1, 1024, 1024, 3).astype("float32")
input_tensor = keras.ops.convert_to_tensor(dummy_image)

# Get the image embedding
embedding = encoder(input_tensor)
print(f"Output embedding shape: {embedding.shape}")
# Expected output: Output embedding shape: (1, 64, 64, 256)

# Test serialization
model_path = "image_encoder.keras"
encoder.save(model_path)
loaded_encoder = keras.models.load_model(model_path)
print("Model serialized and loaded successfully.")

# Verify output consistency
embedding_from_loaded = loaded_encoder(input_tensor)
np.testing.assert_allclose(
    keras.ops.convert_to_numpy(embedding),
    keras.ops.convert_to_numpy(embedding_from_loaded),
    rtol=1e-6, atol=1e-6
)
print("Outputs are consistent after serialization.")
```

**References**:
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers
  for Image Recognition at Scale. *ICLR*.
- Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer
  using Shifted Windows. *ICCV*. (For window attention concepts).
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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WindowedAttentionWithRelPos(layers.Layer):
    """
    Multi-Head Self-Attention with optional Relative Positional Embeddings and Windowing.

    This layer wraps the attention factory to provide windowed attention with relative
    positional bias, which is crucial for capturing spatial relationships in vision tasks.

    **Intent**: To model relationships between patches in a window (or globally) and
    enrich patch representations with contextual information, while supporting relative
    positional encodings for better spatial understanding.

    **Architecture**:
    1.  Input is linearly projected to Query (Q), Key (K), and Value (V).
    2.  Attention scores are computed: `softmax(Q @ K^T / sqrt(d_k))`.
    3.  If `use_rel_pos` is True, a learnable relative positional bias is added
        to the attention scores before the softmax operation.
    4.  The attention scores are used to create a weighted sum of the Values (V).
    5.  The output is passed through a final linear projection.

    Args:
        dim: Integer, the input and output dimension of tokens.
        num_heads: Integer, the number of attention heads. Defaults to 8.
        qkv_bias: Boolean, if True, add a learnable bias to Q, K, V projections.
            Defaults to True.
        use_rel_pos: Boolean, if True, add relative positional embeddings to the
            attention scores. Defaults to False.
        input_size: Tuple of two integers, the height and width of the input
            feature map. Required if `use_rel_pos` is True to initialize the
            relative position bias tables.
        **kwargs: Additional `keras.layers.Layer` arguments.

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
        Creates the layer's weights and builds its sub-layers.

        Args:
            input_shape: Shape tuple of the input.
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
            # DECISION plan-2026-08-03T191222-1d751f81/D-007: `rel_pos` is
            # `(L, C)` = (relative-distance positions, head channels) and ONLY
            # the distance axis may be interpolated. Reshape to a 4-D BATCHED
            # image `(1, 1, L, C)` so `ops.image.resize` maps L -> width and C
            # -> channels, then resize the width axis alone with `size=(1, D)`.
            # Do NOT feed a 3-D `(1, C, L)` tensor here: `ops.image.resize`
            # reads a 3-D input as UNBATCHED `(h, w, c)`, so `(1, C, L)` is
            # read as h=1/w=C/c=L, it resizes the CHANNEL table instead of the
            # distance table, and the following squeeze raises
            # ValueError("Cannot squeeze axis=0, because the dimension is not
            # 1."). See decisions.md D-007.
            shape = ops.shape(rel_pos)
            rel_pos_resized = ops.image.resize(
                ops.reshape(rel_pos, (1, 1, shape[0], shape[1])),
                size=(1, max_rel_dist),
                interpolation='bilinear'
            )
            rel_pos_resized = ops.reshape(rel_pos_resized, (max_rel_dist, shape[1]))
        else:
            rel_pos_resized = rel_pos

        # Calculate relative coordinates.
        # DECISION plan_2026-06-15_e6a0391c/D-004: cast arange (int32) to float32
        # before scaling by the Python float ratio — multiplying an int32 tensor
        # by a float raised "Cannot convert 1.0 to EagerTensor of dtype int32".
        q_coords = ops.cast(ops.expand_dims(ops.arange(q_size), axis=1), "float32") * max(k_size / q_size, 1.0)
        k_coords = ops.cast(ops.expand_dims(ops.arange(k_size), axis=0), "float32") * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + float(k_size - 1) * max(q_size / k_size, 1.0)

        # Gather the embeddings using the coordinates.
        # `keras.ops.gather` does not exist on keras 3.8; `ops.take(..., axis=0)`
        # is the row-gather. (Verified by execution, not assumed: an earlier
        # comment here framed this as a keras.ops limitation in general, which
        # misled a later reader into thinking row-gathering was unavailable.)
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

        # Compute relative biases for height and width.
        # DECISION plan_2026-06-15_e6a0391c/D-004: rel_w output label was a typo
        # ('x' not present in inputs → einsum ValueError); must be 'k' like rel_h.
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


@keras.saving.register_keras_serializable()
class ViTBlock(layers.Layer):
    """
    Transformer Block for the Vision Transformer with Windowing Support.

    This layer implements a standard transformer block with pre-normalization
    (pre-LN) and support for windowed attention for computational efficiency.
    It uses the factory patterns from dl_techniques for attention, FFN, and normalization.

    **Intent**: To serve as the primary building block of the ViT, responsible
    for processing sequences of tokens and refining their representations through
    self-attention and feed-forward networks.

    **Architecture**:
    ```
    Input
      |
      +------------------------ (Residual Connection 1)
      |
      v
    Norm1
      |
      v
    Attention (Windowed or Global with Relative Position Bias)
      |
      v
    Add (Input + Attention Output)
      |
      +------------------------ (Residual Connection 2)
      |
      v
    Norm2
      |
      v
    FFN (Feed-Forward Network)
      |
      v
    Add (Previous Sum + FFN Output)
      |
      v
    Output
    ```

    Args:
        dim: Integer, the embedding dimension.
        num_heads: Integer, number of attention heads.
        mlp_ratio: Float, determines the hidden dimension of the FFN as
            `int(dim * mlp_ratio)`. Defaults to 4.0.
        qkv_bias: Boolean, whether to use bias in the QKV projection. Defaults to True.
        use_rel_pos: Boolean, whether to use relative positional embeddings in
            attention. Defaults to False.
        window_size: Integer, size of the attention window. If 0 or less, global
            attention is used over the entire feature map. Defaults to 0.
        input_size: Tuple of integers, the input feature map resolution (H, W).
            Required for global attention with relative positions.
        normalization_type: String, type of normalization to use. Defaults to 'layer_norm'.
        ffn_type: String, type of FFN to use. Defaults to 'mlp'.
        activation: String, activation function for FFN. Defaults to 'gelu'.
        **kwargs: Additional `keras.layers.Layer` arguments.

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
        self.activation = activation

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

        # DECISION plan-2026-07-30T140922-8af1028f/D-022
        # `activation` is this block's own generic default, pre-filtered
        # against what `ffn_type` accepts. This is NOT hypothetical here:
        # `ffn_type` is declared `Literal['mlp', 'swiglu', 'geglu', 'glu']`
        # and `SwiGLUFFN` takes no `activation` at all, so `swiglu` -- inside
        # this block's own documented surface -- silently dropped it, and
        # `create_ffn_layer` now RAISES on a dropped key. DISCARD is the right
        # semantics for `swiglu` (its SiLU gate is definitional, and there is
        # no renamed analogue to forward it to).
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
            "activation": self.activation,
        })
        return config


@keras.saving.register_keras_serializable()
class ImageEncoderViT(keras.Model):
    """
    The Vision Transformer (ViT) Image Encoder for SAM.

    This model takes an image as input and outputs a grid of powerful feature
    embeddings. It follows a standard ViT architecture but incorporates windowed
    attention in most of its blocks for computational efficiency.

    **Intent**: To serve as a powerful and scalable image feature extractor,
    capable of producing high-quality embeddings for downstream tasks like
    segmentation.

    Args:
        img_size: Integer, the size of the input image (assumed square).
        patch_size: Integer, the size of the image patches.
        in_chans: Integer, the number of input channels (e.g., 3 for RGB).
        embed_dim: Integer, the patch embedding dimension.
        depth: Integer, the number of transformer blocks.
        num_heads: Integer, the number of attention heads in each block.
        mlp_ratio: Float, the ratio for the FFN hidden dimension.
        out_chans: Integer, the number of output channels from the neck module.
        qkv_bias: Boolean, whether to use bias in QKV projections.
        use_rel_pos: Boolean, whether to use relative positional embeddings.
            Defaults to `True`, matching reference SAM (all released variants
            set it). Note that `rel_pos_h`/`rel_pos_w` are zero-initialized, so
            enabling this is numerically inert until the tables are trained.
        window_size: Integer, the size for windowed attention. Set to `0` to
            make every block global, in which case `global_attn_indexes` is
            correctly empty.
        global_attn_indexes: Tuple of integers, indices of `ViTBlock`s that
            should use global attention instead of windowed attention. Must be
            non-empty whenever `window_size > 0`; reference SAM uses four
            evenly-spaced indices (e.g. `(2, 5, 8, 11)` at `depth=12`).

    Raises:
        ValueError: If `window_size > 0` and `global_attn_indexes` is empty —
            that configuration windows every block, so the encoder never
            attains a global receptive field.
        normalization_type: String, type of normalization to use in blocks.
            Defaults to 'layer_norm'.
        ffn_type: String, type of FFN to use in blocks. Defaults to 'mlp'.
        activation: String, activation function for FFN. Defaults to 'gelu'.
        **kwargs: Additional `keras.Model` arguments.

    Input shape:
        4D tensor with shape: `(batch_size, img_size, img_size, in_chans)`.

    Output shape:
        4D tensor with shape: `(batch_size, img_size/patch_size, img_size/patch_size, out_chans)`.
        The neck is a pair of 1x1/3x3 convolutions at stride 1, so it changes the
        channel count only; the spatial grid is set by the patch embedding alone
        (measured: `(1, 64, 64, out_chans)` at `img_size=1024, patch_size=16`).
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

        # DECISION plan-2026-08-03T191222-1d751f81/D-014: refuse the
        # windowed-only encoder. `window_size > 0` with an EMPTY
        # `global_attn_indexes` windows every block, so the receptive field is
        # never global — the shipped defaults did exactly this. Do NOT
        # "simplify" this to an unconditional `global_attn_indexes` non-empty
        # requirement: `window_size == 0` already makes every block global and
        # an empty tuple is then CORRECT, not degenerate (measured: 4/4 global
        # blocks). The `window_size > 0` conjunct is the discriminating half.
        #
        # DECISION plan-2026-08-03T191222-1d751f81/D-026 (F-R3): the SECOND
        # conjunct, `window_size < grid_size`, is equally load-bearing and was
        # missing. A window at least as large as the token grid already covers
        # the whole grid, so every block IS global and an empty
        # `global_attn_indexes` is CORRECT — measured at the legitimate
        # `img_size=224, patch_size=16, window_size=14` (grid 14x14), which the
        # narrower guard refused. Do NOT drop either conjunct: without the
        # first, a global-only encoder (`window_size=0`) is refused; without
        # this one, a window-covers-grid encoder is. See decisions.md D-014,
        # D-026.
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
        self.activation = activation
        self.grid_size = img_size // patch_size

        # CREATE all sub-layers in __init__.
        # DECISION plan_2026-06-16_6e8c78a3/D-009: reuse the shared
        # PatchEmbedding2D with flatten=False (returns the 4D spatial grid) in
        # place of an inline duplicate. Do NOT use the default flatten=True here
        # — the SAM encoder adds a 4D pos_embed and runs windowed attention on
        # the spatial layout. See decisions.md D-009.
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
        Creates the model's own weights.

        For `keras.Model`, sub-layers are built automatically on the first
        call. We only need to create weights that belong directly to this model,
        like the positional embedding.

        Args:
            input_shape: Shape tuple of the input.
        """
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.grid_size, self.grid_size, self.embed_dim),
            initializer="zeros",
            trainable=True
        )
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
            "activation": self.activation,
        })
        return config

# ---------------------------------------------------------------------
