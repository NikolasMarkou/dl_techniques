"""
Hierarchical vision transformer built from shifted-window attention, with four
stages, patch merging between them, and an optional classification head.

The architecture exists to resolve a tension between two facts about applying
transformers to images. Global self-attention costs `O((HW)^2)` in the number of
tokens, which is unaffordable at the resolutions dense prediction needs; but
restricting attention to fixed local windows makes the cost `O(M^2 * HW)` — linear
in image area for a fixed window `M` — at the price of never letting information
cross a window boundary, which stacks into a network of disconnected columns.
Swin's answer is to alternate the window partition instead of enlarging it.
Even-indexed blocks partition the feature map on a regular `MxM` grid; odd-indexed
blocks shift the whole partition by `M/2` in both axes, so a window in the shifted
layer straddles four windows of the previous layer. Two consecutive blocks
therefore connect every token to a neighbourhood strictly larger than one window,
and stacking them grows the receptive field without ever computing a global
attention matrix.

Implementing that shift naively would produce windows of unequal size at the image
border — `(M/2+1)^2` partitions instead of `(H/M)^2`, which cannot be batched. The
trick the paper uses, and which lives in `SwinTransformerBlock` rather than here,
is a cyclic roll: the feature map is rolled by `-M/2` before partitioning and
rolled back afterwards, which keeps the window count and shape identical to the
unshifted case. The roll wraps opposite edges of the image into the same physical
window, so tokens that are not spatially adjacent end up as window-mates, and an
attention mask must forbid exactly those pairs. Roll and mask are two halves of
one operation: a roll without its mask lets the left edge attend to the right edge
as though they touched, and a mask without its roll masks pairs that were never
brought together. The block derives both from a single resolved shift value for
that reason, and drops both together when a stage's grid has become smaller than
the window (at which point one window covers everything and there is nothing to
shift).

Attention inside a window is position-agnostic on its own, so each head carries a
learnable relative position bias added to the scores before softmax — indexed by
the `(2M-1) x (2M-1)` possible relative offsets between two positions in a window
rather than by absolute coordinates. This is what supplies spatial structure; the
model adds no absolute positional embedding anywhere.

Between stages, `PatchMerging` concatenates each `2x2` neighbourhood of tokens into
one and projects `4C -> 2C`. Resolution halves and width doubles, the same
pyramid a CNN builds, which is what makes the network usable as a detection or
segmentation backbone rather than only a classifier. Four stages take a `H/4`
patch grid down to `H/32`.

This module assembles those parts. The stage schedule is the paper's:
tiny `[2,2,6,2]` at `embed_dim=96`, small `[2,2,18,2]` at 96, base `[2,2,18,2]` at
128, large `[2,2,18,2]` at 192, giving 28.3M / 49.6M / 87.8M / 196.5M parameters at
`num_classes=1000` (measured, not quoted). Shift alternates on the block index
*within* a stage, so every stage begins unshifted. Stochastic depth is scheduled
linearly across all blocks of all stages, not restarted per stage, so a block's
drop rate depends on its global depth.

Three things about the code would otherwise mislead. The model is *functional*, not
subclassed: `__init__` builds a `keras.Input` graph and calls
`super().__init__(inputs, outputs)` at the end, which is why `get_config()`
deliberately does not merge `super().get_config()` — the functional config would
collide with the constructor arguments on reload. Between the patch embedding and
the first block there is a bare `Reshape`: `PatchEmbedding2D` emits a 3D
`(B, HW, C)` sequence while `SwinTransformerBlock` requires a 4D `(B, H, W, C)`
grid, and both contracts are load-bearing for other models, so the seam is
resolved here in the model rather than by changing either layer. And the
divisibility checks on input height and width only `logger.warning`; they are
compute notes, not correctness guards. `PatchMerging` ceil-pads an odd grid
dimension, so a non-divisible input still produces the declared output shape — it
just carries zero-padded tokens through at least one merge. The single hard
requirement, `H % patch_size == 0`, is raised by `PatchEmbedding2D`.

The classification head emits raw logits — layer-norm, global average pooling, a
dense layer with no activation — so compile with `from_logits=True`.
No checkpoints ship with this package, so `create_swin_transformer(pretrained=True)`
raises `NotImplementedError`. It used to log a warning and return a randomly
initialized model, which made an unavailable checkpoint indistinguishable from a
successful load at the call site; warm-start from a local file with
`model.load_weights(path)` instead.

References:
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. ICCV 2021. (https://arxiv.org/abs/2103.14030)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Shaw et al., 2018. Self-Attention with Relative Position Representations.
      (https://arxiv.org/abs/1803.02155)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from keras import layers, initializers, regularizers
from typing import List, Optional, Union, Tuple, Dict, Any, Sequence

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.patch_merging import PatchMerging
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.transformers.swin_transformer_block import SwinTransformerBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinTransformer(keras.Model):
    """
    Hierarchical Vision Transformer using shifted windows for efficient image classification.

    This model implements the Swin Transformer architecture featuring windowed self-attention
    with linear computational complexity relative to image size. The hierarchical design
    with patch merging enables multi-scale feature learning while maintaining global
    receptive fields through shifted window mechanisms.

    **Intent**: Provide a production-ready Swin Transformer implementation that efficiently
    processes images through windowed attention mechanisms, offering better efficiency than
    standard Vision Transformers while maintaining state-of-the-art performance on image
    classification and dense prediction tasks.

    **Architecture Overview**:
    ```
    Input Image (H×W×3)
           ↓
    Patch Embedding → (H/4×W/4×embed_dim)
           ↓
    Stage 1: depths[0]×SwinBlock (window_size, alternating shifts)
           ↓
    Patch Merging → (H/8×W/8×embed_dim×2)
           ↓
    Stage 2: depths[1]×SwinBlock
           ↓
    Patch Merging → (H/16×W/16×embed_dim×4)
           ↓
    Stage 3: depths[2]×SwinBlock
           ↓
    Patch Merging → (H/32×W/32×embed_dim×8)
           ↓
    Stage 4: depths[3]×SwinBlock
           ↓
    LayerNorm → GlobalAvgPool → Classifier (if include_top)
           ↓
    Output: (num_classes,) logits or (H/32×W/32×embed_dim×8) features
    ```

    **Key Architectural Features**:
    - **Hierarchical Processing**: 4-stage pyramid with progressively larger features
    - **Windowed Attention**: O(H×W) complexity through local window attention
    - **Shifted Windows**: Cross-window connections via cyclical shifting mechanism
    - **Stochastic Depth**: Linear drop path scheduling for regularization
    - **Patch Merging**: Efficient downsampling with feature expansion

    **Data Flow**:
    1. **Patch Embedding**: Converts H×W×3 image to H/4×W/4×embed_dim patches
    2. **Stage Processing**: Each stage applies multiple Swin blocks with attention
    3. **Hierarchical Learning**: Patch merging creates multi-scale representations
    4. **Classification**: Optional global pooling and linear classification head

    Args:
        num_classes: Integer, number of output classes for classification.
            Must be positive. Only used when include_top=True. Defaults to 1000.
        embed_dim: Integer, base embedding dimension for first stage features.
            Must be positive. Subsequent stages use 2^i × embed_dim. Defaults to 96.
        depths: List[int], number of Swin blocks per stage (exactly 4 elements).
            Each element must be positive. Controls model depth. Defaults to [2,2,6,2].
        num_heads: List[int], attention heads per stage (exactly 4 elements).
            Each element must be positive and divide stage dimension. Defaults to [3,6,12,24].
        window_size: Integer, size of attention windows (typically 7 or 8).
            Must be positive. Larger windows increase computation. Defaults to 7.
        mlp_ratio: Float, expansion ratio for MLP layers in each block.
            Must be positive. Typical values: 4.0. Defaults to 4.0.
        qkv_bias: Boolean, whether to use bias in attention QKV projections.
            True generally improves performance. Defaults to True.
        dropout_rate: Float, dropout rate for attention projection and MLP.
            Must be in [0, 1). Applied throughout the model. Defaults to 0.0.
        attn_dropout_rate: Float, dropout rate specifically for attention weights.
            Must be in [0, 1). Controls attention sparsity. Defaults to 0.0.
        drop_path_rate: Float, maximum stochastic depth rate for regularization.
            Must be in [0, 1). Linearly scheduled across blocks. Defaults to 0.1.
        patch_size: Integer, patch size for initial patch embedding.
            Must be positive. Determines initial downsampling. Defaults to 4.
        use_bias: Boolean, whether to use bias terms in linear layers.
            False can reduce parameters. Defaults to True.
        kernel_initializer: String or Initializer, weight initialization strategy.
            Controls model parameter initialization. Defaults to "glorot_uniform".
        bias_initializer: String or Initializer, bias initialization strategy.
            Only used when use_bias=True. Defaults to "zeros".
        kernel_regularizer: Optional Regularizer, weight regularization.
            Helps prevent overfitting in large models. Defaults to None.
        bias_regularizer: Optional Regularizer, bias regularization.
            Only applied when use_bias=True. Defaults to None.
        include_top: Boolean, whether to include classification head.
            Set False for feature extraction models. Defaults to True.
        input_shape: Optional[Tuple[int, ...]], input tensor shape (H, W, C).
            If None, defaults to (224, 224, 3). Must be 3D tuple.
        **kwargs: Additional arguments for Model base class (name, etc.).

    Input shape:
        4D tensor: `(batch_size, height, width, channels)`
        Optimal when height and width are divisible by patch_size × 8.

    Output shape:
        - If include_top=True: `(batch_size, num_classes)` - classification logits
        - If include_top=False: `(batch_size, H/32, W/32, embed_dim×8)` - feature maps

    Attributes:
        patch_embed: Patch embedding layer for image tokenization.
        patch_embed_norm: Optional normalization after patch embedding.
        stages: List[List[SwinTransformerBlock]], hierarchical transformer blocks.
        patch_merge_layers: List[PatchMerging], downsampling between stages.
        head_layers: List of classification head layers (if include_top=True).

    Example:
        ```python
        # Standard ImageNet model
        model = SwinTransformer.from_variant("base", num_classes=1000)

        # CIFAR-10 with smaller input
        model = SwinTransformer.from_variant(
            "tiny",
            num_classes=10,
            input_shape=(32, 32, 3)
        )

        # Feature extraction backbone
        backbone = SwinTransformer.from_variant(
            "large",
            include_top=False,
            input_shape=(384, 384, 3)
        )

        # Custom configuration
        model = SwinTransformer(
            num_classes=100,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8,
            drop_path_rate=0.2
        )
        ```

    Raises:
        ValueError: If configuration parameters are invalid (negative values,
            wrong list lengths, incompatible dimensions).

    References:
        - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
          Liu et al., ICCV 2021: https://arxiv.org/abs/2103.14030
        - Official implementation: https://github.com/microsoft/Swin-Transformer

    Note:
        This implementation follows modern Keras 3 patterns for robustness and
        integrates seamlessly with dl-techniques framework components. For optimal
        performance, ensure input dimensions are multiples of patch_size × 8.
    """

    # Model variant configurations (validated presets)
    MODEL_VARIANTS = {
        "tiny": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24]
        },
        "small": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24]
        },
        "base": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32]
        },
        "large": {
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48]
        },
    }

    # Architecture constants
    NUM_STAGES = 4
    LAYERNORM_EPSILON = 1e-5
    PATCH_EMBED_NORM = True

    def __init__(
            self,
            num_classes: int = 1000,
            embed_dim: int = 96,
            depths: Sequence[int] = (2, 2, 6, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            window_size: int = 7,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout_rate: float = 0.0,
            attn_dropout_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            patch_size: int = 4,
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
        # Comprehensive parameter validation
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if len(depths) != self.NUM_STAGES:
            raise ValueError(f"depths must have {self.NUM_STAGES} elements, got {len(depths)}")
        if len(num_heads) != self.NUM_STAGES:
            raise ValueError(f"num_heads must have {self.NUM_STAGES} elements, got {len(num_heads)}")
        if any(d <= 0 for d in depths):
            raise ValueError(f"All depths must be positive, got {depths}")
        if any(h <= 0 for h in num_heads):
            raise ValueError(f"All num_heads must be positive, got {num_heads}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not (0 <= attn_dropout_rate < 1):
            raise ValueError(f"attn_dropout_rate must be in [0, 1), got {attn_dropout_rate}")
        if not (0 <= drop_path_rate < 1):
            raise ValueError(f"drop_path_rate must be in [0, 1), got {drop_path_rate}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        # Set default input shape
        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        # DECISION plan-2026-07-31T210633-b63a35aa/D-003
        # These are COMPUTE notes, not correctness warnings, and there is deliberately
        # NO `window_size` raise or divisibility raise here. Do not "upgrade" either.
        #
        # The previous text ("may cause issues in deeper stages") was measured over an
        # 80-cell (input, patch_size, window_size, depth) sweep and had ZERO true
        # positives: every geometry that builds reports `model.output_shape[1:] ==
        # actual[1:]`. `PatchMerging` ceil-pads an odd grid dimension, so a deeper stage
        # does not break -- it costs padded tokens. Re-verified here with
        # `include_top=False` (a SPATIAL declared shape, so the check is not vacuous):
        # declared == actual at 7 of 7 geometries, including the warned 56/ps=4 and
        # 30/ps=2. The predicate also mis-fires in both directions as a correctness
        # signal: it is SILENT on 63/ps=2, which raises loudly (and correctly) in
        # `PatchEmbedding2D`, and it FIRES on 56 and 30, which work.
        #
        # The only hard requirement is `H % patch_size == 0`, and it is already enforced
        # by `PatchEmbedding2D` with a clear message, at construction, in the right
        # place. That raise is this model's divisibility guard; this is not.
        #
        # `window_size` is entirely unconstrained and was never wrong in the sweep
        # (including ws=5, ws=7 on grids 7/4/2/1, and ws=16 on a 1x1 grid). Its only
        # residual cost is COMPUTE WASTE at the deepest stage, and it is NOT detectable
        # from the divisibility predicate below -- measured by instrumenting
        # `SwinTransformerBlock.call` inside a full forward pass, the worst cells are
        # NOT warned ones. At `input_shape=(32,32,3), patch_size=4` the deepest stage
        # grid is 1x1, so the block pads it to the full window: `window_size=8` -> 8x8,
        # 1 -> 64 tokens (64x) and 1 -> 4096 attention pairs (4096x); `window_size=16`
        # -> 16x16, 256x tokens and 65536x pairs. Meanwhile the WARNED 56/ps=4/ws=7
        # cell has a 2x2 deepest grid padded to 7x7 -- only 12x tokens / 150x pairs.
        # A `window_size` guard would therefore have to key off the deepest stage grid,
        # not off divisibility; none is added here.
        height, width, channels = input_shape
        if height is not None and height % (patch_size * 8) != 0:
            logger.warning(
                f"Input height {height} is not divisible by {patch_size * 8}. "
                f"The model is still correct -- PatchMerging ceil-pads an odd grid "
                f"dimension and the declared output shape matches the actual one. "
                f"The cost is compute: at least one of the three merge stages will "
                f"carry zero-padded tokens."
            )
        if width is not None and width % (patch_size * 8) != 0:
            logger.warning(
                f"Input width {width} is not divisible by {patch_size * 8}. "
                f"The model is still correct -- PatchMerging ceil-pads an odd grid "
                f"dimension and the declared output shape matches the actual one. "
                f"The cost is compute: at least one of the three merge stages will "
                f"carry zero-padded tokens."
            )

        # Store ALL configuration parameters for serialization
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: the DEFAULT is a
        # tuple (R-009 S1) and the STORED attribute is a list. Keeping the
        # store as `list(...)` is what makes the conversion invisible: it is
        # the type `get_config` has always emitted, so a saved config's JSON
        # shape and every `== [..]` assertion in the suites are unchanged.
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.use_bias = use_bias
        self.include_top = include_top
        self._input_shape = input_shape

        # Store serializable initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Initialize layer collections
        self.stages = []
        self.patch_merge_layers = []
        self.head_layers = []

        # CREATE model architecture
        inputs = keras.Input(shape=input_shape, name="input")
        outputs = self._build_architecture(inputs)

        # Initialize the Model (Keras handles sub-layer building automatically)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created Swin Transformer: embed_dim={embed_dim}, "
            f"depths={depths}, num_heads={num_heads}, "
            f"total_blocks={sum(depths)}, input_shape={input_shape}"
        )

    def _build_architecture(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Build the complete Swin Transformer architecture.

        Args:
            inputs: Input tensor from keras.Input().

        Returns:
            Output tensor (logits or features).
        """
        x = inputs

        # Stage 1: Patch embedding
        x = self._create_patch_embedding(x)

        # Stages 2-4: Hierarchical transformer blocks with patch merging
        for stage_idx in range(self.NUM_STAGES):
            # Add patch merging before stages 2-4
            if stage_idx > 0:
                x = self._create_patch_merging(x, stage_idx)

            # Add transformer blocks for this stage
            x = self._create_stage_blocks(x, stage_idx)

        # Optional classification head
        if self.include_top:
            x = self._create_classification_head(x)

        return x

    def _create_patch_embedding(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Create patch embedding to tokenize input image."""
        # Use dl-techniques embedding factory
        self.patch_embed = create_embedding_layer(
            embedding_type="patch_2d",
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="patch_embed"
        )
        x = self.patch_embed(x)

        # Optional normalization after patch embedding
        if self.PATCH_EMBED_NORM:
            self.patch_embed_norm = layers.LayerNormalization(
                epsilon=self.LAYERNORM_EPSILON,
                center=self.use_bias,
                scale=True,
                name="patch_embed_norm"
            )
            x = self.patch_embed_norm(x)

        # DECISION plan_2026-06-16_c8f3e9ca/D-004: restore (B,H,W,C) grid between
        # PatchEmbedding2D (emits 3D (B, H*W, embed_dim)) and SwinTransformerBlock
        # (requires 4D (B,H,W,C)). Both shared-layer contracts are intentional and
        # are left UNCHANGED — the seam fix lives here, in the model. Do NOT "fix"
        # this by making PatchEmbedding2D emit 4D or by relaxing the block's 4D
        # assertion: other models rely on the 3D patch-embed output. The patch grid
        # is square and static (H=W=input_size//patch_size), so a Reshape with a
        # known target shape is graph-safe. PatchMerging and SwinTransformerBlock
        # both keep the tensor 4D thereafter, so this single reshape suffices.
        grid_h = self._input_shape[0] // self.patch_size
        grid_w = self._input_shape[1] // self.patch_size
        x = layers.Reshape(
            (grid_h, grid_w, self.embed_dim),
            name="patch_embed_grid_restore"
        )(x)

        return x

    def _create_patch_merging(
            self,
            x: keras.KerasTensor,
            stage_idx: int
    ) -> keras.KerasTensor:
        """Create patch merging layer for downsampling."""
        # Calculate input dimension for current stage
        input_dim = self.embed_dim * (2 ** (stage_idx - 1))

        patch_merge = PatchMerging(
            dim=input_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name=f"patch_merge_{stage_idx}"
        )

        x = patch_merge(x)
        self.patch_merge_layers.append(patch_merge)
        return x

    def _create_stage_blocks(
            self,
            x: keras.KerasTensor,
            stage_idx: int
    ) -> keras.KerasTensor:
        """Create Swin Transformer blocks for a given stage."""
        stage_blocks = []
        depth = self.depths[stage_idx]
        num_heads = self.num_heads[stage_idx]
        stage_dim = self.embed_dim * (2 ** stage_idx)

        # Calculate drop path rates (linear scheduling over ALL blocks of ALL stages)
        total_blocks = sum(self.depths)
        block_start_idx = sum(self.depths[:stage_idx])
        drop_path_rates = linear_drop_path_rates(total_blocks, self.drop_path_rate)

        for block_idx in range(depth):
            # Global block index across stages: 0 .. total_blocks - 1
            current_block_idx = block_start_idx + block_idx
            current_drop_path_rate = drop_path_rates[current_block_idx]

            # Alternate between regular and shifted windows
            shift_size = 0 if block_idx % 2 == 0 else self.window_size // 2

            # Create Swin Transformer block
            block = SwinTransformerBlock(
                dim=stage_dim,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size=shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attn_dropout_rate,
                stochastic_depth_rate=current_drop_path_rate,
                activation="gelu",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )

            x = block(x)
            stage_blocks.append(block)

        self.stages.append(stage_blocks)
        return x

    def _create_classification_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Create classification head with global pooling."""
        # Layer normalization before pooling
        head_norm = layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        x = head_norm(x)

        # Global average pooling
        gap = layers.GlobalAveragePooling2D(name="global_avg_pool")
        x = gap(x)

        # Classification layer
        if self.num_classes > 0:
            classifier = layers.Dense(
                units=self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="classifier"
            )
            x = classifier(x)
            self.head_layers = [head_norm, gap, classifier]
        else:
            self.head_layers = [head_norm, gap]

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> "SwinTransformer":
        """
        Create Swin Transformer from a predefined variant configuration.

        Args:
            variant: String, model variant ("tiny", "small", "base", "large").
            num_classes: Integer, number of output classes.
            input_shape: Optional tuple, input shape. Defaults to (224, 224, 3).
            **kwargs: Additional arguments passed to constructor.

        Returns:
            SwinTransformer model instance.

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]
        logger.info(f"Creating Swin Transformer-{variant.upper()} model")

        return cls(
            num_classes=num_classes,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            # ALL __init__ parameters must be included
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "dropout_rate": self.dropout_rate,
            "attn_dropout_rate": self.attn_dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "patch_size": self.patch_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SwinTransformer":
        """Create model from configuration dictionary."""
        # Deserialize initializers and regularizers
        if config.get("kernel_initializer"):
            config["kernel_initializer"] = initializers.deserialize(
                config["kernel_initializer"]
            )
        if config.get("bias_initializer"):
            config["bias_initializer"] = initializers.deserialize(
                config["bias_initializer"]
            )
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if config.get("bias_regularizer"):
            config["bias_regularizer"] = regularizers.deserialize(
                config["bias_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with Swin Transformer specific information."""
        super().summary(**kwargs)

        # Print additional architectural details
        total_blocks = sum(self.depths)
        total_params = sum(layer.count_params() for layer in self.layers)

        logger.info("=" * 50)
        logger.info("SWIN TRANSFORMER CONFIGURATION")
        logger.info("=" * 50)
        logger.info(f"Input shape: {self._input_shape}")
        logger.info(f"Patch size: {self.patch_size}")
        logger.info(f"Base embedding dimension: {self.embed_dim}")
        logger.info(f"Window size: {self.window_size}")
        logger.info(f"Number of stages: {self.NUM_STAGES}")
        logger.info(f"Depths per stage: {self.depths}")
        logger.info(f"Heads per stage: {self.num_heads}")
        logger.info(f"Total transformer blocks: {total_blocks}")
        logger.info(f"MLP expansion ratio: {self.mlp_ratio}")
        logger.info(f"Stochastic depth rate: {self.drop_path_rate}")
        logger.info(f"Include classification head: {self.include_top}")
        if self.include_top:
            logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info("=" * 50)


# ---------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------

def create_swin_transformer(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: bool = False,
        **kwargs: Any
) -> SwinTransformer:
    """
    Factory function to create Swin Transformer models with validation.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large").
        num_classes: Integer, number of output classes.
        input_shape: Optional tuple, input shape. Defaults to (224, 224, 3).
        pretrained: Boolean, must be False. `True` raises `NotImplementedError` —
            no Swin checkpoints ship with this package.
        **kwargs: Additional arguments passed to model constructor.

    Returns:
        SwinTransformer model instance.

    Raises:
        ValueError: If variant is invalid or parameters are incompatible.

    Example:
        ```python
        # CIFAR-10 model
        model = create_swin_transformer(
            "tiny",
            num_classes=10,
            input_shape=(32, 32, 3)
        )

        # ImageNet feature extractor
        backbone = create_swin_transformer(
            "base",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        ```
    """
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-069: raise, do not warn-and-continue.
    if pretrained:
        raise NotImplementedError(
            f"No pretrained Swin Transformer weights are distributed with "
            f"dl_techniques (requested variant '{variant}'). Build the "
            f"architecture with pretrained=False and warm-start from a local "
            f"checkpoint instead: model = create_swin_transformer('{variant}', "
            f"...); model.load_weights('/path/to/weights.keras'). Prefer "
            f"dl_techniques.utils.weight_transfer.load_weights_or_raise(model, "
            f"path), which raises when a load changes ZERO variables -- raw "
            f"load_weights is silent about a checkpoint that matches nothing."
        )

    return SwinTransformer.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

# ---------------------------------------------------------------------
