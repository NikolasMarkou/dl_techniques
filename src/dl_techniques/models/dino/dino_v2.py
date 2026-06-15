"""
DINOv2 Vision Transformer Implementation - Modern Keras 3 Patterns

This module provides a complete implementation of the DINOv2 (DINO version 2) Vision Transformer
architecture using modern Keras 3 best practices following the refined guide patterns.

Key Features:
- Configurable ViT backbone with multiple size variants (Tiny, Small, Base, Large, Giant)
- Support for register tokens for improved attention quality
- LearnableMultiplier for layer scaling (LayerScale replacement)
- Stochastic depth regularization with proper linear decay
- Flexible positional embedding with interpolation support
- Memory-efficient attention mechanisms
- Pre-normalization architecture for better gradient flow
- Modern Keras 3 patterns with explicit sub-layer building

The implementation is designed to be compatible with both self-supervised pre-training
(DINO, iBOT, KoLeo losses) and downstream supervised fine-tuning tasks.

Architecture Overview:
```
Input Image (B, H, W, C)
     ↓
Patch Embedding → (B, N, D)
     ↓
[CLS] + [REG] + Patches + PosEmb → (B, 1+R+N, D)
     ↓
DINOv2Block_1 → ... → DINOv2Block_L
     ↓
Layer Normalization
     ↓
CLS Token Features (B, D)
```

Where:
- B: Batch size
- H, W: Image height and width
- C: Input channels (typically 3)
- N: Number of patches = (H/P) * (W/P)
- P: Patch size (typically 14)
- D: Embedding dimension
- R: Number of register tokens (0 or 4)
- L: Number of transformer layers (depth)
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers, initializers
from typing import Optional, Union, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.embedding.class_token import ClassTokenPrepend
from dl_techniques.layers.embedding.mask_token import MaskTokenApply
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.layer_scale import LearnableMultiplier


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOv2Block(keras.layers.Layer):
    """
    DINOv2 Transformer Block with LearnableMultiplier scaling and configurable components.

    This block implements the DINOv2 transformer architecture with:
    - Pre-normalization layout (LayerNorm → Attention → Residual)
    - LearnableMultiplier for training stability (replaces LayerScale)
    - Configurable attention mechanisms via factory
    - Configurable FFN types via factory
    - Optional stochastic depth regularization

    **Architecture**:
    ```
    Input x (B, N, D) ────────────────────────────────────────────────────+
       ↓                                                        │
    LayerNorm → MultiHeadAttention → LearnableMultiplier → DropPath ──(+)─→ x_mid
       ↓                                                                   │
    LayerNorm → FFN → LearnableMultiplier → DropPath ──────────────────────(+)─→ Output
    ```

    **Intent**: Implement the core transformer block used in DINOv2 with modern
    enhancements like learnable scaling and stochastic depth for improved training
    stability and regularization.

    Args:
        dim: Embedding dimension. Must be positive and divisible by num_heads.
        num_heads: Number of attention heads. Must be positive.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension. Must be positive.
        attention_type: Type of attention mechanism ('multi_head_attention', etc.).
        ffn_type: Type of FFN ('mlp', 'swiglu', etc.).
        normalization_type: Type of normalization ('layer_norm', 'rms_norm', etc.).
        qkv_bias: Whether to use bias in QKV projection.
        proj_bias: Whether to use bias in attention projection.
        ffn_bias: Whether to use bias in FFN layers.
        stochastic_depth_rate: Stochastic depth drop probability.
        init_values: LearnableMultiplier initialization value (None disables scaling).
        attention_dropout: Dropout rate for attention.
        ffn_dropout: Dropout rate for FFN.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`.
        Same shape as input due to residual connections.

    Example:
        ```python
        # Standard DINOv2 block
        block = DINOv2Block(
            dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            stochastic_depth_rate=0.1,
            init_values=1e-5
        )

        # With SwiGLU FFN (for giant model)
        block = DINOv2Block(
            dim=1536,
            num_heads=24,
            ffn_type='swiglu',
            stochastic_depth_rate=0.3
        )
        ```
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            attention_type: str = 'multi_head',
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            qkv_bias: bool = True,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            stochastic_depth_rate: float = 0.0,  # Renamed from drop_path_rate
            init_values: Optional[float] = None,
            attention_dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        # Store all configuration
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.stochastic_depth_rate = stochastic_depth_rate  # Renamed
        self.init_values = init_values
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout

        # Create sub-layers in __init__ following Modern Keras 3 patterns

        # Normalization layers
        self.norm1 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm1"
        )
        self.norm2 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm2"
        )

        # Attention layer - map parameters appropriately
        attention_args = {
            'num_heads': self.num_heads,
            'dropout_rate': self.attention_dropout
        }

        if self.attention_type == 'multi_head':
            attention_args['dim'] = self.dim
            attention_args['use_bias'] = self.qkv_bias
        else:
            attention_args['dim'] = self.dim

        self.attention = create_attention_layer(
            attention_type=self.attention_type,
            name="attention",
            **attention_args
        )

        # FFN layer
        hidden_dim = int(self.dim * self.mlp_ratio)
        ffn_args = {
            'output_dim': self.dim,
            'dropout_rate': self.ffn_dropout,
            'use_bias': self.ffn_bias
        }

        if self.ffn_type in ['mlp', 'glu', 'geglu']:
            ffn_args['hidden_dim'] = hidden_dim
        elif self.ffn_type in ['swiglu']:
            ffn_args['ffn_expansion_factor'] = self.mlp_ratio

        self.ffn = create_ffn_layer(
            ffn_type=self.ffn_type,
            name="ffn",
            **ffn_args
        )

        # LearnableMultiplier for layer scaling (replaces LayerScale)
        if self.init_values is not None:
            self.ls1 = LearnableMultiplier(
                multiplier_type='CHANNEL',
                initializer=initializers.Constant(self.init_values),
                constraint='non_neg',
                name="ls1"
            )
            self.ls2 = LearnableMultiplier(
                multiplier_type='CHANNEL',
                initializer=initializers.Constant(self.init_values),
                constraint='non_neg',
                name="ls2"
            )
        else:
            self.ls1 = None
            self.ls2 = None

        # Stochastic depth (optional)
        if self.stochastic_depth_rate > 0.0:
            self.drop_path = StochasticDepth(self.stochastic_depth_rate, name="drop_path")
        else:
            self.drop_path = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the transformer block with all sub-components.

        **Critical**: Explicitly builds all sub-layers for proper serialization.
        Following the modern Keras 3 pattern from the refined guide.
        """
        logger.debug(f"Building DINOv2Block with input_shape: {input_shape}")

        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build attention layer
        self.attention.build(input_shape)

        # Build FFN layer
        self.ffn.build(input_shape)

        # Build LearnableMultiplier layers if used
        if self.ls1 is not None:
            self.ls1.build(input_shape)
            self.ls2.build(input_shape)

        # Build stochastic depth if used
        if self.drop_path is not None:
            self.drop_path.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the transformer block."""
        # Pre-norm attention block
        x = self.norm1(inputs, training=training)
        x = self.attention(x, training=training)

        if self.ls1 is not None:
            x = self.ls1(x, training=training)

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        x = inputs + x

        # Pre-norm FFN block
        y = self.norm2(x, training=training)
        y = self.ffn(y, training=training)

        if self.ls2 is not None:
            y = self.ls2(y, training=training)

        if self.drop_path is not None:
            y = self.drop_path(y, training=training)

        x = x + y

        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the layer."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
            "qkv_bias": self.qkv_bias,
            "proj_bias": self.proj_bias,
            "ffn_bias": self.ffn_bias,
            "stochastic_depth_rate": self.stochastic_depth_rate,  # Updated
            "init_values": self.init_values,
            "attention_dropout": self.attention_dropout,
            "ffn_dropout": self.ffn_dropout,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOv2VisionTransformer(keras.Model):
    """
    DINOv2 Vision Transformer backbone implementation following Modern Keras 3 patterns.

    This implementation provides a complete DINOv2 ViT backbone using the functional API
    pattern from the refined guide, similar to the ConvNeXt V1 paradigm. It creates
    the entire model architecture in a _build_model method using keras.Input and
    functional connections.

    **Architecture**:
    ```
    Input Image (B, H, W, C)
         ↓
    PatchEmbed → (B, N, D)
         ↓
    [CLS] + [REG]₀₋ᴿ + Patches + PosEmbed → (B, 1+R+N, D)
         ↓
    TransformerBlock₁ → ... → TransformerBlockₗ
         ↓
    LayerNorm → Split: [CLS] | [REG] | [Patches]
    ```

    **Intent**: Provide the core Vision Transformer backbone used in DINOv2,
    supporting both self-supervised pre-training and downstream fine-tuning
    with configurable architecture and modern training enhancements.

    Args:
        image_size: Input image size (int or tuple). Must be positive.
        patch_size: Patch size for embedding (int or tuple). Must divide image_size evenly.
        in_chans: Number of input channels. Typically 1 or 3.
        embed_dim: Embedding dimension. Must be positive and divisible by num_heads.
        depth: Number of transformer blocks. Must be positive.
        num_heads: Number of attention heads. Must be positive.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim. Must be positive.
        qkv_bias: Enable bias for QKV projections.
        proj_bias: Enable bias for attention projection.
        ffn_bias: Enable bias for FFN layers.
        stochastic_depth_rate: Maximum stochastic depth rate.
        drop_path_uniform: Use uniform drop rate across blocks.
        init_values: LearnableMultiplier initialization value.
        attention_type: Type of attention mechanism.
        ffn_type: Type of feed-forward network.
        normalization_type: Type of normalization.
        num_register_tokens: Number of register tokens to use.
        interpolate_antialias: Use anti-aliasing for positional embedding interpolation.
        interpolate_offset: Offset for positional embedding interpolation.
        include_top: Whether to include the final normalization layer.
        input_shape: Input shape. If None, computed from image_size and in_chans.
        **kwargs: Additional keyword arguments for the Model base class.

    Inputs:
        A 2-element list ``[images, masks]``:
            - images: 4D tensor `(batch_size, height, width, channels)`.
            - masks: 2D boolean tensor `(batch_size, num_patches)` (iBOT mask;
              `True` marks patches replaced by the learnable mask token).

    Output:
        Always a dictionary (output structure is config-fixed, not
        training-dependent) with keys:
            - 'x_norm_clstoken': CLS token features (B, D)
            - 'x_norm_regtokens': Register token features (B, R, D)
            - 'x_norm_patchtokens': Patch token features (B, N, D)
            - 'x_prenorm': Pre-normalization features (B, 1+R+N, D)
            - 'masks': Input masks (echoed through an identity op)

    Example:
        ```python
        # Standard DINOv2-Base
        backbone = DINOv2VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            stochastic_depth_rate=0.1
        )

        # With register tokens
        backbone = DINOv2VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_register_tokens=4
        )
        ```
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        'tiny': {
            'embed_dim': 192,
            'depth': 12,
            'num_heads': 3,
            'mlp_ratio': 4.0,
        },
        'small': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 6,
            'mlp_ratio': 4.0,
        },
        'base': {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
        },
        'large': {
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0,
        },
        'giant': {
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
            'mlp_ratio': 4.0,
            'ffn_type': 'swiglu',  # Giant uses SwiGLU by default
        }
    }

    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,  # Renamed from img_size
            patch_size: Union[int, Tuple[int, int]] = 14,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            stochastic_depth_rate: float = 0.0,  # Renamed from drop_path_rate
            drop_path_uniform: bool = False,
            init_values: Optional[float] = None,
            attention_type: str = 'multi_head',
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            num_register_tokens: int = 0,
            interpolate_antialias: bool = False,
            interpolate_offset: float = 0.1,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> None:
        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if in_chans not in [1, 3]:
            logger.warning(f"Unusual number of input channels: {in_chans}. DINOv2 typically uses 3 channels.")

        # Store configuration
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.ffn_bias = ffn_bias
        self.stochastic_depth_rate = stochastic_depth_rate  # Renamed
        self.drop_path_uniform = drop_path_uniform
        self.init_values = init_values
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.include_top = include_top

        # Computed attributes
        self.num_features = embed_dim
        self.num_tokens = 1  # CLS token
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        # Set input shape
        if input_shape is None:
            input_shape = (*self.image_size, self.in_chans)

        # Validate patch size alignment
        if self.image_size[0] % self.patch_size[0] != 0 or self.image_size[1] % self.patch_size[1] != 0:
            raise ValueError(f"image_size {self.image_size} must be divisible by patch_size {self.patch_size}")

        # Initialize layer lists for tracking
        self.transformer_blocks = []
        self.patch_embed = None
        self.pos_embed = None
        self.norm = None

        # DECISION plan_2026-06-15_e2759fbc/D-009: the iBOT mask token is a GENUINE
        # learnable (1,1,embed_dim) weight via MaskTokenApply (owns its weight in build(),
        # like ClassTokenPrepend), NOT a Dense(use_bias=False, kernel_initializer='zeros')
        # applied to ones(1,N,1). The old Dense-on-ones produced a CONSTANT-ZERO vector at
        # init (degenerate: ops.where(mask, zeros, patch) merely ZEROED masked patches) —
        # the exact "projection-on-ones" hack D-002 already rejected for the CLS token. Do
        # NOT revert to a zero Dense-on-ones; the mask token must be a real add_weight so it
        # learns a non-degenerate iBOT replacement vector. MaskTokenApply is safe to assign
        # pre-super (lazy build). See decisions.md D-009.
        self.mask_token_layer = MaskTokenApply(name='mask_token')

        # DECISION plan_2026-06-15_e2759fbc/D-002: CLS token is a real learnable token via
        # ClassTokenPrepend (owns its (1,1,dim) weight in build()), NOT a Dense-on-ones
        # projection hack. ClassTokenPrepend is safe to assign pre-super (lazy build).
        # Do NOT replace with a Dense(name='cls_token_projection') on ones. See decisions.md D-002.
        self.cls_token_layer = ClassTokenPrepend(name="cls_token")

        # DECISION plan_2026-06-15_e2759fbc/D-003: register-token projection hoisted to
        # __init__ (guarded by num_register_tokens>0), same no-weight-creating-layer-in-Lambda
        # rule as D-001. Insertion uses Concatenate, not a Lambda. See decisions.md D-003.
        self.register_token_projection = None
        if self.num_register_tokens > 0:
            self.register_token_projection = layers.Dense(
                self.embed_dim,
                use_bias=False,
                kernel_initializer=initializers.TruncatedNormal(stddev=1e-6),
                name='register_token_projection'
            )

        # Create inputs and build model using functional API
        inputs = keras.Input(shape=input_shape, name="input_images")

        # DECISION plan_2026-06-15_e2759fbc/D-009: the model contract is 2-input
        # [images, masks]. The former 3rd `is_training` Input was SPURIOUS — once B5/D-005
        # made the output structure always the 5-key dict (no training-dependent branch),
        # nothing in the graph reads `is_training`, so it was dead weight carried only to
        # be coerced by a fragile DINOv2.call override (#13). Do NOT re-add an `is_training`
        # Input here or to the wrapper; there is no branch that needs it. See decisions.md D-009.
        masks_input = keras.Input(shape=(self.num_patches,), dtype=tf.bool, name="input_masks")

        # Build the model
        outputs = self._build_model(inputs, masks_input)

        # DECISION plan_2026-06-15_e2759fbc/D-007: backbone name was hardcoded in
        # super().__init__ while the wrapper (DINOv2._build_model) ALSO passed
        # name='dinov2_backbone' into **kwargs -> duplicate `name` kwarg TypeError on the
        # first-ever wrapper->backbone construction. Fixed on the BACKBONE side: honor a
        # caller-supplied name via kwargs.pop, falling back to the variant-derived default.
        # Do NOT re-add a hardcoded name= alongside **kwargs (re-introduces the collision).
        name = kwargs.pop('name', f'dinov2_vit_{embed_dim}d_{depth}l')

        # Initialize the Model
        super().__init__(
            inputs=[inputs, masks_input],
            outputs=outputs,
            name=name,
            **kwargs
        )

        logger.info(
            f"Created DINOv2 ViT backbone: {embed_dim}d x {depth}l x {num_heads}h, "
            f"patches {self.num_patches}, register_tokens {num_register_tokens}"
        )

    def _build_model(
            self,
            inputs: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build the complete DINOv2 Vision Transformer architecture.

        Args:
            inputs: Input image tensor
            masks: Input mask tensor for iBOT objective

        Returns:
            The 5-key output dict (always — output structure is config-fixed).
        """
        # Build patch embedding
        x = self._build_patch_embedding(inputs)

        # Build token preparation with masks
        x = self._build_token_preparation(x, inputs, masks)

        # Build transformer blocks
        x = self._build_transformer_blocks(x)

        # Build final processing
        outputs = self._build_final_processing(x, masks)

        return outputs

    def _build_patch_embedding(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build patch embedding layer."""
        # Create patch embedding using factory
        self.patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )

        x = self.patch_embed(inputs)
        logger.debug(f"After patch embedding: {x.shape}")
        return x

    def _build_token_preparation(
            self,
            patch_embeddings: keras.KerasTensor,
            original_inputs: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build token preparation with CLS, register tokens, and positional embeddings."""
        # B1: apply iBOT masking via MaskTokenApply, which owns the learnable (1,1,D)
        # mask-token weight in its build() and performs the elementwise select internally.
        # DECISION plan_2026-06-15_e2759fbc/D-009: the masked positions are replaced by a
        # REAL learnable mask token (not a Dense-on-ones zero vector, and not a raw
        # weight-creating layer inside a Lambda). MaskTokenApply((patch_emb, mask)) returns
        # where(expand_dims(mask,-1), mask_token, patch_emb) with mask_token broadcast over
        # the batch. Do NOT revert to a zero Dense-on-ones + an apply_masks ops.where Lambda.
        # See decisions.md D-009.
        x = self.mask_token_layer([patch_embeddings, masks])

        # B2: prepend a real learnable CLS token (B,N,D)->(B,N+1,D) via ClassTokenPrepend,
        # after masking, before pos-embed.
        # DECISION plan_2026-06-15_e2759fbc/D-002: use ClassTokenPrepend, not a Dense-on-ones
        # hack. Do NOT reintroduce a Dense(name='cls_token_projection'). See decisions.md D-002.
        x = self.cls_token_layer(x)

        # B4/B7: positional embeddings. The weight is sized num_patches + num_tokens (CLS),
        # so it accounts for the prepended CLS; PositionalEmbedding.call slices to the seq
        # length. Variable-resolution interpolation is OUT OF SCOPE.
        pos_embed_seq_len = self.num_patches + self.num_tokens
        self.pos_embed = create_embedding_layer(
            'positional_learned',
            max_seq_len=pos_embed_seq_len,
            dim=self.embed_dim,
            name='pos_embed'
        )

        # DECISION plan_2026-06-15_e2759fbc/D-004: flatten pos-embed to x = self.pos_embed(x).
        # The deleted path nested a Lambda inside a Lambda (illegal layer-creation-in-trace),
        # read the WRONG attribute self.pos_embed.pos_embed (correct is .pos_embedding), and
        # ran a Python assert on a symbolic tensor (a trace-time no-op). Do NOT reintroduce
        # _get_interpolated_pos_embed or an add_pos_embed Lambda; var-res is out of scope.
        # See decisions.md D-004.
        x = self.pos_embed(x)

        # B3: insert register tokens after CLS via Concatenate (NOT a Lambda). Cold path on
        # the smoke (num_register_tokens=0); 'large'/'giant' auto-enable 4 registers.
        # DECISION plan_2026-06-15_e2759fbc/D-003: reg_tokens via the hoisted Dense OUTSIDE any
        # Lambda; insertion via Concatenate([cls, reg, rest]). Do NOT use a Lambda or an
        # in-Lambda Dense for register tokens. See decisions.md D-003.
        #
        # DECISION plan_2026-06-15_e2759fbc/D-009: register tokens are inserted AFTER the
        # positional embedding (pos_embed above is sized num_patches + num_tokens = CLS+patches
        # and applied to the (B, N+1, D) post-CLS tensor at line ~688, BEFORE this concat). This
        # is INTENTIONAL: register tokens are DELIBERATELY position-free learnable tokens (Darcet
        # et al. 2023, "Vision Transformers Need Registers") -- they receive NO positional signal
        # by design. The resulting (B, 1+R+N, D) sequence is accepted by the length-agnostic
        # attention blocks + final norm (empirically verified: a 'tiny'+4-register model forwards
        # finite (2,10) and is input-sensitive). Do NOT "fix" this by enlarging pos_embed to
        # 1+R+N or by moving register insertion before pos_embed -- that would WRONGLY give
        # registers a positional signal. See decisions.md D-009 + tests::test_register_tokens_forward.
        if self.num_register_tokens > 0:
            reg_base = self.register_token_projection(
                keras.ops.ones((1, self.num_register_tokens, 1))
            )
            # Broadcast reg tokens (1,R,D) to the runtime batch of x via a pure-op Lambda
            # (no weights), so Concatenate sees a matching batch dim.
            reg_tokens = layers.Lambda(
                lambda a: keras.ops.broadcast_to(
                    a[1],
                    (keras.ops.shape(a[0])[0], self.num_register_tokens, self.embed_dim)
                ),
                name='broadcast_register_tokens'
            )([x, reg_base])
            cls = x[:, :1]
            rest = x[:, 1:]
            x = layers.Concatenate(axis=1, name='add_register_tokens')([cls, reg_tokens, rest])

        return x

    def _build_transformer_blocks(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build transformer blocks with stochastic depth."""
        # Calculate drop path rates
        if self.drop_path_uniform:
            dpr = [self.stochastic_depth_rate] * self.depth
        else:
            dpr = np.linspace(0, self.stochastic_depth_rate, self.depth).tolist()

        # Create transformer blocks
        for i in range(self.depth):
            block = DINOv2Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attention_type=self.attention_type,
                ffn_type=self.ffn_type,
                normalization_type=self.normalization_type,
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                stochastic_depth_rate=dpr[i],  # Updated
                init_values=self.init_values,
                name=f"block_{i}"
            )
            x = block(x)
            self.transformer_blocks.append(block)

        return x

    def _build_final_processing(
            self,
            x: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build final normalization and output processing."""
        x_prenorm = x  # Store pre-normalization features

        # Final normalization
        if self.include_top:
            self.norm = create_normalization_layer(
                normalization_type=self.normalization_type,
                name="norm"
            )
            x_norm = self.norm(x)
        else:
            x_norm = x

        # DECISION plan_2026-06-15_e2759fbc/D-005: ALWAYS return the 5-key dict; do NOT
        # branch on `is_training` via keras.ops.cond. The original code returned a 5-key
        # dict in the training branch but a bare CLS tensor in the inference branch — a
        # mismatched nested structure that keras.ops.cond rejects, and an output whose
        # STRUCTURE depended on `training` (the training-dependent-output anti-pattern).
        # This is a Functional model with a fixed `is_training` Input, so the runtime
        # branch never earned its keep. Each output key is produced by its OWN tensor op
        # (a per-key Lambda returning a single tensor) so Keras can shape-infer each one;
        # a single Lambda returning a Python dict cannot always infer its output shape
        # ("could not infer the shape of the Lambda's output"). The output structure is
        # config-fixed (always the 5-key dict). Do NOT reintroduce ops.cond, a bare-tensor
        # inference branch, an `is_training` Input, or a dict-returning Lambda.
        num_reg = self.num_register_tokens

        cls_token = layers.Lambda(
            lambda t: t[:, 0], name='slice_cls_token'
        )(x_norm)

        if num_reg > 0:
            reg_tokens = layers.Lambda(
                lambda t: t[:, 1:1 + num_reg], name='slice_reg_tokens'
            )(x_norm)
            patch_tokens = layers.Lambda(
                lambda t: t[:, 1 + num_reg:], name='slice_patch_tokens'
            )(x_norm)
        else:
            reg_tokens = layers.Lambda(
                lambda t: t[:, 0:0], name='slice_reg_tokens'
            )(x_norm)
            patch_tokens = layers.Lambda(
                lambda t: t[:, 1:], name='slice_patch_tokens'
            )(x_norm)

        # DECISION plan_2026-06-15_e2759fbc/D-008: route the `masks` input through an
        # identity Lambda before echoing it as an output. Passing the RAW `masks` input
        # KerasTensor straight into the outputs dict makes it both an input AND an output
        # of this Functional model; when the DINOv2 wrapper nests this backbone and feeds
        # its own masks input in, Keras's _build_map sees the input tensor reachable from
        # the backbone op that consumes it and raises "Tensor input_masks ... is part of a
        # cycle". Wrapping in an identity op makes the echoed mask a DISTINCT node, breaking
        # the input-is-output aliasing. Do NOT put the bare `masks` input back in the dict.
        masks_out = layers.Lambda(lambda t: t, name='masks_passthrough')(masks)

        outputs = {
            "x_norm_clstoken": cls_token,
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": x_prenorm,
            "masks": masks_out,
        }

        return outputs

    @classmethod
    def from_variant(
            cls,
            variant: Literal['tiny', 'small', 'base', 'large', 'giant'],
            image_size: Union[int, Tuple[int, int]] = 224,  # Updated
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_register_tokens: int = 0,
            init_values: Optional[float] = 1e-5,
            stochastic_depth_rate: float = 0.0,  # Updated
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> "DINOv2VisionTransformer":
        """
        Create DINOv2 Vision Transformer from predefined variant.

        Args:
            variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
            image_size: Input image size.
            patch_size: Patch size for patch embedding.
            num_register_tokens: Number of register tokens to use.
            init_values: LearnableMultiplier initialization value.
            stochastic_depth_rate: Maximum stochastic depth rate.
            input_shape: Input shape. If None, computed from image_size.
            **kwargs: Additional arguments for the model.

        Returns:
            DINOv2VisionTransformer instance.

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(cls.MODEL_VARIANTS.keys())}")

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating DINOv2-{variant.upper()} Vision Transformer")
        logger.info(f"Configuration: {config}")

        return cls(
            image_size=image_size,  # Updated
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            init_values=init_values,
            stochastic_depth_rate=stochastic_depth_rate,  # Updated
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'image_size': self.image_size,  # Updated
            'patch_size': self.patch_size,
            'in_chans': self.in_chans,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'proj_bias': self.proj_bias,
            'ffn_bias': self.ffn_bias,
            'stochastic_depth_rate': self.stochastic_depth_rate,  # Updated
            'drop_path_uniform': self.drop_path_uniform,
            'init_values': self.init_values,
            'attention_type': self.attention_type,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
            'num_register_tokens': self.num_register_tokens,
            'interpolate_antialias': self.interpolate_antialias,
            'interpolate_offset': self.interpolate_offset,
            'include_top': self.include_top,
        }

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info(f"DINOv2 ViT configuration:")
        logger.info(f"  - Input size: {self.image_size}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Embed dim: {self.embed_dim}")
        logger.info(f"  - Depth: {self.depth}")
        logger.info(f"  - Num heads: {self.num_heads}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Num patches: {self.num_patches}")
        logger.info(f"  - Register tokens: {self.num_register_tokens}")
        logger.info(f"  - Stochastic depth rate: {self.stochastic_depth_rate}")  # Updated
        logger.info(f"  - Init values: {self.init_values}")

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DINOv2(keras.Model):
    """
    Complete DINOv2 Model with classification head following modern Keras 3 patterns.

    This model provides a high-level interface for DINOv2 with automatic
    input/output handling and model variant support. Uses the functional API
    pattern for consistent architecture building.

    **Architecture**:
    ```
    Input (B, H, W, C)
         ↓
    DINOv2VisionTransformer → Features (B, D)
         ↓
    Dense Classifier (optional) → Predictions (B, num_classes)
    ```

    **Intent**: Provide a complete model interface that can be used for both
    pre-training and fine-tuning, with proper functional API implementation
    following modern Keras 3 best practices.

    Args:
        image_size: Input image size (int or tuple).
        patch_size: Patch size for patch embedding (int or tuple).
        num_classes: Number of output classes.
        include_top: Whether to include classification head.
        input_shape: Input shape. If None, computed from image_size.
        **backbone_kwargs: Additional arguments passed to backbone.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        - If `include_top=True`: 2D tensor with shape: `(batch_size, num_classes)`
        - If `include_top=False`: 2D tensor with shape: `(batch_size, embed_dim)`

    Example:
        ```python
        # Pre-training model (no classification head)
        model = DINOv2(
            image_size=224,
            patch_size=14,
            include_top=False,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Fine-tuning model with classification head
        model = DINOv2(
            image_size=224,
            patch_size=14,
            num_classes=1000,
            include_top=True,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # From variant
        model = DINOv2.from_variant('base', num_classes=100)
        ```
    """

    def __init__(
            self,
            image_size: Union[int, Tuple[int, int]] = 224,  # Updated
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_classes: int = 1000,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **backbone_kwargs
    ) -> None:
        # Validate inputs
        if num_classes <= 0 and include_top:
            raise ValueError(f"num_classes must be positive when include_top=True, got {num_classes}")

        # Store configuration
        self.image_size = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
        self.num_classes = num_classes
        self.include_top = include_top
        self.backbone_kwargs = backbone_kwargs

        # Set input shape
        if input_shape is None:
            in_chans = backbone_kwargs.get('in_chans', 3)
            input_shape = (*self.image_size, in_chans)

        # Initialize layer tracking
        self.backbone = None
        self.classifier = None

        # Create inputs
        # DECISION plan_2026-06-15_e2759fbc/D-008: the wrapper's keras.Input names MUST
        # be UNIQUE from the backbone sub-model's internal input names ("input_images",
        # "input_masks"). Sharing names makes Keras alias the wrapper's symbolic inputs with
        # the nested backbone's inputs at super().__init__, producing a graph cycle
        # ("input_masks" collision). Do NOT rename these back to match the backbone —
        # prefix with "dinov2_" so the two input layers stay distinct. See D-008.
        # DECISION plan_2026-06-15_e2759fbc/D-009: the wrapper contract is 2-input
        # [images, masks]. The former 3rd `dinov2_is_training` Input was spurious (unused
        # under the always-dict backbone) and only existed to feed the fragile DINOv2.call
        # rank-0 coercion override (#13), which is now removed. Do NOT re-add it.
        inputs = keras.Input(shape=input_shape, name="dinov2_input_images")

        # For inference, we typically don't need masks, so provide default
        masks = keras.Input(shape=(None,), dtype=tf.bool, name="dinov2_input_masks")

        # Build the model
        outputs = self._build_model(inputs, masks)

        # Initialize the Model
        super().__init__(
            inputs=[inputs, masks],
            outputs=outputs,
            name='dinov2_model'
        )

        logger.info(f"Created DINOv2 complete model with include_top={include_top}")
        if include_top:
            logger.info(f"Classification head for {num_classes} classes")

    def _build_model(
            self,
            inputs: keras.KerasTensor,
            masks: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Build the complete DINOv2 model architecture.

        Args:
            inputs: Input tensor
            masks: Mask tensor

        Returns:
            Output tensor
        """
        # Create backbone
        self.backbone = DINOv2VisionTransformer(
            image_size=self.image_size,  # Updated
            patch_size=self.patch_size,
            name='dinov2_backbone',
            **self.backbone_kwargs
        )

        # DECISION plan_2026-06-15_e2759fbc/D-006: the backbone output is ALWAYS a
        # 5-key dict (see D-005) — slice ["x_norm_clstoken"] directly by subscripting the
        # functional output. Do NOT wrap this in a Lambda + keras.ops.cond on
        # `is_training`: the old code ran a cond whose inference branch returned the WHOLE
        # dict (not the CLS tensor), which is both wrong and a mismatched-structure cond.
        # Dict subscription on a Functional model's output is legal and shape-inferable.
        backbone_output = self.backbone([inputs, masks])
        features = backbone_output["x_norm_clstoken"]

        # Create classifier if needed
        if self.include_top and self.num_classes > 0:
            self.classifier = layers.Dense(
                self.num_classes,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
                name='classifier'
            )
            outputs = self.classifier(features)
        else:
            outputs = features

        return outputs

    @classmethod
    def from_variant(
            cls,
            variant: Literal['tiny', 'small', 'base', 'large', 'giant'],
            image_size: Union[int, Tuple[int, int]] = 224,  # Updated
            patch_size: Union[int, Tuple[int, int]] = 14,
            num_classes: int = 1000,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> "DINOv2":
        """
        Create DINOv2 model from predefined variant.

        Args:
            variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
            image_size: Input image size.
            patch_size: Patch size for patch embedding.
            num_classes: Number of output classes.
            include_top: Whether to include classification head.
            input_shape: Input shape.
            **kwargs: Additional arguments for the model.

        Returns:
            DINOv2 instance.

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in DINOv2VisionTransformer.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(DINOv2VisionTransformer.MODEL_VARIANTS.keys())}")

        config = DINOv2VisionTransformer.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating DINOv2-{variant.upper()} complete model")

        return cls(
            image_size=image_size,  # Updated
            patch_size=patch_size,
            num_classes=num_classes,
            include_top=include_top,
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'image_size': self.image_size,  # Updated
            'patch_size': self.patch_size,
            'num_classes': self.num_classes,
            'include_top': self.include_top,
            **self.backbone_kwargs
        }

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        logger.info(f"DINOv2 Model configuration:")
        logger.info(f"  - Input size: {self.image_size}")
        logger.info(f"  - Patch size: {self.patch_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Backbone config: {self.backbone_kwargs}")

# ---------------------------------------------------------------------

def create_dino_v2(  # Renamed from create_dinov2_model
        variant: Literal['tiny', 'small', 'base', 'large', 'giant'] = 'base',
        image_size: Union[int, Tuple[int, int]] = 224,  # Updated
        patch_size: Union[int, Tuple[int, int]] = 14,
        num_register_tokens: int = 0,
        init_values: Optional[float] = 1e-5,
        stochastic_depth_rate: float = 0.0,  # Updated
        ffn_type: str = 'mlp',
        num_classes: int = 1000,
        include_top: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: bool = False,
        **kwargs
) -> DINOv2:
    """
    Factory function to create DINOv2 model variants with sensible defaults.

    **Recommended Configurations**:
    - **tiny/small/base/large**: Use 'mlp' FFN, 0 or 4 register tokens
    - **giant**: Use 'swiglu' FFN, 4 register tokens, higher stochastic_depth_rate

    Args:
        variant: Size variant ('tiny', 'small', 'base', 'large', 'giant').
        image_size: Input image size.
        patch_size: Patch size for patch embedding.
        num_register_tokens: Number of register tokens (0 or 4 typically).
        init_values: LearnableMultiplier initialization value.
        stochastic_depth_rate: Maximum stochastic depth rate.
        ffn_type: Type of FFN ('mlp' for small models, 'swiglu' for giant).
        num_classes: Number of output classes.
        include_top: Whether to include classification head.
        input_shape: Input shape.
        pretrained: Whether to load pretrained weights (not implemented).
        **kwargs: Additional arguments for the model.

    Returns:
        DINOv2 instance.

    Example:
        ```python
        # Standard DINOv2-Base for ImageNet
        model = create_dino_v2('base', num_classes=1000)

        # DINOv2-Giant with SwiGLU and register tokens
        model = create_dino_v2(
            'giant',
            ffn_type='swiglu',
            num_register_tokens=4,
            stochastic_depth_rate=0.3
        )

        # Pre-training model (no classification head)
        model = create_dino_v2('base', include_top=False)

        # CIFAR-10 model
        model = create_dino_v2(
            'small',
            num_classes=10,
            image_size=32,
            input_shape=(32, 32, 3)
        )
        ```
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    # Use SwiGLU for giant model by default if not specified
    if variant == 'giant' and ffn_type == 'mlp':
        ffn_type = 'swiglu'
        logger.info("Using SwiGLU FFN for giant variant")

    # Set reasonable register tokens for larger models
    if variant in ['large', 'giant'] and num_register_tokens == 0:
        num_register_tokens = 4
        logger.info(f"Setting {num_register_tokens} register tokens for {variant} variant")

    logger.info(f"Creating DINOv2-{variant.upper()} model with:")
    logger.info(f"  - FFN type: {ffn_type}")
    logger.info(f"  - Register tokens: {num_register_tokens}")
    logger.info(f"  - Stochastic depth rate: {stochastic_depth_rate}")  # Updated
    logger.info(f"  - Init values: {init_values}")

    return DINOv2.from_variant(
        variant,
        image_size=image_size,  # Updated
        patch_size=patch_size,
        num_classes=num_classes,
        include_top=include_top,
        input_shape=input_shape,
        num_register_tokens=num_register_tokens,
        init_values=init_values,
        stochastic_depth_rate=stochastic_depth_rate,  # Updated
        ffn_type=ffn_type,
        **kwargs
    )