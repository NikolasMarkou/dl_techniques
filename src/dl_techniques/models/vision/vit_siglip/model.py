"""Vision Transformer with a two-stage convolutional patch-embedding stem.

The "SigLIP" name is inherited and does not describe what this file builds.
SigLIP's contribution is a sigmoid pairwise loss that replaces CLIP's softmax
over the in-batch similarity matrix; its own vision tower uses a single-conv
stem, no CLS token and a MAP pooling head. This file has none of those
three: a two-stage stem, an optional CLS token, and no MAP head. It contains
no text tower and no loss.

What it builds: a standard pre-norm ViT encoder whose patch embedding is
split into two strided convolutions, `Conv2D(embed_dim//2, k=s=patch//2)` ->
LayerNorm -> GELU -> `Conv2D(embed_dim, k=s=2)`, for a total stride of
`2 * (patch // 2)`. That forces `patch_size` to be even in both dimensions;
an odd value gives a stride smaller than the declared patch size, so the
constructor raises instead of letting the mismatch surface as a reshape
failure later. Attention, normalization and FFN come from the
`dl_techniques` factories.

The class name and `SCALE_CONFIGS` keys keep `SigLIP` for source
compatibility. A published SigLIP number does not apply to a model built
here.

References:
    - Zhai et al., 2023. Sigmoid Loss for Language Image Pre-Training (SigLIP).
      (https://arxiv.org/abs/2303.15343)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words (ViT).
      (https://arxiv.org/abs/2010.11929)
    - Lee et al., 2019. Set Transformer. (https://arxiv.org/abs/1810.00825)
"""

import keras
from keras import ops, layers, activations, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

SigLIPScale = Literal['tiny', 'small', 'base', 'large', 'huge']
PoolingMode = Literal['cls', 'mean', 'max']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.vit_siglip.model")
class SigLIPVisionTransformer(keras.Model):
    """Vision Transformer with a two-stage convolutional patch-embedding stem.

    The `SigLIP` in the name is historical; SigLIP itself is a sigmoid
    contrastive loss, not a patch-embedding scheme, and its own tower uses a
    single-conv stem, no CLS token and a MAP head, none of which this class
    shares (see the module docstring). This class builds a standard pre-norm
    ViT encoder whose patch embedding is split into two strided
    convolutions, with every sub-component created through the
    dl_techniques factories.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, H, W, C]                  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Conv2D(embed_dim/2, k=s=patch/2)    │
        │  → LayerNorm → GELU                  │
        │  → Conv2D(embed_dim, k=s=2)          │
        │  patch_size must be even             │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  reshape to [B, N, D]                │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  prepend CLS token → [B, N+1, D]     │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  + learned positional embedding      │
        │  → Dropout(pos_dropout_rate)         │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  TransformerLayer × num_layers       │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  final norm ('pre' only)             │
        └───────────────┬──────────────────────┘
                        │
            ┌───────────┴────────────┐
            ▼                        ▼
        include_top=True        include_top=False
            │                        │
        ┌───────────────┐   ┌────────────────────────────┐
        │ x[:, 0]       │   │ pooling='cls'  → x[:, 0]   │
        │ [Dropout]     │   │ pooling='mean' → mean over │
        │ Dense(classes)│   │ pooling='max'  → max  over │
        │               │   │   the whole sequence       │
        │ → [B, classes]│   │ pooling=None → [B,N+1,D]   │
        └───────────────┘   └────────────────────────────┘

    Scales:

    .. code-block:: text

        scale     embed_dim   heads   layers   mlp_ratio
        tiny         192        3       12        4.0
        small        384        6       12        4.0
        base         768       12       12        4.0
        large       1024       16       24        4.0
        huge        1280       16       32        4.0

    :param input_shape: Input image shape ``(height, width, channels)``. Must
        have positive dimensions divisible by ``patch_size``. Defaults to
        ``(224, 224, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param num_classes: Number of output classes. Must be positive. Only used
        when ``include_top=True``. Defaults to 1000.
    :type num_classes: int
    :param scale: Model scale, one of ``'tiny'``, ``'small'``, ``'base'``,
        ``'large'``, ``'huge'``. Defaults to ``'base'``.
    :type scale: SigLIPScale
    :param patch_size: Patch size; an int gives square patches. Must be even
        in both dimensions and divide the image dimensions. Defaults to 16.
    :type patch_size: Union[int, Tuple[int, int]]
    :param include_top: Whether to include the classification head. When
        False the model is a feature extractor. Defaults to True.
    :type include_top: bool
    :param pooling: Pooling strategy, used only when ``include_top=False``:
        ``'cls'`` reads the CLS token, ``'mean'``/``'max'`` pool the whole
        sequence including the CLS token, ``None`` returns the full
        sequence. Defaults to None.
    :type pooling: Optional[PoolingMode]
    :param dropout_rate: General dropout rate, applied in the transformer
        layers and before the classification head. Defaults to 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout rate for attention weights.
        Defaults to 0.0.
    :type attention_dropout_rate: float
    :param pos_dropout_rate: Dropout rate after the positional embedding.
        Defaults to 0.0.
    :type pos_dropout_rate: float
    :param kernel_initializer: Weight initializer for every layer. Defaults
        to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Weight regularizer for every layer. Defaults
        to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: Bias initializer for every layer. Defaults to
        ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Bias regularizer for every layer. Defaults to
        None.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param normalization_type: Normalization identifier passed to
        ``create_normalization_layer``. Defaults to ``'layer_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'post'`` (default) or ``'pre'``. The
        final normalization layer is created only when ``'pre'``.
    :type normalization_position: Literal['pre', 'post']
    :param ffn_type: Feed-forward network identifier passed to the factory.
        Defaults to ``'mlp'``.
    :type ffn_type: FFNType
    :param activation: Activation for the FFN. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param name: Model name; auto-generated as
        ``siglip_vision_transformer_<scale>`` when None.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If ``input_shape`` is not a positive 3-tuple, if
        ``patch_size`` is invalid, odd, or does not divide the image
        dimensions, if ``num_classes`` is not positive, if ``scale`` or
        ``pooling`` is unrecognized, or if any dropout rate leaves
        ``[0, 1]``.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``, with height and
        width divisible by the corresponding patch dimensions.

    Output shape:
        - ``include_top=True``: ``(batch_size, num_classes)``, logits with no
          softmax applied.
        - ``include_top=False, pooling='cls'|'mean'|'max'``:
          ``(batch_size, embed_dim)``.
        - ``include_top=False, pooling=None``:
          ``(batch_size, num_patches + 1, embed_dim)``.

    :ivar embed_dim: Embedding dimension, fixed by ``scale``.
    :vartype embed_dim: int
    :ivar num_heads: Attention head count, fixed by ``scale``.
    :vartype num_heads: int
    :ivar num_layers: Transformer depth, fixed by ``scale``.
    :vartype num_layers: int
    :ivar num_patches: Total number of image patches.
    :vartype num_patches: int
    :ivar max_seq_len: ``num_patches + 1``, counting the CLS token.
    :vartype max_seq_len: int
    :ivar siglip_patch_embed: The two-stage patch-embedding layers. The
        attribute name keeps the ``siglip_`` prefix for source
        compatibility; the scheme is not SigLIP's own (see the module
        docstring).
    :vartype siglip_patch_embed: keras.Sequential
    :ivar transformer_layers: The encoder stack.
    :vartype transformer_layers: list[TransformerLayer]

    Example:
        .. code-block:: python

            model = SigLIPVisionTransformer(
                input_shape=(224, 224, 3),
                num_classes=1000,
                scale='base'
            )

            feature_model = SigLIPVisionTransformer(
                input_shape=(224, 224, 3),
                scale='base',
                include_top=False,
                pooling='cls'
            )

    Note:
        The head emits logits, so compile with ``from_logits=True``.
    """

    # Scale configurations: [embed_dim, num_heads, num_layers, mlp_ratio]
    SCALE_CONFIGS: Dict[str, Tuple[int, int, int, float]] = {
        "tiny": (192, 3, 12, 4.0),    # SigLIP ViT-Tiny
        "small": (384, 6, 12, 4.0),   # SigLIP ViT-Small
        "base": (768, 12, 12, 4.0),   # SigLIP ViT-Base
        "large": (1024, 16, 24, 4.0), # SigLIP ViT-Large
        "huge": (1280, 16, 32, 4.0),  # SigLIP ViT-Huge
    }

    # `MODEL_VARIANTS` is the canonical name across `models/` (see
    # `models/CLAUDE.md` § House Model Module Shape). `SCALE_CONFIGS` remains the
    # definition because the `scale=` constructor argument and the tests already
    # name it; this is an alias to the same dict, not a copy.
    MODEL_VARIANTS = SCALE_CONFIGS

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            num_classes: int = 1000,
            scale: SigLIPScale = "base",
            patch_size: Union[int, Tuple[int, int]] = 16,
            include_top: bool = True,
            pooling: Optional[PoolingMode] = None,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            pos_dropout_rate: float = 0.0,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            normalization_type: NormalizationType = "layer_norm",
            normalization_position: Literal['pre', 'post'] = "post",
            ffn_type: FFNType = "mlp",
            activation: Union[str, callable] = "gelu",
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize SigLIP Vision Transformer model."""
        # Auto-generate name if not provided
        if name is None:
            name = f"siglip_vision_transformer_{scale}"

        super().__init__(name=name, **kwargs)

        # Validate and store input_shape
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
            raise ValueError(f"input_shape must be a 3-tuple (height, width, channels), got {input_shape}")

        img_h, img_w, img_c = input_shape
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")

        # Validate and normalize patch_size
        if isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {patch_size}")
            patch_h = patch_w = patch_size
        else:
            if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
                raise ValueError(f"patch_size must be int or tuple of 2 ints, got {patch_size}")
            patch_h, patch_w = patch_size
            if patch_h <= 0 or patch_w <= 0:
                raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")

        # The two-stage stem has total stride 2*(patch//2), which equals patch
        # only when patch is even; an odd patch size would surface as an opaque reshape error later.
        if patch_h % 2 != 0 or patch_w % 2 != 0:
            raise ValueError(
                f"patch_size must be even in both dimensions, got {patch_size}: "
                f"the two-stage patch-embedding stem has total stride "
                f"2*(patch//2), which equals the patch size only for even "
                f"values (an odd {patch_h} would give a stride of "
                f"{2 * (patch_h // 2)})."
            )

        # Validate divisibility for patch extraction
        if img_h % patch_h != 0:
            raise ValueError(f"Image height ({img_h}) must be divisible by patch height ({patch_h})")
        if img_w % patch_w != 0:
            raise ValueError(f"Image width ({img_w}) must be divisible by patch width ({patch_w})")

        # Validate other parameters
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        if scale not in self.SCALE_CONFIGS:
            raise ValueError(f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}")

        if pooling not in [None, "cls", "mean", "max"]:
            raise ValueError(f"Unsupported pooling: {pooling}. Choose from [None, 'cls', 'mean', 'max']")

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(f"pos_dropout_rate must be between 0 and 1, got {pos_dropout_rate}")

        # Store ALL configuration parameters for serialization
        self.input_shape_config = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.scale = str(scale)
        self.patch_size = (patch_h, patch_w)
        self.include_top = bool(include_top)
        self.pooling = pooling
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.pos_dropout_rate = float(pos_dropout_rate)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = bias_regularizer
        self.normalization_type = str(normalization_type)
        self.normalization_position = str(normalization_position)
        self.ffn_type = str(ffn_type)
        self.activation = activation

        # Get model configuration from scale
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate derived parameters
        self.intermediate_size = int(self.embed_dim * self.mlp_ratio)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate derived parameters
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        # Create all sub-layers in __init__; they are unbuilt until build() runs.
        # Uses factories for consistent component creation.

        # Two-stage patch embedding -- this module's own, not SigLIP's (module docstring)
        self.siglip_patch_embed = self._create_siglip_patch_embedding()

        # Positional embedding using factory
        self.pos_embed = create_embedding_layer(
            'positional_learned',
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout_rate=self.pos_dropout_rate,
            name="pos_embed"
        )

        # Transformer layers using existing TransformerLayer
        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type="multi_head",
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                activation=self.activation,
                use_bias=True,
                # DECISION plan-2026-08-23T091307-9a110062/D-560: each block gets its own
                # clone_initializer() copy; a shared instance replays one draw across all blocks. See decisions.md.
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Final normalization using factory - only for pre-norm
        self.norm = None
        if self.normalization_position == 'pre':
            self.norm = create_normalization_layer(
                self.normalization_type,
                name="final_norm"
            )

        # Classification components (if include_top)
        self.head_dropout = None
        self.head = None
        if self.include_top:
            if self.dropout_rate > 0.0:
                self.head_dropout = layers.Dropout(self.dropout_rate, name="head_dropout")

            self.head = layers.Dense(
                self.num_classes,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="head"
            )

        # Feature-extraction pooling via the shared SequencePooling layer.
        # DECISION plan-2026-07-15T144225-5b25d9f1/D-001: pool via SequencePooling with the CLS
        # token included in mean/max (no exclude_positions); adding it would drop CLS from the output. See decisions.md.
        self.pool = None
        if self.pooling == "cls":
            self.pool = SequencePooling(strategy="cls", name="seq_pool")
        elif self.pooling in ("mean", "max"):
            self.pool = SequencePooling(strategy=self.pooling, name="seq_pool")

        # CLS token weight (created in build())
        self.cls_token = None

        logger.info(f"Created SigLIPVisionTransformer-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")
        logger.info(f"Image shape: {self.input_shape_config}, Patch size: {self.patch_size}, Num patches: {self.num_patches}")

    def _create_siglip_patch_embedding(self) -> keras.Sequential:
        """Build the two-stage patch-embedding stem (see the module docstring).

        :return: Sequential model implementing the two-stage patch embedding.
        :rtype: keras.Sequential
        """
        patch_h, patch_w = self.patch_size

        return keras.Sequential([
            # Stage 1: Coarse-grained patching
            layers.Conv2D(
                filters=self.embed_dim // 2,
                kernel_size=(patch_h // 2, patch_w // 2),
                strides=(patch_h // 2, patch_w // 2),
                padding='valid',
                use_bias=True,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='patch_embed_conv1'
            ),
            # DECISION plan-2026-08-17T183311-79c63e38/D-028: route through the factory,
            # not layers.LayerNormalization directly; that default eps=1e-3 is a 100x divergence
            # from the model's final norm (1e-6). See decisions.md.
            create_normalization_layer('layer_norm', name='patch_embed_norm1'),
            layers.Activation('gelu', name='patch_embed_activation1'),

            # Stage 2: Refinement to final embedding dimension
            layers.Conv2D(
                filters=self.embed_dim,
                kernel_size=2,
                strides=2,
                padding='valid',
                use_bias=True,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='patch_embed_conv2'
            ),
        ], name='siglip_patch_embed')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the CLS token and explicitly build every sub-layer.

        Each sub-layer is built in computational order rather than left to a
        lazy first call, so the weight tree materializes on ``.keras``
        reload.

        :param input_shape: Input shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not 4D.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        # Create CLS token weight
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )

        # Build all sub-layers in computational order
        # SigLIP patch embedding
        dummy_input_shape = (None,) + self.input_shape_config
        self.siglip_patch_embed.build(dummy_input_shape)

        # Positional embedding
        pos_input_shape = (None, self.max_seq_len, self.embed_dim)
        self.pos_embed.build(pos_input_shape)

        # Transformer layers
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)

        # Final normalization (only for pre-norm)
        if self.norm is not None:
            self.norm.build(pos_input_shape)

        # Classification head components
        if self.include_top:
            head_input_shape = (None, self.embed_dim)
            if self.head_dropout is not None:
                self.head_dropout.build(head_input_shape)
            self.head.build(head_input_shape)

        # Feature-extraction pooling
        if self.pool is not None:
            self.pool.build(pos_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the SigLIP Vision Transformer.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Model output. ``(batch, num_classes)`` logits with
            ``include_top=True``; otherwise the pooled features
            ``(batch, embed_dim)`` or, when ``pooling is None``, the full
            sequence ``(batch, max_seq_len, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(inputs)[0]

        # Shape: (batch_size, patch_h, patch_w, embed_dim).
        x = self.siglip_patch_embed(inputs, training=training)

        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        num_patches_h = img_h // patch_h
        num_patches_w = img_w // patch_w

        x = ops.reshape(x, [batch_size, num_patches_h * num_patches_w, self.embed_dim])

        # Prepend the CLS token: (batch_size, seq_len, embed_dim).
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)

        # Positional embedding includes its own dropout.
        x = self.pos_embed(x, training=training)

        for layer in self.transformer_layers:
            x = layer(x, training=training)

        if self.norm is not None:
            x = self.norm(x, training=training)

        if self.include_top:
            cls_token = x[:, 0, :]
            if self.head_dropout is not None:
                cls_token = self.head_dropout(cls_token, training=training)
            x = self.head(cls_token)
            return x
        else:
            # cls / mean / max all route through SequencePooling, built without
            # exclude_positions, so the CLS token is included in mean/max.
            if self.pool is not None:
                return self.pool(x)
            return x

    def get_cls_token(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Return the CLS token from forward-pass features.

        :param features: Vision features from the forward pass.
        :type features: keras.KerasTensor
        :return: CLS token of shape ``(batch_size, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        return features[:, 0, :]

    def get_patch_tokens(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Return the patch tokens from forward-pass features, dropping the CLS token.

        :param features: Vision features from the forward pass.
        :type features: keras.KerasTensor
        :return: Patch tokens of shape ``(batch_size, num_patches, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        return features[:, 1:, :]

    def get_spatial_features(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape patch tokens back to a spatial grid, for dense prediction tasks.

        :param features: Vision features from the forward pass.
        :type features: keras.KerasTensor
        :return: Spatial features of shape
            ``(batch_size, patch_height, patch_width, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        patch_tokens = self.get_patch_tokens(features)
        batch_size = ops.shape(patch_tokens)[0]

        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        num_patches_h = img_h // patch_h
        num_patches_w = img_w // patch_w

        return ops.reshape(
            patch_tokens,
            [batch_size, num_patches_h, num_patches_w, self.embed_dim]
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape for a given input shape.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape, depending on ``include_top`` and ``pooling``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not 4D.
        """
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        batch_size = input_shape[0]

        if self.include_top:
            return (batch_size, self.num_classes)
        else:
            if self.pooling in ["cls", "mean", "max"]:
                return (batch_size, self.embed_dim)
            else:
                return (batch_size, self.max_seq_len, self.embed_dim)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            **kwargs: Any
    ) -> "SigLIPVisionTransformer":
        """Create a SigLIPVisionTransformer model from a predefined variant.

        :param variant: One of ``'tiny'``, ``'small'``, ``'base'``,
            ``'large'``, ``'huge'``.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input image shape ``(height, width, channels)``.
        :type input_shape: Tuple[int, int, int]
        :param kwargs: Additional keyword arguments for the constructor.
        :return: A new ``SigLIPVisionTransformer`` instance.
        :rtype: SigLIPVisionTransformer
        :raises ValueError: If ``variant`` is not recognized.

        Example:
            .. code-block:: python

                model = SigLIPVisionTransformer.from_variant("base", num_classes=1000)
                model = SigLIPVisionTransformer.from_variant(
                    "small", num_classes=10, input_shape=(32, 32, 3), patch_size=4
                )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        return cls(
            input_shape=input_shape,
            num_classes=num_classes,
            scale=variant,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        Includes every ``__init__`` parameter, which is what makes the round
        trip lossless.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes,
            "scale": self.scale,
            "patch_size": self.patch_size,
            "include_top": self.include_top,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "pos_dropout_rate": self.pos_dropout_rate,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            # DECISION plan-2026-08-23T091307-9a110062/D-400: use the shared
            # activation_serialization pair, not an inline isinstance(str) check;
            # keras.activations.serialize rejects a bare dl_techniques key like 'mish'. See decisions.md.
            "activation": serialize_activation(self.activation),
        })
        return config

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
            custom_objects: Optional[Dict[str, Any]] = None
    ) -> "SigLIPVisionTransformer":
        """Recreate a model from its serialized configuration.

        The only key needing explicit handling is ``activation``:
        ``get_config`` writes a serialized form for callables, and it has to
        turn back into a callable before ``__init__`` hands it to
        ``TransformerLayer``. Every other key is already resolved by
        ``initializers.get`` / ``regularizers.get`` inside ``__init__``.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :param custom_objects: Optional mapping of names to custom callables,
            used to resolve an activation that is not registered with
            ``keras.saving.register_keras_serializable``.
        :type custom_objects: Optional[Dict[str, Any]]
        :return: A new ``SigLIPVisionTransformer`` instance.
        :rtype: SigLIPVisionTransformer
        """
        config = dict(config)
        activation = config.get("activation")
        if activation is not None and not isinstance(activation, str):
            config["activation"] = deserialize_activation(
                activation, custom_objects=custom_objects
            )
        return cls(**config)

    def get_feature_extractor(self) -> "SigLIPVisionTransformer":
        """Return a feature-extractor twin of this model.

        Copies every configuration value, sets ``include_top=False`` and
        ``pooling='cls'``. Note this constructs a new, randomly initialized
        model; it does not transfer this instance's weights.

        :return: New ``SigLIPVisionTransformer`` instance configured for
            CLS-token feature extraction.
        :rtype: SigLIPVisionTransformer
        :raises ValueError: If the model was not properly initialized.
        """
        if not hasattr(self, 'input_shape_config') or not self.input_shape_config:
            raise ValueError("Model must be properly initialized before creating feature extractor")

        return SigLIPVisionTransformer(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            pos_dropout_rate=self.pos_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            normalization_type=self.normalization_type,
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            activation=self.activation,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Print detailed model summary with architecture information."""
        logger.info("SigLIP Vision Transformer Model Summary")
        logger.info(f"Scale: {self.scale}")
        logger.info(f"Input Shape: {self.input_shape_config}")
        logger.info(f"Patch Size: {self.patch_size}")
        logger.info(f"Number of Patches: {self.num_patches}")
        logger.info(f"Sequence Length: {self.max_seq_len}")
        logger.info(f"Embedding Dimension: {self.embed_dim}")
        logger.info(f"Number of Heads: {self.num_heads}")
        logger.info(f"Number of Layers: {self.num_layers}")
        logger.info(f"MLP Ratio: {self.mlp_ratio}")
        logger.info(f"Intermediate Size: {self.intermediate_size}")
        logger.info(f"Dropout Rate: {self.dropout_rate}")
        logger.info(f"Attention Dropout Rate: {self.attention_dropout_rate}")
        logger.info(f"Positional Dropout Rate: {self.pos_dropout_rate}")
        logger.info(f"Normalization Type: {self.normalization_type}")
        logger.info(f"Normalization Position: {self.normalization_position}")
        logger.info(f"FFN Type: {self.ffn_type}")
        logger.info(f"Activation: {self.activation}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self.built:
            logger.info(f"Total Parameters: {self.count_params():,}")

        # Additional architecture information
        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        logger.info(f"Patches per dimension: {img_h // patch_h} x {img_w // patch_w}")
        logger.info("Two-stage SigLIP patch embedding architecture")


# ---------------------------------------------------------------------
# Factory Functions for Convenient Model Creation
# ---------------------------------------------------------------------


def create_siglip_vision_transformer(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 1000,
        scale: SigLIPScale = "base",
        patch_size: Union[int, Tuple[int, int]] = 16,
        include_top: bool = True,
        pooling: Optional[PoolingMode] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        pos_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: Literal['pre', 'post'] = "post",
        ffn_type: FFNType = "mlp",
        activation: Union[str, callable] = "gelu",
        **kwargs: Any
) -> SigLIPVisionTransformer:
    """Create a SigLIP-style Vision Transformer with a two-stage patch stem.

    Delegates argument validation and construction to
    :class:`SigLIPVisionTransformer`.

    :param input_shape: Input image shape ``(height, width, channels)``; must
        be compatible with ``patch_size``.
    :type input_shape: Tuple[int, int, int]
    :param num_classes: Number of output classes. Only used when
        ``include_top=True``.
    :type num_classes: int
    :param scale: Model scale, one of ``'tiny'``, ``'small'``, ``'base'``,
        ``'large'``, ``'huge'``.
    :type scale: SigLIPScale
    :param patch_size: Patch size; an int gives square patches. Must be even
        in both dimensions.
    :type patch_size: Union[int, Tuple[int, int]]
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param pooling: Feature-extraction pooling when ``include_top=False``:
        ``'cls'``, ``'mean'``, ``'max'`` or None.
    :type pooling: Optional[PoolingMode]
    :param dropout_rate: General dropout rate.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-weight dropout rate.
    :type attention_dropout_rate: float
    :param pos_dropout_rate: Post-positional-embedding dropout rate.
    :type pos_dropout_rate: float
    :param kernel_initializer: Weight initializer for every layer.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Weight regularizer for every layer.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_initializer: Bias initializer for every layer.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Bias regularizer for every layer.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param normalization_type: Normalization identifier.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'post'`` (default) or ``'pre'``.
    :type normalization_position: Literal['pre', 'post']
    :param ffn_type: Feed-forward network identifier.
    :type ffn_type: FFNType
    :param activation: FFN activation.
    :type activation: Union[str, callable]
    :param kwargs: Additional keyword arguments forwarded to the
        :class:`SigLIPVisionTransformer` constructor.
    :return: ``SigLIPVisionTransformer`` model instance.
    :rtype: SigLIPVisionTransformer
    :raises ValueError: Propagated from :class:`SigLIPVisionTransformer`'s own
        validation.

    Example:
        .. code-block:: python

            model = create_siglip_vision_transformer(
                input_shape=(224, 224, 3),
                num_classes=1000,
                scale='base'
            )

            feature_model = create_siglip_vision_transformer(
                input_shape=(384, 384, 3),
                scale='small',
                include_top=False,
                pooling='cls',
                normalization_type='rms_norm',
                ffn_type='swiglu'
            )
    """
    # Argument validation lives in SigLIPVisionTransformer.__init__, not here;
    # a second, hand-kept copy is how the even-patch_size constraint (C-15)
    # could be added to the constructor while an older duplicate pre-empted it.
    model = SigLIPVisionTransformer(
        input_shape=input_shape,
        num_classes=num_classes,
        scale=scale,
        patch_size=patch_size,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        pos_dropout_rate=pos_dropout_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        activation=activation,
        **kwargs
    )

    # 10000 patches is a practical upper bound before memory becomes a concern.
    if model.num_patches > 10000:
        logger.warning(
            f"Large number of patches ({model.num_patches}) may cause memory issues"
        )

    logger.info(f"SigLIPVisionTransformer-{scale} created successfully")
    logger.info(
        f"Configuration: {model.num_patches} patches, {num_classes} classes"
    )
    return model


# ---------------------------------------------------------------------