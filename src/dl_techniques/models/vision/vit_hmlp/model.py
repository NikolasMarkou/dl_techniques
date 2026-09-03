"""Vision Transformer with a hierarchical MLP (hMLP) patch stem.

A standard ViT embeds each patch with one linear projection. A convolutional
stem improves accuracy but breaks masked self-supervised pretraining, because
its receptive field crosses patch boundaries and lets a masked patch's
neighbors leak into its embedding. The hMLP stem keeps the accuracy gain
without the leak: each patch runs through a hierarchy of linear projections,
normalization and non-linearity, independently of every other patch. No
operation crosses a patch boundary, so masking before or after the stem gives
identical results, at under 1% extra FLOPs.

`stem_norm_layer` chooses BatchNorm (better accuracy) or LayerNorm (more
stable at small batch sizes) inside the stem. Everything after the stem is a
standard pre-norm transformer encoder built from the repo's TransformerLayer,
with attention, FFN and normalization type selectable through the usual
factories.

References:
    - Touvron et al., 2022. Three things everyone should know about Vision
      Transformers. (https://arxiv.org/abs/2203.09795)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Bao et al., 2021. BEiT: BERT Pre-Training of Image Transformers.
      (https://arxiv.org/abs/2106.08254)
    - He et al., 2021. Masked Autoencoders Are Scalable Vision Learners.
      (https://arxiv.org/abs/2111.06377)
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.hierarchical_mlp_stem import HierarchicalMLPStem
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

VitScale = Literal['tiny', 'small', 'base', 'large', 'huge']
PoolingMode = Literal['cls', 'mean', 'max']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
StemNormLayer = Literal['batch', 'layer']


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.vit_hmlp.model")
class ViTHMLP(keras.Model):
    """Vision Transformer with a hierarchical MLP stem.

    Processes each patch independently through a hierarchy of linear
    projections, then runs a standard pre-norm transformer encoder. The stem
    never mixes information across patch boundaries, so it is safe to mask
    before or after it runs.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, H, W, C]                  │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  HierarchicalMLPStem                 │
        │  4x4 stride-4, then 2x2 stride-2     │
        │  stages → [B, N, D]                  │
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
        │  pre-norm by default                 │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  final norm over the whole sequence  │
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

    hMLP stem stages:

    .. code-block:: text

        input patch grid
              ▼
        4x4 stride-4 conv
              ▼
        2x2 stride-2 stage           three stages total for the
              ▼                       default 16-pixel patch:
        2x2 stride-2 stage            4x4 -> 8x8 -> 16x16
              ▼
        patch embedding [B, N, D]

        each stage stays inside its own patch: no operation reaches
        across a patch boundary, so masking before or after the stem
        gives identical results.

    Scales:

    .. code-block:: text

        scale     embed_dim   heads   layers   mlp_ratio
        tiny         192        3       12        4.0
        small        384        6       12        4.0
        base         768       12       12        4.0
        large       1024       16       24        4.0
        huge        1280       16       32        4.0

    CLS handling in mean and max pooling:

    .. code-block:: text

        pooling='cls' reads x[:, 0] alone

        pooling='mean'/'max' pool the whole sequence, CLS token
        included (no exclude_positions; this differs from the
        sibling vit package, which excludes the CLS token)

    :param input_shape: Input image shape ``(height, width, channels)``. Must
        have positive dimensions divisible by ``patch_size``. Defaults to
        ``(224, 224, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param num_classes: Number of output classes. Must be positive. Only used
        when ``include_top=True``. Defaults to 1000.
    :type num_classes: int
    :param scale: Model scale, one of ``'tiny'``, ``'small'``, ``'base'``,
        ``'large'``, ``'huge'``. Defaults to ``'base'``.
    :type scale: VitScale
    :param patch_size: Patch size; an int gives square patches. Image
        dimensions must be divisible by it. Defaults to 16.
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
    :param stem_norm_layer: Normalization inside the hMLP stem: ``'batch'``
        (default, better accuracy) or ``'layer'`` (more stable at small
        batch sizes).
    :type stem_norm_layer: StemNormLayer
    :param kernel_initializer: Weight initializer for every layer. Defaults
        to ``'he_normal'``.
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
    :param normalization_position: ``'pre'`` (default) or ``'post'``.
    :type normalization_position: Literal['pre', 'post']
    :param ffn_type: Feed-forward network identifier passed to the factory.
        Defaults to ``'mlp'``.
    :type ffn_type: FFNType
    :param activation: Activation for the FFN. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param use_stochastic_depth: Whether transformer layers apply stochastic
        depth. Defaults to False.
    :type use_stochastic_depth: bool
    :param stochastic_depth_rate: Maximum drop-path rate, used only when
        ``use_stochastic_depth=True``. Defaults to 0.1.
    :type stochastic_depth_rate: float
    :param name: Model name; auto-generated as
        ``vision_transformer_hmlp_<scale>`` when None.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If ``input_shape`` is not a positive 3-tuple, if
        ``patch_size`` is invalid or does not divide the image dimensions, if
        ``num_classes`` is not positive, if ``scale``, ``pooling`` or
        ``stem_norm_layer`` is unrecognized, or if any rate leaves ``[0, 1]``.

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
    :ivar stem: The hierarchical MLP stem.
    :vartype stem: HierarchicalMLPStem
    :ivar transformer_layers: The encoder stack.
    :vartype transformer_layers: list[TransformerLayer]

    Example:
        .. code-block:: python

            model = ViTHMLP(
                input_shape=(224, 224, 3),
                num_classes=1000,
                scale='base'
            )

            feature_model = ViTHMLP(
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
        "tiny": (192, 3, 12, 4.0),  # ViT-Tiny
        "small": (384, 6, 12, 4.0),  # ViT-Small
        "base": (768, 12, 12, 4.0),  # ViT-Base
        "large": (1024, 16, 24, 4.0),  # ViT-Large
        "huge": (1280, 16, 32, 4.0),  # ViT-Huge
    }

    # `MODEL_VARIANTS` is the canonical name across `models/` (see
    # `models/CLAUDE.md` § House Model Module Shape). `SCALE_CONFIGS` is kept as
    # the definition because tests and the `scale=` constructor argument already
    # name it; this is an alias to the same dict, not a copy.
    MODEL_VARIANTS = SCALE_CONFIGS

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            num_classes: int = 1000,
            scale: VitScale = "base",
            patch_size: Union[int, Tuple[int, int]] = 16,
            include_top: bool = True,
            pooling: Optional[PoolingMode] = None,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            pos_dropout_rate: float = 0.0,
            stem_norm_layer: StemNormLayer = "batch",
            kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            normalization_type: NormalizationType = "layer_norm",
            normalization_position: Literal['pre', 'post'] = "pre",
            ffn_type: FFNType = "mlp",
            activation: Union[str, callable] = "gelu",
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize Vision Transformer model with Hierarchical MLP stem."""
        # Auto-generate name if not provided
        if name is None:
            name = f"vision_transformer_hmlp_{scale}"

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

        if stem_norm_layer not in ["batch", "layer"]:
            raise ValueError(f"Unsupported stem_norm_layer: {stem_norm_layer}. Choose from ['batch', 'layer']")

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout_rate must be between 0 and 1, got {attention_dropout_rate}")

        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(f"pos_dropout_rate must be between 0 and 1, got {pos_dropout_rate}")

        if not (0.0 <= stochastic_depth_rate <= 1.0):
            raise ValueError(f"stochastic_depth_rate must be between 0 and 1, got {stochastic_depth_rate}")

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
        self.stem_norm_layer = str(stem_norm_layer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = bias_regularizer
        self.normalization_type = str(normalization_type)
        self.normalization_position = str(normalization_position)
        self.ffn_type = str(ffn_type)
        self.activation = activation
        self.use_stochastic_depth = bool(use_stochastic_depth)
        self.stochastic_depth_rate = float(stochastic_depth_rate)

        # Get model configuration from scale
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate derived parameters
        self.intermediate_size = int(self.embed_dim * self.mlp_ratio)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate derived parameters
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # Create all sub-layers in __init__; they are unbuilt until build() runs.
        # Uses factories for consistent component creation where available.

        # Hierarchical MLP Stem (specialized component, direct instantiation)
        self.stem = HierarchicalMLPStem(
            embed_dim=self.embed_dim,
            img_size=self.input_shape_config[:2],
            patch_size=self.patch_size,
            in_channels=img_c,
            norm_layer=self.stem_norm_layer,
            name="hierarchical_mlp_stem"
        )

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
        # Linear stochastic depth schedule across the transformer stack
        drop_path_rates = linear_drop_path_rates(
            self.num_layers, self.stochastic_depth_rate
        )
        for i in range(self.num_layers):
            # Stochastic depth rate for this layer (gate is outside the schedule)
            layer_drop_rate = (
                drop_path_rates[i] if self.use_stochastic_depth else 0.0
            )

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
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=layer_drop_rate,
                activation=self.activation,
                # DECISION plan-2026-08-23T091307-9a110062/D-560: each block gets its own
                # clone_initializer() copy; a shared instance replays one draw across all blocks. See decisions.md.
                use_bias=True,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Final normalization using factory
        self.norm = create_normalization_layer(
            self.normalization_type,
            name="norm"
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

        logger.info(f"Created VisionTransformer-hMLP-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")
        logger.info(
            f"Image shape: {self.input_shape_config}, Patch size: {self.patch_size}, Num patches: {self.num_patches}")

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
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        # Build all sub-layers in computational order
        # Hierarchical MLP Stem
        dummy_input_shape = (None,) + self.input_shape_config
        self.stem.build(dummy_input_shape)

        # Positional embedding
        pos_input_shape = (None, self.max_seq_len, self.embed_dim)
        self.pos_embed.build(pos_input_shape)

        # Transformer layers
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)

        # Final normalization
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

        logger.info(f"Built VisionTransformer-hMLP-{self.scale} with {self.num_patches} patches")

        # Parent build must run last.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the Vision Transformer with hMLP stem.

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
        # Patch embedding: (batch_size, num_patches, embed_dim).
        x = self.stem(inputs, training=training)

        # Prepend the CLS token: (batch_size, seq_len, embed_dim).
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)

        # Positional embedding includes its own dropout.
        x = self.pos_embed(x, training=training)

        for layer in self.transformer_layers:
            x = layer(x, training=training)

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
            "stem_norm_layer": self.stem_norm_layer,
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
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
            custom_objects: Optional[Dict[str, Any]] = None
    ) -> "ViTHMLP":
        """Recreate a model from its serialized configuration.

        ``get_config`` serializes the initializers, regularizers and
        activation, so they need deserializing back into objects here;
        otherwise the raw config dicts reach ``__init__`` and get stored
        (and re-serialized) as dicts.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :param custom_objects: Optional mapping of names to custom callables,
            used to resolve an activation that is not registered with
            ``keras.saving.register_keras_serializable``.
        :type custom_objects: Optional[Dict[str, Any]]
        :return: A new ``ViTHMLP`` instance.
        :rtype: ViTHMLP
        """
        config = dict(config)
        for key in ("kernel_initializer", "bias_initializer"):
            if config.get(key) is not None:
                config[key] = initializers.deserialize(config[key])
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) is not None:
                config[key] = regularizers.deserialize(config[key])
        activation = config.get("activation")
        if activation is not None and not isinstance(activation, str):
            config["activation"] = deserialize_activation(
                activation, custom_objects=custom_objects
            )
        return cls(**config)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            **kwargs: Any
    ) -> "ViTHMLP":
        """Create a ViTHMLP model from a predefined variant.

        :param variant: One of ``'tiny'``, ``'small'``, ``'base'``,
            ``'large'``, ``'huge'``.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input image shape ``(height, width, channels)``.
        :type input_shape: Tuple[int, int, int]
        :param kwargs: Additional keyword arguments for the ``ViTHMLP``
            constructor.
        :return: A new ``ViTHMLP`` instance.
        :rtype: ViTHMLP
        :raises ValueError: If ``variant`` is not recognized.

        Example:
            .. code-block:: python

                model = ViTHMLP.from_variant("base", num_classes=1000)
                model = ViTHMLP.from_variant(
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

    def get_feature_extractor(self) -> "ViTHMLP":
        """Return a feature-extractor twin of this model.

        Copies every configuration value, sets ``include_top=False`` and
        ``pooling='cls'``. Note this constructs a new, randomly initialized
        model; it does not transfer this instance's weights.

        :return: New ``ViTHMLP`` instance configured for CLS-token feature
            extraction.
        :rtype: ViTHMLP
        :raises ValueError: If the model was not properly initialized.
        """
        if not hasattr(self, 'input_shape_config') or not self.input_shape_config:
            raise ValueError("Model must be properly initialized before creating feature extractor")

        return ViTHMLP(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            pos_dropout_rate=self.pos_dropout_rate,
            stem_norm_layer=self.stem_norm_layer,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            normalization_type=self.normalization_type,
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            activation=self.activation,
            use_stochastic_depth=self.use_stochastic_depth,
            stochastic_depth_rate=self.stochastic_depth_rate,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Print detailed model summary with architecture information."""
        logger.info("Vision Transformer with hMLP Stem Model Summary")
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
        logger.info(f"Stem Normalization: {self.stem_norm_layer}")
        logger.info(f"Transformer Normalization Type: {self.normalization_type}")
        logger.info(f"Normalization Position: {self.normalization_position}")
        logger.info(f"FFN Type: {self.ffn_type}")
        logger.info(f"Activation: {self.activation}")
        logger.info(f"Use Stochastic Depth: {self.use_stochastic_depth}")
        logger.info(f"Stochastic Depth Rate: {self.stochastic_depth_rate}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self.built:
            logger.info(f"Total Parameters: {self.count_params():,}")

        # Additional architecture information
        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        logger.info(f"Patches per dimension: {img_h // patch_h} x {img_w // patch_w}")
        logger.info("hMLP Stem Processing: 4×4 → 8×8 → 16×16 pixels (patch_size=16)")


# ---------------------------------------------------------------------
# Factory Functions for Convenient Model Creation
# ---------------------------------------------------------------------


def create_vit_hmlp(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 1000,
        scale: VitScale = "base",
        patch_size: Union[int, Tuple[int, int]] = 16,
        include_top: bool = True,
        pooling: Optional[PoolingMode] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        pos_dropout_rate: float = 0.0,
        stem_norm_layer: StemNormLayer = "batch",
        kernel_initializer: Union[str, initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        normalization_type: NormalizationType = "layer_norm",
        normalization_position: Literal['pre', 'post'] = "pre",
        ffn_type: FFNType = "mlp",
        activation: Union[str, callable] = "gelu",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any
) -> ViTHMLP:
    """Create a Vision Transformer with a hierarchical MLP stem.

    Validates its own parameters, then delegates model construction to
    :class:`ViTHMLP`.

    :param input_shape: Input image shape ``(height, width, channels)``; must
        be compatible with ``patch_size``.
    :type input_shape: Tuple[int, int, int]
    :param num_classes: Number of output classes. Only used when
        ``include_top=True``.
    :type num_classes: int
    :param scale: Model scale, one of ``'tiny'``, ``'small'``, ``'base'``,
        ``'large'``, ``'huge'``.
    :type scale: VitScale
    :param patch_size: Patch size; an int gives square patches.
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
    :param stem_norm_layer: Normalization inside the hMLP stem: ``'batch'``
        (better accuracy) or ``'layer'`` (more stable at small batch sizes).
    :type stem_norm_layer: StemNormLayer
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
    :param normalization_position: ``'pre'`` or ``'post'``.
    :type normalization_position: Literal['pre', 'post']
    :param ffn_type: Feed-forward network identifier.
    :type ffn_type: FFNType
    :param activation: FFN activation.
    :type activation: Union[str, callable]
    :param use_stochastic_depth: Whether layers apply stochastic depth.
    :type use_stochastic_depth: bool
    :param stochastic_depth_rate: Maximum drop-path rate.
    :type stochastic_depth_rate: float
    :param kwargs: Additional keyword arguments forwarded to the
        :class:`ViTHMLP` constructor.
    :return: ``ViTHMLP`` model instance.
    :rtype: ViTHMLP
    :raises ValueError: If any parameter is invalid.

    Example:
        .. code-block:: python

            model = create_vit_hmlp(
                input_shape=(224, 224, 3),
                num_classes=1000,
                scale='base'
            )

            feature_model = create_vit_hmlp(
                input_shape=(384, 384, 3),
                scale='small',
                include_top=False,
                pooling='cls',
                normalization_type='rms_norm',
                ffn_type='swiglu'
            )
    """
    # Validate basic parameters before model creation
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3-element tuple/list, got {input_shape}")

    if any(dim <= 0 for dim in input_shape):
        raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")

    # Validate patch_size and ensure compatibility with input_shape
    if isinstance(patch_size, int):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        patch_h = patch_w = patch_size
    else:
        if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
            raise ValueError(f"patch_size must be int or 2-element tuple/list, got {patch_size}")
        patch_h, patch_w = patch_size
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")

    img_h, img_w = input_shape[:2]
    if img_h % patch_h != 0:
        raise ValueError(f"Image height ({img_h}) must be divisible by patch height ({patch_h})")
    if img_w % patch_w != 0:
        raise ValueError(f"Image width ({img_w}) must be divisible by patch width ({patch_w})")

    # Calculate and validate number of patches
    num_patches = (img_h // patch_h) * (img_w // patch_w)
    if num_patches <= 0:
        raise ValueError(f"Number of patches must be positive, got {num_patches}")
    # 10000 patches is a practical upper bound before memory becomes a concern.
    if num_patches > 10000:
        logger.warning(f"Large number of patches ({num_patches}) may cause memory issues")

    # Create model instance
    model = ViTHMLP(
        input_shape=input_shape,
        num_classes=num_classes,
        scale=scale,
        patch_size=patch_size,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        pos_dropout_rate=pos_dropout_rate,
        stem_norm_layer=stem_norm_layer,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        activation=activation,
        use_stochastic_depth=use_stochastic_depth,
        stochastic_depth_rate=stochastic_depth_rate,
        **kwargs
    )

    logger.info(f"VisionTransformer-hMLP-{scale} created successfully")
    logger.info(f"Configuration: {num_patches} patches ({img_h // patch_h}x{img_w // patch_w}), {num_classes} classes")
    logger.info(f"hMLP Stem: Progressive processing with {stem_norm_layer} normalization")
    return model


# ---------------------------------------------------------------------
# Utility Functions for Masked Self-Supervised Learning
# ---------------------------------------------------------------------


def create_inputs_with_masking(
        batch_size: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.4,
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Create masked input images and a matching mask, for self-supervised training.

    Because the hMLP stem processes each patch independently, masking can be
    applied before or after it with the same result, which is what MAE and
    BeiT-style training needs.

    :param batch_size: Batch size for the generated data.
    :type batch_size: int
    :param image_size: Image dimensions ``(height, width)``.
    :type image_size: Tuple[int, int]
    :param patch_size: Patch dimensions ``(height, width)``.
    :type patch_size: Tuple[int, int]
    :param mask_ratio: Fraction of patches to mask, in ``[0.0, 1.0]``.
    :type mask_ratio: float
    :return: ``(images, mask)``, where ``mask`` is 1 for a masked patch and 0
        for a visible one.
    :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
    :raises ValueError: If ``mask_ratio`` leaves ``[0.0, 1.0]``.

    Example:
        .. code-block:: python

            images, mask = create_inputs_with_masking(batch_size=32, mask_ratio=0.75)
            model = create_vit_hmlp(scale='base')
            model.build(images.shape)
            masked_patches, _ = apply_mask_after_stem(model.stem, images, mask)
    """
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")

    # Create random images
    images = keras.random.normal([batch_size, image_size[0], image_size[1], 3])

    # Calculate number of patches
    num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
    num_mask = int(mask_ratio * num_patches)

    # Create mask for each sample in batch
    masks = []
    for _ in range(batch_size):
        # Create random mask for this sample
        indices = keras.random.shuffle(ops.arange(num_patches, dtype='int32'))[:num_mask]
        mask_sample = ops.zeros([num_patches], dtype='float32')

        # Set masked positions to 1
        mask_sample = ops.scatter_update(
            mask_sample,
            ops.expand_dims(indices, 1),
            ops.ones([num_mask], dtype='float32')
        )
        masks.append(mask_sample)

    mask = ops.stack(masks, axis=0)

    logger.info(f"Created masked inputs: {batch_size} samples, {mask_ratio:.1%} masking ratio")
    return images, mask


def apply_mask_after_stem(
        stem: HierarchicalMLPStem,
        images: keras.KerasTensor,
        mask: keras.KerasTensor
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Process images through the hMLP stem, then zero out the masked patches.

    Since the stem never mixes information across patch boundaries, masking
    after it gives the same result as masking before it.

    :param stem: Hierarchical MLP stem instance.
    :type stem: HierarchicalMLPStem
    :param images: Input images of shape ``(batch_size, height, width, channels)``.
    :type images: keras.KerasTensor
    :param mask: Mask tensor of shape ``(batch_size, num_patches)``, 1 for a
        masked patch.
    :type mask: keras.KerasTensor
    :return: ``(masked_patches, mask)``, where masked patches are zeroed.
    :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]

    Example:
        .. code-block:: python

            model = create_vit_hmlp(scale='base')
            model.build((None, 224, 224, 3))
            images, mask = create_inputs_with_masking(batch_size=4, mask_ratio=0.75)
            masked_patches, mask = apply_mask_after_stem(model.stem, images, mask)
    """
    # Shape: (batch_size, num_patches, embed_dim).
    patches = stem(images)

    # Shape: (batch_size, num_patches, 1).
    mask_expanded = ops.expand_dims(mask, -1)

    # Broadcast mask to match patch dimensions.
    embed_dim = ops.shape(patches)[-1]
    mask_expanded = ops.repeat(mask_expanded, embed_dim, axis=-1)

    # Apply mask: multiply by (1 - mask) to zero out masked patches
    masked_patches = patches * (1 - mask_expanded)

    logger.info("Applied mask to hMLP stem output - no information leakage between patches")
    return masked_patches, mask


# ---------------------------------------------------------------------