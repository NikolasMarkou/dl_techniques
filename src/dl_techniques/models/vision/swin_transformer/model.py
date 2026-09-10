"""
SwinTransformer: a hierarchical vision transformer built from shifted-window attention.

Defines :class:`SwinTransformer`, four stages of windowed attention blocks with patch
merging between them and an optional classification head, plus the
:func:`create_swin_transformer` factory. Global self-attention costs ``O((HW)^2)`` in
token count, too expensive at the resolutions dense prediction needs. Restricting
attention to fixed local windows brings that to ``O(M^2 * HW)``, linear in image area,
but stops information from crossing a window boundary. Swin alternates the partition
rather than enlarging it: even blocks partition on a regular grid, odd blocks shift it
by half a window, so two consecutive blocks connect every token to a neighbourhood
wider than one window. ``PatchMerging`` halves resolution and doubles width between
stages. There is no absolute positional embedding; each head learns a relative position
bias inside its window. The class subclasses ``keras.Model`` but builds its graph with
the functional API in ``__init__``, so it overrides neither ``build`` nor ``call`` and
the input shape is fixed at construction. No checkpoints ship with this package, so
``pretrained=True`` raises ``NotImplementedError``.

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
from typing import Optional, Union, Tuple, Dict, Any, Sequence

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.pooling.patch_merging import PatchMerging
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.transformers.swin_transformer_block import SwinTransformerBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# The official implementation's `_init_weights` value, not Keras' glorot_uniform.
# DECISION plan-2026-08-23T091307-9a110062/D-502: keep this a config dict, not an
# Initializer instance, which replays the same draw on every use. See decisions.md.
REFERENCE_KERNEL_INITIALIZER: Dict[str, Any] = {
    "class_name": "TruncatedNormal",
    "config": {"stddev": 0.02},
}

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.swin_transformer.model")
class SwinTransformer(keras.Model):
    """Classify images with shifted-window attention over a feature pyramid.

    Windowed self-attention keeps computation linear in image size, patch merging
    builds the pyramid, and the half-window shift on every second block widens the
    receptive field without a global attention matrix. The graph is assembled in
    ``__init__`` with the functional API, so the input shape is fixed at construction
    and Keras builds the sub-layers itself.

    Architecture:

    .. code-block:: text

        input [B, H, W, 3]
                 │
                 ▼
        ┌───────────────────┐
        │ patch_embed       │
        └───────────────────┘
                 │  [B, H/4 * W/4, embed_dim]
                 ▼
        ┌───────────────────┐
        │ patch_embed_norm  │
        └───────────────────┘
                 │  reshape back to a 4D grid
                 ▼
        ┌───────────────────┐
        │ stage 0 blocks    │  depths[0]
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ patch_merge_1     │
        └───────────────────┘
                 │  [B, H/8, W/8, embed_dim*2]
                 ▼
        ┌───────────────────┐
        │ stage 1 blocks    │  depths[1]
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ patch_merge_2     │
        └───────────────────┘
                 │  [B, H/16, W/16, embed_dim*4]
                 ▼
        ┌───────────────────┐
        │ stage 2 blocks    │  depths[2]
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ patch_merge_3     │
        └───────────────────┘
                 │  [B, H/32, W/32, embed_dim*8]
                 ▼
        ┌───────────────────┐
        │ stage 3 blocks    │  depths[3]
        └───────────────────┘
                 │
           ┌─────┴─────┐
           ▼           ▼
        include_top   else
           │           │
           ▼           ▼
        head_norm    features
        avg pool     [B, H/32, W/32, embed_dim*8]
        classifier
           │
           ▼
        logits [B, num_classes]

    Per-block schedule:

    .. code-block:: text

        block b in stage s   shift_size          stochastic depth
        b even               0                   rates[k]
        b odd                window_size // 2    rates[k]

        k = sum(depths[:s]) + b
        rates = linear from 0 to drop_path_rate over sum(depths) blocks

    Consecutive blocks pair a regular partition with a shifted one.

    Variants:

    .. code-block:: text

        variant  embed_dim  depths          num_heads
        tiny        96      [2, 2, 6, 2]    [3, 6, 12, 24]
        small       96      [2, 2, 18, 2]   [3, 6, 12, 24]
        base       128      [2, 2, 18, 2]   [4, 8, 16, 32]
        large      192      [2, 2, 18, 2]   [6, 12, 24, 48]

    :param num_classes: Number of output classes. Must be positive whatever
        ``include_top`` is, though only the head reads it.
    :type num_classes: int
    :param embed_dim: Base embedding dimension for the first stage; later
        stages use ``2**i * embed_dim``.
    :type embed_dim: int
    :param depths: Number of Swin blocks per stage, exactly 4 elements.
    :type depths: Sequence[int]
    :param num_heads: Attention heads per stage, exactly 4 elements.
    :type num_heads: Sequence[int]
    :param window_size: Attention window size, typically 7 or 8.
    :type window_size: int
    :param mlp_ratio: Expansion ratio for each block's MLP.
    :type mlp_ratio: float
    :param qkv_bias: Whether to use bias in attention QKV projections.
    :type qkv_bias: bool
    :param dropout_rate: Dropout rate for attention projection and MLP, in [0, 1).
    :type dropout_rate: float
    :param attn_dropout_rate: Dropout rate applied to attention weights, in [0, 1).
    :type attn_dropout_rate: float
    :param drop_path_rate: Maximum stochastic depth rate, in [0, 1), scheduled
        linearly across all blocks of all stages.
    :type drop_path_rate: float
    :param patch_size: Patch size for the initial patch embedding.
    :type patch_size: int
    :param use_bias: Whether to use bias terms in linear layers. Also sets
        ``center`` on the two LayerNormalization layers.
    :type use_bias: bool
    :param kernel_initializer: Weight initializer. ``None`` resolves to
        ``TruncatedNormal(stddev=0.02)``, the official implementation's convention.
    :type kernel_initializer: str, dict, or keras.initializers.Initializer
    :param bias_initializer: Bias initializer, used when ``use_bias=True``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Weight regularizer.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param bias_regularizer: Bias regularizer, used when ``use_bias=True``.
    :type bias_regularizer: keras.regularizers.Regularizer or None
    :param include_top: Whether to include the classification head.
    :type include_top: bool
    :param input_shape: Input tensor shape ``(H, W, C)``. Defaults to ``(224, 224, 3)``.
        The patch grid is computed from it, so it fixes the resolution.
    :type input_shape: tuple or None
    :param **kwargs: Additional arguments for the Keras ``Model`` base class.

    :raises ValueError: If a size or rate is out of range, if ``depths`` or
        ``num_heads`` does not have 4 elements, or if ``input_shape`` is not 3D.

    :ivar patch_embed: Patch embedding layer.
    :vartype patch_embed: keras.layers.Layer
    :ivar patch_embed_norm: Normalization after patch embedding.
    :vartype patch_embed_norm: keras.layers.LayerNormalization
    :ivar stages: Nested list of :class:`SwinTransformerBlock` per stage.
    :vartype stages: List[List[SwinTransformerBlock]]
    :ivar patch_merge_layers: :class:`PatchMerging` layers between stages.
    :vartype patch_merge_layers: List[PatchMerging]
    :ivar head_layers: Norm, pooling and classifier, when ``include_top=True``.
    :vartype head_layers: List[keras.layers.Layer]

    Input shape:
        4D tensor: `(batch_size, height, width, channels)`
        Optimal when height and width are divisible by patch_size x 8.

    Output shape:
        - If include_top=True: `(batch_size, num_classes)` - classification logits
        - If include_top=False: `(batch_size, H/32, W/32, embed_dim*8)` - feature maps

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

    References:
        - Official implementation: https://github.com/microsoft/Swin-Transformer

    Note:
        A height or width not divisible by patch_size x 8 only costs compute:
        PatchMerging ceil-pads an odd grid and the declared output shape still holds.
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
            kernel_initializer: Optional[Union[str, Dict[str, Any], initializers.Initializer]] = None,
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
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

        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        # DECISION plan-2026-07-31T210633-b63a35aa/D-003: warn, never raise, on a
        # non-divisible height or width; the hard guard lives in PatchEmbedding2D.
        # See decisions.md.
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

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: store as a list; get_config
        # has always emitted lists and a tuple changes that shape. See decisions.md.
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

        # Resolved from a None sentinel rather than a module-level dict default,
        # so `initializers.get` produces a fresh instance per model.
        if kernel_initializer is None:
            kernel_initializer = REFERENCE_KERNEL_INITIALIZER
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.stages = []
        self.patch_merge_layers = []
        self.head_layers = []

        inputs = keras.Input(shape=input_shape, name="input")
        outputs = self._build_architecture(inputs)

        # super().__init__ runs last, since the functional graph has to exist first.
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created Swin Transformer: embed_dim={embed_dim}, "
            f"depths={depths}, num_heads={num_heads}, "
            f"total_blocks={sum(depths)}, input_shape={input_shape}"
        )

    def _build_architecture(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Assemble patch embedding, the four stages, the merges and the head.

        :param inputs: Input tensor from ``keras.Input()``.
        :type inputs: keras.KerasTensor
        :return: Output tensor (logits or features).
        :rtype: keras.KerasTensor
        """
        x = inputs

        x = self._create_patch_embedding(x)

        for stage_idx in range(self.NUM_STAGES):
            if stage_idx > 0:
                x = self._create_patch_merging(x, stage_idx)

            x = self._create_stage_blocks(x, stage_idx)

        if self.include_top:
            x = self._create_classification_head(x)

        return x

    def _create_patch_embedding(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Tokenize the image, normalize, and restore the 4D patch grid.

        :param x: Input image tensor.
        :type x: keras.KerasTensor
        :return: Grid of patch embeddings ``(B, H/patch, W/patch, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-23T091307-9a110062/D-540: every consumer gets its own
        # clone_initializer copy; a shared instance replays one draw. See decisions.md.
        self.patch_embed = create_embedding_layer(
            embedding_type="patch_2d",
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="patch_embed"
        )
        x = self.patch_embed(x)

        if self.PATCH_EMBED_NORM:
            self.patch_embed_norm = layers.LayerNormalization(
                epsilon=self.LAYERNORM_EPSILON,
                center=self.use_bias,
                scale=True,
                name="patch_embed_norm"
            )
            x = self.patch_embed_norm(x)

        # DECISION plan_2026-06-16_c8f3e9ca/D-004: reshape to 4D here, not by changing
        # PatchEmbedding2D's 3D output or the block's 4D input. See decisions.md.
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
        """Halve the grid and double the width before ``stage_idx``.

        :param x: Feature grid from the previous stage.
        :type x: keras.KerasTensor
        :param stage_idx: Index of the stage this merge feeds, 1 to 3.
        :type stage_idx: int
        :return: Downsampled feature grid.
        :rtype: keras.KerasTensor
        """
        input_dim = self.embed_dim * (2 ** (stage_idx - 1))

        patch_merge = PatchMerging(
            dim=input_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
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
        """Add one stage's blocks, alternating regular and shifted windows.

        :param x: Feature grid entering the stage.
        :type x: keras.KerasTensor
        :param stage_idx: Index of the stage, 0 to 3.
        :type stage_idx: int
        :return: Feature grid after ``depths[stage_idx]`` blocks.
        :rtype: keras.KerasTensor
        """
        stage_blocks = []
        depth = self.depths[stage_idx]
        num_heads = self.num_heads[stage_idx]
        stage_dim = self.embed_dim * (2 ** stage_idx)

        # Rates are scheduled over all blocks of all stages, not per stage.
        total_blocks = sum(self.depths)
        block_start_idx = sum(self.depths[:stage_idx])
        drop_path_rates = linear_drop_path_rates(total_blocks, self.drop_path_rate)

        for block_idx in range(depth):
            current_block_idx = block_start_idx + block_idx
            current_drop_path_rate = drop_path_rates[current_block_idx]

            shift_size = 0 if block_idx % 2 == 0 else self.window_size // 2

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
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )

            x = block(x)
            stage_blocks.append(block)

        self.stages.append(stage_blocks)
        return x

    def _create_classification_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Normalize, pool over the grid, and project to class logits.

        :param x: Final feature grid.
        :type x: keras.KerasTensor
        :return: Logits ``(B, num_classes)``.
        :rtype: keras.KerasTensor
        """
        head_norm = layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        x = head_norm(x)

        gap = layers.GlobalAveragePooling2D(name="global_avg_pool")
        x = gap(x)

        if self.num_classes > 0:
            classifier = layers.Dense(
                units=self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
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
        """Create a Swin Transformer from a predefined variant configuration.

        :param variant: Model variant (``"tiny"``, ``"small"``, ``"base"``, ``"large"``).
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input shape. Defaults to ``(224, 224, 3)``.
        :type input_shape: tuple or None
        :param **kwargs: Additional arguments passed to the constructor, overriding
            the variant's own values.
        :return: Configured :class:`SwinTransformer` instance.
        :rtype: SwinTransformer
        :raises ValueError: If ``variant`` is not recognized, or a resolved argument
            fails the constructor's checks.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127: copy the preset before the
        # update, or a caller's override poisons MODEL_VARIANTS. See decisions.md.
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        config.update(kwargs)
        logger.info(f"Creating Swin Transformer-{variant.upper()} model")

        return cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the Keras base config plus every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
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
        """Create a model from a configuration dictionary.

        :param config: Dict as returned by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new model.
        :rtype: SwinTransformer
        """
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
        """Print the Keras summary, then log the architecture settings.

        :param **kwargs: Forwarded to ``keras.Model.summary``.
        """
        super().summary(**kwargs)

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
    """Build a Swin Transformer from a named variant, with input validation.

    :param variant: Model variant (``"tiny"``, ``"small"``, ``"base"``, ``"large"``).
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Input shape. Defaults to ``(224, 224, 3)``.
    :type input_shape: tuple or None
    :param pretrained: Must be ``False``; ``True`` raises ``NotImplementedError``
        since no Swin checkpoints ship with this package.
    :type pretrained: bool
    :param **kwargs: Additional arguments passed to the model constructor.
    :return: Configured :class:`SwinTransformer` instance.
    :rtype: SwinTransformer
    :raises ValueError: If ``variant`` is invalid or parameters are incompatible.
    :raises NotImplementedError: If ``pretrained=True``.

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
