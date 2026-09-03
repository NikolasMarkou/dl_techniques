"""
FastVLM: a hybrid convolution/transformer image classifier, built by the
``FastVLM`` class, with size presets from "nano" to "huge" via
``FastVLM.from_variant`` / ``create_fastvlm``.

The first two stages mix tokens with convolutional RepMixer blocks, cheap
because they never compute pairwise attention; the last stage switches to
transformer attention, giving the network global context only where it is
worth the cost. Resolution halves after each of the first two stages, so
the stack goes from a full-resolution stem down to ``H/16`` before the
attention stage runs.

This is a hybrid classifier assembled from this repository's own blocks,
not a weight-compatible port of a published FastViT/FastVLM checkpoint.
``dl_techniques.layers.repmixer_block.RepMixerBlock`` shares a name with
FastViT's RepMixer but is a different construction; the FastViT port lives
in ``dl_techniques.layers.fastvit``. No pretrained weights are distributed
for this package.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2023. MobileOne: An Improved One Millisecond Mobile
      Backbone. (https://arxiv.org/abs/2206.04040)
    - Dosovitskiy et al., 2021. An Image is Worth 16x16 Words.
      (https://arxiv.org/abs/2010.11929)
"""

import keras
from typing import Optional, Union, List, Dict, Any, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention.factory import AttentionType
from dl_techniques.layers.repmixer_block import RepMixerBlock, ConvolutionalStem

from .components import AttentionBlockVLM
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.fastvlm.model")
class FastVLM(keras.Model):
    """
    Hybrid convolution/transformer classifier: RepMixer stages, then attention.

    The model has not been trained or benchmarked in this repository; no
    performance claim is made for it.

    Architecture:

    .. code-block:: text

        input [B, H, W, 3]
               │
               ▼
        ┌─────────────────────┐
        │  ConvolutionalStem   │
        └──────────┬───────────┘  [B, H/4, W/4, embed_dims[0]]
                    ▼
        ┌─────────────────────┐
        │  Stage 1: RepMixer   │  x depths[0]
        │  + downsample        │
        └──────────┬───────────┘  [B, H/8, W/8, embed_dims[1]]
                    ▼
        ┌─────────────────────┐
        │  Stage 2: RepMixer   │  x depths[1]
        │  + downsample        │
        └──────────┬───────────┘  [B, H/16, W/16, embed_dims[2]]
                    ▼
        ┌─────────────────────┐
        │  Stage 3: Attention  │  x depths[2], no downsample
        └──────────┬───────────┘
                    │
          ┌─────────┴─────────┐
          ▼ (include_top)      ▼ (include_top=False)
        ┌───────────────┐   output [B, H/16, W/16, embed_dims[-1]]
        │  GAP + Dense   │
        └───────┬────────┘
                 ▼
        output [B, num_classes]

    Variants (``MODEL_VARIANTS``):

    .. code-block:: text

        name    embed_dims        depths     mlp_ratio  use_se
        nano    [24, 48, 96]      [1, 2, 3]  2.0        False
        tiny    [32, 64, 128]     [2, 3, 4]  3.0        False
        small   [48, 96, 192]     [3, 4, 6]  4.0        False
        base    [64, 128, 256]    [3, 4, 6]  4.0        False
        large   [96, 192, 384]    [4, 6, 8]  4.0        True
        huge    [128, 256, 512]   [6, 8, 12] 4.0        True

    :param num_classes: Number of output classes. Use ``0`` for feature
        extraction only. Defaults to ``1000``.
    :type num_classes: int
    :param embed_dims: Feature dimension per stage, 3 positive values.
        Defaults to ``[64, 128, 256]``.
    :type embed_dims: Optional[List[int]]
    :param depths: Block count per stage, 3 non-negative values. Defaults
        to ``[3, 4, 6]``.
    :type depths: Optional[List[int]]
    :param num_heads: Attention-head count per stage, 3 positive values
        each dividing the matching ``embed_dims`` entry. ``None`` derives
        ``max(1, dim // 32)`` per stage.
    :type num_heads: Optional[List[int]]
    :param mlp_ratio: FFN expansion ratio in the transformer and RepMixer
        blocks. Defaults to ``4.0``.
    :type mlp_ratio: float
    :param dropout_rate: Dropout rate applied throughout, in ``[0, 1]``.
        Defaults to ``0.0``.
    :type dropout_rate: float
    :param drop_path_rate: Stochastic-depth rate, in ``[0, 1]``. Defaults
        to ``0.1``.
    :type drop_path_rate: float
    :param use_se: Whether MobileOne blocks use Squeeze-and-Excitation.
        Defaults to ``False``.
    :type use_se: bool
    :param attention_type: ``'multi_head'``, ``'window'`` or
        ``'group_query'`` for the stage-3 blocks. Defaults to
        ``'group_query'``, the only option that carries positional
        information into stage 3 (see the DECISION comment on
        ``attention_type`` below).
    :type attention_type: str
    :param use_layer_scale: Whether attention blocks use layer scaling.
        Defaults to ``True``.
    :type use_layer_scale: bool
    :param attention_max_seq_len: RoPE table length for stage-3 blocks,
        consumed only when ``attention_type`` is ``'group_query'``. The
        stage-3 grid is ``(H/16) * (W/16)`` tokens; the default ``2048``
        covers inputs up to roughly 720px. Defaults to ``2048``.
    :type attention_max_seq_len: int
    :param activation: Activation used throughout. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for conv kernels. Defaults to
        ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param include_top: Whether to include the classification head.
        Defaults to ``True``.
    :type include_top: bool
    :param input_shape: Input shape. ``None`` defaults to ``(224, 224, 3)``.
    :type input_shape: Optional[Tuple[int, ...]]
    :param kwargs: Additional keyword arguments for the ``Model`` base
        class.

    Input shape:
        4D tensor: ``(batch_size, height, width, 3)``.

    Output shape:
        ``(batch_size, num_classes)`` if ``include_top`` and
        ``num_classes > 0``; otherwise
        ``(batch_size, H/16, W/16, embed_dims[-1])``.

    :ivar stem: The ``ConvolutionalStem`` doing initial feature extraction.
    :ivar stages: The three stage blocks (RepMixer, RepMixer, Attention).
    :ivar head: The classification head, or ``None`` when not built.
    :ivar downsample_layers: The two downsampling convs between stages.

    Example:
        .. code-block:: python

            model = FastVLM.from_variant("base", num_classes=1000)
            model.compile(optimizer='adamw', loss='categorical_crossentropy')

            backbone = FastVLM.from_variant("base", include_top=False)
            features = backbone(images)  # (B, H/16, W/16, 256)
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "nano": {
            "embed_dims": [24, 48, 96],
            "depths": [1, 2, 3],
            "num_heads": [1, 2, 3],
            "mlp_ratio": 2.0,
            "dropout_rate": 0.0,
            "drop_path_rate": 0.0,
            "use_se": False
        },
        "tiny": {
            "embed_dims": [32, 64, 128],
            "depths": [2, 3, 4],
            "num_heads": [1, 2, 4],
            "mlp_ratio": 3.0,
            "dropout_rate": 0.0,
            "drop_path_rate": 0.05,
            "use_se": False
        },
        "small": {
            "embed_dims": [48, 96, 192],
            "depths": [3, 4, 6],
            "num_heads": [2, 3, 6],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.1,
            "use_se": False
        },
        "base": {
            "embed_dims": [64, 128, 256],
            "depths": [3, 4, 6],
            "num_heads": [2, 4, 8],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.1,
            "use_se": False
        },
        "large": {
            "embed_dims": [96, 192, 384],
            "depths": [4, 6, 8],
            "num_heads": [3, 6, 12],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.2,
            "use_se": True
        },
        "huge": {
            "embed_dims": [128, 256, 512],
            "depths": [6, 8, 12],
            "num_heads": [4, 8, 16],
            "mlp_ratio": 4.0,
            "dropout_rate": 0.1,
            "drop_path_rate": 0.3,
            "use_se": True
        }
    }

    def __init__(
            self,
            num_classes: int = 1000,
            embed_dims: Optional[List[int]] = None,
            depths: Optional[List[int]] = None,
            num_heads: Optional[List[int]] = None,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.1,
            use_se: bool = False,
            attention_type: AttentionType = 'group_query',
            use_layer_scale: bool = True,
            attention_max_seq_len: int = 2048,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
        # None sentinels, not shared mutable list defaults.
        if input_shape is None:
            input_shape = (224, 224, 3)
        if embed_dims is None:
            embed_dims = [64, 128, 256]
        if depths is None:
            depths = [3, 4, 6]

        if num_classes < 0:
            raise ValueError(f"num_classes must be non-negative, got {num_classes}")
        if len(embed_dims) != 3:
            raise ValueError(f"embed_dims must have 3 elements, got {len(embed_dims)}")
        if len(depths) != 3:
            raise ValueError(f"depths must have 3 elements, got {len(depths)}")
        if any(dim <= 0 for dim in embed_dims):
            raise ValueError("All embed_dims must be positive")
        if any(depth < 0 for depth in depths):
            raise ValueError("All depths must be non-negative")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= drop_path_rate <= 1.0):
            raise ValueError(f"drop_path_rate must be between 0 and 1, got {drop_path_rate}")
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        if num_heads is None:
            num_heads = [max(1, dim // 32) for dim in embed_dims]
        elif len(num_heads) != 3:
            raise ValueError(f"num_heads must have 3 elements, got {len(num_heads)}")

        for i, (dim, heads) in enumerate(zip(embed_dims, num_heads)):
            if heads <= 0:
                raise ValueError(f"All num_heads must be positive, got {heads} at index {i}")
            if dim % heads != 0:
                raise ValueError(
                    f"embed_dims[{i}] ({dim}) must be divisible by num_heads[{i}] ({heads})"
                )

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.use_se = use_se
        # DECISION plan-2026-08-18T140459-7991552f/D-044: default is 'group_query',
        # not 'multi_head' -- 'multi_head' has no RoPE and no positional embedding is added elsewhere, so stage 3 could not represent position (measured 5.36e-07 max deviation under a spatial permutation, i.e. exactly equivariant). See decisions.md.
        self.attention_type = attention_type
        self.attention_max_seq_len = attention_max_seq_len
        self.use_layer_scale = use_layer_scale
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self._input_shape = input_shape

        inputs = keras.Input(shape=input_shape)

        outputs = self._build_model(inputs)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created FastVLM model for input {input_shape} "
            f"with {sum(depths)} blocks total"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the FastVLM model architecture.

        This method runs once, from ``__init__``, to trace the ``outputs``
        tensor handed to ``super().__init__(inputs=, outputs=)``. The
        ``training=None`` on every sublayer call below is trace-only: Keras
        dispatches the real forward pass to ``call()`` below, which threads
        the caller's ``training`` value, so the value traced here never
        reaches a real forward pass.
        """
        # DECISION plan-2026-08-22T035419-a11304c8/D-010: keep training=None here,
        # not training=training -- call() below is what actually runs; this method only traces shapes once. See decisions.md.
        x = inputs

        self.stem = ConvolutionalStem(
            out_channels=self.embed_dims[0],
            use_se=self.use_se,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            name='stem'
        )
        x = self.stem(x, training=None)

        self.stages = []
        self.downsample_layers = []

        # Stage 1: RepMixer blocks + downsample
        stage1_blocks = []
        for i in range(self.depths[0]):
            stage1_blocks.append(
                RepMixerBlock(
                    dim=self.embed_dims[0],
                    expansion_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    name=f'stage1_block_{i}'
                )
            )
        stage1 = keras.Sequential(stage1_blocks, name='stage1')
        self.stages.append(stage1)
        x = stage1(x, training=None)

        downsample_1_2 = keras.layers.Conv2D(
            filters=self.embed_dims[1],
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='downsample_1_2'
        )
        self.downsample_layers.append(downsample_1_2)
        x = downsample_1_2(x, training=None)

        stage2_blocks = []
        for i in range(self.depths[1]):
            stage2_blocks.append(
                RepMixerBlock(
                    dim=self.embed_dims[1],
                    expansion_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    name=f'stage2_block_{i}'
                )
            )
        stage2 = keras.Sequential(stage2_blocks, name='stage2')
        self.stages.append(stage2)
        x = stage2(x, training=None)

        downsample_2_3 = keras.layers.Conv2D(
            filters=self.embed_dims[2],
            kernel_size=3,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            name='downsample_2_3'
        )
        self.downsample_layers.append(downsample_2_3)
        x = downsample_2_3(x, training=None)

        stage3_blocks = []
        for i in range(self.depths[2]):
            block_drop_rate = self.drop_path_rate * (
                    (sum(self.depths[:2]) + i) / (sum(self.depths) - 1)
            ) if sum(self.depths) > 1 else 0.0

            # DECISION plan-2026-08-11T201945-91938f65/D-003: block_drop_rate feeds
            # stochastic_depth_rate, never dropout_rate -- mixing the two collapses independent knobs. See decisions.md.
            stage3_blocks.append(
                AttentionBlockVLM(
                    dim=self.embed_dims[2],
                    num_heads=self.num_heads[2],
                    mlp_ratio=self.mlp_ratio,
                    attention_type=self.attention_type,
                    dropout_rate=self.dropout_rate,
                    use_stochastic_depth=True,
                    stochastic_depth_rate=block_drop_rate,
                    use_layer_scale=self.use_layer_scale,
                    max_seq_len=self.attention_max_seq_len,
                    name=f'stage3_attention_{i}'
                )
            )
        stage3 = keras.Sequential(stage3_blocks, name='stage3')
        self.stages.append(stage3)
        x = stage3(x, training=None)

        if self.include_top and self.num_classes > 0:
            head_layers = [
                keras.layers.GlobalAveragePooling2D(name='gap'),
                keras.layers.Dense(
                    self.num_classes,
                    kernel_initializer=self.kernel_initializer,
                    name='classifier'
                )
            ]

            if self.dropout_rate > 0.0:
                head_layers.insert(-1, keras.layers.Dropout(self.dropout_rate, name='head_dropout'))

            self.head = keras.Sequential(head_layers, name='classification_head')
            x = self.head(x, training=None)
        else:
            self.head = None

        return x

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through FastVLM."""
        x = self.stem(inputs, training=training)
        x = self.stages[0](x, training=training)
        x = self.downsample_layers[0](x, training=training)
        x = self.stages[1](x, training=training)
        x = self.downsample_layers[1](x, training=training)
        x = self.stages[2](x, training=training)

        if self.head is not None:
            x = self.head(x, training=training)

        return x

    def extract_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """
        Return the stem and each stage's output feature map.

        :param inputs: Input tensor of shape ``(batch_size, height, width, 3)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Four feature tensors, in order: stem output
            ``[B, H/4, W/4, embed_dims[0]]``, stage 1 output (same shape),
            stage 2 output ``[B, H/8, W/8, embed_dims[1]]``, and stage 3
            output ``[B, H/16, W/16, embed_dims[2]]``.
        :rtype: List[keras.KerasTensor]
        """
        features = []

        x = self.stem(inputs, training=training)
        features.append(x)

        x = self.stages[0](x, training=training)
        features.append(x)

        x = self.downsample_layers[0](x, training=training)
        x = self.stages[1](x, training=training)
        features.append(x)

        x = self.downsample_layers[1](x, training=training)
        x = self.stages[2](x, training=training)
        features.append(x)

        return features

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any
    ) -> "FastVLM":
        """
        Create a FastVLM model from a predefined variant.

        :param variant: One of ``"nano"``, ``"tiny"``, ``"small"``,
            ``"base"``, ``"large"``, ``"huge"``.
        :type variant: str
        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: Input shape. ``None`` defaults to
            ``(224, 224, 3)``.
        :type input_shape: Optional[Tuple[int, ...]]
        :param kwargs: Additional arguments passed to the constructor,
            overriding the variant's entries.
        :return: A configured ``FastVLM`` instance.
        :rtype: FastVLM
        :raises ValueError: If ``variant`` is not recognized.

        Example:
            .. code-block:: python

                model = FastVLM.from_variant("base", num_classes=1000)
                backbone = FastVLM.from_variant("base", include_top=False)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        logger.info(f"Creating FastVLM-{variant.upper()} model")
        logger.info(f"Configuration: {config}")

        config.update(kwargs)

        return cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'embed_dims': self.embed_dims,
            'depths': self.depths,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'use_se': self.use_se,
            'attention_type': self.attention_type,
            'attention_max_seq_len': self.attention_max_seq_len,
            'use_layer_scale': self.use_layer_scale,
            'activation': keras.activations.serialize(
                keras.activations.get(self.activation)
            ),
            'kernel_initializer': keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            'include_top': self.include_top,
            'input_shape': self._input_shape,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVLM":
        """Create a model from a configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: A ``FastVLM`` instance.
        :rtype: FastVLM
        """
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = keras.initializers.deserialize(
                config['kernel_initializer']
            )

        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional information."""
        super().summary(**kwargs)

        total_blocks = sum(self.depths)
        logger.info(f"FastVLM configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Embed dimensions: {self.embed_dims}")
        logger.info(f"  - Stage depths: {self.depths}")
        logger.info(f"  - Attention heads: {self.num_heads}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - MLP ratio: {self.mlp_ratio}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Use SE: {self.use_se}")
        logger.info(f"  - Attention type: {self.attention_type}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top and self.num_classes > 0:
            logger.info(f"  - Number of classes: {self.num_classes}")


def create_fastvlm(
        variant: str = "base",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs: Any
) -> FastVLM:
    """
    Create a FastVLM model from a named variant.

    :param variant: One of ``FastVLM.MODEL_VARIANTS`` (``"nano"``,
        ``"tiny"``, ``"small"``, ``"base"``, ``"large"``, ``"huge"``).
    :type variant: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: Input shape. ``None`` defaults to
        ``(224, 224, 3)``.
    :type input_shape: Optional[Tuple[int, ...]]
    :param kwargs: Additional arguments passed to the model constructor,
        overriding the variant's entries.
    :return: A configured ``FastVLM`` instance.
    :rtype: FastVLM
    :raises ValueError: If ``variant`` is not a known variant name, or if
        any resolved argument is out of range.

    Example:
        .. code-block:: python

            model = create_fastvlm("tiny", num_classes=10, input_shape=(32, 32, 3))
            backbone = create_fastvlm("base", include_top=False)
    """
    return FastVLM.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )
