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
``dl_techniques.layers.conv_blocks.repmixer_block.RepMixerBlock`` shares a name with
FastViT's RepMixer but is a different construction; the FastViT port lives
in ``dl_techniques.layers.fastvit``. No pretrained weights are distributed
for this package.

The classification head is the FastViT reference classifier: ``conv_exp``
(a depthwise 3x3 ``MobileOneBlock`` with squeeze-and-excitation that widens
the last stage by ``CLS_RATIO``), global average pooling, dropout, and one
linear ``Dense``. The model returns its logits.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2023. MobileOne: An Improved One Millisecond Mobile
      Backbone. (https://arxiv.org/abs/2206.04040)
    - Dosovitskiy et al., 2021. An Image is Worth 16x16 Words.
      (https://arxiv.org/abs/2010.11929)
    - Reference classifier: apple/ml-fastvit ``models/fastvit.py`` and
      apple/ml-fastvlm ``llava/model/multimodal_encoder/mobileclip/mci.py``
      (``conv_exp``, ``gap``, ``head``), and timm ``models/fastvit.py``
      (``final_conv``, ``ClassifierHead``).
"""

import keras
from typing import Optional, Union, List, Dict, Any, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention.factory import AttentionType
from dl_techniques.layers.conv_blocks.mobile_one_block import MobileOneBlock
from dl_techniques.layers.conv_blocks.repmixer_block import RepMixerBlock, ConvolutionalStem
from dl_techniques.layers.heads.vision import VisionTaskType
from dl_techniques.layers.heads.vision.factory import build_model_vision_head
from dl_techniques.utils.model_build import refuse_unknown_model_kwargs
from dl_techniques.utils.weight_transfer import (
    load_weights_from_checkpoint,
    pretrained_path_or_raise,
)

from .components import AttentionBlockVLM
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.activations.common import (
    resolve_activation,
    serialize_activation,
)

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
        ┌──────────────────────┐
        │  ConvolutionalStem   │
        └──────────┬───────────┘  [B, H/4, W/4, embed_dims[0]]
                   ▼
        ┌──────────────────────┐
        │  Stage 1: RepMixer   │  x depths[0]
        │  + downsample        │
        └──────────┬───────────┘  [B, H/8, W/8, embed_dims[1]]
                   ▼
        ┌──────────────────────┐
        │  Stage 2: RepMixer   │  x depths[1]
        │  + downsample        │
        └──────────┬───────────┘  [B, H/16, W/16, embed_dims[2]]
                   ▼
        ┌──────────────────────┐
        │  Stage 3: Attention  │  x depths[2], no downsample
        └───────────┬──────────┘
                    │
          ┌─────────┴─────────────────┐
          ▼ (include_top)             ▼ (include_top=False)
        ┌────────────────────────┐   output [B, H/16, W/16, embed_dims[-1]]
        │  conv_exp:             │
        │  MobileOneBlock 3x3,   │
        │  depthwise, SE,        │
        │  x CLS_RATIO channels  │
        └───────────┬────────────┘   [B, H/16, W/16, 2 * embed_dims[-1]]
                    ▼
        ┌────────────────────────┐
        │  classification_head:  │
        │  GAP -> [Dropout] ->   │
        │  Dense, no activation  │
        └───────────┬────────────┘
                    ▼
        logits [B, num_classes]

    The head is the FastViT reference classifier (apple/ml-fastvit and
    apple/ml-fastvlm build ``conv_exp``, ``gap`` and ``head`` only in
    classification mode; timm builds the same ``final_conv`` and
    ``ClassifierHead``). ``conv_exp`` belongs to the head: it is built only
    with ``include_top``, so the headless output is the raw stage-3 map. The
    pool, dropout and ``Dense`` are a shared ``ClassificationHead`` with no
    hidden block and no normalization; its ``'logits'`` are the model output.
    Compile with ``from_logits=True``.

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
    :param dropout_rate: Dropout rate applied throughout, in ``[0, 1)``,
        including the head's dropout before the classifier (timm's
        ``ClassifierHead`` ``drop_rate``). Defaults to ``0.0``.
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
    :param activation: Activation used throughout, ``conv_exp`` included.
        The head applies no activation to its logits. Its
        ``activation_type`` is the activation's name (``'relu'`` for
        ``activation='relu'``), which is what a hidden block added through
        ``head_config_overrides={'block_dims': ...}`` uses. A name the head's
        activation factory cannot build (``'tanh'``, a custom callable)
        raises ``ValueError`` only if such a block is requested; the
        reference head has none. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for every conv and ``Dense``
        kernel, ``conv_exp`` and the head's classifier included. Defaults to
        ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param include_top: Whether to include ``conv_exp`` and the
        classification head. Defaults to ``True``.
    :type include_top: bool
    :param input_shape: Input shape. ``None`` defaults to ``(224, 224, 3)``.
    :type input_shape: Optional[Tuple[int, ...]]
    :param head_config_overrides: Optional dict merged into the
        ``ClassificationHead`` configuration (e.g. ``{'block_dims': (256,)}``).
        ``name``, ``num_classes`` and ``task_type`` belong to the model and
        raise ``ValueError``. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: ``name``, ``trainable`` or ``dtype`` for the ``Model``
        base class. Any other keyword raises ``ValueError``: Keras'
        functional ``Model`` constructor would drop it silently.

    :raises ValueError: If an argument is out of range, a
        ``head_config_overrides`` key is reserved, or a keyword is unknown.

    Input shape:
        4D tensor: ``(batch_size, height, width, 3)``.

    Output shape:
        Logits ``(batch_size, num_classes)`` if ``include_top`` and
        ``num_classes > 0``; otherwise
        ``(batch_size, H/16, W/16, embed_dims[-1])``.

    :ivar stem: The ``ConvolutionalStem`` doing initial feature extraction.
    :ivar stages: The three stage blocks (RepMixer, RepMixer, Attention).
    :ivar conv_exp: The head's widening ``MobileOneBlock``, or ``None`` when
        no head is built.
    :ivar head: The ``ClassificationHead``, or ``None`` when not built.
    :ivar downsample_layers: The two downsampling convs between stages.
    :ivar activation: The resolved activation callable, produced by
        ``resolve_activation``. Note this is a resolved callable, not the raw
        constructor spec.

    Example:
        .. code-block:: python

            model = FastVLM.from_variant("base", num_classes=1000)
            model.compile(
                optimizer='adamw',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            )

            backbone = FastVLM.from_variant("base", include_top=False)
            features = backbone(images)  # (B, H/16, W/16, 256)
    """

    #: ``conv_exp`` widens the last stage by this factor: ``cls_ratio=2.0``
    #: in apple/ml-fastvit ``FastViT.__init__``, apple/ml-fastvlm ``mci.py``
    #: and timm ``FastVit.__init__``.
    CLS_RATIO = 2.0

    #: Layer names of the head. A backbone-only load skips exactly these, and
    #: every other weight-bearing layer must load.
    HEAD_LAYER_NAMES = ("conv_exp", "classification_head")

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
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        refuse_unknown_model_kwargs(kwargs, type(self).__name__)
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
        # [0, 1), not [0, 1]: the head's classifier dropout refuses 1.0.
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1 (1 excluded), got {dropout_rate}"
            )
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
        # DECISION plan-2026-09-15T094955-31fbe3db/D-008
        # Restores Invariant 1 (resolve activation in __init__, never deferred to
        # get_config/save time) — see plan.md Step 7 / decisions.md D-008. Do not
        # revert to storing the raw `activation` argument here.
        self.activation = resolve_activation(activation)
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self._input_shape = input_shape

        self._head_config_overrides = dict(head_config_overrides) if head_config_overrides else None

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
            self.conv_exp, self.head = self._build_head()
            x = self.conv_exp(x, training=None)
            x = self.head(x, training=None)['logits']
        else:
            self.conv_exp = None
            self.head = None

        return x

    def _build_head(self) -> Tuple[MobileOneBlock, keras.layers.Layer]:
        """Build the FastViT reference classifier: ``conv_exp``, then the head.

        ``conv_exp`` takes the reference's structural literals (kernel 3,
        stride 1, depthwise, ``CLS_RATIO`` times wider, squeeze-and-
        excitation, one conv branch). Everything else in it is this model's
        ``MobileOneBlock`` convention, the one its stem uses: Keras'
        BatchNorm epsilon (1e-3) and the block's own SE defaults. The pool,
        dropout and linear classifier are a ``ClassificationHead`` with no
        hidden block and no normalization, whose classifier draws from the
        model's ``kernel_initializer``.

        :return: ``(conv_exp, classification_head)``.
        :rtype: Tuple[MobileOneBlock, keras.layers.Layer]
        :raises ValueError: If ``head_config_overrides`` names a reserved
            key, or adds a hidden block while the model's activation has no
            counterpart in the head's activation factory.
        """
        conv_exp = MobileOneBlock(
            out_channels=int(self.embed_dims[-1] * self.CLS_RATIO),
            kernel_size=3,
            stride=1,
            group_size=1,
            use_se=True,
            num_conv_branches=1,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            name='conv_exp'
        )
        head = build_model_vision_head(
            VisionTaskType.CLASSIFICATION,
            {
                'num_classes': self.num_classes,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
                'block_dims': (),
                'normalization_type': None,
                # The model's activation by name. No reference block applies it;
                # a block added by an override does, and the head's activation
                # factory refuses a name it cannot build.
                'activation_type': getattr(
                    self.activation, '__name__', str(self.activation)),
                'dropout_rate': 0.0,
                'classifier_dropout_rate': self.dropout_rate,
                'kernel_initializer': self.kernel_initializer,
                # Deterministic layer name: by-name weight transfer depends on it.
                'name': 'classification_head',
            },
            self._head_config_overrides,
        )
        return conv_exp, head

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through FastVLM.

        :param inputs: Images ``(batch, height, width, 3)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Logits ``(batch, num_classes)`` with a head, otherwise the
            stage-3 map ``(batch, H/16, W/16, embed_dims[-1])``.
        :rtype: keras.KerasTensor
        """
        x = self.stem(inputs, training=training)
        x = self.stages[0](x, training=training)
        x = self.downsample_layers[0](x, training=training)
        x = self.stages[1](x, training=training)
        x = self.downsample_layers[1](x, training=training)
        x = self.stages[2](x, training=training)

        if self.head is not None:
            x = self.conv_exp(x, training=training)
            x = self.head(x, training=training)['logits']

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
        pretrained: Union[bool, str] = False,
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
        :param pretrained: ``False`` (default): random initialization. A
            ``str``: the path of a local ``.keras`` FastVLM checkpoint whose
            backbone is copied into the model by layer name, headed or
            headless. The checkpoint's head (``conv_exp`` and
            ``classification_head``) is never copied; the model's own head,
            if any, keeps its fresh initialization, so the class count may
            differ. Every backbone layer loads or the call raises (another
            depth or width is refused). To restore a whole model, head
            included, use ``keras.models.load_model``. ``True`` raises
            ``NotImplementedError``: no weights are distributed.
        :type pretrained: Union[bool, str]
        :param kwargs: Additional arguments passed to the constructor,
            overriding the variant's entries.
        :return: A configured ``FastVLM`` instance.
        :rtype: FastVLM
        :raises ValueError: If ``variant`` is not recognized, or a
            ``pretrained`` checkpoint does not fit the backbone.
        :raises NotImplementedError: If ``pretrained`` is True.
        :raises FileNotFoundError: If a ``pretrained`` path does not exist.
            Raised before the model is built.

        Example:
            .. code-block:: python

                model = FastVLM.from_variant("base", num_classes=1000)
                backbone = FastVLM.from_variant("base", include_top=False)
                warm = FastVLM.from_variant(
                    "base", num_classes=10, pretrained="fastvlm_base.keras")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        weights_path = pretrained_path_or_raise(pretrained, "FastVLM")

        config = cls.MODEL_VARIANTS[variant].copy()

        logger.info(f"Creating FastVLM-{variant.upper()} model")
        logger.info(f"Configuration: {config}")

        config.update(kwargs)

        model = cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **config
        )
        if weights_path is not None:
            model._load_backbone(weights_path)
        return model

    def _load_backbone(self, weights_path: str) -> None:
        """Copy every backbone layer of a ``.keras`` checkpoint into this model.

        The model is functional, so it is already built. Both sides' head
        layers (:attr:`HEAD_LAYER_NAMES`) are skipped and may be absent; any
        other layer that does not load, or a checkpoint layer with no home
        here, raises inside ``load_weights_from_checkpoint`` before a weight
        is copied.

        :param weights_path: Path of an existing ``.keras`` checkpoint.
        :type weights_path: str
        :raises ValueError: If the checkpoint does not fit the backbone, or
            the transfer report leaves a backbone layer unloaded.
        """
        report = load_weights_from_checkpoint(
            self,
            weights_path,
            skip_prefixes=self.HEAD_LAYER_NAMES,
            strict=True,
            allow_unloaded=self.HEAD_LAYER_NAMES,
        )
        # DECISION plan-2026-10-14T062616-81938e81/D-019: assert the backbone
        # loaded; do NOT trust the transfer's silence. A skip prefix that also
        # matched a backbone layer would leave it fresh with no error raised.
        backbone = [
            layer.name for layer in self.layers
            if layer.weights and not layer.name.startswith(self.HEAD_LAYER_NAMES)
        ]
        unloaded = [name for name in backbone if name not in report.loaded]
        if unloaded:
            raise ValueError(
                f"FastVLM checkpoint {weights_path!r} left backbone layer(s) "
                f"{unloaded} unloaded; a backbone load is all or nothing.\n"
                f"{report.summary_string()}"
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
            'activation': serialize_activation(self.activation),
            'kernel_initializer': keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            'include_top': self.include_top,
            'input_shape': self._input_shape,
            'head_config_overrides': self._head_config_overrides,
        })
        return config

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

# ---------------------------------------------------------------------

def create_fastvlm(
        variant: str = "base",
        pretrained: Union[bool, str] = False,
        **kwargs: Any
) -> FastVLM:
    """
    Create a FastVLM model from a named variant.

    A one-line delegate to :meth:`FastVLM.from_variant`, which owns every
    argument, ``pretrained`` included.

    :param variant: One of ``FastVLM.MODEL_VARIANTS`` (``"nano"``,
        ``"tiny"``, ``"small"``, ``"base"``, ``"large"``, ``"huge"``).
    :type variant: str
    :param pretrained: ``False``, or the path of a local ``.keras``
        checkpoint whose backbone is loaded. ``True`` raises
        ``NotImplementedError``. See :meth:`FastVLM.from_variant`.
    :type pretrained: Union[bool, str]
    :param kwargs: ``num_classes``, ``input_shape``, ``head_config_overrides``
        and any constructor argument, overriding the variant's entries.
    :return: A configured ``FastVLM`` instance.
    :rtype: FastVLM
    :raises ValueError: If ``variant`` is not a known variant name, or if
        any resolved argument is out of range or unknown.
    :raises NotImplementedError: If ``pretrained`` is True.
    :raises FileNotFoundError: If a ``pretrained`` path does not exist.

    Example:
        .. code-block:: python

            model = create_fastvlm("tiny", num_classes=10, input_shape=(32, 32, 3))
            backbone = create_fastvlm("base", include_top=False)
    """
    return FastVLM.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------
