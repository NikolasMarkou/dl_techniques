"""
A segmentation U-Net whose convolutional blocks carry transformer-like context.

The architecture starts from an observation about why transformer U-Nets win on
medical segmentation. Their advantage is usually attributed to attention itself,
but two more prosaic properties travel with it: every position sees the whole
image, and features are exchanged *across* scales rather than only within a
matched encoder-decoder pair. ACC-UNet's thesis is that both properties can be
obtained with pooling and 1x1 convolutions alone, and therefore without paying
attention's `O(N^2)` cost in the number of pixels `N` -- which at segmentation
resolutions is the dominant term.

Long-range context comes from hierarchical neighborhood aggregation. For a block
with `k` levels, the feature map is average- and max-pooled at strides
`2, 4, ..., 2^(k-1)`, each summary is resized back to full resolution, all of
them are concatenated with the untouched input to give `C * (2k - 1)` channels,
and a 1x1 convolution learns how to weigh them. Each pixel is thus compared not
to every other pixel but to the mean and the peak of its neighborhood at several
radii, which costs `O(N * k)`. The mean carries texture, the max carries the
salient activation, and their difference at a given radius is what tells a pixel
whether it sits inside a homogeneous region or on a boundary.

`k` shrinks with depth: `[3, 3, 3, 2, 1]` down the encoder and `[2, 2, 3, 3]` up
the decoder. This is deliberate -- a stride-4 pool at the bottleneck already spans
a large fraction of the image, so extra levels there summarize nearly the same
thing. The endpoint is worth stating plainly, because it is easy to misread as a
milder version of the same block: at `k = 1` the HANC layer pools nothing at all
and degenerates to a 1x1 projection of the input. The bottleneck level therefore
carries no hierarchical context whatsoever; its receptive field is what the two
stacked depthwise convolutions give it.

Each block itself is an inverted bottleneck -- 1x1 expansion by `inv_factor`,
depthwise 3x3, the HANC aggregation, a 1x1 projection to the requested width, and
squeeze-excitation. The residual shortcut exists only when input and output width
agree, which in practice means the second block of every level; the first block
of a level changes width and runs without one. Decoder level 3 uses
`inv_factor=4` where every other block uses 3.

Skip connections get two stages of treatment, both aimed at the semantic gap
between a shallow encoder feature and the deep decoder feature it is concatenated
with. ResPath first passes each of the four pre-bottleneck levels through a stack
of residual conv-SE blocks, `[4, 3, 2, 1]` of them from the shallowest level
down -- most refinement where the gap is widest. MLFC then resizes all four levels
to each level's resolution in turn, concatenates, compiles back down with 1x1
convolutions and adds the result residually, so a shallow feature acquires deep
semantics and a deep feature reacquires spatial detail. The bottleneck bypasses
both stages and goes straight into the decoder.

Two implementation choices are not what the parameter names suggest.
`mlfc_iterations` does not set `MLFCLayer.num_iterations`; it creates that many
*separate* single-iteration MLFC layers applied in sequence. The two forms are
not equivalent, because `MLFCLayer` applies its per-level squeeze-excitation once
after its internal loop -- stacking three layers therefore recalibrates channels
three times, not once. And `input_channels` is a constructor argument rather than
something inferred at build time because `HANCBlock` fixes its expansion width
from the channel count at construction; the model must know the input depth
before it ever sees a tensor.

Spatial dimensions must be divisible by 16. Four stride-2 pools and four
stride-2 transposed convolutions cannot round-trip an odd dimension, and the
decoder's concatenate is where that shows up, so the model raises at the first
call with a static shape rather than failing deeper with a shape mismatch.
Dynamic (`None`) dimensions are accepted and unchecked -- the divisibility
contract still holds at run time, it just cannot be verified at trace time.

The head applies its activation: sigmoid for `num_classes == 1`, softmax
otherwise. The model therefore emits probabilities, not logits, and must be
compiled with `from_logits=False`.

References:
    - Ibtehaz & Kihara, 2023. ACC-UNet: A Completely Convolutional UNet Model
      for the 2020s. MICCAI 2023.
      (https://arxiv.org/abs/2308.13680)
    - Ronneberger et al., 2015. U-Net: Convolutional Networks for Biomedical
      Image Segmentation. (https://arxiv.org/abs/1505.04597)
    - Ibtehaz & Rahman, 2020. MultiResUNet: Rethinking the U-Net Architecture
      for Multimodal Biomedical Image Segmentation. Neural Networks.
      (the origin of the ResPath skip refinement)
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear
      Bottlenecks. (https://arxiv.org/abs/1801.04381)
    - Hu et al., 2018. Squeeze-and-Excitation Networks.
      (https://arxiv.org/abs/1709.01507)
    - Zhao et al., 2017. Pyramid Scene Parsing Network.
      (https://arxiv.org/abs/1612.01105)
"""

import keras
from typing import Optional, Union, Tuple, Any, List, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.res_path import ResPath
from dl_techniques.layers.hanc_block import HANCBlock
from dl_techniques.layers.multi_level_feature_compilation import MLFCLayer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.accunet.model")
class AccUNet(keras.Model):
    """ACC-UNet: a completely convolutional UNet with transformer-like context.

    A U-Net whose convolution blocks are replaced by :class:`HANCBlock`, which
    obtains long-range context from hierarchical neighborhood aggregation
    (average- and max-pooling at strides ``2, 4, ..., 2^(k-1)``, resized back
    and fused by a 1x1 convolution) rather than from attention, at ``O(N * k)``
    instead of ``O(N^2)``. Five encoder levels of two blocks each descend
    through four stride-2 max pools; four decoder levels ascend through
    stride-2 transposed convolutions. The four pre-bottleneck skips are
    refined by :class:`ResPath` and then jointly compiled by a stack of
    :class:`MLFCLayer`; the bottleneck bypasses both and feeds the decoder
    directly. ``k`` shrinks with depth and reaches 1 at the bottleneck, where
    the HANC layer pools nothing and degenerates to a 1x1 projection.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │   Input [B, H, W, input_channels]    │
        │   H, W must be divisible by 16       │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────┐                       ┌──────────────────────┐
        │ Enc L0: 2×HANC(F, 3) │──skip 0──┐            │ Dec L3: 2×HANC(F, 3) │
        └──────────┬───────────┘          │            └──────────▲───────────┘
              MaxPool /2                  │                  ConvT /2
        ┌──────────▼───────────┐          │            ┌──────────┴───────────┐
        │ Enc L1: 2×HANC(2F,3) │──skip 1──┤            │ Dec L2: 2×HANC(2F,3) │
        └──────────┬───────────┘          │            └──────────▲───────────┘
              MaxPool /2                  │                  ConvT /2
        ┌──────────▼───────────┐          │            ┌──────────┴───────────┐
        │ Enc L2: 2×HANC(4F,3) │──skip 2──┤            │ Dec L1: 2×HANC(4F,2) │
        └──────────┬───────────┘          │            └──────────▲───────────┘
              MaxPool /2                  │                  ConvT /2
        ┌──────────▼───────────┐          │            ┌──────────┴───────────┐
        │ Enc L3: 2×HANC(8F,2) │──skip 3──┤            │ Dec L0: 2×HANC(8F,2) │
        └──────────┬───────────┘          │            └──────────▲───────────┘
              MaxPool /2                  │                  ConvT /2
        ┌──────────▼───────────┐          ▼                       │
        │ Enc L4 (bottleneck)  │   ┌───────────────┐              │
        │   2×HANC(16F, k=1)   │   │ ResPath ×4    │              │
        │   k=1 → NO pooling,  │   │ MLFC ×N       │──────────────┘
        │   a 1×1 projection   │   └───────────────┘   (concat with
        └──────────┬───────────┘    skip refinement     upsampled x)
                   │
                   └──────────────► decoder (bypasses ResPath/MLFC)
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Conv 1×1 → num_classes              │
        │  → sigmoid (num_classes == 1)        │
        │  → softmax (num_classes  > 1)        │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, H, W, num_classes]       │
        │  PROBABILITIES, not logits           │
        │  → compile with from_logits=False    │
        └──────────────────────────────────────┘

    **Per-level configuration:**

    .. code-block:: text

        encoder   L0     L1     L2     L3     L4 (bottleneck)
        filters   F      2F     4F     8F     16F
        k         3      3      3      2      1
        inv       3      3      3      3      3
        respath   4      3      2      1      -- (bypasses skips)

        decoder   L0     L1     L2     L3
        filters   8F     4F     2F     F
        k         2      2      3      3
        inv       3      3      3      4

        F = base_filters (default 32)

    :param input_channels: Number of input channels (e.g. 3 for RGB, 1 for
        grayscale). Must be positive. This is a CONSTRUCTOR argument rather
        than something inferred at build time because :class:`HANCBlock` fixes
        its expansion width from the channel count at construction.
    :type input_channels: int
    :param num_classes: Number of output classes for segmentation. Must be
        positive. Also selects the head activation: 1 gives sigmoid, more than
        1 gives softmax.
    :type num_classes: int
    :param base_filters: Base filter count. The five encoder levels use
        ``[base_filters, ×2, ×4, ×8, ×16]``. Must be positive. Defaults to 32.
    :type base_filters: int
    :param mlfc_iterations: How many MLFC layers to stack. NOTE this does NOT
        set ``MLFCLayer.num_iterations``; it creates that many SEPARATE
        single-iteration layers applied in sequence, which is not equivalent
        because each layer applies its squeeze-excitation once. Must be
        positive. Defaults to 3.
    :type mlfc_iterations: int
    :param kernel_initializer: Initializer for every convolution kernel.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every convolution
        kernel. Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``input_channels``, ``num_classes``,
        ``base_filters`` or ``mlfc_iterations`` is not positive. Also raised at
        the first call when a statically-known height or width is not divisible
        by 16.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, input_channels)``.
        Height and width must be divisible by 16; ``None`` dims are accepted
        and left unchecked.

    Output shape:
        4D tensor with shape ``(batch_size, height, width, num_classes)``,
        already passed through sigmoid or softmax.

    :ivar encoder_blocks: Per encoder level, the two ``HANCBlock`` instances.
    :vartype encoder_blocks: List[List[HANCBlock]]
    :ivar pooling_layers: The four stride-2 max pools between encoder levels.
    :vartype pooling_layers: List[keras.layers.Layer]
    :ivar decoder_upsamples: The four stride-2 transposed convolutions.
    :vartype decoder_upsamples: List[keras.layers.Layer]
    :ivar decoder_blocks: Per decoder level, the two ``HANCBlock`` instances.
    :vartype decoder_blocks: List[List[HANCBlock]]
    :ivar res_paths: The four ``ResPath`` layers refining the skips.
    :vartype res_paths: List[ResPath]
    :ivar mlfc_layers: The stacked ``MLFCLayer`` instances.
    :vartype mlfc_layers: List[MLFCLayer]
    :ivar output_conv: Final 1x1 convolution to ``num_classes``.
    :vartype output_conv: keras.layers.Conv2D
    :ivar output_activation: Final sigmoid or softmax.
    :vartype output_activation: keras.layers.Layer

    Example:
        .. code-block:: python

            # Binary segmentation
            model = AccUNet(input_channels=3, num_classes=1)

            # Multi-class segmentation
            model = AccUNet(input_channels=1, num_classes=5)

            # Custom configuration
            model = AccUNet(
                input_channels=3,
                num_classes=2,
                base_filters=64,
                mlfc_iterations=4,
                kernel_regularizer=keras.regularizers.L2(1e-4)
            )

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

    Note:
        The head applies its own activation, so the model emits PROBABILITIES,
        not logits: compile with ``from_logits=False``.

        Following modern Keras 3 patterns, all sub-layers are created in
        ``__init__`` without helper methods, which keeps serialization and
        build handling correct.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        base_filters: int = 32,
        mlfc_iterations: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the AccUNet model instance.

        :param input_channels: Number of input channels.
        :type input_channels: int
        :param num_classes: Number of output segmentation classes.
        :type num_classes: int
        :param base_filters: Base filter count for the five levels.
        :type base_filters: int
        :param mlfc_iterations: Number of stacked single-iteration MLFC layers.
        :type mlfc_iterations: int
        :param kernel_initializer: Initializer for convolution kernels.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional regularizer for convolution kernels.
        :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Additional keyword arguments for ``keras.Model``.
        :raises ValueError: If any of the four scalar arguments is not positive.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if base_filters <= 0:
            raise ValueError(f"base_filters must be positive, got {base_filters}")
        if mlfc_iterations <= 0:
            raise ValueError(f"mlfc_iterations must be positive, got {mlfc_iterations}")

        # Store ALL configuration parameters for serialization
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.mlfc_iterations = mlfc_iterations
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Calculate filter sizes for each level
        self.filter_sizes = [
            base_filters,        # Level 0: 32
            base_filters * 2,    # Level 1: 64
            base_filters * 4,    # Level 2: 128
            base_filters * 8,    # Level 3: 256
            base_filters * 16    # Level 4 (bottleneck): 512
        ]

        # CREATE all sub-layers in __init__ following Modern Keras 3 pattern
        # No helper methods - all layers created here directly

        # === ENCODER BLOCKS ===
        self.encoder_blocks: List[List[HANCBlock]] = []

        for level in range(5):  # 5 encoder levels
            if level == 0:
                # First level: input_channels -> base_filters
                input_ch = input_channels
                output_ch = self.filter_sizes[0]
                k = 3
            else:
                # Other levels: prev_filters -> curr_filters
                input_ch = self.filter_sizes[level - 1]
                output_ch = self.filter_sizes[level]
                # Determine k based on level (as per paper)
                if level <= 2:
                    k = 3
                elif level == 3:
                    k = 2
                else:  # level 4 (bottleneck)
                    k = 1

            # Create 2 HANC blocks per level
            block1 = HANCBlock(
                filters=output_ch,
                input_channels=input_ch,  # FIX: First block always takes input_ch for the level
                k=k,
                inv_factor=3,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'encoder_l{level}_block1'
            )

            block2 = HANCBlock(
                filters=output_ch,
                input_channels=output_ch,  # Second block always has same in/out
                k=k,
                inv_factor=3,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'encoder_l{level}_block2'
            )

            self.encoder_blocks.append([block1, block2])

        # === POOLING LAYERS ===
        self.pooling_layers: List[keras.layers.Layer] = []
        for level in range(4):  # Only 4 pooling layers (not needed for bottleneck)
            pool = keras.layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                name=f'pool_{level}'
            )
            self.pooling_layers.append(pool)

        # === DECODER BLOCKS ===
        self.decoder_upsamples: List[keras.layers.Layer] = []
        self.decoder_blocks: List[List[HANCBlock]] = []

        for level in range(4):  # 4 decoder levels
            # Upsample layer
            # curr_filters = self.filter_sizes[4 - level]  # 512, 256, 128, 64
            next_filters = self.filter_sizes[3 - level]  # 256, 128, 64, 32

            upsample = keras.layers.Conv2DTranspose(
                filters=next_filters,
                kernel_size=2,
                strides=2,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'upsample_{level}'
            )
            self.decoder_upsamples.append(upsample)

            # Determine k and inv_factor based on level
            if level <= 1:
                k = 2
                inv_factor = 3
            elif level == 2:
                k = 3
                inv_factor = 3
            else:  # level 3
                k = 3
                inv_factor = 4  # Special case as per paper

            # Decoder blocks (2 per level)
            # Input: next_filters (from upsample) + next_filters (from skip) = 2*next_filters
            block1 = HANCBlock(
                filters=next_filters,
                input_channels=2 * next_filters,  # Concatenated channels
                k=k,
                inv_factor=inv_factor,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'decoder_l{level}_block1'
            )

            block2 = HANCBlock(
                filters=next_filters,
                input_channels=next_filters,  # Output of block1
                k=k,
                inv_factor=inv_factor,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'decoder_l{level}_block2'
            )

            self.decoder_blocks.append([block1, block2])

        # === SKIP CONNECTION PROCESSING ===
        # ResPath layers for each encoder level (except bottleneck)
        self.res_paths: List[ResPath] = []
        res_path_blocks = [4, 3, 2, 1]  # Number of blocks for each level

        for level in range(4):
            res_path = ResPath(
                channels=self.filter_sizes[level],
                num_blocks=res_path_blocks[level],
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'res_path_{level}'
            )
            self.res_paths.append(res_path)

        # MLFC layers for multi-level feature compilation
        self.mlfc_layers: List[MLFCLayer] = []
        channels_list = self.filter_sizes[:4]  # First 4 levels only

        for i in range(self.mlfc_iterations):
            mlfc = MLFCLayer(
                channels_list=channels_list,
                num_iterations=1,  # Each MLFC layer does 1 iteration
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'mlfc_{i}'
            )
            self.mlfc_layers.append(mlfc)

        # === OUTPUT LAYER ===
        # Output convolution and activation
        self.output_conv = keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='output_conv'
        )

        if self.num_classes == 1:
            # Binary segmentation
            self.output_activation = keras.layers.Activation("sigmoid", name='output_activation')
        else:
            # Multi-class segmentation
            self.output_activation = keras.layers.Softmax(name='output_activation')

        # === CONCATENATION LAYERS ===
        # Create concatenation layers for decoder skip connections
        self.concat_layers: List[keras.layers.Layer] = []
        for level in range(4):
            concat = keras.layers.Concatenate(axis=-1, name=f'concat_{level}')
            self.concat_layers.append(concat)

    @staticmethod
    def _validate_spatial_dims(shape: Tuple[Optional[int], ...]) -> None:
        """Reject a statically-known height or width that is not a multiple of 16.

        DECISION plan_2026-05-10_bdb2c84d/D-001: AccUNet requires H, W divisible
        by 16. ``padding='same'`` on MaxPooling2D was tried first but
        ``Conv2DTranspose(strides=2, padding='same')`` always emits ``2 * H_in``
        which cannot recover odd dims, so the decoder Concatenate still
        mismatched. Failing loudly is the honest contract; the trainer must
        resize inputs accordingly.

        :param shape: Input shape; positions 1 and 2 are height and width.
            ``None`` dims are accepted and left unchecked (dynamic shape).
        :type shape: Tuple[Optional[int], ...]
        :raises ValueError: If a known height or width is not divisible by 16.
        """
        if len(shape) < 3:
            return
        h, w = shape[1], shape[2]
        for name, dim in (("height", h), ("width", w)):
            if dim is not None and dim % 16 != 0:
                raise ValueError(
                    f"AccUNet requires input {name} divisible by 16 "
                    f"(4 stride-2 downsamples + matched stride-2 upsamples), "
                    f"got {name}={dim}. Resize inputs to a multiple of 16."
                )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the model.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, input_channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Segmentation map of shape
            ``(batch_size, height, width, num_classes)``, already activated
            (sigmoid or softmax), i.e. probabilities rather than logits.
        :rtype: keras.KerasTensor
        :raises ValueError: If a statically-known spatial dimension is not
            divisible by 16.
        """
        # Validate spatial dims at first call (skipped for dynamic shapes).
        self._validate_spatial_dims(tuple(inputs.shape))

        # === ENCODER FORWARD PASS ===
        encoder_features: List[keras.KerasTensor] = []
        x = inputs

        for level in range(5):
            # Apply encoder blocks
            for block in self.encoder_blocks[level]:
                x = block(x, training=training)

            # Store features for skip connections (except bottleneck)
            if level < 4:
                encoder_features.append(x)
                x = self.pooling_layers[level](x)
            else:
                # Bottleneck features (level 4)
                bottleneck_features = x

        # === SKIP CONNECTION PROCESSING ===
        # Apply ResPath to encoder features
        processed_features: List[keras.KerasTensor] = []
        for level, features in enumerate(encoder_features):
            processed = self.res_paths[level](features, training=training)
            processed_features.append(processed)

        # Apply MLFC layers iteratively
        for mlfc_layer in self.mlfc_layers:
            processed_features = mlfc_layer(processed_features, training=training)

        # === DECODER FORWARD PASS ===
        x = bottleneck_features

        for level in range(4):
            # Upsample
            x = self.decoder_upsamples[level](x)

            # Concatenate with processed skip connection features
            skip_features = processed_features[3 - level]  # Reverse order
            x = self.concat_layers[level]([x, skip_features])

            # Apply decoder blocks
            for block in self.decoder_blocks[level]:
                x = block(x, training=training)

        # === OUTPUT LAYER ===
        x = self.output_conv(x)
        x = self.output_activation(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Dictionary containing all model configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'base_filters': self.base_filters,
            'mlfc_iterations': self.mlfc_iterations,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AccUNet':
        """Create a model instance from its configuration.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: AccUNet model instance.
        :rtype: AccUNet
        """
        return cls(**config)

# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_acc_unet(
    input_channels: int,
    num_classes: int,
    base_filters: int = 32,
    mlfc_iterations: int = 3,
    input_shape: Optional[Tuple[int, int]] = None,
    **kwargs: Any
) -> keras.Model:
    """Create an ACC-UNet wrapped in the Keras Functional API.

    :param input_channels: Number of input channels.
    :type input_channels: int
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param base_filters: Base filter count. Defaults to 32.
    :type base_filters: int
    :param mlfc_iterations: Number of stacked MLFC layers. Defaults to 3.
    :type mlfc_iterations: int
    :param input_shape: Input spatial dimensions ``(height, width)``, each a
        multiple of 16. ``None`` builds over a dynamic ``(None, None)`` shape,
        in which case the divisibility contract still holds at run time but
        cannot be checked at trace time.
    :type input_shape: Optional[Tuple[int, int]]
    :param kwargs: Additional arguments forwarded to :class:`AccUNet`.
    :return: Functional ``keras.Model`` named ``ACC_UNet``.
    :rtype: keras.Model

    Example:
        .. code-block:: python

            # Binary segmentation with fixed input size
            model = create_acc_unet(
                input_channels=3,
                num_classes=1,
                input_shape=(256, 256)
            )

            # Multi-class with dynamic input size
            model = create_acc_unet(
                input_channels=1,
                num_classes=5,
                base_filters=64
            )
    """
    if input_shape is not None:
        input_spec = keras.Input(shape=input_shape + (input_channels,))
    else:
        input_spec = keras.Input(shape=(None, None, input_channels))

    # Create the model instance
    acc_unet = AccUNet(
        input_channels=input_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        mlfc_iterations=mlfc_iterations,
        **kwargs
    )

    # Build the model by calling it
    outputs = acc_unet(input_spec)

    # Create functional model
    model = keras.Model(inputs=input_spec, outputs=outputs, name='ACC_UNet')

    return model


def create_acc_unet_binary(
    input_channels: int,
    input_shape: Optional[Tuple[int, int]] = None,
    base_filters: int = 32,
    mlfc_iterations: int = 3,
    **kwargs: Any
) -> keras.Model:
    """Create an ACC-UNet for binary segmentation.

    :param input_channels: Number of input channels.
    :type input_channels: int
    :param input_shape: Input spatial dimensions ``(height, width)``. ``None``
        uses a dynamic shape.
    :type input_shape: Optional[Tuple[int, int]]
    :param base_filters: Base filter count. Defaults to 32.
    :type base_filters: int
    :param mlfc_iterations: Number of stacked MLFC layers. Defaults to 3.
    :type mlfc_iterations: int
    :param kwargs: Additional arguments forwarded to :class:`AccUNet`.
    :return: Functional model with a single sigmoid-activated output channel.
    :rtype: keras.Model

    Example:
        .. code-block:: python

            # Grayscale medical image segmentation
            model = create_acc_unet_binary(
                input_channels=1,
                input_shape=(512, 512),
                base_filters=32
            )

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['binary_accuracy', 'dice_coefficient']
            )
    """
    return create_acc_unet(
        input_channels=input_channels,
        num_classes=1,  # Binary segmentation
        base_filters=base_filters,
        mlfc_iterations=mlfc_iterations,
        input_shape=input_shape,
        **kwargs
    )


def create_acc_unet_multiclass(
    input_channels: int,
    num_classes: int,
    input_shape: Optional[Tuple[int, int]] = None,
    base_filters: int = 32,
    mlfc_iterations: int = 3,
    **kwargs: Any
) -> keras.Model:
    """Create an ACC-UNet for multi-class segmentation.

    :param input_channels: Number of input channels.
    :type input_channels: int
    :param num_classes: Number of output classes; must be greater than 1.
    :type num_classes: int
    :param input_shape: Input spatial dimensions ``(height, width)``. ``None``
        uses a dynamic shape.
    :type input_shape: Optional[Tuple[int, int]]
    :param base_filters: Base filter count. Defaults to 32.
    :type base_filters: int
    :param mlfc_iterations: Number of stacked MLFC layers. Defaults to 3.
    :type mlfc_iterations: int
    :param kwargs: Additional arguments forwarded to :class:`AccUNet`.
    :return: Functional model with a softmax-activated output.
    :rtype: keras.Model
    :raises ValueError: If ``num_classes`` is not greater than 1.

    Example:
        .. code-block:: python

            # RGB image with 5 semantic classes
            model = create_acc_unet_multiclass(
                input_channels=3,
                num_classes=5,
                input_shape=(256, 256),
                base_filters=64
            )

            model.compile(
                optimizer='adamw',
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy', 'mean_iou']
            )
    """
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1 for multi-class segmentation, got {num_classes}")

    return create_acc_unet(
        input_channels=input_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        mlfc_iterations=mlfc_iterations,
        input_shape=input_shape,
        **kwargs
    )

# ---------------------------------------------------------------------