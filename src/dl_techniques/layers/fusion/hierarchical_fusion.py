"""Hierarchical fusion of a feature pyramid into one map at the finest resolution.

``HierarchicalScaleFusion`` takes feature maps ordered fine to coarse, each half the
spatial size of the previous one, and returns one map at the size of the first. It
never concatenates channels and never resizes all levels to full resolution. Each
level is projected at its own resolution, then a top-down cascade moves the running
map up one octave at a time with a sub-pixel convolution (3x3 conv to 4 * filters,
then depth-to-space) and merges it into the next level by a normalized weighted sum.
A sum of per-level convolutions equals one convolution over the concatenated levels,
and a 3x3 conv followed by depth-to-space contains nearest-upsample followed by a
3x3 conv, so the layer spans concat-then-conv while each level is computed at its
native size. Inputs are channels-last. Level ``i + 1`` must have spatial size
``ceil(H_i / 2)``, which is what stride-2 layers with ``padding="same"`` produce.

The projections and refine blocks are ``ConvBlock`` instances, so the normalization
comes from the norms factory and its epsilon is the factory default (``1e-6``), not
the Keras batch-norm default (``1e-3``). The depth-to-space step is
``PixelShuffle2D``. Both are shared layers, not local re-implementations.

References:
    - Lin et al., 2017. Feature Pyramid Networks for Object Detection. (https://arxiv.org/abs/1612.03144)
    - Shi et al., 2016. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. (https://arxiv.org/abs/1609.05158)
    - Tan et al., 2020. EfficientDet: Scalable and Efficient Object Detection. (https://arxiv.org/abs/1911.09070)
"""

import keras
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import initializer_spec
from dl_techniques.layers.conv_blocks.conv_block import ConvBlock
from dl_techniques.layers.pooling.pixel_unshuffle import PixelShuffle2D
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

def check_pyramid_shapes(
        shapes: Sequence[Sequence[Optional[int]]],
        num_levels: int,
) -> None:
    """Validate a fine-to-coarse list of channels-last shapes.

    The one place the pyramid contract lives: ``build``, ``call`` and
    ``compute_output_shape`` all route through it. Unknown (``None``) spatial
    sizes pass, because a static check cannot judge them.

    :param shapes: One ``(batch, height, width, channels)`` shape per level.
    :param num_levels: Number of levels the layer was configured for.
    :raises ValueError: If the level count differs from ``num_levels``, a shape is not
        rank 4, or a known spatial size breaks ``H_{i+1} = ceil(H_i / 2)``.
    """
    if len(shapes) != num_levels:
        raise ValueError(
            f"Expected {num_levels} feature maps, got {len(shapes)}."
        )
    for i, shape in enumerate(shapes):
        if len(shape) != 4:
            raise ValueError(
                f"Level {i} must be a rank-4 channels-last shape, got {tuple(shape)}."
            )
    for i in range(num_levels - 1):
        for axis in (1, 2):
            fine = shapes[i][axis]
            coarse = shapes[i + 1][axis]
            if fine is not None and coarse is not None and coarse != -(-fine // 2):
                raise ValueError(
                    f"Level {i + 1} axis {axis} is {coarse}, expected "
                    f"ceil({fine} / 2) = {-(-fine // 2)}."
                )

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.fusion.hierarchical_fusion")
class HierarchicalScaleFusion(keras.layers.Layer):
    """Fuse a fine-to-coarse feature pyramid top-down into one finest-level map.

    The coarsest level is projected to ``filters`` channels. At every finer level the
    running map is upsampled x2, cropped to that level's size, merged with the
    level's own projection, and refined by a residual 3x3 block. Coarse levels pass
    through more refinement steps than fine ones, which mirrors the encoder depth
    that produced them.

    Architecture:

    .. code-block:: text

        x[L-1]  [B, H_{L-1}, W_{L-1}, N_{L-1}]
                 │
                 ▼
        ┌─────────────────────┐
        │ lateral_{L-1}       │  1x1 conv + norm
        └─────────────────────┘
                 │  [B, H_{L-1}, W_{L-1}, filters]
                 ▼
        ┌─────────────────────┐
        │ fuse step L-2       │◄── x[L-2]
        └─────────────────────┘
                 │  [B, H_{L-2}, W_{L-2}, filters]
                 ▼
                ...
                 │
                 ▼
        ┌─────────────────────┐
        │ fuse step 0         │◄── x[0]
        └─────────────────────┘
                 │
                 ▼
        output [B, H_0, W_0, filters]

    Fuse step i:

    .. code-block:: text

        y  [B, H_{i+1}, W_{i+1}, filters]    x[i]  [B, H_i, W_i, N_i]
                 │                                │
                 ▼                                ▼
        ┌─────────────────────┐        ┌─────────────────────┐
        │ up_conv_i           │        │ lateral_i           │
        │ 3x3, 4 * filters    │        │ 1x1 conv + norm     │
        └─────────────────────┘        └─────────────────────┘
                 │                                │
                 ▼                                │
          pixel_shuffle_i (x2)                    │
          crop to [H_i, W_i]                      │
                 │                                │
                 └──────► weighted sum ◄──────────┘
                               │
                               ├────────────┐
                               ▼            │
                    ┌─────────────────────┐ │
                    │ refine_i            │ │
                    │ 3x3 conv+norm+act   │ │
                    └─────────────────────┘ │
                               │            │
                               ▼            │
                               + ◄──────────┘
                               │
                               ▼
                    y  [B, H_i, W_i, filters]

    With ``upsample="bilinear"``, up_conv_i and pixel_shuffle_i become a bilinear x2.
    With ``weighted_fusion=False``, the weighted sum is a plain sum.

    :param filters: Channels of every projected level and of the output.
    :type filters: int
    :param num_levels: Number of pyramid levels, at least 2. Fixes how many sub-layers
        exist, so they are all created in ``__init__``.
    :type num_levels: int
    :param upsample: ``"subpixel"`` (learned) or ``"bilinear"`` (fixed).
    :type upsample: str
    :param weighted_fusion: Merge with ``(w_a * a + w_b * b) / (w_a + w_b + eps)``,
        ``w = relu(learned scalar)``, one pair per fuse step.
    :type weighted_fusion: bool
    :param normalization_type: Key of the norms factory used after every conv.
    :type normalization_type: str
    :param activation_type: Activation in the refine blocks, an activations-factory
        key or a Keras activation name.
    :type activation_type: str
    :param epsilon: Stabilizer in the weighted sum.
    :type epsilon: float
    :param kernel_initializer: Initializer for every conv kernel. Stored as a spec, so
        each conv draws independently.
    :type kernel_initializer: str or dict or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for every conv kernel.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    :type kwargs: Any

    :raises ValueError: If ``filters`` is not positive, ``num_levels`` is below 2,
        ``epsilon`` is not positive, or ``upsample`` is not ``"subpixel"`` or
        ``"bilinear"``.

    Input shape:
        List of ``num_levels`` tensors ``[B, H_i, W_i, N_i]``, fine to coarse, with
        ``H_{i+1} = ceil(H_i / 2)`` and ``W_{i+1} = ceil(W_i / 2)``.

    Output shape:
        ``[B, H_0, W_0, filters]``.

    Example:
        >>> f0 = keras.Input((128, 128, 32))
        >>> f1 = keras.Input((64, 64, 64))
        >>> f2 = keras.Input((32, 32, 128))
        >>> y = HierarchicalScaleFusion(64, num_levels=3)([f0, f1, f2])
    """

    def __init__(
            self,
            filters: int,
            num_levels: int,
            upsample: str = "subpixel",
            weighted_fusion: bool = True,
            normalization_type: str = "batch_norm",
            activation_type: str = "gelu",
            epsilon: float = 1e-4,
            kernel_initializer: Any = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if num_levels < 2:
            raise ValueError(f"num_levels must be at least 2, got {num_levels}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if upsample not in ("subpixel", "bilinear"):
            raise ValueError(
                f"upsample must be 'subpixel' or 'bilinear', got {upsample!r}"
            )

        self.filters = filters
        self.num_levels = num_levels
        self.upsample = upsample
        self.weighted_fusion = weighted_fusion
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.epsilon = epsilon
        self.kernel_initializer = initializer_spec(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.laterals: List[ConvBlock] = [
            ConvBlock(
                filters,
                kernel_size=1,
                normalization_type=normalization_type,
                activation_type="linear",
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"lateral_{i}",
            )
            for i in range(num_levels)
        ]

        # Created for both upsample modes, built only for the one `call` runs.
        self.up_convs: List[keras.layers.Conv2D] = []
        self.pixel_shuffles: List[PixelShuffle2D] = []
        self.up_bilinears: List[keras.layers.UpSampling2D] = []
        self.refines: List[ConvBlock] = []
        for i in range(num_levels - 1):
            self.up_convs.append(keras.layers.Conv2D(
                4 * filters,
                3,
                padding="same",
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"up_conv_{i}",
            ))
            self.pixel_shuffles.append(PixelShuffle2D(2, name=f"pixel_shuffle_{i}"))
            self.up_bilinears.append(keras.layers.UpSampling2D(
                2, interpolation="bilinear", name=f"up_bilinear_{i}"
            ))
            self.refines.append(ConvBlock(
                filters,
                kernel_size=3,
                normalization_type=normalization_type,
                activation_type=activation_type,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"refine_{i}",
            ))

        self.fusion_logits = None

    def build(self, input_shape: Sequence[Tuple[Optional[int], ...]]) -> None:
        """Build the sub-layers that ``call`` runs, and the fusion weights.

        :param input_shape: List of ``num_levels`` shapes, fine to coarse.
        :raises ValueError: If the level count is wrong or a known spatial size
            breaks ``H_{i+1} = ceil(H_i / 2)``.
        """
        shapes = [tuple(s) for s in input_shape]
        check_pyramid_shapes(shapes, self.num_levels)

        projected = []
        for lateral, shape in zip(self.laterals, shapes):
            lateral.build(shape)
            projected.append(lateral.compute_output_shape(shape))

        for i in range(self.num_levels - 1):
            if self.upsample == "subpixel":
                self.up_convs[i].build(projected[i + 1])
                self.pixel_shuffles[i].build(
                    self.up_convs[i].compute_output_shape(projected[i + 1])
                )
            else:
                self.up_bilinears[i].build(projected[i + 1])
            self.refines[i].build(projected[i])

        if self.weighted_fusion:
            self.fusion_logits = self.add_weight(
                name="fusion_logits",
                shape=(self.num_levels - 1, 2),
                initializer="ones",
                trainable=True,
            )
        super().build(input_shape)

    def _upsample(self, i: int, y: Any, height: Any, width: Any) -> Any:
        if self.upsample == "subpixel":
            y = self.pixel_shuffles[i](self.up_convs[i](y))
        else:
            y = self.up_bilinears[i](y)
        # Odd H_i gives 2 * ceil(H_i / 2) = H_i + 1; the crop removes that extra row.
        return y[:, :height, :width, :]

    def _merge(self, i: int, lateral: Any, upsampled: Any) -> Any:
        if not self.weighted_fusion:
            return lateral + upsampled
        w = keras.ops.relu(self.fusion_logits[i])
        return (w[0] * lateral + w[1] * upsampled) / (w[0] + w[1] + self.epsilon)

    def call(self, inputs: Sequence[Any], training: Optional[bool] = None) -> Any:
        """Fuse the pyramid top-down.

        :param inputs: List of ``num_levels`` feature maps, fine to coarse.
        :param training: Forwarded to the normalization layers.
        :return: Fused map ``[B, H_0, W_0, filters]``.
        :raises ValueError: If the level count or a known spatial size is wrong.
        """
        check_pyramid_shapes([tuple(x.shape) for x in inputs], self.num_levels)

        y = self.laterals[-1](inputs[-1], training=training)
        for i in range(self.num_levels - 2, -1, -1):
            lateral = self.laterals[i](inputs[i], training=training)
            shape = keras.ops.shape(lateral)
            y = self._merge(i, lateral, self._upsample(i, y, shape[1], shape[2]))
            y = y + self.refines[i](y, training=training)
        return y

    def compute_output_shape(
            self, input_shape: Sequence[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        shapes = [tuple(s) for s in input_shape]
        check_pyramid_shapes(shapes, self.num_levels)
        return shapes[0][:-1] + (self.filters,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "num_levels": self.num_levels,
                "upsample": self.upsample,
                "weighted_fusion": self.weighted_fusion,
                "normalization_type": self.normalization_type,
                "activation_type": self.activation_type,
                "epsilon": self.epsilon,
                "kernel_initializer": self.kernel_initializer,
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

# ---------------------------------------------------------------------
