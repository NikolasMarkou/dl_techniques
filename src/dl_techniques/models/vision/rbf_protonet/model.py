"""``RBFProtoNet``, a small CIFAR-style CNN backbone for an RBF prototype-classification head.

A CNN reduces a 32x32 RGB image to a flat feature vector; a later step attaches
an ``RBFLayer`` (``dl_techniques.layers.mixtures.radial_basis_function``) as a
distance-based class-prototype classification head on top of that vector, so
each class gets one learned, inspectable prototype in feature space instead of
a distributed softmax-of-logits weight matrix. This module currently implements
only the CNN half of that composition: the backbone reduces an input image to a
``feature_dim``-wide pooled vector and returns it directly from ``call()``. The
RBF head is added in a later revision of this same class (see the class
docstring's Note); no RBF/mixture code is imported here.

The backbone follows the ``stem_type='cifar'`` precedent from
``dl_techniques.models.vision.resnet.model.ResNet``: a single 3x3 stride-1 stem
(no max-pool, since a 32x32 input has no resolution to spare) followed by a
handful of stride-2 downsampling stages that both widen channels and shrink the
spatial map (32x32 -> 16x16 -> 8x8 -> 4x4), ending in a global average pool.
Every conv+norm+activation triple is built via ``ConvBlock``
(``dl_techniques.layers.conv_blocks.conv_block``), which itself dispatches to
the ``create_normalization_layer``/``resolve_activation_layer`` factories,
rather than hand-rolling ``Conv2D`` + ``BatchNormalization`` + activation in
this file.

This package intentionally ships without a ``MODEL_VARIANTS`` table (see
``models/CLAUDE.md`` Section "When the shape does not apply" / the guide's
Section 5.6): there is exactly one baseline architecture here, and inventing
named size multipliers with no genuine use case would be exactly the
"forcing an ill-fitting template" anti-pattern the house shape warns against.

References:
    - He et al., 2016. Deep Residual Learning for Image Recognition
      (the ``stem_type='cifar'`` precedent this backbone's stem mirrors).
      (https://arxiv.org/abs/1512.03385)
    - Snell, Swersky & Zemel, 2017. Prototypical Networks for Few-shot Learning
      (the "one prototype vector per class in feature space" framing this
      backbone's pooled output feeds into once the RBF head is attached).
      (https://arxiv.org/abs/1703.05175)
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.conv_blocks.conv_block import ConvBlock
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.rbf_protonet.model")
class RBFProtoNet(keras.Model):
    """
    CIFAR-style CNN backbone that pools an image to a flat feature vector.

    A single 3x3 stride-1 stem (``stem_type='cifar'`` precedent, no max-pool)
    feeds a configurable number of stride-2 downsampling stages, each built
    from one ``ConvBlock`` (conv + factory-selected normalization + factory-
    selected activation). A global average pool at the end collapses the final
    stage's feature map to a flat vector of width ``feature_dim``; a ``Dense``
    projection is inserted only when the last stage's channel width does not
    already equal ``feature_dim`` (the default configuration needs none, since
    the last stage is already ``feature_dim`` wide).

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │       Input [B, 32, 32, 3]            │
        └───────────────┬────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Stem: ConvBlock 3x3 /1               │  32x32 -> 32x32
        │  (no pooling -- input has no spare    │
        │   resolution to give up)              │
        └───────────────┬────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 1: ConvBlock 3x3 /2            │  32x32 -> 16x16
        └───────────────┬────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 2: ConvBlock 3x3 /2            │  16x16 -> 8x8
        └───────────────┬────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Stage 3: ConvBlock 3x3 /2            │  8x8   -> 4x4
        └───────────────┬────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  GlobalAveragePooling2D               │  4x4xF -> F
        │  (+ Dense projection to feature_dim,  │
        │   only if the last stage width != F)  │
        └───────────────┬────────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output: [B, feature_dim]              │
        └──────────────────────────────────────┘

    Note:
        This class currently implements only the CNN backbone described
        above; ``call()`` returns the pooled ``feature_dim``-wide vector
        directly. A prototype-classification head built from ``RBFLayer``
        (``output_mode='normalized'``) is wired onto this same class in a
        later revision, at which point ``call()`` is extended (not replaced)
        to run the backbone and then the head. No RBF/mixture import exists
        in this file by design -- it is added alongside the head.

    :param input_shape: Input shape ``(height, width, channels)`` excluding
        the batch dimension. Defaults to ``(32, 32, 3)`` (CIFAR).
    :type input_shape: Tuple[int, int, int]
    :param stem_filters: Channel width of the single 3x3 stride-1 stem
        convolution. Defaults to ``32``.
    :type stem_filters: int
    :param filters_per_stage: Channel width of each stride-2 downsampling
        stage, in order. Defaults to ``[64, 128, 128]`` (three stages,
        32x32 -> 16x16 -> 8x8 -> 4x4 on a 32x32 input).
    :type filters_per_stage: Optional[List[int]]
    :param feature_dim: Width of the pooled feature vector this backbone
        returns. Defaults to ``128``.
    :type feature_dim: int
    :param normalization_type: Normalization layer identifier passed to
        every ``ConvBlock`` (see ``create_normalization_layer``). Defaults
        to ``'batch_norm'``.
    :type normalization_type: str
    :param normalization_kwargs: Optional kwargs forwarded to every
        ``ConvBlock``'s normalization factory call. ``None`` resolves to
        ``{}``.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param activation_type: Activation identifier passed to every
        ``ConvBlock`` (see ``resolve_activation_layer``). Defaults to
        ``'relu'``.
    :type activation_type: str
    :param kernel_regularizer: Optional regularizer applied to every
        convolution kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param dropout_rate: Dropout rate applied inside every downsampling
        stage's ``ConvBlock`` (``0.0`` disables it). Not applied in the stem.
        Defaults to ``0.0``.
    :type dropout_rate: float
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``input_shape`` is not 3D, if ``stem_filters`` or
        ``feature_dim`` is not positive, or if ``filters_per_stage`` is empty
        or contains a non-positive value.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        2D tensor ``(batch_size, feature_dim)``.

    Example:
        >>> backbone = RBFProtoNet(input_shape=(32, 32, 3))
        >>> features = backbone(keras.random.normal((4, 32, 32, 3)))
        >>> features.shape
        TensorShape([4, 128])
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (32, 32, 3),
            stem_filters: int = 32,
            filters_per_stage: Optional[List[int]] = None,
            feature_dim: int = 128,
            normalization_type: str = "batch_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_type: str = "relu",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")
        if stem_filters <= 0:
            raise ValueError(f"stem_filters must be positive, got {stem_filters}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        filters_per_stage = (
            list(filters_per_stage) if filters_per_stage is not None else [64, 128, 128]
        )
        if len(filters_per_stage) == 0:
            raise ValueError("filters_per_stage must be non-empty")
        if any(f <= 0 for f in filters_per_stage):
            raise ValueError(
                f"every entry of filters_per_stage must be positive, got {filters_per_stage}"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0,1], got {dropout_rate}")

        self.input_shape_config = tuple(input_shape)
        self.input_height, self.input_width, self.input_channels = input_shape
        self.stem_filters = stem_filters
        self.filters_per_stage = filters_per_stage
        # DECISION plan-2026-09-16-7dfede94/D-006: default feature_dim=128, not
        # 256 or 512. Step 1's standalone smoke test measured RBFLayer(units=100,
        # output_mode='normalized') gradients on centers/gamma at D in
        # {128, 256, 512}; 128 already fully passed, with the STRONGEST
        # gradients of the three candidates (larger D bought nothing here).
        # Do not "fix" this default upward assuming a wider feature vector
        # would help the downstream RBF head -- that assumption was tested and
        # refuted. See decisions.md D-006 (Anchor-Refs).
        self.feature_dim = feature_dim
        self.normalization_type = normalization_type
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}
        self.activation_type = activation_type
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate

        self._build_stem()

        self.stages: List[ConvBlock] = []
        for stage_idx in range(len(self.filters_per_stage)):
            self._build_stage(stage_idx)

        self._build_head()

        logger.info(
            f"Created RBFProtoNet backbone with {len(self.filters_per_stage)} "
            f"downsampling stages for input {self.input_shape_config}, "
            f"pooled feature_dim={self.feature_dim}"
        )

    def _build_stem(self) -> None:
        """Build the single 3x3 stride-1 CIFAR-style stem (no pooling)."""
        self.stem = ConvBlock(
            filters=self.stem_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            normalization_type=self.normalization_type,
            normalization_kwargs=dict(self.normalization_kwargs),
            activation_type=self.activation_type,
            kernel_regularizer=self.kernel_regularizer,
            name="stem",
        )

    def _build_stage(self, stage_idx: int) -> None:
        """Build one stride-2 downsampling stage.

        :param stage_idx: Index of the stage to build.
        """
        stage = ConvBlock(
            filters=self.filters_per_stage[stage_idx],
            kernel_size=3,
            strides=2,
            padding="same",
            normalization_type=self.normalization_type,
            normalization_kwargs=dict(self.normalization_kwargs),
            activation_type=self.activation_type,
            dropout_rate=self.dropout_rate,
            kernel_regularizer=self.kernel_regularizer,
            name=f"stage{stage_idx + 1}",
        )
        self.stages.append(stage)

    def _build_head(self) -> None:
        """Build the global average pool and, if needed, a projection Dense.

        A projection is created only when the last stage's channel width
        does not already equal ``feature_dim`` -- the default configuration
        (last stage width ``128`` == ``feature_dim`` ``128``) needs none.
        """
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")

        if self.filters_per_stage[-1] != self.feature_dim:
            self.feature_proj = keras.layers.Dense(
                units=self.feature_dim,
                kernel_initializer="he_normal",
                kernel_regularizer=self.kernel_regularizer,
                name="feature_proj",
            )
        else:
            self.feature_proj = None

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from `input_shape` by tracing `call`.

        :param input_shape: Shape (or nest of shapes) of the input to `call`.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the stem, downsampling stages, and pooling head.

        :param inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        :param training: Whether batch norm and dropout run in training mode.
        :return: Pooled feature tensor `(batch_size, feature_dim)`.
        """
        x = self.stem(inputs, training=training)

        for stage in self.stages:
            x = stage(x, training=training)

        pooled = self.gap(x)
        if self.feature_proj is not None:
            pooled = self.feature_proj(pooled)

        return pooled

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "input_shape": self.input_shape_config,
            "stem_filters": self.stem_filters,
            "filters_per_stage": self.filters_per_stage,
            "feature_dim": self.feature_dim,
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
            "activation_type": self.activation_type,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            "dropout_rate": self.dropout_rate,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RBFProtoNet":
        """Create a model from its `get_config()` output."""
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)


# ---------------------------------------------------------------------
