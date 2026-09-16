"""``RBFProtoNet``, a small CIFAR-style CNN backbone with an RBF prototype-classification head.

A CNN reduces a 32x32 RGB image to a flat feature vector; an ``RBFLayer``
(``dl_techniques.layers.mixtures.radial_basis_function``, built through
``create_mixture_layer('rbf', ...)``) is then applied as a distance-based
class-prototype classification head on top of that vector, so each class gets
one learned, inspectable prototype in feature space instead of a distributed
softmax-of-logits weight matrix. ``call()`` runs the backbone, then the head,
and returns a per-class probability vector (``output_mode='normalized'``, see
D-002) that sums to 1.0 along the last axis.

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

This module also hosts ``CliffordRBFProtoNet``, a second backbone/head pairing
that swaps the CNN backbone above for an isotropic Clifford-algebra backbone
(``CliffordNetBlock``, ``dl_techniques.layers.geometric.clifford_block``) while
reusing the same RBF prototype-classification head design. Both classes
register under the same module path (``dl_techniques.models.rbf_protonet.model``)
since they are defined in this one file -- see decisions.md D-001.

References:
    - He et al., 2016. Deep Residual Learning for Image Recognition
      (the ``stem_type='cifar'`` precedent this backbone's stem mirrors).
      (https://arxiv.org/abs/1512.03385)
    - Snell, Swersky & Zemel, 2017. Prototypical Networks for Few-shot Learning
      (the "one prototype vector per class in feature space" framing this
      backbone's pooled output feeds into once the RBF head is attached).
      (https://arxiv.org/abs/1703.05175)
    - Ji, 2026. (arXiv:2601.06793) -- the Clifford-algebra backbone
      ``CliffordRBFProtoNet`` reuses (``CliffordNetBlock``).
    - Brandstetter et al., 2023. Clifford Neural Layers for PDE Modeling.
      (https://arxiv.org/abs/2209.04934)
    - Ruhe et al., 2023. Clifford Group Equivariant Neural Networks.
      (https://arxiv.org/abs/2302.06594)
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.conv_blocks.conv_block import ConvBlock
from dl_techniques.layers.mixtures.factory import create_mixture_layer
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.drop_path import linear_drop_path_rates

# ---------------------------------------------------------------------


def _create_rbf_head(
        num_classes: int,
        repulsion_strength: float,
        min_distance: float,
        name: str = "rbf_head",
) -> keras.layers.Layer:
    """Build the RBF prototype-classification head shared by both backbones.

    The ONE genuinely-reused piece of construction logic between
    :class:`RBFProtoNet` and :class:`CliffordRBFProtoNet` -- both call this
    function from their own ``_build_rbf_head`` rather than duplicating the
    ``create_mixture_layer`` call inline. See decisions.md D-002.

    # DECISION plan-2026-09-16T064227-b83f88ad/D-002: 'normalized' is the ONLY
    # supported output_mode for this head -- do not expose output_mode as a
    # parameter here defaulting to 'basis'. RBFLayer's own docstring and
    # decisions.md D-002 both document 'basis' as barely-trainable at this
    # scale; making it reachable here would reintroduce that footgun through
    # both models' config surfaces at once. See decisions.md D-002 for the
    # full trade-off.

    :param num_classes: Number of RBF prototype units (= number of output
        classes).
    :type num_classes: int
    :param repulsion_strength: Strength of the RBF head's center-repulsion
        penalty, forwarded verbatim to ``create_mixture_layer('rbf', ...)``.
    :type repulsion_strength: float
    :param min_distance: Minimum desired distance between RBF centers,
        forwarded verbatim to ``create_mixture_layer('rbf', ...)``.
    :type min_distance: float
    :param name: Layer name. Defaults to ``"rbf_head"``.
    :type name: str
    :return: A configured RBF mixture layer (``output_mode='normalized'``).
    :rtype: keras.layers.Layer
    """
    return create_mixture_layer(
        "rbf",
        units=num_classes,
        output_mode="normalized",
        repulsion_strength=repulsion_strength,
        min_distance=min_distance,
        name=name,
    )


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.rbf_protonet.model")
class RBFProtoNet(keras.Model):
    """
    CIFAR-style CNN backbone with an RBF prototype-classification head.

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

        ┌────────────────────────────────────────┐
        │       Input [B, 32, 32, 3]             │
        └───────────────┬────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────┐
        │  Stem: ConvBlock 3x3 /1                │  32x32 -> 32x32
        │  (no pooling -- input has no spare     │
        │   resolution to give up)               │
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  Stage 1: ConvBlock 3x3 /2             │  32x32 -> 16x16
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  Stage 2: ConvBlock 3x3 /2             │  16x16 -> 8x8
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  Stage 3: ConvBlock 3x3 /2             │  8x8   -> 4x4
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  GlobalAveragePooling2D                │  4x4xF -> F
        │  (+ Dense projection to feature_dim,   │
        │   only if the last stage width != F)   │
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  RBFLayer head (output_mode=           │  F -> num_classes
        │  'normalized', via create_mixture_     │
        │  layer('rbf', ...))                    │
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  Output: [B, num_classes]              │
        │  (probabilities, sums to 1.0 per row)  │
        └────────────────────────────────────────┘

    Note:
        The RBF head is built via ``create_mixture_layer('rbf', units=
        num_classes, output_mode='normalized', ...)`` (see D-002 in this
        plan's ``decisions.md``: ``'normalized'`` is the ONLY supported
        mode for this model -- ``'basis'`` is not exposed as a togglable
        knob). ``call()`` runs the backbone and then the head, returning a
        per-class probability vector.

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
    :param num_classes: Number of RBF prototype units (= number of output
        classes) in the head. Must be a positive int. Defaults to ``100``
        (CIFAR-100).
    :type num_classes: int
    :param repulsion_strength: Strength of the RBF head's center-repulsion
        penalty, forwarded verbatim to ``create_mixture_layer('rbf', ...)``.
        Defaults to ``0.1`` (the factory's own default), which is exactly
        the value Step 1's smoke test measured at ``units=100, feature_dim=
        128``: repulsion loss stayed at 0.14%-1.14% of total loss, nowhere
        near the 50% dominance floor -- see decisions.md D-006. Not
        re-tuned here since the measured default already passes.
    :type repulsion_strength: float
    :param min_distance: Minimum desired distance between RBF centers,
        forwarded verbatim to ``create_mixture_layer('rbf', ...)``. Defaults
        to ``1.0`` (the factory's own default), for the same reason as
        ``repulsion_strength`` above.
    :type min_distance: float
    :param pretrained: If ``True``, raises ``NotImplementedError`` --
        this package distributes no pretrained weights. Defaults to
        ``False``.
    :type pretrained: bool
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``input_shape`` is not 3D, if ``stem_filters``,
        ``feature_dim`` or ``num_classes`` is not positive, or if
        ``filters_per_stage`` is empty or contains a non-positive value.
    :raises NotImplementedError: If ``pretrained=True``.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        2D tensor ``(batch_size, num_classes)``, rows summing to 1.0.

    Example:
        >>> model = RBFProtoNet(input_shape=(32, 32, 3), num_classes=100)
        >>> probs = model(keras.random.normal((4, 32, 32, 3)))
        >>> probs.shape
        TensorShape([4, 100])
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
            num_classes: int = 100,
            repulsion_strength: float = 0.1,
            min_distance: float = 1.0,
            pretrained: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if pretrained:
            raise NotImplementedError(
                "RBFProtoNet ships no pretrained weights; pretrained=True is "
                "not supported."
            )

        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")
        if stem_filters <= 0:
            raise ValueError(f"stem_filters must be positive, got {stem_filters}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive int, got {num_classes}")

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
        # DECISION plan-2026-09-16T052349-7dfede94/D-006: default feature_dim=128, not
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
        self.num_classes = num_classes
        self.repulsion_strength = repulsion_strength
        self.min_distance = min_distance
        self.pretrained = pretrained

        self._build_stem()

        self.stages: List[ConvBlock] = []
        for stage_idx in range(len(self.filters_per_stage)):
            self._build_stage(stage_idx)

        self._build_head()
        self._build_rbf_head()

        logger.info(
            f"Created RBFProtoNet with {len(self.filters_per_stage)} "
            f"downsampling stages for input {self.input_shape_config}, "
            f"pooled feature_dim={self.feature_dim}, "
            f"num_classes={self.num_classes}"
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

    def _build_rbf_head(self) -> None:
        """Build the RBF prototype-classification head.

        # DECISION plan-2026-09-16T052349-7dfede94/D-002: 'normalized' is the ONLY
        # supported output_mode for this model's head -- do not expose
        # output_mode as a constructor knob defaulting to 'basis'. RBFLayer's
        # own docstring and decisions.md D-002 both document 'basis' as
        # barely-trainable at this scale; making it reachable here would
        # reintroduce that footgun through this model's own config surface.
        # See decisions.md D-002 for the full trade-off.

        Delegates to the module-level ``_create_rbf_head`` helper shared with
        :class:`CliffordRBFProtoNet` -- see decisions.md D-002 (this plan's own
        entry, which documents the extraction; the anchor comment above is the
        UNCHANGED, preserved decision from the prior plan that originally
        justified 'normalized'-only).
        """
        self.rbf_head = _create_rbf_head(
            self.num_classes,
            self.repulsion_strength,
            self.min_distance,
        )

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
        :return: Per-class probability tensor `(batch_size, num_classes)`,
            rows summing to 1.0 (``output_mode='normalized'`` contract).
        """
        x = self.stem(inputs, training=training)

        for stage in self.stages:
            x = stage(x, training=training)

        pooled = self.gap(x)
        if self.feature_proj is not None:
            pooled = self.feature_proj(pooled)

        return self.rbf_head(pooled, training=training)

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
            "num_classes": self.num_classes,
            "repulsion_strength": self.repulsion_strength,
            "min_distance": self.min_distance,
            "pretrained": self.pretrained,
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


def create_rbf_protonet(
        num_classes: int = 100,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        pretrained: bool = False,
        **kwargs: Any,
) -> RBFProtoNet:
    """Create an :class:`RBFProtoNet` model.

    Thin delegating factory, per the house model module shape -- no extra
    logic beyond forwarding to the constructor.

    :param num_classes: Number of RBF prototype units (= output classes).
        Defaults to ``100`` (CIFAR-100).
    :type num_classes: int
    :param input_shape: Input shape ``(height, width, channels)`` excluding
        the batch dimension. Defaults to ``(32, 32, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param pretrained: If ``True``, raises ``NotImplementedError`` (no
        pretrained weights are distributed). Defaults to ``False``.
    :type pretrained: bool
    :param kwargs: Additional keyword arguments forwarded to
        :class:`RBFProtoNet`.
    :return: A configured, uncompiled :class:`RBFProtoNet` model.
    :rtype: RBFProtoNet
    """
    return RBFProtoNet(
        input_shape=input_shape,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.rbf_protonet.model")
class CliffordRBFProtoNet(keras.Model):
    """
    Isotropic Clifford-algebra backbone with an RBF prototype-classification head.

    A single stride-``patch_size`` patch-embedding stem (``Conv2D`` + batch
    normalization) reduces spatial resolution once, followed by ``depth``
    identical ``CliffordNetBlock`` residual blocks at a fixed ``channels``
    width (no per-stage downsampling -- every existing ``CliffordNetBlock``
    consumer in this repo, e.g. ``CliffordNet``, is isotropic; see
    decisions.md D-003). A global average pool collapses the final feature
    map to a flat vector, optionally projected to ``feature_dim`` when
    ``channels != feature_dim``, then fed into the same RBF prototype head
    design used by :class:`RBFProtoNet`.

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │       Input [B, 32, 32, 3]             │
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  Stem: Conv2D 3x3 /patch_size + BN     │  32x32 -> 16x16 (patch_size=2)
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  depth x CliffordNetBlock (channels)   │  16x16 -> 16x16
        │  x = x + drop_path(block(x))           │  (isotropic, no downsampling)
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  GlobalAveragePooling2D                │  16x16xC -> C
        │  (+ Dense projection to feature_dim,   │
        │   only if channels != feature_dim)     │
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  RBFLayer head (output_mode=           │  F -> num_classes
        │  'normalized', via create_mixture_     │
        │  layer('rbf', ...))                    │
        └───────────────┬────────────────────────┘
                        ▼
        ┌────────────────────────────────────────┐
        │  Output: [B, num_classes]              │
        │  (probabilities, sums to 1.0 per row)  │
        └────────────────────────────────────────┘

    Note:
        The RBF head is built via ``create_mixture_layer('rbf', units=
        num_classes, output_mode='normalized', ...)`` (see decisions.md
        D-002: ``'normalized'`` is the ONLY supported mode for this model --
        ``'basis'`` is not exposed as a togglable knob). ``call()`` runs the
        stem, the isotropic Clifford block stack, the pooling head and then
        the RBF head, returning a per-class probability vector.

    :param input_shape: Input shape ``(height, width, channels)`` excluding
        the batch dimension. Defaults to ``(32, 32, 3)`` (CIFAR).
    :type input_shape: Tuple[int, int, int]
    :param num_classes: Number of RBF prototype units (= number of output
        classes) in the head. Must be a positive int. Defaults to ``100``
        (CIFAR-100).
    :type num_classes: int
    :param channels: Fixed channel width of every ``CliffordNetBlock`` in the
        backbone (isotropic -- no per-stage widening). Defaults to ``128``
        (the ``nano``-equivalent configuration, decisions.md D-004).
    :type channels: int
    :param depth: Number of identical ``CliffordNetBlock`` residual blocks.
        Defaults to ``12``.
    :type depth: int
    :param shifts: Shift amounts forwarded to every ``CliffordNetBlock``.
        Defaults to ``(1, 2)``.
    :type shifts: Tuple[int, ...]
    :param patch_size: Stride of the patch-embedding stem's ``Conv2D``.
        Defaults to ``2``.
    :type patch_size: int
    :param use_global_context: Whether every ``CliffordNetBlock`` uses global
        context. Defaults to ``False``.
    :type use_global_context: bool
    :param drop_path_rate: Maximum (last-block) stochastic-depth drop
        probability; per-block rates are ramped linearly from ``0`` to this
        value via ``linear_drop_path_rates(depth, max_rate=drop_path_rate)``.
        Defaults to ``0.1`` (matching the ``stochastic_depth_rate=0.1`` value
        ``CliffordNet``'s own ``nano`` variant example uses -- see
        decisions.md D-006; ``CliffordNet``'s own constructor default is
        ``0.0``, but the trained ``nano`` example uses ``0.1``).
    :type drop_path_rate: float
    :param feature_dim: Width of the pooled feature vector fed to the RBF
        head. Defaults to ``128``.
    :type feature_dim: int
    :param repulsion_strength: Strength of the RBF head's center-repulsion
        penalty, forwarded verbatim to ``create_mixture_layer('rbf', ...)``.
        Defaults to ``0.1`` (the factory's own default).
    :type repulsion_strength: float
    :param min_distance: Minimum desired distance between RBF centers,
        forwarded verbatim to ``create_mixture_layer('rbf', ...)``. Defaults
        to ``1.0`` (the factory's own default).
    :type min_distance: float
    :param kernel_regularizer: Optional regularizer applied to the stem's
        convolution kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param pretrained: If ``True``, raises ``NotImplementedError`` -- this
        package distributes no pretrained weights. Defaults to ``False``.
    :type pretrained: bool
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``input_shape`` is not 3D, or if ``channels``,
        ``depth``, ``patch_size``, ``feature_dim`` or ``num_classes`` is not
        positive.
    :raises NotImplementedError: If ``pretrained=True``.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        2D tensor ``(batch_size, num_classes)``, rows summing to 1.0.
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (32, 32, 3),
            num_classes: int = 100,
            channels: int = 128,
            depth: int = 12,
            shifts: Tuple[int, ...] = (1, 2),
            patch_size: int = 2,
            use_global_context: bool = False,
            drop_path_rate: float = 0.1,
            feature_dim: int = 128,
            repulsion_strength: float = 0.1,
            min_distance: float = 1.0,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            pretrained: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if pretrained:
            raise NotImplementedError(
                "CliffordRBFProtoNet ships no pretrained weights; "
                "pretrained=True is not supported."
            )

        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive int, got {num_classes}")

        self.input_shape_config = tuple(input_shape)
        self.input_height, self.input_width, self.input_channels = input_shape
        self.num_classes = num_classes
        self.channels = channels
        self.depth = depth
        self.shifts = tuple(shifts)
        self.patch_size = patch_size
        self.use_global_context = use_global_context
        self.drop_path_rate = drop_path_rate
        self.feature_dim = feature_dim
        self.repulsion_strength = repulsion_strength
        self.min_distance = min_distance
        self.kernel_regularizer = kernel_regularizer
        self.pretrained = pretrained

        self._build_stem()
        self._build_blocks()
        self._build_head()
        self._build_rbf_head()

        logger.info(
            f"Created CliffordRBFProtoNet with depth={self.depth} "
            f"CliffordNetBlocks at channels={self.channels} for input "
            f"{self.input_shape_config}, pooled feature_dim={self.feature_dim}, "
            f"num_classes={self.num_classes}"
        )

    def _build_stem(self) -> None:
        """Build the patch-embedding stem: strided ``Conv2D`` + ``BatchNormalization``.

        # DECISION plan-2026-09-16T064227-b83f88ad/D-005: this exact composition
        # (Conv2D(strides=patch_size) -> BatchNormalization(momentum=0.9))
        # was validated by step 1's standalone smoke test before being
        # folded into this class. Do not swap in ConvBlock or a different
        # normalization momentum without re-running that smoke test --
        # this specific pairing is what was measured NaN/Inf-clean with
        # live gradients across all 12 downstream blocks. See decisions.md
        # D-005.
        """
        self.stem_conv = keras.layers.Conv2D(
            filters=self.channels,
            kernel_size=3,
            strides=self.patch_size,
            padding="same",
            use_bias=True,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv",
        )
        self.stem_bn = keras.layers.BatchNormalization(
            momentum=0.9,
            name="stem_bn",
        )

    def _build_blocks(self) -> None:
        """Build the isotropic stack of ``depth`` identical ``CliffordNetBlock``s.

        Per-block stochastic-depth rates are ramped linearly from ``0`` to
        ``drop_path_rate`` via ``linear_drop_path_rates`` -- the same
        schedule ``CliffordNet.call()`` uses. ``self.blocks`` and
        ``self.drop_paths`` are stored as parallel lists for ``call()``
        (step 4) to iterate: ``x = x + drop_paths[i](blocks[i](x))``.
        """
        drop_rates = linear_drop_path_rates(self.depth, max_rate=self.drop_path_rate)

        self.blocks: List[CliffordNetBlock] = []
        self.drop_paths: List[StochasticDepth] = []
        for i in range(self.depth):
            block = CliffordNetBlock(
                channels=self.channels,
                shifts=list(self.shifts),
                use_global_context=self.use_global_context,
                name=f"clifford_block_{i}",
            )
            drop_path = StochasticDepth(
                drop_path_rate=drop_rates[i],
                name=f"drop_path_{i}",
            )
            self.blocks.append(block)
            self.drop_paths.append(drop_path)

    def _build_head(self) -> None:
        """Build the global average pool and, if needed, a projection Dense.

        A projection is created only when ``channels`` does not already
        equal ``feature_dim`` -- mirrors ``RBFProtoNet._build_head``'s exact
        conditional logic.
        """
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")

        if self.channels != self.feature_dim:
            self.feature_proj = keras.layers.Dense(
                units=self.feature_dim,
                kernel_initializer="he_normal",
                kernel_regularizer=self.kernel_regularizer,
                name="feature_proj",
            )
        else:
            self.feature_proj = None

    def _build_rbf_head(self) -> None:
        """Build the RBF prototype-classification head.

        Delegates to the module-level ``_create_rbf_head`` helper shared with
        :class:`RBFProtoNet` -- see decisions.md D-002.
        """
        self.rbf_head = _create_rbf_head(
            self.num_classes,
            self.repulsion_strength,
            self.min_distance,
        )

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
        """Run the stem, the isotropic Clifford block stack, and the pooling head.

        :param inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        :param training: Whether batch norm and stochastic depth run in training mode.
        :return: Per-class probability tensor `(batch_size, num_classes)`,
            rows summing to 1.0 (``output_mode='normalized'`` contract).
        """
        x = self.stem_conv(inputs)
        x = self.stem_bn(x, training=training)

        for block, drop_path in zip(self.blocks, self.drop_paths):
            x = x + drop_path(block(x, training=training), training=training)

        pooled = self.gap(x)
        if self.feature_proj is not None:
            pooled = self.feature_proj(pooled)

        return self.rbf_head(pooled, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes,
            "channels": self.channels,
            "depth": self.depth,
            "shifts": self.shifts,
            "patch_size": self.patch_size,
            "use_global_context": self.use_global_context,
            "drop_path_rate": self.drop_path_rate,
            "feature_dim": self.feature_dim,
            "repulsion_strength": self.repulsion_strength,
            "min_distance": self.min_distance,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            "pretrained": self.pretrained,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordRBFProtoNet":
        """Create a model from its `get_config()` output."""
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)


# ---------------------------------------------------------------------


def create_clifford_rbf_protonet(
        num_classes: int = 100,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        pretrained: bool = False,
        **kwargs: Any,
) -> CliffordRBFProtoNet:
    """Create a :class:`CliffordRBFProtoNet` model.

    Thin delegating factory, per the house model module shape -- no extra
    logic beyond forwarding to the constructor.

    :param num_classes: Number of RBF prototype units (= output classes).
        Defaults to ``100`` (CIFAR-100).
    :type num_classes: int
    :param input_shape: Input shape ``(height, width, channels)`` excluding
        the batch dimension. Defaults to ``(32, 32, 3)``.
    :type input_shape: Tuple[int, int, int]
    :param pretrained: If ``True``, raises ``NotImplementedError`` (no
        pretrained weights are distributed). Defaults to ``False``.
    :type pretrained: bool
    :param kwargs: Additional keyword arguments forwarded to
        :class:`CliffordRBFProtoNet`.
    :return: A configured, uncompiled :class:`CliffordRBFProtoNet` model.
    :rtype: CliffordRBFProtoNet
    """
    if pretrained:
        raise NotImplementedError(
            "CliffordRBFProtoNet has no pretrained weights; pass "
            "pretrained=False and use model.load_weights(path) for warm starts."
        )
    return CliffordRBFProtoNet(
        input_shape=input_shape,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )

# ---------------------------------------------------------------------

