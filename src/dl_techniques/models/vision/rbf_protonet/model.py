"""``RBFProtoNet``, a small CIFAR-style CNN backbone with an RBF prototype-classification head.

A CNN reduces a 32x32 RGB image to a flat feature vector; an ``RBFLayer``
(``dl_techniques.layers.mixtures.radial_basis_function``, built through
``create_mixture_layer('rbf', ...)``) is then applied as a distance-based
class-prototype classification head on top of that vector, so each class gets
one learned, inspectable prototype in feature space instead of a distributed
softmax-of-logits weight matrix. ``call()`` runs the backbone, then the head,
and returns a per-class probability vector (``output_mode='normalized'``) that
sums to 1.0 along the last axis.

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

Neither model takes an ``input_shape``: both end in a global average pool, so
they run at any spatial size and nothing in them would read one. A caller that
still passes ``input_shape`` (or ``input_dim``) gets a ``ValueError`` naming it
instead of Keras's warning (``refuse_input_shape``). Neither distributes
pretrained weights, and the factories refuse ``pretrained``
(``refuse_pretrained``); a warm start is ``model.load_weights(path)``.

This package intentionally ships without a ``MODEL_VARIANTS`` table (see
``models/AGENTS.md`` Section "When the shape does not apply" / the guide's
Section 5.6): there is exactly one baseline architecture here, and inventing
named size multipliers with no genuine use case would be exactly the
"forcing an ill-fitting template" anti-pattern the house shape warns against.

This module also hosts ``CliffordRBFProtoNet``, a second backbone/head pairing
that swaps the CNN backbone above for an isotropic Clifford-algebra backbone
(``CliffordNetBlock``, ``dl_techniques.layers.geometric.clifford_block``) while
reusing the same RBF prototype-classification head design. Both classes
register under the same module path (``dl_techniques.models.rbf_protonet.model``)
since they are defined in this one file.

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
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.conv_blocks.conv_block import ConvBlock
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.mixtures.factory import create_mixture_layer
from dl_techniques.utils.model_build import (
    materialize_sublayers,
    refuse_input_shape,
    validate_rates,
)
from dl_techniques.utils.validation import positive_int
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.weight_transfer import refuse_pretrained
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.drop_path import linear_drop_path_rates

# ---------------------------------------------------------------------

#: BatchNormalization momentum and epsilon of the CliffordRBFProtoNet patch-embedding stem.
#: The norm factory's own epsilon default is ``1e-6``, not the Keras ``1e-3`` the stem was
#: measured with, so both values are stated. They are this model's own: the stem's
#: composition was validated at exactly these values (anchor above ``_build_stem``).
# DECISION plan-2026-10-26T125959-7af851e5/D-049: these two constants are this model's own
# although ``cliffordnet.model`` defines constants of the same names and values. Do NOT
# replace them with an import of those: the stem was validated at exactly these values
# (the D-005 anchor at ``_build_stem``), and an import would let a retune of CliffordNet's
# stem silently change this model's stem and the starting point of every archive. A
# duplicated pair of numbers is the price of keeping the two models independent. See decisions.md.
STEM_BN_MOMENTUM = 0.9
STEM_BN_EPSILON = 1e-3

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
        num_classes, output_mode='normalized', ...)``: ``'normalized'`` is the
        ONLY supported mode for this model -- ``'basis'`` is not exposed as a
        togglable knob. ``call()`` runs the backbone and then the head,
        returning a per-class probability vector.

    :param num_classes: Number of RBF prototype units (= number of output
        classes) in the head. Must be a positive integer (a numpy integer is
        accepted). Defaults to ``100`` (CIFAR-100).
    :type num_classes: int
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
    :param kernel_regularizer: Optional regularizer (instance, name or
        serialized config) applied to every convolution kernel and to the
        feature projection, when there is one.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param dropout_rate: Dropout rate applied inside every downsampling
        stage's ``ConvBlock`` (``0.0`` disables it). Not applied in the stem.
        Must be in ``[0, 1)``. Defaults to ``0.0``.
    :type dropout_rate: float
    :param repulsion_strength: Strength of the RBF head's center-repulsion
        penalty, forwarded verbatim to ``create_mixture_layer('rbf', ...)``.
        Defaults to ``0.1`` (the factory's own default). Measured at
        ``units=100, feature_dim=128`` the repulsion loss stayed at
        0.14%-1.14% of the total loss, far from dominating it, so the
        default was not re-tuned.
    :type repulsion_strength: float
    :param min_distance: Minimum desired distance between RBF centers,
        forwarded verbatim to ``create_mixture_layer('rbf', ...)``. Defaults
        to ``1.0`` (the factory's own default), for the same reason as
        ``repulsion_strength`` above.
    :type min_distance: float
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``num_classes`` is not a positive integer, if
        ``stem_filters`` or ``feature_dim`` is not positive, if
        ``filters_per_stage`` is empty or contains a non-positive value, if
        ``dropout_rate`` is outside ``[0, 1)``, or if a keyword is not an
        argument of the model: ``input_shape`` and ``input_dim`` are refused by
        name (the model runs at any spatial size), anything else by Keras.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``; any
        height and width the strides allow.

    Output shape:
        2D tensor ``(batch_size, num_classes)``, rows summing to 1.0.

    Example:
        >>> model = RBFProtoNet(num_classes=100)
        >>> probs = model(keras.random.normal((4, 32, 32, 3)))
        >>> probs.shape
        TensorShape([4, 100])
    """

    def __init__(
            self,
            num_classes: int = 100,
            stem_filters: int = 32,
            filters_per_stage: Optional[List[int]] = None,
            feature_dim: int = 128,
            normalization_type: str = "batch_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_type: str = "relu",
            kernel_regularizer: Optional[
                Union[str, Dict[str, Any], keras.regularizers.Regularizer]
            ] = None,
            dropout_rate: float = 0.0,
            repulsion_strength: float = 0.1,
            min_distance: float = 1.0,
            **kwargs: Any,
    ) -> None:
        refuse_input_shape(
            kwargs, type(self).__name__,
            "it ends in a global average pool and runs at any spatial size")
        super().__init__(**kwargs)

        num_classes = positive_int(num_classes, "num_classes", type(self).__name__)
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
        validate_rates(dropout_rate=dropout_rate)

        self.num_classes = num_classes
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
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.dropout_rate = dropout_rate
        self.repulsion_strength = repulsion_strength
        self.min_distance = min_distance

        self._build_stem()

        self.stages: List[ConvBlock] = []
        for stage_idx in range(len(self.filters_per_stage)):
            self._build_stage(stage_idx)

        self._build_head()
        self._build_rbf_head()

        logger.info(
            f"Created RBFProtoNet with {len(self.filters_per_stage)} "
            f"downsampling stages, pooled feature_dim={self.feature_dim}, "
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

    # DECISION plan-2026-09-16T052349-7dfede94/D-002: 'normalized' is the ONLY
    # supported output_mode for this model's head -- do not expose
    # output_mode as a constructor knob defaulting to 'basis'. RBFLayer's
    # own docstring and decisions.md D-002 both document 'basis' as
    # barely-trainable at this scale; making it reachable here would
    # reintroduce that footgun through this model's own config surface.
    # See decisions.md D-002 for the full trade-off.
    def _build_rbf_head(self) -> None:
        """Build the RBF prototype-classification head (``output_mode='normalized'``)."""
        self.rbf_head = create_mixture_layer(
            "rbf",
            units=self.num_classes,
            output_mode="normalized",
            repulsion_strength=self.repulsion_strength,
            min_distance=self.min_distance,
            name="rbf_head",
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
            "num_classes": self.num_classes,
            "stem_filters": self.stem_filters,
            "filters_per_stage": list(self.filters_per_stage),
            "feature_dim": self.feature_dim,
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
            "activation_type": self.activation_type,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "dropout_rate": self.dropout_rate,
            "repulsion_strength": self.repulsion_strength,
            "min_distance": self.min_distance,
        }
        base_config = super().get_config()
        return {**base_config, **config}


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.rbf_protonet.model")
class CliffordRBFProtoNet(keras.Model):
    """
    Isotropic Clifford-algebra backbone with an RBF prototype-classification head.

    A single stride-``patch_size`` patch-embedding stem (a ``ConvStem`` of one
    ``Conv2D`` and a batch normalization) reduces spatial resolution once,
    followed by ``depth`` identical ``CliffordNetBlock`` residual blocks at a
    fixed ``channels`` width (no per-stage downsampling -- every existing
    ``CliffordNetBlock`` consumer in this repo, e.g. ``CliffordNet``, is
    isotropic). A global average pool collapses the final feature map to a flat
    vector, optionally projected to ``feature_dim`` when ``channels !=
    feature_dim``, then fed into the same RBF prototype head design used by
    :class:`RBFProtoNet`.

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
        num_classes, output_mode='normalized', ...)``: ``'normalized'`` is the
        ONLY supported mode for this model -- ``'basis'`` is not exposed as a
        togglable knob. ``call()`` runs the stem, the isotropic Clifford block
        stack, the pooling head and then the RBF head, returning a per-class
        probability vector.

    :param num_classes: Number of RBF prototype units (= number of output
        classes) in the head. Must be a positive integer (a numpy integer is
        accepted). Defaults to ``100`` (CIFAR-100).
    :type num_classes: int
    :param channels: Fixed channel width of every ``CliffordNetBlock`` in the
        backbone (isotropic -- no per-stage widening). Defaults to ``128``
        (the ``nano``-equivalent configuration).
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
        probability, in ``[0, 1)``; per-block rates are ramped linearly from
        ``0`` to this value via ``linear_drop_path_rates(depth,
        max_rate=drop_path_rate)``. Defaults to ``0.1`` (the
        ``stochastic_depth_rate=0.1`` value ``CliffordNet``'s own ``nano``
        variant example uses; ``CliffordNet``'s own constructor default is
        ``0.0``).
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
    :param kernel_regularizer: Optional regularizer (instance, name or
        serialized config) applied to the stem's convolution kernel and to the
        feature projection, when there is one.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class.

    :raises ValueError: If ``num_classes`` is not a positive integer, if
        ``channels``, ``depth``, ``patch_size`` or ``feature_dim`` is not positive,
        if ``drop_path_rate`` is outside ``[0, 1)``, or if a keyword is not an
        argument of the model: ``input_shape`` and ``input_dim`` are refused by
        name (the model runs at any spatial size), anything else by Keras.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``; any
        height and width.

    Output shape:
        2D tensor ``(batch_size, num_classes)``, rows summing to 1.0.

    Example:
        >>> model = CliffordRBFProtoNet(num_classes=100, depth=4)
        >>> probs = model(keras.random.normal((4, 32, 32, 3)))
        >>> probs.shape
        TensorShape([4, 100])
    """

    def __init__(
            self,
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
            kernel_regularizer: Optional[
                Union[str, Dict[str, Any], keras.regularizers.Regularizer]
            ] = None,
            **kwargs: Any,
    ) -> None:
        refuse_input_shape(
            kwargs, type(self).__name__,
            "it ends in a global average pool and runs at any spatial size")
        super().__init__(**kwargs)

        num_classes = positive_int(num_classes, "num_classes", type(self).__name__)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        validate_rates(drop_path_rate=drop_path_rate)

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
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self._build_stem()
        self._build_blocks()
        self._build_head()
        self._build_rbf_head()

        logger.info(
            f"Created CliffordRBFProtoNet with depth={self.depth} "
            f"CliffordNetBlocks at channels={self.channels}, "
            f"pooled feature_dim={self.feature_dim}, "
            f"num_classes={self.num_classes}"
        )

    # DECISION plan-2026-09-16T064227-b83f88ad/D-005: this exact composition
    # (Conv2D(strides=patch_size) -> BatchNormalization(momentum=0.9))
    # was validated by step 1's standalone smoke test before being
    # folded into this class. Do not swap in ConvBlock or a different
    # normalization momentum without re-running that smoke test --
    # this specific pairing is what was measured NaN/Inf-clean with
    # live gradients across all 12 downstream blocks. See decisions.md
    # D-005.
    def _build_stem(self) -> None:
        """Build the patch-embedding stem: a ``ConvStem`` of strided ``Conv2D`` + ``BatchNormalization``.

        The composition is ``Conv2D(strides=patch_size)`` ->
        ``BatchNormalization(momentum=STEM_BN_MOMENTUM, epsilon=STEM_BN_EPSILON)``,
        no activation, no pool. The standing test
        ``TestCliffordRBFProtoNetD005StandingSmoke`` in
        ``tests/test_models/test_rbf_protonet/test_clifford_model.py`` re-runs
        the smoke the anchor above asks for.
        """
        self.stem = ConvStem(
            filters=self.channels,
            kernel_size=3,
            strides=self.patch_size,
            padding="same",
            use_bias=True,
            normalization_type="batch_norm",
            normalization_kwargs={"momentum": STEM_BN_MOMENTUM, "epsilon": STEM_BN_EPSILON},
            activation_type="linear",
            kernel_regularizer=self.kernel_regularizer,
            name="stem",
        )

    def _build_blocks(self) -> None:
        """Build the isotropic stack of ``depth`` identical ``CliffordNetBlock``s.

        Per-block stochastic-depth rates are ramped linearly from ``0`` to
        ``drop_path_rate`` via ``linear_drop_path_rates`` -- the same
        schedule ``CliffordNet.call()`` uses. ``self.blocks`` and
        ``self.drop_paths`` are parallel lists, named ``block_{i}`` and
        ``block_{i}_drop_path``, that ``call()`` iterates:
        ``x = x + drop_paths[i](blocks[i](x))``.
        """
        drop_rates = linear_drop_path_rates(self.depth, max_rate=self.drop_path_rate)

        self.blocks: List[CliffordNetBlock] = []
        self.drop_paths: List[StochasticDepth] = []
        for i in range(self.depth):
            block = CliffordNetBlock(
                channels=self.channels,
                shifts=list(self.shifts),
                use_global_context=self.use_global_context,
                name=f"block_{i}",
            )
            drop_path = StochasticDepth(
                drop_path_rate=drop_rates[i],
                name=f"block_{i}_drop_path",
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

    # DECISION plan-2026-09-16T064227-b83f88ad/D-002: 'normalized' is the ONLY
    # supported output_mode for this head -- do not expose output_mode as a
    # constructor knob defaulting to 'basis' (RBFLayer's own docstring and
    # decisions.md D-002 document 'basis' as barely-trainable at this scale).
    # The pin is written inline here and in RBFProtoNet._build_rbf_head, one
    # call per model; a test asserts both heads are 'normalized'.
    def _build_rbf_head(self) -> None:
        """Build the RBF prototype-classification head (``output_mode='normalized'``)."""
        self.rbf_head = create_mixture_layer(
            "rbf",
            units=self.num_classes,
            output_mode="normalized",
            repulsion_strength=self.repulsion_strength,
            min_distance=self.min_distance,
            name="rbf_head",
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
        x = self.stem(inputs, training=training)

        for block, drop_path in zip(self.blocks, self.drop_paths):
            x = x + drop_path(block(x, training=training), training=training)

        pooled = self.gap(x)
        if self.feature_proj is not None:
            pooled = self.feature_proj(pooled)

        return self.rbf_head(pooled, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "channels": self.channels,
            "depth": self.depth,
            "shifts": list(self.shifts),
            "patch_size": self.patch_size,
            "use_global_context": self.use_global_context,
            "drop_path_rate": self.drop_path_rate,
            "feature_dim": self.feature_dim,
            "repulsion_strength": self.repulsion_strength,
            "min_distance": self.min_distance,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}


# ---------------------------------------------------------------------


#: What ``refuse_pretrained`` tells a caller of either factory to do instead. These models have a
#: backbone and an ``rbf_head`` and no distributed weights, ``keras.models.load_model`` refuses an
#: archive written before the ``pretrained`` and ``input_shape`` knobs were removed (their config
#: carries both keys), and ``load_weights`` restores such an archive into a model of the same
#: architecture, so that is the advice the shared message gives.
WARM_START_HINT = (
    "Build the model and call model.load_weights(path) to warm-start it from a saved model "
    "of the same architecture, or pass pretrained=False to random-init."
)


def create_rbf_protonet(
        num_classes: int = 100,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
) -> RBFProtoNet:
    """Create an :class:`RBFProtoNet` model.

    Thin delegating factory, per the house model module shape -- no extra
    logic beyond refusing ``pretrained`` and forwarding to the constructor.

    :param num_classes: Number of RBF prototype units (= output classes).
        Defaults to ``100`` (CIFAR-100).
    :type num_classes: int
    :param pretrained: Any truthy value, a path included, raises
        ``NotImplementedError``: no pretrained weights are distributed and a
        warm start is ``model.load_weights(path)``. Defaults to ``False``.
    :type pretrained: Union[bool, str]
    :param kwargs: Additional keyword arguments forwarded to
        :class:`RBFProtoNet`.
    :return: A configured, uncompiled :class:`RBFProtoNet` model.
    :rtype: RBFProtoNet
    :raises NotImplementedError: If ``pretrained`` is truthy.
    :raises ValueError: If a keyword is not an argument of :class:`RBFProtoNet`; the
        message names it. ``input_shape`` was a parameter of this factory once and
        is refused, not ignored.
    """
    refuse_pretrained(pretrained, "RBFProtoNet", WARM_START_HINT)
    return RBFProtoNet(num_classes=num_classes, **kwargs)


def create_clifford_rbf_protonet(
        num_classes: int = 100,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
) -> CliffordRBFProtoNet:
    """Create a :class:`CliffordRBFProtoNet` model.

    Thin delegating factory, per the house model module shape -- no extra
    logic beyond refusing ``pretrained`` and forwarding to the constructor.

    :param num_classes: Number of RBF prototype units (= output classes).
        Defaults to ``100`` (CIFAR-100).
    :type num_classes: int
    :param pretrained: Any truthy value, a path included, raises
        ``NotImplementedError``: no pretrained weights are distributed and a
        warm start is ``model.load_weights(path)``. Defaults to ``False``.
    :type pretrained: Union[bool, str]
    :param kwargs: Additional keyword arguments forwarded to
        :class:`CliffordRBFProtoNet`.
    :return: A configured, uncompiled :class:`CliffordRBFProtoNet` model.
    :rtype: CliffordRBFProtoNet
    :raises NotImplementedError: If ``pretrained`` is truthy.
    :raises ValueError: If a keyword is not an argument of :class:`CliffordRBFProtoNet`; the
        message names it. ``input_shape`` was a parameter of this factory once and
        is refused, not ignored.
    """
    refuse_pretrained(pretrained, "CliffordRBFProtoNet", WARM_START_HINT)
    return CliffordRBFProtoNet(num_classes=num_classes, **kwargs)

# ---------------------------------------------------------------------
