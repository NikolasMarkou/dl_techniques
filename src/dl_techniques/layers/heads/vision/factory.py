"""Vision task heads, and the factory that picks one.

This module holds eight layer classes and five module-level helpers. The
classes turn a backbone feature map into task predictions. The helpers pick a
class and configure it.

The heads take a feature map, not an image. Nothing here knows which backbone
produced it, so the same head works on a ResNet, a ViT or a ConvNeXt stage.

Classes
-------
* :class:`BaseVisionHead` -- shared construction. Six heads inherit from it.
  It has no ``call``.
* :class:`DetectionHead` -- class scores and box offsets, per anchor.
* :class:`SegmentationHead` -- one map of per-pixel class logits, upsampled
  towards input size. The only head here that returns a bare tensor.
* :class:`DepthEstimationHead` -- one depth map, plus the tensor it was
  scaled from (progressive decoder) or the last decoder features (DPT).
* :class:`ClassificationHead` -- one label for the whole image. Returns
  pre-softmax ``'logits'`` and their softmax under ``'probabilities'``.
* :class:`InstanceSegmentationHead` -- detection outputs plus per-instance
  mask logits.
* :class:`EnhancementHead` -- a restored or upscaled image.
* :class:`MultiTaskHead` -- several heads behind one layer. It does not
  inherit from :class:`BaseVisionHead`.

No head applies an activation to a class score. The class map, the instance
masks and the detection scores are logits; compile with ``from_logits=True``
or apply ``softmax``/``sigmoid`` yourself. The one output squashed inside a
head is the progressive :class:`DepthEstimationHead`'s sigmoid, a range map
that the depth is scaled from, not a class score. Its DPT decoder applies
``dpt_output_activation`` instead (``'linear'`` by default).

Helpers
-------
* :func:`create_vision_head` -- task type to head. It handles 10 of the 37
  ``VisionTaskType`` members and raises ``ValueError`` for the other 27.
* :func:`create_enhancement_head` -- build an :class:`EnhancementHead`, with
  a default scale factor for super-resolution.
* :func:`create_multi_task_head` -- build a :class:`MultiTaskHead` from any
  of three configuration formats.
* :class:`HeadConfiguration` -- two keyword-argument presets per task type:
  default and efficient.
* :func:`merge_head_config_overrides` -- merge a model's
  ``head_config_overrides`` into its head config, refusing the keys the model
  owns (``name``, ``num_classes``, ``task_type``).

The task types and the configuration objects live in
``vision/task_types.py``. The ``VisionTaskType`` docstring there lists the 10
dispatched members and names all 27 that raise. Read it rather than deriving
a second split here.

Example
-------
>>> from dl_techniques.layers.heads.vision import create_vision_head
>>> head = create_vision_head('classification', num_classes=10)
>>> out = head(features)
>>> sorted(out)
['logits', 'probabilities']

``out['logits']`` is the linear classifier output. ``out['probabilities']``
is ``softmax(out['logits'])``. Compile a model that returns the logits with
``from_logits=True``.
"""

import keras
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any, Literal, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...activations import ActivationType
from ...conv_blocks.conv_block import ConvBlock
from ...dense_block import DenseBlock
from ...ffn.factory import create_ffn_layer, FFNType
from ...attention import AttentionType
from ...attention.factory import (
    create_attention_layer,
    assemble_attention_config,
)
from ...norms import create_normalization_layer, NormalizationType
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.tensors import is_power_of_two, is_single_shape
from .task_types import VisionTaskType, TaskConfiguration
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Common-layer flag refusal
# ---------------------------------------------------------------------

def _refuse_common_layers_call_never_runs(
        kwargs: Dict[str, Any],
        head_name: str,
        flags: Tuple[str, ...]
) -> None:
    """
    Default each common-layer argument a head's ``call`` never applies to
    off (``False`` or ``0.0``), and refuse a value that would turn it on.

    Interface contract, shared by :class:`ClassificationHead` (``use_ffn``)
    and :class:`EnhancementHead` (``use_attention``, ``use_ffn``,
    ``dropout_rate``):

    * ``kwargs`` is the head's ``**kwargs`` before they reach
      :class:`BaseVisionHead`. Each name in ``flags`` that is absent is set to
      ``False`` in place (``0.0`` for ``dropout_rate``), so the base class
      builds and applies nothing for it.
    * A name passed with a truthy value (``True``, or a rate above 0) raises
      ``ValueError`` naming the head and the argument. Nothing is built then.
    * Returns ``None``.

    Call it BEFORE ``super().__init__``: the base class builds the common
    layers inside its own ``__init__``.

    :param kwargs: The head's keyword arguments, mutated in place.
    :type kwargs: Dict[str, Any]
    :param head_name: Class name, for the error message.
    :type head_name: str
    :param flags: Argument names the head's ``call`` never applies.
    :type flags: Tuple[str, ...]
    :return: None.
    :rtype: None
    :raises ValueError: If any name in ``flags`` is passed as truthy.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-020
    # Refuse, do NOT accept-and-ignore: a stored flag that builds nothing is a
    # dead knob (it silently changed no architecture for every caller that
    # passed it), and a built-but-unapplied layer is weight no gradient
    # reaches. See decisions.md D-020.
    for flag in flags:
        off = 0.0 if flag == 'dropout_rate' else False
        value = kwargs.setdefault(flag, off)
        if value:
            raise ValueError(
                f"{head_name} never applies what {flag}={value!r} asks for: "
                f"its call() runs no such layer. Pass {flag}={off!r} or leave "
                f"it unset."
            )


def _validated_block_dims(
        block_dims: Optional[Sequence[int]]
) -> Optional[Tuple[int, ...]]:
    """
    Check a head's ``block_dims`` argument and freeze it to a tuple.

    Interface contract, shared by :class:`ClassificationHead` and
    :class:`EnhancementHead`, whose ``block_dims`` means "the width of each
    hidden block, in order":

    * ``None`` is returned unchanged; each head resolves it to its own
      default stack.
    * Any other value is returned as a tuple. ``()`` is valid and means no
      hidden block.
    * An entry that is not a positive ``int`` (``bool`` included, since it is
      an ``int`` subclass) raises ``ValueError`` naming the whole argument.

    :param block_dims: The caller's ``block_dims``.
    :type block_dims: Optional[Sequence[int]]
    :return: ``None``, or the widths as a tuple.
    :rtype: Optional[Tuple[int, ...]]
    :raises ValueError: If an entry is not a positive integer.
    """
    if block_dims is None:
        return None
    block_dims = tuple(block_dims)
    bad = [d for d in block_dims
           if isinstance(d, bool) or not isinstance(d, int) or d <= 0]
    if bad:
        raise ValueError(
            f"block_dims must hold positive integers, got {block_dims}"
        )
    return block_dims


# ---------------------------------------------------------------------
# Base Head Class
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class BaseVisionHead(keras.layers.Layer):
    """
    Shared construction for the vision task heads.

    This class builds the layers the heads have in common and stores the
    configuration they read. It has no ``call``. Each subclass applies the
    stages it wants, then runs its own task layers.

    ``__init__`` calls ``_create_common_layers``, which builds ``attention``
    and ``ffn`` only when the matching flag asks for them. Every subclass
    ``call`` applies both when they exist, so the head carries no layer its
    forward pass skips.

    **Architecture Overview:**

    .. code-block:: text

        _create_common_layers builds these two:

        ┌─────────────────────────────────┐
        │ attention      if use_attention │
        │ ffn                  if use_ffn │
        └─────────────────────────────────┘

        _common_processed_shape predicts this much of a
        subclass forward pass, and no more:

        input_shape (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        the subclass task layers

    There is no common normalization or dropout stage. Each head normalizes
    and drops inside its own ``ConvBlock`` or ``DenseBlock``, and those read
    ``normalization_type`` and ``dropout_rate`` from this class.

    Two heads run less than this picture. :class:`ClassificationHead` runs
    ``attention`` only and refuses ``use_ffn=True``. :class:`EnhancementHead`
    runs neither, applies no dropout, and refuses ``use_attention=True``,
    ``use_ffn=True`` and ``dropout_rate > 0``. :class:`MultiTaskHead` does not inherit from this class at all. Read the
    subclass diagram, not this one, for what a given head runs.

    :param hidden_dim: Working width of the head. The sub-blocks and the FFN
        use it as their channel count.
    :type hidden_dim: int
    :param normalization_type: Which registered normalization to build, and
        the value handed to each sub-block.
    :type normalization_type: NormalizationType
    :param activation_type: Activation each sub-block uses.
    :type activation_type: ActivationType
    :param dropout_rate: Dropout rate handed to the sub-blocks, the FFN and
        the attention layer.
    :type dropout_rate: float
    :param use_attention: Whether to build an attention layer.
    :type use_attention: bool
    :param attention_type: Which registered attention type to build.
    :type attention_type: AttentionType
    :param use_ffn: Whether to build an FFN block. It defaults to ``True``
        here; :class:`ClassificationHead` and :class:`EnhancementHead`
        default it to ``False`` and refuse ``True``.
    :type use_ffn: bool
    :param ffn_type: Which registered FFN type to build.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: The FFN inner width is
        ``hidden_dim * ffn_expansion_factor``.
    :type ffn_expansion_factor: int
    :param kwargs: Additional arguments for the base Layer class.

    :ivar attention: Attention layer, built when ``use_attention`` is set.
    :vartype attention: keras.layers.Layer
    :ivar ffn: FFN block, built when ``use_ffn`` is set.
    :vartype ffn: keras.layers.Layer
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            normalization_type: NormalizationType = 'layer_norm',
            activation_type: ActivationType = 'gelu',
            dropout_rate: float = 0.1,
            use_attention: bool = False,
            attention_type: AttentionType = 'multi_head',
            use_ffn: bool = True,
            ffn_type: FFNType = 'mlp',
            ffn_expansion_factor: int = 4,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration and build the common layers.

        See the class docstring for what each argument means and which heads
        ignore it.

        :param hidden_dim: Working width of the head.
        :type hidden_dim: int
        :param normalization_type: Which registered normalization to build.
        :type normalization_type: NormalizationType
        :param activation_type: Activation the sub-blocks use.
        :type activation_type: ActivationType
        :param dropout_rate: Dropout rate handed to the sub-blocks.
        :type dropout_rate: float
        :param use_attention: Whether to build an attention layer.
        :type use_attention: bool
        :param attention_type: Which registered attention type to build.
        :type attention_type: AttentionType
        :param use_ffn: Whether to build an FFN block.
        :type use_ffn: bool
        :param ffn_type: Which registered FFN type to build.
        :type ffn_type: FFNType
        :param ffn_expansion_factor: FFN inner-width multiplier.
        :type ffn_expansion_factor: int
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        # Store configuration
        self.hidden_dim = hidden_dim
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.use_ffn = use_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor

        # Create common layers
        self._create_common_layers()

    def _create_common_layers(self) -> None:
        """
        Build the common layers the head's ``call`` applies.

        ``attention`` is built when ``use_attention`` is set and ``ffn`` when
        ``use_ffn`` is set; nothing else. The attention branch lives in
        ``_create_attention_layer``.

        Called from ``__init__``, following the package rule that layers are
        created in ``__init__``.

        :return: None.
        :rtype: None
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-020
        # Do NOT build a common norm or Dropout here again: no head's call()
        # applied either (measured per head at step 6), the norm carried
        # weights no gradient reached, and creating a Dropout draws a default
        # seed that shifts every later initializer. See decisions.md D-020.

        # Optional attention mechanism
        self._create_attention_layer()

        # Optional FFN block
        if self.use_ffn:
            if self.ffn_type == 'swiglu':
                self.ffn = create_ffn_layer(
                    'swiglu',
                    output_dim=self.hidden_dim,
                    ffn_expansion_factor=self.ffn_expansion_factor,
                    dropout_rate=self.dropout_rate,
                    name=f'{self.name}_ffn'
                )
            else:
                self.ffn = create_ffn_layer(
                    self.ffn_type,
                    hidden_dim=self.hidden_dim * self.ffn_expansion_factor,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    name=f'{self.name}_ffn'
                )

    def _create_attention_layer(self) -> None:
        """
        Build ``attention`` when ``use_attention`` is set, else nothing.

        The one place the attention branch is written;
        ``_create_common_layers`` calls it.

        The attention branch has three cases. ``'multi_head'`` and ``'cbam'``
        get their own argument sets. Every other type goes through
        ``assemble_attention_config``, which drops any keyword the type does
        not declare. ``create_attention_layer`` itself raises ``ValueError``
        on such a keyword. So passing ``dim`` to a type that has no such
        parameter would fail construction outright.

        :return: None.
        :rtype: None
        """
        if self.use_attention:
            if self.attention_type == 'multi_head':
                self.attention = create_attention_layer(
                    'multi_head',
                    dim=self.hidden_dim,
                    num_heads=8,
                    dropout_rate=self.dropout_rate,
                    name=f'{self.name}_attention'
                )
            elif self.attention_type == 'cbam':
                self.attention = create_attention_layer(
                    'cbam',
                    channels=self.hidden_dim,
                    ratio=8,
                    name=f'{self.name}_cbam'
                )
            else:
                # DECISION plan-2026-08-17T183311-79c63e38/D-023
                # 11 of the 33 registered attention types do not declare `dim`:
                # 'cbam' (handled above), 'channel', 'spatial', the four
                # 'tripseN', 'capsule_routing', 'fnet', 'hopfield', 'non_local'.
                # Do NOT pass `dim` unconditionally. See decisions.md D-023.
                self.attention = create_attention_layer(
                    self.attention_type,
                    name=f'{self.name}_attention',
                    **assemble_attention_config(
                        self.attention_type, {'dim': self.hidden_dim}
                    )
                )

    def _common_processed_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Report the shape after the attention and FFN stages.

        This mirrors the ``if self.use_attention: ... if self.use_ffn: ...``
        block that most subclass ``call`` methods run before their task
        layers. Subclass ``build`` methods use it to size what comes next.

        :class:`ClassificationHead` and :class:`EnhancementHead` do not use
        this method at all.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape after the optional attention and FFN stages.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(input_shape)
        if self.use_attention:
            shape = tuple(self.attention.compute_output_shape(shape))
        if self.use_ffn:
            shape = tuple(self.ffn.compute_output_shape(shape))
        return shape

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the common layers that exist.

        Each is built on the raw input shape, because each stage is
        shape-preserving in the width the head runs at.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        # The common layers each consume the raw input feature map.
        if self.use_attention:
            self.attention.build(input_shape)
        if self.use_ffn:
            self.ffn.build(input_shape)

        super().build(input_shape)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Report the output shape.

        The base head transforms nothing, so this returns the input shape
        unchanged. Every concrete subclass overrides it.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying the nine constructor arguments, on top
            of the base Layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'dropout_rate': self.dropout_rate,
            'use_attention': self.use_attention,
            'attention_type': self.attention_type,
            'use_ffn': self.use_ffn,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor
        })
        return config


# ---------------------------------------------------------------------
# Detection Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class DetectionHead(BaseVisionHead):
    """
    Class scores and box offsets, one set per anchor per location.

    Two branches run on the same common-processed features. Each is a
    ``ConvBlock`` followed by a 1x1 conv. Neither branch changes the spatial
    dimensions, because every conv here is stride 1 with ``'same'`` padding.

    Neither output conv applies an activation. The scores are raw logits and
    the offsets are raw values, so a loss that expects logits is the right
    one here.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)  or  (B, N, D) with input_format="sequence"
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼ x, common-processed
                 ┌───────┴────────────┐
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ cls_conv         │ │ reg_conv         │
        │ ConvBlock 3x3    │ │ ConvBlock 3x3    │
        └────────┬─────────┘ └────────┬─────────┘
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ cls_head         │ │ reg_head         │
        │ Conv2D 1x1       │ │ Conv2D 1x1       │
        └──────────────────┘ └──────────────────┘
         'classifications'      'regressions'

        classifications: (B, H, W, num_anchors * num_classes)
        regressions:     (B, H, W, num_anchors * bbox_dims)

    :class:`InstanceSegmentationHead` holds one of these as a sub-layer.
    :func:`create_vision_head` also returns this class for keypoint
    detection, unchanged.

    Input shape:
        ``(batch, height, width, channels)`` for spatial format, or
        ``(batch, seq_len, dim)`` for sequence format (ViT patch tokens).

    Output shape:
        ``{'classifications': (batch, height, width, num_anchors *
        num_classes), 'regressions': (batch, height, width, num_anchors *
        bbox_dims)}``.

    :param num_classes: Number of object classes.
    :type num_classes: int
    :param num_anchors: Number of anchor boxes per location.
    :type num_anchors: int
    :param bbox_dims: Values per box. 4 for the usual corner or centre form.
    :type bbox_dims: int
    :param input_format: Input format, ``"spatial"`` for (B,H,W,C) or
        ``"sequence"`` for (B,N,D) ViT patch tokens. Defaults to ``"spatial"``.
    :type input_format: Literal["spatial", "sequence"]
    :param patch_grid_size: Tuple ``(H, W)`` of patch grid dimensions for
        ``input_format="sequence"``. Required when using sequence format.
    :type patch_grid_size: Optional[Tuple[int, int]]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar cls_conv: Classification-branch ``ConvBlock``.
    :vartype cls_conv: ConvBlock
    :ivar cls_head: Classification output conv.
    :vartype cls_head: keras.layers.Conv2D
    :ivar reg_conv: Regression-branch ``ConvBlock`` (if ``use_regression=True``).
    :vartype reg_conv: Optional[ConvBlock]
    :ivar reg_head: Regression output conv (if ``use_regression=True``).
    :vartype reg_head: Optional[keras.layers.Conv2D]
    """

    def __init__(
            self,
            num_classes: int,
            num_anchors: int = 9,
            bbox_dims: int = 4,
            input_format: Literal["spatial", "sequence"] = "spatial",
            patch_grid_size: Optional[Tuple[int, int]] = None,
            use_regression: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the two branches.

        :param num_classes: Number of object classes.
        :type num_classes: int
        :param num_anchors: Number of anchor boxes per location.
        :type num_anchors: int
        :param bbox_dims: Values per box.
        :type bbox_dims: int
        :param input_format: Input format, ``"spatial"`` or ``"sequence"``.
        :type input_format: Literal["spatial", "sequence"]
        :param patch_grid_size: Patch grid (H, W) for sequence input format.
        :type patch_grid_size: Optional[Tuple[int, int]]
        :param use_regression: Whether to create the regression branch.
            For keypoint detection and other tasks that only need classification,
            set to ``False`` to avoid dead weights. Defaults to ``True``.
        :type use_regression: bool
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.bbox_dims = bbox_dims
        self.input_format = input_format
        self.patch_grid_size = patch_grid_size
        self.use_regression = use_regression

        if self.input_format == "sequence" and self.patch_grid_size is None:
            raise ValueError("patch_grid_size is required when input_format='sequence'")

        # Create detection-specific layers
        self._create_detection_layers()

    def _create_detection_layers(self) -> None:
        """
        Build the classification and regression branches.

        Each branch is a 3x3 ``ConvBlock`` at ``hidden_dim`` channels, then a
        1x1 conv with no activation.

        :return: None.
        :rtype: None
        """

        # Classification branch
        self.cls_conv = ConvBlock(
            filters=self.hidden_dim,
            kernel_size=3,
            normalization_type=self.normalization_type,
            activation_type=self.activation_type,
            dropout_rate=self.dropout_rate,
            name="cls_conv"
        )

        self.cls_head = layers.Conv2D(
            filters=self.num_anchors * self.num_classes,
            kernel_size=1,
            padding='same',
            name='cls_head'
        )

        # Regression branch (optional)
        if self.use_regression:
            self.reg_conv = ConvBlock(
                filters=self.hidden_dim,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                dropout_rate=self.dropout_rate,
                name="reg_conv"
            )

            self.reg_head = layers.Conv2D(
                filters=self.num_anchors * self.bbox_dims,
                kernel_size=1,
                padding='same',
                name='reg_head'
            )
        else:
            self.reg_conv = None
            self.reg_head = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build both branches on the common-processed shape.

        For sequence input, attention and ffn are applied on the sequence
        (before reshaping), so they are built on the sequence shape.
        The conv branches are built on the spatial shape after reshaping.

        :param input_shape: Input feature-map shape (spatial) or sequence shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        # For sequence input, build attention/ffn on sequence shape,
        # and conv branches on spatial shape
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            batch, seq_len, dim = input_shape
            h, w = self.patch_grid_size
            spatial_shape = (batch, h, w, dim)
            
            # Build common layers (attention, ffn) on sequence shape
            if self.use_attention:
                self.attention.build(input_shape)
            if self.use_ffn:
                self.ffn.build(input_shape)
            
            # Build conv branches on spatial shape
            feature_shape = self._common_processed_shape(spatial_shape)
        else:
            # Build common layers on spatial shape
            if self.use_attention:
                self.attention.build(input_shape)
            if self.use_ffn:
                self.ffn.build(input_shape)
            feature_shape = self._common_processed_shape(input_shape)

        self.cls_conv.build(feature_shape)
        self.cls_head.build(self.cls_conv.compute_output_shape(feature_shape))

        if self.use_regression:
            self.reg_conv.build(feature_shape)
            self.reg_head.build(self.reg_conv.compute_output_shape(feature_shape))

        # attention and ffn are built above, on the sequence or spatial shape
        # call() applies them to, so only the Layer bookkeeping remains.
        keras.layers.Layer.build(self, input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stages, then both branches.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)`` for spatial format, or ``(batch, seq_len, dim)``
            for sequence format (ViT patch tokens).
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'classifications'`` and ``'regressions'``.
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # For sequence input, apply common processing (attention, ffn) on the
        # sequence BEFORE reshaping to spatial, since attention layers expect
        # 3D input (batch, seq_len, dim).
        if self.input_format == "sequence":
            # Apply common processing on sequence
            if self.use_attention:
                x = self.attention(x, training=training)
            if self.use_ffn:
                x = self.ffn(x, training=training)

            # Now reshape to spatial for conv branches
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            batch_size = ops.shape(x)[0]
            h, w = self.patch_grid_size
            x = ops.reshape(x, (batch_size, h, w, -1))
        else:
            # Spatial input: apply common processing on spatial features
            if self.use_attention:
                x = self.attention(x, training=training)
            if self.use_ffn:
                x = self.ffn(x, training=training)

        # Classification branch
        cls_features = self.cls_conv(x, training=training)
        cls_output = self.cls_head(cls_features)

        # Regression branch (optional)
        if self.use_regression:
            reg_features = self.reg_conv(x, training=training)
            reg_output = self.reg_head(reg_features)
        else:
            reg_output = None

        if self.use_regression:
            return {
                'classifications': cls_output,
                'regressions': reg_output
            }
        else:
            return {
                'classifications': cls_output
            }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the output shapes.

        Spatial dimensions are preserved. Only the channel count changes.

        :param input_shape: Shape ``(batch, height, width, channels)`` for spatial,
            or ``(batch, seq_len, dim)`` for sequence format.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'classifications'`` and optionally ``'regressions'``
            shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        if self.input_format == "sequence":
            batch = input_shape[0]
            h, w = self.patch_grid_size
        else:
            batch, height, width = input_shape[0], input_shape[1], input_shape[2]
            h, w = height, width
        output_shapes = {
            'classifications': (batch, h, w, self.num_anchors * self.num_classes)
        }
        if self.use_regression:
            output_shapes['regressions'] = (batch, h, w, self.num_anchors * self.bbox_dims)
        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``num_anchors``,
            ``bbox_dims``, ``input_format``, ``patch_grid_size`` and ``use_regression``,
            on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_anchors': self.num_anchors,
            'bbox_dims': self.bbox_dims,
            'input_format': self.input_format,
            'patch_grid_size': self.patch_grid_size,
            'use_regression': self.use_regression
        })
        return config


# ---------------------------------------------------------------------
# Segmentation Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class SegmentationHead(BaseVisionHead):
    """
    One map of per-pixel class logits, upsampled towards the input resolution.

    Three ``ConvBlock`` stages refine the features, then transposed convs
    double the resolution once per stage, then a linear 1x1 conv emits one
    channel per class. The output is pre-softmax: compile a per-pixel
    cross-entropy with ``from_logits=True``, or apply
    ``keras.ops.softmax(out, axis=-1)`` for class probabilities.

    This head accepts a single feature map or a list of them. With a list and
    ``use_skip_connections``, the last map is the one refined and the rest
    become skips in reverse order. One skip is concatenated after each
    refine block, while skips remain.

    Two skip channel modes are supported:
    - ``"auto"`` (default): refine blocks halve channels each step
      (``hidden_dim``, ``hidden_dim//2``, ``hidden_dim//4``), skip
      concatenation dynamically widens the tensor.
    - ``"explicit"``: caller provides ``explicit_skip_channels`` list
      matching the encoder feature channels (e.g., ``[32, 64, 128, 256]``
      for AccUNet). Refine block output channels are derived from the
      corresponding skip channels.

    This is the only head in the module that returns a bare tensor. Every
    other one returns a dict.

    **Architecture Overview (auto mode):**

    .. code-block:: text

        inputs: (B, H, W, C), or a list of feature maps
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ list input: x = the last map,   │
        │ skips = the rest, reversed      │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ refine_blocks: 3 x ConvBlock    │
        │ concat skips[i] after block i   │
        └────────────────┬────────────────┘
                         ▼ channels: hidden_dim, then halved
        ┌─────────────────────────────────┐
        │ upsample_layers                 │
        │ Conv2DTranspose stride 2        │
        └────────────────┬────────────────┘
                         ▼ log2(upsampling_factor) of them
        ┌─────────────────────────────────┐
        │ seg_head: Conv2D 1x1     linear │
        └────────────────┬────────────────┘
                         ▼
        (B, H*s, W*s, num_classes), s = 2 ** the count above

    The refine blocks halve their channel count as they go: ``hidden_dim``,
    then ``hidden_dim // 2``, then ``hidden_dim // 4``. Every upsample layer
    uses ``hidden_dim // 8``, one halving further still. Concatenating a skip
    widens the tensor again, so the built shapes follow ``call`` rather than
    the block widths alone.

    The output resolution is ``upsampling_factor`` times the input: one
    stride-2 transposed conv per doubling, ``log2(upsampling_factor)`` of
    them. With the default ``upsampling_factor=4`` that is 2 layers. A factor
    that is not a power of two raises ``ValueError``.

    Input shape:
        ``(batch, height, width, channels)``, or a list of such shapes.

    Output shape:
        ``(batch, out_height, out_width, num_classes)``.

    :param num_classes: Number of segmentation classes.
    :type num_classes: int
    :param upsampling_factor: Output resolution over input resolution, a
        power of two. ``log2`` of it stride-2 transposed convs are built.
    :type upsampling_factor: int
    :param use_skip_connections: Concatenate list inputs as skips. Ignored
        when the input is a single tensor.
    :type use_skip_connections: bool
    :param skip_channel_mode: ``"auto"`` (default) or ``"explicit"``. In
        explicit mode, ``explicit_skip_channels`` must be provided.
    :type skip_channel_mode: Literal["auto", "explicit"]
    :param explicit_skip_channels: List of channel counts for each skip
        connection, ordered from deepest to shallowest (matching the reversed
        encoder features). Required when ``skip_channel_mode="explicit"``.
    :type explicit_skip_channels: Optional[List[int]]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar refine_blocks: The three refinement ``ConvBlock`` layers.
    :vartype refine_blocks: List[ConvBlock]
    :ivar upsample_layers: The stride-2 ``Conv2DTranspose`` layers.
    :vartype upsample_layers: List[keras.layers.Conv2DTranspose]
    :ivar seg_head: Output ``Conv2D(num_classes)``, linear: it emits the
        class logits.
    :vartype seg_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            num_classes: int,
            upsampling_factor: int = 4,
            use_skip_connections: bool = True,
            skip_channel_mode: Literal["auto", "explicit"] = "auto",
            explicit_skip_channels: Optional[List[int]] = None,
            input_format: Literal["spatial", "sequence"] = "spatial",
            patch_grid_size: Optional[Tuple[int, int]] = None,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the segmentation layers.

        :param num_classes: Number of segmentation classes.
        :type num_classes: int
        :param upsampling_factor: Output over input resolution, a power of
            two.
        :type upsampling_factor: int
        :param use_skip_connections: Concatenate list inputs as skips.
        :type use_skip_connections: bool
        :param skip_channel_mode: ``"auto"`` or ``"explicit"``.
        :type skip_channel_mode: Literal["auto", "explicit"]
        :param explicit_skip_channels: Channel counts for explicit skip mode.
        :type explicit_skip_channels: Optional[List[int]]
        :param input_format: Input format, ``"spatial"`` for (B,H,W,C) or
            ``"sequence"`` for (B,N,D) ViT patch tokens. Defaults to ``"spatial"``.
        :type input_format: Literal["spatial", "sequence"]
        :param patch_grid_size: Tuple ``(H, W)`` of patch grid dimensions for
            ``input_format="sequence"``. Required when using sequence format.
        :type patch_grid_size: Optional[Tuple[int, int]]
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.upsampling_factor = upsampling_factor
        self.use_skip_connections = use_skip_connections
        self.skip_channel_mode = skip_channel_mode
        self.explicit_skip_channels = explicit_skip_channels
        self.input_format = input_format
        self.patch_grid_size = patch_grid_size

        if self.input_format == "sequence" and self.patch_grid_size is None:
            raise ValueError("patch_grid_size is required when input_format='sequence'")

        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is None:
            raise ValueError("explicit_skip_channels is required when skip_channel_mode='explicit'")

        # DECISION plan-2026-10-14T062616-81938e81/D-028
        # One stride-2 transposed conv per doubling: log2(factor) of them. Do
        # NOT go back to int(sqrt(factor)): it equals log2 only at 2, 4, 16
        # and 32, so factor 8 gave 4x and 64 gave 256x, and FFTNet (factor =
        # patch_size) returned a map that was not image-sized for patch 8.
        # See decisions.md D-028.
        if (isinstance(upsampling_factor, bool)
                or not isinstance(upsampling_factor, int)
                or not is_power_of_two(upsampling_factor)):
            raise ValueError(
                f"upsampling_factor must be a power of two (1, 2, 4, 8, ...), "
                f"got {upsampling_factor!r}"
            )
        self._num_upsamples = upsampling_factor.bit_length() - 1

        self._create_segmentation_layers()

    def _create_segmentation_layers(self) -> None:
        """
        Build the refine stack, the upsample stack and the output conv.

        In "auto" mode, refine channel counts halve each step.
        In "explicit" mode, refine block output channels match the
        corresponding skip channels (from explicit_skip_channels).

        :return: None.
        :rtype: None
        """

        # Feature refinement blocks
        self.refine_blocks = []

        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is not None:
            # Explicit mode: refine blocks output channels to match skip channels
            # We have 3 refine blocks, explicit_skip_channels has 4 entries (L0-L3)
            # Refine block i output should match skip_channels[-(i+2)] for concatenation
            skip_ch = self.explicit_skip_channels
            for i in range(3):
                # Output channels = skip_channels[-(i+2)] // 2 (half before concat)
                out_channels = skip_ch[-(i+2)] // 2
                self.refine_blocks.append(
                    ConvBlock(
                        filters=out_channels,
                        kernel_size=3,
                        normalization_type=self.normalization_type,
                        activation_type=self.activation_type,
                        dropout_rate=self.dropout_rate,
                        name=f'refine_block_{i}'
                    )
                )
        else:
            # Auto mode: halve channels each step
            channels = self.hidden_dim
            for i in range(3):
                self.refine_blocks.append(
                    ConvBlock(
                        filters=channels,
                        kernel_size=3,
                        normalization_type=self.normalization_type,
                        activation_type=self.activation_type,
                        dropout_rate=self.dropout_rate,
                        name=f'refine_block_{i}'
                    )
                )
                channels = channels // 2

        # Upsampling layers
        self.upsample_layers = []
        for i in range(self._num_upsamples):
            # Determine filter count for upsample layer
            if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is not None:
                # After 3 refine blocks, channels = skip_ch[-3] // 2 = skip_ch[0] // 2
                # Actually let's compute it properly: after 3 refinements in explicit mode
                # refine_0 out = skip_ch[1]//2, after concat with skip_ch[0] -> skip_ch[1]//2 + skip_ch[0]
                # refine_1 out = skip_ch[2]//2, after concat with skip_ch[1] -> skip_ch[2]//2 + skip_ch[1]
                # refine_2 out = skip_ch[3]//2, after concat with skip_ch[2] -> skip_ch[3]//2 + skip_ch[2]
                # This is complex, so we'll let build() handle the shapes
                filters = None  # Will be determined in build
            else:
                # Auto mode: after 3 halvings from hidden_dim
                filters = self.hidden_dim // 8
            
            self.upsample_layers.append(
                layers.Conv2DTranspose(
                    filters=filters if filters else self.hidden_dim // 8,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    activation=self.activation_type,
                    name=f'upsample_{i}'
                )
            )

        # Final segmentation layer.
        # DECISION plan-2026-10-14T062616-81938e81/D-003
        # Linear: the class map is logits. Do NOT put activation='softmax'
        # back: a from_logits=True loss would double-softmax, and a
        # from_logits=False one hides the probabilities inside the head where
        # no consumer can choose. Range maps (the depth sigmoid) are the only
        # squashed outputs in this module. See decisions.md D-003.
        self.seg_head = layers.Conv2D(
            filters=self.num_classes,
            kernel_size=1,
            padding='same',
            name='seg_head'
        )

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build every sub-layer on the shape ``call`` will hand it.

        This repeats the skip-concatenation arithmetic of ``call``, so a
        multi-scale input builds the same widths it will run with. The common
        layers are built on the single highest-level map.

        In explicit mode, validates that skip shapes match explicit_skip_channels.

        For sequence input format, the input is reshaped to spatial in call(),
        so we build the layers for the spatial shape.

        :param input_shape: One feature-map shape, or a list of shapes for a
            multi-scale input.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: None.
        :rtype: None
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-027
        # Decide "list of shapes" by the ELEMENT type (is_single_shape), NOT
        # by isinstance(input_shape, list): load_model rebuilds this layer
        # from a JSON build_config, where ONE shape arrives as a list of ints,
        # and the list test then read it as several one-int shapes, so every
        # saved model holding this head failed to load. See decisions.md D-027.
        shape_list = not is_single_shape(input_shape)

        # For sequence input, compute the spatial shape after reshaping
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            if shape_list:
                spatial_shapes = []
                for shape in input_shape:
                    batch, seq_len, dim = shape
                    h, w = self.patch_grid_size
                    spatial_shapes.append((batch, h, w, dim))
                input_shape = spatial_shapes
            else:
                batch, seq_len, dim = input_shape
                h, w = self.patch_grid_size
                input_shape = (batch, h, w, dim)

        if shape_list and self.use_skip_connections:
            base_shape = input_shape[-1]
            skip_shapes = input_shape[:-1][::-1]
        else:
            base_shape = input_shape[-1] if shape_list else input_shape
            skip_shapes = []

        # In explicit mode, validate skip channels
        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is not None:
            if len(skip_shapes) != len(self.explicit_skip_channels):
                raise ValueError(
                    f"explicit_skip_channels has {len(self.explicit_skip_channels)} entries "
                    f"but {len(skip_shapes)} skip inputs provided"
                )
            for i, (skip_shape, exp_ch) in enumerate(zip(skip_shapes, self.explicit_skip_channels)):
                if skip_shape[-1] != exp_ch:
                    raise ValueError(
                        f"Skip {i} channels {skip_shape[-1]} != explicit_skip_channels[{i}]={exp_ch}"
                    )

        shape = self._common_processed_shape(base_shape)

        for i, refine_block in enumerate(self.refine_blocks):
            refine_block.build(shape)
            shape = tuple(refine_block.compute_output_shape(shape))
            if self.use_skip_connections and i < len(skip_shapes):
                skip_channels = skip_shapes[i][-1]
                merged = (
                    shape[-1] + skip_channels
                    if shape[-1] is not None and skip_channels is not None
                    else None
                )
                shape = shape[:-1] + (merged,)

        # Rebuild upsample layers with correct filters if needed
        if self.skip_channel_mode == "explicit" and self.explicit_skip_channels is not None:
            # After all refine blocks, shape[-1] is the channel count
            final_channels = shape[-1]
            for i, upsample_layer in enumerate(self.upsample_layers):
                upsample_layer.filters = final_channels
                upsample_layer.build(shape)
                shape = tuple(upsample_layer.compute_output_shape(shape))
        else:
            for upsample_layer in self.upsample_layers:
                upsample_layer.build(shape)
                shape = tuple(upsample_layer.compute_output_shape(shape))

        self.seg_head.build(shape)

        # Common layers operate on the single highest-level feature map.
        super().build(base_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Refine, concatenate skips, upsample, then classify each pixel.

        :param inputs: One feature map, or a list of them with the
            highest-level map last. For sequence format, input is (B, N, D).
        :type inputs: Union[keras.KerasTensor, List[keras.KerasTensor]]
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Per-pixel class logits, pre-softmax. This head returns no
            dict.
        :rtype: keras.KerasTensor
        """

        # Handle sequence input format (ViT patch tokens)
        x = inputs
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            if isinstance(x, list):
                # Convert each element in the list
                x = [self._reshape_sequence_to_spatial(xi) for xi in x]
            else:
                x = self._reshape_sequence_to_spatial(x)

        # Handle multi-scale inputs if skip connections are used
        if isinstance(x, list) and self.use_skip_connections:
            # The last entry is the highest-level map. The rest are skips,
            # reversed so the deepest one is consumed first.
            x_main = x[-1]
            skip_features = x[:-1][::-1]
        else:
            x_main = x if not isinstance(x, list) else x[-1]
            skip_features = []

        # Apply common processing
        if self.use_attention:
            x_main = self.attention(x_main, training=training)
        if self.use_ffn:
            x_main = self.ffn(x_main, training=training)

        # Refinement and upsampling
        for i, refine_block in enumerate(self.refine_blocks):
            x_main = refine_block(x_main, training=training)

            # Add skip connections if available
            if self.use_skip_connections and i < len(skip_features):
                x_main = ops.concatenate([x_main, skip_features[i]], axis=-1)

        # Upsample to original resolution
        for upsample_layer in self.upsample_layers:
            x_main = upsample_layer(x_main)

        # Final segmentation output
        seg_output = self.seg_head(x_main)

        return seg_output

    def _reshape_sequence_to_spatial(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape sequence (B, N, D) to spatial (B, H, W, D)."""
        batch_size = ops.shape(x)[0]
        h, w = self.patch_grid_size
        return ops.reshape(x, (batch_size, h, w, -1))

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """
        Report the output shape.

        Each of the ``log2(upsampling_factor)`` transposed convs doubles
        height and width. The output conv emits ``num_classes``
        channels. A list input is measured by its last entry.

        For sequence input format, the input shape (B, N, D) is reshaped
        to spatial (B, H, W, D) using patch_grid_size before computing.

        :param input_shape: One feature-map shape, or a list of shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        :return: Shape ``(batch, out_height, out_width, num_classes)``.
        :rtype: Tuple[Optional[int], ...]
        """
        base_shape = (
            input_shape if is_single_shape(input_shape) else input_shape[-1]
        )
        batch = base_shape[0]

        # Handle sequence input format
        if self.input_format == "sequence":
            if self.patch_grid_size is None:
                raise ValueError("patch_grid_size must be set for sequence input format")
            # base_shape is (B, N, D) -> reshape to (B, H, W, D)
            height, width = self.patch_grid_size
        else:
            height, width = base_shape[1], base_shape[2]

        scale = self.upsampling_factor
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        return (batch, out_height, out_width, self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``upsampling_factor``,
            ``use_skip_connections``, ``skip_channel_mode``,
            ``explicit_skip_channels``, ``input_format``, and ``patch_grid_size``,
            on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'upsampling_factor': self.upsampling_factor,
            'use_skip_connections': self.use_skip_connections,
            'skip_channel_mode': self.skip_channel_mode,
            'explicit_skip_channels': self.explicit_skip_channels,
            'input_format': self.input_format,
            'patch_grid_size': self.patch_grid_size
        })
        return config


# ---------------------------------------------------------------------
# Depth Estimation Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class DepthEstimationHead(BaseVisionHead):
    """
    A depth map, at eight times the input resolution by default.

    With the default ``decoder_style='progressive'``, three stages each
    refine the features and then double height and width, so the output is
    ``8H`` by ``8W``. A sigmoid conv produces a normalized map, which is then
    scaled into ``[min_depth, max_depth]``. The diagram below is this mode.
    ``decoder_style='dpt'`` is a different decoder, described after it.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ 3 x (depth_conv_blocks[i], then │
        │ depth_upsamples[i] stride 2)    │
        └────────────────┬────────────────┘
                         ▼ 8x spatial, channels halved 3x
        ┌─────────────────────────────────┐
        │ depth_head: Conv2D 3x3  sigmoid │
        └────────────────┬────────────────┘
                         ▼ depth_normalized, in [0, 1]
                 ┌───────┴────────────┐
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ use_log_depth:   │ │ passed through   │
        │ exp(...) or a    │ │ with no change   │
        │ linear rescale   │ │                  │
        └────────┬─────────┘ └────────┬─────────┘
                 ▼                    ▼
              'depth'            'confidence'

        depth, confidence: (B, 8H, 8W, output_channels)

    ``'confidence'`` is ``depth_normalized`` itself, returned unchanged. It
    is the sigmoid output the depth was scaled from, not a second estimate
    and not an uncertainty. Treat it as a confidence only if you have checked
    that reading against your data.

    The scaling has two forms. With ``use_log_depth`` the map is read as a
    position in log space: ``exp(x * (log(max_depth) - log(min_depth)) +
    log(min_depth))``. Otherwise it is read linearly: ``x * (max_depth -
    min_depth) + min_depth``.

    :func:`create_vision_head` reuses this class for two other tasks. Surface
    normals get ``output_channels=3`` and optical flow gets
    ``output_channels=2``. Neither changes the scaling, so a flow vector
    comes out squeezed into the depth range.

    ``decoder_style='dpt'`` (the DepthAnything decoder) runs one
    ``Conv2D -> BatchNormalization -> activation`` stage per ``dpt_dims``
    entry, with a 2x bilinear upsample after each of the first
    ``log2(dpt_upsample_factor)`` stages, then ``dpt_output_conv``. Its
    ``'depth'`` is that conv's output under ``dpt_output_activation``
    (``'linear'`` by default, so no sigmoid), and ``min_depth``,
    ``max_depth`` and ``use_log_depth`` are stored but NOT applied.
    ``'confidence'`` is the last stage's features, ``dpt_dims[-1]`` channels
    wide, not a map shaped like ``'depth'``.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        Progressive: ``{'depth': (batch, 8 * height, 8 * width,
        output_channels), 'confidence': the same shape}``. DPT, with
        ``s = dpt_upsample_factor``: ``{'depth': (batch, s * height,
        s * width, output_channels), 'confidence': (batch, s * height,
        s * width, dpt_dims[-1])}``.

    :param output_channels: Channel count of the output map. 1 for depth.
    :type output_channels: int
    :param min_depth: Lower end of the output range.
    :type min_depth: float
    :param max_depth: Upper end of the output range.
    :type max_depth: float
    :param use_log_depth: Scale in log space instead of linearly.
    :type use_log_depth: bool
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar depth_conv_blocks: The three refinement ``ConvBlock`` s
        (progressive decoder).
    :vartype depth_conv_blocks: List[ConvBlock]
    :ivar depth_upsamples: The three stride-2 ``Conv2DTranspose`` layers,
        each applied after the ``ConvBlock`` of the same index.
    :vartype depth_upsamples: List[keras.layers.Conv2DTranspose]
    :ivar depth_head: Output ``Conv2D(output_channels)`` with sigmoid.
    :vartype depth_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            output_channels: int = 1,
            min_depth: float = 0.1,
            max_depth: float = 100.0,
            use_log_depth: bool = True,
            decoder_style: Literal["progressive", "dpt"] = "progressive",
            dpt_dims: Optional[List[int]] = None,
            dpt_use_bias: bool = False,
            dpt_activation: Union[str, callable] = "relu",
            dpt_output_activation: Union[str, callable] = "linear",
            dpt_upsample_factor: int = 16,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the depth layers.

        :param output_channels: Channel count of the output map.
        :type output_channels: int
        :param min_depth: Lower end of the output range.
        :type min_depth: float
        :param max_depth: Upper end of the output range.
        :type max_depth: float
        :param use_log_depth: Scale in log space instead of linearly.
        :type use_log_depth: bool
        :param decoder_style: ``"progressive"`` (default, 3x ConvBlock+Conv2DTranspose)
            or ``"dpt"`` (DPTDecoder-compatible: Conv2D+BN+UpSampling2D).
        :type decoder_style: Literal["progressive", "dpt"]
        :param dpt_dims: Channel dimensions per DPT decoder stage.
            Defaults to ``[256, 128, 64, 32]``.
        :type dpt_dims: Optional[List[int]]
        :param dpt_use_bias: Whether Conv2D layers use bias in DPT mode.
        :type dpt_use_bias: bool
        :param dpt_activation: Activation for DPT decoder stages.
        :type dpt_activation: Union[str, callable]
        :param dpt_output_activation: Activation for final DPT output conv.
        :type dpt_output_activation: Union[str, callable]
        :param dpt_upsample_factor: Total upsampling factor for DPT mode (power of 2).
        :type dpt_upsample_factor: int
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        # Set defaults for DepthEstimationHead before calling parent constructor
        kwargs.setdefault('use_attention', False)
        kwargs.setdefault('use_ffn', False)
        super().__init__(**kwargs)

        self.output_channels = output_channels
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_log_depth = use_log_depth
        self.decoder_style = decoder_style
        self.dpt_dims = dpt_dims
        self.dpt_use_bias = dpt_use_bias
        self.dpt_activation = dpt_activation
        self.dpt_output_activation = dpt_output_activation
        self.dpt_upsample_factor = dpt_upsample_factor

        if self.decoder_style not in ("progressive", "dpt"):
            raise ValueError(f"decoder_style must be 'progressive' or 'dpt', got {decoder_style}")

        self._create_depth_layers()

    def _create_depth_layers(self) -> None:
        """
        Build the refinement stack and the depth conv.

        Two architectures based on decoder_style:
        - "progressive": Three ``[ConvBlock, Conv2DTranspose]`` pairs.
        - "dpt": DPTDecoder-compatible conv+bn+upsample stages.

        :return: None.
        :rtype: None
        """

        if self.decoder_style == "progressive":
            # Progressive upsampling with refinement (original behavior).
            #
            # DECISION plan-2026-10-14T062616-81938e81/D-027
            # Two FLAT lists, paired by index. Do NOT nest them back into a
            # list of [conv, upsample] pairs: Keras does not track a layer
            # inside a nested list for saving, so a .keras round trip
            # silently restored none of these six kernels (measured: the
            # reloaded depth differed by up to 17.3 with no warning). See
            # decisions.md D-027.
            self.depth_conv_blocks = []
            self.depth_upsamples = []
            channels = self.hidden_dim

            for i in range(3):
                self.depth_conv_blocks.append(
                    ConvBlock(
                        filters=channels,
                        kernel_size=3,
                        normalization_type=self.normalization_type,
                        activation_type=self.activation_type,
                        name=f'depth_conv_{i}'
                    )
                )
                self.depth_upsamples.append(
                    layers.Conv2DTranspose(
                        filters=channels // 2,
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        name=f'depth_upsample_{i}'
                    )
                )
                channels = channels // 2

            # Depth prediction layer
            self.depth_head = layers.Conv2D(
                filters=self.output_channels,
                kernel_size=3,
                padding='same',
                # Sigmoid keeps the output in [0, 1]. call() then scales it into
                # the [min_depth, max_depth] range.
                activation='sigmoid',
                name='depth_head'
            )
        else:
            # DPTDecoder-compatible architecture
            from dl_techniques.utils.activation_serialization import deserialize_activation

            self.dpt_dims = self.dpt_dims if self.dpt_dims is not None else [256, 128, 64, 32]
            self.dpt_activation = deserialize_activation(self.dpt_activation)
            self.dpt_output_activation = deserialize_activation(self.dpt_output_activation)

            # Validate upsample_factor: must be a power of 2 and <= 2**len(dims).
            if self.dpt_upsample_factor < 1 or (self.dpt_upsample_factor & (self.dpt_upsample_factor - 1)) != 0:
                raise ValueError(
                    f"dpt_upsample_factor must be a positive power of 2, got {self.dpt_upsample_factor}"
                )
            # Number of 2x upsamples to insert.
            self._num_dpt_upsamples = 0
            uf = self.dpt_upsample_factor
            while uf > 1:
                uf //= 2
                self._num_dpt_upsamples += 1
            if self._num_dpt_upsamples > len(self.dpt_dims):
                raise ValueError(
                    f"dpt_upsample_factor=2**{self._num_dpt_upsamples} requires at least "
                    f"{self._num_dpt_upsamples} decoder stages, but len(dpt_dims)={len(self.dpt_dims)}"
                )

            # All sublayer dims are shape-independent, so create them here in
            # __init__ for stable layer tracking across (de)serialization.
            self.dpt_conv_layers: List[keras.layers.Conv2D] = []
            self.dpt_bn_layers: List[keras.layers.BatchNormalization] = []
            self.dpt_act_layers: List[keras.layers.Layer] = []
            self.dpt_up_layers: List[Optional[keras.layers.Layer]] = []

            # Create convolutional layers for each dimension
            for i, dim in enumerate(self.dpt_dims):
                # Convolutional layer
                conv = keras.layers.Conv2D(
                    filters=dim,
                    kernel_size=3,
                    padding='same',
                    kernel_initializer=self.kernel_initializer if hasattr(self, 'kernel_initializer') else 'he_normal',
                    kernel_regularizer=self.kernel_regularizer if hasattr(self, 'kernel_regularizer') else None,
                    use_bias=self.dpt_use_bias,
                    name=f'dpt_conv_{i}'
                )
                self.dpt_conv_layers.append(conv)

                # Batch normalization layer
                # Use epsilon=1e-5 to match torch reference (DPTDecoder default)
                bn = keras.layers.BatchNormalization(
                    epsilon=1e-5, name=f'dpt_bn_{i}'
                )
                self.dpt_bn_layers.append(bn)

                # Activation layer
                activation_layer = keras.layers.Activation(
                    self.dpt_activation,
                    name=f'dpt_act_{i}'
                )
                self.dpt_act_layers.append(activation_layer)

                # Upsample after the first _num_dpt_upsamples stages (one 2x per stage).
                if i < self._num_dpt_upsamples:
                    up = keras.layers.UpSampling2D(
                        size=(2, 2), interpolation='bilinear', name=f'dpt_upsample_{i}'
                    )
                else:
                    up = None
                self.dpt_up_layers.append(up)

            # Final output layer
            self.dpt_output_conv = keras.layers.Conv2D(
                filters=self.output_channels,
                kernel_size=3,
                padding='same',
                kernel_initializer=self.kernel_initializer if hasattr(self, 'kernel_initializer') else 'he_normal',
                kernel_regularizer=self.kernel_regularizer if hasattr(self, 'kernel_regularizer') else None,
                activation=self.dpt_output_activation,
                use_bias=True,
                name='dpt_output_conv'
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the depth stack on the common-processed shape.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        if self.decoder_style == "progressive":
            shape = self._common_processed_shape(input_shape)

            for conv_block, upsample in zip(self.depth_conv_blocks,
                                            self.depth_upsamples):
                conv_block.build(shape)
                shape = tuple(conv_block.compute_output_shape(shape))
                upsample.build(shape)
                shape = tuple(upsample.compute_output_shape(shape))

            self.depth_head.build(shape)
        else:
            # DPT style: build all dpt layers
            shape = self._common_processed_shape(input_shape)
            for conv, bn, act, up in zip(
                    self.dpt_conv_layers,
                    self.dpt_bn_layers,
                    self.dpt_act_layers,
                    self.dpt_up_layers,
            ):
                conv.build(shape)
                shape = conv.compute_output_shape(shape)
                bn.build(shape)
                act.build(shape)
                if up is not None:
                    shape = up.compute_output_shape(shape)
            self.dpt_output_conv.build(shape)

        # The common attention and ffn, on the raw input shape.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Refine, upsample, predict, then scale into the depth range.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'depth'`` and ``'confidence'``, where
            ``'confidence'`` is the unscaled sigmoid output (progressive)
            or the pre-activation output (dpt).
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply common processing
        if self.use_attention:
            x = self.attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)

        if self.decoder_style == "progressive":
            # Progressive refinement and upsampling
            for conv_block, upsample in zip(self.depth_conv_blocks,
                                            self.depth_upsamples):
                x = conv_block(x, training=training)
                x = upsample(x)

            # Predict normalized depth
            depth_normalized = self.depth_head(x)

            # Scale to actual depth range
            if self.use_log_depth:
                # Convert from log space
                log_min = ops.log(self.min_depth)
                log_max = ops.log(self.max_depth)
                depth = ops.exp(depth_normalized * (log_max - log_min) + log_min)
            else:
                # Linear scaling
                depth = depth_normalized * (self.max_depth - self.min_depth) + self.min_depth

            return {
                'depth': depth,
                # This is the sigmoid output the depth was scaled from, not a
                # separate estimate. Read the class docstring before using it.
                'confidence': depth_normalized
            }
        else:
            # DPT style: conv -> bn -> act -> upsample
            for conv, bn, act, up in zip(
                    self.dpt_conv_layers,
                    self.dpt_bn_layers,
                    self.dpt_act_layers,
                    self.dpt_up_layers,
            ):
                x = conv(x)
                x = bn(x, training=training)
                x = act(x)
                if up is not None:
                    x = up(x)

            # Final output conv (no sigmoid, activation handled by dpt_output_activation)
            depth = self.dpt_output_conv(x)

            return {
                'depth': depth,
                'confidence': x  # Pre-output features as confidence proxy
            }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the two output shapes.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'depth'`` and ``'confidence'`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]

        if self.decoder_style == "progressive":
            # Three transposed-conv upsamples, stride 2 each.
            scale = 8
        else:
            # DPT style: upsample_factor determines scale
            scale = self.dpt_upsample_factor

        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        shape = (batch, out_height, out_width, self.output_channels)
        if self.decoder_style == "progressive":
            return {'depth': shape, 'confidence': shape}
        # DPT's 'confidence' is the last decoder stage's features, so it has
        # that stage's width, not output_channels.
        return {
            'depth': shape,
            'confidence': (batch, out_height, out_width, self.dpt_dims[-1]),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``output_channels``, ``min_depth``,
            ``max_depth``, ``use_log_depth``, ``decoder_style``,
            ``dpt_dims``, ``dpt_use_bias``, ``dpt_activation``,
            ``dpt_output_activation``, and ``dpt_upsample_factor``,
            on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'use_log_depth': self.use_log_depth,
            'decoder_style': self.decoder_style,
            'dpt_dims': self.dpt_dims,
            'dpt_use_bias': self.dpt_use_bias,
            'dpt_activation': self.dpt_activation,
            'dpt_output_activation': self.dpt_output_activation,
            'dpt_upsample_factor': self.dpt_upsample_factor
        })
        return config


# ---------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class ClassificationHead(BaseVisionHead):
    """
    One label for a whole image.

    Spatial features are collapsed to one vector per image, optionally
    normalized, pushed through ``len(block_dims)`` dense blocks, and scored by
    a linear ``Dense``. Use this for image classification.

    The defaults (``block_dims=None``) give two hidden blocks, ``hidden_dim``
    then ``hidden_dim // 2`` wide. A model that reproduces a paper sets the
    shape itself: ``block_dims=()`` is pool then linear classifier (ResNet's
    GAP + FC), ``pooled_normalization_type='layer_norm'`` with
    ``block_dims=()`` is ConvNeXt's GAP + LayerNorm + FC, and
    ``block_dims=(4096, 4096)`` with ``normalization_type=None`` is AlexNet's
    fc6/fc7. ``classifier_dropout_rate`` puts a dropout right before the
    classifier (MobileNetV2's ``Dropout(0.2)`` then ``Linear``).
    ``normalization_kwargs`` reaches every norm the head builds (MobileNetV4's
    BatchNorm epsilon). ``block_use_bias=False`` drops the hidden Dense bias
    that a following BatchNorm would leave untrained. ``kernel_initializer``,
    ``bias_initializer``, ``kernel_regularizer`` and ``bias_regularizer``
    reach every ``Dense`` the head builds, the classifier included, so a
    model's own knobs govern its head.

    This head builds and applies ``attention`` (when ``use_attention`` is
    set) and nothing else from :class:`BaseVisionHead`, so it carries exactly
    the weights its ``call`` runs. Its ``call`` runs no FFN, so ``use_ffn``
    defaults to ``False`` and ``use_ffn=True`` raises ``ValueError``.
    ``build`` matches ``call``: it propagates the attention shape only.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
                 ┌───────┴────────────┐
                 ▼                    ▼
         use_global_pooling       otherwise
        ┌──────────────────┐ ┌──────────────────┐
        │ pooling          │ │ Flatten          │
        │ avg or max       │ │ made in call()   │
        └────────┬─────────┘ └────────┬─────────┘
                 └───────┬────────────┘
                         ▼ (B, C) or (B, H*W*C)
        ┌─────────────────────────────────┐
        │ pooled_norm          (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ dense_blocks: one DenseBlock    │
        │ per block_dims entry (0 or more)│
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ classifier_dropout   (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ classifier: Dense(num_classes)  │
        │ linear, no activation           │
        └────────────────┬────────────────┘
                         ▼
                 ┌───────┴────────────┐
                 ▼                    ▼
              'logits'      softmax(logits, axis=-1)
                                      ▼
                               'probabilities'

    The two keys hold different tensors. ``'logits'`` is the linear
    classifier output, the input a ``from_logits=True`` loss expects.
    ``'probabilities'`` is ``keras.ops.softmax(logits, axis=-1)``, computed in
    the layer's compute dtype exactly as the NLP ``TextClassificationHead``
    does. Under ``mixed_float16`` it is therefore a float16 softmax.

    With ``use_global_pooling=False`` the flatten is a ``layers.Flatten()``
    created inside ``call`` on every invocation. It is not a stored
    sub-layer, so it appears in no weight list. ``Flatten`` holds no weights,
    so this costs correctness nothing.

    Input shape:
        ``(batch, height, width, channels)``, or ``(batch, seq_len, dim)``
        with ``pooling_type='cls_token'``.

    Output shape:
        ``{'logits': (batch, num_classes),
        'probabilities': (batch, num_classes)}``.

    :param num_classes: Number of classes.
    :type num_classes: int
    :param use_global_pooling: Collapse the spatial dimensions by pooling.
        When ``False``, ``call`` flattens them instead.
    :type use_global_pooling: bool
    :param pooling_type: How to collapse the input when
        ``use_global_pooling`` is set: ``'avg'`` or ``'max'`` pooling for 4D
        feature maps, ``'cls_token'`` (position 0) for 3D sequence inputs.
    :type pooling_type: Literal['avg', 'max', 'cls_token']
    :param block_dims: Width of each hidden ``DenseBlock``, in order. ``None``
        means ``(hidden_dim, hidden_dim // 2)``. ``()`` builds no hidden
        block, so the pooled vector goes straight to the classifier. Each
        block is Dense, then ``normalization_type`` (none when that is
        ``None``), ``activation_type`` and ``dropout_rate``.
    :type block_dims: Optional[Sequence[int]]
    :param pooled_normalization_type: Normalization applied to the pooled
        (or flattened) vector before the first block or the classifier.
        ``None`` (the default) builds none.
    :type pooled_normalization_type: Optional[NormalizationType]
    :param normalization_kwargs: Extra arguments for every norm the head
        builds: the pooled norm and each block's norm (for example
        ``{'epsilon': 1e-3}``). ``None`` passes nothing, so the norms factory
        default epsilon (1e-6) applies.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param classifier_dropout_rate: Dropout applied to the classifier's input,
        after the last block (or the pooled vector when there is none). 0
        (the default) builds no dropout.
    :type classifier_dropout_rate: float
    :param block_use_bias: Whether each hidden block's ``Dense`` has a bias.
        Set ``False`` when the blocks use ``batch_norm``: the norm subtracts
        the per-feature batch mean, so a bias before it receives an
        identically-zero gradient and never trains. The classifier always
        has a bias.
    :type block_use_bias: bool
    :param kernel_initializer: Initializer for every ``Dense`` kernel in the
        head, the classifier included. Each ``Dense`` gets its own
        ``clone_initializer`` copy, so no two kernels replay one draw.
    :type kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param bias_initializer: Initializer for every ``Dense`` bias in the head.
    :type bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for every ``Dense`` kernel in the
        head, the classifier included. ``None`` adds no loss.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for every ``Dense`` bias in the head,
        the classifier included. ``None`` adds no loss.
    :type bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :raises ValueError: If ``block_dims`` holds a width that is not a
        positive integer, or ``classifier_dropout_rate`` is outside
        ``[0, 1)``.

    :ivar pooling: Global pooling layer, built only when
        ``use_global_pooling`` is set.
    :vartype pooling: keras.layers.Layer
    :ivar pooled_norm: Normalization of the pooled vector, built only when
        ``pooled_normalization_type`` is set.
    :vartype pooled_norm: keras.layers.Layer
    :ivar dense_blocks: One ``DenseBlock`` per ``block_dims`` entry, named
        ``dense_block``, ``dense_block_1``, ... in order.
    :vartype dense_blocks: List[DenseBlock]
    :ivar classifier_dropout: Dropout before the classifier, built only when
        ``classifier_dropout_rate > 0``.
    :vartype classifier_dropout: keras.layers.Dropout
    :ivar classifier: Output ``Dense(num_classes)``, linear (no activation).
    :vartype classifier: keras.layers.Dense
    """

    def __init__(
            self,
            num_classes: int,
            use_global_pooling: bool = True,
            pooling_type: Literal['avg', 'max', 'cls_token'] = 'avg',
            block_dims: Optional[Sequence[int]] = None,
            pooled_normalization_type: Optional[NormalizationType] = None,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            classifier_dropout_rate: float = 0.0,
            block_use_bias: bool = True,
            kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the classification layers.

        :param num_classes: Number of classes.
        :type num_classes: int
        :param use_global_pooling: Collapse spatial dimensions by pooling.
        :type use_global_pooling: bool
        :param pooling_type: ``'avg'``, ``'max'`` or ``'cls_token'``.
        :type pooling_type: Literal['avg', 'max', 'cls_token']
        :param block_dims: Hidden block widths; ``None`` is
            ``(hidden_dim, hidden_dim // 2)``, ``()`` is none.
        :type block_dims: Optional[Sequence[int]]
        :param pooled_normalization_type: Normalization of the pooled vector,
            or ``None``.
        :type pooled_normalization_type: Optional[NormalizationType]
        :param normalization_kwargs: Extra arguments for every head norm.
        :type normalization_kwargs: Optional[Dict[str, Any]]
        :param classifier_dropout_rate: Dropout on the classifier's input.
        :type classifier_dropout_rate: float
        :param block_use_bias: Whether each hidden block's ``Dense`` has a
            bias.
        :type block_use_bias: bool
        :param kernel_initializer: Initializer for every ``Dense`` kernel.
        :type kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
        :param bias_initializer: Initializer for every ``Dense`` bias.
        :type bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
        :param kernel_regularizer: Regularizer for every ``Dense`` kernel.
        :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
        :param bias_regularizer: Regularizer for every ``Dense`` bias.
        :type bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If a ``block_dims`` entry is not a positive int,
            ``classifier_dropout_rate`` is outside ``[0, 1)``, or
            ``use_ffn`` is ``True``.
        """
        _refuse_common_layers_call_never_runs(
            kwargs, type(self).__name__, ('use_ffn',))
        super().__init__(**kwargs)

        block_dims = _validated_block_dims(block_dims)
        if not 0.0 <= classifier_dropout_rate < 1.0:
            raise ValueError(
                f"classifier_dropout_rate must be in [0, 1), got "
                f"{classifier_dropout_rate}"
            )

        self.num_classes = num_classes
        self.use_global_pooling = use_global_pooling
        self.pooling_type = pooling_type
        self.block_dims = block_dims
        self.pooled_normalization_type = pooled_normalization_type
        self.normalization_kwargs = dict(normalization_kwargs or {})
        self.classifier_dropout_rate = classifier_dropout_rate
        self.block_use_bias = bool(block_use_bias)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self._create_classification_layers()

    def _resolved_block_dims(self) -> Tuple[int, ...]:
        """
        The hidden block widths this head builds.

        :return: ``block_dims``, or ``(hidden_dim, hidden_dim // 2)`` when it
            is ``None``.
        :rtype: Tuple[int, ...]
        """
        if self.block_dims is None:
            return (self.hidden_dim, self.hidden_dim // 2)
        return self.block_dims

    def _create_classification_layers(self) -> None:
        """
        Build the pooling layer, the pooled norm, the dense blocks and the
        classifier.

        Pooling is built only when ``use_global_pooling`` is set, the pooled
        norm only when ``pooled_normalization_type`` is set. One dense block
        is built per resolved ``block_dims`` entry.

        For ``pooling_type='cls_token'`` (used with 3D sequence inputs like
        Vision Transformer outputs), no pooling layer is built; instead the
        first token (position 0) is extracted directly in ``call``.

        :return: None.
        :rtype: None
        """

        # Global pooling
        if self.use_global_pooling and self.pooling_type in ('avg', 'max'):
            if self.pooling_type == 'avg':
                self.pooling = layers.GlobalAveragePooling2D()
            else:
                self.pooling = layers.GlobalMaxPooling2D()
        # For 'cls_token', no pooling layer is built; extraction happens in call()

        if self.pooled_normalization_type is not None:
            self.pooled_norm = create_normalization_layer(
                self.pooled_normalization_type,
                name='pooled_norm',
                **self.normalization_kwargs
            )

        # Dense layers for classification.
        #
        # Named explicitly. `DenseBlock` builds a `Dense` (plus a norm) inside
        # itself, so an unnamed pair takes Keras' auto suffixes -- and those
        # suffixes depend on how many same-named layers the PROCESS has already
        # created. A `.keras` round trip re-enters the counter, so the reloaded
        # model's blocks came back as `dense_block_2` against the original's
        # `dense_block`, and every weight path under them differed. Naming them
        # makes the path stable across a save/load, which is what by-name weight
        # transfer and any weight-path-keyed comparison rely on.
        #
        # DECISION plan-2026-10-14T062616-81938e81/D-013
        # Every Dense here, the classifier included, takes the head's own
        # initializers and regularizer, each kernel through a fresh
        # clone_initializer() copy. Do NOT drop them back to the Dense
        # defaults: the consuming model's kernel_initializer and
        # kernel_regularizer would then silently stop at its backbone. Do NOT
        # hand one initializer instance to every Dense: a seedless Keras 3
        # initializer replays one draw at every site. See decisions.md D-013.
        self.dense_blocks = [
            DenseBlock(
                units=units,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                dropout_rate=self.dropout_rate,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.block_use_bias,
                normalization_kwargs=dict(self.normalization_kwargs),
                name='dense_block' if index == 0 else f'dense_block_{index}'
            )
            for index, units in enumerate(self._resolved_block_dims())
        ]

        if self.classifier_dropout_rate > 0.0:
            self.classifier_dropout = layers.Dropout(
                self.classifier_dropout_rate, name='classifier_dropout'
            )

        # Final classifier: linear. The softmax for 'probabilities' is
        # applied in call(), so 'logits' stays the pre-softmax score.
        self.classifier = layers.Dense(
            units=self.num_classes,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='classifier'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the classification layers on the shapes ``call`` produces.

        This does not use ``_common_processed_shape``, because ``call``
        applies attention and skips the FFN. The flatten branch computes the
        flat width here rather than building a layer, since ``call`` makes
        its own ``Flatten``.

        :param input_shape: Input feature-map shape (4D) or sequence shape (3D).
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = tuple(input_shape)
        # Classification call() applies attention only (no FFN) before pooling.
        if self.use_attention:
            shape = tuple(self.attention.compute_output_shape(shape))

        if self.use_global_pooling and self.pooling_type in ('avg', 'max'):
            self.pooling.build(shape)
            shape = tuple(self.pooling.compute_output_shape(shape))
        elif self.use_global_pooling and self.pooling_type == 'cls_token':
            # cls_token extracts position 0: (B, seq_len, D) -> (B, D)
            shape = (shape[0], shape[-1])
        else:
            # call() flattens via an inline (stateless) Flatten layer.
            flat = 1
            for dim in shape[1:]:
                flat = None if (dim is None or flat is None) else flat * dim
            shape = (shape[0], flat)

        if self.pooled_normalization_type is not None:
            self.pooled_norm.build(shape)

        for dense_block in self.dense_blocks:
            dense_block.build(shape)
            shape = tuple(dense_block.compute_output_shape(shape))

        if self.classifier_dropout_rate > 0.0:
            self.classifier_dropout.build(shape)
        self.classifier.build(shape)

        # The common attention (the only common layer this head has), on the
        # raw input shape.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Pool or flatten, apply the pooled norm if built, run the dense
        blocks, then classify.

        No FFN runs here, whatever ``use_ffn`` says; none is built.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)`` for 4D, or ``(batch, seq_len, dim)`` for 3D sequences.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to the dense blocks
            and to attention.
        :type training: Optional[bool]
        :return: ``{'logits': z, 'probabilities': softmax(z, axis=-1)}``,
            where ``z`` is the linear classifier output.
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x, training=training)

        # Global pooling
        if self.use_global_pooling:
            if self.pooling_type == 'cls_token':
                # Extract CLS token at position 0 for sequence inputs (B, seq_len, D) -> (B, D)
                x = x[:, 0, :]
            else:
                x = self.pooling(x)
        else:
            x = layers.Flatten()(x)

        if self.pooled_normalization_type is not None:
            x = self.pooled_norm(x, training=training)

        # Dense layers
        for dense_block in self.dense_blocks:
            x = dense_block(x, training=training)

        if self.classifier_dropout_rate > 0.0:
            x = self.classifier_dropout(x, training=training)

        # DECISION plan-2026-10-14T062616-81938e81/D-001
        # Linear logits, then their softmax in the COMPUTE dtype, exactly as the
        # NLP TextClassificationHead does. Do NOT put activation='softmax' back on
        # the classifier: 'logits' would again be probabilities. Do NOT cast to
        # float32 before the softmax: that would split the vision and NLP heads'
        # convention; measured under mixed_float16 the float16 softmax is finite
        # and within one float16 ulp of 1. See decisions.md D-001.
        logits = self.classifier(x)
        probabilities = keras.ops.softmax(logits, axis=-1)

        return {
            'logits': logits,
            'probabilities': probabilities
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the two output shapes.

        Both keys carry ``(batch, num_classes)``: the logits and their
        softmax have the same shape.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'logits'`` and ``'probabilities'`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch = input_shape[0]
        shape = (batch, self.num_classes)
        return {'logits': shape, 'probabilities': shape}

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``use_global_pooling``,
            ``pooling_type``, ``block_dims``, ``pooled_normalization_type``,
            ``normalization_kwargs``, ``classifier_dropout_rate``,
            ``block_use_bias`` and the
            serialized initializers and regularizers, on top of the base
            configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'use_global_pooling': self.use_global_pooling,
            'pooling_type': self.pooling_type,
            'block_dims': (
                None if self.block_dims is None else list(self.block_dims)
            ),
            'pooled_normalization_type': self.pooled_normalization_type,
            'normalization_kwargs': dict(self.normalization_kwargs),
            'classifier_dropout_rate': self.classifier_dropout_rate,
            'block_use_bias': self.block_use_bias,
            'kernel_initializer': keras.initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(
                self.bias_initializer),
            'kernel_regularizer': (
                keras.regularizers.serialize(self.kernel_regularizer)
                if self.kernel_regularizer is not None else None
            ),
            'bias_regularizer': (
                keras.regularizers.serialize(self.bias_regularizer)
                if self.bias_regularizer is not None else None
            ),
        })
        return config


# ---------------------------------------------------------------------
# Instance Segmentation Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class InstanceSegmentationHead(BaseVisionHead):
    """
    Boxes, class scores and per-instance masks from one feature map.

    Two branches run on the same common-processed features. One is a full
    :class:`DetectionHead` instance held as a sub-layer, which supplies
    ``'classifications'`` and ``'regressions'``. The other is three
    ``ConvBlock`` stages and a linear 1x1 conv, which supplies
    ``'instance_masks'``.

    Every class-like output here is a logit. ``'instance_masks'`` holds mask
    logits: apply ``keras.ops.sigmoid`` for per-pixel mask probabilities, or
    compile a binary cross-entropy with ``from_logits=True``. The key keeps
    its name; its values are not in ``[0, 1]``.

    The inner detection head is built with ``use_attention=False`` and
    ``use_ffn=False``, because this head has already applied both. Without
    that the same two stages would run twice.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼ x, common-processed
                 ┌───────┴────────────┐
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ detection_head   │ │ mask_conv_blocks │
        │ a DetectionHead  │ │ 3 x ConvBlock    │
        └────────┬─────────┘ └────────┬─────────┘
                 │                    ▼
                 │           ┌──────────────────┐
                 │           │ mask_head 1x1    │
                 │           │ linear           │
                 │           └────────┬─────────┘
                 ▼                    ▼
         'classifications'     'instance_masks'
           'regressions'

        classifications: (B, H, W, num_anchors * num_classes)
        regressions:     (B, H, W, num_anchors * bbox_dims)
        instance_masks:  (B, H, W, num_instances)

    Spatial dimensions are preserved throughout. Every conv here is stride 1
    with ``'same'`` padding.

    ``mask_size`` is stored and serialized but no layer reads it. The masks
    come out at the input feature-map resolution.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'classifications': (batch, height, width, num_anchors *
        num_classes), 'regressions': (batch, height, width, num_anchors *
        bbox_dims), 'instance_masks': (batch, height, width,
        num_instances)}``.

    :param num_classes: Number of object classes, forwarded to the inner
        detection head.
    :type num_classes: int
    :param num_instances: Channel count of the mask output. One channel per
        instance slot.
    :type num_instances: int
    :param mask_size: Stored for serialization. No layer reads it.
    :type mask_size: Tuple[int, int]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar detection_head: The inner :class:`DetectionHead` instance.
    :vartype detection_head: DetectionHead
    :ivar mask_conv_blocks: The three mask ``ConvBlock`` layers.
    :vartype mask_conv_blocks: List[ConvBlock]
    :ivar mask_head: Output ``Conv2D(num_instances)``, linear: it emits the
        mask logits.
    :vartype mask_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            num_classes: int,
            num_instances: int = 100,
            mask_size: Tuple[int, int] = (28, 28),
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the two branches.

        :param num_classes: Number of object classes.
        :type num_classes: int
        :param num_instances: Channel count of the mask output.
        :type num_instances: int
        :param mask_size: Stored for serialization only.
        :type mask_size: Tuple[int, int]
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_instances = num_instances
        self.mask_size = mask_size

        # Create sub-heads
        self.detection_head = DetectionHead(
            num_classes=num_classes,
            hidden_dim=self.hidden_dim,
            normalization_type=self.normalization_type,
            activation_type=self.activation_type,
            dropout_rate=self.dropout_rate,
            # Attention and the FFN have already run in this head, so the
            # inner detection head must not repeat them.
            use_attention=False,
            use_ffn=False,
            name='detection_head'
        )

        self._create_mask_layers()

    def _create_mask_layers(self) -> None:
        """
        Build the mask branch.

        Three ``ConvBlock`` stages at ``hidden_dim`` channels, then a linear
        1x1 conv emitting ``num_instances`` channels of mask logits.

        :return: None.
        :rtype: None
        """

        # Mask feature extraction. Named explicitly, as in EnhancementHead:
        # an auto-suffixed name makes the weight paths process-dependent.
        self.mask_conv_blocks = [
            ConvBlock(
                filters=self.hidden_dim,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                name=f'mask_conv_{i}'
            )
            for i in range(3)
        ]

        # Mask prediction head.
        # DECISION plan-2026-10-14T062616-81938e81/D-003
        # Linear: 'instance_masks' carries mask logits under its old name. Do
        # NOT put activation='sigmoid' back: the sigmoid belongs to the loss
        # (from_logits=True) or the caller, as for every class-like output of
        # layers/heads. See decisions.md D-003.
        self.mask_head = layers.Conv2D(
            filters=self.num_instances,
            kernel_size=1,
            name='mask_head'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the detection sub-head and the mask branch.

        Both consume the shape ``_common_processed_shape`` predicts, not the
        raw input shape. The common attention and FFN are built last, by the
        base class, on the raw input shape.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        feature_shape = self._common_processed_shape(input_shape)

        # Detection sub-head consumes the common-processed features.
        self.detection_head.build(feature_shape)

        shape = tuple(feature_shape)
        for mask_conv in self.mask_conv_blocks:
            mask_conv.build(shape)
            shape = tuple(mask_conv.compute_output_shape(shape))
        self.mask_head.build(shape)

        # The common attention and ffn, on the raw input shape.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stages, then both branches.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'classifications'`` (class logits),
            ``'regressions'`` and ``'instance_masks'`` (mask logits).
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply common processing
        if self.use_attention:
            x = self.attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)

        # Get detection outputs
        detection_outputs = self.detection_head(x, training=training)

        # Mask prediction branch
        mask_features = x
        for mask_conv in self.mask_conv_blocks:
            mask_features = mask_conv(mask_features, training=training)

        instance_masks = self.mask_head(mask_features)

        return {
            'classifications': detection_outputs['classifications'],
            'regressions': detection_outputs['regressions'],
            'instance_masks': instance_masks
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the three output shapes.

        The two detection shapes come from the inner
        :class:`DetectionHead`. The mask shape keeps the input spatial
        dimensions and carries ``num_instances`` channels.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'classifications'``, ``'regressions'`` and
            ``'instance_masks'`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        detection_shapes = self.detection_head.compute_output_shape(input_shape)
        return {
            'classifications': detection_shapes['classifications'],
            'regressions': detection_shapes['regressions'],
            'instance_masks': (batch, height, width, self.num_instances)
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``num_classes``, ``num_instances`` and
            ``mask_size``, on top of the base configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_instances': self.num_instances,
            'mask_size': self.mask_size
        })
        return config


# ---------------------------------------------------------------------
# Enhancement Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class EnhancementHead(BaseVisionHead):
    """
    A restored or upscaled image, built from a feature map.

    Use this for denoising, super-resolution and the other image-to-image
    tasks. ``ConvBlock`` stages refine the features, one per ``block_dims``
    entry (three at ``hidden_dim`` by default). Then one conv produces the
    image: a transposed conv when ``scale_factor > 1``, and a 3x3
    same-resolution conv otherwise.

    ``block_dims=()`` builds no refinement block, so the head is exactly one
    3x3 conv to ``output_channels``: the tail conv of a restoration network
    such as SCUNet's ``m_tail`` (``output_use_bias=False``) or DarkIR's
    ``ending`` (bias kept). Any long skip or global residual around that
    conv is the model's to apply; the head adds none.

    This head applies none of the common stages and no dropout, so it builds
    none: ``use_attention`` and ``use_ffn`` default to ``False``,
    ``dropout_rate`` to ``0.0``, and any other value raises ``ValueError``
    (its ``ConvBlock`` stages take no dropout). ``build`` does not call
    ``_common_processed_shape`` either. The head does read ``hidden_dim``,
    ``normalization_type`` and ``activation_type``, which it hands to its own
    ``ConvBlock`` stages. Its output is unactivated: a restored image, not a
    score.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ enhance_blocks: one ConvBlock   │ default: 3 x hidden_dim
        │ per block_dims entry (0 or more)│ () : none
        └────────────────┬────────────────┘
                          ▼
                 ┌────────┴────────┐
                 ▼                 ▼
          scale_factor > 1     otherwise
        ┌─────────────────┐ ┌─────────────────┐
        │ upsample        │ │ output_conv     │
        │ Conv2DTranspose │ │ Conv2D 3x3      │
        └────────┬────────┘ └────────┬────────┘
                 └─────────┬─────────┘
                           ▼
                       'enhanced'

        enhanced: (B, H*s, W*s, output_channels) when s > 1,
                  else (B, H, W, output_channels). s = scale_factor.

    The class name is frozen. This class once lived inside
    ``create_enhancement_head`` as a closure-local registered class, and was
    lifted to module scope. Do NOT re-nest it there. Its decorator is now
    ``@register_dl_technique("dl_techniques.layers.heads.vision.factory")``,
    so ``get_registered_name`` resolves
    ``dl_techniques.layers.heads.vision.factory>EnhancementHead``. The helper
    also mints a legacy ``Custom>EnhancementHead`` alias. That alias is what a
    checkpoint written before 2026-08-29 reads, and it was verified on
    2026-08-29 to resolve to this same class. It is keyed on the bare class
    name, so a rename would drop it and break those archives.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'enhanced': (batch, out_height, out_width, output_channels)}``.

    :param output_channels: Channel count of the output image.
    :type output_channels: int
    :param scale_factor: Spatial upscaling factor. Above 1 selects the
        transposed-conv path; 1 selects the same-resolution conv.
    :type scale_factor: int
    :param block_dims: Width of each refinement ``ConvBlock``, in order.
        ``None`` means ``(hidden_dim, hidden_dim, hidden_dim)``. ``()``
        builds no block, so the output conv reads the input directly.
    :type block_dims: Optional[Sequence[int]]
    :param output_use_bias: Whether the output conv (``upsample`` or
        ``output_conv``) has a bias. Defaults to ``True``.
    :type output_use_bias: bool
    :param kwargs: Forwarded to :class:`BaseVisionHead`.

    :raises ValueError: If ``use_attention`` or ``use_ffn`` is ``True``,
        ``dropout_rate`` is above 0, or ``block_dims`` holds a width that is
        not a positive integer.

    :ivar enhance_blocks: One refinement ``ConvBlock`` per resolved
        ``block_dims`` entry, named ``enhance_block_{i}``.
    :vartype enhance_blocks: List[ConvBlock]
    :ivar upsample: Transposed conv, built only when ``scale_factor > 1``.
    :vartype upsample: keras.layers.Conv2DTranspose
    :ivar output_conv: Output conv, built only when ``scale_factor <= 1``.
    :vartype output_conv: keras.layers.Conv2D
    """

    def __init__(
            self,
            output_channels: int = 3,
            scale_factor: int = 1,
            block_dims: Optional[Sequence[int]] = None,
            output_use_bias: bool = True,
            **kwargs
    ):
        """
        Store the configuration and build the enhancement layers.

        No common layer is built: this head's ``call`` applies none. Only one
        of ``upsample`` and ``output_conv`` is built, chosen by
        ``scale_factor``.

        :param output_channels: Channel count of the output image.
        :type output_channels: int
        :param scale_factor: Spatial upscaling factor.
        :type scale_factor: int
        :param block_dims: Refinement block widths; ``None`` is three blocks
            at ``hidden_dim``, ``()`` is none.
        :type block_dims: Optional[Sequence[int]]
        :param output_use_bias: Whether the output conv has a bias.
        :type output_use_bias: bool
        :param kwargs: Forwarded to :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``use_attention`` or ``use_ffn`` is ``True``,
            ``dropout_rate`` is above 0, or a ``block_dims`` entry is not a
            positive integer.
        """
        _refuse_common_layers_call_never_runs(
            kwargs, type(self).__name__,
            ('use_attention', 'use_ffn', 'dropout_rate'))
        super().__init__(**kwargs)
        self.output_channels = output_channels
        self.scale_factor = scale_factor
        self.block_dims = _validated_block_dims(block_dims)
        self.output_use_bias = bool(output_use_bias)
        resolved_dims = (
            (self.hidden_dim,) * 3 if self.block_dims is None
            else self.block_dims
        )

        # Enhancement-specific layers. Named explicitly: unnamed, each took
        # Keras' process-global auto suffix (`conv_block_102`, `conv2d_20`),
        # so two instances of one model disagreed by weight path and a
        # `.keras` reload drifted every path under the head.
        self.enhance_blocks = [
            ConvBlock(
                filters=filters,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                name=f'enhance_block_{i}'
            )
            for i, filters in enumerate(resolved_dims)
        ]

        if self.scale_factor > 1:
            # For super-resolution
            self.upsample = layers.Conv2DTranspose(
                filters=self.output_channels,
                kernel_size=3,
                strides=self.scale_factor,
                padding='same',
                use_bias=self.output_use_bias,
                name='upsample'
            )
        else:
            # For denoising and other tasks
            self.output_conv = layers.Conv2D(
                filters=self.output_channels,
                kernel_size=3,
                padding='same',
                use_bias=self.output_use_bias,
                name='output_conv'
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the refinement stack and the output conv.

        Each ``ConvBlock`` is built on the shape the previous one returns.
        The head owns no common layer, so the base ``build`` only marks it
        built.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = tuple(input_shape)
        for block in self.enhance_blocks:
            block.build(shape)
            shape = tuple(block.compute_output_shape(shape))

        if self.scale_factor > 1:
            self.upsample.build(shape)
        else:
            self.output_conv.build(shape)

        # This head owns no common layer, so this only marks it built.
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Refine the features, then produce the output image.

        No common stage runs here, and none is built. See the class
        docstring.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to each ``ConvBlock``.
        :type training: Optional[bool]
        :return: Dict with a single ``'enhanced'`` key.
        :rtype: Dict[str, keras.KerasTensor]
        """
        x = inputs

        for block in self.enhance_blocks:
            x = block(x, training=training)

        if self.scale_factor > 1:
            x = self.upsample(x)
        else:
            x = self.output_conv(x)

        return {'enhanced': x}

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the output shape.

        With ``scale_factor > 1`` the transposed conv scales height and width
        by ``scale_factor``. Otherwise the output conv keeps them. Channels
        become ``output_channels`` either way.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with the ``'enhanced'`` output shape.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        scale = self.scale_factor if self.scale_factor > 1 else 1
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        return {'enhanced': (batch, out_height, out_width, self.output_channels)}

    def get_config(self):
        """
        Return the constructor arguments for serialization.

        Adds this head's four arguments to the base configuration.

        :return: Config dict carrying ``output_channels``, ``scale_factor``,
            ``block_dims`` and ``output_use_bias``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'scale_factor': self.scale_factor,
            'block_dims': (
                None if self.block_dims is None else list(self.block_dims)
            ),
            'output_use_bias': self.output_use_bias,
        })
        return config


# ---------------------------------------------------------------------
# Multi-Task Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class MultiTaskHead(keras.layers.Layer):
    """
    Several vision heads behind one layer.

    Each entry in ``task_configs`` becomes one head. The heads are
    independent: they share the input, not any weights. This class does not
    inherit from :class:`BaseVisionHead`, so it builds no common layers of
    its own.

    A dict input is read per task name. A task name missing from the dict
    falls back to the ``'shared'`` entry. A single tensor is sent to every
    head unchanged.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, H, W, C), or a dict keyed by task name
        with a 'shared' fallback for any name it omits
                                   │
                      ┌────────────┴────────────┐
                      ▼            ▼            ▼
                 ┌───────────┐┌───────────┐┌───────────┐
                 │ head[t1]  ││ head[t2]  ││ head[tN]  │
                 └─────┬─────┘└─────┬─────┘└─────┬─────┘
                       ▼            ▼            ▼
                  outputs[t1]  outputs[t2]  outputs[tN]

        Each head is one of five classes:

        DetectionHead        SegmentationHead
        DepthEstimationHead  ClassificationHead
        InstanceSegmentationHead

        Any other task type raises ValueError.

    Every head returns a dict, so ``call`` returns a dict of dicts: task name
    to that head's own output keys.

    Input shape:
        ``(batch, height, width, channels)``, or a dict of such shapes keyed
        by task name.

    Output shape:
        Dict mapping each task name to that head's output shapes.

    :param task_configs: Dict mapping a task name to that head's config. Each
        config carries a ``'task_type'`` key naming one of the five classes,
        as a :class:`VisionTaskType` member or its string value.
        ``get_config`` stores the string value, so the head survives a
        ``.keras`` round trip.
    :type task_configs: Dict[str, Dict[str, Any]]
    :param shared_backbone_dim: Default ``hidden_dim`` for any task config
        that does not set one.
    :type shared_backbone_dim: int
    :param use_task_specific_attention: Value forced onto every task head's
        ``use_attention``. A per-task ``use_attention`` is overwritten.
    :type use_task_specific_attention: bool
    :param kwargs: Additional arguments for the base Layer class.

    :ivar task_heads: Dict mapping each task name to its head instance. Each
        head is named ``'<task name>_head'`` unless its config sets ``name``.
    :vartype task_heads: Dict[str, keras.layers.Layer]

    :raises ValueError: From ``_create_task_heads``, when a config names a
        task type with no head.
    """

    def __init__(
            self,
            task_configs: Dict[str, Dict[str, Any]],
            shared_backbone_dim: int = 256,
            use_task_specific_attention: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration and build one head per task.

        :param task_configs: Dict mapping a task name to that head's config.
            Each config carries a ``'task_type'`` key.
        :type task_configs: Dict[str, Dict[str, Any]]
        :param shared_backbone_dim: Default ``hidden_dim`` for any task
            config that does not set one.
        :type shared_backbone_dim: int
        :param use_task_specific_attention: Value forced onto every task
            head's ``use_attention``.
        :type use_task_specific_attention: bool
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises ValueError: If a config names a task type with no head.
        """
        super().__init__(**kwargs)

        self.task_configs = task_configs
        self.shared_backbone_dim = shared_backbone_dim
        self.use_task_specific_attention = use_task_specific_attention

        self._create_task_heads()

    def _create_task_heads(self) -> None:
        """
        Build one head per entry in ``task_configs``.

        Each entry names a task type and carries that head's keyword
        arguments. ``hidden_dim`` defaults to ``shared_backbone_dim`` and
        ``use_attention`` is forced to ``use_task_specific_attention``, so a
        per-task ``use_attention`` in the caller dict is overwritten.

        Five task types are accepted. Any other raises.

        :return: None.
        :rtype: None
        :raises ValueError: If a config names a task type with no head.
        """

        self.task_heads = {}

        for task_name, config in self.task_configs.items():
            # Copy before pop: do NOT mutate the caller's config dict (the
            # entries of self.task_configs are the caller's objects). Mutating
            # them stripped 'task_type' as a side-effect and broke get_config()
            # round-trips and repeated construction. See decisions.md / SC6.
            config = dict(config)
            task_type = config.pop('task_type')
            if isinstance(task_type, str):
                task_type = VisionTaskType.from_string(task_type)

            # Add shared configuration
            config['hidden_dim'] = config.get('hidden_dim', self.shared_backbone_dim)
            config['use_attention'] = self.use_task_specific_attention
            # Named after the task, unless the caller names it: an unnamed
            # head takes a process-global Keras suffix ('detection_head_2'),
            # which a .keras reload re-enters, so every weight path under it
            # moved across a round trip.
            config.setdefault('name', f'{task_name}_head')

            # Create appropriate head
            if task_type == VisionTaskType.DETECTION:
                self.task_heads[task_name] = DetectionHead(**config)
            elif task_type == VisionTaskType.SEGMENTATION:
                self.task_heads[task_name] = SegmentationHead(**config)
            elif task_type == VisionTaskType.DEPTH_ESTIMATION:
                self.task_heads[task_name] = DepthEstimationHead(**config)
            elif task_type == VisionTaskType.CLASSIFICATION:
                self.task_heads[task_name] = ClassificationHead(**config)
            elif task_type == VisionTaskType.INSTANCE_SEGMENTATION:
                self.task_heads[task_name] = InstanceSegmentationHead(**config)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build every task head on its own input shape.

        A dict ``input_shape`` gives per-task shapes, falling back to its
        ``'shared'`` entry. A single shape is used for every task.

        :param input_shape: One shared feature-map shape, or a dict of
            per-task shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        :return: None.
        :rtype: None
        """
        for task_name, task_head in self.task_heads.items():
            if isinstance(input_shape, dict):
                task_input_shape = input_shape.get(task_name, input_shape.get('shared'))
            else:
                task_input_shape = input_shape
            task_head.build(task_input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, Dict[str, keras.KerasTensor]]:
        """
        Run every task head and collect the outputs.

        A dict input is read per task name, falling back to its ``'shared'``
        entry. A single tensor is sent to every head unchanged.

        :param inputs: One feature map, or a dict of per-task feature maps.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Keras training flag, forwarded to each head.
        :type training: Optional[bool]
        :return: Dict mapping each task name to that head's output dict.
        :rtype: Dict[str, Dict[str, keras.KerasTensor]]
        """

        outputs = {}

        # Handle different input formats
        if isinstance(inputs, dict):
            # Task-specific inputs
            for task_name, task_head in self.task_heads.items():
                task_input = inputs.get(task_name, inputs.get('shared'))
                outputs[task_name] = task_head(task_input, training=training)
        else:
            # Shared input for all tasks
            for task_name, task_head in self.task_heads.items():
                outputs[task_name] = task_head(inputs, training=training)

        return outputs

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
    ) -> Dict[str, Any]:
        """
        Report each task head's output shape.

        Delegates to every head's own ``compute_output_shape``. A dict
        ``input_shape`` gives per-task shapes, falling back to its
        ``'shared'`` entry. A single shape is used for every task.

        :param input_shape: One shared feature-map shape, or a dict of
            per-task shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        :return: Dict mapping each task name to that head's output shape.
        :rtype: Dict[str, Any]
        """
        output_shapes = {}
        for task_name, task_head in self.task_heads.items():
            if isinstance(input_shape, dict):
                task_input_shape = input_shape.get(task_name, input_shape.get('shared'))
            else:
                task_input_shape = input_shape
            output_shapes[task_name] = task_head.compute_output_shape(task_input_shape)
        return output_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        ``task_configs`` is stored as the caller passed it, except that a
        :class:`VisionTaskType` ``task_type`` becomes its string value.
        Nothing here mutates the caller's dicts, so a round trip rebuilds the
        same heads.

        :return: Config dict carrying ``task_configs``,
            ``shared_backbone_dim`` and ``use_task_specific_attention``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        # DECISION plan-2026-10-14T062616-81938e81/D-027
        # Store each task_type as its string value. Do NOT store the caller's
        # VisionTaskType member: an Enum is not JSON-serializable, so saving
        # any model holding this head raised TypeError. _create_task_heads
        # parses the string back. See decisions.md D-027.
        task_configs = {}
        for task_name, task_config in self.task_configs.items():
            task_config = dict(task_config)
            task_type = task_config['task_type']
            if isinstance(task_type, VisionTaskType):
                task_config['task_type'] = task_type.value
            task_configs[task_name] = task_config
        config.update({
            'task_configs': task_configs,
            'shared_backbone_dim': self.shared_backbone_dim,
            'use_task_specific_attention': self.use_task_specific_attention
        })
        return config


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------

def create_vision_head(
        task_type: Union[VisionTaskType, str],
        **kwargs: Any
) -> BaseVisionHead:
    """
    Build the vision head that serves a task type.

    Ten of the 37 ``VisionTaskType`` members reach a head. The other 27 raise
    ``ValueError``. There is no fallback head and never has been: a task with
    no implementation fails loudly rather than returning a plausible wrong
    one.

    Five members get a class of their own. Three reuse another task's class,
    two of them with a different output channel count. Denoising and
    super-resolution route through :func:`create_enhancement_head`.

    **Architecture Overview:**

    .. code-block:: text

        task_type: VisionTaskType, or a str that from_string parses
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          one of these 10     the other 27
                 │                 │
                 ▼                 ▼
           head instance     ValueError raised

        DETECTION              DetectionHead
        SEGMENTATION           SegmentationHead
        INSTANCE_SEGMENTATION  InstanceSegmentationHead
        CLASSIFICATION         ClassificationHead
        DEPTH_ESTIMATION       DepthEstimationHead
        SURFACE_NORMALS        DepthEstimationHead, 3 channels
        OPTICAL_FLOW           DepthEstimationHead, 2 channels
        KEYPOINT_DETECTION     DetectionHead
        DENOISING              EnhancementHead
        SUPER_RESOLUTION       EnhancementHead, scale_factor 2

    The ``VisionTaskType`` docstring in ``vision/task_types.py`` names all 27
    members that raise. This function is the arbiter if the two disagree.

    :param task_type: Task the head is for, as an enum member or its string
        value.
    :type task_type: Union[VisionTaskType, str]
    :param kwargs: Forwarded to the chosen head class.
    :return: Configured vision head.
    :rtype: BaseVisionHead
    :raises ValueError: If the task type is one of the other 27.
    """

    # Convert string to VisionTaskType if needed
    if isinstance(task_type, str):
        task_type = VisionTaskType.from_string(task_type)

    # Create appropriate head based on task type
    if task_type == VisionTaskType.DETECTION:
        return DetectionHead(**kwargs)

    elif task_type == VisionTaskType.SEGMENTATION:
        return SegmentationHead(**kwargs)

    elif task_type == VisionTaskType.DEPTH_ESTIMATION:
        return DepthEstimationHead(**kwargs)

    elif task_type == VisionTaskType.CLASSIFICATION:
        return ClassificationHead(**kwargs)

    elif task_type == VisionTaskType.INSTANCE_SEGMENTATION:
        return InstanceSegmentationHead(**kwargs)

    elif task_type == VisionTaskType.SURFACE_NORMALS:
        # Surface normals use similar architecture to depth
        return DepthEstimationHead(output_channels=3, **kwargs)

    elif task_type == VisionTaskType.OPTICAL_FLOW:
        # Optical flow predicts 2D motion vectors
        return DepthEstimationHead(output_channels=2, **kwargs)

    elif task_type == VisionTaskType.KEYPOINT_DETECTION:
        # Keypoint detection is similar to detection with different outputs
        return DetectionHead(**kwargs)

    elif task_type in [VisionTaskType.DENOISING, VisionTaskType.SUPER_RESOLUTION]:
        # Image enhancement tasks
        return create_enhancement_head(task_type, **kwargs)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def create_enhancement_head(
        task_type: VisionTaskType,
        **kwargs: Any
) -> BaseVisionHead:
    """
    Build an :class:`EnhancementHead` for a restoration task.

    The only thing this function decides is the default scale factor.
    Super-resolution gets ``scale_factor=2`` unless the caller passed one.
    Every other task type is left alone, so the head keeps its own default of
    1 and takes the same-resolution path.

    The class itself lives at module scope. This function does not define it.

    :param task_type: Task the head is for. Only
        ``VisionTaskType.SUPER_RESOLUTION`` changes the outcome.
    :type task_type: VisionTaskType
    :param kwargs: Forwarded to :class:`EnhancementHead`.
    :return: Configured enhancement head.
    :rtype: BaseVisionHead
    """

    if task_type == VisionTaskType.SUPER_RESOLUTION:
        kwargs['scale_factor'] = kwargs.get('scale_factor', 2)

    return EnhancementHead(**kwargs)


def create_multi_task_head(
        task_configuration: Union[TaskConfiguration, List[VisionTaskType], Dict[str, Dict]],
        **kwargs: Any
) -> MultiTaskHead:
    """
    Build a :class:`MultiTaskHead` from any of three configuration formats.

    A :class:`TaskConfiguration` contributes its enabled tasks. A list
    contributes its entries, parsing any string through
    ``VisionTaskType.from_string``. A dict is used as the task-config mapping
    unchanged. In the first two cases each task's own options are taken from
    the ``kwargs`` entry named after the task's value (``detection={'num_classes':
    80}``), and that entry is removed from ``kwargs``.

    The remaining ``kwargs`` go to :class:`MultiTaskHead`.

    :param task_configuration: A :class:`TaskConfiguration`, a list of
        :class:`VisionTaskType` members or strings, or a dict mapping task
        names to config dicts.
    :type task_configuration: Union[TaskConfiguration, List[VisionTaskType], Dict[str, Dict]]
    :param kwargs: Per-task option dicts, keyed by task value (list and
        :class:`TaskConfiguration` forms only), plus any argument for
        :class:`MultiTaskHead`.
    :return: Configured multi-task head.
    :rtype: MultiTaskHead
    :raises ValueError: If ``task_configuration`` is none of the three
        accepted types.
    """

    if isinstance(task_configuration, TaskConfiguration):
        # Convert TaskConfiguration to dict
        task_configs = {}
        for task in task_configuration.get_enabled_tasks():
            task_configs[task.value] = {
                'task_type': task,
                **kwargs.pop(task.value, {})
            }

    elif isinstance(task_configuration, list):
        # List of TaskTypes
        task_configs = {}
        for task in task_configuration:
            if isinstance(task, str):
                task = VisionTaskType.from_string(task)
            # pop, not get: a per-task dict left in kwargs would also reach
            # MultiTaskHead(**kwargs), which refuses it as an unknown
            # keyword, so this form could never take a per-task option.
            task_configs[task.value] = {
                'task_type': task,
                **kwargs.pop(task.value, {})
            }

    elif isinstance(task_configuration, dict):
        # Already a configuration dict
        task_configs = task_configuration

    else:
        raise ValueError(f"Invalid task_configuration type: {type(task_configuration)}")

    return MultiTaskHead(task_configs=task_configs, **kwargs)


# ---------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------

#: Head-config keys every :class:`HeadConfiguration` preset forces, per task
#: type, because that task's head never applies the layer the key would
#: build and refuses ``True`` (:func:`_refuse_common_layers_call_never_runs`).
#: Applied last by both presets, so a preset never builds an unusable
#: config.
PRESET_FIXED_KEYS: Dict[VisionTaskType, Dict[str, Any]] = {
    VisionTaskType.CLASSIFICATION: {'use_ffn': False},
    VisionTaskType.DENOISING: {
        'use_attention': False, 'use_ffn': False, 'dropout_rate': 0.0},
    VisionTaskType.SUPER_RESOLUTION: {
        'use_attention': False, 'use_ffn': False, 'dropout_rate': 0.0},
}

class HeadConfiguration:
    """
    Two keyword-argument presets for the vision heads.

    This class builds no layer. Each static method returns a dict you pass to
    a head constructor or to :func:`create_vision_head`. The presets differ in
    width, dropout, attention and FFN choice. Every head either preset
    configures builds and runs (pinned by the head tests).

    A third, ``get_high_performance_config``, was deleted on 2026-10-14: its
    ``attention_type='differential'`` needs ``num_heads`` and ``head_dim``,
    which the heads' attention builder never passes, so it raised
    ``ValueError`` for every task type and had built no head.

    **Architecture Overview:**

    .. code-block:: text

        get_default_config(task_type)
                  │
                  ▼
        base_config, 7 keys shared by every task
                  │
                  ▼
        + the task_specific entry, for 5 of the task types
                  │
                  ▼
        get_efficient_config
        overrides 6 keys
                  │
                  ▼
        both presets, last: the PRESET_FIXED_KEYS entry for the task

    ``get_efficient_config`` calls ``get_default_config`` first, so it
    inherits the task-specific keys and then overrides the shared ones.

    Both presets then apply :data:`PRESET_FIXED_KEYS` for their task type,
    last, so no preset asks a head for a layer its ``call`` never runs:
    classification gets ``use_ffn=False`` and the two enhancement tasks get
    ``use_attention=False``, ``use_ffn=False`` and ``dropout_rate=0.0``.
    Those heads refuse the other values. For them the FFN keys
    (``ffn_type``, ``ffn_expansion_factor``) are inert.

    Example:
        >>> cfg = HeadConfiguration.get_default_config(
        ...     VisionTaskType.CLASSIFICATION)
        >>> cfg['num_classes']
        1000
    """

    @staticmethod
    def get_default_config(task_type: VisionTaskType) -> Dict[str, Any]:
        """
        Get the default preset for a task type.

        Seven shared keys are always returned. Five task types add their own
        keys on top: detection, segmentation, depth estimation,
        classification and instance segmentation. Then the task's
        :data:`PRESET_FIXED_KEYS` entry is applied, which turns off the
        common layers classification and the two enhancement tasks never
        run. Any other task type gets the seven shared keys and nothing more.

        :param task_type: Task the preset is for.
        :type task_type: VisionTaskType
        :return: Keyword arguments for the matching head class.
        :rtype: Dict[str, Any]
        """

        base_config = {
            'hidden_dim': 256,
            'normalization_type': 'layer_norm',
            'activation_type': 'gelu',
            'dropout_rate': 0.1,
            'use_ffn': True,
            'ffn_type': 'mlp',
            'ffn_expansion_factor': 4
        }

        task_specific = {
            VisionTaskType.DETECTION: {
                # 80 classes is the COCO default.
                'num_classes': 80,
                'num_anchors': 9,
                'bbox_dims': 4,
                'use_attention': False
            },
            VisionTaskType.SEGMENTATION: {
                # 21 classes is the VOC default.
                'num_classes': 21,
                'upsampling_factor': 4,
                'use_skip_connections': True,
                'use_attention': True,
                'attention_type': 'cbam'
            },
            VisionTaskType.DEPTH_ESTIMATION: {
                'output_channels': 1,
                'min_depth': 0.1,
                'max_depth': 100.0,
                'use_log_depth': True,
                'use_attention': False
            },
            VisionTaskType.CLASSIFICATION: {
                # 1000 classes is the ImageNet default. No attention: the
                # 'multi_head' this preset once named takes a (B, N, D)
                # sequence and raised ValueError on the 4D map that 'avg'
                # pooling expects (measured 2026-10-14).
                'num_classes': 1000,
                'use_global_pooling': True,
                'pooling_type': 'avg',
                'use_attention': False,
            },
            VisionTaskType.INSTANCE_SEGMENTATION: {
                'num_classes': 80,
                'num_instances': 100,
                'mask_size': (28, 28),
                'use_attention': True,
                'attention_type': 'cbam'
            }
        }

        config = base_config.copy()
        if task_type in task_specific:
            config.update(task_specific[task_type])
        config.update(PRESET_FIXED_KEYS.get(task_type, {}))

        return config

    @staticmethod
    def get_efficient_config(task_type: VisionTaskType) -> Dict[str, Any]:
        """
        Get the lightweight preset for a task type.

        Starts from :meth:`get_default_config` and overrides six keys: a
        narrower ``hidden_dim``, no dropout, no attention, and a GLU FFN with
        a smaller expansion. :data:`PRESET_FIXED_KEYS` is applied last.

        :param task_type: Task the preset is for.
        :type task_type: VisionTaskType
        :return: Keyword arguments for the matching head class.
        :rtype: Dict[str, Any]
        """

        config = HeadConfiguration.get_default_config(task_type)
        config.update({
            'hidden_dim': 128,
            'dropout_rate': 0.0,
            'use_attention': False,
            'use_ffn': True,
            # 'glu' is the more efficient registered FFN type.
            'ffn_type': 'glu',
            'ffn_expansion_factor': 2
        })
        config.update(PRESET_FIXED_KEYS.get(task_type, {}))
        return config


# ---------------------------------------------------------------------
# Model-side head_config_overrides merge
# ---------------------------------------------------------------------

#: Head-config keys a model owns and a ``head_config_overrides`` dict may not
#: replace. ``name`` fixes the head's weight paths (by-name weight transfer
#: reads them), ``num_classes`` must agree with the model's own
#: ``num_classes`` and its ``get_config``, and ``task_type`` picks the head
#: class.
RESERVED_HEAD_OVERRIDE_KEYS: Tuple[str, ...] = ('name', 'num_classes', 'task_type')


def merge_head_config_overrides(
        head_config: Dict[str, Any],
        overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge a model's ``head_config_overrides`` into its head config.

    Interface contract, shared by every model that builds a vision head from
    a config dict plus caller overrides:

    * ``head_config`` is not mutated. The return value is a new dict.
    * ``overrides`` of ``None`` or ``{}`` returns a plain copy.
    * An override naming any of :data:`RESERVED_HEAD_OVERRIDE_KEYS` raises
      ``ValueError`` naming every offending key. Nothing is merged then.
    * Every other override key replaces or adds the matching entry. Unknown
      keys are not checked here: the head constructor rejects them.

    A plain ``dict.update`` let ``num_classes`` silently desynchronise the
    head from the model's config, and let ``name`` collide with the model's
    own ``name=`` argument as a ``TypeError``.

    :param head_config: The model's head configuration.
    :type head_config: Dict[str, Any]
    :param overrides: Caller overrides, or ``None``.
    :type overrides: Optional[Dict[str, Any]]
    :return: The merged configuration.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``overrides`` names a reserved key.
    """
    merged = dict(head_config)
    if not overrides:
        return merged

    reserved = sorted(set(overrides) & set(RESERVED_HEAD_OVERRIDE_KEYS))
    if reserved:
        raise ValueError(
            f"head_config_overrides may not set {reserved}: the model owns "
            f"{list(RESERVED_HEAD_OVERRIDE_KEYS)}. Pass num_classes to the "
            f"model itself; the head name and task type are fixed."
        )

    merged.update(overrides)
    return merged
