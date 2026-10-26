"""Base vision task head and its shared construction.

:class:`BaseVisionHead` builds the layers every vision head has in common
(attention, FFN) when their flags are set, and stores the configuration the
subclasses read. It has no ``call``.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras

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
from ..validation import positive_int, resolve_stage_knobs
from .task_types import VisionTaskType, TaskConfiguration
from dl_techniques.utils.keras_registration import register_dl_technique

# registration decorator is imported above

from .utils import (
    STAGE_KNOB_DEFAULTS,
    _drop_off_stage_knobs,
    _UNRUNNABLE_ATTENTION_TYPES,
    _SPATIAL_ATTENTION_TYPES,
    _require_hidden_dim_at_least,
)


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
    inside its own ``ConvBlock`` or ``DenseBlock``, which read
    ``normalization_type`` from this class. ``dropout_rate`` reaches the
    class-score blocks (detection, segmentation refine, classification) and
    the attention/FFN stage; the dense-regression decoders (depth, normals,
    flow, the instance-mask branch, enhancement) take no dropout, as their
    reference decoders have none.

    Two heads run less than this picture. :class:`ClassificationHead` runs
    ``attention`` only and refuses ``use_ffn=True``. :class:`EnhancementHead`
    runs neither, applies no dropout, and refuses ``use_attention=True``,
    ``use_ffn=True`` and ``dropout_rate > 0``. :class:`MultiTaskHead` does not inherit from this class at all. Read the
    subclass diagram, not this one, for what a given head runs.

    :param hidden_dim: Working width of the head, a positive integer. The
        sub-blocks and the FFN use it as their channel count.
    :type hidden_dim: int
    :param normalization_type: Which registered normalization to build, and
        the value handed to each sub-block.
    :type normalization_type: NormalizationType
    :param activation_type: Activation each sub-block uses.
    :type activation_type: ActivationType
    :param dropout_rate: Dropout rate handed to the class-score sub-blocks,
        the FFN and the attention layer. The dense-regression heads default
        it to 0.0 and refuse a rate no layer would apply.
    :type dropout_rate: float
    :param use_attention: Whether to build an attention layer.
    :type use_attention: bool
    :param attention_type: Which registered attention type to build.
        ``None`` (the default) means ``'multi_head'`` when ``use_attention``
        is on; with it off, any value raises ``ValueError`` and ``None`` is
        stored (D-030's rule).
    :type attention_type: Optional[AttentionType]
    :param use_ffn: Whether to build an FFN block. It defaults to ``True``
        here; :class:`ClassificationHead` and :class:`EnhancementHead`
        default it to ``False`` and refuse ``True``.
    :type use_ffn: bool
    :param ffn_type: Which registered FFN type to build. ``None`` (the
        default) means ``'mlp'`` when ``use_ffn`` is on; with it off, any
        value raises and ``None`` is stored.
    :type ffn_type: Optional[FFNType]
    :param ffn_expansion_factor: The FFN inner width is
        ``hidden_dim * ffn_expansion_factor``. ``None`` (the default) means
        4 when ``use_ffn`` is on; with it off, any value raises and ``None``
        is stored.
    :type ffn_expansion_factor: Optional[int]
    :param kwargs: Additional arguments for the base Layer class.

    :ivar attention: Attention layer, built when ``use_attention`` is set.
    :vartype attention: keras.layers.Layer
    :ivar ffn: FFN block, built when ``use_ffn`` is set.
    :vartype ffn: keras.layers.Layer
    """

    #: The ``(H, W)`` grid a sequence-format head places its tokens on. Only
    #: :class:`DetectionHead` and :class:`SegmentationHead` take a
    #: ``patch_grid_size`` argument; every other head holds no grid, so a
    #: spatial attention type cannot run on a sequence it is given.
    patch_grid_size: Optional[Tuple[int, int]] = None

    def __init__(
            self,
            hidden_dim: int = 256,
            normalization_type: NormalizationType = 'layer_norm',
            activation_type: ActivationType = 'gelu',
            dropout_rate: float = 0.1,
            use_attention: bool = False,
            attention_type: Optional[AttentionType] = None,
            use_ffn: bool = True,
            ffn_type: Optional[FFNType] = None,
            ffn_expansion_factor: Optional[int] = None,
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
        :param attention_type: Which registered attention type to build;
            ``None`` means ``'multi_head'`` with the stage on. Refused with
            the stage off.
        :type attention_type: Optional[AttentionType]
        :param use_ffn: Whether to build an FFN block.
        :type use_ffn: bool
        :param ffn_type: Which registered FFN type to build; ``None`` means
            ``'mlp'`` with the stage on. Refused with the stage off.
        :type ffn_type: Optional[FFNType]
        :param ffn_expansion_factor: FFN inner-width multiplier; ``None``
            means 4 with the stage on. Refused with the stage off.
        :type ffn_expansion_factor: Optional[int]
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises ValueError: If a stage knob is given for a stage that is off.
        """
        super().__init__(**kwargs)

        # Store configuration
        self.hidden_dim = positive_int(
            hidden_dim, 'hidden_dim', type(self).__name__)
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_ffn = use_ffn
        # DECISION plan-2026-10-14T062616-81938e81/D-210
        # A stage knob for a stage that is off raises and is stored as None.
        # Do NOT refuse only a non-default value: the default would still be
        # stored and written to get_config for a layer that does not exist.
        # See decisions.md D-210 (D-030's rule).
        knobs = resolve_stage_knobs(
            type(self).__name__,
            STAGE_KNOB_DEFAULTS,
            {'use_attention': use_attention, 'use_ffn': use_ffn},
            {'attention_type': attention_type, 'ffn_type': ffn_type,
             'ffn_expansion_factor': ffn_expansion_factor},
        )
        self.attention_type = knobs['attention_type']
        self.ffn_type = knobs['ffn_type']
        self.ffn_expansion_factor = knobs['ffn_expansion_factor']

        # DECISION plan-2026-10-14T062616-81938e81/D-089: refuse a type the
        # head can build but never run, here, naming the reason. Do not let it
        # through to fail inside call() far from the argument (it used to).
        if self.use_attention and self.attention_type in _UNRUNNABLE_ATTENTION_TYPES:
            raise ValueError(
                f"{type(self).__name__}: attention_type="
                f"{self.attention_type!r} cannot run in a vision head: "
                f"{_UNRUNNABLE_ATTENTION_TYPES[self.attention_type]}."
            )

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
            shape = self._attention_output_shape(shape)
        if self.use_ffn:
            shape = tuple(self.ffn.compute_output_shape(shape))
        return shape

    # DECISION plan-2026-10-14T062616-81938e81/D-071: the attention stage runs
    # in the layout its TYPE takes, whatever layout the head holds: a sequence
    # type sees a map flattened to H*W tokens and restored, a spatial type
    # (_SPATIAL_ATTENTION_TYPES) sees a sequence head's tokens on the patch
    # grid and flattened back. Do NOT hand the layer the head's own layout
    # again: the default 'multi_head' raised on every spatial head, and 'cbam'
    # on the sequence-format detection head (FFTNet). Do NOT move a sequence
    # head's attention after its reshape instead: 'multi_head' then fails
    # there. See decisions.md D-071.
    def _attention_takes_spatial(self, rank: int) -> bool:
        """
        Whether the attention layer takes a rank-4 map, for a head input of
        ``rank``; ``ValueError`` when a spatial type meets a grid-less
        sequence.

        :param rank: Rank of the tensor the head holds (3 or 4).
        :type rank: int
        :return: ``True`` for a type in ``_SPATIAL_ATTENTION_TYPES``.
        :rtype: bool
        :raises ValueError: If a spatial type meets a rank-3 input and the
            head has no ``patch_grid_size`` to place the tokens on.
        """
        spatial = self.attention_type in _SPATIAL_ATTENTION_TYPES
        if spatial and rank == 3 and self.patch_grid_size is None:
            raise ValueError(
                f"{type(self).__name__}: attention_type={self.attention_type!r} "
                f"takes a (batch, height, width, channels) map, but this head "
                f"holds a (batch, tokens, channels) sequence and no "
                f"patch_grid_size to place the tokens on."
            )
        return spatial

    def _attention_input_shape(
            self,
            shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        The shape the attention layer sees when the head holds ``shape``.

        :param shape: A rank-4 map shape or a rank-3 sequence shape.
        :type shape: Tuple[Optional[int], ...]
        :return: The shape in the layout the attention type takes.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(shape)
        spatial = self._attention_takes_spatial(len(shape))
        if len(shape) == 4 and not spatial:
            tokens = (None if shape[1] is None or shape[2] is None
                      else shape[1] * shape[2])
            return (shape[0], tokens, shape[3])
        if len(shape) == 3 and spatial:
            return (shape[0], *self.patch_grid_size, shape[2])
        return shape

    def _attention_output_shape(
            self,
            shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        The shape :meth:`_apply_attention` returns for an input of ``shape``,
        in the head's own layout.

        :param shape: A rank-4 map shape or a rank-3 sequence shape.
        :type shape: Tuple[Optional[int], ...]
        :return: ``shape`` with the attention output width.
        :rtype: Tuple[Optional[int], ...]
        """
        shape = tuple(shape)
        inner = tuple(self.attention.compute_output_shape(
            self._attention_input_shape(shape)))
        return shape[:-1] + (inner[-1],)

    def _apply_attention(
            self,
            x: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run ``attention`` in the layout its type takes; return the head's
        layout.

        Interface contract, shared by every head ``call`` that applies the
        common attention: ``x`` is the rank-4 map or rank-3 sequence the
        head holds; the result has the same rank and leading dimensions,
        with the attention output width. Where the layouts already agree the
        layer is called on ``x`` itself, so nothing changes for a
        configuration that ran before. Row-major reshapes, so token ``i``
        is grid position ``(i // W, i % W)`` both ways.

        :param x: The head's features.
        :type x: keras.KerasTensor
        :param training: Keras training flag, forwarded to the layer.
        :type training: Optional[bool]
        :return: The attended features, in ``x``'s layout.
        :rtype: keras.KerasTensor
        :raises ValueError: See :meth:`_attention_takes_spatial`.
        """
        rank = len(x.shape)
        spatial = self._attention_takes_spatial(rank)
        batch = keras.ops.shape(x)[0]
        if rank == 4 and not spatial:
            height = x.shape[1] if x.shape[1] is not None else keras.ops.shape(x)[1]
            width = x.shape[2] if x.shape[2] is not None else keras.ops.shape(x)[2]
            tokens = keras.ops.reshape(x, (batch, -1, x.shape[-1]))
            y = self.attention(tokens, training=training)
            return keras.ops.reshape(y, (batch, height, width, -1))
        if rank == 3 and spatial:
            grid_h, grid_w = self.patch_grid_size
            grid = keras.ops.reshape(x, (batch, grid_h, grid_w, x.shape[-1]))
            y = self.attention(grid, training=training)
            return keras.ops.reshape(y, (batch, grid_h * grid_w, -1))
        return self.attention(x, training=training)

    def _drop_unread_hidden_dim(self, given_hidden_dim: Any, reason: str) -> None:
        """
        Refuse, and store ``None`` for, a ``hidden_dim`` no layer reads.

        Interface contract, for a head mode whose task layers are sized by
        other arguments (:class:`SegmentationHead` in explicit mode, its one
        caller since the DPT mode of :class:`DepthEstimationHead` was
        deleted, D-222): call it after
        ``BaseVisionHead.__init__`` with the caller's ``hidden_dim`` (``None``
        when not given; pop a ``None`` from ``kwargs`` before ``__init__`` so
        the base validates its default). With the attention or FFN stage on,
        ``hidden_dim`` is their width and is kept. Otherwise a given value
        raises ``ValueError`` and ``self.hidden_dim`` becomes ``None``, which
        ``get_config`` writes and ``from_config`` reads back.

        :param given_hidden_dim: The caller's ``hidden_dim``, or ``None``.
        :type given_hidden_dim: Any
        :param reason: What sizes the task layers instead, for the message.
        :type reason: str
        :return: None.
        :rtype: None
        :raises ValueError: If a ``hidden_dim`` was given and no stage reads it.
        """
        if self.use_attention or self.use_ffn:
            return
        if given_hidden_dim is not None:
            raise ValueError(
                f"{type(self).__name__}: {reason}, and neither the attention "
                f"nor the FFN stage is on, so no layer reads hidden_dim; got "
                f"hidden_dim={given_hidden_dim!r}."
            )
        self.hidden_dim = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the common layers that exist.

        The FFN is built on the raw input shape and the attention on the same
        shape in the layout its type takes; each stage is shape-preserving in
        the width the head runs at.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        if self.use_attention:
            self.attention.build(self._attention_input_shape(input_shape))
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
