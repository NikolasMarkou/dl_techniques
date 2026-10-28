"""Classification head: one label for a whole image."""

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...dense_block import DenseBlock
from ...norms import create_normalization_layer, NormalizationType
from dl_techniques.initializers import clone_initializer
from ..validation import positive_int
from dl_techniques.utils.keras_registration import register_dl_technique
from .base import BaseVisionHead
from .utils import _refuse_common_layers_call_never_runs, _validated_block_dims, _require_hidden_dim_at_least


@register_dl_technique("dl_techniques.layers.heads.vision.classification")
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
    ``classifier_use_bias=False`` builds the classifier without a bias, as a
    reference head such as RADConvNet's does.
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

    With ``use_global_pooling=False`` the flatten is a ``keras.layers.Flatten()``
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
    :param classifier_use_bias: Whether the classifier ``Dense`` has a bias.
        ``False`` builds a classifier with only a kernel.
    :type classifier_use_bias: bool
    :param block_use_bias: Whether each hidden block's ``Dense`` has a bias.
        Set ``False`` when the blocks use ``batch_norm``: the norm subtracts
        the per-feature batch mean, so a bias before it receives an
        identically-zero gradient and never trains. The classifier's bias is
        ``classifier_use_bias``.
    :type block_use_bias: bool
    :param kernel_initializer: Initializer for every ``Dense`` kernel in the
        head, the classifier included. Each ``Dense`` gets its own
        ``clone_initializer`` copy, so a seedless initializer draws afresh at
        every kernel. An initializer that carries an explicit ``seed`` reuses
        that seed at every kernel, by contract.
    :type kernel_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param bias_initializer: Initializer for every ``Dense`` bias in the head.
        When no ``Dense`` has a bias (``classifier_use_bias=False`` and no
        hidden block with a bias) nothing reads it: a value other than the
        default is refused and the head stores ``None``.
    :type bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for every ``Dense`` kernel in the
        head, the classifier included. ``None`` adds no loss.
    :type kernel_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for every ``Dense`` bias in the head,
        the classifier included. ``None`` adds no loss. Refused, like
        ``bias_initializer``, when no ``Dense`` has a bias.
    :type bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :raises ValueError: If ``block_dims`` holds a width that is not a
        positive integer, ``classifier_dropout_rate`` is outside ``[0, 1)``,
        ``pooling_type`` is not ``'avg'``, ``'max'`` or ``'cls_token'``,
        ``num_classes`` is not a positive integer, ``hidden_dim < 2``
        with the default ``block_dims``, or a ``bias_initializer`` or
        ``bias_regularizer`` is given while no ``Dense`` has a bias.

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
    :ivar classifier: Output ``Dense(num_classes)``, linear (no activation),
        with a bias unless ``classifier_use_bias`` is ``False``.
    :vartype classifier: keras.layers.Dense
    """

    #: The ``pooling_type`` values ``call`` implements: global average and
    #: max pooling of a 4D map, and position 0 of a 3D sequence.
    _POOLING_TYPES: Tuple[str, ...] = ('avg', 'max', 'cls_token')

    def __init__(
            self,
            num_classes: int,
            use_global_pooling: bool = True,
            pooling_type: Literal['avg', 'max', 'cls_token'] = 'avg',
            block_dims: Optional[Sequence[int]] = None,
            pooled_normalization_type: Optional[NormalizationType] = None,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            classifier_dropout_rate: float = 0.0,
            classifier_use_bias: bool = True,
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
        :param classifier_use_bias: Whether the classifier ``Dense`` has a
            bias.
        :type classifier_use_bias: bool
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
            ``classifier_dropout_rate`` is outside ``[0, 1)``,
            ``use_ffn`` is ``True``, ``num_classes`` is not a positive int,
            ``pooling_type`` is not one of ``_POOLING_TYPES``, or the default
            blocks would get a width of 0 (``hidden_dim < 2``), or a
            ``bias_initializer`` or ``bias_regularizer`` is given while no
            ``Dense`` has a bias.
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
        if pooling_type not in self._POOLING_TYPES:
            raise ValueError(
                f"ClassificationHead: pooling_type must be one of "
                f"{list(self._POOLING_TYPES)}, got {pooling_type!r}"
            )
        if block_dims is None:
            _require_hidden_dim_at_least(
                self.hidden_dim, 2, type(self).__name__,
                "the default block_dims are (hidden_dim, hidden_dim // 2)")

        self.num_classes = positive_int(
            num_classes, 'num_classes', type(self).__name__)
        self.use_global_pooling = use_global_pooling
        self.pooling_type = pooling_type
        self.block_dims = block_dims
        self.pooled_normalization_type = pooled_normalization_type
        self.normalization_kwargs = dict(normalization_kwargs or {})
        self.classifier_dropout_rate = classifier_dropout_rate
        self.classifier_use_bias = bool(classifier_use_bias)
        self.block_use_bias = bool(block_use_bias)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._set_bias_arguments(bias_initializer, bias_regularizer)

        self._create_classification_layers()

    def _set_bias_arguments(
            self,
            bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer],
            bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]],
    ) -> None:
        """
        Store ``bias_initializer`` and ``bias_regularizer``, or refuse them.

        A ``Dense`` reads them only when it has a bias. When the classifier
        has none and no hidden block has one, a non-default value is refused
        and both are stored as ``None``, which round-trips. An explicit
        ``'zeros'`` is the default, not a value, so it is accepted.

        :param bias_initializer: The caller's bias initializer.
        :type bias_initializer: Union[str, Dict[str, Any], keras.initializers.Initializer]
        :param bias_regularizer: The caller's bias regularizer.
        :type bias_regularizer: Optional[Union[str, Dict[str, Any], keras.regularizers.Regularizer]]
        :return: None.
        :rtype: None
        :raises ValueError: If no ``Dense`` has a bias and either argument
            holds a non-default value.
        """
        initializer = keras.initializers.get(bias_initializer)
        regularizer = keras.regularizers.get(bias_regularizer)
        a_block_has_a_bias = self.block_use_bias and bool(self._resolved_block_dims())
        if self.classifier_use_bias or a_block_has_a_bias:
            self.bias_initializer = initializer
            self.bias_regularizer = regularizer
            return

        given = []
        if initializer is not None and not isinstance(
                initializer, keras.initializers.Zeros):
            given.append(f"bias_initializer={bias_initializer!r}")
        if regularizer is not None:
            given.append(f"bias_regularizer={bias_regularizer!r}")
        if given:
            raise ValueError(
                f"{type(self).__name__} has no Dense with a bias "
                f"(classifier_use_bias=False and no hidden block with a "
                f"bias), so no layer reads {' or '.join(given)}."
            )
        self.bias_initializer = None
        self.bias_regularizer = None

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
                self.pooling = keras.layers.GlobalAveragePooling2D()
            else:
                self.pooling = keras.layers.GlobalMaxPooling2D()
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
            self.classifier_dropout = keras.layers.Dropout(
                self.classifier_dropout_rate, name='classifier_dropout'
            )

        # Final classifier: linear. The softmax for 'probabilities' is
        # applied in call(), so 'logits' stays the pre-softmax score.
        self.classifier = keras.layers.Dense(
            units=self.num_classes,
            use_bias=self.classifier_use_bias,
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
            shape = self._attention_output_shape(shape)

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
            x = self._apply_attention(x, training=training)

        # Global pooling
        if self.use_global_pooling:
            if self.pooling_type == 'cls_token':
                # Extract CLS token at position 0 for sequence inputs (B, seq_len, D) -> (B, D)
                x = x[:, 0, :]
            else:
                x = self.pooling(x)
        else:
            x = keras.layers.Flatten()(x)

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
            ``classifier_use_bias``, ``block_use_bias`` and the
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
            'classifier_use_bias': self.classifier_use_bias,
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
