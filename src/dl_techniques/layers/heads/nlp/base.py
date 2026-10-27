"""``BaseNLPHead``: shared construction and pooling for every NLP task head."""

import inspect
import keras
from typing import Dict, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...activations import ActivationType
from ...dense_block import DenseBlock
from ...ffn import create_ffn_layer, FFNType
from ...attention import AttentionType
from ...attention.factory import create_attention_layer, assemble_attention_config
from ...norms import create_normalization_layer, NormalizationType
from ...sequence_pooling import SequencePooling
from ..masking import fill_masked
from ..validation import (
    pool_or_passthrough,
    positive_int,
    resolve_stage_knobs,
)
from .task_types import NLPTaskConfig
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import NLP_STAGE_KNOB_DEFAULTS, _DELEGATED_POOLING_STRATEGIES, _POOLING_TYPES


@register_dl_technique("dl_techniques.layers.heads.nlp.base")
class BaseNLPHead(keras.layers.Layer):
    """
    Shared construction and pooling for every NLP task head.

    This class builds the layers the task heads have in common and stores the
    configuration they read. It has no ``call``. A subclass runs the stages it
    needs and then applies its own output layer.

    ``__init__`` calls ``_create_common_layers``, which always builds ``norm``
    and ``dropout``, and builds ``intermediate``, ``task_attention``, ``ffn``
    and a pooler only when the matching flag asks for them.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, S, D) or (B, D), or a dict holding
        'hidden_states' and an optional 'attention_mask'
                          │
                          ▼
        ┌───────────────────────────────────┐
        │ _pool_sequence         (optional) │ (B,S,D)->(B,D)
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │ norm  ->  dropout                 │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │ intermediate           (optional) │ -> hidden_size
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │ task_attention         (optional) │
        └─────────────────┬─────────────────┘
                          ▼
        ┌───────────────────────────────────┐
        │ ffn                    (optional) │
        └─────────────────┬─────────────────┘
                          ▼
                subclass output layer

    Not every subclass runs every stage, and a head owns only the stages its
    ``call`` runs (D-062). Four never apply ``task_attention``:
    :class:`TextClassificationHead` (its features are pooled to rank 2
    first), :class:`TextGenerationHead`, :class:`TextSimilarityHead` and
    :class:`MultipleChoiceHead`. Two never apply ``ffn``:
    :class:`QuestionAnsweringHead` and :class:`MultipleChoiceHead`. Three
    never pool: :class:`TokenClassificationHead`,
    :class:`QuestionAnsweringHead` and :class:`TextGenerationHead`. Each of
    them refuses the matching flag with ``ValueError`` rather than building a
    layer it would never call.

    **Widths (D-062):** task attention and the FFN run at the width the
    features have when they arrive, ``hidden_size`` after the intermediate
    block and ``input_dim`` without it; the FFN emits ``hidden_size``. The
    output layer of every subclass is built at the width ``call`` feeds it,
    :meth:`_output_width`.

    **Padding (D-063, D-072):** a dict input's ``'attention_mask'`` reaches
    every stage that mixes positions: the pooling and the task attention
    (:meth:`_apply_task_attention`, which refuses a mask for an attention
    type that takes none). The per-token stages need none.

    **Pooling (D-002):**

    .. code-block:: text

        sequence (B, S, D)  +  attention_mask (B, S)
                            │
                 ┌──────────┴──────────┐
           cls/mean/max/last      'attention'
                 ▼                     ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ SequencePooling  │ │ Dense(1, tanh)   │
        │ (shared layer,   │ │ mask -> softmax  │
        │  mask aware)     │ │ weighted sum     │
        └────────┬─────────┘ └────────┬─────────┘
                 └─────────┬──────────┘
                           ▼
                     pooled (B, D)

              any other value -> ValueError

    ``_DELEGATED_POOLING_STRATEGIES`` names the four strategies on the left.
    They are handed to the shared ``SequencePooling`` layer. ``'attention'``
    is not, and stays inline in ``_pool_sequence``. The two paths are not
    interchangeable. ``SequencePooling('attention')`` is ``AttentionPooling``,
    which scores a ``Dense(hidden, tanh)`` projection against a learnable
    context vector. That is a different mechanism and a different weight set
    from the ``Dense(1, tanh)`` scorer here.

    Delegating ``cls``, ``mean`` and ``max`` was checked against the inline
    code it replaced. Mask-aware mean and max agree with the old values within
    atol 1e-6 for any sequence with at least one valid token. That check is
    the reason the delegation is safe, and it is recorded here because the
    plan that made the decision no longer exists.

    :param task_config: The task, as an :class:`NLPTaskConfig` or a dict of
        its fields (the form ``get_config`` writes). It is copied.
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
    :param input_dim: Dimension of input features from foundation model.
    :type input_dim: int
    :param normalization_type: Type of normalization to use.
    :type normalization_type: NormalizationType
    :param activation_type: Type of activation function.
    :type activation_type: ActivationType
    :param use_pooling: Whether to use pooling for sequence-level tasks.
    :type use_pooling: bool
    :param pooling_type: Which pooling strategy to use. ``'last'`` reads the
        last position kept by the mask, so it is the right choice for a causal
        backbone. The default ``'cls'`` reads position 0, which suits a
        bidirectional encoder. See the D-023 note in ``_create_common_layers``.
    :type pooling_type: Literal['mean', 'max', 'cls', 'last', 'attention']
    :param use_intermediate: Whether to build the intermediate DenseBlock.
    :type use_intermediate: bool
    :param intermediate_size: Width of the intermediate layer. Defaults to
        ``input_dim`` when not given.
    :type intermediate_size: Optional[int]
    :param use_task_attention: Whether to build a task-specific attention
        layer. Four subclasses refuse it; see above.
    :type use_task_attention: bool
    :param attention_type: Which registered attention type to build.
        ``None`` (the default) means ``'multi_head'`` when
        ``use_task_attention`` is on; with it off, any value raises
        ``ValueError`` and ``None`` is stored (D-211).
    :type attention_type: Optional[AttentionType]
    :param use_ffn: Whether to build an FFN block.
    :type use_ffn: bool
    :param ffn_type: Which registered FFN type to build. ``None`` (the
        default) means ``'mlp'`` when ``use_ffn`` is on; with it off, any
        value raises and ``None`` is stored.
    :type ffn_type: Optional[FFNType]
    :param ffn_expansion_factor: The FFN's inner width is
        ``hidden_size * ffn_expansion_factor``. ``None`` (the default) means
        4 when ``use_ffn`` is on; with it off, any value raises and ``None``
        is stored.
    :type ffn_expansion_factor: Optional[int]
    :param initializer_range: Standard deviation of the truncated normal
        initializer used by every Dense layer in this package.
    :type initializer_range: float
    :param kwargs: Additional arguments for the base Layer class. A default
        ``name`` of ``"<task name>_head"`` is set when none is given.

    :ivar hidden_size: Working width of the head. Taken from
        ``task_config.hidden_size`` when set, otherwise ``intermediate_size``.
    :vartype hidden_size: int
    :ivar norm: Normalization layer, always built.
    :vartype norm: keras.layers.Layer
    :ivar dropout: Dropout layer, always built.
    :vartype dropout: keras.layers.Dropout
    :ivar sequence_pooler: Shared SequencePooling layer, or ``None``.
    :vartype sequence_pooler: Optional[SequencePooling]
    :ivar attention_pooling: Inline ``Dense(1, tanh)`` scorer used by the
        ``'attention'`` strategy, or ``None``.
    :vartype attention_pooling: Optional[keras.layers.Dense]
    :ivar task_attention: Task attention layer, or ``None``.
    :vartype task_attention: Optional[keras.layers.Layer]
    :ivar intermediate: Intermediate DenseBlock, or ``None``.
    :vartype intermediate: Optional[DenseBlock]
    :ivar ffn: FFN block, or ``None``.
    :vartype ffn: Optional[keras.layers.Layer]

    :raises ValueError: From ``__init__``: ``input_dim`` or
        ``intermediate_size`` not a positive integer, ``pooling_type`` not one
        of the five supported values, a flag for a stage the subclass never
        runs, or an attention/FFN knob for a stage that is off; each names
        the head and the argument.
    """

    def __init__(
            self,
            task_config: Union[NLPTaskConfig, Dict[str, Any]],
            input_dim: int,
            normalization_type: NormalizationType = 'layer_norm',
            activation_type: ActivationType = 'gelu',
            use_pooling: bool = True,
            pooling_type: Literal['mean', 'max', 'cls', 'last', 'attention'] = 'cls',
            use_intermediate: bool = True,
            intermediate_size: Optional[int] = None,
            use_task_attention: bool = False,
            attention_type: Optional[AttentionType] = None,
            use_ffn: bool = False,
            ffn_type: Optional[FFNType] = None,
            ffn_expansion_factor: Optional[int] = None,
            initializer_range: float = 0.02,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration and build the common layers.

        See the class docstring for what each parameter means. The arguments
        are validated, every one is stored on the instance, then
        ``_create_common_layers`` builds the shared layers.

        :param task_config: The task config, or a dict of its fields.
        :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
        :param input_dim: Width of the incoming features.
        :type input_dim: int
        :param normalization_type: Which registered normalization to build.
        :type normalization_type: NormalizationType
        :param activation_type: Activation used by the intermediate block.
        :type activation_type: ActivationType
        :param use_pooling: Whether to build a pooler.
        :type use_pooling: bool
        :param pooling_type: Which pooling strategy to use.
        :type pooling_type: Literal['mean', 'max', 'cls', 'last', 'attention']
        :param use_intermediate: Whether to build the intermediate DenseBlock.
        :type use_intermediate: bool
        :param intermediate_size: Width of the intermediate layer. Defaults to
            ``input_dim``.
        :type intermediate_size: Optional[int]
        :param use_task_attention: Whether to build task attention.
        :type use_task_attention: bool
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
        :param initializer_range: Truncated normal standard deviation.
        :type initializer_range: float
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        """
        owner = type(self).__name__
        # A dict arrives from get_config; an instance is copied, so editing the
        # caller's object later cannot change this head or its config.
        task_config = NLPTaskConfig.coerce(task_config)
        input_dim = positive_int(input_dim, 'input_dim', owner)
        if intermediate_size is not None:
            intermediate_size = positive_int(
                intermediate_size, 'intermediate_size', owner)
        if not isinstance(pooling_type, str) or pooling_type not in _POOLING_TYPES:
            raise ValueError(
                f"{owner}: pooling_type must be one of {list(_POOLING_TYPES)}, "
                f"got {pooling_type!r}."
            )
        self._refuse_unapplied_stages(use_pooling, use_task_attention, use_ffn)

        # Set a default name only when 'name' is absent from kwargs. This
        # stops 'name' being passed twice during deserialization, because the
        # saved config already carries it.
        kwargs.setdefault('name', f"{task_config.name}_head")
        super().__init__(**kwargs)

        # Store configuration
        self.task_config = task_config
        self.input_dim = input_dim
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.use_pooling = use_pooling
        self.pooling_type = pooling_type
        self.use_intermediate = use_intermediate
        self.intermediate_size = intermediate_size or input_dim
        self.use_task_attention = use_task_attention
        self.use_ffn = use_ffn
        # DECISION plan-2026-10-14T062616-81938e81/D-211
        # A stage knob for a stage that is off raises and is stored as None,
        # as on the vision heads (D-210). Do NOT go back to storing the
        # defaults: they were written to get_config for a layer that does not
        # exist. See decisions.md D-211.
        knobs = resolve_stage_knobs(
            owner,
            NLP_STAGE_KNOB_DEFAULTS,
            {'use_task_attention': use_task_attention, 'use_ffn': use_ffn},
            {'attention_type': attention_type, 'ffn_type': ffn_type,
             'ffn_expansion_factor': ffn_expansion_factor},
        )
        self.attention_type = knobs['attention_type']
        self.ffn_type = knobs['ffn_type']
        self.ffn_expansion_factor = knobs['ffn_expansion_factor']
        self.initializer_range = initializer_range

        # Set hidden size from config or use intermediate size
        self.hidden_size = task_config.hidden_size or self.intermediate_size

        # DECISION plan-2026-10-14T062616-81938e81/D-062
        # Task attention and the FFN run at the width the features have when
        # they arrive: hidden_size after the intermediate block, input_dim
        # without it. Do NOT build them at hidden_size unconditionally: with
        # the intermediate block off and hidden_size != input_dim every stage
        # after it was built for a width it was never fed (review NLP-1, eight
        # settings failed). See decisions.md D-062.
        self._stage_width = self.hidden_size if use_intermediate else self.input_dim

        # Create common layers (following Golden Rule: CREATE in __init__)
        self._create_common_layers()

    # Which optional stages a subclass's call applies. BaseNLPHead has no call
    # and allows all three; each head narrows them, and __init__ refuses a flag
    # that would build a stage its call never runs (D-062).
    _POOLS = True
    _APPLIES_TASK_ATTENTION = True
    _APPLIES_FFN = True

    def _refuse_unapplied_stages(
            self,
            use_pooling: bool,
            use_task_attention: bool,
            use_ffn: bool,
    ) -> None:
        """
        Refuse a flag that would build a stage this head's ``call`` never runs.

        :param use_pooling: The caller's ``use_pooling``.
        :type use_pooling: bool
        :param use_task_attention: The caller's ``use_task_attention``.
        :type use_task_attention: bool
        :param use_ffn: The caller's ``use_ffn``.
        :type use_ffn: bool
        :return: None.
        :rtype: None
        :raises ValueError: Naming the head and the argument.
        """
        owner = type(self).__name__
        for flag, on, applied, stage in (
                ('use_pooling', use_pooling, self._POOLS, 'pools no sequence'),
                ('use_task_attention', use_task_attention,
                 self._APPLIES_TASK_ATTENTION, 'never applies task attention'),
                ('use_ffn', use_ffn, self._APPLIES_FFN, 'never applies an FFN'),
        ):
            if on and not applied:
                raise ValueError(
                    f"{owner}: {flag}=True would build a layer this head never "
                    f"runs; its call {stage}. Leave {flag} off."
                )

    def _output_width(self) -> int:
        """
        Return the width of the features ``call`` hands the output layer.

        ``hidden_size`` when the intermediate block or the FFN ran (each emits
        that width), otherwise ``input_dim``; task attention keeps the width it
        is given.

        :return: The width.
        :rtype: int
        """
        if self.use_intermediate or self.use_ffn:
            return self.hidden_size
        return self.input_dim

    def _pool_input(
            self,
            hidden_states: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """
        Pool a sequence to one vector per row, or pass a pooled input through.

        Interface contract, shared by :class:`TextClassificationHead` and both
        forms of :class:`TextSimilarityHead`: a rank-3 ``hidden_states`` is
        pooled under ``attention_mask`` by :meth:`_pool_sequence`; a rank-2
        one is already pooled and returned unchanged. The rank is read once,
        statically.

        :param hidden_states: ``(batch, seq, width)`` or ``(batch, width)``.
        :type hidden_states: keras.KerasTensor
        :param attention_mask: ``(batch, seq)`` keep mask, or ``None``.
        :type attention_mask: Optional[keras.KerasTensor]
        :return: ``(batch, width)``.
        :rtype: keras.KerasTensor
        :raises ValueError: On an unknown rank, a rank other than 2 or 3, a
            mask for a rank-2 input, or a rank-3 input with
            ``use_pooling=False``; each names the head.
        """
        owner = type(self).__name__

        def pool(sequence, mask):
            if not self.use_pooling:
                raise ValueError(
                    f"{owner}: a rank-3 hidden_states needs pooling, but the "
                    "head was built with use_pooling=False. Pass a pooled "
                    "(batch, width) input or build it with use_pooling=True."
                )
            return self._pool_sequence(sequence, mask)

        return pool_or_passthrough(
            hidden_states, attention_mask, pool, owner=owner,
            features_name='hidden_states', mask_name='attention_mask')

    def _create_common_layers(self) -> None:
        """
        Build the layers every NLP head shares.

        ``norm`` and ``dropout`` are always built. The pooler,
        ``task_attention``, ``intermediate`` and ``ffn`` are built only when
        their flags ask for them, so a head never carries weights it will not
        use. Called from ``__init__``, following the package rule that layers
        are created in ``__init__``.

        :return: None.
        :rtype: None
        """

        # Dropout layer
        self.dropout = keras.layers.Dropout(
            self.task_config.dropout_rate,
            name=f"{self.name}_dropout"
        )

        # Optional normalization
        self.norm = create_normalization_layer(
            self.normalization_type,
            name=f"{self.name}_norm"
        )

        # Optional pooling for sequence-level tasks.
        #
        # DECISION plan_2026-06-08_8b32ca51/D-002: cls/mean/max/last delegate
        # to the shared SequencePooling layer. 'attention' stays inline. Do NOT
        # route it through SequencePooling('attention'): that is AttentionPooling,
        # a different mechanism with a different weight set, so it changes pooled
        # values and breaks existing checkpoints. Owning plan gone; see docstring.

        # DECISION plan-2026-08-17T183311-79c63e38/D-023: 'last' is allowed here;
        # the default stays 'cls'. A causal backbone must pass pooling_type='last':
        # under 'cls', perturbing token 5 moved the logits by exactly 0.000e+00
        # while token 0 moved them by 6.205e-02. Do NOT simplify 'last' to
        # inputs[:, -1, :]; it must skip right padding. See decisions.md D-023.
        self.sequence_pooler = None
        self.attention_pooling = None
        if self.use_pooling and self.pooling_type in _DELEGATED_POOLING_STRATEGIES:
            self.sequence_pooler = SequencePooling(
                strategy=self.pooling_type,
                name=f"{self.name}_sequence_pooler"
            )
        if self.use_pooling and self.pooling_type == 'attention':
            # Attention pooling, kept inline. See the D-002 note above.
            self.attention_pooling = keras.layers.Dense(
                1,
                activation='tanh',
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name=f"{self.name}_attention_pooling"
            )

        # Optional task-specific attention
        self.task_attention = None
        if self.use_task_attention:
            # DECISION plan-2026-08-17T183311-79c63e38/D-023: pre-filter our own
            # generic defaults to the keys this attention_type accepts.
            # create_attention_layer raises on an undeclared key, and 14 of the
            # 33 registered types declare neither `dim` nor `dropout_rate`, or
            # only one. Do NOT declare `dim` on those entries. See D-023.
            attn_defaults = {'dim': self._stage_width}
            # 'sliding_window' was dropped from both tuples below on 2026-08-27.
            # It is not an ATTENTION_REGISTRY key and never has been, so
            # `create_attention_layer` raises `Unknown attention type` before
            # either branch can matter. The 1-D banded variant is 'window_band'.
            # Adding that name here would be a new feature, not a fix.
            if self.attention_type in ('multi_head', 'window'):
                attn_defaults['num_heads'] = 8
            if self.attention_type == 'window':
                attn_defaults['window_size'] = 7
            attn_defaults['dropout_rate'] = self.task_config.dropout_rate

            self.task_attention = create_attention_layer(
                self.attention_type,
                name=f"{self.name}_attention",
                **assemble_attention_config(self.attention_type, attn_defaults)
            )

        # Optional intermediate layer
        self.intermediate = None
        if self.use_intermediate:
            self.intermediate = DenseBlock(
                units=self.hidden_size,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                dropout_rate=self.task_config.dropout_rate,
                name=f"{self.name}_intermediate"
            )

        # Optional FFN block
        self.ffn = None
        if self.use_ffn:
            self.ffn = create_ffn_layer(
                self.ffn_type,
                hidden_dim=self.hidden_size * self.ffn_expansion_factor,
                output_dim=self.hidden_size,
                dropout_rate=self.task_config.dropout_rate,
                name=f"{self.name}_ffn"
            )

    def _apply_task_attention(
            self,
            hidden_states: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor],
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """
        Run ``task_attention`` over the real tokens only.

        Interface contract, shared by the heads that apply task attention
        (:class:`TokenClassificationHead`, :class:`QuestionAnsweringHead`):
        ``hidden_states`` is ``(batch, seq, stage_width)`` and
        ``attention_mask`` a ``(batch, seq)`` keep mask (1 = real) or
        ``None``. With no mask the layer is called exactly as before the mask
        existed; with one it reaches the layer's ``attention_mask``.

        :param hidden_states: The per-token features.
        :type hidden_states: keras.KerasTensor
        :param attention_mask: Keep mask, or ``None``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: The attended features, the same shape.
        :rtype: keras.KerasTensor
        :raises ValueError: If a mask is given and the attention type's
            ``call`` takes no ``attention_mask``.
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-072: the padding mask
        # reaches the task attention, so a real token never attends over
        # padding. Do NOT call the layer without it when a mask was given
        # (every real position mixed in the padded ones), and do NOT drop a
        # mask in silence for a type that cannot take one ('performer';
        # 'linear' accepts mask= and discards it): refuse instead. No mask
        # keeps the old call, so mask-free outputs are bit-identical. See
        # decisions.md D-072.
        if attention_mask is None:
            return self.task_attention(hidden_states, training=training)
        takes_mask = 'attention_mask' in inspect.signature(
            self.task_attention.call).parameters
        if not takes_mask:
            raise ValueError(
                f"{type(self).__name__}: attention_type={self.attention_type!r} "
                f"takes no attention_mask, so its task attention would attend "
                f"over padding; drop the mask or choose a type that takes one "
                f"(e.g. 'multi_head')."
            )
        return self.task_attention(
            hidden_states, attention_mask=attention_mask, training=training)

    def _pool_sequence(
            self,
            sequence: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Reduce a sequence to one vector per batch element.

        Four strategies are handed to the shared ``SequencePooling`` layer.
        The fifth, ``'attention'``, is computed here. See the class docstring
        for the diagram and for why the two paths stay separate.

        :param sequence: Sequence tensor ``(batch, seq_length, hidden_dim)``.
        :type sequence: keras.KerasTensor
        :param attention_mask: Optional mask ``(batch, seq_length)``. Passed on
            to ``SequencePooling``, and used to select the dtype's mask
            sentinel over masked scores in the inline branch.
        :type attention_mask: Optional[keras.KerasTensor]
        :return: Pooled representation ``(batch, hidden_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``pooling_type`` is none of the five supported
            values.
        """
        # DECISION plan_2026-06-08_8b32ca51/D-002: the four strategies in
        # _DELEGATED_POOLING_STRATEGIES (cls/mean/max/last) delegate to the
        # shared SequencePooling facade, created in _create_common_layers.
        # 'attention' stays inline. Do NOT route it through SequencePooling:
        # different mechanism, different weights. Owning plan gone.
        if self.pooling_type in _DELEGATED_POOLING_STRATEGIES:
            # The `mask=` argument matters for 'last' (D-023). It is what makes
            # the gather land on the last real token rather than on padding.
            return self.sequence_pooler(sequence, mask=attention_mask)

        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_pooling(sequence)
            attention_weights = keras.ops.squeeze(attention_weights, axis=-1)

            if attention_mask is not None:
                # DECISION plan-2026-08-31T134711-6271592d/D-007
                # SELECT the sentinel, never add it. `w * m + (1 - m) * -1e9`
                # is `0 * -inf = NaN` at every KEPT position once `-1e9`
                # overflows float16. Do NOT reintroduce that form, and do NOT
                # hardcode the magnitude; `mask_sentinel` derives it per dtype.
                attention_weights = fill_masked(
                    attention_weights, attention_mask, self.compute_dtype)

            attention_weights = keras.ops.softmax(attention_weights, axis=-1)
            attention_weights = keras.ops.expand_dims(attention_weights, axis=-1)

            return keras.ops.sum(sequence * attention_weights, axis=1)

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build every common sub-layer that was created.

        A dict input shape is read through its ``'hidden_states'`` key. A
        rank-3 shape is treated as a sequence and a shorter one as already
        pooled, which decides the shape the normalization layer is built for.
        Each optional sub-layer is built only when it exists.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # Determine input shape
        if isinstance(input_shape, dict):
            hidden_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            hidden_shape = input_shape

        # Build normalization layer.
        # Rank 3 is a sequence; anything shorter is already pooled.
        if len(hidden_shape) == 3:
            norm_input_shape = hidden_shape
        else:
            norm_input_shape = (hidden_shape[0], self.input_dim)

        self.norm.build(norm_input_shape)

        # Build the delegated sequence pooler if there is one (D-002).
        if self.sequence_pooler is not None:
            self.sequence_pooler.build(hidden_shape)

        # Build attention pooling if needed
        if self.attention_pooling is not None:
            self.attention_pooling.build(hidden_shape)

        # Build task attention if needed
        if self.task_attention is not None:
            # Task attention runs at the stage width (D-062).
            self.task_attention.build((hidden_shape[0], hidden_shape[1], self._stage_width))

        # Build intermediate layer if needed
        if self.intermediate is not None:
            # Intermediate can receive pooled or sequence input
            if self.use_pooling and len(hidden_shape) == 3:
                intermediate_input = (hidden_shape[0], self.input_dim)
            else:
                intermediate_input = norm_input_shape
            self.intermediate.build(intermediate_input)

        # Build FFN if needed
        if self.ffn is not None:
            ffn_input = (hidden_shape[0], hidden_shape[1] if len(hidden_shape) == 3 else 1,
                         self._stage_width)
            self.ffn.build(ffn_input)

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """
        Return a placeholder output shape.

        The base class does not know what a head produces, so it reports one
        ``'output'`` entry with an unknown width. Every subclass overrides this
        with its real output keys.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: ``{'output': (batch_size, None)}``.
        :rtype: Dict[str, Tuple]
        """
        # Base implementation - subclasses override with specific shapes
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
        else:
            batch_size = input_shape[0] if input_shape else None

        return {'output': (batch_size, None)}

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        ``task_config`` is written as a plain dict of every field
        (:meth:`NLPTaskConfig.to_dict`). There is no ``from_config`` override:
        Keras passes the config to ``__init__``, which takes the dict as it
        is, so the caller's config is never modified.

        :return: Config dict accepted by ``from_config``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'task_config': self.task_config.to_dict(),
            'input_dim': self.input_dim,
            'normalization_type': self.normalization_type,
            'activation_type': self.activation_type,
            'use_pooling': self.use_pooling,
            'pooling_type': self.pooling_type,
            'use_intermediate': self.use_intermediate,
            'intermediate_size': self.intermediate_size,
            'use_task_attention': self.use_task_attention,
            'attention_type': self.attention_type,
            'use_ffn': self.use_ffn,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'initializer_range': self.initializer_range,
        })
        return config
