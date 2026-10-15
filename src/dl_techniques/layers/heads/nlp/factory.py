"""NLP task heads, and the factory that picks one.

This module holds eight layer classes and three module-level factory
functions. The classes turn a foundation model's hidden states into task
predictions. The functions pick a class and configure it.

The heads take hidden states, not tokens. Nothing here knows which backbone
produced them, so the same head works on BERT, GPT or T5 output.

Classes
-------
* :class:`BaseNLPHead` -- shared construction and pooling. Every other head
  except :class:`MultiTaskNLPHead` inherits from it. It has no ``call``.
* :class:`TextClassificationHead` -- one label per sequence.
* :class:`TokenClassificationHead` -- one label per token.
* :class:`QuestionAnsweringHead` -- start and end logits over the sequence.
* :class:`TextSimilarityHead` -- one embedding, or a score for a pair.
* :class:`TextGenerationHead` -- logits over the vocabulary per position.
* :class:`MultipleChoiceHead` -- one score per candidate answer.
* :class:`MultiTaskNLPHead` -- several heads behind one layer.

Helpers
-------
* :func:`get_head_class` -- task type to head class, read from
  ``NLP_HEAD_DISPATCH``. Raises for a task type that has no head.
* :func:`create_nlp_head` -- build one head from a config.
* :func:`create_multi_task_nlp_head` -- build a multi-task head.

The task types, the dispatch table and the config dataclass live in
``nlp/task_types.py``.

Example
-------
>>> from dl_techniques.layers.heads.nlp import create_nlp_head
>>> from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
>>> cfg = NLPTaskConfig(
...     name='sentiment',
...     task_type=NLPTaskType.SENTIMENT_ANALYSIS,
...     num_classes=3,
... )
>>> head = create_nlp_head(cfg, input_dim=768)
>>> out = head(hidden_states)
>>> sorted(out)
['logits', 'probabilities']
"""

import keras
from typing import Dict, List, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...activations import ActivationType
from ...dense_block import DenseBlock
from ...ffn import create_ffn_layer, FFNType
from ...attention import AttentionType
from ...attention.factory import (
    create_attention_layer,
    assemble_attention_config,
)
from ...norms import create_normalization_layer, NormalizationType
from ...sequence_pooling import SequencePooling

from ..validation import positive_int, static_rank
from .task_types import NLP_HEAD_DISPATCH, NLPTaskConfig, NLPTaskType
from dl_techniques.utils.dtype_policy import mask_sentinel, stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# The four pooling strategies BaseNLPHead hands to the shared SequencePooling
# layer. Two sites read this tuple: the construction guard in
# `_create_common_layers` and the dispatch in `_pool_sequence`. It is declared
# once so the two cannot drift. A strategy listed in only one of them either
# builds a pooler nothing calls, or falls through to `_pool_sequence`'s
# "Unknown pooling type" raise. 'attention' is not here on purpose: it stays
# inline, for the reason given in the D-002 note below.
_DELEGATED_POOLING_STRATEGIES = ('cls', 'mean', 'max', 'last')

#: Every ``pooling_type`` a head accepts: the delegated four plus the inline
#: ``'attention'`` scorer. ``__init__`` refuses anything else.
_POOLING_TYPES = _DELEGATED_POOLING_STRATEGIES + ('attention',)


def _split_inputs(inputs: Any) -> Tuple[Any, Any]:
    """
    Return ``(hidden_states, attention_mask)`` from a head's input.

    Interface contract, shared by every NLP head's ``call``: a dict yields its
    ``'hidden_states'`` and its ``'attention_mask'`` (``None`` when absent);
    anything else is the hidden states themselves, with no mask.

    :param inputs: A tensor, or a dict with ``'hidden_states'``.
    :type inputs: Any
    :return: The hidden states and the mask or ``None``.
    :rtype: Tuple[Any, Any]
    """
    if isinstance(inputs, dict):
        return inputs['hidden_states'], inputs.get('attention_mask')
    return inputs, None


def _is_pair(value: Any) -> bool:
    """
    Tell a pair of inputs (or input shapes) from a single one.

    Interface contract, used by :class:`TextSimilarityHead`'s ``build``,
    ``compute_output_shape`` and ``call``: true for a tuple or list of two
    elements that are each a tensor, a dict, or a shape (a tuple or list).
    A single shape ``(batch, width)`` is a 2-tuple of integers (or ``None``)
    and is not a pair.

    :param value: An input, or an input shape.
    :type value: Any
    :return: Whether it is a pair.
    :rtype: bool
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(isinstance(v, (dict, list, tuple)) or hasattr(v, 'shape') for v in value)


def _apply_per_row(layer: keras.layers.Layer, x: Any, training: Optional[bool]) -> Any:
    """
    Apply ``layer`` to every vector along the last axis of ``x``.

    Interface contract, shared by the intermediate block of the token, QA,
    generation and multiple-choice heads: ``x`` of shape ``(..., width)`` is
    folded to ``(rows, width)``, run through ``layer``, and unfolded to
    ``(..., layer width)``. The leading axes may be dynamic. The output width
    is the layer's, not the input's.

    :param layer: The per-row layer (a ``DenseBlock``).
    :type layer: keras.layers.Layer
    :param x: A rank-3 or higher tensor.
    :type x: Any
    :param training: Keras training flag.
    :type training: Optional[bool]
    :return: ``layer`` applied per row, leading axes restored.
    :rtype: Any
    """
    shape = keras.ops.shape(x)
    flat = keras.ops.reshape(x, (-1, shape[-1]))
    flat = layer(flat, training=training)
    return keras.ops.reshape(flat, (*shape[:-1], -1))

# ---------------------------------------------------------------------
# Base NLP Head Class
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
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
    :type attention_type: AttentionType
    :param use_ffn: Whether to build an FFN block.
    :type use_ffn: bool
    :param ffn_type: Which registered FFN type to build.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: The FFN's inner width is
        ``hidden_size * ffn_expansion_factor``.
    :type ffn_expansion_factor: int
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
        of the five supported values, or a flag for a stage the subclass never
        runs; each names the head and the argument.
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
            attention_type: AttentionType = 'multi_head',
            use_ffn: bool = False,
            ffn_type: FFNType = 'mlp',
            ffn_expansion_factor: int = 4,
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
        :param attention_type: Which registered attention type to build.
        :type attention_type: AttentionType
        :param use_ffn: Whether to build an FFN block.
        :type use_ffn: bool
        :param ffn_type: Which registered FFN type to build.
        :type ffn_type: FFNType
        :param ffn_expansion_factor: FFN inner-width multiplier.
        :type ffn_expansion_factor: int
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
        self.attention_type = attention_type
        self.use_ffn = use_ffn
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
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
        rank = static_rank(hidden_states, 'hidden_states', owner)
        if rank == 2:
            if attention_mask is not None:
                raise ValueError(
                    f"{owner}: attention_mask was given for a rank-2 "
                    "hidden_states. A rank-2 input is already pooled, so there "
                    "is nothing to mask; drop the mask or pass the sequence."
                )
            return hidden_states
        if rank != 3:
            raise ValueError(
                f"{owner}: hidden_states must be rank 3 (a sequence) or rank 2 "
                f"(pooled), got rank {rank}."
            )
        if not self.use_pooling:
            raise ValueError(
                f"{owner}: a rank-3 hidden_states needs pooling, but the head "
                "was built with use_pooling=False. Pass a pooled (batch, width) "
                "input or build it with use_pooling=True."
            )
        return self._pool_sequence(hidden_states, attention_mask)

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
                keep = keras.ops.cast(attention_mask, dtype="bool")
                sentinel = keras.ops.cast(
                    mask_sentinel(self.compute_dtype), attention_weights.dtype
                )
                attention_weights = keras.ops.where(keep, attention_weights, sentinel)

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


# ---------------------------------------------------------------------
# Text Classification Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
class TextClassificationHead(BaseNLPHead):
    """
    One label for a whole sequence.

    Use this for sentiment analysis, topic classification, intent detection
    and the other sequence-level tasks. It also serves regression, by setting
    ``num_classes=1``.

    A rank-3 input is pooled to one vector per batch element first. A rank-2
    input is taken as already pooled and goes straight into the common stage.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, S, D), (B, D), or a dict
                        │
                        ▼
        ┌───────────────────────────────┐
        │ _pool_sequence     (optional) │ rank 3 only
        └───────────────┬───────────────┘
                        ▼ (B, D)
        ┌───────────────────────────────┐
        │ norm  ->  dropout             │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ intermediate       (optional) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ ffn                (optional) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ classifier: Dense(num_classes)│
        └───────────────┬───────────────┘
                        ▼ logits (B, num_classes)
                ┌───────┴────────┐
                ▼                ▼
             'logits'     'probabilities'
                          = softmax(logits)

    There is no task-attention stage: the features are rank 2 from the pooling
    on, so ``use_task_attention=True`` raises ``ValueError`` (D-062). The
    classifier is built at the width ``call`` feeds it: ``hidden_size`` after
    the intermediate block or the FFN, else ``input_dim``. With
    ``use_pooling=False`` the head takes only an already pooled ``(B, D)``
    input; a sequence raises ``ValueError``.

    Input shape:
        ``(batch, seq_length, input_dim)`` or ``(batch, input_dim)``, or a
        dict with ``'hidden_states'`` and an optional ``'attention_mask'``.

    Output shape:
        ``{'logits': (batch, num_classes),
        'probabilities': (batch, num_classes)}``.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``task_config`` and
        ``input_dim`` are required there.

    :ivar classifier: Output ``Dense(num_classes)`` layer.
    :vartype classifier: keras.layers.Dense

    :raises ValueError: If ``task_config.num_classes`` is ``None``, or
        ``use_task_attention`` is set.
    """

    _APPLIES_TASK_ATTENTION = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Build the base head, then the classifier.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_config.num_classes`` is ``None``.
        """
        super().__init__(**kwargs)

        if self.task_config.num_classes is None:
            raise ValueError("num_classes must be specified for classification tasks")

        # Classification layer, created in __init__ per the package rule.
        self.classifier = keras.layers.Dense(
            self.task_config.num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_classifier"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then the classifier.

        The classifier's input width is the one ``call`` feeds it,
        :meth:`BaseNLPHead._output_width`.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # First build common layers
        super().build(input_shape)

        classifier_input_dim = self._output_width()

        # Classifier receives features after processing
        batch_size = input_shape.get('hidden_states', (None,))[0] \
            if isinstance(input_shape, dict) else input_shape[0]
        classifier_input_shape = (batch_size, classifier_input_dim)
        self.classifier.build(classifier_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """
        Report the two output shapes.

        Both keys carry ``(batch, num_classes)``.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: Shapes of ``'logits'`` and ``'probabilities'``.
        :rtype: Dict[str, Tuple]
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
        else:
            batch_size = input_shape[0] if input_shape else None

        return {
            'logits': (batch_size, self.task_config.num_classes),
            'probabilities': (batch_size, self.task_config.num_classes)
        }

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Pool if needed, run the common stage, then classify.

        :param inputs: A sequence tensor, an already-pooled tensor, or a dict
            with ``'hidden_states'`` and an optional ``'attention_mask'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'logits', 'probabilities'}``. The probabilities are the
            softmax of the logits over the last axis.
        :rtype: Dict[str, keras.KerasTensor]
        """
        hidden_states, attention_mask = _split_inputs(inputs)

        # A sequence is pooled to one vector per row; a rank-2 input already is.
        hidden_states = self._pool_input(hidden_states, attention_mask)

        # Apply common processing
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            hidden_states = self.intermediate(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Classification
        logits = self.classifier(hidden_states)
        probabilities = keras.ops.softmax(logits, axis=-1)

        return {
            'logits': logits,
            'probabilities': probabilities
        }


# ---------------------------------------------------------------------
# Token Classification Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
class TokenClassificationHead(BaseNLPHead):
    """
    One label per token.

    Use this for named entity recognition, part-of-speech tagging and other
    token-level tasks. The head never pools, because the sequence axis has to
    survive to the output; ``pooling_type`` is validated but not read (a
    language model passes its default pooling to every head it builds).

    The output keys depend on ``task_config.use_crf``. With a CRF the head
    emits logits only, and the decoding is left to the CRF. Without one it
    also emits an argmax over the label axis.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, S, D) or a dict
                        │
                        ▼
        ┌───────────────────────────────┐
        │ norm  ->  dropout             │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ intermediate       (optional) │
        │ (B,S,D) -> (B*S,D) -> (B,S,H) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ task_attention     (optional) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ ffn                (optional) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ token_classifier: num_classes │
        └───────────────┬───────────────┘
                        ▼ (B, S, num_classes)
                ┌───────┴────────┐
              use_crf         no crf
                ▼                ▼
           {'logits'}      {'logits',
                            'predictions'}
                            = argmax(logits)

    Note:
        ``use_crf`` records the intent. No CRF layer is built here, so a
        CRF-flagged head currently just drops the ``'predictions'`` key.

    Input shape:
        ``(batch, seq_length, input_dim)``, or a dict with
        ``'hidden_states'``.

    Output shape:
        ``{'logits': (batch, seq_length, num_classes)}``, plus
        ``{'predictions': (batch, seq_length)}`` when ``use_crf`` is false.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``use_pooling`` defaults
        to ``False``; ``True`` raises ``ValueError``.

    :ivar token_classifier: Per-token ``Dense(num_classes)`` layer.
    :vartype token_classifier: keras.layers.Dense
    :ivar use_crf: Mirror of ``task_config.use_crf``.
    :vartype use_crf: bool

    :raises ValueError: If ``task_config.num_classes`` is ``None``.
    """

    _POOLS = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Default pooling off, then build the per-token classifier.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_config.num_classes`` is ``None``.
        """
        # A token head never pools; an explicit use_pooling=True is refused.
        kwargs.setdefault('use_pooling', False)
        super().__init__(**kwargs)

        if self.task_config.num_classes is None:
            raise ValueError("num_classes must be specified for token classification")

        # Token classifier, created in __init__ per the package rule.
        self.token_classifier = keras.layers.Dense(
            self.task_config.num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_token_classifier"
        )

        # There is no CRF layer here yet. The flag is recorded so the output
        # keys match what a CRF consumer expects; a real CRF would be a
        # separate layer.
        if self.task_config.use_crf:
            self.use_crf = True
        else:
            self.use_crf = False

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then the per-token classifier.

        The classifier's input width is the one ``call`` feeds it,
        :meth:`BaseNLPHead._output_width`; task attention keeps its width.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # First build common layers
        super().build(input_shape)

        # Determine input shape for the classifier
        if isinstance(input_shape, dict):
            seq_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            seq_shape = input_shape

        # The width call feeds the classifier (D-062).
        classifier_input_dim = self._output_width()

        # Classifier receives processed sequence
        classifier_input_shape = (seq_shape[0], seq_shape[1], classifier_input_dim)
        self.token_classifier.build(classifier_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """
        Report the output shapes, with or without ``'predictions'``.

        ``'predictions'`` is present only when ``use_crf`` is false, which
        matches what ``call`` returns.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: ``{'logits': ...}``, plus ``'predictions'`` when no CRF.
        :rtype: Dict[str, Tuple]
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
            seq_length = input_shape.get('hidden_states', (None, None))[1]
        else:
            batch_size = input_shape[0] if input_shape else None
            seq_length = input_shape[1] if len(input_shape) > 1 else None

        output_shapes = {
            'logits': (batch_size, seq_length, self.task_config.num_classes)
        }

        if not self.use_crf:
            output_shapes['predictions'] = (batch_size, seq_length)

        return output_shapes

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stage per token, then label every position.

        :param inputs: A sequence tensor, or a dict with ``'hidden_states'``.
            Any ``'attention_mask'`` in the dict is ignored by this head.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'logits'}``, plus ``'predictions'`` when ``use_crf`` is
            false.
        :rtype: Dict[str, keras.KerasTensor]
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            sequence_output = inputs['hidden_states']
        else:
            sequence_output = inputs

        # Apply processing to each token
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            # One row per token, then the original layout back.
            hidden_states = _apply_per_row(self.intermediate, hidden_states, training)

        if self.use_task_attention:
            hidden_states = self.task_attention(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Token classification
        logits = self.token_classifier(hidden_states)

        outputs = {'logits': logits}

        if not self.use_crf:
            # Simple argmax predictions
            predictions = keras.ops.argmax(logits, axis=-1)
            outputs['predictions'] = predictions

        return outputs


# ---------------------------------------------------------------------
# Question Answering Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
class QuestionAnsweringHead(BaseNLPHead):
    """
    Start and end scores for an extractive answer span.

    Two independent ``Dense(1)`` layers read the same processed sequence. Each
    produces one score per position, so the answer span is chosen by taking an
    argmax over each of them. The head never pools, because the sequence axis
    has to survive to the output.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, S, D) or a dict (+ attention_mask)
                        │
                        ▼
        ┌───────────────────────────────┐
        │ norm  ->  dropout             │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ intermediate       (optional) │
        │ (B,S,D) -> (B*S,D) -> (B,S,H) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ task_attention     (optional) │
        └───────────────┬───────────────┘
                        ▼
                ┌───────┴──────────┐
                ▼                   ▼
        ┌────────────────┐ ┌────────────────┐
        │ start Dense(1) │ │ end Dense(1)   │
        └───────┬────────┘ └───────┬────────┘
                ▼                   ▼
          squeeze -> (B,S)   squeeze -> (B,S)
                │                   │
                └───────┬──────────┘
                        ▼
        ┌───────────────────────────────┐
        │ where(keep, x, mask_sentinel) │
        │ (optional, needs the mask)    │
        └───────────────┬───────────────┘
                        ▼
          {'start_logits', 'end_logits'}

    There is no FFN stage: ``call`` stops applying common layers after
    ``task_attention``, so ``use_ffn=True`` raises ``ValueError`` (D-062).

    Input shape:
        ``(batch, seq_length, input_dim)``, or a dict with
        ``'hidden_states'`` and an optional ``'attention_mask'``.

    Output shape:
        ``{'start_logits': (batch, seq_length),
        'end_logits': (batch, seq_length)}``.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``use_pooling`` defaults
        to ``False``; ``True`` raises ``ValueError``. ``pooling_type`` is
        validated but not read.

    :ivar start_classifier: ``Dense(1)`` scoring the span start.
    :vartype start_classifier: keras.layers.Dense
    :ivar end_classifier: ``Dense(1)`` scoring the span end.
    :vartype end_classifier: keras.layers.Dense
    """

    _POOLS = False
    _APPLIES_FFN = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Default pooling off, then build the two span scorers.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        """
        # A span head never pools; an explicit use_pooling=True is refused.
        kwargs.setdefault('use_pooling', False)
        super().__init__(**kwargs)

        # Span scorers, created in __init__ per the package rule.
        self.start_classifier = keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_start"
        )

        self.end_classifier = keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_end"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then both span scorers.

        Both scorers take the same shape, at the width ``call`` feeds them,
        :meth:`BaseNLPHead._output_width`.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # First build common layers
        super().build(input_shape)

        # Determine input shape for the classifiers
        if isinstance(input_shape, dict):
            seq_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            seq_shape = input_shape

        # The width call feeds both scorers (D-062).
        classifier_input_dim = self._output_width()

        # Classifiers receive processed sequence
        classifier_input_shape = (seq_shape[0], seq_shape[1], classifier_input_dim)
        self.start_classifier.build(classifier_input_shape)
        self.end_classifier.build(classifier_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """
        Report the two span-score shapes.

        Both keys carry ``(batch, seq_length)``.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: Shapes of ``'start_logits'`` and ``'end_logits'``.
        :rtype: Dict[str, Tuple]
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
            seq_length = input_shape.get('hidden_states', (None, None))[1]
        else:
            batch_size = input_shape[0] if input_shape else None
            seq_length = input_shape[1] if len(input_shape) > 1 else None

        return {
            'start_logits': (batch_size, seq_length),
            'end_logits': (batch_size, seq_length)
        }

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stage, then score every position twice.

        When an ``'attention_mask'`` is given, masked positions are replaced
        by ``utils.dtype_policy.mask_sentinel`` in both score vectors so an
        argmax cannot land on padding. Selection, not an additive bias: the
        additive form NaNs every KEPT position under ``mixed_float16``.

        :param inputs: A sequence tensor, or a dict with ``'hidden_states'``
            and an optional ``'attention_mask'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'start_logits', 'end_logits'}``, both ``(batch,
            seq_length)``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        sequence_output, attention_mask = _split_inputs(inputs)

        # Apply processing
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            hidden_states = _apply_per_row(self.intermediate, hidden_states, training)

        if self.use_task_attention:
            hidden_states = self.task_attention(hidden_states, training=training)

        # Predict start and end positions
        start_logits = keras.ops.squeeze(self.start_classifier(hidden_states), axis=-1)
        end_logits = keras.ops.squeeze(self.end_classifier(hidden_states), axis=-1)

        # Push masked positions to the dtype's mask sentinel so an argmax
        # cannot select padding.
        if attention_mask is not None:
            # DECISION plan-2026-08-31T134711-6271592d/D-007
            # These are span logits read by an ARGMAX, not a softmax: the
            # sentinel only has to lose, so there is no rescue semantics and
            # `apply_attention_mask` is deliberately NOT used here. Select the
            # sentinel; the additive form NaNs every KEPT position under fp16.
            keep = keras.ops.cast(attention_mask, dtype="bool")
            sentinel = keras.ops.cast(
                mask_sentinel(self.compute_dtype), start_logits.dtype
            )
            start_logits = keras.ops.where(keep, start_logits, sentinel)
            end_logits = keras.ops.where(keep, end_logits, sentinel)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }


# ---------------------------------------------------------------------
# Text Similarity Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
class TextSimilarityHead(BaseNLPHead):
    """
    One embedding, or a similarity score for a pair of texts.

    The head has two input forms and returns a different dict for each. Give
    it a pair of sequences (a 2-tuple or a 2-list, what a functional model
    hands a two-input layer) and it scores the pair. Give it a single
    sequence and it returns that sequence's embedding, with no score.

    Each sequence may be a dict carrying its own ``'attention_mask'``; the
    pooling honours it, in both forms (D-063). Both forms share
    ``_process_sequence``, which runs ``norm``, ``dropout``, the optional
    ``intermediate`` and ``ffn``, then a ``projection`` Dense to
    ``hidden_size``. There is no task-attention stage, so
    ``use_task_attention=True`` raises ``ValueError`` (D-062).

    **Architecture Overview:**

    .. code-block:: text

        inputs: (seq1, seq2) or [seq1, seq2], or one tensor or dict
                            │
                 ┌──────────┴──────────┐
               pair                  single
                 ▼                    ▼
           pool each under       pool under its
           its own mask          mask if rank 3
                 │                    │
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ _process_sequence│ │ _process_sequence│
        │ twice: emb1,emb2 │ │ -> embeddings    │
        └────────┬─────────┘ └────────┬─────────┘
                 ▼                    ▼
        ┌──────────────────┐    {'embeddings'}
        │ similarity_fn    │
        │ (table below)    │
        └────────┬─────────┘
                 ▼
          {'similarity_score'}
          plus 'embeddings_1' and 'embeddings_2'
          when output_embeddings is set

    **Similarity Functions:**

    .. code-block:: text

        similarity_function   computation             out
        -------------------   ---------------------   ----
        'cosine'              sum(L2(e1) * L2(e2))    (B,)
        'dot'                 sum(e1 * e2)            (B,)
        'learned'             concat[e1, e2, e1*e2,   (B,)
                              abs(e1-e2)]
                              -> Dense(hidden_size)
                              -> Dense(1) -> squeeze

    The cosine branch divides by ``max(norm, 1e-8)``, so a zero vector gives a
    score of 0 rather than a NaN.

    Input shape:
        A pair (tuple or list) of ``(batch, seq_length, input_dim)`` tensors
        or dicts, or a single tensor of that shape or of ``(batch,
        input_dim)``, or a single dict. A dict may hold ``'attention_mask'``.

    Output shape:
        Pair form: ``{'similarity_score': (batch,)}``, plus
        ``'embeddings_1'`` and ``'embeddings_2'`` of ``(batch, hidden_size)``
        when ``output_embeddings`` is set. Single form:
        ``{'embeddings': (batch, hidden_size)}``.

    :param output_embeddings: Whether the pair form also returns the two
        embeddings. The single form always returns its embedding.
    :type output_embeddings: bool
    :param similarity_function: Which of the three scoring rules to use.
    :type similarity_function: Literal['cosine', 'dot', 'learned']
    :param kwargs: Passed to :class:`BaseNLPHead`.

    :ivar projection: ``Dense(hidden_size)`` applied to every embedding.
    :vartype projection: keras.layers.Dense
    :ivar similarity_layers: The two Dense layers of the ``'learned'`` rule.
        Empty for the other two rules. A flat list of layers round-trips
        through ``.keras`` by value (measured,
        ``test_every_nlp_head_survives_a_keras_round_trip.py``).
    :vartype similarity_layers: List[keras.layers.Dense]

    :raises ValueError: If ``similarity_function`` is none of the three, or
        ``use_task_attention`` is set.
    """

    _APPLIES_TASK_ATTENTION = False

    def __init__(
            self,
            output_embeddings: bool = True,
            similarity_function: Literal['cosine', 'dot', 'learned'] = 'cosine',
            **kwargs: Any
    ) -> None:
        """
        Build the base head, the projection, and the learned scorer.

        The two ``similarity_layers`` are built only for the ``'learned'``
        rule. For ``'cosine'`` and ``'dot'`` the list stays empty.

        :param output_embeddings: Whether the pair form also returns the two
            embeddings.
        :type output_embeddings: bool
        :param similarity_function: Which of the three scoring rules to use.
        :type similarity_function: Literal['cosine', 'dot', 'learned']
        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        """
        if similarity_function not in ('cosine', 'dot', 'learned'):
            raise ValueError(
                "TextSimilarityHead: similarity_function must be one of "
                f"['cosine', 'dot', 'learned'], got {similarity_function!r}."
            )
        super().__init__(**kwargs)

        self.output_embeddings = output_embeddings
        self.similarity_function = similarity_function

        # Projection layer, created in __init__ per the package rule.
        self.projection = keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_projection"
        )

        # Learned similarity function layers
        self.similarity_layers = []
        if similarity_function == 'learned':
            self.similarity_layers = [
                keras.layers.Dense(
                    self.hidden_size,
                    activation=self.activation_type,
                    name=f"{self.name}_sim_hidden"
                ),
                keras.layers.Dense(1, name=f"{self.name}_sim_output")
            ]

    def build(self, input_shape: Union[Tuple, Dict, List]) -> None:
        """
        Build the common layers, the projection, and the learned scorer.

        A 2-element input shape is the pair form. Both sides go through the
        same layers, so the first element's shape is enough to build them.

        :param input_shape: One shape, a dict of shapes, or a 2-element list
            or tuple of either for the pair form.
        :type input_shape: Union[Tuple, Dict, List]
        :return: None.
        :rtype: None
        """
        # A pair builds from its first side: both go through the same layers.
        if _is_pair(input_shape):
            base_input_shape = input_shape[0]
        else:
            base_input_shape = input_shape

        # First build common layers
        super().build(base_input_shape)

        # The width call feeds the projection (D-062).
        projection_input_dim = self._output_width()

        projection_input_shape = (None, projection_input_dim)
        self.projection.build(projection_input_shape)

        # Build similarity layers if needed
        if self.similarity_function == 'learned':
            # Combined features: emb1, emb2, emb1*emb2, abs(emb1-emb2)
            combined_input_shape = (None, self.hidden_size * 4)
            for layer in self.similarity_layers:
                layer.build(combined_input_shape)
                combined_input_shape = (None, layer.units if hasattr(layer, 'units') else 1)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict, List]) -> Dict[str, Tuple]:
        """
        Report the output shapes for whichever input form was given.

        A 2-element shape reports ``'similarity_score'``, plus the two
        embeddings when ``output_embeddings`` is set. Anything else reports a
        single ``'embeddings'`` entry.

        :param input_shape: One shape, a dict of shapes, or a 2-element list
            or tuple of either for the pair form.
        :type input_shape: Union[Tuple, Dict, List]
        :return: Output shapes for the matching input form.
        :rtype: Dict[str, Tuple]
        """
        # A pooled single shape (batch, width) is a 2-tuple too; _is_pair
        # tells them apart by what the two elements are.
        if _is_pair(input_shape):
            # Pairwise input
            if isinstance(input_shape[0], dict):
                batch_size = input_shape[0].get('hidden_states', (None,))[0]
            else:
                batch_size = input_shape[0][0] if input_shape[0] else None

            outputs = {'similarity_score': (batch_size,)}
            if self.output_embeddings:
                outputs['embeddings_1'] = (batch_size, self.hidden_size)
                outputs['embeddings_2'] = (batch_size, self.hidden_size)
            return outputs
        else:
            # Single input - return embeddings
            if isinstance(input_shape, dict):
                batch_size = input_shape.get('hidden_states', (None,))[0]
            else:
                batch_size = input_shape[0] if input_shape else None

            return {'embeddings': (batch_size, self.hidden_size)}

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor], Tuple],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Score a pair of sequences, or embed a single one.

        :param inputs: A pair (tuple or list) of sequences for the pair form,
            or a single sequence tensor or dict with ``'hidden_states'``. Each
            dict may carry ``'attention_mask'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor], Tuple]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'similarity_score'}`` for the pair form, with
            ``'embeddings_1'`` and ``'embeddings_2'`` when
            ``output_embeddings`` is set. ``{'embeddings'}`` for the single
            form.
        :rtype: Dict[str, keras.KerasTensor]
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-063
        # Each side is pooled under its own attention_mask, through
        # _pool_input and therefore SequencePooling (or the inline attention
        # scorer, D-002). Do NOT strip a dict side to its hidden_states: the
        # pair form scored identically with and without masks, and the single
        # form, the one every headed language model feeds, let padding move the
        # embedding by 0.026 (review NLP-2). See decisions.md D-063.
        if _is_pair(inputs):
            first, second = inputs
            emb1 = self._process_sequence(
                self._pool_input(*_split_inputs(first)), training)
            emb2 = self._process_sequence(
                self._pool_input(*_split_inputs(second)), training)

            # Compute similarity
            if self.similarity_function == 'cosine':
                # Cosine similarity. The floor keeps a zero vector from
                # producing a NaN.
                # DECISION plan-2026-08-31T134711-6271592d/D-009: derive the
                # floor from the compute dtype. A bare `1e-8` is exactly 0.0 in
                # float16, so `keras.ops.maximum(norm, 1e-8)` is `keras.ops.maximum(norm,
                # 0.0)` and an all-padding (zero) embedding divides 0/0 = NaN.
                # MEASURED at HEAD under mixed_float16: [nan nan].
                norm_floor = stability_floor(self.compute_dtype, 1e-8)
                emb1_norm = emb1 / keras.ops.maximum(
                    keras.ops.norm(emb1, axis=-1, keepdims=True), norm_floor
                )
                emb2_norm = emb2 / keras.ops.maximum(
                    keras.ops.norm(emb2, axis=-1, keepdims=True), norm_floor
                )
                similarity = keras.ops.sum(emb1_norm * emb2_norm, axis=-1)

            elif self.similarity_function == 'dot':
                # Dot product
                similarity = keras.ops.sum(emb1 * emb2, axis=-1)

            elif self.similarity_function == 'learned':
                # Learned similarity
                combined = keras.ops.concatenate([emb1, emb2, emb1 * emb2, keras.ops.abs(emb1 - emb2)], axis=-1)
                for layer in self.similarity_layers:
                    combined = layer(combined)
                similarity = keras.ops.squeeze(combined, axis=-1)

            outputs = {'similarity_score': similarity}
            if self.output_embeddings:
                outputs['embeddings_1'] = emb1
                outputs['embeddings_2'] = emb2

            return outputs

        else:
            # Single sequence - return embeddings
            pooled = self._pool_input(*_split_inputs(inputs))
            embeddings = self._process_sequence(pooled, training)

            return {'embeddings': embeddings}

    def _process_sequence(
            self,
            sequence: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run one pooled sequence through the shared layers.

        Applies ``norm``, ``dropout``, the optional ``intermediate`` and
        ``ffn``, then the ``projection`` Dense. ``task_attention`` is not
        applied here.

        :param sequence: Pooled features ``(batch, width)``.
        :type sequence: keras.KerasTensor
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: Embedding ``(batch, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        hidden_states = self.norm(sequence)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            hidden_states = self.intermediate(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Project to embedding space
        embeddings = self.projection(hidden_states)

        return embeddings

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        Adds this head's two extra arguments to the base config.

        :return: Config dict accepted by ``from_config``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_embeddings': self.output_embeddings,
            'similarity_function': self.similarity_function,
        })
        return config


# ---------------------------------------------------------------------
# Text Generation Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
class TextGenerationHead(BaseNLPHead):
    """
    Logits over the vocabulary, one distribution per position.

    Use this for autoregressive generation, masked language modeling,
    summarization and completion. The head never pools, because the sequence
    axis has to survive to the output.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, S, D) or a dict
                        │
                        ▼
        ┌───────────────────────────────┐
        │ norm  ->  dropout             │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ intermediate       (optional) │
        │ (B,S,D) -> (B*S,D) -> (B,S,H) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ ffn                (optional) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ lm_head -> vocabulary_size    │
        └───────────────┬───────────────┘
                        ▼
        {'logits'} (B, S, vocabulary_size)

    There is no task-attention stage: ``call`` goes straight from the
    intermediate block to the FFN, so ``use_task_attention=True`` raises
    ``ValueError`` (D-062).

    Input shape:
        ``(batch, seq_length, input_dim)``, or a dict with
        ``'hidden_states'``.

    Output shape:
        ``{'logits': (batch, seq_length, vocabulary_size)}``.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``use_pooling`` defaults
        to ``False``; ``True`` raises ``ValueError``. ``pooling_type`` is
        validated but not read.

    :ivar lm_head: Output ``Dense(vocabulary_size)`` layer.
    :vartype lm_head: keras.layers.Dense

    :raises ValueError: If ``task_config.vocabulary_size`` is ``None``.
    """

    _POOLS = False
    _APPLIES_TASK_ATTENTION = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Default pooling off, then build the language modeling head.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_config.vocabulary_size`` is ``None``.
        """
        # A generation head never pools; an explicit use_pooling=True is refused.
        kwargs.setdefault('use_pooling', False)
        super().__init__(**kwargs)

        if self.task_config.vocabulary_size is None:
            raise ValueError("vocabulary_size must be specified for generation tasks")

        # Language modeling head, created in __init__ per the package rule.
        self.lm_head = keras.layers.Dense(
            self.task_config.vocabulary_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_lm_head"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then the language modeling head.

        The head's input width is the one ``call`` feeds it,
        :meth:`BaseNLPHead._output_width`.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # First build common layers
        super().build(input_shape)

        # Determine input shape for the LM head
        if isinstance(input_shape, dict):
            seq_shape = input_shape.get('hidden_states', (None, None, self.input_dim))
        else:
            seq_shape = input_shape

        # The width call feeds the LM head (D-062).
        lm_input_dim = self._output_width()

        # LM head receives processed sequence
        lm_input_shape = (seq_shape[0], seq_shape[1], lm_input_dim)
        self.lm_head.build(lm_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """
        Report the vocabulary logits shape.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: ``{'logits': (batch, seq_length, vocabulary_size)}``.
        :rtype: Dict[str, Tuple]
        """
        if isinstance(input_shape, dict):
            batch_size = input_shape.get('hidden_states', (None,))[0]
            seq_length = input_shape.get('hidden_states', (None, None))[1]
        else:
            batch_size = input_shape[0] if input_shape else None
            seq_length = input_shape[1] if len(input_shape) > 1 else None

        return {'logits': (batch_size, seq_length, self.task_config.vocabulary_size)}

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stage per token, then score the vocabulary.

        There is no task-attention stage (``use_task_attention`` is refused).

        :param inputs: A sequence tensor, or a dict with ``'hidden_states'``.
            Any ``'attention_mask'`` in the dict is ignored by this head.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'logits'}`` of shape ``(batch, seq_length,
            vocabulary_size)``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            sequence_output = inputs['hidden_states']
        else:
            sequence_output = inputs

        # Apply processing
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            hidden_states = _apply_per_row(self.intermediate, hidden_states, training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Predict token logits
        logits = self.lm_head(hidden_states)

        return {'logits': logits}


# ---------------------------------------------------------------------
# Multiple Choice Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
class MultipleChoiceHead(BaseNLPHead):
    """
    One score per candidate answer.

    Every choice is scored independently by the same ``Dense(1)``, then the
    scores are softmaxed across the choice axis. The interesting part is the
    reshape ladder: the choice axis is folded into the batch axis so the
    pooler sees ordinary sequences, then unfolded again.

    ``B`` is the batch size, ``C`` the number of choices, ``S`` the sequence
    length and ``D`` the input width.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (B, C, S, D) or (B, C, D), or a dict
                          │
                  ┌───────┴────────┐
               rank 4            rank 3
                  ▼                │
        ┌────────────────────┐     │
        │ reshape to         │     │
        │ (B*C, S, D)        │     │
        └──────────┬─────────┘     │
                   ▼               │
        ┌────────────────────┐     │
        │ _pool_sequence     │     │
        │ -> (B*C, D)        │     │
        └──────────┬─────────┘     │
                   ▼               │
        ┌────────────────────┐     │
        │ reshape to (B,C,D) │     │
        └──────────┬─────────┘     │
                   └────┬──────────┘
                        ▼ pooled (B, C, D)
        ┌───────────────────────────────┐
        │ norm  ->  dropout             │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ intermediate       (optional) │
        │ (B,C,D) -> (B*C,D) -> (B,C,D) │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ scorer: Dense(1) -> squeeze   │
        └───────────────┬───────────────┘
                        ▼ logits (B, C)
                ┌───────┴────────┐
                ▼                ▼
             'logits'     'probabilities'
                          = softmax(logits)

    There is no task-attention stage and no FFN stage: ``call`` goes straight
    from the intermediate block to the scorer, so ``use_task_attention=True``
    and ``use_ffn=True`` raise ``ValueError`` (D-062). The intermediate block
    keeps its own output width, ``hidden_size`` (it used to be reshaped back to
    ``input_dim``, which failed whenever the two differed, review NLP-4).

    A dict input may carry ``'attention_mask'`` of shape ``(batch,
    num_choices, seq_length)``; it is folded with the hidden states, so each
    choice is pooled under its own mask (D-063, review NLP-3). An
    already-pooled input takes no mask.

    Input shape:
        ``(batch, num_choices, seq_length, input_dim)``, or an already-pooled
        ``(batch, num_choices, input_dim)``, or a dict with
        ``'hidden_states'`` and an optional ``'attention_mask'``.

    Output shape:
        ``{'logits': (batch, num_choices),
        'probabilities': (batch, num_choices)}``.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``task_config`` and
        ``input_dim`` are required there.

    :ivar scorer: ``Dense(1)`` applied to every choice.
    :vartype scorer: keras.layers.Dense
    """

    _APPLIES_TASK_ATTENTION = False
    _APPLIES_FFN = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Build the base head, then the per-choice scorer.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        # Per-choice scorer, created in __init__ per the package rule.
        self.scorer = keras.layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_scorer"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then the scorer.

        The scorer sees one row per choice, so only its width matters: the
        one ``call`` feeds it, :meth:`BaseNLPHead._output_width`.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # First build common layers
        super().build(input_shape)

        # The width call feeds the scorer (D-062).
        scorer_input_dim = self._output_width()

        # Scorer receives pooled representations for each choice
        scorer_input_shape = (None, scorer_input_dim)
        self.scorer.build(scorer_input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Tuple]:
        """
        Report the two output shapes.

        The number of choices is read from axis 1 of the input shape. Both
        keys carry ``(batch, num_choices)``.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: Shapes of ``'logits'`` and ``'probabilities'``.
        :rtype: Dict[str, Tuple]
        """
        if isinstance(input_shape, dict):
            hidden_shape = input_shape.get('hidden_states', (None, None))
        else:
            hidden_shape = input_shape

        batch_size = hidden_shape[0] if hidden_shape else None
        num_choices = hidden_shape[1] if len(hidden_shape) > 1 else None

        return {
            'logits': (batch_size, num_choices),
            'probabilities': (batch_size, num_choices)
        }

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Pool every choice, then score them against each other.

        A rank-4 input is folded to ``(batch * num_choices, seq_length,
        input_dim)``, pooled under the folded mask, and unfolded back. A rank-3
        input is taken as already pooled and skips all three steps.

        :param inputs: ``(batch, num_choices, seq_length, input_dim)``, an
            already-pooled ``(batch, num_choices, input_dim)``, or a dict with
            such a ``'hidden_states'`` entry and an optional
            ``'attention_mask'`` ``(batch, num_choices, seq_length)``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'logits', 'probabilities'}``, both ``(batch,
            num_choices)``. The probabilities are a softmax across choices.
        :rtype: Dict[str, keras.KerasTensor]
        """
        hidden_states, attention_mask = _split_inputs(inputs)
        owner = type(self).__name__

        # Rank 4 is [batch, choices, seq, hidden]; rank 3 is already pooled.
        rank = static_rank(hidden_states, 'hidden_states', owner)
        if rank == 4:
            if not self.use_pooling:
                raise ValueError(
                    f"{owner}: a rank-4 hidden_states needs pooling, but the "
                    "head was built with use_pooling=False. Pass pooled "
                    "(batch, num_choices, width) choices or build it with "
                    "use_pooling=True."
                )
            # Fold the choice axis into the batch axis so the pooler sees
            # ordinary sequences, each under its own mask row (D-063).
            shape = keras.ops.shape(hidden_states)
            folded = keras.ops.reshape(hidden_states, (-1, shape[2], shape[3]))
            folded_mask = None
            if attention_mask is not None:
                folded_mask = keras.ops.reshape(attention_mask, (-1, shape[2]))
            pooled = self._pool_sequence(folded, folded_mask)
            pooled = keras.ops.reshape(pooled, (shape[0], shape[1], -1))
        elif rank == 3:
            if attention_mask is not None:
                raise ValueError(
                    f"{owner}: attention_mask was given for rank-3 (already "
                    "pooled) hidden_states. There is nothing to mask; drop the "
                    "mask or pass the rank-4 choices."
                )
            pooled = hidden_states
        else:
            raise ValueError(
                f"{owner}: hidden_states must be rank 4 (choices of sequences) "
                f"or rank 3 (pooled choices), got rank {rank}."
            )

        # Process each choice
        hidden_states = self.norm(pooled)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            # One row per choice; the block's own width comes back (NLP-4).
            hidden_states = _apply_per_row(self.intermediate, hidden_states, training)

        # Score each choice
        logits = self.scorer(hidden_states)
        logits = keras.ops.squeeze(logits, axis=-1)

        return {
            'logits': logits,
            'probabilities': keras.ops.softmax(logits, axis=-1)
        }


# ---------------------------------------------------------------------
# Multi-Task NLP Head
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.heads.nlp.factory")
class MultiTaskNLPHead(keras.layers.Layer):
    """
    Several task heads behind one layer.

    One head is built per entry in ``task_configs``, using
    :func:`get_head_class` to pick the class. This layer does not inherit from
    :class:`BaseNLPHead`; it owns the sub-heads and routes to them.

    ``call`` takes an optional keyword-only ``task_name``. Given one, it runs
    that head alone and returns that head's own output dict. Given ``None``,
    it runs every head and returns a dict of those dicts, keyed by task name.
    ``compute_output_shape`` and the symbolic call follow ``task_name`` too.

    With ``use_task_specific_projections`` set, each task first goes through
    its own ``Dense`` projection, so the heads can work at different widths.

    The wrapper and every sub-head hold their own copies of the task configs
    (D-065): editing a config after construction changes nothing here. The
    caller's outer dict was never aliased (Keras stores a dict assigned to a
    layer as a ``TrackedDict`` copy, measured; review NLP-5 refuted for it).

    **Architecture Overview:**

    .. code-block:: text

        inputs (shared features), task_name
                        │
                 ┌──────┴──────────────────┐
           task_name given            task_name None
                 ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ known task?      │  │ loop every task  │
        │ else ValueError  │  │ in task_heads    │
        └────────┬─────────┘  └────────┬─────────┘
                 ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ task_projections │  │ task_projections │
        │ [name] (optional)│  │ [name] (optional)│
        └────────┬─────────┘  └────────┬─────────┘
                 ▼                     ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ task_heads[name] │  │ task_heads[name] │
        └────────┬─────────┘  └────────┬─────────┘
                 ▼                     ▼
          that head's dict     {task: head dict}
                               one entry per task

    Input shape:
        Whatever the sub-heads accept. A tensor is passed to every head
        unchanged. A dict is passed through with its ``'hidden_states'``
        entry replaced when a projection applies.

    Output shape:
        The selected head's output dict, or a dict of every head's output dict
        keyed by task name.

    :param task_configs: Task name to :class:`NLPTaskConfig`, or to a dict of
        its fields (the form ``get_config`` writes). One head is built per
        entry. Must not be empty.
    :type task_configs: Dict[str, Union[NLPTaskConfig, Dict[str, Any]]]
    :param shared_input_dim: Width of the shared features coming in.
    :type shared_input_dim: int
    :param use_task_specific_projections: Whether each task gets its own
        Dense projection before its head.
    :type use_task_specific_projections: bool
    :param kwargs: Additional arguments for the base Layer class.

    :ivar task_heads: Task name to head layer. A dict of layers round-trips
        through ``.keras`` by value (measured; Keras tracks it by key).
    :vartype task_heads: Dict[str, BaseNLPHead]
    :ivar task_projections: Task name to projection layer. Empty when
        ``use_task_specific_projections`` is false.
    :vartype task_projections: Dict[str, keras.layers.Dense]

    :raises ValueError: If ``task_configs`` is empty or not a dict, from
        :func:`get_head_class` when a task type has no head, and from ``call``
        when ``task_name`` is not a known task.
    """

    def __init__(
            self,
            task_configs: Dict[str, Union[NLPTaskConfig, Dict[str, Any]]],
            shared_input_dim: int,
            use_task_specific_projections: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Build one head, and optionally one projection, per task.

        :param task_configs: Task name to :class:`NLPTaskConfig` or a dict of
            its fields.
        :type task_configs: Dict[str, Union[NLPTaskConfig, Dict[str, Any]]]
        :param shared_input_dim: Width of the shared features coming in.
        :type shared_input_dim: int
        :param use_task_specific_projections: Whether each task gets its own
            Dense projection before its head.
        :type use_task_specific_projections: bool
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_configs`` is empty or not a dict, or a
            task type has no implemented head.
        """
        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError(
                "MultiTaskNLPHead: task_configs must be a non-empty dict of task "
                f"name to NLPTaskConfig, got {task_configs!r}."
            )
        shared_input_dim = positive_int(
            shared_input_dim, 'shared_input_dim', 'MultiTaskNLPHead')
        super().__init__(**kwargs)

        # DECISION plan-2026-10-14T062616-81938e81/D-065
        # Copy every config (NLPTaskConfig.coerce). Do NOT store the caller's
        # objects: setting num_classes=7 on one after construction left the
        # built classifier at 3 units while get_config reported 7, so a rebuild
        # disagreed with the saved weights (review NLP-5, inner configs). See
        # decisions.md D-065.
        configs = {name: NLPTaskConfig.coerce(cfg) for name, cfg in task_configs.items()}
        self.task_configs = configs
        self.shared_input_dim = shared_input_dim
        self.use_task_specific_projections = use_task_specific_projections

        # Heads and projections, created in __init__ per the package rule.
        self.task_heads = {}
        self.task_projections = {}

        for task_name, task_config in configs.items():
            # Create appropriate head
            head_class = get_head_class(task_config.task_type)

            # Set input dimension
            if use_task_specific_projections:
                projection_dim = task_config.hidden_size or shared_input_dim
                self.task_projections[task_name] = keras.layers.Dense(
                    projection_dim,
                    name=f"{task_name}_projection"
                )
                input_dim = projection_dim
            else:
                input_dim = shared_input_dim

            # Create head
            self.task_heads[task_name] = head_class(
                task_config=task_config,
                input_dim=input_dim
            )

    def _head_input_shape(self, task_name: str, input_shape: Union[Tuple, Dict]) -> Union[Tuple, Dict]:
        """
        Return the input shape a task's head sees, after its projection.

        :param task_name: A configured task name.
        :type task_name: str
        :param input_shape: Shape of the shared input, or a dict of shapes.
        :type input_shape: Union[Tuple, Dict]
        :return: The same shape, with the feature width replaced by the
            projection's when projections are on.
        :rtype: Union[Tuple, Dict]
        """
        if not self.use_task_specific_projections:
            return input_shape
        width = self.task_configs[task_name].hidden_size or self.shared_input_dim
        if isinstance(input_shape, dict):
            hidden = tuple(input_shape['hidden_states'])
            return {**input_shape, 'hidden_states': hidden[:-1] + (width,)}
        return tuple(input_shape)[:-1] + (width,)

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build every projection, then every head.

        With projections on, each head is built for the width its own
        projection emits rather than for ``shared_input_dim``.

        :param input_shape: Shape of the shared input, or a dict of shapes
            carrying a ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        hidden_shape = (input_shape['hidden_states']
                        if isinstance(input_shape, dict) else input_shape)
        for projection in self.task_projections.values():
            projection.build(tuple(hidden_shape))

        for task_name, head in self.task_heads.items():
            head.build(self._head_input_shape(task_name, input_shape))

        super().build(input_shape)

    def _selected_heads(self, task_name: Optional[str]) -> Dict[str, keras.layers.Layer]:
        """
        Return the heads a call runs: one for ``task_name``, else all of them.

        :param task_name: A configured task name, or ``None``.
        :type task_name: Optional[str]
        :return: Task name to head.
        :rtype: Dict[str, keras.layers.Layer]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        if task_name is None:
            return dict(self.task_heads)
        if task_name not in self.task_heads:
            raise ValueError(
                f"Unknown task: {task_name!r}. Configured tasks: "
                f"{sorted(self.task_heads)}."
            )
        return {task_name: self.task_heads[task_name]}

    def _project(self, task_name: str, inputs: Any, symbolic: bool = False) -> Any:
        """
        Apply a task's projection to the shared input, if projections are on.

        :param task_name: A configured task name.
        :type task_name: str
        :param inputs: The shared input, a tensor or a dict.
        :type inputs: Any
        :param symbolic: Use the projection's ``compute_output_spec`` (for a
            symbolic call) instead of calling it.
        :type symbolic: bool
        :return: The input the task's head takes.
        :rtype: Any
        """
        if not self.use_task_specific_projections:
            return inputs
        projection = self.task_projections[task_name]
        apply = projection.compute_output_spec if symbolic else projection
        if isinstance(inputs, dict):
            return {**inputs, 'hidden_states': apply(inputs['hidden_states'])}
        return apply(inputs)

    def compute_output_shape(
            self,
            input_shape: Union[Tuple, Dict],
            *,
            task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the output shapes ``call`` returns for the same ``task_name``.

        :param input_shape: Shape of the shared input, or a dict of shapes
            carrying a ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output shapes, or task name to output shapes.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        shapes = {
            name: head.compute_output_shape(self._head_input_shape(name, input_shape))
            for name, head in self._selected_heads(task_name).items()
        }
        return shapes[task_name] if task_name is not None else shapes

    def compute_output_spec(
            self,
            inputs: Any,
            training: Optional[bool] = None,
            *,
            task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return the symbolic outputs ``call`` returns for the same ``task_name``.

        Keras' default routes a symbolic call through ``compute_output_shape``
        with the input shapes only, so ``task_name`` never reached it and a
        symbolic call returned every task's outputs (the D-060 defect of the
        VLM wrapper). Each selected head reports its own spec instead.

        :param inputs: Symbolic shared inputs.
        :type inputs: Any
        :param training: Unused; accepted because ``call`` takes it.
        :type training: Optional[bool]
        :param task_name: Keyword only, as in ``call``.
        :type task_name: Optional[str]
        :return: That head's output spec, or task name to output spec.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``task_name`` names no configured task.
        """
        specs = {
            name: head.compute_output_spec(self._project(name, inputs, symbolic=True))
            for name, head in self._selected_heads(task_name).items()
        }
        return specs[task_name] if task_name is not None else specs

    # DECISION plan-2026-10-14T062616-81938e81/D-065
    # task_name is keyword-only and training keeps the second positional slot,
    # as in MultiTaskVLMHead (D-060). Do NOT move task_name back in front of
    # training: a positional training flag landed in task_name. See
    # decisions.md D-065.
    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None,
            *,
            task_name: Optional[str] = None,
    ) -> Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]:
        """
        Run one task head, or all of them.

        :param inputs: Shared features from the foundation model, as a tensor
            or as a dict with ``'hidden_states'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :param task_name: Keyword only. Which task to run; ``None`` runs every
            task.
        :type task_name: Optional[str]
        :return: One head's output dict when ``task_name`` is given. A dict of
            every head's output dict, keyed by task name, when it is ``None``.
        :rtype: Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]
        :raises ValueError: If ``task_name`` is not a known task.
        """
        outputs = {
            name: head(self._project(name, inputs), training=training)
            for name, head in self._selected_heads(task_name).items()
        }
        return outputs[task_name] if task_name is not None else outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        Each :class:`NLPTaskConfig` is written as a plain dict of every field
        (:meth:`NLPTaskConfig.to_dict`). There is no ``from_config`` override:
        ``__init__`` takes those dicts as they are, so the caller's config is
        never modified.

        :return: Config dict accepted by ``from_config``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'task_configs': {
                name: tc.to_dict() for name, tc in self.task_configs.items()
            },
            'shared_input_dim': self.shared_input_dim,
            'use_task_specific_projections': self.use_task_specific_projections,
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

#: Head class name, as :data:`NLP_HEAD_DISPATCH` writes it, to the class.
_HEAD_CLASSES: Dict[str, type] = {
    cls.__name__: cls for cls in (
        TextClassificationHead,
        TokenClassificationHead,
        QuestionAnsweringHead,
        TextSimilarityHead,
        TextGenerationHead,
        MultipleChoiceHead,
    )
}


def get_head_class(task_type: Union[NLPTaskType, str]) -> type:
    """
    Look up the head class for a task type.

    The class comes from :data:`~.task_types.NLP_HEAD_DISPATCH`, the one
    table from task type to head (D-064). It covers 24 of the 37 members of
    :class:`NLPTaskType`. **The other 13 raise ``ValueError``.** That is a
    guard, not a gap in the table. This function used to end with
    ``head_mapping.get(task_type, TextClassificationHead)``. Asking for
    ``MACHINE_TRANSLATION`` then returned a text classifier that built, trained
    and produced plausible nonsense. Nothing ever failed. A task with no head
    is a caller error and is now reported as one.

    To add support for one of the 13, implement or choose a head and add its
    row to the table. Do not restore the fallback.

    **Architecture Overview:**

    .. code-block:: text

        task_type (member or string value)
            │
            ▼
        ┌──────────────────────┐        ┌────────────┐
        │ in NLP_HEAD_DISPATCH?│─ no ──►│ ValueError │
        └──────────┬───────────┘        └────────────┘
                   │ yes
                   ▼
        ┌──────────────────────────────────────┐
        │ one of the six leaf head classes:    │
        │ TextClassificationHead  (10 tasks)   │
        │ TokenClassificationHead  (4 tasks)   │
        │ TextGenerationHead       (4 tasks)   │
        │ TextSimilarityHead       (3 tasks)   │
        │ QuestionAnsweringHead    (2 tasks)   │
        │ MultipleChoiceHead       (1 task)    │
        └──────────────────────────────────────┘

    :param task_type: The NLP task type, as a member or its string value.
    :type task_type: Union[NLPTaskType, str]
    :return: The head class for that task type.
    :rtype: type
    :raises ValueError: If no head is implemented for ``task_type``, or the
        string names no task type. The message lists every supported task
        type.
    """
    if isinstance(task_type, str):
        task_type = NLPTaskType.from_string(task_type)
    head_name = NLP_HEAD_DISPATCH.get(task_type)
    # No silent fallback (see the docstring). A translation task quietly
    # became a classifier, and nothing failed. To add one of the 13 missing
    # types, add its row to NLP_HEAD_DISPATCH on purpose.
    if head_name is None:
        supported = sorted(t.name for t in NLP_HEAD_DISPATCH)
        raise ValueError(
            f"No NLP head is implemented for task type '{task_type.name}'. "
            f"Supported task types: {supported}. "
            f"(This previously returned a TextClassificationHead silently, which produced "
            f"a working but wrong head for tasks like MACHINE_TRANSLATION.)"
        )
    return _HEAD_CLASSES[head_name]


def create_nlp_head(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        input_dim: int,
        **kwargs: Any
) -> BaseNLPHead:
    """
    Build one NLP head from a task config.

    The task type in the config picks the class through
    :func:`get_head_class`. A plain dict is converted to an
    :class:`NLPTaskConfig` first, and may give ``task_type`` as its string
    value. Extra keyword arguments go to the head's constructor unchanged, so
    anything :class:`BaseNLPHead` accepts can be set here.

    **Architecture Overview:**

    .. code-block:: text

        task_config (NLPTaskConfig or dict)
                │
                ▼
        ┌──────────────────────────┐
        │ NLPTaskConfig.coerce     │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ get_head_class(task_type)│
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ head_class(task_config,  │
        │   input_dim, **kwargs)   │
        └──────────────────────────┘

    :param task_config: An :class:`NLPTaskConfig`, or a dict of its fields.
    :type task_config: Union[NLPTaskConfig, Dict[str, Any]]
    :param input_dim: Width of the features the foundation model emits.
    :type input_dim: int
    :param kwargs: Extra keyword arguments for the head constructor.
    :return: A configured head for that task.
    :rtype: BaseNLPHead
    :raises ValueError: If the task type has no implemented head, or a field
        or argument is invalid.
    :raises TypeError: If ``task_config`` is neither form.
    """
    task_config = NLPTaskConfig.coerce(task_config)
    head_class = get_head_class(task_config.task_type)
    return head_class(task_config=task_config, input_dim=input_dim, **kwargs)


def create_multi_task_nlp_head(
        task_configs: Union[List[Union[NLPTaskConfig, Dict[str, Any]]],
                            Dict[str, Union[NLPTaskConfig, Dict[str, Any]]]],
        input_dim: int,
        **kwargs: Any
) -> MultiTaskNLPHead:
    """
    Build a :class:`MultiTaskNLPHead` from several task configs.

    A list is turned into a dict keyed by each config's ``name``; two configs
    with one name are refused, because keying would keep only the last. Pass a
    dict directly to choose the keys yourself. Either form may hold dicts of
    config fields. An empty list or dict is refused.

    :param task_configs: A list of configs, or a dict of task name to config.
    :type task_configs: Union[List[...], Dict[str, ...]]
    :param input_dim: Width of the shared features. Becomes the head's
        ``shared_input_dim``.
    :type input_dim: int
    :param kwargs: Extra keyword arguments for :class:`MultiTaskNLPHead`.
    :return: The configured multi-task head.
    :rtype: MultiTaskNLPHead
    :raises ValueError: If the list repeats a name, the set is empty, or any
        task type has no implemented head.
    """
    if isinstance(task_configs, list):
        configs = [NLPTaskConfig.coerce(c) for c in task_configs]
        names = [c.name for c in configs]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"create_multi_task_nlp_head: task names {repeated} appear more "
                "than once; a list is keyed by name, so all but the last would "
                "be dropped. Give each config its own name, or pass a dict."
            )
        task_configs = {c.name: c for c in configs}

    return MultiTaskNLPHead(
        task_configs=task_configs,
        shared_input_dim=input_dim,
        **kwargs
    )
