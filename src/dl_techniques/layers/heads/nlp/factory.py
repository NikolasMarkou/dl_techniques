"""NLP task heads, and the factory that picks one.

This module holds eight layer classes and four module-level helpers. The
classes turn a foundation model's hidden states into task predictions. The
helpers pick a class and configure it.

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
* :func:`get_head_class` -- task type to head class. Raises for a task type
  that has no head.
* :func:`create_nlp_head` -- build one head from a config.
* :func:`create_multi_task_nlp_head` -- build a multi-task head.
* :class:`NLPHeadConfiguration` -- default keyword arguments per task type.

The task types and the config dataclass live in ``nlp/task_types.py``.

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
from keras import layers, ops
from typing import Dict, List, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...activations import ActivationType
from ...standard_blocks import DenseBlock
from ...ffn import create_ffn_layer, FFNType
from ...attention import AttentionType
from ...attention.factory import (
    create_attention_layer,
    assemble_attention_config,
)
from ...norms import create_normalization_layer, NormalizationType
from ...sequence_pooling import SequencePooling

from .task_types import NLPTaskType, NLPTaskConfig
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

    Not every subclass runs every stage, whatever its flags say. Three never
    apply ``task_attention``: :class:`TextGenerationHead`,
    :class:`TextSimilarityHead` and :class:`MultipleChoiceHead`. Two never
    apply ``ffn``: :class:`QuestionAnsweringHead` and
    :class:`MultipleChoiceHead`. So a flag can build a layer the head will
    never call. Read the subclass diagram, not this one, for what a given head
    actually runs.

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
    interchangeable: ``SequencePooling('attention')`` is ``AttentionPooling``,
    which scores a ``Dense(hidden, tanh)`` projection against a learnable
    context vector, so it has both a different mechanism and a different
    weight set than the ``Dense(1, tanh)`` scorer here.

    Delegating ``cls``, ``mean`` and ``max`` was checked against the inline
    code it replaced. Mask-aware mean and max agree with the old values within
    atol 1e-6 for any sequence with at least one valid token. That check is
    the reason the delegation is safe, and it is recorded here because the
    plan that made the decision no longer exists.

    :param task_config: NLPTaskConfig object with task configuration.
    :type task_config: NLPTaskConfig
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
        layer. Three subclasses build it but never call it; see above.
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

    :raises ValueError: From ``_pool_sequence``, when ``pooling_type`` is not
        one of the five supported values.
    """

    def __init__(
            self,
            task_config: NLPTaskConfig,
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

        See the class docstring for what each parameter means. Every argument
        is stored on the instance, then ``_create_common_layers`` builds the
        shared layers.

        :param task_config: NLPTaskConfig object with task configuration.
        :type task_config: NLPTaskConfig
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

        # Create common layers (following Golden Rule: CREATE in __init__)
        self._create_common_layers()

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
        self.dropout = layers.Dropout(
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
            self.attention_pooling = layers.Dense(
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
            attn_defaults = {'dim': self.hidden_size}
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
            to ``SequencePooling``, and used to push masked scores to -1e9 in
            the inline branch.
        :type attention_mask: Optional[keras.KerasTensor]
        :return: Pooled representation ``(batch, hidden_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``pooling_type`` is none of the five supported
            values.
        """
        # DECISION plan_2026-06-08_8b32ca51/D-002: cls/mean/max delegate to the
        # shared SequencePooling facade (created in _create_common_layers).
        # The 'attention' branch stays inline — do NOT route it through
        # SequencePooling (different mechanism + weights). See decisions.md.
        if self.pooling_type in _DELEGATED_POOLING_STRATEGIES:
            # The `mask=` argument matters for 'last' (D-023). It is what makes
            # the gather land on the last real token rather than on padding.
            return self.sequence_pooler(sequence, mask=attention_mask)

        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_pooling(sequence)
            attention_weights = ops.squeeze(attention_weights, axis=-1)

            if attention_mask is not None:
                mask = ops.cast(attention_mask, dtype=attention_weights.dtype)
                attention_weights = attention_weights * mask + (1 - mask) * -1e9

            attention_weights = ops.softmax(attention_weights, axis=-1)
            attention_weights = ops.expand_dims(attention_weights, axis=-1)

            return ops.sum(sequence * attention_weights, axis=1)

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
            # Task attention operates on normalized features
            self.task_attention.build((hidden_shape[0], hidden_shape[1], self.hidden_size))

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
            ffn_input = (hidden_shape[0], hidden_shape[1] if len(hidden_shape) == 3 else 1, self.hidden_size)
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

        ``task_config`` is a dataclass, so it is flattened to a plain dict of
        its ten serializable fields. :meth:`from_config` rebuilds it.

        :return: Config dict accepted by :meth:`from_config`.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'task_config': {
                'name': self.task_config.name,
                'task_type': self.task_config.task_type.value,
                'num_classes': self.task_config.num_classes,
                'dropout_rate': self.task_config.dropout_rate,
                'hidden_size': self.task_config.hidden_size,
                'loss_weight': self.task_config.loss_weight,
                'label_smoothing': self.task_config.label_smoothing,
                'use_crf': self.task_config.use_crf,
                'use_attention_pooling': self.task_config.use_attention_pooling,
                'vocabulary_size': getattr(self.task_config, 'vocabulary_size', None),
            },
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseNLPHead":
        """
        Rebuild a head from a saved config.

        The flattened ``task_config`` dict is turned back into an
        :class:`NLPTaskConfig` before the class is constructed.

        :param config: Config dict produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new head of this class.
        :rtype: BaseNLPHead
        """
        # Reconstruct task config
        task_config_dict = config.pop('task_config')
        task_config = NLPTaskConfig(
            name=task_config_dict['name'],
            task_type=NLPTaskType(task_config_dict['task_type']),
            num_classes=task_config_dict.get('num_classes'),
            dropout_rate=task_config_dict.get('dropout_rate', 0.1),
            hidden_size=task_config_dict.get('hidden_size'),
            loss_weight=task_config_dict.get('loss_weight', 1.0),
            label_smoothing=task_config_dict.get('label_smoothing', 0.0),
            use_crf=task_config_dict.get('use_crf', False),
            use_attention_pooling=task_config_dict.get('use_attention_pooling', False),
            vocabulary_size=task_config_dict.get('vocabulary_size'),
        )
        config['task_config'] = task_config
        return cls(**config)


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
        │ task_attention     (optional) │ rank 3 only
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

    ``task_attention`` is built when ``use_task_attention`` is set, but it
    only runs while the features are still rank 3. After pooling they are rank
    2, so with the default ``use_pooling=True`` it never runs.

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

    :raises ValueError: If ``task_config.num_classes`` is ``None``.
    """

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
        self.classifier = layers.Dense(
            self.task_config.num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_classifier"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then the classifier.

        The classifier's input width is ``hidden_size`` when any of
        ``ffn``, ``task_attention`` or ``intermediate`` is enabled, because
        each of those rewrites the feature width. Otherwise it is
        ``input_dim``.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # First build common layers
        super().build(input_shape)

        # The dimension depends on whether transformative layers are used.
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            classifier_input_dim = self.hidden_size
        else:
            classifier_input_dim = self.input_dim

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
        # Handle different input formats
        if isinstance(inputs, dict):
            hidden_states = inputs['hidden_states']
            attention_mask = inputs.get('attention_mask', None)
        else:
            hidden_states = inputs
            attention_mask = None

        # Pool a sequence input down to one vector per batch element.
        # Rank 3 is [batch, seq_len, hidden]; anything else is already pooled.
        if len(ops.shape(hidden_states)) == 3:
            hidden_states = self._pool_sequence(hidden_states, attention_mask)

        # Apply common processing
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            hidden_states = self.intermediate(hidden_states, training=training)

        if self.use_task_attention and len(ops.shape(hidden_states)) == 3:
            hidden_states = self.task_attention(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Classification
        logits = self.classifier(hidden_states)
        probabilities = ops.softmax(logits, axis=-1)

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
    token-level tasks. Pooling is forced off, because the sequence axis has to
    survive to the output.

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

    :param kwargs: Passed to :class:`BaseNLPHead`. ``use_pooling`` is
        overwritten with ``False`` before it gets there.

    :ivar token_classifier: Per-token ``Dense(num_classes)`` layer.
    :vartype token_classifier: keras.layers.Dense
    :ivar use_crf: Mirror of ``task_config.use_crf``.
    :vartype use_crf: bool

    :raises ValueError: If ``task_config.num_classes`` is ``None``.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Force pooling off, then build the per-token classifier.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_config.num_classes`` is ``None``.
        """
        # Token classification doesn't use pooling
        kwargs['use_pooling'] = False
        super().__init__(**kwargs)

        if self.task_config.num_classes is None:
            raise ValueError("num_classes must be specified for token classification")

        # Token classifier, created in __init__ per the package rule.
        self.token_classifier = layers.Dense(
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

        The classifier's input width is ``hidden_size`` when any of ``ffn``,
        ``task_attention`` or ``intermediate`` is enabled, because each of
        those rewrites the feature width. Otherwise it is ``input_dim``.

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

        # Determine the correct input dimension for the classifier
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            classifier_input_dim = self.hidden_size
        else:
            classifier_input_dim = self.input_dim

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
            # Flatten the sequence axis so the dense block sees one row per
            # token, then restore the original layout.
            batch_size, seq_len, hidden_dim = ops.shape(hidden_states)
            hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))
            hidden_states_flat = self.intermediate(hidden_states_flat, training=training)
            hidden_states = ops.reshape(hidden_states_flat, (batch_size, seq_len, -1))

        if self.use_task_attention:
            hidden_states = self.task_attention(hidden_states, training=training)

        if self.use_ffn:
            hidden_states = self.ffn(hidden_states, training=training)

        # Token classification
        logits = self.token_classifier(hidden_states)

        outputs = {'logits': logits}

        if not self.use_crf:
            # Simple argmax predictions
            predictions = ops.argmax(logits, axis=-1)
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
    argmax over each of them. Pooling is forced off, because the sequence axis
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
        │ mask fill  x*m + (1-m)*-1e9   │
        │ (optional, needs the mask)    │
        └───────────────┬───────────────┘
                        ▼
          {'start_logits', 'end_logits'}

    There is no FFN stage. ``call`` stops applying common layers after
    ``task_attention``, so ``use_ffn`` builds a block this head never runs.

    Input shape:
        ``(batch, seq_length, input_dim)``, or a dict with
        ``'hidden_states'`` and an optional ``'attention_mask'``.

    Output shape:
        ``{'start_logits': (batch, seq_length),
        'end_logits': (batch, seq_length)}``.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``use_pooling`` is
        overwritten with ``False`` before it gets there.

    :ivar start_classifier: ``Dense(1)`` scoring the span start.
    :vartype start_classifier: keras.layers.Dense
    :ivar end_classifier: ``Dense(1)`` scoring the span end.
    :vartype end_classifier: keras.layers.Dense
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Force pooling off, then build the two span scorers.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        """
        # QA doesn't use pooling
        kwargs['use_pooling'] = False
        super().__init__(**kwargs)

        # Span scorers, created in __init__ per the package rule.
        self.start_classifier = layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_start"
        )

        self.end_classifier = layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_end"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then both span scorers.

        Both scorers take the same shape. Its width is ``hidden_size`` when any
        of ``ffn``, ``task_attention`` or ``intermediate`` is enabled, and
        ``input_dim`` otherwise.

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

        # Determine the correct input dimension for the classifiers.
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            classifier_input_dim = self.hidden_size
        else:
            classifier_input_dim = self.input_dim

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

        When an ``'attention_mask'`` is given, masked positions are pushed to
        -1e9 in both score vectors so an argmax cannot land on padding.

        :param inputs: A sequence tensor, or a dict with ``'hidden_states'``
            and an optional ``'attention_mask'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'start_logits', 'end_logits'}``, both ``(batch,
            seq_length)``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            sequence_output = inputs['hidden_states']
            attention_mask = inputs.get('attention_mask', None)
        else:
            sequence_output = inputs
            attention_mask = None

        # Apply processing
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            batch_size, seq_len, hidden_dim = ops.shape(hidden_states)
            hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))
            hidden_states_flat = self.intermediate(hidden_states_flat, training=training)
            hidden_states = ops.reshape(hidden_states_flat, (batch_size, seq_len, -1))

        if self.use_task_attention:
            hidden_states = self.task_attention(hidden_states, training=training)

        # Predict start and end positions
        start_logits = ops.squeeze(self.start_classifier(hidden_states), axis=-1)
        end_logits = ops.squeeze(self.end_classifier(hidden_states), axis=-1)

        # Push masked positions to -1e9 so an argmax cannot select padding.
        if attention_mask is not None:
            # Cast attention_mask to the same dtype as logits
            mask = ops.cast(attention_mask, dtype=start_logits.dtype)
            start_logits = start_logits * mask + (1 - mask) * -1e9
            end_logits = end_logits * mask + (1 - mask) * -1e9

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
    it a 2-tuple of sequences and it scores the pair. Give it a single
    sequence and it returns that sequence's embedding, with no score.

    Both forms share ``_process_sequence``, which runs ``norm``, ``dropout``,
    the optional ``intermediate`` and ``ffn``, then a ``projection`` Dense to
    ``hidden_size``. Neither form applies ``task_attention``, whatever
    ``use_task_attention`` says.

    **Architecture Overview:**

    .. code-block:: text

        inputs: (seq1, seq2) tuple, or one tensor or dict
                            │
                 ┌──────────┴──────────┐
              2-tuple                single
                 ▼                    ▼
           pool seq1, seq2       pool if rank 3
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
        A 2-tuple of ``(batch, seq_length, input_dim)`` tensors or dicts, or a
        single tensor of that shape, or a single dict.

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
        Empty for the other two rules.
    :vartype similarity_layers: List[keras.layers.Dense]
    """

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
        super().__init__(**kwargs)

        self.output_embeddings = output_embeddings
        self.similarity_function = similarity_function

        # Projection layer, created in __init__ per the package rule.
        self.projection = layers.Dense(
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
                layers.Dense(
                    self.hidden_size,
                    activation=self.activation_type,
                    name=f"{self.name}_sim_hidden"
                ),
                layers.Dense(1, name=f"{self.name}_sim_output")
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
        # Handle tuple input (pairwise) by using first element shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            base_input_shape = input_shape[0]
        else:
            base_input_shape = input_shape

        # First build common layers
        super().build(base_input_shape)

        # Input width for the projection layer. task_attention is not part of
        # the condition because this head never applies it.
        if self.use_ffn or self.use_intermediate:
            projection_input_dim = self.hidden_size
        else:
            projection_input_dim = self.input_dim

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
        # Handle different input formats
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
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

        :param inputs: A 2-tuple of sequences for the pair form, or a single
            sequence tensor or dict with ``'hidden_states'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor], Tuple]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'similarity_score'}`` for the pair form, with
            ``'embeddings_1'`` and ``'embeddings_2'`` when
            ``output_embeddings`` is set. ``{'embeddings'}`` for the single
            form.
        :rtype: Dict[str, keras.KerasTensor]
        """
        # Handle different input formats
        if isinstance(inputs, tuple) and len(inputs) == 2:
            # Pairwise similarity
            seq1, seq2 = inputs
            if isinstance(seq1, dict):
                seq1 = seq1['hidden_states']
            if isinstance(seq2, dict):
                seq2 = seq2['hidden_states']

            # Pool sequences
            if len(ops.shape(seq1)) == 3:
                seq1 = self._pool_sequence(seq1)
                seq2 = self._pool_sequence(seq2)

            # Process sequences
            emb1 = self._process_sequence(seq1, training)
            emb2 = self._process_sequence(seq2, training)

            # Compute similarity
            if self.similarity_function == 'cosine':
                # Cosine similarity. The 1e-8 floor keeps a zero vector from
                # producing a NaN.
                emb1_norm = emb1 / ops.maximum(ops.norm(emb1, axis=-1, keepdims=True), 1e-8)
                emb2_norm = emb2 / ops.maximum(ops.norm(emb2, axis=-1, keepdims=True), 1e-8)
                similarity = ops.sum(emb1_norm * emb2_norm, axis=-1)

            elif self.similarity_function == 'dot':
                # Dot product
                similarity = ops.sum(emb1 * emb2, axis=-1)

            elif self.similarity_function == 'learned':
                # Learned similarity
                combined = ops.concatenate([emb1, emb2, emb1 * emb2, ops.abs(emb1 - emb2)], axis=-1)
                for layer in self.similarity_layers:
                    combined = layer(combined)
                similarity = ops.squeeze(combined, axis=-1)

            outputs = {'similarity_score': similarity}
            if self.output_embeddings:
                outputs['embeddings_1'] = emb1
                outputs['embeddings_2'] = emb2

            return outputs

        else:
            # Single sequence - return embeddings
            if isinstance(inputs, dict):
                hidden_states = inputs['hidden_states']
            else:
                hidden_states = inputs

            if len(ops.shape(hidden_states)) == 3:
                hidden_states = self._pool_sequence(hidden_states)

            embeddings = self._process_sequence(hidden_states, training)

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
    summarization and completion. Pooling is forced off, because the sequence
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

    There is no task-attention stage. ``call`` goes straight from the
    intermediate block to the FFN, so setting ``use_task_attention=True``
    builds an attention layer this head never runs. Do not add the stage to
    the diagram from the base class; read ``call``.

    Input shape:
        ``(batch, seq_length, input_dim)``, or a dict with
        ``'hidden_states'``.

    Output shape:
        ``{'logits': (batch, seq_length, vocabulary_size)}``.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``use_pooling`` is
        overwritten with ``False`` before it gets there.

    :ivar lm_head: Output ``Dense(vocabulary_size)`` layer.
    :vartype lm_head: keras.layers.Dense

    :raises ValueError: If ``task_config.vocabulary_size`` is ``None``.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Force pooling off, then build the language modeling head.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``task_config.vocabulary_size`` is ``None``.
        """
        # Generation doesn't use pooling
        kwargs['use_pooling'] = False
        super().__init__(**kwargs)

        if self.task_config.vocabulary_size is None:
            raise ValueError("vocabulary_size must be specified for generation tasks")

        # Language modeling head, created in __init__ per the package rule.
        self.lm_head = layers.Dense(
            self.task_config.vocabulary_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_lm_head"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then the language modeling head.

        The head's input width is ``hidden_size`` when any of ``ffn``,
        ``task_attention`` or ``intermediate`` is enabled, and ``input_dim``
        otherwise.

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

        # Determine the correct input dimension for the LM head.
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            lm_input_dim = self.hidden_size
        else:
            lm_input_dim = self.input_dim

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

        ``task_attention`` is never applied here, even when
        ``use_task_attention`` is set.

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
            batch_size, seq_len, hidden_dim = ops.shape(hidden_states)
            hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))
            hidden_states_flat = self.intermediate(hidden_states_flat, training=training)
            hidden_states = ops.reshape(hidden_states_flat, (batch_size, seq_len, -1))

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

    There is no task-attention stage and no FFN stage. ``call`` goes straight
    from the intermediate block to the scorer, so ``use_task_attention`` and
    ``use_ffn`` build layers this head never runs.

    Input shape:
        ``(batch, num_choices, seq_length, input_dim)``, or an already-pooled
        ``(batch, num_choices, input_dim)``, or a dict with
        ``'hidden_states'``.

    Output shape:
        ``{'logits': (batch, num_choices),
        'probabilities': (batch, num_choices)}``.

    :param kwargs: Passed to :class:`BaseNLPHead`. ``task_config`` and
        ``input_dim`` are required there.

    :ivar scorer: ``Dense(1)`` applied to every choice.
    :vartype scorer: keras.layers.Dense
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Build the base head, then the per-choice scorer.

        :param kwargs: Passed to :class:`BaseNLPHead`.
        :return: None.
        :rtype: None
        """
        super().__init__(**kwargs)

        # Per-choice scorer, created in __init__ per the package rule.
        self.scorer = layers.Dense(
            1,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name=f"{self.name}_scorer"
        )

    def build(self, input_shape: Union[Tuple, Dict]) -> None:
        """
        Build the common layers, then the scorer.

        The scorer sees one row per choice, so only its width matters. That
        width is ``hidden_size`` when any of ``ffn``, ``task_attention`` or
        ``intermediate`` is enabled, and ``input_dim`` otherwise.

        :param input_shape: Shape of the input, or a dict of shapes carrying a
            ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: None.
        :rtype: None
        """
        # First build common layers
        super().build(input_shape)

        # Determine the correct input dimension for the scorer.
        if self.use_ffn or self.use_task_attention or self.use_intermediate:
            scorer_input_dim = self.hidden_size
        else:
            scorer_input_dim = self.input_dim

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
        input_dim)``, pooled, and unfolded back. A rank-3 input is taken as
        already pooled and skips all three steps.

        :param inputs: ``(batch, num_choices, seq_length, input_dim)``, an
            already-pooled ``(batch, num_choices, input_dim)``, or a dict with
            such a ``'hidden_states'`` entry.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'logits', 'probabilities'}``, both ``(batch,
            num_choices)``. The probabilities are a softmax across choices.
        :rtype: Dict[str, keras.KerasTensor]
        """
        if isinstance(inputs, dict):
            hidden_states = inputs['hidden_states']
        else:
            hidden_states = inputs

        # Reshape to handle multiple choices
        batch_size, num_choices = ops.shape(hidden_states)[:2]

        # Pool each choice if the sequence axis is still there.
        # Rank 4 is [batch, choices, seq, hidden]; rank 3 is already pooled.
        if len(ops.shape(hidden_states)) == 4:
            # Fold the choice axis into the batch axis so the pooler sees
            # ordinary sequences.
            hidden_states = ops.reshape(
                hidden_states,
                (batch_size * num_choices,) + ops.shape(hidden_states)[2:]
            )
            pooled = self._pool_sequence(hidden_states)
            pooled = ops.reshape(pooled, (batch_size, num_choices, -1))
        else:
            pooled = hidden_states

        # Process each choice
        hidden_states = self.norm(pooled)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            # Flatten choices into rows so the dense block sees one row each.
            hidden_shape = ops.shape(hidden_states)
            hidden_flat = ops.reshape(hidden_states, (-1, hidden_shape[-1]))
            hidden_flat = self.intermediate(hidden_flat, training=training)
            hidden_states = ops.reshape(hidden_flat, hidden_shape)

        # Score each choice
        logits = self.scorer(hidden_states)
        logits = ops.squeeze(logits, axis=-1)

        return {
            'logits': logits,
            'probabilities': ops.softmax(logits, axis=-1)
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

    ``call`` takes an optional ``task_name``. Given one, it runs that head
    alone and returns that head's own output dict. Given ``None``, it runs
    every head and returns a dict of those dicts, keyed by task name.

    With ``use_task_specific_projections`` set, each task first goes through
    its own ``Dense`` projection, so the heads can work at different widths.

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

    :param task_configs: Task name to :class:`NLPTaskConfig`. One head is
        built per entry.
    :type task_configs: Dict[str, NLPTaskConfig]
    :param shared_input_dim: Width of the shared features coming in.
    :type shared_input_dim: int
    :param use_task_specific_projections: Whether each task gets its own
        Dense projection before its head.
    :type use_task_specific_projections: bool
    :param kwargs: Additional arguments for the base Layer class.

    :ivar task_heads: Task name to head layer.
    :vartype task_heads: Dict[str, BaseNLPHead]
    :ivar task_projections: Task name to projection layer. Empty when
        ``use_task_specific_projections`` is false.
    :vartype task_projections: Dict[str, keras.layers.Dense]

    :raises ValueError: From :func:`get_head_class`, when a task type has no
        head. From ``call``, when ``task_name`` is not a known task.
    """

    def __init__(
            self,
            task_configs: Dict[str, NLPTaskConfig],
            shared_input_dim: int,
            use_task_specific_projections: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Build one head, and optionally one projection, per task.

        :param task_configs: Task name to :class:`NLPTaskConfig`.
        :type task_configs: Dict[str, NLPTaskConfig]
        :param shared_input_dim: Width of the shared features coming in.
        :type shared_input_dim: int
        :param use_task_specific_projections: Whether each task gets its own
            Dense projection before its head.
        :type use_task_specific_projections: bool
        :param kwargs: Additional arguments for the base Layer class.
        :return: None.
        :rtype: None
        :raises ValueError: If a task type has no implemented head.
        """
        super().__init__(**kwargs)

        self.task_configs = task_configs
        self.shared_input_dim = shared_input_dim
        self.use_task_specific_projections = use_task_specific_projections

        # Heads and projections, created in __init__ per the package rule.
        self.task_heads = {}
        self.task_projections = {}

        for task_name, task_config in task_configs.items():
            # Create appropriate head
            head_class = get_head_class(task_config.task_type)

            # Set input dimension
            if use_task_specific_projections:
                projection_dim = task_config.hidden_size or shared_input_dim
                self.task_projections[task_name] = layers.Dense(
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
        # Build task projections if needed
        if self.use_task_specific_projections:
            for task_name, projection in self.task_projections.items():
                if isinstance(input_shape, dict):
                    hidden_shape = input_shape.get('hidden_states', (None, None, self.shared_input_dim))
                else:
                    hidden_shape = input_shape

                # Projections work on flattened features
                if len(hidden_shape) == 3:
                    projection_input = (None, self.shared_input_dim)
                else:
                    projection_input = hidden_shape
                projection.build(projection_input)

        # Build task heads
        for task_name, head in self.task_heads.items():
            if self.use_task_specific_projections:
                # Head receives projected features
                task_config = self.task_configs[task_name]
                head_input_dim = task_config.hidden_size or self.shared_input_dim
                if isinstance(input_shape, dict):
                    head_input = {'hidden_states': (None, None, head_input_dim)}
                else:
                    head_input = (None, None, head_input_dim) if len(input_shape) == 3 else (None, head_input_dim)
            else:
                head_input = input_shape

            head.build(head_input)

        super().build(input_shape)

    def compute_output_shape(self, input_shape: Union[Tuple, Dict]) -> Dict[str, Dict[str, Tuple]]:
        """
        Report every head's output shapes, keyed by task name.

        This is the ``task_name=None`` shape. There is no shape method for the
        single-task call, because that returns one head's dict directly.

        :param input_shape: Shape of the shared input, or a dict of shapes
            carrying a ``'hidden_states'`` entry.
        :type input_shape: Union[Tuple, Dict]
        :return: Task name to that head's output shape dict.
        :rtype: Dict[str, Dict[str, Tuple]]
        """
        output_shapes = {}

        for task_name, head in self.task_heads.items():
            # Get shape for each task head
            if self.use_task_specific_projections:
                task_config = self.task_configs[task_name]
                head_input_dim = task_config.hidden_size or self.shared_input_dim
                if isinstance(input_shape, dict):
                    hidden_states_shape = input_shape.get('hidden_states')
                    head_input = {'hidden_states': (hidden_states_shape[0],
                                                   hidden_states_shape[1],
                                                   head_input_dim)}
                else:
                    shape = (input_shape[0], input_shape[1], head_input_dim) if len(input_shape) == 3 else \
                        (input_shape[0], head_input_dim)
                    head_input = shape
            else:
                head_input = input_shape

            output_shapes[task_name] = head.compute_output_shape(head_input)

        return output_shapes

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            task_name: Optional[str] = None,
            training: Optional[bool] = None
    ) -> Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]:
        """
        Run one task head, or all of them.

        :param inputs: Shared features from the foundation model, as a tensor
            or as a dict with ``'hidden_states'``.
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param task_name: Which task to run. ``None`` runs every task.
        :type task_name: Optional[str]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: One head's output dict when ``task_name`` is given. A dict of
            every head's output dict, keyed by task name, when it is ``None``.
        :rtype: Union[Dict[str, Dict[str, keras.KerasTensor]], Dict[str, keras.KerasTensor]]
        :raises ValueError: If ``task_name`` is not a known task.
        """
        # Handle single task
        if task_name is not None:
            if task_name not in self.task_heads:
                raise ValueError(f"Unknown task: {task_name}")

            task_inputs = inputs

            # Apply task-specific projection if needed
            if self.use_task_specific_projections:
                if isinstance(task_inputs, dict):
                    hidden_states = task_inputs['hidden_states']
                    hidden_states = self.task_projections[task_name](hidden_states)
                    task_inputs = {**task_inputs, 'hidden_states': hidden_states}
                else:
                    task_inputs = self.task_projections[task_name](task_inputs)

            return self.task_heads[task_name](task_inputs, training=training)

        # Run all tasks
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            task_inputs = inputs

            # Apply task-specific projection if needed
            if self.use_task_specific_projections:
                if isinstance(task_inputs, dict):
                    hidden_states = task_inputs['hidden_states']
                    hidden_states = self.task_projections[task_name](hidden_states)
                    task_inputs = {**task_inputs, 'hidden_states': hidden_states}
                else:
                    task_inputs = self.task_projections[task_name](task_inputs)

            outputs[task_name] = task_head(task_inputs, training=training)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        Each :class:`NLPTaskConfig` is flattened to a plain dict of six
        fields. :meth:`from_config` rebuilds them.

        :return: Config dict accepted by :meth:`from_config`.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'task_configs': {
                name: {
                    'name': tc.name,
                    'task_type': tc.task_type.value,
                    'num_classes': tc.num_classes,
                    'dropout_rate': tc.dropout_rate,
                    'hidden_size': tc.hidden_size,
                    'vocabulary_size': getattr(tc, 'vocabulary_size', None),
                }
                for name, tc in self.task_configs.items()
            },
            'shared_input_dim': self.shared_input_dim,
            'use_task_specific_projections': self.use_task_specific_projections,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiTaskNLPHead":
        """
        Rebuild a multi-task head from a saved config.

        Each flattened task-config dict is turned back into an
        :class:`NLPTaskConfig` before the layer is constructed.

        :param config: Config dict produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new multi-task head.
        :rtype: MultiTaskNLPHead
        """
        # Reconstruct task configs
        task_configs_dict = config.pop('task_configs')
        task_configs = {}

        for name, tc_dict in task_configs_dict.items():
            task_configs[name] = NLPTaskConfig(
                name=tc_dict['name'],
                task_type=NLPTaskType(tc_dict['task_type']),
                num_classes=tc_dict.get('num_classes'),
                dropout_rate=tc_dict.get('dropout_rate', 0.1),
                hidden_size=tc_dict.get('hidden_size'),
                vocabulary_size=tc_dict.get('vocabulary_size'),
            )

        config['task_configs'] = task_configs
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def get_head_class(task_type: NLPTaskType) -> type:
    """
    Look up the head class for a task type.

    The mapping is an explicit dict. It covers 24 of the 37
    :class:`NLPTaskType` members. **The other 13 raise ``ValueError``.** That
    is a guard, not a gap in the table: this function used to end with
    ``head_mapping.get(task_type, TextClassificationHead)``, so asking for
    ``MACHINE_TRANSLATION`` returned a text classifier that built, trained and
    produced plausible nonsense. Nothing ever failed. A task with no head is a
    caller error and is now reported as one.

    To add support for one of the 13, implement or choose a head and add the
    entry. Do not restore the fallback.

    **Architecture Overview:**

    .. code-block:: text

        task_type
            │
            ▼
        ┌──────────────────────┐        ┌────────────┐
        │ in head_mapping?     │─ no ──►│ ValueError │
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

    :param task_type: The NLP task type to look up.
    :type task_type: NLPTaskType
    :return: The head class for that task type.
    :rtype: type
    :raises ValueError: If no head is implemented for ``task_type``. The
        message lists every supported task type.
    """
    head_mapping = {
        # Classification tasks
        NLPTaskType.TEXT_CLASSIFICATION: TextClassificationHead,
        NLPTaskType.SENTIMENT_ANALYSIS: TextClassificationHead,
        NLPTaskType.EMOTION_DETECTION: TextClassificationHead,
        NLPTaskType.INTENT_CLASSIFICATION: TextClassificationHead,
        NLPTaskType.TOPIC_CLASSIFICATION: TextClassificationHead,
        NLPTaskType.SPAM_DETECTION: TextClassificationHead,

        # Token classification
        NLPTaskType.TOKEN_CLASSIFICATION: TokenClassificationHead,
        NLPTaskType.NAMED_ENTITY_RECOGNITION: TokenClassificationHead,
        NLPTaskType.PART_OF_SPEECH_TAGGING: TokenClassificationHead,
        NLPTaskType.SEQUENCE_LABELING: TokenClassificationHead,

        # QA and span tasks
        NLPTaskType.QUESTION_ANSWERING: QuestionAnsweringHead,
        NLPTaskType.SPAN_EXTRACTION: QuestionAnsweringHead,

        # Similarity tasks
        NLPTaskType.TEXT_SIMILARITY: TextSimilarityHead,
        NLPTaskType.PARAPHRASE_DETECTION: TextSimilarityHead,
        NLPTaskType.DUPLICATE_DETECTION: TextSimilarityHead,

        # Generation tasks
        NLPTaskType.TEXT_GENERATION: TextGenerationHead,
        NLPTaskType.MASKED_LANGUAGE_MODELING: TextGenerationHead,
        NLPTaskType.TEXT_SUMMARIZATION: TextGenerationHead,
        NLPTaskType.TEXT_COMPLETION: TextGenerationHead,

        # Multiple choice
        NLPTaskType.MULTIPLE_CHOICE: MultipleChoiceHead,

        # NLI can use classification head with 3 classes
        NLPTaskType.NATURAL_LANGUAGE_INFERENCE: TextClassificationHead,

        # Regression tasks use classification head with num_classes=1
        NLPTaskType.TEXT_REGRESSION: TextClassificationHead,
        NLPTaskType.READABILITY_SCORING: TextClassificationHead,
        NLPTaskType.QUALITY_SCORING: TextClassificationHead,
    }

    # No silent fallback. This used to be
    #     return head_mapping.get(task_type, TextClassificationHead)
    # so 13 of the 37 NLPTaskType members -- MACHINE_TRANSLATION,
    # DIALOGUE_GENERATION, RELATION_EXTRACTION, DEPENDENCY_PARSING,
    # COREFERENCE_RESOLUTION, SEMANTIC_ROLE_LABELING and friends -- returned a
    # text classification head. It builds, it trains, and it produces plausible
    # nonsense. A translation task quietly became a classifier, and nothing
    # failed. The sibling `heads/vision/factory.py` already raises on an
    # unsupported task, and this now matches it. To add one of the missing
    # types, map it here on purpose. Do not restore the fallback.
    if task_type not in head_mapping:
        supported = sorted(t.name for t in head_mapping)
        raise ValueError(
            f"No NLP head is implemented for task type '{task_type.name}'. "
            f"Supported task types: {supported}. "
            f"(This previously returned a TextClassificationHead silently, which produced "
            f"a working but wrong head for tasks like MACHINE_TRANSLATION.)"
        )

    return head_mapping[task_type]


def create_nlp_head(
        task_config: Union[NLPTaskConfig, Dict[str, Any]],
        input_dim: int,
        **kwargs: Any
) -> BaseNLPHead:
    """
    Build one NLP head from a task config.

    The task type in the config picks the class through
    :func:`get_head_class`. A plain dict is converted to an
    :class:`NLPTaskConfig` first. Extra keyword arguments go to the head's
    constructor unchanged, so anything :class:`BaseNLPHead` accepts can be set
    here.

    **Architecture Overview:**

    .. code-block:: text

        task_config (NLPTaskConfig or dict)
                │
                ▼
        ┌──────────────────────────┐
        │ dict -> NLPTaskConfig    │
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
    :raises ValueError: If the task type has no implemented head.
    """
    # Convert dict to NLPTaskConfig if needed
    if isinstance(task_config, dict):
        task_config = NLPTaskConfig(**task_config)

    # Get appropriate head class
    head_class = get_head_class(task_config.task_type)

    # Create head with configuration
    return head_class(
        task_config=task_config,
        input_dim=input_dim,
        **kwargs
    )


def create_multi_task_nlp_head(
        task_configs: Union[List[NLPTaskConfig], Dict[str, NLPTaskConfig]],
        input_dim: int,
        **kwargs: Any
) -> MultiTaskNLPHead:
    """
    Build a :class:`MultiTaskNLPHead` from several task configs.

    A list is turned into a dict keyed by each config's ``name``. Pass a dict
    directly to choose the keys yourself.

    :param task_configs: A list of :class:`NLPTaskConfig`, or a dict of task
        name to config.
    :type task_configs: Union[List[NLPTaskConfig], Dict[str, NLPTaskConfig]]
    :param input_dim: Width of the shared features. Becomes the head's
        ``shared_input_dim``.
    :type input_dim: int
    :param kwargs: Extra keyword arguments for :class:`MultiTaskNLPHead`.
    :return: The configured multi-task head.
    :rtype: MultiTaskNLPHead
    :raises ValueError: If any task type has no implemented head.
    """
    # Convert list to dict if needed
    if isinstance(task_configs, list):
        task_configs = {config.name: config for config in task_configs}

    return MultiTaskNLPHead(
        task_configs=task_configs,
        shared_input_dim=input_dim,
        **kwargs
    )


# ---------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------

class NLPHeadConfiguration:
    """
    Suggested keyword arguments per task type.

    This is a lookup table, not a layer. Nothing in this module calls it. Pass
    the result to :func:`create_nlp_head` when you want sensible defaults
    instead of writing them out.

    **Task Defaults:**

    .. code-block:: text

        Every task gets these five:
          dropout_rate 0.1
          normalization_type 'layer_norm'
          activation_type 'gelu'
          use_intermediate True
          initializer_range 0.02

        task_type              extra keys
        --------------------   -----------------------------
        TEXT_CLASSIFICATION    pooling 'cls', no task attn
        TOKEN_CLASSIFICATION   no pooling, no crf, no attn
        QUESTION_ANSWERING     no pooling, multi_head attn
        TEXT_SIMILARITY        pooling 'mean', cosine score
        TEXT_GENERATION        no pooling, swiglu ffn,
                               vocabulary_size 32000

    A task type not in the table gets the five common keys only.
    """

    @staticmethod
    def get_default_config(task_type: NLPTaskType) -> Dict[str, Any]:
        """
        Return the suggested keyword arguments for a task type.

        Five keys are always present. Five task types add a few more. Any
        other task type gets the five common keys unchanged.

        :param task_type: The NLP task type to look up.
        :type task_type: NLPTaskType
        :return: A fresh dict, safe for the caller to mutate.
        :rtype: Dict[str, Any]
        """
        base_config = {
            'dropout_rate': 0.1,
            'normalization_type': 'layer_norm',
            'activation_type': 'gelu',
            'use_intermediate': True,
            'initializer_range': 0.02,
        }

        task_specific = {
            NLPTaskType.TEXT_CLASSIFICATION: {
                'use_pooling': True,
                'pooling_type': 'cls',
                'use_task_attention': False,
            },
            NLPTaskType.TOKEN_CLASSIFICATION: {
                'use_pooling': False,
                'use_crf': False,
                'use_task_attention': False,
            },
            NLPTaskType.QUESTION_ANSWERING: {
                'use_pooling': False,
                'use_task_attention': True,
                'attention_type': 'multi_head',
            },
            NLPTaskType.TEXT_SIMILARITY: {
                'use_pooling': True,
                'pooling_type': 'mean',
                'output_embeddings': True,
                'similarity_function': 'cosine',
            },
            NLPTaskType.TEXT_GENERATION: {
                'use_pooling': False,
                'vocabulary_size': 32000,
                'use_ffn': True,
                'ffn_type': 'swiglu',
            },
        }

        config = base_config.copy()
        if task_type in task_specific:
            config.update(task_specific[task_type])

        return config