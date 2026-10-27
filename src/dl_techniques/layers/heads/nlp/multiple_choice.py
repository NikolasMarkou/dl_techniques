"""``MultipleChoiceHead``: one score per candidate answer."""

import keras
from typing import Dict, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import static_rank
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _apply_per_row, _split_inputs
from .base import BaseNLPHead


@register_dl_technique("dl_techniques.layers.heads.nlp.multiple_choice")
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
