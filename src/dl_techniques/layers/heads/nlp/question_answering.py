"""``QuestionAnsweringHead``: start and end logits over a sequence."""

import keras
from typing import Dict, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from ..masking import fill_masked
from .utils import _apply_per_row, _split_inputs
from .base import BaseNLPHead


@register_dl_technique("dl_techniques.layers.heads.nlp.question_answering")
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

        When an ``'attention_mask'`` is given, the task attention (if on)
        attends over real tokens only, and masked positions are replaced
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
            hidden_states = self._apply_task_attention(
                hidden_states, attention_mask, training)

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
            start_logits = fill_masked(
                start_logits, attention_mask, self.compute_dtype)
            end_logits = fill_masked(
                end_logits, attention_mask, self.compute_dtype)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }
