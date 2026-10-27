"""``TextGenerationHead``: vocabulary logits per position."""

import keras
from typing import Dict, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _apply_per_row
from .base import BaseNLPHead


@register_dl_technique("dl_techniques.layers.heads.nlp.text_generation")
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
