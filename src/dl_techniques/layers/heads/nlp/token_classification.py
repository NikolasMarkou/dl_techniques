"""``TokenClassificationHead``: one label per token."""

import keras
from typing import Dict, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _apply_per_row, _split_inputs
from .base import BaseNLPHead


@register_dl_technique("dl_techniques.layers.heads.nlp.token_classification")
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

        :param inputs: A sequence tensor, or a dict with ``'hidden_states'``
            and an optional ``'attention_mask'``, which reaches the task
            attention only (every other stage is per token).
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether the call is a training step.
        :type training: Optional[bool]
        :return: ``{'logits'}``, plus ``'predictions'`` when ``use_crf`` is
            false.
        :raises ValueError: If a mask is given to a task attention type that
            takes none.
        :rtype: Dict[str, keras.KerasTensor]
        """
        sequence_output, attention_mask = _split_inputs(inputs)

        # Apply processing to each token
        hidden_states = self.norm(sequence_output)
        hidden_states = self.dropout(hidden_states, training=training)

        if self.use_intermediate:
            # One row per token, then the original layout back.
            hidden_states = _apply_per_row(self.intermediate, hidden_states, training)

        if self.use_task_attention:
            hidden_states = self._apply_task_attention(
                hidden_states, attention_mask, training)

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
