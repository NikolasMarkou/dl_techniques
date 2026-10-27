"""``TextClassificationHead``: one label for a whole sequence."""

import keras
from typing import Dict, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _split_inputs
from .base import BaseNLPHead


@register_dl_technique("dl_techniques.layers.heads.nlp.text_classification")
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
