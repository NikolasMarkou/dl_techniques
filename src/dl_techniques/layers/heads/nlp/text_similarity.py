"""``TextSimilarityHead``: one embedding, or a score for a pair."""

import keras
from typing import Dict, List, Optional, Union, Tuple, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.dtype_policy import stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _is_pair, _split_inputs
from .base import BaseNLPHead


@register_dl_technique("dl_techniques.layers.heads.nlp.text_similarity")
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
