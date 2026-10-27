"""Shared constants and helpers for the NLP heads."""

import keras
from typing import Dict, Optional, Tuple, Any

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

#: The task-attention and FFN knobs each flag gates, with the value a knob
#: takes when its stage is on and the caller gave ``None`` (D-210; the vision
#: heads' table is ``vision.factory.STAGE_KNOB_DEFAULTS``).
NLP_STAGE_KNOB_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'use_task_attention': {'attention_type': 'multi_head'},
    'use_ffn': {'ffn_type': 'mlp', 'ffn_expansion_factor': 4},
}
