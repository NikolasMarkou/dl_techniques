"""Shared tied-embedding LM-head projection.

Five `models/language/` packages independently implement the same weight-tied
LM-head projection (`logits = hidden_states @ embedding_weights.T`, optionally
`+ bias`), used only when the model's `tie_word_embeddings` / `tie_weights` /
`use_weight_tying` flag is `True`: `gpt2/gpt2.py`, `hnet/model.py`,
`wave_field/model.py`, `zamba2/model.py` (unconditional there -- it has no
untied path) and `masked_language_model/clm.py` (the one site with a learned
bias). This module extracts the one expression that is genuinely identical
across all five -- the matmul/transpose/cast -- into a single helper.

The surrounding tying-resolution control flow (eager `__init__`/`build`-time
resolution at four sites vs. `clm.py`'s deferred `build()`/`call()`-time
resolution with a learned bias fallback) is deliberately NOT unified here: it
differs enough per site that forcing one shape onto it would be a leaky
abstraction. See `plan-2026-09-12T123331-28fd855f`'s `decisions.md` D-002.
"""

import keras


# DECISION plan-2026-09-12T123331-28fd855f/D-002
# WHAT NOT TO DO: do not fold the "build lm_head only if untied" branch into
# this helper, and do not place it in `layers/heads/nlp/` or as a Keras
# `Layer`. Only the matmul/transpose/cast/bias expression is identical
# across all 5 call sites -- the surrounding tying-resolution control flow
# (eager __init__/build-time at 4 sites vs. clm.py's deferred build()/call()
# with a learned bias) differs enough that unifying it would be a leaky
# abstraction. See decisions.md D-002.
def tied_embedding_logits(
        hidden_states: keras.KerasTensor,
        embedding_weights: keras.KerasTensor,
        *,
        bias: keras.KerasTensor = None,
) -> keras.KerasTensor:
    """Project hidden states to vocabulary logits through a tied embedding table.

    Casts ``embedding_weights`` to ``hidden_states.dtype`` before the matmul.
    This cast is a no-op at ``float32`` (the default dtype policy, same dtype
    in and out) and load-bearing under ``mixed_float16``: the embedding table
    is created at ``float32`` (Keras variables default to the layer's
    variable dtype regardless of the compute dtype policy), and a
    ``float32`` weight matrix cannot matmul against a ``float16`` hidden
    state without an explicit cast.

    :param hidden_states: Backbone output, shape ``(batch, seq_len, hidden_size)``.
    :type hidden_states: keras.KerasTensor
    :param embedding_weights: The token embedding table being reused as the
        LM head, shape ``(vocab_size, hidden_size)``.
    :type embedding_weights: keras.KerasTensor
    :param bias: Optional additive bias, broadcastable against the output
        logits shape ``(batch, seq_len, vocab_size)``. Not added when ``None``
        (the default).
    :type bias: keras.KerasTensor, optional
    :return: Logits of shape ``(batch, seq_len, vocab_size)``.
    :rtype: keras.KerasTensor
    """
    embedding_weights = keras.ops.cast(embedding_weights, hidden_states.dtype)
    logits = keras.ops.matmul(hidden_states, keras.ops.transpose(embedding_weights))
    if bias is not None:
        logits = logits + bias
    return logits
