"""Shared tied-embedding LM-head projection.

Five `models/language/` packages independently implement the same weight-tied
LM-head projection (`logits = hidden_states @ embedding_weights.T`, optionally
`+ bias`), used only when the model's `tie_word_embeddings` / `tie_weights` /
`use_weight_tying` flag is `True`: `gpt2/gpt2.py`, `hnet/model.py`,
`wave_field/model.py`, `zamba2/model.py` (unconditional there -- it has no
untied path) and `masked_language_model/clm.py` (the one site with a learned
bias). This module extracts the one expression that is genuinely identical
across all five -- the matmul/transpose/float32-cast -- into a single helper.

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
#
# DECISION plan-2026-09-12T123331-28fd855f/D-004
# WHAT NOT TO DO: do not cast either OPERAND (hidden_states or
# embedding_weights) to try to control the output dtype under mixed
# precision. D-003 assumed a float32 embedding table needs an explicit
# operand cast to matmul against a float16 hidden state; D-004 measured
# (real eager forward pass through a model's own `AutocastScope`, not an
# isolated `ops.matmul` probe) that this premise is false in this repo's
# Keras 3.8.0/TF 2.18 stack -- `AutocastScope` already autocasts the
# embedding variable to the compute dtype at the point it is read inside
# `call()`, so both operands are already float16 under `mixed_float16`
# regardless of any operand-level cast. The only mechanism proven to
# produce float32 real-call output is casting the matmul RESULT (below),
# after the matmul, unconditionally. See decisions.md D-004.
def tied_embedding_logits(
        hidden_states: keras.KerasTensor,
        embedding_weights: keras.KerasTensor,
        *,
        bias: keras.KerasTensor = None,
) -> keras.KerasTensor:
    """Project hidden states to vocabulary logits through a tied embedding table.

    Operand dtypes are used as-is -- neither ``hidden_states`` nor
    ``embedding_weights`` is cast before the matmul. Under this repo's Keras
    3.8.0/TF 2.18 stack, a real forward pass runs inside the calling layer's
    ``AutocastScope``, which already autocasts the embedding variable to the
    active compute dtype at the point it is read; an explicit operand cast
    measures as a no-op there (see ``decisions.md`` D-004 for the real-model
    measurement method -- an actual eager forward pass, not an isolated
    ``keras.ops.matmul`` probe, which is what led an earlier revision of this
    docstring to the wrong conclusion).

    The matmul's RESULT is unconditionally cast to ``float32`` -- and any
    ``bias`` is added after that cast -- so this function ALWAYS returns
    ``float32`` logits regardless of the input dtype. This mirrors the
    standard Keras mixed-precision convention of keeping loss-facing outputs
    (softmax / cross-entropy inputs) in ``float32`` even when the rest of the
    model runs its compute in a lower-precision policy, the same convention a
    final ``Dense`` layer follows under a ``mixed_float16`` dtype policy.

    :param hidden_states: Backbone output, shape ``(batch, seq_len, hidden_size)``.
    :type hidden_states: keras.KerasTensor
    :param embedding_weights: The token embedding table being reused as the
        LM head, shape ``(vocab_size, hidden_size)``.
    :type embedding_weights: keras.KerasTensor
    :param bias: Optional additive bias, broadcastable against the output
        logits shape ``(batch, seq_len, vocab_size)``. Not added when ``None``
        (the default). Added after the float32 cast, so it is safe under any
        input dtype.
    :type bias: keras.KerasTensor, optional
    :return: Logits of shape ``(batch, seq_len, vocab_size)``, always ``float32``.
    :rtype: keras.KerasTensor
    """
    logits = keras.ops.matmul(hidden_states, keras.ops.transpose(embedding_weights))
    logits = keras.ops.cast(logits, "float32")
    if bias is not None:
        logits = logits + keras.ops.cast(bias, "float32")
    return logits
