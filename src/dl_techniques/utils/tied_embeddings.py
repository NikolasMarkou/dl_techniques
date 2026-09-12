"""Shared tied-embedding LM-head projection.

Seven `models/` sites independently implement the same weight-tied LM-head
projection (`logits = hidden_states @ embedding_weights.T`, optionally
`+ bias`), used only when the model's `tie_word_embeddings` / `tie_weights` /
`use_weight_tying` / `use_shared_embedding` flag is `True`: `gpt2/gpt2.py`,
`hnet/model.py`, `wave_field/model.py`, `zamba2/model.py` (unconditional there
-- it has no untied path), `masked_language_model/clm.py` (the one site with a
learned bias), `vision/cliffordnet/lm.py`, and `vision_language/nano_vlm/model.py`
(two call sites, `call()` and `generate()`). This module extracts the one
expression that is genuinely identical across all of them -- the
matmul/transpose/floor-to-float32/bias -- into a single helper.

The surrounding tying-resolution control flow (eager `__init__`/`build`-time
resolution at most sites vs. `clm.py`'s deferred `build()`/`call()`-time
resolution with a learned bias fallback) is deliberately NOT unified here: it
differs enough per site that forcing one shape onto it would be a leaky
abstraction. See `plan-2026-09-12T123331-28fd855f`'s `decisions.md` D-002.
"""

from typing import Optional

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
# produce float32 real-call output is promoting the matmul RESULT (below),
# after the matmul. See decisions.md D-004.
#
# DECISION plan-2026-09-12T123331-28fd855f/D-005
# WHAT NOT TO DO: do not cast the matmul result to float32
# UNCONDITIONALLY. D-004's unconditional cast passed this plan's own
# mixed_float16 tests but turned 2 of this repo's repo-wide
# `test_precision_arm_family.py` guards RED (`gpt2`, `wave_field`) --
# those guards assert every float output of an UNPINNED charged package
# stays at the compute dtype under `mixed_float16` unless the package is
# explicitly registered with `expected_compute_dtype=` (see
# `precision_arm_subjects.py`'s `ideogram4` precedent). An unconditional
# cast also silently DOWNCASTS a float64 caller to float32, a latent
# collision with `assert_float64_arm`. D-005 measured that a FLOOR
# semantic -- promote only when the matmul's natural result dtype is
# NARROWER than float32 (float16/bfloat16), leave float32-or-wider alone
# -- closes the float64 collision while still promoting mixed_float16
# (whose natural matmul result is float16, narrower than float32) exactly
# as before. The residual mixed_float16 deviation for gpt2/wave_field is
# handled by a registered precision-arm pin, not by reverting this floor.
# See decisions.md D-005.
def tied_embedding_logits(
        hidden_states: keras.KerasTensor,
        embedding_weights: keras.KerasTensor,
        *,
        bias: Optional[keras.KerasTensor] = None,
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

    The matmul's RESULT is promoted to ``float32`` only when its natural
    dtype is NARROWER than ``float32`` (``float16`` / ``bfloat16``, e.g. the
    matmul result under a ``mixed_float16`` dtype policy); a result that is
    already ``float32`` or wider (e.g. ``float64``) is left untouched. This
    is a deliberate, registered design choice for loss-facing numerical
    stability -- keeping softmax/cross-entropy inputs at or above
    ``float32`` precision even when the rest of the model computes in a
    lower-precision policy -- and it is NOT automatic Keras dtype-policy
    behavior: a plain final ``Dense`` layer under ``mixed_float16`` returns
    ``float16``, not ``float32`` (measured; see decisions.md D-005). Because
    this floor deviates from a charged package's compute dtype under
    ``mixed_float16``, every caller reached by a package registered in
    ``tests/test_models/precision_arm_subjects.py``'s ``CHARGED_PACKAGES``
    must also register an ``expected_compute_dtype="float32"`` pin there
    (precedent: ``ideogram4``) -- this module cannot see or update that
    registry itself, so keeping the two in sync is the caller's
    responsibility whenever a new tied site is wired to this helper.

    Any ``bias`` is cast to the (possibly-promoted) result dtype and added
    after promotion, so it never re-widens or re-narrows the output.

    :param hidden_states: Backbone output, shape ``(batch, seq_len, hidden_size)``.
    :type hidden_states: keras.KerasTensor
    :param embedding_weights: The token embedding table being reused as the
        LM head, shape ``(vocab_size, hidden_size)``.
    :type embedding_weights: keras.KerasTensor
    :param bias: Optional additive bias, broadcastable against the output
        logits shape ``(batch, seq_len, vocab_size)``. Not added when ``None``
        (the default).
    :type bias: keras.KerasTensor, optional
    :return: Logits of shape ``(batch, seq_len, vocab_size)``. ``float32`` or
        wider under any policy whose natural matmul result would otherwise be
        narrower than ``float32``; unchanged (e.g. ``float64``) otherwise.
    :rtype: keras.KerasTensor
    """
    logits = keras.ops.matmul(hidden_states, keras.ops.transpose(embedding_weights))
    # Dtype-name lookup uses `getattr(d, "name", None) or str(d)`, this repo's
    # own idiom for reading a dtype's name -- NOT `keras.backend.standardize_dtype`
    # (a banned Keras-2 residue) and not a bare `str(d)` (wrong for a tf.DType).
    # See `plan-2026-09-03T033750-9bdf25f4`'s decisions.md D-007 (established
    # at `layers/attention/common.py`), reused here rather than re-derived.
    result_dtype_name = getattr(logits.dtype, "name", None) or str(logits.dtype)
    if result_dtype_name in ("float16", "bfloat16"):
        logits = keras.ops.cast(logits, "float32")
    if bias is not None:
        logits = logits + keras.ops.cast(bias, logits.dtype)
    return logits
