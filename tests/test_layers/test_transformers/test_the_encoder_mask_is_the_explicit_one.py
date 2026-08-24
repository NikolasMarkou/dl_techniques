"""R-038 root cause **RD-6**: the auto-mask ``TextEncoder`` never consumed.

Plan ``plan-2026-08-22T035419-a11304c8``, ruling **D-055**.

The inventory recorded ``Layer 'positional_embeddings' (of type
PositionalEmbedding) was passed an input with a mask attached to it. However,
this layer does not support masking and will therefore destroy the mask
information. Downstream layers will not see the mask.`` and asked whether that
lets attention see padding.

**It does -- and ``supports_masking = True`` on ``PositionalEmbedding`` does not
fix it.** Measured on a seeded ``TextEncoder`` fed ``[[5, 7, 0, 0]]`` with no
explicit ``attention_mask``:

=========================================  ==============================
arm                                        ``max|f([5,7,0,0])[:, :2] - f([5,7])|``
=========================================  ==============================
``PositionalEmbedding.supports_masking``   ``1.290977e-02``
  = False (as shipped)
``PositionalEmbedding.supports_masking``   ``1.290977e-02`` -- IDENTICAL
  = True
explicit ``attention_mask=[[1,1,0,0]]``    ``2.384186e-07`` (float32 noise)
=========================================  ==============================

Declaring ``supports_masking`` only relocated the warning to
``transformer_layer_0`` / ``MultiHeadAttention`` / ``MultiHeadCrossAttention``,
none of which support masking either. The auto-mask was therefore consumed by
NOTHING, was numerically inert (forward output bit-identical, max abs diff
``0.0``, at ``mask_zero`` True vs False), and its only observable effect was the
warning. D-055 sets ``mask_zero=False`` by default -- the same finding and the
same repair as ``models/distilbert/model.py`` (D-018).

These tests pin BOTH halves: the working mask, and the documented limitation.
A test that only pinned the working mask would let someone "restore"
``mask_zero=True`` and reintroduce the warning without failing anything.
"""

import warnings
from typing import List

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.transformers.text_encoder import TextEncoder
from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding

_IDS_PADDED = np.array([[5, 7, 0, 0]], dtype="int32")
_IDS_SHORT = np.array([[5, 7]], dtype="int32")


def _encoder(**overrides) -> TextEncoder:
    keras.utils.set_random_seed(0)
    kwargs = dict(
        vocab_size=32, embed_dim=16, depth=2, num_heads=2, max_seq_len=8,
        positional_type="learned", output_mode="mean", dropout_rate=0.0,
        attention_dropout_rate=0.0, use_cls_token=False,
    )
    kwargs.update(overrides)
    return TextEncoder(**kwargs)


def _user_warnings(fn) -> List[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def test_positional_embedding_declares_that_it_is_mask_safe():
    """A position-wise add cannot corrupt a per-timestep mask."""
    layer = PositionalEmbedding(max_seq_len=8, dim=16)
    assert layer.supports_masking is True


def test_no_layer_reports_destroying_a_mask_in_a_default_encoder():
    """The whole W-11 family, at its source: there is no mask to destroy."""
    encoder = _encoder()
    messages = _user_warnings(
        lambda: encoder.get_sequence_features(_IDS_PADDED, training=False)
    )
    destroyed = [m for m in messages if "destroy the mask information" in m]
    assert destroyed == [], (
        "a layer reported destroying an attached mask. The encoder must not "
        "create an auto-mask nothing in its chain consumes:\n  "
        + "\n  ".join(destroyed)
    )


def test_the_word_embedding_does_not_advertise_masking_it_cannot_deliver():
    encoder = _encoder()
    assert encoder.word_embeddings.mask_zero is False, (
        "mask_zero=True attaches a Keras mask that PositionalEmbedding, "
        "TransformerLayer, MultiHeadAttention and MultiHeadCrossAttention all "
        "drop. See D-055."
    )


def test_the_explicit_attention_mask_makes_padding_invisible():
    """The masking path that DOES work -- the anti-vacuity arm."""
    encoder = _encoder()
    masked = ops.convert_to_numpy(encoder.get_sequence_features(
        _IDS_PADDED,
        attention_mask=np.array([[1, 1, 0, 0]], dtype="int32"),
        training=False,
    ))
    short = ops.convert_to_numpy(
        encoder.get_sequence_features(_IDS_SHORT, training=False)
    )
    gap = float(np.max(np.abs(masked[:, :2, :] - short)))
    assert gap < 1e-5, (
        f"with an explicit attention_mask the first two positions still differ "
        f"from the unpadded sequence by {gap:.6e}; the mask is not reaching "
        f"attention"
    )


def test_without_the_explicit_mask_padding_IS_visible_and_that_is_documented():
    """The limitation, pinned so it cannot be mistaken for masked behaviour.

    If this ever starts passing at ``< 1e-5``, someone has wired real Keras mask
    propagation through the stack -- good news, but the docstrings and D-055
    must be updated in the same change rather than left claiming otherwise.
    """
    encoder = _encoder()
    unmasked = ops.convert_to_numpy(
        encoder.get_sequence_features(_IDS_PADDED, training=False)
    )
    short = ops.convert_to_numpy(
        encoder.get_sequence_features(_IDS_SHORT, training=False)
    )
    gap = float(np.max(np.abs(unmasked[:, :2, :] - short)))
    assert gap > 1e-4, (
        f"the padded and unpadded runs now agree to {gap:.6e} without an "
        f"explicit attention_mask. Keras mask propagation appears to work now; "
        f"update D-055 and the TextEncoder docstring, then change this test."
    )


def test_mask_zero_is_numerically_inert_here():
    """The flag's only effect was the warning -- measured, not asserted."""
    outputs = {}
    for flag in (True, False):
        encoder = _encoder(embedding_args={"mask_zero": flag})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outputs[flag] = ops.convert_to_numpy(
                encoder.get_sequence_features(_IDS_PADDED, training=False)
            )
    delta = float(np.max(np.abs(outputs[True] - outputs[False])))
    assert delta == 0.0, (
        f"mask_zero is no longer numerically inert (max abs diff {delta:.6e}). "
        f"D-055's repair rests on that inertness; re-derive it."
    )
