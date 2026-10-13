"""A wrapper around a length-guarded backbone survives a refused first call.

The sequence-length guard (``dl_techniques.utils.sequence_length_guard``)
drops the cached step functions of the model it guards, so a refused first
``predict`` leaves THAT model usable. A pre-training wrapper is a different
model with step functions of its own, and the refusal fires inside the
wrapper's first build:

``CausalLanguageModel``
    Its ``build`` caught every backbone build error and only logged it, so the
    wrapper was marked built around an unbuilt backbone (weight tying resolved
    against no variables, so an untied head was made) and its own
    ``predict_function`` was left broken by the failed first trace. Measured
    before the fix: ``predict(20)`` refused, then ``predict(8)`` twice
    ``ValueError: Structures don't have the same nested structure``; eager
    ``call(20)`` then ``call(8)`` ``TypeError``.
``MaskedLanguageModel``
    Its ``build`` re-raises the backbone error but never dropped its own step
    functions.

Each arm: the over-long call is refused with the backbone's message, then
two legal calls succeed with finite logits, and the CLM's head is the tied
head a never-refused wrapper gets.
"""

from typing import Any, Callable, Dict, NamedTuple

import keras
import numpy as np
import pytest

from dl_techniques.models.common.masked_language_model import (
    CausalLanguageModel,
    MaskedLanguageModel,
)
from ..tiny_language_models import TINY, VOCAB

# ---------------------------------------------------------------------

BATCH = 2
LIMIT = 16
OVER_LONG = LIMIT + 4
LEGAL = 8

# ---------------------------------------------------------------------


class Wrapped(NamedTuple):
    """One wrapper over one guarded backbone.

    :param name: Test id.
    :param factory: Builds a fresh, unbuilt wrapper whose backbone refuses a
        sequence longer than ``LIMIT``.
    :param limit_name: The backbone's limit attribute, named by its message.
    """

    name: str
    factory: Callable[[], keras.Model]
    limit_name: str


def _clm_over_qwen3() -> keras.Model:
    # Qwen3 returns vocabulary logits itself, so the wrapper adds no head.
    tiny = TINY["qwen3"]
    backbone = tiny.cls()(**{**tiny.kwargs, "max_seq_len": LIMIT})
    return CausalLanguageModel(
        backbone=backbone, vocab_size=VOCAB, skip_head=True,
        verify_causality=False,
    )


def _clm_over_bert() -> keras.Model:
    # A backbone returning `last_hidden_state`: the wrapper ties its own head
    # to the backbone's token table, which exists only once the backbone is
    # built. `verify_causality=False`: BERT is bidirectional; this is about
    # the head, as in `test_the_embedding_locator_finds_a_keras_embedding`.
    tiny = TINY["bert"]
    backbone = tiny.cls()(**{**tiny.kwargs, "max_position_embeddings": LIMIT})
    return CausalLanguageModel(
        backbone=backbone, vocab_size=VOCAB, verify_causality=False
    )


def _mlm_over_bert() -> keras.Model:
    tiny = TINY["bert"]
    encoder = tiny.cls()(**{**tiny.kwargs, "max_position_embeddings": LIMIT})
    return MaskedLanguageModel(
        encoder=encoder, vocab_size=VOCAB, mask_token_id=VOCAB - 1
    )


WRAPPED = [
    Wrapped("clm_over_qwen3", _clm_over_qwen3, "max_seq_len"),
    Wrapped("clm_over_bert", _clm_over_bert, "max_position_embeddings"),
    Wrapped("mlm_over_bert", _mlm_over_bert, "max_position_embeddings"),
]
WRAPPED_IDS = [w.name for w in WRAPPED]


def _feed(seq_len: int) -> Dict[str, np.ndarray]:
    ids = np.random.RandomState(3).randint(0, VOCAB - 4, (BATCH, seq_len))
    ids = ids.astype("int32")
    return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def _assert_legal_logits(out: Any, seq_len: int) -> None:
    out = keras.ops.convert_to_numpy(out)
    assert out.shape == (BATCH, seq_len, VOCAB)
    assert np.isfinite(out).all()


def _assert_head_as_never_refused(model: keras.Model) -> None:
    """A CLM with a head ties it to the backbone's table, as a fresh one does."""
    if isinstance(model, CausalLanguageModel) and not model.skip_head:
        assert model.use_weight_tying, "the head was built untied"
        assert model.output_layer is None

# ---------------------------------------------------------------------


@pytest.mark.parametrize("wrapped", WRAPPED, ids=WRAPPED_IDS)
def test_a_refused_first_predict_leaves_the_wrapper_usable(wrapped):
    model = wrapped.factory()
    with pytest.raises(ValueError, match=wrapped.limit_name):
        model.predict(_feed(OVER_LONG), verbose=0)
    for _ in range(2):
        _assert_legal_logits(model.predict(_feed(LEGAL), verbose=0), LEGAL)
    _assert_head_as_never_refused(model)


@pytest.mark.parametrize("wrapped", WRAPPED, ids=WRAPPED_IDS)
def test_a_refused_first_eager_call_leaves_the_wrapper_usable(wrapped):
    model = wrapped.factory()
    with pytest.raises(ValueError, match=wrapped.limit_name):
        model(_feed(OVER_LONG))
    for _ in range(2):
        _assert_legal_logits(model(_feed(LEGAL)), LEGAL)
    _assert_head_as_never_refused(model)


def test_the_clm_still_tolerates_any_other_backbone_build_error(caplog):
    """Only a length refusal escapes ``CausalLanguageModel.build``.

    Any other backbone build error keeps the logged-warning path (the
    wrapper's D-049 contract): here a backbone whose ``build`` raises a plain
    ``ValueError`` is still wrapped and built.
    """

    class _FailingBuild(keras.layers.Layer):
        hidden_size = 8

        def build(self, input_shape):
            raise ValueError("not a length refusal")

        def call(self, inputs):
            return {"last_hidden_state": inputs}

    model = CausalLanguageModel(
        backbone=_FailingBuild(), vocab_size=VOCAB, verify_causality=False,
        tie_weights=False,
    )
    with caplog.at_level("WARNING"):
        model.build((None, None))
    assert model.built
    assert "not a length refusal" in caplog.text
