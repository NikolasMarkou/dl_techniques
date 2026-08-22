"""`CausalLanguageModel` must locate a Keras `Embedding`'s weight matrix.

Rationale
---------
`_locate_embedding_weights`'s "HF style" branch read
``self.backbone.embeddings.word_embeddings.weight``. ``weight`` is the PyTorch
spelling; a Keras `Embedding` calls it ``embeddings``, and reading EITHER
attribute on an unbuilt layer RAISES rather than returning None. MEASURED at
HEAD:

    CausalLanguageModel(backbone=BERT(...))
      AttributeError: 'Embedding' object has no attribute 'weight'.
                      Did you mean: 'weights'?

i.e. the class could not be constructed on this repository's OWN BERT -- the
most obvious backbone there is -- and the failure was at construction, not at
some exotic call path.

The repair matches on the variable's SHAPE first and only then falls back to
attribute names, so it works for a built layer of any provenance and degrades
to `None` (weight tying simply disabled) instead of crashing.

See decisions.md D-035 (plan-2026-08-19T163559-499b6f0e).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.bert.bert import BERT
from dl_techniques.models.masked_language_model.clm import CausalLanguageModel

VOCAB = 64
HIDDEN = 32
SEQ = 12


@pytest.fixture(scope="module")
def model() -> CausalLanguageModel:
    backbone = BERT(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=1, num_heads=2,
        intermediate_size=64, max_position_embeddings=64,
        hidden_dropout_rate=0.0, attention_probs_dropout_rate=0.0,
    )
    # `verify_causality=False`: BERT is bidirectional by construction, so the
    # causality probe would (correctly) reject it. This test is about the
    # EMBEDDING LOCATOR, not about causality.
    model = CausalLanguageModel(
        backbone=backbone, vocab_size=VOCAB, verify_causality=False)
    # Build before the locator is inspected: `.variables` is empty on an
    # unbuilt layer, so a shape-matching locator can only work post-build.
    ids = np.zeros((2, SEQ), dtype="int32")
    model({"input_ids": ids, "attention_mask": np.ones((2, SEQ), "int32")},
          training=False)
    return model


def test_the_model_constructs_on_a_bert_backbone(model):
    """RED at HEAD: AttributeError: 'Embedding' object has no attribute 'weight'."""
    assert model is not None


def test_the_embedding_matrix_is_located_with_the_right_shape(model):
    """The locator itself, asked once its subject actually exists."""
    located = model._locate_embedding_weights()
    assert located is not None, (
        "`_locate_embedding_weights` returned None for a plain Keras "
        "`Embedding` on a BUILT backbone -- the shape match is not working"
    )
    assert tuple(located.shape) == (VOCAB, HIDDEN), (
        f"located a variable of shape {tuple(located.shape)}, expected "
        f"({VOCAB}, {HIDDEN}) -- the shape match is the whole point of the "
        f"locator, so a wrong shape is worse than None"
    )


def test_weight_tying_is_actually_active(model):
    """UN-PINNED at step 17.1 of plan-2026-08-19T163559-499b6f0e.

    This carried `xfail(strict=True)` with the reason "`BERT` implements no
    `build()`, so it is marked built with ZERO variables and the locator runs
    before anything exists to find". `BERT.build` now exists (decisions.md
    D-049) and the pin's own stated closing condition is met.

    MEASURED across the change, on the module fixture above:

    ==================================== ========== =========
    quantity                             BEFORE     AFTER
    ==================================== ========== =========
    `BERT.build((None, 12))` tensors      0          17
    `BERT.build((None, 12))` params       0          12,768
    `embedding_weights`                   None       FOUND
    `use_weight_tying`                    False      True
    `CausalLanguageModel` variable count  19         18
    ==================================== ========== =========

    The variable count DROPS by one because the untied fallback was creating a
    whole `Dense(vocab_size)`; tying replaces it with a single `output_bias`.
    """
    assert model.embedding_weights is not None
    assert model.use_weight_tying is True


def test_the_locator_returns_none_rather_than_raising_on_a_bare_object(model):
    """Degradation contract: an unrecognised owner must not crash."""
    assert model._embedding_variable_of(object()) is None


def test_the_forward_pass_runs(model):
    ids = np.random.RandomState(0).randint(1, VOCAB, size=(2, SEQ)).astype("int32")
    logits = np.array(model(
        {"input_ids": ids, "attention_mask": np.ones((2, SEQ), "int32")},
        training=False))
    assert logits.shape == (2, SEQ, VOCAB)
    assert np.all(np.isfinite(logits))
