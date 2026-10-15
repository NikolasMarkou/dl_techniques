"""The NLP similarity and multiple-choice heads pool under the caller's padding mask.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md`` items NLP-2 and NLP-3).
Measured at ``846589b7f`` with mean pooling:

* NLP-2, ``TextSimilarityHead`` pair form: each side's dict was cut down to
  its ``'hidden_states'``, so a pair scored with masks and without them gave
  bit-identical scores. Reproduced. The single-sequence form, the one every
  headed language model feeds (``{'hidden_states', 'attention_mask'}``), had
  the same hole: flooding the padded positions with 100 moved the embeddings
  by 0.026.
* NLP-3, ``MultipleChoiceHead``: the dict path read ``'hidden_states'`` only,
  and flooding the padded positions of every choice moved the logits by
  0.024. Reproduced.
* A ``TextSimilarityHead`` pair given as a list (what a functional model
  hands a two-input layer) took the single-sequence branch and failed inside
  the norm (``'list' object has no attribute 'shape'``).

The masks reach the pooling, which is ``SequencePooling`` for
``cls``/``mean``/``max``/``last`` and the inline scorer for ``attention``
(D-002), so the rule for a masked reduction lives in one place. A mask for an
already pooled input is refused: there is nothing left to mask.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType, create_nlp_head

D = 8
B, S, C = 2, 6, 3
POOLINGS = ("cls", "mean", "max", "last", "attention")

_RNG = np.random.default_rng(7)
SEQ = _RNG.normal(size=(B, S, D)).astype("float32")
SEQ2 = _RNG.normal(size=(B, S, D)).astype("float32")
MASK = np.ones((B, S), "float32")
MASK[1, 3:] = 0.0
CHOICES = _RNG.normal(size=(B, C, S, D)).astype("float32")
CHOICE_MASK = np.ones((B, C, S), "float32")
CHOICE_MASK[0, 1, 2:] = 0.0
CHOICE_MASK[1, :, 4:] = 0.0


def _flood(x, mask):
    out = x.copy()
    out[mask == 0] = 100.0
    return out


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _similarity(pooling, function="cosine"):
    return create_nlp_head(
        NLPTaskConfig(name="s", task_type=NLPTaskType.TEXT_SIMILARITY),
        input_dim=D, pooling_type=pooling, similarity_function=function)


def _choice(pooling):
    return create_nlp_head(
        NLPTaskConfig(name="m", task_type=NLPTaskType.MULTIPLE_CHOICE),
        input_dim=D, pooling_type=pooling)


# ---------------------------------------------------------------------
# TextSimilarityHead
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pooling", POOLINGS)
def test_padding_never_reaches_a_single_sequence_embedding(pooling) -> None:
    head = _similarity(pooling)
    clean = head({"hidden_states": SEQ, "attention_mask": MASK})["embeddings"]
    flooded = head({"hidden_states": _flood(SEQ, MASK), "attention_mask": MASK})["embeddings"]
    np.testing.assert_allclose(_np(flooded), _np(clean), rtol=0, atol=1e-5)


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("function", ["cosine", "dot", "learned"])
@pytest.mark.parametrize("pooling", POOLINGS)
def test_padding_never_reaches_a_pair_score(pooling, function, container) -> None:
    head = _similarity(pooling, function)
    mask2 = np.ones((B, S), "float32")
    mask2[0, 2:] = 0.0

    def score(a, b):
        return _np(head(container([
            {"hidden_states": a, "attention_mask": MASK},
            {"hidden_states": b, "attention_mask": mask2},
        ]))["similarity_score"])

    np.testing.assert_allclose(
        score(_flood(SEQ, MASK), _flood(SEQ2, mask2)), score(SEQ, SEQ2),
        rtol=0, atol=1e-5)


@pytest.mark.parametrize("pooling", ("mean", "max", "last", "attention"))
def test_the_pair_mask_changes_the_score(pooling) -> None:
    """Anti-vacuity: with a partial mask the score differs from the unmasked one."""
    head = _similarity(pooling)
    masked = head(({"hidden_states": SEQ, "attention_mask": MASK},
                   {"hidden_states": SEQ2}))["similarity_score"]
    unmasked = head(({"hidden_states": SEQ}, {"hidden_states": SEQ2}))["similarity_score"]
    assert np.abs(_np(masked) - _np(unmasked)).max() > 1e-4


@pytest.mark.parametrize("pooling", POOLINGS)
def test_an_all_keep_mask_is_no_mask(pooling) -> None:
    head = _similarity(pooling)
    keep = np.ones((B, S), "float32")
    a = head({"hidden_states": SEQ, "attention_mask": keep})["embeddings"]
    b = head({"hidden_states": SEQ})["embeddings"]
    np.testing.assert_allclose(_np(a), _np(b), rtol=0, atol=1e-6)


def test_a_mask_for_a_pooled_similarity_input_is_refused() -> None:
    head = _similarity("mean")
    with pytest.raises(ValueError, match="TextSimilarityHead.*attention_mask"):
        head({"hidden_states": SEQ.mean(axis=1), "attention_mask": MASK})


# ---------------------------------------------------------------------
# MultipleChoiceHead
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pooling", POOLINGS)
def test_padding_never_reaches_a_choice_score(pooling) -> None:
    head = _choice(pooling)
    clean = head({"hidden_states": CHOICES, "attention_mask": CHOICE_MASK})["logits"]
    flooded = head({"hidden_states": _flood(CHOICES, CHOICE_MASK),
                    "attention_mask": CHOICE_MASK})["logits"]
    np.testing.assert_allclose(_np(flooded), _np(clean), rtol=0, atol=1e-5)


@pytest.mark.parametrize("pooling", POOLINGS)
def test_each_choice_pools_under_its_own_mask(pooling) -> None:
    """A choice scored alone (batch of one, its own mask) scores as in the batch."""
    head = _choice(pooling)
    together = _np(head({"hidden_states": CHOICES, "attention_mask": CHOICE_MASK})["logits"])
    for b in range(B):
        alone = _np(head({"hidden_states": CHOICES[b:b + 1],
                          "attention_mask": CHOICE_MASK[b:b + 1]})["logits"])
        np.testing.assert_allclose(alone[0], together[b], rtol=0, atol=1e-5)


@pytest.mark.parametrize("pooling", POOLINGS)
def test_an_all_keep_choice_mask_is_no_mask(pooling) -> None:
    head = _choice(pooling)
    a = head({"hidden_states": CHOICES, "attention_mask": np.ones((B, C, S), "float32")})
    b = head({"hidden_states": CHOICES})
    np.testing.assert_allclose(_np(a["logits"]), _np(b["logits"]), rtol=0, atol=1e-6)


def test_a_mask_for_pooled_choices_is_refused() -> None:
    head = _choice("mean")
    with pytest.raises(ValueError, match="MultipleChoiceHead.*attention_mask"):
        head({"hidden_states": CHOICES.mean(axis=2), "attention_mask": CHOICE_MASK[..., 0]})


def test_a_masked_choice_head_runs_in_a_functional_model() -> None:
    head = _choice("mean")
    inputs = {"hidden_states": keras.Input(shape=(C, S, D), name="hidden_states"),
              "attention_mask": keras.Input(shape=(C, S), name="attention_mask")}
    model = keras.Model(inputs, head(inputs)["logits"])
    eager = _np(head({"hidden_states": CHOICES, "attention_mask": CHOICE_MASK})["logits"])
    predicted = model.predict({"hidden_states": CHOICES, "attention_mask": CHOICE_MASK},
                              verbose=0)
    # predict runs a compiled graph; on the GPU its matmuls may take TF32, which
    # moved these logits by 2.1e-05 against the eager call (measured, GPU 1).
    np.testing.assert_allclose(predicted, eager, rtol=0, atol=1e-4)
