"""Guard: four models whose `build()` did not materialise what it promised.

Plan ``plan-2026-08-19T163559-499b6f0e``, step 17.1. Rules ``R-002`` / ``R-066``
/ ``R-069``.

The lazy-build contract this repository writes down is: ``.build(shape)`` alone,
with no forward pass, must materialise every weight, and a public entry point
must not hand back an unbuilt model. Four subjects broke it, each differently,
and each with a measured symptom rather than a style complaint:

======================================= ============================== ==============
subject                                 BEFORE                         AFTER
======================================= ============================== ==============
``MaskedAutoencoder.build``             8 of 51 tensors / 1,976 of     51 / 51,
                                        203,643 params + `UserWarning` 203,643, no
                                                                       warning
``BERT.build``                          **did not exist**: 0 tensors   17 / 12,768,
                                        / 0 params + `UserWarning`     no warning
``CausalLanguageModel`` weight tying    ``embedding_weights=None``,    FOUND, True
  on a BERT backbone                    ``use_weight_tying=False``
``CausalLanguageModel`` `.keras` round  **RAISES** `ValueError`        loads, delta
  trip                                  (1 object could not be loaded) exactly 0.0
``create_hierarchical_reasoning_model`` ``built=False``;               ``built=True``,
                                        ``count_params()`` RAISES      231,298 / 94
======================================= ============================== ==============

The BERT row is the load-bearing one: `CausalLanguageModel.build` locates the
embedding matrix by SHAPE over ``backbone.variables``, so a backbone that is
"built" with zero variables silently disabled weight tying — which is ON BY
DEFAULT — and made the save and load halves take different branches. That is the
`.keras` raise. One `build()` closes all three.

See ``decisions.md`` D-048, D-049, D-051.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest
import keras

from dl_techniques.models.bert.bert import BERT
from dl_techniques.models.masked_autoencoder.mae import MaskedAutoencoder
from dl_techniques.models.masked_language_model.clm import CausalLanguageModel
from dl_techniques.models.hierarchical_reasoning_model.model import (
    create_hierarchical_reasoning_model,
)

VOCAB = 64
HIDDEN = 32
SEQ = 12


def _build_and_capture_warnings(model, input_shape):
    """`.build(shape)` with the Keras "no build() method" UserWarning captured.

    Keras reports under-materialisation as a `UserWarning` and then exits 0, so
    "did the build run" cannot be answered by an exit code. This returns both
    the post-build weight census and the warning count.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.build(input_shape)
        user_warnings = [
            str(w.message) for w in caught
            if issubclass(w.category, UserWarning)
            and "does not have a `build()` method" in str(w.message)
        ]
    return len(model.weights), model.count_params(), user_warnings


# ---------------------------------------------------------------------------
# MaskedAutoencoder — R-002
# ---------------------------------------------------------------------------

def _tiny_encoder():
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    for _ in range(4):
        x = keras.layers.Conv2D(8, 3, strides=2, padding="same")(x)
    return keras.Model(inputs, x, name="tiny_enc")


def test_masked_autoencoder_build_materialises_every_weight():
    model = MaskedAutoencoder(
        encoder=_tiny_encoder(), patch_size=16, input_shape=(32, 32, 3))
    tensors, params, warns = _build_and_capture_warnings(model, (None, 32, 32, 3))

    assert warns == [], f"Keras reported an unbuilt layer: {warns}"
    assert (tensors, params) == (51, 203643), (
        f"`.build(shape)` reached {tensors} tensors / {params} params; the model "
        "has 51 / 203,643 (measured at HEAD: 8 / 1,976)"
    )


def test_masked_autoencoder_build_agrees_with_the_first_call():
    """ANTI-VACUITY: the post-build census must equal the post-call census."""
    model = MaskedAutoencoder(
        encoder=_tiny_encoder(), patch_size=16, input_shape=(32, 32, 3))
    model.build((None, 32, 32, 3))
    after_build = (len(model.weights), model.count_params())
    model(np.random.RandomState(0).rand(2, 32, 32, 3).astype("float32"),
          training=False)
    assert (len(model.weights), model.count_params()) == after_build


# ---------------------------------------------------------------------------
# BERT — R-002, and the root cause of the two CLM rows
# ---------------------------------------------------------------------------

def _bert():
    return BERT(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=1, num_heads=2,
        intermediate_size=64, max_position_embeddings=64,
        hidden_dropout_rate=0.0, attention_probs_dropout_rate=0.0,
    )


def test_bert_build_materialises_every_weight():
    tensors, params, warns = _build_and_capture_warnings(_bert(), (None, SEQ))

    assert warns == [], f"Keras reported an unbuilt layer: {warns}"
    assert (tensors, params) == (17, 12768), (
        f"`BERT.build((None, {SEQ}))` reached {tensors} tensors / {params} "
        "params; measured at HEAD: 0 / 0"
    )


def test_bert_build_agrees_with_the_first_call():
    """ANTI-VACUITY. At HEAD this pair read 0/0 then 17/12,768."""
    model = _bert()
    model.build((None, SEQ))
    after_build = (len(model.weights), model.count_params())
    ids = np.zeros((2, SEQ), dtype="int32")
    model({"input_ids": ids, "attention_mask": np.ones((2, SEQ), "int32")},
          training=False)
    assert (len(model.weights), model.count_params()) == after_build


def test_bert_build_accepts_the_dict_spelling():
    """`build()` is called with whatever `call()` is called with."""
    model = _bert()
    model.build({"input_ids": (None, SEQ), "attention_mask": (None, SEQ)})
    assert len(model.weights) == 17


def test_bert_build_rejects_a_shape_that_is_not_input_ids():
    with pytest.raises(ValueError, match="input_ids"):
        _bert().build((None, SEQ, HIDDEN))


def test_bert_compute_output_shape_answers_without_a_call():
    shapes = _bert().compute_output_shape((None, SEQ))
    assert shapes["last_hidden_state"] == (None, SEQ, HIDDEN)
    assert shapes["attention_mask"] == (None, SEQ)


# ---------------------------------------------------------------------------
# CausalLanguageModel — R-069 (weight tying + the `.keras` round trip)
# ---------------------------------------------------------------------------

@pytest.fixture
def clm_on_bert():
    # `verify_causality=False`: BERT is bidirectional by construction and the
    # causality probe would (correctly) reject it. These tests are about the
    # BUILD contract.
    model = CausalLanguageModel(
        backbone=_bert(), vocab_size=VOCAB, verify_causality=False)
    ids = np.zeros((2, SEQ), dtype="int32")
    model({"input_ids": ids, "attention_mask": np.ones((2, SEQ), "int32")},
          training=False)
    return model


def test_weight_tying_is_actually_active(clm_on_bert):
    """RED at HEAD: `embedding_weights` None, `use_weight_tying` False.

    This is the test `test_the_embedding_locator_finds_a_keras_embedding.py`
    carried as `xfail(strict=True)`, restated here now that `BERT.build` exists.
    """
    assert clm_on_bert.embedding_weights is not None
    assert clm_on_bert.use_weight_tying is True


def test_the_causal_language_model_survives_a_keras_round_trip(clm_on_bert):
    """RED at HEAD: `ValueError: A total of 1 objects could not be loaded`."""
    rng = np.random.RandomState(0)
    inputs = {
        "input_ids": rng.randint(1, VOCAB, (2, SEQ)).astype("int32"),
        "attention_mask": np.ones((2, SEQ), "int32"),
    }
    # Perturb every weight off its initializer, so a fresh-random reload cannot
    # coincide with the saved state.
    for weight in clm_on_bert.weights:
        weight.assign(rng.standard_normal(weight.shape).astype("float32") * 0.05)

    before = np.asarray(keras.ops.convert_to_numpy(
        clm_on_bert(inputs, training=False)))

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "clm.keras")
        clm_on_bert.save(path)
        restored = keras.models.load_model(path, compile=False)

    after = np.asarray(keras.ops.convert_to_numpy(restored(inputs, training=False)))

    assert len(restored.weights) == len(clm_on_bert.weights)
    # ANTI-VACUITY: the comparison is only meaningful if the output varies.
    assert float(np.ptp(before)) > 1e-3, "the forward output is nearly constant"
    assert float(np.max(np.abs(before - after))) == 0.0


# ---------------------------------------------------------------------------
# hierarchical_reasoning_model — R-066
# ---------------------------------------------------------------------------

def test_the_hrm_factory_returns_a_built_model():
    """RED at HEAD: `built=False`, and `count_params()` RAISES.

    `src/train/hrm/train_hrm.py` calls `count_params()` inside
    `HRMTrainer.__init__`, so the trainer module could not start at all.
    """
    model = create_hierarchical_reasoning_model(
        vocab_size=16, seq_len=8, embed_dim=32, num_puzzle_identifiers=4)

    assert model.built is True
    assert model.count_params() == 231298
    assert len(model.weights) == 94
