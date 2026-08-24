"""BLT must train through stock `fit()` with XLA on.

Rationale
---------
`PatchingLayer.compute_patch_ids` derived the expansion length from the DATA --
``ops.max(ops.sum(patch_lengths, axis=1))`` -- and fed it to `ops.arange`, so the
layer's OUTPUT SHAPE depended on its INPUT VALUES. XLA rejects that outright.
MEASURED at HEAD on GPU 1, a micro BLT, seq 16, batch 2, one epoch:

    jit_compile=True   InvalidArgumentError: Input 1 to node
                       byte_latent_transformer_1/range with op Range must be a
                       compile-time constant. XLA compilation requires that
                       operator arguments that represent shapes ... be
                       evaluated to concrete values.

Keras 3.8's `fit()` defaults to ``jit_compile="auto"``, which selects XLA on a
GPU, so `src/train/blt/train_blt.py` -- which compiles at that default -- could
not train at all. The remedy passes the STATIC sequence length in from the
caller's own byte tensor; it is not a guess, because `PatchingLayer` builds the
lengths as occupancy counts of a per-byte assignment and the row sum is exactly
`seq_len` structurally.

The test runs a REAL `fit()`, because the failure is a compilation failure: it
cannot be reproduced by calling the layer, and it is GREEN on CPU by
construction, so a CPU-only test passes against the defect it exists to catch.

See decisions.md D-034 (plan-2026-08-19T163559-499b6f0e).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.byte_latent_transformer.model import create_blt_model

# ---------------------------------------------------------------------

SEQ = 16
VOCAB = 260
BATCH = 2


@pytest.fixture
def data():
    x = np.random.RandomState(0).randint(
        0, VOCAB, size=(BATCH * 2, SEQ)).astype("int32")
    y = np.random.RandomState(1).randint(
        0, VOCAB, size=(BATCH * 2, SEQ)).astype("int32")
    return x, y


def _model():
    keras.utils.set_random_seed(0)
    return create_blt_model("micro", vocab_size=VOCAB, max_sequence_length=SEQ)


@pytest.mark.parametrize("jit_compile", [True, False])
def test_fit_completes(data, jit_compile):
    """`jit_compile=True` is the arm that raised; `False` is its control."""
    x, y = data
    model = _model()
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  jit_compile=jit_compile)
    history = model.fit(x, y, batch_size=BATCH, epochs=1, verbose=0)
    loss = history.history["loss"][-1]
    assert np.isfinite(loss), (
        f"fit(jit_compile={jit_compile}) produced a non-finite loss: {loss!r}"
    )


def test_compute_patch_ids_expands_to_the_supplied_length(data):
    """The remedy's own contract, checked directly."""
    from keras import ops

    model = _model()
    x, _ = data
    entropy_logits = model.entropy_model(x, training=False)
    entropy = model.entropy_model.compute_entropy(entropy_logits)
    patch_lengths = model.patcher(entropy, training=False)

    patch_ids = model.patcher.compute_patch_ids(patch_lengths, seq_len=SEQ)
    assert tuple(ops.shape(patch_ids)) == (BATCH * 2, SEQ)


def test_the_supplied_length_agrees_with_the_data_derived_one(data):
    """Anti-vacuity: a supplied length that DISAGREED would silently truncate.

    The whole remedy rests on the structural claim that the patch lengths sum
    to `seq_len` for any threshold. This asserts that claim on real data, so a
    change to `PatchingLayer.call` that breaks it is caught here rather than as
    a shape error two layers downstream.
    """
    from keras import ops

    model = _model()
    x, _ = data
    entropy_logits = model.entropy_model(x, training=False)
    entropy = model.entropy_model.compute_entropy(entropy_logits)
    patch_lengths = model.patcher(entropy, training=False)

    row_sums = np.array(ops.sum(patch_lengths, axis=1))
    assert np.all(row_sums == SEQ), (
        f"patch length row sums are {row_sums.tolist()}, not {SEQ}; the "
        f"occupancy-count invariant `compute_patch_ids` relies on is broken"
    )

    supplied = np.array(model.patcher.compute_patch_ids(
        patch_lengths, seq_len=SEQ))
    derived = np.array(model.patcher.compute_patch_ids(patch_lengths))
    np.testing.assert_array_equal(
        supplied, derived,
        err_msg="passing seq_len changed the patch ids; the remedy is not "
                "behaviour-preserving"
    )
