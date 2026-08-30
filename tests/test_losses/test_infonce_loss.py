"""Tests for :mod:`dl_techniques.losses.infonce_loss`.

The load-bearing artifact in this file is :func:`_reference_simcse_loss`: a **verbatim,
op-for-op transcription** of the trainer-private ``SimCSELoss.call`` body at
``src/train/embeddings_experimental/train_embeddings.py:260-281`` (read 2026-08-30).

It is transcribed from *that* file and never from
:class:`dl_techniques.losses.infonce_loss.SymmetricInfoNCELoss`. That independence is the
whole point: an oracle copied from the implementation it grades passes forever and proves
nothing. The trainer module is deliberately **not imported** -- importing it pulls in the
tensorflow-heavy study pipeline, and a live training run is executing against that file.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.infonce_loss import SymmetricInfoNCELoss

# ---------------------------------------------------------------------
# Reference oracle -- verbatim transcription, do not "simplify"
# ---------------------------------------------------------------------


def _reference_simcse_loss(y_pred, temperature):
    """The reference ``SimCSELoss.call`` body, transcribed op-for-op.

    Source: ``src/train/embeddings_experimental/train_embeddings.py:260-281``.

    :param y_pred: Stacked views, ``(batch, 2, embed_dim)``.
    :param temperature: Softmax temperature over the cosine similarities.
    :returns: The reference **scalar** loss, ``mean(forward + backward) / 2``.
    """
    view_a = y_pred[:, 0, :]
    view_b = y_pred[:, 1, :]
    logits = keras.ops.matmul(
        view_a, keras.ops.transpose(view_b)
    ) / temperature
    targets = keras.ops.arange(keras.ops.shape(logits)[0])
    forward = keras.losses.sparse_categorical_crossentropy(
        targets, logits, from_logits=True
    )
    backward = keras.losses.sparse_categorical_crossentropy(
        targets, keras.ops.transpose(logits), from_logits=True
    )
    return keras.ops.mean(forward + backward) / 2.0


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------


def _l2_normalized_views(batch: int, dim: int, seed: int) -> np.ndarray:
    """Build an L2-normalized ``(batch, 2, dim)`` float32 array.

    Unit-norm rows are what make the logits genuine cosine similarities, which is the
    regime the study's temperature (0.05) is calibrated for.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((batch, 2, dim)).astype("float32")
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return (raw / norms).astype("float32")


# ---------------------------------------------------------------------
# H-2: numerical equivalence with the reference implementation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch", [2, 4, 8, 64])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_the_loss_reproduces_the_reference_simcse_loss(batch, seed):
    """The reduced loss equals the trainer's scalar ``SimCSELoss`` value.

    A failure here means the library loss is **not** a drop-in replacement for
    ``train_embeddings.py``'s local ``SimCSELoss``, and the deferred trainer swap would
    silently change the in-flight study's numbers. Do NOT loosen the tolerance to make
    this pass -- diagnose the divergence.
    """
    y_pred = _l2_normalized_views(batch=batch, dim=256, seed=seed)
    temperature = 0.05

    expected = float(keras.ops.convert_to_numpy(
        _reference_simcse_loss(keras.ops.convert_to_tensor(y_pred), temperature)
    ))

    loss_fn = SymmetricInfoNCELoss(temperature=temperature)
    actual = float(keras.ops.convert_to_numpy(
        loss_fn(keras.ops.zeros((batch,), dtype="float32"), y_pred)
    ))

    assert actual == pytest.approx(expected, abs=1e-6, rel=0.0), (
        f"batch={batch} seed={seed}: new loss {actual!r} != reference {expected!r}. "
        f"The library loss is not equivalent to the trainer's SimCSELoss."
    )
