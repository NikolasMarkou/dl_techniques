"""`CLIPTrainer.train_step` / `val_step` must reach the optimizer.

Rationale
---------
Both called ``self.loss_fn(None, outputs)``. `keras.losses.Loss.__call__`
converts ``y_true`` to a tensor before dispatching to `call`, so a ``None``
label raises. MEASURED at HEAD, end to end on a tiny real CLIP on GPU 1:

    ValueError: None values not supported.
      at src/train/clip/train_clip.py, line 287, in train_step

This is device- and data-INDEPENDENT: it raises on the first step, always, so
the shipped `src/train/clip/` pipeline could not train at all. The repo already
had the remedy next door in `src/train/cliffordnet/train_clip.py` under a D-003
anchor; it is now a single shared entry point,
`CLIPContrastiveLoss.reduced_loss`.

Anti-vacuity
------------
Asserting "does not raise" is a weak oracle, so the tests also pin the loss
VALUE: for uniform (all-zero) logits the symmetric contrastive loss is exactly
``log(batch_size)``, an arithmetic fact independent of any weight.

See decisions.md D-030 (plan-2026-08-19T163559-499b6f0e).
"""

import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss
from dl_techniques.models.clip.model import create_clip_model
from train.clip.train_clip import CLIPTrainer

# ---------------------------------------------------------------------

BATCH = 4
CONTEXT = 8
VOCAB = 64
IMAGE_SIZE = 32


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_reduced_loss_is_log_of_the_batch_size_for_uniform_logits(batch_size):
    """The shared entry point, pinned to an arithmetic fact.

    With every logit equal, each row of the softmax is uniform over
    `batch_size` classes, so the cross-entropy against the diagonal is exactly
    ``log(batch_size)`` in both directions and the weighted sum is too.
    """
    loss_fn = CLIPContrastiveLoss()
    zeros = keras.ops.convert_to_tensor(
        np.zeros((batch_size, batch_size), dtype="float32"))
    value = float(np.array(loss_fn.reduced_loss(
        {"logits_per_image": zeros, "logits_per_text": zeros})))
    assert value == pytest.approx(math.log(batch_size), rel=1e-5), (
        f"reduced_loss on uniform logits is {value!r}, not log({batch_size}) = "
        f"{math.log(batch_size)!r}"
    )


def test_the_old_spelling_is_the_defect_not_a_style_choice():
    """Isolating arm: `loss_fn(None, ...)` must still raise.

    If this ever stops raising, `reduced_loss` is no longer load-bearing and the
    whole family can be reconsidered -- but until then, a reviewer who
    "simplifies" a call site back to `loss_fn(None, ...)` is re-introducing a
    trainer-breaking defect, and this test says so.
    """
    loss_fn = CLIPContrastiveLoss()
    zeros = keras.ops.convert_to_tensor(np.zeros((BATCH, BATCH), dtype="float32"))
    with pytest.raises((ValueError, TypeError)):
        loss_fn(None, {"logits_per_image": zeros, "logits_per_text": zeros})


@pytest.fixture(scope="module")
def trainer():
    keras.utils.set_random_seed(0)
    model = create_clip_model(
        image_size=IMAGE_SIZE, patch_size=16, vision_layers=1, vision_width=32,
        vision_heads=2, vision_kv_heads=2, vocab_size=VOCAB,
        context_length=CONTEXT, text_layers=1, text_width=32, text_heads=2,
        text_kv_heads=2, embed_dim=16, ffn_multiple_of=8,
    )
    return CLIPTrainer(model=model, loss_fn=CLIPContrastiveLoss(),
                       optimizer=keras.optimizers.Adam(1e-4))


@pytest.fixture(scope="module")
def batch():
    images = np.random.RandomState(0).randn(
        BATCH, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float32")
    texts = np.random.RandomState(1).randint(
        0, VOCAB, size=(BATCH, CONTEXT)).astype("int32")
    return tf.constant(images), tf.constant(texts)


def test_train_step_completes_and_returns_a_finite_loss(trainer, batch):
    """End-to-end arm: this raised `None values not supported` at HEAD."""
    result = trainer.train_step(batch)
    loss = float(np.array(result["loss"]))
    assert np.isfinite(loss), f"train_step returned a non-finite loss: {loss!r}"


def test_val_step_completes_and_returns_a_finite_loss(trainer, batch):
    """The same defect lived at the second call site, `val_step`."""
    result = trainer.val_step(batch)
    loss = float(np.array(result["val_loss"]))
    assert np.isfinite(loss), f"val_step returned a non-finite loss: {loss!r}"
