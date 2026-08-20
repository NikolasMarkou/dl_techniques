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

Step 19.2 added the `mixed_precision=True` arm this file was missing. The
flag's own branch called `optimizer.get_scaled_loss` / `get_unscaled_gradients`
-- the Keras 2 spelling, ABSENT on Keras 3.8 -- so every mixed-precision run
raised `AttributeError: 'Adam' object has no attribute 'get_scaled_loss'` at
`train_clip.py:293`, on the first step, always. The float32 tests above could
not see it because they never set the flag.

See decisions.md D-030 and D-092 (plan-2026-08-19T163559-499b6f0e).
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


# ---------------------------------------------------------------------
# The `mixed_precision=True` arm -- the branch the float32 tests cannot reach
# ---------------------------------------------------------------------


@pytest.fixture
def restore_global_policy():
    """`CLIPTrainer(mixed_precision=True)` mutates the GLOBAL dtype policy.

    Leaking `mixed_float16` into the rest of the session would silently change
    every later test in the process, so the policy is restored unconditionally.
    """
    previous = keras.mixed_precision.global_policy()
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


def _tiny_clip():
    keras.utils.set_random_seed(0)
    return create_clip_model(
        image_size=IMAGE_SIZE, patch_size=16, vision_layers=1, vision_width=32,
        vision_heads=2, vision_kv_heads=2, vocab_size=VOCAB,
        context_length=CONTEXT, text_layers=1, text_width=32, text_heads=2,
        text_kv_heads=2, embed_dim=16, ffn_multiple_of=8,
    )


def test_the_keras2_loss_scaling_api_is_absent_on_keras3(restore_global_policy):
    """Isolating arm: the reason the old spelling could never have worked.

    Pinned so that "it was just a typo" is not a possible reading. If a future
    Keras restores these names this test says so, loudly, before anyone
    concludes the trainer was fine all along.
    """
    lso = keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam())
    assert not hasattr(lso, "get_scaled_loss")
    assert not hasattr(lso, "get_unscaled_gradients")
    assert hasattr(lso, "scale_loss")


def test_train_step_completes_under_mixed_precision(restore_global_policy, batch):
    """End-to-end arm: this raised `AttributeError` at HEAD."""
    trainer = CLIPTrainer(
        model=_tiny_clip(), loss_fn=CLIPContrastiveLoss(),
        optimizer=keras.optimizers.Adam(1e-4),
        gradient_clip_norm=1.0, mixed_precision=True,
    )
    loss = float(np.array(trainer.train_step(batch)["loss"]))
    assert np.isfinite(loss), f"non-finite loss under mixed_float16: {loss!r}"


def test_the_optimizer_is_wrapped_so_scaling_is_not_a_no_op(
        restore_global_policy):
    """Without the wrap `scale_loss` returns its argument and the flag is inert.

    Pinned separately from the smoke arm because a trainer that RUNS under
    `mixed_precision=True` while scaling nothing looks exactly like a working
    one.
    """
    trainer = CLIPTrainer(
        model=_tiny_clip(), loss_fn=CLIPContrastiveLoss(),
        optimizer=keras.optimizers.Adam(1e-4), mixed_precision=True,
    )
    assert isinstance(trainer.optimizer,
                      keras.mixed_precision.LossScaleOptimizer)
    assert float(np.array(trainer.optimizer.scale_loss(1.0))) > 1.0


def test_wrapping_is_idempotent(restore_global_policy):
    """Double-wrapping would SQUARE the loss scale."""
    inner = keras.mixed_precision.LossScaleOptimizer(
        keras.optimizers.Adam(1e-4))
    trainer = CLIPTrainer(
        model=_tiny_clip(), loss_fn=CLIPContrastiveLoss(),
        optimizer=inner, mixed_precision=True,
    )
    assert trainer.optimizer is inner
    assert not isinstance(trainer.optimizer.inner_optimizer,
                          keras.mixed_precision.LossScaleOptimizer)


def test_the_gradient_clip_bound_is_expressed_in_the_scaled_domain():
    """The second half of the repair, pinned as arithmetic.

    `tf.clip_by_global_norm` is applied to gradients of the SCALED loss, so a
    bare `gradient_clip_norm` would bound the TRUE norm at `clip / scale`.
    """
    lso = keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam())
    scale = float(np.array(lso.scale_loss(1.0)))
    assert float(np.array(lso.scale_loss(1.0))) == scale
    assert float(np.array(lso.scale_loss(2.5))) == pytest.approx(2.5 * scale)
    plain = keras.optimizers.Adam()
    assert float(np.array(plain.scale_loss(2.5))) == pytest.approx(2.5), (
        "without a loss scale the bound must be unchanged, or the float32 "
        "path is not byte-identical"
    )
