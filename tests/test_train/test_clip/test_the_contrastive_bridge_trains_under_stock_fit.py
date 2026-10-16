"""The CLIP trainer's stock-``fit`` bridge must train CLIP, temperature included.

Rationale
---------
`src/train/clip/train_clip.py` used to train through a hand-written
``CLIPTrainer`` (a ``GradientTape`` ``train_step`` and its own ``fit``). Its
history is two shipped defects, both on the first step, always:

* it called ``self.loss_fn(None, outputs)``; `keras.losses.Loss.__call__`
  converts ``y_true`` to a tensor before dispatching, so a ``None`` label
  raised ``ValueError: None values not supported`` (D-030 of
  plan-2026-08-19T163559-499b6f0e);
* its ``mixed_precision=True`` branch called the Keras 2 loss-scaling API,
  absent on Keras 3.8 (D-092 there), and had to wrap the optimizer and scale
  the clip bound by hand because the loop never called ``compile``.

Since plan-2026-10-14T062616-81938e81 D-036 the trainer compiles CLIP through
`compile_for_contrastive_training`: `StackedCLIPLogits` stacks the two logit
matrices into one ``(N, 2, N)`` output, `StackedCLIPContrastiveLoss` and the
stacked metrics unstack it, and stock ``fit`` does the rest. These tests pin
that bridge: its loss is the repo loss on the model's own output, a stock step
moves every CLIP variable (``logit_scale``, the learnable temperature,
included), its metrics read the right matrix, and under ``mixed_float16`` the
stock ``compile`` does what the old loop did by hand.

Anti-vacuity: the loss VALUE is pinned on uniform logits, where the symmetric
contrastive loss is exactly ``log(batch_size)``, an arithmetic fact
independent of any weight.
"""

import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss
from dl_techniques.models.vision_language.clip.model import CLIP
from train.clip.train_clip import build_clip_optimizer
from train.common.contrastive import (
    StackedCLIPAccuracy,
    StackedCLIPContrastiveLoss,
    StackedCLIPLogits,
    StackedCLIPRecallAtK,
    compile_for_contrastive_training,
)

# ---------------------------------------------------------------------

BATCH = 4
CONTEXT = 8
VOCAB = 64
IMAGE_SIZE = 32


def _tiny_clip() -> CLIP:
    keras.utils.set_random_seed(0)
    return CLIP(
        image_size=IMAGE_SIZE, patch_size=16, vision_layers=1, vision_width=32,
        vision_heads=2, vision_kv_heads=2, vocab_size=VOCAB,
        context_length=CONTEXT, text_layers=1, text_width=32, text_heads=2,
        text_kv_heads=2, embed_dim=16, ffn_multiple_of=8,
    )


def _inputs() -> dict:
    """Distinct images and distinct, unpadded token rows (ids >= 2)."""
    images = np.random.RandomState(0).rand(
        BATCH, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float32")
    texts = np.random.RandomState(1).randint(
        2, VOCAB, size=(BATCH, CONTEXT)).astype("int32")
    assert len({row.tobytes() for row in texts}) == BATCH
    return {"image": images, "text": texts}


def _fit_dataset(steps: int) -> tf.data.Dataset:
    dummy = np.zeros((BATCH,), dtype="float32")
    return tf.data.Dataset.from_tensors((_inputs(), dummy)).repeat(steps)


# ---------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_the_bridge_loss_is_log_of_the_batch_size_for_uniform_logits(batch_size):
    """With every logit equal each softmax row is uniform over ``batch_size``
    classes, so the cross-entropy is ``log(batch_size)`` in both directions."""
    stacked = keras.ops.zeros((batch_size, 2, batch_size))
    value = float(np.array(StackedCLIPContrastiveLoss()(
        np.zeros((batch_size,), dtype="float32"), stacked)))
    assert value == pytest.approx(math.log(batch_size), rel=1e-5), (
        f"bridge loss on uniform logits is {value!r}, not log({batch_size}) = "
        f"{math.log(batch_size)!r}"
    )


def test_the_old_spelling_is_the_defect_not_a_style_choice():
    """Isolating arm: ``loss_fn(None, ...)`` must still raise.

    It is why `StackedCLIPContrastiveLoss` calls the inner loss's ``call`` and
    why the trainer's batches carry a dummy ``y``. If this stops raising, both
    can be reconsidered.
    """
    loss_fn = CLIPContrastiveLoss()
    zeros = keras.ops.convert_to_tensor(np.zeros((BATCH, BATCH), dtype="float32"))
    with pytest.raises((ValueError, TypeError)):
        loss_fn(None, {"logits_per_image": zeros, "logits_per_text": zeros})


def test_the_bridge_loss_is_the_repo_loss_on_the_model_output():
    """Stacking and unstacking copy values; nothing else may differ.

    The reference is `CLIPContrastiveLoss.call` on the CLIP's own output dict,
    reduced by the mean, exactly as the old loop reduced it.
    """
    clip = _tiny_clip()
    inputs = _inputs()
    outputs = clip(inputs, training=False)
    expected = np.array(keras.ops.mean(CLIPContrastiveLoss().call(None, outputs)))

    stacked = StackedCLIPLogits(clip)(inputs, training=False)
    np.testing.assert_array_equal(
        np.array(stacked[:, 0]), np.array(outputs["logits_per_image"]))
    np.testing.assert_array_equal(
        np.array(stacked[:, 1]), np.array(outputs["logits_per_text"]))
    value = np.array(StackedCLIPContrastiveLoss()(
        np.zeros((BATCH,), dtype="float32"), stacked))
    np.testing.assert_array_equal(value, expected)


def test_the_inner_loss_adds_no_second_temperature():
    """CLIP multiplies by its learnable ``exp(logit_scale)``; a loss-side
    temperature would compound it."""
    wrapper = compile_for_contrastive_training(_tiny_clip(), keras.optimizers.Adam(1e-3))
    assert isinstance(wrapper.loss, StackedCLIPContrastiveLoss)
    assert wrapper.loss.inner.apply_temperature is False


# ---------------------------------------------------------------------
# A stock step trains the model
# ---------------------------------------------------------------------


def test_a_stock_fit_step_moves_every_variable_including_logit_scale():
    """Two stock ``fit`` steps; every trainable CLIP variable changes value.

    ``logit_scale`` is the learnable log-temperature and lives on the CLIP
    itself (``CLIP.build``); the wrapper owns no variable of its own. A
    ``stop_gradient`` anywhere in the path leaves a variable at its init, and
    the failure names it.
    """
    clip = _tiny_clip()
    clip(_inputs(), training=False)
    wrapper = compile_for_contrastive_training(clip, keras.optimizers.Adam(1e-3))
    tracked = list(clip.trainable_variables)
    assert [v.path for v in wrapper.trainable_variables] == [v.path for v in tracked]
    assert "logit_scale" in [v.name for v in tracked]
    before = {v.path: np.array(v.numpy(), copy=True) for v in tracked}

    history = wrapper.fit(_fit_dataset(2), epochs=2, steps_per_epoch=1, verbose=0)

    losses = history.history["loss"]
    assert len(losses) == 2 and all(np.isfinite(v) for v in losses), losses
    frozen = [v.path for v in tracked if np.array_equal(before[v.path], v.numpy())]
    assert not frozen, f"left at their init after 2 stock fit steps: {frozen}"


def test_stock_evaluate_returns_a_finite_loss_and_bounded_metrics():
    """The evaluation path (the old ``val_step``) runs through stock ``evaluate``."""
    wrapper = compile_for_contrastive_training(_tiny_clip(), keras.optimizers.Adam(1e-3))
    result = wrapper.evaluate(_fit_dataset(1), verbose=0, return_dict=True)
    assert set(result) == {"loss", "clip_i2t_accuracy", "clip_i2t_recall@5"}, result
    assert np.isfinite(result["loss"]), result
    assert 0.0 <= result["clip_i2t_accuracy"] <= 1.0, result
    assert 0.0 <= result["clip_i2t_recall@5"] <= 1.0, result


def test_the_metrics_read_the_image_to_text_matrix():
    """Index 0 of the stacked axis is ``logits_per_image``.

    The loss is symmetric under swapping the two matrices (weights 0.5/0.5),
    so only the metrics can see a swap. Here the image->text matrix ranks every
    pair right and the text->image matrix ranks every pair wrong.
    """
    n = 6
    right = 10.0 * np.eye(n, dtype="float32")
    wrong = 10.0 * np.roll(np.eye(n, dtype="float32"), 1, axis=1)
    stacked = keras.ops.convert_to_tensor(np.stack([right, wrong], axis=1))
    accuracy, recall = StackedCLIPAccuracy(), StackedCLIPRecallAtK(k=1)
    accuracy.update_state(np.zeros((n,), "float32"), stacked)
    recall.update_state(np.zeros((n,), "float32"), stacked)
    assert float(accuracy.result()) == 1.0
    assert float(recall.result()) == 1.0


# ---------------------------------------------------------------------
# mixed_float16: what stock compile does that the old loop did by hand
# ---------------------------------------------------------------------


@pytest.fixture
def mixed_float16():
    """Set the GLOBAL policy for one test and restore it unconditionally.

    Leaking ``mixed_float16`` into the rest of the session would silently
    change every later test in the process.
    """
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


def test_the_keras2_loss_scaling_api_is_absent_on_keras3():
    """Isolating arm: the reason the old loop's spelling could never have worked.

    If a future Keras restores these names this test says so, loudly.
    """
    lso = keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam())
    assert not hasattr(lso, "get_scaled_loss")
    assert not hasattr(lso, "get_unscaled_gradients")
    assert hasattr(lso, "scale_loss")


def test_a_stock_fit_step_is_finite_under_mixed_precision(mixed_float16):
    """The trainer's own optimizer, a ``mixed_float16`` CLIP, one stock step."""
    clip = _tiny_clip()
    assert clip.compute_dtype == "float16"
    wrapper = compile_for_contrastive_training(clip, build_clip_optimizer(1e-3, 0))
    history = wrapper.fit(_fit_dataset(1), epochs=1, verbose=0)
    loss = history.history["loss"][0]
    assert np.isfinite(loss), f"non-finite loss under mixed_float16: {loss!r}"


def test_compile_wraps_the_optimizer_so_scaling_is_not_a_no_op(mixed_float16):
    """Without the wrap ``scale_loss`` returns its argument and fp16 gradients
    underflow; a run that scales nothing looks exactly like a working one."""
    wrapper = compile_for_contrastive_training(_tiny_clip(), build_clip_optimizer(1e-3, 0))
    assert isinstance(wrapper.optimizer, keras.mixed_precision.LossScaleOptimizer)
    assert float(np.array(wrapper.optimizer.scale_loss(1.0))) > 1.0


def test_wrapping_is_idempotent(mixed_float16):
    """Double-wrapping would SQUARE the loss scale."""
    inner = keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam(1e-4))
    wrapper = compile_for_contrastive_training(_tiny_clip(), inner)
    assert wrapper.optimizer is inner
    assert not isinstance(inner.inner_optimizer,
                          keras.mixed_precision.LossScaleOptimizer)


def test_the_clip_bound_applies_to_the_true_gradient_norm():
    """The trainer clips by global norm 1.0, on UNSCALED gradients.

    The old loop had to scale the bound by hand (clipping scaled gradients at
    a bare 1.0 bounds the true norm at ``1 / scale``). Stock ``fit`` hands
    `LossScaleOptimizer.apply` the scaled gradients; it unscales them and the
    inner optimizer clips. MEASURED here with SGD(lr=1) and the same clip
    setting: a true gradient of norm 10 moves the variable by exactly 1.
    """
    assert build_clip_optimizer(1e-3, 0).global_clipnorm == 1.0

    lso = keras.mixed_precision.LossScaleOptimizer(
        keras.optimizers.SGD(learning_rate=1.0, global_clipnorm=1.0))
    w = keras.Variable(np.zeros((4,), dtype="float32"))
    lso.build([w])
    scale = float(np.array(lso.scale_loss(1.0)))
    assert scale > 1.0
    true_grad = np.array([10.0, 0.0, 0.0, 0.0], dtype="float32")
    lso.apply([keras.ops.convert_to_tensor(true_grad * scale)], [w])
    np.testing.assert_allclose(np.array(w.numpy()), [-1.0, 0.0, 0.0, 0.0], rtol=1e-6)
