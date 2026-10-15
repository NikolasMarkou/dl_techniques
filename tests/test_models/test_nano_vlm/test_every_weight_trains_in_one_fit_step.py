"""After one optimizer step, every NanoVLM trainable variable has trained.

Until plan-2026-10-14T062616-81938e81 (D-037) a tied NanoVLM built a
`shared_output_projection` Dense (768 x 32000 = 24,576,000 parameters on
'base'), saved and counted it, and never called it: its kernel was the one
trainable variable with a `None` gradient, waived by name in `test_model.py`.
The Dense is deleted; a tied model builds no output Dense at all.

Two objectives, because they answer different questions:

* every position of the output (`full_length_loss`): every weight is on the
  model's backward graph and moves in one stock `fit` step;
* the trainer's objective, `NanoVLMLoss`, which scores only the TEXT
  positions: no `None` gradient, tied and untied, and every variable moves in
  one stock `fit` step. Until step 9.1 (D-040) this was a strict xfail: the
  last `MultiModalFusion` block's vision-side sub-layers (12 variables) fed
  only the vision positions the loss drops, so their gradient was identically
  zero. The vision tokens are now a prefix of one causal decoder that every
  text position reads, and the fusion layer is gone.

The decoder has 3 blocks: the dead layers were in the LAST of a multi-block
stack, so a one-block model could not show them.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.nano_vlm_loss import NanoVLMLoss
from dl_techniques.models.vision_language.nano_vlm import NanoVLM

from ..test_sam.dead_component_oracle import fit_one_step_moved_variables

VOCAB = 32
TEXT_LEN = 8
BATCH = 2
#: 32px images in 8px patches: (4 // 2) ** 2 pixel-shuffled tokens + CLS.
VISION_LEN = 5


def _model(tied: bool) -> NanoVLM:
    model = NanoVLM.from_variant(
        "mini", vocab_size=VOCAB, use_shared_embedding=tied, output_dropout_rate=0.0,
        vision_config={"img_size": 32, "patch_size": 8, "embed_dim": 16, "depth": 1,
                       "num_heads": 2},
        text_config={"embed_dim": 16, "depth": 3, "num_heads": 2, "max_seq_len": 16},
    )
    model.build({"images": (None, 32, 32, 3), "text_tokens": (None, TEXT_LEN)})
    assert model.num_vision_tokens == VISION_LEN
    return model


def _batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    images = rng.uniform(-1.0, 1.0, (BATCH, 32, 32, 3)).astype("float32")
    # No padding id 0, so every label position is scored.
    tokens = rng.integers(1, VOCAB, (BATCH, TEXT_LEN)).astype("int32")
    return {"images": images, "text_tokens": tokens}, tokens


def _gradients(model, loss_fn, x, y):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x, training=True))
    grads = tape.gradient(loss, model.trainable_variables)
    return {v.path: g for v, g in zip(model.trainable_variables, grads)}


@pytest.mark.parametrize("tied", [True, False])
def test_no_trainable_variable_has_a_none_gradient_under_the_training_loss(tied):
    model = _model(tied)
    x, y = _batch()
    grads = _gradients(model, NanoVLMLoss(), x, y)
    none = sorted(path for path, g in grads.items() if g is None)
    assert none == [], f"{len(none)} of {len(grads)} variables get no gradient: {none}"
    assert all(np.all(np.isfinite(keras.ops.convert_to_numpy(g))) for g in grads.values())
    assert model.ties_output_to_embeddings is tied
    assert (model.output_projection is None) is tied


@pytest.mark.parametrize("tied", [True, False])
def test_one_fit_step_on_every_position_moves_every_trainable_variable(tied):
    """Stock `compile`/`fit`, one step, Adam (no decay, so a zero gradient
    cannot move a weight): every trainable variable moves."""
    model = _model(tied)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    x, tokens = _batch()
    rng = np.random.default_rng(1)
    full_labels = rng.integers(0, VOCAB, (BATCH, VISION_LEN + TEXT_LEN)).astype("int32")
    full_labels[:, VISION_LEN:] = tokens

    report = fit_one_step_moved_variables(model, x, full_labels, batch_size=BATCH)

    assert report.unmoved == (), report.summary()
    assert report.total == len(model.trainable_variables)
    assert np.isfinite(report.final_loss)


@pytest.mark.parametrize("tied", [True, False])
def test_one_fit_step_under_the_training_loss_moves_every_trainable_variable(tied):
    model = _model(tied)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=NanoVLMLoss())
    x, tokens = _batch()
    report = fit_one_step_moved_variables(model, x, tokens, batch_size=BATCH)
    assert report.unmoved == (), report.summary()
    assert report.total == len(model.trainable_variables)
    # The vision side is in the population this asserts on.
    assert any("modality_projection" in path for path in report.moved), report.summary()
    assert any("vision_encoder" in path for path in report.moved), report.summary()
