"""After one optimizer step, no NanoVLM trainable variable has a `None` gradient.

Until plan-2026-10-14T062616-81938e81 (D-037) a tied NanoVLM built a
`shared_output_projection` Dense (768 x 32000 = 24,576,000 parameters on
'base'), saved and counted it, and never called it: its kernel was the one
trainable variable with a `None` gradient, waived by name in `test_model.py`.
The Dense is deleted; a tied model builds no output Dense at all.

Two objectives, because they answer different questions:

* every position of the output (`full_length_loss`): every weight is on the
  model's backward graph and moves in one stock `fit` step;
* the trainer's objective, `NanoVLMLoss`, which scores only the TEXT
  positions: no `None` gradient, tied and untied. That this objective leaves
  the last fusion block's vision-side sub-layers at an identically-zero
  gradient is a measured, open defect pinned by a strict xfail below.
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
#: (32 // 16) ** 2 patches + CLS.
VISION_LEN = 5


def _model(tied: bool, num_fusion_layers: int = 2) -> NanoVLM:
    model = NanoVLM.from_variant(
        "mini", vocab_size=VOCAB, use_shared_embedding=tied, output_dropout_rate=0.0,
        vision_config={"img_size": 32, "embed_dim": 16, "depth": 1, "num_heads": 2},
        text_config={"embed_dim": 16, "depth": 1, "num_heads": 2, "max_seq_len": 16},
        fusion_config={"attention_config": {"num_heads": 2},
                       "num_fusion_layers": num_fusion_layers},
    )
    model.build({"images": (None, 32, 32, 3), "text_tokens": (None, TEXT_LEN)})
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MEASURED 2026-10-15 (plan-2026-10-14T062616-81938e81 step 9): under "
    "NanoVLMLoss, which scores only the text positions, the LAST fusion "
    "block's vision-side sub-layers (cross_attn_<L-1>_0_to_1, norm_<L-1>_0, "
    "ffn_<L-1>_0: 12 variables) feed only the vision positions the loss drops, "
    "so their gradient is identically zero and they never train. The fix is "
    "a fusion design change (text attends to vision, vision does not attend "
    "to text), reported, not made, in step 9."))
def test_one_fit_step_under_the_training_loss_moves_every_trainable_variable():
    model = _model(tied=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=NanoVLMLoss())
    x, tokens = _batch()
    report = fit_one_step_moved_variables(model, x, tokens, batch_size=BATCH)
    assert report.unmoved == (), report.summary()
