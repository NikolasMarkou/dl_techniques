"""The ``mixed_float16`` BACKWARD pass of TreeTransformer has no non-finite gradient.

plan-2026-10-11T170958-dc4d2202 D-014. ``GroupAttention.call`` symmetrizes the
break probabilities with ``sqrt(a * a^T + 1e-9)``. In float16 the constant
``1e-9`` is exactly 0 and the softmax is exactly 0 off the adjacent band, so
``sqrt(0)`` has an infinite derivative and ``0 * inf = NaN`` reaches 29 of the
49 headless gradients (29 of 55 with a text-classification head) while the
forward stays finite. ``test_mixed_float16_no_nan`` in ``test_model.py`` is
forward-only and never saw it.

The expression now runs in float32 and is cast back to the compute dtype. (The
float32 forward is untouched: the cast is then a no-op, which the headless
controls pin with ``np.array_equal``.)
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.tree_transformer.model import TreeTransformer

_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3)
_HEADS = {"headless": {}, "headed": dict(include_top=True, task_config=_CLF)}


@pytest.fixture
def mixed_float16():
    original = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(original)


def _loss(output):
    """Mean squared output in float32 (the dict's logits when it is one)."""
    if isinstance(output, dict):
        output = output.get("logits", output.get("last_hidden_state"))
    return tf.reduce_mean(tf.cast(output, "float32") ** 2)


@pytest.mark.parametrize("padded", [False, True], ids=["no_padding", "padded"])
@pytest.mark.parametrize("head", sorted(_HEADS))
def test_every_gradient_is_finite_under_mixed_float16(mixed_float16, head, padded):
    keras.utils.set_random_seed(0)
    model = TreeTransformer(
        vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
        intermediate_size=64, max_len=16, **_HEADS[head])
    ids = np.random.default_rng(0).integers(1, 60, (4, 12)).astype("int32")
    mask = np.ones((4, 12), "int32")
    if padded:
        mask[:, 9:] = 0
    with tf.GradientTape() as tape:
        loss = _loss(model({"input_ids": ids, "attention_mask": mask}, training=True))
    variables = model.trainable_variables
    grads = tape.gradient(loss, variables)

    assert variables, "no trainable variable: the arm proves nothing"
    bad = [v.path for v, g in zip(variables, grads)
           if g is None or not np.isfinite(tf.convert_to_tensor(g).numpy()).all()]
    assert np.isfinite(float(loss))
    assert not bad, f"{len(bad)} of {len(variables)} gradients non-finite: {bad[:4]}"

