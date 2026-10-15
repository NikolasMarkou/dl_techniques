"""Both MobileCLIP models train under stock ``compile``/``fit``.

MobileCLIP has no trainer (``grep -rn mobile_clip src/train`` is empty), so
this is its training proof (plan-2026-10-14T062616-81938e81 D-011): a tiny
model per class, synthetic image/caption pairs, the repo's
``CLIPContrastiveLoss`` -- the loss ``src/train/clip/train_clip.py`` trains
with -- and two stock ``fit`` steps. Asserted: every logged loss is finite and
EVERY trainable variable of BOTH towers, plus ``logit_scale``, changed value.
A tower that is built but bypassed, or cut off by a ``stop_gradient``, keeps
its init and is named in the failure.

The loss bridge (``_StackedLogitsModel`` / ``_StackedClipLoss``) is the one in
``test_mobile_clip_v2_end_to_end.py``: the model's dict output is stacked into
one ``(N, 2, N)`` tensor, because Keras' ``CompileLoss`` broadcasts one loss
over every leaf of a structured output. For v1, which emits features but no
logits, the bridge forms them with ``compute_clip_logits``.

Fixture rules (GUIDE section 8 and the Step 7.1 record): batch 4, because a
BatchNormalization at batch 1 zeroes its gradients; per-sample distinct
images and token sequences, the EOT (numeric maximum) at the last position
because both text towers pool at ``argmax`` of the ids.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision_language.mobile_clip.mobile_clip_v1 import (
    MobileClipModel,
)

from .test_mobile_clip_v2 import _IMG as _V2_IMG
from .test_mobile_clip_v2 import _SEQ as _V2_SEQ
from .test_mobile_clip_v2 import _VOCAB as _V2_VOCAB
from .test_mobile_clip_v2 import _tiny_model as _tiny_v2
from .test_mobile_clip_v2_end_to_end import _StackedClipLoss, _StackedLogitsModel

_BATCH = 4
_STEPS = 2

_V1_IMG = 64
_V1_SEQ = 8
_V1_VOCAB = 64


def _tiny_v1() -> MobileClipModel:
    """``s0`` (MobileNetV3Small stand-in, non-causal text) at 64px, tiny text."""
    row = MobileClipModel.MODEL_VARIANTS["s0"]
    return MobileClipModel.from_variant(
        "s0",
        embed_dim=16,
        image_config={
            **row["image_config"],
            "image_size": _V1_IMG,
            "projection_dropout_rate": 0.0,
        },
        text_config={
            **row["text_config"],
            "vocab_size": _V1_VOCAB,
            "max_seq_len": _V1_SEQ,
            "embed_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_size": 64,
            "dropout_rate": 0.0,
            "attention_dropout_rate": 0.0,
        },
    )


def _pairs(image_size: int, seq: int, vocab: int, seed: int):
    rng = np.random.default_rng(seed)
    images = rng.random((_BATCH, image_size, image_size, 3), dtype=np.float32)
    tokens = rng.integers(1, vocab - 1, size=(_BATCH, seq)).astype("int32")
    tokens[:, -1] = vocab - 1
    assert len({row.tobytes() for row in tokens}) == _BATCH
    return images, tokens


_SUBJECTS = {
    "v1": (_tiny_v1, (_V1_IMG, _V1_SEQ, _V1_VOCAB)),
    "v2": (_tiny_v2, (_V2_IMG, _V2_SEQ, _V2_VOCAB)),
}


@pytest.mark.parametrize("subject", sorted(_SUBJECTS))
def test_two_stock_fit_steps_move_every_variable_of_both_towers(subject):
    keras.utils.set_random_seed(11)
    build, (image_size, seq, vocab) = _SUBJECTS[subject]
    clip_model = build()
    images, tokens = _pairs(image_size, seq, vocab, seed=5)
    inputs = {"image": images, "text": tokens}
    clip_model(inputs, training=False)  # materialize before the snapshot

    towers = {
        "image_encoder": clip_model.image_encoder.trainable_variables,
        "text_encoder": clip_model.text_encoder.trainable_variables,
    }
    assert all(towers.values()), {k: len(v) for k, v in towers.items()}
    tracked = list(clip_model.trainable_variables)
    in_towers = {id(v) for vs in towers.values() for v in vs}
    rest = [v for v in tracked if id(v) not in in_towers]
    assert [v.name for v in rest] == ["logit_scale"], [v.path for v in rest]
    before = {id(v): np.array(v.numpy(), copy=True) for v in tracked}

    wrapper = _StackedLogitsModel(clip_model)
    wrapper.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=_StackedClipLoss(),
    )
    dummy = np.zeros((_BATCH,), dtype="float32")
    dataset = tf.data.Dataset.from_tensors((inputs, dummy)).repeat(_STEPS)
    history = wrapper.fit(dataset, epochs=_STEPS, steps_per_epoch=1, verbose=0)

    losses = [float(v) for v in history.history["loss"]]
    assert len(losses) == _STEPS
    assert all(np.isfinite(v) for v in losses), losses

    frozen = {
        name: [v.path for v in vs if np.array_equal(before[id(v)], v.numpy())]
        for name, vs in {**towers, "logit_scale": rest}.items()
    }
    assert not any(frozen.values()), (
        f"{subject}: trainable variables left at their init after {_STEPS} "
        f"fit steps: {frozen}"
    )
