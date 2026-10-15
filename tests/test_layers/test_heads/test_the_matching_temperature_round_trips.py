"""``ImageTextMatchingHead`` keeps its starting temperature in its config.

Guard for plan-2026-10-14T062616-81938e81 step 11.2 (the user's heads review,
``research/reviews/20261015_heads_review.md`` item VLM-5). Measured at
``8b4182f9d``: ``get_config`` left ``temperature`` out ("a weight, will be
saved automatically"), so ``from_config(get_config())`` of a head built with
``temperature=0.5`` started at the default 0.07. The ``.keras`` round trip
was not affected, because the trained value is a weight and is restored with
the other weights; the constructor value was lost everywhere a config is
rebuilt without weights (a clone, a multi-task rebuild, a fresh fine-tune).
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vlm import (
    ImageTextMatchingHead,
    MultiTaskVLMHead,
    VLMTaskConfig,
    VLMTaskType,
    create_multi_task_vlm_head,
)

DIM = 16


def _head(**kwargs) -> ImageTextMatchingHead:
    return ImageTextMatchingHead(
        task_config=VLMTaskConfig(name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                                  hidden_size=DIM), **kwargs)


def test_the_config_holds_the_starting_temperature() -> None:
    assert _head(temperature=0.5).get_config()["temperature"] == pytest.approx(0.5)


def test_a_rebuilt_head_starts_at_the_same_temperature() -> None:
    rebuilt = ImageTextMatchingHead.from_config(_head(temperature=0.5).get_config())
    assert float(keras.ops.convert_to_numpy(rebuilt.temperature)) == pytest.approx(0.5)


def test_a_keras_round_trip_restores_the_trained_value_and_the_starting_one() -> None:
    head = _head(temperature=0.5)
    v = keras.Input(shape=(5, DIM), name="vision_features")
    t = keras.Input(shape=(4, DIM), name="text_features")
    model = keras.Model([v, t], head({"vision_features": v, "text_features": t}))
    head.temperature.assign(0.3)  # what training would have moved it to
    xs = [np.random.default_rng(0).normal(size=(2, 5, DIM)).astype("float32"),
          np.random.default_rng(1).normal(size=(2, 4, DIM)).astype("float32")]
    before = model.predict(xs, verbose=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "itm.keras")
        model.save(path)
        restored = keras.models.load_model(path)
    head_back = [layer for layer in restored.layers
                 if isinstance(layer, ImageTextMatchingHead)][0]
    assert float(keras.ops.convert_to_numpy(head_back.temperature)) == pytest.approx(0.3)
    assert head_back.get_config()["temperature"] == pytest.approx(0.5)
    after = restored.predict(xs, verbose=0)
    np.testing.assert_allclose(before["logits"], after["logits"], rtol=0, atol=1e-6)


def test_a_shared_temperature_survives_a_multi_task_rebuild() -> None:
    head = create_multi_task_vlm_head(
        [VLMTaskConfig(name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
                       hidden_size=DIM)], temperature=0.25)
    rebuilt = MultiTaskVLMHead.from_config(head.get_config())
    assert float(keras.ops.convert_to_numpy(
        rebuilt.task_heads["itm"].temperature)) == pytest.approx(0.25)
