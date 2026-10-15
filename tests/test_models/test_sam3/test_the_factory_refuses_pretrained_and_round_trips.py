"""`create_sam3` is SAM 3's one factory: it refuses `pretrained` and round-trips.

GUIDE § 1: one ``create_<model>(variant, pretrained=False, **kwargs)`` that
only delegates to ``from_variant``, and ``pretrained=True`` raises
``NotImplementedError`` when no weights are distributed. SAM 3 had no factory,
and ``Sam3Image.from_variant(..., pretrained=True)`` died on Keras'
"Unrecognized keyword arguments" ``ValueError`` (plan-2026-10-14T062616-81938e81
Step 11). Sam3Image has no backbone/head split, so a path is refused like
``True`` (the D-033 ruling).

The round trip uses ``tiny``, the smallest variant, compared BY VALUE: every
weight and every output, with the zero-initialized box head randomized first
so ``pred_boxes`` is not trivially equal on both sides (see test_model.py).
"""

import keras
import numpy as np
import pytest

import dl_techniques.models.vision_language.sam.sam3 as sam3_package
import dl_techniques.models.vision_language.sam.sam3.sam3_image as model_module
from dl_techniques.models.vision_language.sam.sam3 import Sam3Image, create_sam3

from .test_model import randomize

SMALLEST = "tiny"


def test_the_factory_is_bound_in_the_leaf_namespace():
    assert "create_sam3" in sam3_package.__all__
    assert sam3_package.create_sam3 is model_module.create_sam3


@pytest.mark.parametrize("pretrained", [True, "/some/sam3_tiny.keras"])
@pytest.mark.parametrize("entry", ["factory", "from_variant"])
def test_pretrained_is_refused_before_anything_is_built(monkeypatch, entry, pretrained):
    """A truthy value, a path included, raises before the backbone is made."""
    def no_construction(*args, **kwargs):
        raise AssertionError("a component was constructed before the refusal")

    monkeypatch.setattr(model_module, "Sam3ViTDetBackbone", no_construction)
    build = create_sam3 if entry == "factory" else Sam3Image.from_variant
    with pytest.raises(NotImplementedError, match="load_model"):
        build(SMALLEST, pretrained=pretrained)


def test_an_unknown_keyword_is_refused():
    with pytest.raises(ValueError, match="bogus_kw"):
        create_sam3(SMALLEST, bogus_kw=1)


def test_the_factory_round_trips_the_smallest_variant_by_value(tmp_path):
    keras.utils.set_random_seed(5)
    model = create_sam3(SMALLEST)
    assert isinstance(model, Sam3Image)
    model.build(None)
    randomize(model.transformer.bbox_embed[-1], seed=11)
    rng = np.random.default_rng(7)
    batch = {
        "image": rng.standard_normal((2, 32, 32, 3)).astype("float32"),
        "token_ids": rng.integers(1, 64, (2, 8)).astype("int32"),
        "token_padding_mask": np.array([[False] * 8, [False] * 5 + [True] * 3]),
    }
    before = model(batch, training=False)

    path = tmp_path / "sam3_tiny.keras"
    model.save(path)
    restored = keras.models.load_model(path)

    assert type(restored) is Sam3Image
    assert len(restored.weights) == len(model.weights) > 0
    for source, target in zip(model.weights, restored.weights):
        np.testing.assert_array_equal(
            np.array(source), np.array(target), err_msg=source.path)
    after = restored(batch, training=False)
    assert set(after) == set(before)
    for key, value in before.items():
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(value),
            keras.ops.convert_to_numpy(after[key]), err_msg=key)
    # Anti-vacuity: the three per-query outputs carry information (a constant
    # tensor compares equal for free). `presence_logit` is exactly 0.0 on a
    # fresh model, so it is compared above but cannot carry this check.
    for key in ("pred_logits", "pred_boxes", "pred_masks"):
        assert float(np.std(keras.ops.convert_to_numpy(before[key]))) > 0.0, key
