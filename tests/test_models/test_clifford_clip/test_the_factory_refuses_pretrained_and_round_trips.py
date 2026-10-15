"""`create_clifford_clip` is CliffordCLIP's one factory: it refuses `pretrained`
and round-trips.

GUIDE § 1: one ``create_<model>(variant, pretrained=False, **kwargs)`` that
only delegates to ``from_variant``, and ``pretrained=True`` raises
``NotImplementedError`` when no weights are distributed. CliffordCLIP had no
factory (its README documented the absence) and
``CliffordCLIP.from_variant(..., pretrained=True)`` died on Keras'
"Unrecognized keyword arguments" ``ValueError`` (plan-2026-10-14T062616-81938e81
Step 11). A dual encoder has no backbone/head split, so a path is refused like
``True`` (the D-033 ruling). ``vocab_size``, ``image_size`` and
``context_length`` describe the data, not the variant, so both entry points
require them (D-045; the Mamba ``vocab_size`` precedent).

The round trip uses ``nano``, the smallest variant, at the smallest resolution
its four-stage ladder accepts, compared BY VALUE: every weight and every output.
"""

import inspect

import keras
import numpy as np
import pytest

import dl_techniques.models.vision_language.clifford_clip as package
import dl_techniques.models.vision_language.clifford_clip.model as model_module
from dl_techniques.models.vision_language.clifford_clip import (
    CliffordCLIP,
    create_clifford_clip,
)

SMALLEST = "nano"
DATA = dict(vocab_size=64, image_size=64, context_length=16)


def test_the_factory_is_bound_in_the_leaf_namespace():
    assert "create_clifford_clip" in package.__all__
    assert package.create_clifford_clip is model_module.create_clifford_clip


@pytest.mark.parametrize("entry", [create_clifford_clip, CliffordCLIP.from_variant])
def test_the_data_sizes_are_required(entry):
    params = inspect.signature(entry).parameters
    for name in ("variant", "vocab_size", "image_size", "context_length"):
        assert params[name].default is inspect.Parameter.empty, name
    assert params["pretrained"].default is False
    with pytest.raises(TypeError):
        entry(SMALLEST, vocab_size=64)


@pytest.mark.parametrize("pretrained", [True, "/some/cliffordclip_nano.keras"])
@pytest.mark.parametrize("entry", ["factory", "from_variant"])
def test_pretrained_is_refused_before_anything_is_built(monkeypatch, entry, pretrained):
    """A truthy value, a path included, raises before the model is made."""
    def no_construction(self, *args, **kwargs):
        raise AssertionError("the model was constructed before the refusal")

    monkeypatch.setattr(CliffordCLIP, "__init__", no_construction)
    build = create_clifford_clip if entry == "factory" else CliffordCLIP.from_variant
    with pytest.raises(NotImplementedError, match="load_model"):
        build(SMALLEST, pretrained=pretrained, **DATA)


def test_an_unknown_keyword_is_refused():
    with pytest.raises(ValueError, match="bogus_kw"):
        create_clifford_clip(SMALLEST, bogus_kw=1, **DATA)


def test_the_positional_order_matches_from_variant():
    model = create_clifford_clip(SMALLEST, 64, 64, 16, vision_patch_size=4)
    assert (model.vocab_size, model.image_size, model.context_length) == (64, 64, 16)


def test_the_factory_round_trips_the_smallest_variant_by_value(tmp_path):
    keras.utils.set_random_seed(5)
    model = create_clifford_clip(
        SMALLEST, vision_patch_size=4, dropout_rate=0.0,
        vision_stochastic_depth_rate=0.0, text_stochastic_depth_rate=0.0, **DATA)
    model.build({"image": (None, 64, 64, 3), "text": (None, 16)})
    rng = np.random.default_rng(0)
    inputs = {
        "image": rng.standard_normal((3, 64, 64, 3)).astype("float32"),
        "text": rng.integers(1, 64, (3, 16)).astype("int32"),
    }
    before = model(inputs, training=False)

    path = tmp_path / "cliffordclip_nano.keras"
    model.save(path)
    restored = keras.models.load_model(path)

    assert type(restored) is CliffordCLIP
    assert len(restored.weights) == len(model.weights) > 0
    for source, target in zip(model.weights, restored.weights):
        np.testing.assert_array_equal(
            np.array(source), np.array(target), err_msg=source.path)
    after = restored(inputs, training=False)
    for key in ("image_features", "text_features", "logits_per_image",
                "logits_per_text"):
        value = keras.ops.convert_to_numpy(before[key])
        assert float(np.std(value)) > 0.0, key
        np.testing.assert_array_equal(
            value, keras.ops.convert_to_numpy(after[key]), err_msg=key)
