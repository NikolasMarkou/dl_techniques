"""`create_sam` is SAM 1's one factory: it refuses `pretrained` and round-trips.

GUIDE § 1: one ``create_<model>(variant, pretrained=False, **kwargs)`` that
only delegates to ``from_variant``, and ``pretrained=True`` raises
``NotImplementedError`` when no weights are distributed. SAM 1 had no factory,
and ``SAM.from_variant(..., pretrained=True)`` died on Keras' "Unrecognized
keyword arguments" ``ValueError`` (plan-2026-10-14T062616-81938e81 Step 11).
SAM has no backbone/head split, so a path is refused like ``True`` (the D-033
ruling): a local archive loads with ``keras.models.load_model(path)``.

The round trip uses ``vit_b``, the smallest published variant (the encoder
geometry is fixed at 1024 px), compared BY VALUE: every weight and the
``low_res_logits`` of one prompt.
"""

import keras
import numpy as np
import pytest

import dl_techniques.models.vision_language.sam.sam1 as sam1_package
import dl_techniques.models.vision_language.sam.sam1.model as model_module
from dl_techniques.models.vision_language.sam.sam1 import SAM, create_sam

SMALLEST = "vit_b"


def test_the_factory_is_bound_in_the_leaf_namespace():
    assert "create_sam" in sam1_package.__all__
    assert sam1_package.create_sam is model_module.create_sam


@pytest.mark.parametrize("pretrained", [True, "/some/sam_vit_b.keras"])
@pytest.mark.parametrize("entry", ["factory", "from_variant"])
def test_pretrained_is_refused_before_anything_is_built(monkeypatch, entry, pretrained):
    """A truthy value, a path included, raises before the encoder is made."""
    def no_construction(*args, **kwargs):
        raise AssertionError("a component was constructed before the refusal")

    monkeypatch.setattr(model_module, "ImageEncoderViT", no_construction)
    build = create_sam if entry == "factory" else SAM.from_variant
    with pytest.raises(NotImplementedError, match="load_model"):
        build(SMALLEST, pretrained=pretrained)


def test_an_unknown_variant_is_refused_first():
    with pytest.raises(ValueError, match="Unknown variant"):
        create_sam("vit_s", pretrained=True)


def test_an_unknown_keyword_is_refused():
    with pytest.raises(ValueError, match="bogus_kw"):
        create_sam(SMALLEST, bogus_kw=1)


def test_the_factory_round_trips_the_smallest_variant_by_value(tmp_path):
    keras.utils.set_random_seed(5)
    model = create_sam(SMALLEST)
    assert isinstance(model, SAM)
    inputs = {
        "image": np.random.default_rng(0).uniform(
            0.0, 255.0, (1, 64, 64, 3)).astype("float32"),
        "points": (
            keras.ops.convert_to_tensor([[[20.0, 40.0]]]),
            keras.ops.convert_to_tensor([[1]]),
        ),
        "original_size": keras.ops.convert_to_tensor((64, 64)),
    }
    reference = keras.ops.convert_to_numpy(
        model(inputs, multimask_output=True)["low_res_logits"])
    # Anti-vacuity: a constant output would pass any equality below.
    assert float(np.std(reference)) > 0.0

    path = tmp_path / "sam_vit_b.keras"
    model.save(path)
    restored = keras.models.load_model(path)

    assert type(restored) is SAM
    assert len(restored.weights) == len(model.weights) > 0
    for source, target in zip(model.weights, restored.weights):
        np.testing.assert_array_equal(
            np.array(source), np.array(target), err_msg=source.path)
    np.testing.assert_array_equal(
        reference,
        keras.ops.convert_to_numpy(
            restored(inputs, multimask_output=True)["low_res_logits"]),
    )
