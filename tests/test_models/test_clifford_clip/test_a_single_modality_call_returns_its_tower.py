"""CliffordCLIP keeps CLIP's partial ``call`` contract, and reports the shapes it returns.

Rationale
---------
The vanilla CLIP (``models/vision_language/clip/model.py``) answers an
image-only or a text-only input with that tower's features and omits the
logit keys, so a caption bank can be encoded without a dummy image batch.
At 56b696fe2 CliffordCLIP read ``inputs["image"]`` and ``inputs["text"]``
unconditionally and raised ``KeyError`` on either single-modality input
(the user's review ``research/2026_clifford_full_system_plan.md`` § 1,
row 7; plan-2026-10-14T062616-81938e81 D-054). Its ``compute_output_shape``
also reported square ``(batch, batch)`` logits taken from one modality's
batch, so an image batch of 2 against a caption batch of 3 was reported as
``(2, 2)`` while ``call`` returned ``(2, 3)``.

Every input form is checked against ``encode_image``/``encode_text`` by
value, and ``compute_output_shape`` against the tensors ``call`` returns.
The two batches differ, so a transposed or square logit shape cannot pass.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP

_IMG_BATCH, _TXT_BATCH = 2, 3
_FEATURES = {"image_features", "text_features"}
_LOGITS = {"logits_per_image", "logits_per_text", "logit_scale"}


@pytest.fixture(scope="module")
def model():
    m = CliffordCLIP.from_variant(
        "nano", vocab_size=97, image_size=64, context_length=16,
        vision_patch_size=4, dropout_rate=0.0,
        vision_stochastic_depth_rate=0.0, text_stochastic_depth_rate=0.0)
    m.build({"image": (None, 64, 64, 3), "text": (None, 16)})
    return m


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(4)
    images = rng.standard_normal((_IMG_BATCH, 64, 64, 3)).astype("float32")
    tokens = rng.integers(1, 97, size=(_TXT_BATCH, 16)).astype("int32")
    tokens[0, 9:] = 0  # right-padded caption: the last-non-pad anchor moves
    return images, tokens


def _forms(images, tokens):
    """Every accepted input form: (label, inputs, keys expected)."""
    return [
        ("dict image", {"image": images}, {"image_features"}),
        ("dict text", {"text": tokens}, {"text_features"}),
        ("dict both", {"image": images, "text": tokens}, _FEATURES | _LOGITS),
        ("tuple image", (images,), {"image_features"}),
        ("tuple text", (None, tokens), {"text_features"}),
        ("tuple both", (images, tokens), _FEATURES | _LOGITS),
    ]


def _shape_of(inputs):
    if isinstance(inputs, dict):
        return {k: tuple(v.shape) for k, v in inputs.items()}
    return tuple(None if v is None else tuple(v.shape) for v in inputs)


@pytest.mark.parametrize("index", range(6))
def test_each_input_form_returns_exactly_its_keys(model, data, index):
    label, inputs, keys = _forms(*data)[index]
    out = model(inputs, training=False)
    assert set(out) == keys, label


@pytest.mark.parametrize("index", range(6))
def test_each_tower_output_is_that_tower_s_encoding(model, data, index):
    images, tokens = data
    label, inputs, keys = _forms(images, tokens)[index]
    out = model(inputs, training=False)
    if "image_features" in keys:
        np.testing.assert_array_equal(
            np.asarray(out["image_features"]),
            np.asarray(model.encode_image(images, training=False)), err_msg=label)
    if "text_features" in keys:
        np.testing.assert_array_equal(
            np.asarray(out["text_features"]),
            np.asarray(model.encode_text(tokens, training=False)), err_msg=label)


def test_the_paired_logits_are_scaled_cosines(model, data):
    images, tokens = data
    out = model({"image": images, "text": tokens}, training=False)
    img = np.asarray(out["image_features"], dtype=np.float64)
    txt = np.asarray(out["text_features"], dtype=np.float64)
    scale = float(np.asarray(out["logit_scale"]))
    np.testing.assert_allclose(np.asarray(out["logits_per_image"]), scale * img @ txt.T,
                               rtol=0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(out["logits_per_text"]), scale * txt @ img.T,
                               rtol=0, atol=1e-4)


@pytest.mark.parametrize("index", range(6))
def test_compute_output_shape_matches_call(model, data, index):
    label, inputs, _ = _forms(*data)[index]
    out = model(inputs, training=False)
    reported = model.compute_output_shape(_shape_of(inputs))
    assert reported == {k: tuple(v.shape) for k, v in out.items()}, label


@pytest.mark.parametrize("inputs", [{}, (), (None, None), {"image": None, "text": None}],
                         ids=["empty dict", "empty tuple", "two Nones", "dict of Nones"])
def test_an_input_with_no_modality_is_refused(model, inputs):
    with pytest.raises(ValueError, match="image.*text|text.*image"):
        model(inputs, training=False)
    with pytest.raises(ValueError, match="image.*text|text.*image"):
        model.compute_output_shape(inputs)
