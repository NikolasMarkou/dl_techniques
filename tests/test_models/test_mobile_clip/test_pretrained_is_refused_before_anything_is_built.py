"""``pretrained`` is owned by ``from_variant`` and refused there, for v1 and v2.

No MobileCLIP checkpoint ships with this package, and a dual encoder has no
task head, so there is no backbone-only load to offer for a path either: a
whole ``.keras`` archive loads with ``keras.models.load_model(path)``. Until
plan-2026-10-14T062616-81938e81 Step 8 the raise sat in the two factories and
``from_variant`` did not take ``pretrained`` at all (Keras' "Unrecognized
keyword arguments" ``ValueError``). Each factory is now one ``return`` of
``from_variant``.

The refusal must come BEFORE construction. ``bogus_kw`` makes the order
observable: had the model been constructed first, Keras would raise
``ValueError: Unrecognized keyword arguments`` instead of
``NotImplementedError``.
"""

import inspect

import pytest

from dl_techniques.models.vision_language.mobile_clip import (
    MobileClipModel,
    MobileClipV2Model,
    create_mobile_clip_model,
    create_mobile_clip_v2,
)

_ENTRIES = {
    "v1.from_variant": (MobileClipModel.from_variant, "s0"),
    "v1.create_mobile_clip_model": (create_mobile_clip_model, "s0"),
    "v2.from_variant": (MobileClipV2Model.from_variant, "mobileclip2_s0"),
    "v2.create_mobile_clip_v2": (create_mobile_clip_v2, "mobileclip2_s0"),
}


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_pretrained_true_raises_before_construction(entry):
    fn, variant = _ENTRIES[entry]
    with pytest.raises(NotImplementedError, match=r"keras\.models\.load_model"):
        fn(variant, pretrained=True, bogus_kw=1)


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_a_path_is_refused_too(entry, tmp_path):
    """The file exists, so the refusal cannot be a missing-file error."""
    fn, variant = _ENTRIES[entry]
    archive = tmp_path / "mobile_clip.keras"
    archive.write_bytes(b"not read")
    with pytest.raises(NotImplementedError, match=r"keras\.models\.load_model"):
        fn(variant, pretrained=str(archive))


@pytest.mark.parametrize("entry", sorted(_ENTRIES))
def test_an_unknown_variant_is_still_named_first(entry):
    """Variant validation keeps running before the pretrained refusal."""
    fn, _variant = _ENTRIES[entry]
    with pytest.raises(ValueError, match="no_such_variant"):
        fn("no_such_variant", pretrained=True)


@pytest.mark.parametrize("cls", [MobileClipModel, MobileClipV2Model])
def test_from_variant_owns_the_pretrained_parameter(cls):
    params = inspect.signature(cls.from_variant).parameters
    assert params["pretrained"].default is False


@pytest.mark.parametrize("cls", [MobileClipModel, MobileClipV2Model])
def test_unknown_constructor_kwargs_are_refused(cls):
    """Both are subclassed models, so Keras refuses an unknown keyword."""
    variant = sorted(cls.MODEL_VARIANTS)[0]
    with pytest.raises(ValueError, match="bogus_kw"):
        cls.from_variant(variant, bogus_kw=1)
