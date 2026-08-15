"""`TabMModel.from_variant` must construct every preset it advertises.

Guard for C-33 (plan-2026-08-14T233721-d4f9beb2, step 30). At HEAD before the
fix, ``from_variant`` copied ``MODEL_VARIANTS[variant]`` -- every entry of which
carries a ``"description"`` key -- and splatted the whole dict into ``cls(...)``.
``TabMModel.__init__`` has no ``description`` parameter and forwards unknown
``**kwargs`` to ``keras.Model.__init__``, which raises
``ValueError: Unrecognized keyword arguments passed to TabMModel: {'description': ...}``.
All six presets were therefore unreachable through the documented entry point,
and nothing in the suite called ``from_variant`` at all.

The parametrization is driven off the real ``MODEL_VARIANTS`` dict rather than a
hand-written list, so a variant added later is covered automatically.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.tabm.model import TabMModel


VARIANT_IDS = sorted(TabMModel.MODEL_VARIANTS.keys())


@pytest.fixture(scope="module")
def variant_kwargs():
    return dict(n_num_features=6, cat_cardinalities=[3, 4], n_classes=2)


def test_every_variant_dict_carries_a_description():
    """Anti-vacuity: the defect only exists because the key is really there."""
    for name, cfg in TabMModel.MODEL_VARIANTS.items():
        assert "description" in cfg, f"variant {name!r} lost its description"


@pytest.mark.parametrize("variant", VARIANT_IDS)
def test_from_variant_constructs(variant, variant_kwargs):
    model = TabMModel.from_variant(variant, **variant_kwargs)
    assert isinstance(model, TabMModel)


@pytest.mark.parametrize("variant", VARIANT_IDS)
def test_from_variant_actually_applies_the_variant_config(variant, variant_kwargs):
    """A `from_variant` that constructed but ignored the preset would pass a
    bare construction check. Pin the fields the preset actually sets."""
    cfg = TabMModel.MODEL_VARIANTS[variant]
    model = TabMModel.from_variant(variant, **variant_kwargs)

    assert model.hidden_dims == cfg["hidden_dims"]
    assert model.k == cfg["k"]
    assert model.arch_type == cfg["arch_type"]
    # The metadata key must not have leaked onto the model as an attribute.
    assert not hasattr(model, "description")


def test_from_variant_model_runs_forward(variant_kwargs):
    """One end-to-end pass so the constructed preset is a usable model."""
    model = TabMModel.from_variant("micro", **variant_kwargs)
    x_num = np.random.rand(4, 6).astype("float32")
    x_cat = np.stack(
        [
            np.random.randint(0, 3, size=4),
            np.random.randint(0, 4, size=4),
        ],
        axis=-1,
    ).astype("int32")
    out = model([keras.ops.convert_to_tensor(x_num), keras.ops.convert_to_tensor(x_cat)])
    arr = keras.ops.convert_to_numpy(out)
    assert arr.shape[0] == 4
    assert arr.shape[1] == TabMModel.MODEL_VARIANTS["micro"]["k"]
    assert np.all(np.isfinite(arr))


def test_unknown_variant_still_raises(variant_kwargs):
    with pytest.raises(ValueError, match="Unknown variant"):
        TabMModel.from_variant("not-a-variant", **variant_kwargs)
