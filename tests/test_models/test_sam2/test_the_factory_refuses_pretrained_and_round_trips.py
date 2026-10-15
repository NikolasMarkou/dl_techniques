"""`create_sam2` is a one-line factory that refuses `pretrained` and round-trips.

GUIDE § 1: one ``create_<model>(variant, pretrained=False, **kwargs)`` that
only delegates to ``from_variant``, and ``pretrained=True`` raises
``NotImplementedError`` when no weights are distributed. ``create_sam2``
restated seven of ``from_variant``'s overrides as its own ``None``-defaulted
parameters and had no ``pretrained``; ``pretrained=True`` died on Keras'
"Unrecognized keyword arguments" ``ValueError`` (plan-2026-10-14T062616-81938e81
Step 11). SAM 2 has no backbone/head split, so a path is refused like ``True``
(the D-033 ruling). The overrides still reach the model through ``**kwargs``,
and an explicit ``None`` still defers to the variant table (S-3).
"""

import inspect

import keras
import numpy as np
import pytest

import dl_techniques.models.vision_language.sam.sam2 as sam2_package
import dl_techniques.models.vision_language.sam.sam2.model as model_module
from dl_techniques.models.vision_language.sam.sam2 import SAM2, create_sam2

from .test_model import images, max_abs_diff, pin_object_score

SMALLEST = "tiny"


def test_the_factory_only_delegates():
    """No restated override list: ``variant``, ``pretrained``, ``**kwargs``."""
    params = inspect.signature(create_sam2).parameters
    assert list(params) == ["variant", "pretrained", "kwargs"]
    assert params["variant"].default == "hiera_l"
    assert params["pretrained"].default is False
    assert sam2_package.create_sam2 is model_module.create_sam2


@pytest.mark.parametrize("pretrained", [True, "/some/sam2_tiny.keras"])
@pytest.mark.parametrize("entry", ["factory", "from_variant"])
def test_pretrained_is_refused_before_anything_is_built(monkeypatch, entry, pretrained):
    """A truthy value, a path included, raises before the encoder is made."""
    class NoConstruction:
        @staticmethod
        def from_variant(*args, **kwargs):
            raise AssertionError("a component was constructed before the refusal")

    monkeypatch.setattr(model_module, "SAM2ImageEncoder", NoConstruction)
    build = create_sam2 if entry == "factory" else SAM2.from_variant
    with pytest.raises(NotImplementedError, match="load_model"):
        build(SMALLEST, pretrained=pretrained)


def test_the_overrides_still_reach_the_model_through_kwargs():
    model = create_sam2(SMALLEST, num_maskmem=3, multimask_output=False)
    assert model.num_maskmem == 3 and model.multimask_output is False
    # S-3: an explicit None defers to the table.
    assert create_sam2(SMALLEST, num_maskmem=None).num_maskmem == \
        SAM2.MODEL_VARIANTS[SMALLEST]["num_maskmem"]


def test_an_unknown_keyword_is_refused():
    with pytest.raises(ValueError, match="bogus_kw"):
        create_sam2(SMALLEST, bogus_kw=1)


def test_the_factory_round_trips_the_smallest_variant_by_value(tmp_path):
    keras.utils.set_random_seed(5)
    model = create_sam2(SMALLEST)
    model.build(None)
    x = images()
    # A positive object score keeps the real decoder output on both sides
    # (a negative one makes both uniformly NO_OBJ_SCORE; see test_model.py).
    pin_object_score(model, 5.0)
    reference = model({"image": x})["low_res_logits"]
    assert float(np.std(keras.ops.convert_to_numpy(reference))) > 0.0

    path = tmp_path / "sam2_tiny.keras"
    model.save(path)
    restored = keras.models.load_model(path)

    assert type(restored) is SAM2
    assert len(restored.weights) == len(model.weights) > 0
    for source, target in zip(model.weights, restored.weights):
        assert source.path.endswith(target.path) or target.path.endswith(source.path)
        assert max_abs_diff(source, target) == 0.0, source.path
    assert max_abs_diff(reference, restored({"image": x})["low_res_logits"]) == 0.0
