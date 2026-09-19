"""The "ConvNeXt typically uses 4 stages" warning must not cry wolf on a built-in variant.

The audit of the first normalized ConvNeXt run (plan-2026-09-19T040641-db6932ec, F6)
found the same WARNING three times in ``run.log`` for the deliberate 2-stage ``cifar10``
variant: once at construction and once for each reload of a saved model, because a
reload passes ``depths`` as an explicit argument, so a check keyed on "did the caller
pass depths" (or on ``from_variant``) would fire again there. The check therefore
compares the VALUES against the class's ``MODEL_VARIANTS`` table.

A custom non-4-stage ``depths`` (matching no variant) must still be warned about.
"""

import logging

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2

FAMILIES = pytest.mark.parametrize("cls", [ConvNeXtV1, ConvNeXtV2], ids=["v1", "v2"])
STAGE_WARNING = "typically uses 4 stages"
# The cifar10 variant's depths with narrow dims: the check reads depths only.
BUILT_IN_DEPTHS = [5, 5]
TINY = dict(dims=[8, 16], num_classes=4, input_shape=(32, 32, 3))


def _stage_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING and STAGE_WARNING in r.getMessage()]


@FAMILIES
def test_built_in_variant_via_from_variant_does_not_warn(cls, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="dl"):
        cls.from_variant("cifar10", num_classes=4, input_shape=(32, 32, 3))
    assert _stage_warnings(caplog) == [], [r.getMessage() for r in caplog.records]


@FAMILIES
def test_the_same_depths_through_the_constructor_do_not_warn(cls, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="dl"):
        cls(depths=BUILT_IN_DEPTHS, **TINY)
    assert _stage_warnings(caplog) == [], [r.getMessage() for r in caplog.records]


@FAMILIES
def test_a_config_round_trip_and_a_keras_reload_do_not_warn(cls, tmp_path, caplog) -> None:
    model = cls(depths=BUILT_IN_DEPTHS, **TINY)
    model(np.zeros((1, 32, 32, 3), dtype="float32"))  # a subclassed model saves only once built
    path = tmp_path / "tiny.keras"
    model.save(path)
    with caplog.at_level(logging.DEBUG, logger="dl"):
        cls.from_config(model.get_config())
        keras.models.load_model(path)
    assert _stage_warnings(caplog) == [], [r.getMessage() for r in caplog.records]


@FAMILIES
def test_custom_non_four_stage_depths_still_warn(cls, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="dl"):
        cls(depths=[2, 2, 2], dims=[8, 16, 32], num_classes=4, input_shape=(32, 32, 3))
    warned = _stage_warnings(caplog)
    assert len(warned) == 1 and "got 3 stages" in warned[0].getMessage(), (
        [r.getMessage() for r in caplog.records]
    )


@FAMILIES
def test_a_four_stage_model_never_warns(cls, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="dl"):
        cls(depths=[1, 1, 1, 1], dims=[8, 16, 32, 64], num_classes=4, input_shape=(32, 32, 3))
    assert _stage_warnings(caplog) == []
