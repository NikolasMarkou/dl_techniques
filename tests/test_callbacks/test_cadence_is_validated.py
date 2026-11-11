"""Every callback with a cadence argument refuses a cadence below 1 at construction.

A cadence of 0 reached ``(epoch + 1) % 0`` inside ``fit`` and raised ``ZeroDivisionError``
at the end of an epoch (six callbacks, killing every callback listed after it); a negative
value silently ran every epoch (``% -1``) or every second one; three callbacks clamped
the value to 1 and hid the typo. ``SpatialLossLogger``, ``EMAShadowCallback``,
``PeriodicReassignCallback`` and ``SelfIteratePoolCallback`` already raised. The rule is
one ``ValueError("<arg> must be >= 1, got ...")`` raised inline before any side effect.
"""

from __future__ import annotations

import importlib
import inspect
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import numpy as np
import pytest
from unittest.mock import MagicMock

from dl_techniques.callbacks.analyzer_callback import EpochAnalyzerCallback
from dl_techniques.callbacks.coco_map_callback import COCOMAPCallback
from dl_techniques.callbacks.coco_multitask_visualization import (
    COCOMultiTaskPredictionGridCallback,
)
from dl_techniques.callbacks.convunext_bottleneck_monitor import (
    ConvUnextBottleneckMonitorCallback,
)
from dl_techniques.callbacks.depth_visualization import (
    DepthMetricsCurveCallback,
    DepthPredictionGridCallback,
)
from dl_techniques.callbacks.jepa_visualization import (
    LatentMaskOverlayCallback,
    PatchPredictionErrorCallback,
)
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback


# ---------------------------------------------------------------------
# One factory per class: (cadence argument name, builder(out_dir, value))
# ---------------------------------------------------------------------


def _rgb(n: int = 2) -> np.ndarray:
    return np.zeros((n, 4, 4, 3), dtype="float32")


CASES = {
    "ConvUnextBottleneckMonitorCallback": (
        "monitor_freq",
        lambda out, v: ConvUnextBottleneckMonitorCallback(
            full_model=MagicMock(), val_batch=_rgb(), output_dir=out, monitor_freq=v
        ),
    ),
    "EpochAnalyzerCallback": (
        "epoch_frequency",
        lambda out, v: EpochAnalyzerCallback(output_dir=str(out), epoch_frequency=v),
    ),
    "DepthPredictionGridCallback": (
        "frequency",
        lambda out, v: DepthPredictionGridCallback(
            val_rgb=_rgb(), val_depth=_rgb()[..., :1], val_mask=_rgb()[..., :1],
            output_dir=str(out), frequency=v,
        ),
    ),
    "DepthMetricsCurveCallback": (
        "frequency",
        lambda out, v: DepthMetricsCurveCallback(output_dir=str(out), frequency=v),
    ),
    "TrainingCurvesCallback": (
        "frequency",
        lambda out, v: TrainingCurvesCallback(output_dir=str(out), frequency=v),
    ),
    "COCOMultiTaskPredictionGridCallback": (
        "frequency",
        lambda out, v: COCOMultiTaskPredictionGridCallback(
            val_rgb=_rgb(), val_seg=np.zeros((2, 4, 4), "int32"),
            val_cls=np.zeros((2, 80), "float32"), output_dir=str(out), frequency=v,
        ),
    ),
    "LatentMaskOverlayCallback": (
        "frequency",
        lambda out, v: LatentMaskOverlayCallback(
            eval_pixels=_rgb(), output_dir=str(out), frequency=v
        ),
    ),
    "PatchPredictionErrorCallback": (
        "frequency",
        lambda out, v: PatchPredictionErrorCallback(
            eval_pixels=_rgb(), output_dir=str(out), frequency=v
        ),
    ),
    "COCOMAPCallback": (
        "frequency",
        lambda out, v: COCOMAPCallback(
            val_loader=MagicMock(), image_size=64, num_classes=2, reg_max=4,
            strides=(8, 16), frequency=v,
        ),
    ),
}

#: Constructor arguments that mean "run every N units" in the callbacks package.
CADENCE_ARGUMENTS = frozenset(
    {"frequency", "monitor_freq", "epoch_frequency", "log_every", "update_every",
     "every_n_epochs", "regen_freq"}
)

#: Classes that already raised before this rule, validated by their own test files.
ALREADY_VALIDATED = frozenset(
    {"SpatialLossLogger", "EMAShadowCallback", "PeriodicReassignCallback",
     "SelfIteratePoolCallback"}
)


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_a_cadence_below_one_is_refused_before_any_side_effect(tmp_path, name, value):
    arg, build = CASES[name]
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=f"{arg} must be >= 1, got {value}"):
        build(out, value)
    assert not out.exists(), f"{name} created its output dir before validating {arg}"


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_cadence_of_one_still_constructs(tmp_path, name):
    arg, build = CASES[name]
    callback = build(tmp_path / "out", 1)
    assert isinstance(callback, keras.callbacks.Callback)


def test_the_real_fit_no_longer_dies_at_the_end_of_the_first_epoch(tmp_path):
    """The reported failure: ``TrainingCurvesCallback(frequency=0)`` died inside ``fit``."""
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer="sgd", loss="mse")
    with pytest.raises(ValueError, match="frequency must be >= 1, got 0"):
        model.fit(
            np.zeros((4, 2), "float32"), np.zeros((4, 1), "float32"), epochs=1,
            verbose=0,
            callbacks=[TrainingCurvesCallback(output_dir=str(tmp_path), frequency=0)],
        )


def _classes_with_a_cadence_argument() -> set[str]:
    package = importlib.import_module("dl_techniques.callbacks")
    found: set[str] = set()
    for module_name in sorted(
        {m for m in package._LAZY_ATTRIBUTES.values()}
    ):
        module = importlib.import_module(f"dl_techniques.callbacks.{module_name}")
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not issubclass(cls, keras.callbacks.Callback):
                continue
            if CADENCE_ARGUMENTS & set(inspect.signature(cls.__init__).parameters):
                found.add(cls_name)
    return found


def test_no_callback_with_a_cadence_argument_is_left_out_of_this_rule():
    """A new callback that takes a cadence must be added here or it is unchecked."""
    found = _classes_with_a_cadence_argument()
    assert found == set(CASES) | ALREADY_VALIDATED, (
        f"unlisted: {sorted(found - set(CASES) - ALREADY_VALIDATED)}; "
        f"stale: {sorted((set(CASES) | ALREADY_VALIDATED) - found)}"
    )
