"""
Callbacks Module
================

Keras 3 callbacks for training control, schedules, diagnostics, visualization
and evaluation. Every public class is importable from the package root:

.. code-block:: python

    from dl_techniques.callbacks import (
        ConsecutiveIncreaseEarlyStopping,
        EMAShadowCallback,
        TrainingCurvesCallback,
    )

Importing a submodule directly still works and is unchanged.

Why the names load lazily
-------------------------
The submodules pull in heavy or optional dependencies:

- ``analyzer_callback`` imports ``dl_techniques.analyzer``;
- ``ema_shadow_callback`` imports ``models.vision.depth_anything``;
- ``spatial_loss_logger`` imports ``layers.regularization``;
- ``coco_map_callback`` raises ``ImportError`` at import time without
  ``pycocotools``.

Importing them all eagerly would make ``import dl_techniques.callbacks`` slow,
and would make the whole package unusable on any machine without
``pycocotools``. Each name is therefore resolved on first access (PEP 562
``__getattr__``) and cached in the module namespace.

Side effect of laziness: a callback decorated with ``register_dl_technique``
is registered when its module is first imported, not when this package is.
Keras never deserializes callbacks, so nothing depends on that timing.

Optional dependencies
---------------------
``COCOMAPCallback`` needs ``pycocotools``. It is importable from the package
root, but it is left out of ``__all__`` so that ``from dl_techniques.callbacks
import *`` does not fail where ``pycocotools`` is not installed.
"""

import importlib
from typing import TYPE_CHECKING, Any, Dict, List

# ---------------------------------------------------------------------
# Name -> submodule map. The single source of truth for the public API.
# Add a new callback here; `__all__` is derived from this table.
# ---------------------------------------------------------------------

_LAZY_ATTRIBUTES: Dict[str, str] = {
    # analyzer_callback
    "EpochAnalyzerCallback": "analyzer_callback",
    # anneal_harmonic_exponent
    "HarmonicExponentAnnealingCallback": "anneal_harmonic_exponent",
    # coco_map_callback (requires pycocotools)
    "COCOMAPCallback": "coco_map_callback",
    # coco_multitask_visualization
    "COCOMultiTaskPredictionGridCallback": "coco_multitask_visualization",
    "COCO_CLASS_NAMES": "coco_multitask_visualization",
    # consecutive_increase_early_stopping
    "ConsecutiveIncreaseEarlyStopping": "consecutive_increase_early_stopping",
    # convunext_bottleneck_monitor
    "ConvUnextBottleneckMonitorCallback": "convunext_bottleneck_monitor",
    # depth_visualization
    "DepthPredictionGridCallback": "depth_visualization",
    "DepthMetricsCurveCallback": "depth_visualization",
    # ema_shadow_callback (schedules re-exported from teacher_ema)
    "EMAShadowCallback": "ema_shadow_callback",
    "cosine_ema_schedule": "ema_shadow_callback",
    "linear_ema_schedule": "ema_shadow_callback",
    # jepa_visualization
    "LatentMaskOverlayCallback": "jepa_visualization",
    "PatchPredictionErrorCallback": "jepa_visualization",
    # noise_sigma_curriculum
    "NoiseSigmaCurriculumCallback": "noise_sigma_curriculum",
    # periodic_reassign
    "PeriodicReassignCallback": "periodic_reassign",
    # self_iterate_pool
    "SelfIteratePoolCallback": "self_iterate_pool",
    # spatial_loss_logger
    "SpatialLossLogger": "spatial_loss_logger",
    # temperature_annealing
    "TemperatureAnnealingCallback": "temperature_annealing",
    # training_curves
    "TrainingCurvesCallback": "training_curves",
}

#: Names whose module needs a dependency that is not always installed.
#: Importable by name, excluded from ``__all__`` so a star import stays safe.
_OPTIONAL_DEPENDENCY_ATTRIBUTES = frozenset({"COCOMAPCallback"})

__all__: List[str] = sorted(
    name for name in _LAZY_ATTRIBUTES if name not in _OPTIONAL_DEPENDENCY_ATTRIBUTES
)

# ---------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Import the submodule that defines ``name`` on first access.

    :param name: Attribute requested from the package.
    :type name: str
    :return: The requested class, function or constant.
    :rtype: Any
    :raises AttributeError: If ``name`` is not part of the public API.
    :raises ImportError: If the defining submodule cannot be imported, for
        example ``COCOMAPCallback`` without ``pycocotools``. The submodule's
        own message is kept.
    """
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    # Cache, so later lookups bypass __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """List module globals plus every lazily available name, optional ones included."""
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))


# ---------------------------------------------------------------------
# Static imports for type checkers and IDEs only. Never executed at runtime.
# ---------------------------------------------------------------------

if TYPE_CHECKING:  # pragma: no cover
    from .analyzer_callback import EpochAnalyzerCallback
    from .anneal_harmonic_exponent import HarmonicExponentAnnealingCallback
    from .coco_map_callback import COCOMAPCallback
    from .coco_multitask_visualization import (
        COCO_CLASS_NAMES,
        COCOMultiTaskPredictionGridCallback,
    )
    from .consecutive_increase_early_stopping import ConsecutiveIncreaseEarlyStopping
    from .convunext_bottleneck_monitor import ConvUnextBottleneckMonitorCallback
    from .depth_visualization import (
        DepthMetricsCurveCallback,
        DepthPredictionGridCallback,
    )
    from .ema_shadow_callback import (
        EMAShadowCallback,
        cosine_ema_schedule,
        linear_ema_schedule,
    )
    from .jepa_visualization import (
        LatentMaskOverlayCallback,
        PatchPredictionErrorCallback,
    )
    from .noise_sigma_curriculum import NoiseSigmaCurriculumCallback
    from .periodic_reassign import PeriodicReassignCallback
    from .self_iterate_pool import SelfIteratePoolCallback
    from .spatial_loss_logger import SpatialLossLogger
    from .temperature_annealing import TemperatureAnnealingCallback
    from .training_curves import TrainingCurvesCallback