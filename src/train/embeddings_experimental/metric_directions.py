"""The ONE producer of "is higher better?" for this study.

Direction was previously encoded in two places -- implicitly in
``sweep.collect_results`` (which reduced every metric with ``min``) and
explicitly in ``report.HEADLINE_METRICS`` -- and they disagreed. The result was
that ``mlm_val_accuracy_best`` held the WORST accuracy while the report
described it as a maximize metric. A second place to encode direction is a
second place for it to drift, so both readers now come here.
"""

from typing import Dict, Optional

# ---------------------------------------------------------------------

__all__ = ["METRIC_DIRECTIONS", "direction_of"]

#: Bare metric name -- as it appears in a Keras history dict -- to direction.
METRIC_DIRECTIONS: Dict[str, str] = {
    "loss": "min",
    "val_loss": "min",
    "accuracy": "max",
    "val_accuracy": "max",
}


def direction_of(metric: str) -> Optional[str]:
    """Return ``"min"``, ``"max"`` or ``None`` for an unknown metric.

    ``None`` rather than a guessed default: a new Keras metric appearing in a
    history must not silently receive the wrong reduction, which is exactly how
    the accuracy bug survived. Callers are expected to skip the ``_best``
    reduction and log when they get ``None``.

    :param metric: Bare metric name, e.g. ``"val_accuracy"``.
    :type metric: str
    :return: The direction, or ``None`` if the metric is not registered.
    :rtype: str | None
    """
    return METRIC_DIRECTIONS.get(metric)
