"""The two analyzer functions must report the value the live loss computes.

Both analyzers used to clip probabilities without renormalizing (the `call` of
each loss does both) and crashed on the plain-list inputs their own docstring
examples pass.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.decoupled_information_loss import (
    DecoupledInformationLoss,
    analyze_decoupled_information_loss,
)
from dl_techniques.losses.focal_uncertainty_loss import (
    FocalUncertaintyLoss,
    analyze_focal_uncertainty_loss,
)

# ---------------------------------------------------------------------

_Y_TRUE = np.eye(3, dtype="float32")[[0, 1, 2, 1]]

# Rows deliberately sum to 0.5, so skipping the renormalization is visible.
_HALF_MASS_ROWS = np.array(
    [[0.30, 0.15, 0.05], [0.05, 0.35, 0.10], [0.10, 0.10, 0.30], [0.20, 0.20, 0.10]],
    dtype="float32",
)
_LOGITS = np.array(
    [[2.0, 1.0, 0.1], [0.5, 2.1, 0.3], [0.2, 0.8, 1.9], [0.1, 1.2, 0.4]],
    dtype="float32",
)

_CASES = [
    (
        DecoupledInformationLoss,
        analyze_decoupled_information_loss,
        dict(uncertainty_weight=0.2, diversity_weight=0.05),
    ),
    (
        FocalUncertaintyLoss,
        analyze_focal_uncertainty_loss,
        dict(uncertainty_weight=0.2),
    ),
]


@pytest.mark.parametrize("loss_cls, analyzer, kwargs", _CASES)
def test_analyzer_total_equals_the_live_loss_on_unnormalized_rows(loss_cls, analyzer, kwargs) -> None:
    # DECISION plan-2026-11-11T131349-13b28b37/D-029: on rows summing to 0.5 the
    # analyzer total sat 0.228 (decoupled) away from float(loss(...)); the
    # renormalization after the clip is what closes the gap.
    loss_fn = loss_cls(from_logits=False, **kwargs)

    live = float(keras.ops.convert_to_numpy(loss_fn(_Y_TRUE, _HALF_MASS_ROWS)))
    reported = analyzer(loss_fn, _Y_TRUE, _HALF_MASS_ROWS)["total_loss"]

    assert reported == pytest.approx(live, abs=1e-5)


@pytest.mark.parametrize("loss_cls, analyzer, kwargs", _CASES)
def test_analyzer_total_equals_the_live_loss_on_logits(loss_cls, analyzer, kwargs) -> None:
    loss_fn = loss_cls(from_logits=True, **kwargs)

    live = float(keras.ops.convert_to_numpy(loss_fn(_Y_TRUE, _LOGITS)))
    reported = analyzer(loss_fn, _Y_TRUE, _LOGITS)["total_loss"]

    assert reported == pytest.approx(live, abs=1e-5)


@pytest.mark.parametrize("from_logits", [True, False])
@pytest.mark.parametrize("loss_cls, analyzer, kwargs", _CASES)
def test_analyzer_accepts_plain_lists(loss_cls, analyzer, kwargs, from_logits) -> None:
    """The docstring examples pass nested lists; `y_pred.dtype` used to raise."""
    loss_fn = loss_cls(from_logits=from_logits, **kwargs)
    source = _LOGITS if from_logits else _HALF_MASS_ROWS

    from_lists = analyzer(loss_fn, _Y_TRUE.tolist(), source.tolist())
    from_arrays = analyzer(loss_fn, _Y_TRUE, source)

    assert from_lists["total_loss"] == pytest.approx(from_arrays["total_loss"], abs=1e-6)
