"""``ExogenousBlock`` honours ``theta_regularizer`` and ``theta_initializer``.

The block used to replace the theta heads that ``NBeatsBlock`` built with two
bare ``Dense`` layers. Those carried neither knob, so the NBEATSx
``theta_regularizer`` was accepted, serialized and never charged on an
exogenous stack: ``len(block.losses) == 0`` under ``L2(1.0)``.
"""

import numpy as np
import keras
import pytest

from dl_techniques.layers.time_series.nbeatsx_blocks import ExogenousBlock


def _block(use_tcn: bool, **kwargs) -> ExogenousBlock:
    block = ExogenousBlock(
        exogenous_dim=3, tcn_filters=8, use_tcn=use_tcn, units=16,
        backcast_length=12, forecast_length=6, **kwargs,
    )
    residual = np.ones((2, 12), dtype="float32")
    exog = (np.ones((2, 12, 3), dtype="float32"), np.ones((2, 6, 3), dtype="float32"))
    block(residual, exogenous_inputs=exog)
    return block


@pytest.mark.parametrize("use_tcn", [True, False])
def test_the_theta_regularizer_is_charged(use_tcn: bool) -> None:
    block = _block(use_tcn, theta_regularizer=keras.regularizers.L2(1.0))
    assert len(block.losses) == 2


@pytest.mark.parametrize("use_tcn", [True, False])
def test_the_theta_initializer_reaches_both_heads(use_tcn: bool) -> None:
    block = _block(use_tcn, theta_initializer="ones")
    for head in (block.theta_backcast, block.theta_forecast):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(head.kernel), 1.0)
