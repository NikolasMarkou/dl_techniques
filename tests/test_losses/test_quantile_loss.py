"""`QuantileLoss` configuration: quantiles must survive a config round trip exactly."""

import json

import keras
import numpy as np
import pytest

from dl_techniques.losses.quantile_loss import QuantileLoss

# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "quantiles",
    [[0.1, 0.5, 0.9], [0.05, 0.95], [0.025, 0.25, 0.5, 0.75, 0.975]],
)
def test_get_config_emits_the_quantiles_as_given(quantiles) -> None:
    # DECISION plan-2026-11-11T131349-13b28b37/D-029: get_config read the
    # quantiles back out of a float32 tensor, so [0.1, 0.5, 0.9] came back as
    # [0.10000000149011612, 0.5, 0.8999999761581421]. Keep a Python float list.
    loss = QuantileLoss(quantiles=quantiles)

    assert loss.get_config()["quantiles"] == quantiles


def test_config_round_trip_is_a_fixed_point() -> None:
    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9], normalize=True)

    rebuilt = QuantileLoss.from_config(loss.get_config())

    assert rebuilt.get_config()["quantiles"] == [0.1, 0.5, 0.9]
    assert rebuilt.normalize is True
    json.dumps(loss.get_config())


def test_serialize_keras_object_round_trip_keeps_quantiles() -> None:
    loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(loss)
    )

    assert restored.get_config()["quantiles"] == [0.1, 0.5, 0.9]


def test_the_loss_value_is_unchanged_by_keeping_a_float_list() -> None:
    """The computation still runs on the float32 tensor: pinball oracle."""
    rng = np.random.RandomState(0)
    y_true = rng.randn(4, 6).astype("float32")
    y_pred = rng.randn(4, 6, 3).astype("float32")
    q = np.array([0.1, 0.5, 0.9], dtype="float32")

    err = y_true[..., None] - y_pred
    expected = np.maximum(q * err, (q - 1) * err).mean()

    got = float(keras.ops.convert_to_numpy(QuantileLoss(quantiles=[0.1, 0.5, 0.9])(y_true, y_pred)))

    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_quantiles_outside_the_unit_interval_still_raise() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        QuantileLoss(quantiles=[0.0, 0.5])
