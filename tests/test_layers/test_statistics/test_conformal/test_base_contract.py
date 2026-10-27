"""One contract, run over every layer of the ``conformal`` package.

The three layers (:class:`ConformalInterval`, :class:`ConformalQuantileHead`,
:class:`ConformalPredictionSet`) share :class:`ConformalLayer`: one frozen scalar weight
``conformal_q_hat``, ``set_q_hat`` and ``calibrate(scores, alpha)`` to fill it, and a
``get_config`` that stores the CURRENT value as ``q_hat_init``. This module pins that
shared shape once, for all of them, so a layer cannot drift from its siblings; the
layer-specific behaviour is in ``test_interval.py``, ``test_quantile_head.py`` and
``test_prediction_set.py``.
"""

from dataclasses import dataclass
from typing import Callable

import keras
import numpy as np
import pytest

import dl_techniques.layers.statistics.conformal as package
from dl_techniques.layers.statistics.conformal import (
    ConformalInterval,
    ConformalLayer,
    ConformalPredictionSet,
    ConformalQuantileHead,
)
from dl_techniques.utils.conformal_denoiser_intervals import conformal_quantile

CAL = 0.37


@dataclass(frozen=True)
class Subject:
    name: str
    cls: type
    module: str
    make: Callable[..., ConformalLayer]
    frozen: bool
    #: what an infinite fitted value is stored as (the set layer caps it)
    unbounded: float


SUBJECTS = [
    Subject("interval", ConformalInterval, "interval",
            lambda **kw: ConformalInterval(**kw), True, float("inf")),
    Subject("quantile_head", ConformalQuantileHead, "quantile_head",
            lambda **kw: ConformalQuantileHead(forecast_length=2, output_dim=2, **kw),
            False, float("inf")),
    Subject("prediction_set", ConformalPredictionSet, "prediction_set",
            lambda **kw: ConformalPredictionSet(**kw), True, 2.0),
]
IDS = [s.name for s in SUBJECTS]
X = np.random.default_rng(0).standard_normal((6, 8)).astype("float32")


def _built(subject: Subject, **kw) -> ConformalLayer:
    layer = subject.make(**kw)
    layer(X)
    return layer


def _q(layer: ConformalLayer) -> float:
    return float(keras.ops.convert_to_numpy(layer.q_hat))


def _outputs(model: keras.Model) -> list:
    out = model(X, training=False)
    out = out if isinstance(out, (list, tuple)) else [out]
    return [keras.ops.convert_to_numpy(t) for t in out]


def test_the_subjects_are_exactly_the_layers_of_the_package():
    found = set(ConformalLayer.__subclasses__())
    assert found == {s.cls for s in SUBJECTS} and len(SUBJECTS) == 3
    assert set(package.__all__) >= {c.__name__ for c in found} | {"ConformalLayer"}


@pytest.mark.parametrize("subject", SUBJECTS, ids=IDS)
class TestSharedShape:

    def test_registration_under_the_package_key(self, subject, registration_contract):
        registration_contract(subject.cls)
        assert keras.saving.get_registered_name(subject.cls) == (
            f"dl_techniques.layers.statistics.conformal.{subject.module}>{subject.cls.__name__}"
        )
        assert issubclass(subject.cls, ConformalLayer)

    def test_the_weight_is_one_frozen_scalar_in_the_variable_dtype(self, subject, dtype_policy):
        layer = _built(subject)
        assert [w.path.split("/")[-1] for w in layer.non_trainable_weights] == ["conformal_q_hat"]
        assert layer.q_hat.shape == () and layer.q_hat.trainable is False
        assert layer.q_hat.dtype == ("float64" if dtype_policy == "float64" else "float32")
        assert all(v is not layer.q_hat for v in layer.trainable_variables)
        assert (layer.trainable is False) == subject.frozen

    def test_an_unbuilt_layer_has_no_weight_and_refuses_both_fits(self, subject):
        layer = subject.make()
        assert layer.q_hat is None
        with pytest.raises(ValueError, match="is not built"):
            layer.set_q_hat(0.1)
        with pytest.raises(ValueError, match="is not built"):
            layer.calibrate(np.array([0.1, 0.2]), alpha=0.1)

    def test_set_q_hat_assigns_and_refuses_negative_and_nan(self, subject):
        layer = _built(subject)
        layer.set_q_hat(CAL)
        assert _q(layer) == pytest.approx(CAL, abs=1e-6)
        for bad in (-0.2, float("nan")):
            with pytest.raises(ValueError, match="q_hat must be non-negative"):
                layer.set_q_hat(bad)
        assert _q(layer) == pytest.approx(CAL, abs=1e-6)  # a refusal leaves it unchanged

    def test_an_infinite_value_is_stored_as_the_subjects_unbounded_value(self, subject):
        layer = _built(subject)
        layer.set_q_hat(float("inf"))
        assert _q(layer) == subject.unbounded

    def test_the_constructor_refuses_a_bad_q_hat_init(self, subject):
        for bad in (-0.1, float("nan")):
            with pytest.raises(ValueError, match="q_hat_init must be non-negative"):
                subject.make(q_hat_init=bad)

    def test_calibrate_is_the_exact_order_statistic(self, subject):
        layer = _built(subject)
        scores = np.random.default_rng(1).uniform(0.0, 0.9, size=99)
        layer.calibrate(scores, alpha=0.1)
        assert _q(layer) == pytest.approx(conformal_quantile(scores, 0.1), abs=1e-6)
        # n = 9, alpha = 0.1: k = ceil(10 * 0.9) = 9, the LARGEST score
        nine = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.8, 0.4, 0.7, 0.6])
        layer.calibrate(nine, alpha=0.1)
        assert _q(layer) == pytest.approx(0.9, abs=1e-6)

    def test_too_few_scores_give_the_unbounded_value(self, subject):
        layer = _built(subject)
        layer.calibrate(np.array([0.2, 0.4, 0.1]), alpha=0.1)  # k = 4 > n = 3
        assert _q(layer) == subject.unbounded

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
    def test_alpha_outside_the_open_unit_interval_is_refused(self, subject, alpha):
        with pytest.raises(ValueError, match="alpha must be in"):
            _built(subject).calibrate(np.array([0.1, 0.2]), alpha=alpha)

    def test_the_config_stores_the_current_calibration(self, subject):
        layer = _built(subject)
        assert layer.get_config()["q_hat_init"] == layer.q_hat_init  # before: construction value
        layer.set_q_hat(CAL)
        cfg = layer.get_config()
        assert cfg["q_hat_init"] == pytest.approx(CAL, abs=1e-6)
        clone = type(layer).from_config(cfg)
        clone(X)
        assert _q(clone) == pytest.approx(CAL, abs=1e-6)

    def test_keras_round_trip_keeps_a_calibrated_value_on_values(self, subject, tmp_path):
        inp = keras.Input(X.shape[1:])
        layer = subject.make(name="layer_under_test")
        out = layer(inp)
        model = keras.Model(inp, list(out) if isinstance(out, (list, tuple)) else out)
        layer.set_q_hat(CAL)
        before = _outputs(model)
        path = tmp_path / "m.keras"
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={subject.cls.__name__: subject.cls})
        rel = loaded.get_layer("layer_under_test")
        for a, b in zip(layer.weights, rel.weights):  # weights first, before any call
            assert a.path == b.path
            np.testing.assert_array_equal(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))
        assert _q(rel) == pytest.approx(CAL, abs=1e-6)
        for a, b in zip(before, _outputs(loaded)):
            np.testing.assert_array_equal(a, b)
