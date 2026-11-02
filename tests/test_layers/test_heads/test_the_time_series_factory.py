"""``create_time_series_head`` builds the head its task type names, and refuses the rest.

Each member of ``TimeSeriesTaskType`` reaches its own class through the one
table, builds by member and by string, runs a forward pass with the documented
shape, and round-trips through ``.keras``. An unknown task and an undeclared
keyword raise.
"""

import os
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers import heads as heads_package
from dl_techniques.layers.heads import task_types as task_types_module
from dl_techniques.layers.heads import time_series
from dl_techniques.layers.heads.time_series import (
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
    QuantileHead,
    QuantileSequenceHead,
    TimeSeriesTaskType,
    TimestralQuantileHead,
    create_time_series_head,
)
from dl_techniques.layers.heads.time_series.dispatch import (
    TIME_SERIES_HEAD_DISPATCH,
    get_head_class,
)

# (task, head class, constructor kwargs, input shape without batch, output shapes without batch)
_CASES: List[Tuple[TimeSeriesTaskType, type, Dict[str, Any], Tuple[int, ...], List[Tuple[int, ...]]]] = [
    (TimeSeriesTaskType.QUANTILE_FORECAST, QuantileHead,
     {"num_quantiles": 3, "output_length": 24, "enforce_monotonicity": True},
     (16,), [(24, 3)]),
    (TimeSeriesTaskType.QUANTILE_SEQUENCE, QuantileSequenceHead,
     {"num_quantiles": 3}, (12, 16), [(12, 3)]),
    (TimeSeriesTaskType.GAUSSIAN_LIKELIHOOD, GaussianLikelihoodHead,
     {"units": 2}, (12, 16), [(12, 2), (12, 2)]),
    (TimeSeriesTaskType.NEGATIVE_BINOMIAL_LIKELIHOOD, NegativeBinomialLikelihoodHead,
     {"units": 2}, (12, 16), [(12, 2), (12, 2)]),
    (TimeSeriesTaskType.TIMESTRAL_QUANTILE, TimestralQuantileHead,
     {"d_model": 16, "patch_size": 4, "quantile_levels": (0.1, 0.5, 0.9), "width_factor": 2.0},
     (6, 16), [(24, 3)]),
]
_IDS = [case[0].value for case in _CASES]


def _as_list(outputs: Any) -> List[Any]:
    return list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]


def test_every_task_type_has_exactly_one_row() -> None:
    assert set(TIME_SERIES_HEAD_DISPATCH) == set(TimeSeriesTaskType)


def test_the_table_holds_exactly_the_five_exported_heads() -> None:
    assert set(TIME_SERIES_HEAD_DISPATCH.values()) == {
        QuantileHead, QuantileSequenceHead, GaussianLikelihoodHead,
        NegativeBinomialLikelihoodHead, TimestralQuantileHead,
    }
    assert len(TIME_SERIES_HEAD_DISPATCH) == 5


def test_the_surface_is_exported_at_every_level() -> None:
    for name in ("TimeSeriesTaskType", "create_time_series_head"):
        assert name in time_series.__all__
        assert name in heads_package.__all__
    assert task_types_module.TimeSeriesTaskType is TimeSeriesTaskType


@pytest.mark.parametrize("task,head_class,kwargs,_input,_output", _CASES, ids=_IDS)
def test_a_member_and_its_string_build_the_same_class(
        task, head_class, kwargs, _input, _output) -> None:
    assert type(create_time_series_head(task, **kwargs)) is head_class
    assert type(create_time_series_head(task.value, **kwargs)) is head_class


def test_a_task_string_is_stripped_and_lowercased() -> None:
    assert get_head_class(" Quantile_Sequence ") is QuantileSequenceHead


@pytest.mark.parametrize("task,_head,kwargs,input_shape,output_shapes", _CASES, ids=_IDS)
def test_the_head_returns_its_documented_shapes(
        task, _head, kwargs, input_shape, output_shapes) -> None:
    head = create_time_series_head(task, **kwargs)
    inputs = np.random.default_rng(0).normal(size=(2, *input_shape)).astype("float32")
    outputs = _as_list(head(inputs))
    assert [tuple(o.shape) for o in outputs] == [(2, *s) for s in output_shapes]


@pytest.mark.parametrize("task,_head,kwargs,input_shape,_output", _CASES, ids=_IDS)
def test_the_head_survives_a_keras_round_trip(
        task, _head, kwargs, input_shape, _output, tmp_path) -> None:
    inputs = keras.Input(shape=input_shape)
    outputs = create_time_series_head(task, **kwargs)(inputs)
    model = keras.Model(inputs, outputs)

    path = os.path.join(tmp_path, "head.keras")
    model.save(path)
    loaded = keras.models.load_model(path)

    batch = np.random.default_rng(1).normal(size=(2, *input_shape)).astype("float32")
    for before, after in zip(_as_list(model(batch, training=False)),
                             _as_list(loaded(batch, training=False))):
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(before),
            keras.ops.convert_to_numpy(after),
            atol=1e-6, rtol=0,
        )


def test_an_unknown_task_string_raises() -> None:
    with pytest.raises(ValueError, match="Invalid task type"):
        create_time_series_head("quantile")


def test_a_value_that_is_no_task_type_raises() -> None:
    with pytest.raises(ValueError):
        create_time_series_head(3)


def test_an_undeclared_keyword_raises() -> None:
    with pytest.raises(ValueError):
        create_time_series_head("gaussian_likelihood", units=1, num_quantiles=3)


def test_a_missing_required_argument_raises() -> None:
    with pytest.raises(TypeError):
        create_time_series_head("quantile_forecast", num_quantiles=3)
