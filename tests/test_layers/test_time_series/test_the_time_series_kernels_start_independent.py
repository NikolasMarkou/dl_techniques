"""No two kernels of one time_series layer start as the same random sample.

One shared ``Initializer`` instance replays one sample at every site it is
handed to (``dl_techniques.initializers.clone``). Measured at HEAD before the
fix: N-BEATS ``dense2 == dense3 == dense4`` and ``theta_backcast ==
theta_forecast`` bit-for-bit; ``TemporalFusionLayer`` started
``attention_generator == lag_projector`` and ``gate_generator ==
context_forecaster``; ``ForecastabilityGate``'s gate kernel was its hidden
kernel's prefix (Pearson r = 1.0). Each site now gets ``clone_initializer``.

Two kernels alias when the shorter flattened one equals, up to the fan-based
scale, the longer one's prefix: a same-shape pair is then bit-identical, a
different-shape pair has prefix correlation 1.0. The bound 0.9 sits far above
what two independent draws of these sizes give and far below the 1.0 replay.
"""

import itertools
from typing import Callable, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.adaptive_lag_attention import AdaptiveLagAttentionLayer
from dl_techniques.layers.time_series.forecasting_layers import ForecastabilityGate
from dl_techniques.layers.time_series.nbeats_blocks import GenericBlock, SeasonalityBlock, TrendBlock
from dl_techniques.layers.time_series.nbeatsx_blocks import ExogenousBlock
from dl_techniques.layers.time_series.temporal_fusion import TemporalFusionLayer

PREFIX_CORRELATION_BOUND = 0.9


def _ones(*shape):
    return np.ones(shape, dtype="float32")


def _kernels(layer: keras.layers.Layer) -> Dict[str, np.ndarray]:
    return {
        v.path: keras.ops.convert_to_numpy(v).ravel()
        for v in layer.trainable_weights
        if v.path.endswith("kernel") and int(np.prod(v.shape)) > 2
    }


def _nbeats(cls):
    def build():
        block = cls(units=16, thetas_dim=4, backcast_length=12, forecast_length=6)
        block(_ones(2, 12))
        return block
    return build


def _exogenous():
    block = ExogenousBlock(exogenous_dim=3, tcn_filters=8, use_tcn=False, units=16,
                           backcast_length=12, forecast_length=6)
    block(_ones(2, 12), exogenous_inputs=(_ones(2, 12, 3), _ones(2, 6, 3)))
    return block


def _adaptive_lag():
    layer = AdaptiveLagAttentionLayer(num_lags=8)
    layer([_ones(2, 8), _ones(2, 8)])
    return layer


def _temporal_fusion():
    layer = TemporalFusionLayer(output_dim=1, num_lags=4, project_lags=True)
    layer([_ones(2, 4), _ones(2, 4)])
    return layer


def _gate():
    gate = ForecastabilityGate(hidden_units=8, forecast_length=6)
    x = _ones(2, 6, 4)
    gate(x, x, x)
    return gate


BUILDERS: Dict[str, Callable[[], keras.layers.Layer]] = {
    "trend": _nbeats(TrendBlock),
    "seasonality": _nbeats(SeasonalityBlock),
    "generic": _nbeats(GenericBlock),
    "exogenous": _exogenous,
    "adaptive_lag": _adaptive_lag,
    "temporal_fusion": _temporal_fusion,
    "forecastability_gate": _gate,
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_no_kernel_replays_another(name: str) -> None:
    kernels = _kernels(BUILDERS[name]())
    assert len(kernels) >= 2, f"{name}: expected at least two kernels, got {sorted(kernels)}"

    aliased = []
    for (path_a, a), (path_b, b) in itertools.combinations(kernels.items(), 2):
        short, long_ = (a, b) if a.size <= b.size else (b, a)
        r = np.corrcoef(short, long_[:short.size])[0, 1]
        if r > PREFIX_CORRELATION_BOUND:
            aliased.append((path_a, path_b, round(float(r), 6)))
    assert not aliased, f"{name}: kernels start as one replayed sample: {aliased}"
