"""RED proof: a trend-only N-BEATS must not predict exactly zero on the graph path.

``TrendBlock`` wrote its polynomial basis with ``add_weight(initializer='zeros')`` +
``.assign()`` inside ``build()``. Keras 3 runs a symbolic build pass inside a
``StatelessScope`` whenever a sublayer is first reached from a parent's ``call()``,
and that scope records the assign and discards it. ``NBeatsNet.predict()`` /
``fit()`` take exactly that path, so the basis was all zeros and, since
``backcast = theta @ basis_matrix``, the whole trend stack emitted exactly 0.0:
the forecast was identically zero and every block handed the NEXT block a residual
bit-identical to its own input.

The path dependence is why this survived the suite: calling the model EAGERLY
(``model(x)``) builds the blocks outside the stateless scope and produced a correct
basis, so a test written that way -- or one that calls ``block.build(...)`` directly --
cannot see the defect. Both assertions below go through ``predict()``.
"""

import numpy as np

from dl_techniques.models.time_series.nbeats.nbeats import NBeatsNet

BACKCAST_LENGTH = 24
FORECAST_LENGTH = 6


def _model() -> NBeatsNet:
    return NBeatsNet(
        backcast_length=BACKCAST_LENGTH,
        forecast_length=FORECAST_LENGTH,
        stack_types=["trend"],
        nb_blocks_per_stack=2,
        thetas_dim=[3],
        hidden_layer_units=16,
        use_normalization=False,
    )


class TestTrendStackOnTheGraphPath:
    def test_forecast_is_not_identically_zero(self) -> None:
        x = np.random.RandomState(0).randn(4, BACKCAST_LENGTH, 1).astype("float32") * 3.0
        forecast, _ = _model().predict(x, verbose=0)
        assert float(np.abs(forecast).max()) > 1e-6, (
            "trend-only N-BEATS predicts exactly zero through predict(): the polynomial "
            "basis did not survive the symbolic build pass"
        )

    def test_final_residual_is_not_the_unmodified_input(self) -> None:
        x = np.random.RandomState(1).randn(4, BACKCAST_LENGTH, 1).astype("float32") * 3.0
        _, residual = _model().predict(x, verbose=0)
        assert not np.array_equal(
            residual.reshape(4, BACKCAST_LENGTH), x.reshape(4, BACKCAST_LENGTH)
        ), "the doubly-residual stack explained nothing: residual is bit-identical to input"

    def test_basis_matrix_is_populated_after_the_graph_build(self) -> None:
        model = _model()
        model.predict(
            np.zeros((2, BACKCAST_LENGTH, 1), dtype="float32"), verbose=0
        )
        basis = np.asarray(model.blocks[0][0].backcast_basis_matrix)
        np.testing.assert_allclose(basis[0], np.ones(BACKCAST_LENGTH), atol=1e-6)
