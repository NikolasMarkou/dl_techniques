"""RED proof: N-BEATS basis matrices must survive Keras 3's stateless build pass.

Keras 3 runs a symbolic build pass inside a ``StatelessScope`` whenever a sublayer is
first reached from a PARENT layer's ``call()`` -- which is exactly what
``NBeatsNet.call`` does under ``predict()``/``fit()``. That scope RECORDS a
``.assign()`` and then DISCARDS it, so ``TrendBlock`` and ``SeasonalityBlock``, whose
basis matrices were written with ``add_weight(initializer='zeros')`` + ``.assign()``
in ``build()``, carried an all-zero basis in every real model. Since
``backcast = theta @ basis_matrix``, the basis IS the block's entire output map, so
both heads emitted exactly zero while the model still trained and still reported a
loss.

**Every test here builds the layer through a parent layer's ``call()``.** A test that
calls ``layer.build(...)`` directly is precisely the test that missed this defect: the
direct path never enters the stateless scope and always looked correct
(``tests/test_layers/test_time_series/test_nbeats_blocks.py`` builds that way).

Assertions compare the table against its CLOSED FORM, not against a shape and not
against a bare "is non-zero" -- individual basis entries are legitimately 0.0 (a
Fourier sine row starts at exactly 0.0), so only a value comparison discriminates.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.time_series.nbeats_blocks import (
    SeasonalityBlock,
    TrendBlock,
)


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call()`` reaches the child, forcing the stateless build.

    :param child: The layer under test; it is built by being called from here.
    :type child: keras.layers.Layer
    """

    def __init__(self, child: keras.layers.Layer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        out = self.child(inputs)
        return out[0] if isinstance(out, (list, tuple)) else out


def _build_through_parent(child: keras.layers.Layer, feature_dim: int) -> keras.layers.Layer:
    """Build ``child`` the way a real model does: from inside a parent's ``call()``."""
    parent = _Parent(child)
    parent(keras.Input(shape=(feature_dim,)))
    assert child.built, "child was not built through the parent's call()"
    return child


BACKCAST_LENGTH = 12
FORECAST_LENGTH = 6
TOTAL_LENGTH = BACKCAST_LENGTH + FORECAST_LENGTH


def _normalized_time() -> np.ndarray:
    """Closed form of the trend block's normalized time axis over [-1, 1)."""
    idx = np.arange(TOTAL_LENGTH, dtype=np.float64)
    return 2.0 * (idx - BACKCAST_LENGTH) / TOTAL_LENGTH


class TestTrendBlockPolynomialBasis:
    """The polynomial basis must equal powers of normalized time after a parent build."""

    @pytest.fixture()
    def block(self) -> TrendBlock:
        return _build_through_parent(
            TrendBlock(
                units=8,
                thetas_dim=3,
                backcast_length=BACKCAST_LENGTH,
                forecast_length=FORECAST_LENGTH,
                normalize_basis=False,
            ),
            BACKCAST_LENGTH,
        )

    def test_backcast_basis_equals_powers_of_normalized_time(self, block: TrendBlock) -> None:
        t = _normalized_time()[:BACKCAST_LENGTH]
        expected = np.stack([np.ones_like(t), t, t ** 2]).astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(block.backcast_basis_matrix), expected, atol=1e-6
        )

    def test_forecast_basis_equals_powers_of_normalized_time(self, block: TrendBlock) -> None:
        t = _normalized_time()[BACKCAST_LENGTH:]
        expected = np.stack([np.ones_like(t), t, t ** 2]).astype(np.float32)
        np.testing.assert_allclose(
            np.asarray(block.forecast_basis_matrix), expected, atol=1e-6
        )


class TestSeasonalityBlockFourierBasis:
    """The Fourier basis must equal cos/sin rows at harmonic frequencies."""

    @pytest.fixture()
    def block(self) -> SeasonalityBlock:
        return _build_through_parent(
            SeasonalityBlock(
                units=8,
                thetas_dim=4,
                backcast_length=BACKCAST_LENGTH,
                forecast_length=FORECAST_LENGTH,
                normalize_basis=False,
            ),
            BACKCAST_LENGTH,
        )

    @staticmethod
    def _expected(indices: np.ndarray) -> np.ndarray:
        rows = []
        for harmonic in (1, 2):
            freq = 2.0 * np.pi * harmonic / TOTAL_LENGTH
            rows.append(np.cos(freq * indices))
            rows.append(np.sin(freq * indices))
        return np.stack(rows).astype(np.float32)

    def test_backcast_basis_equals_the_fourier_rows(self, block: SeasonalityBlock) -> None:
        indices = np.arange(BACKCAST_LENGTH, dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(block.backcast_basis_matrix), self._expected(indices), atol=1e-6
        )

    def test_forecast_basis_equals_the_fourier_rows(self, block: SeasonalityBlock) -> None:
        indices = np.arange(
            BACKCAST_LENGTH, BACKCAST_LENGTH + FORECAST_LENGTH, dtype=np.float64
        )
        np.testing.assert_allclose(
            np.asarray(block.forecast_basis_matrix), self._expected(indices), atol=1e-6
        )


class TestTrendBlockIsNotAConstantZeroMap:
    """The end-to-end assertion a user would have noticed, and that was missing.

    With an all-zero basis the block emits exactly 0.0 from both heads: the forecast
    contributes nothing and the doubly-residual stack degenerates because the residual
    ``inputs - backcast`` is bit-identical to ``inputs``.
    """

    def test_forecast_and_backcast_are_not_identically_zero_through_a_parent(self) -> None:
        block = _build_through_parent(
            TrendBlock(
                units=8,
                thetas_dim=3,
                backcast_length=BACKCAST_LENGTH,
                forecast_length=FORECAST_LENGTH,
            ),
            BACKCAST_LENGTH,
        )
        x = ops.convert_to_tensor(
            np.random.RandomState(0).randn(4, BACKCAST_LENGTH).astype("float32") * 3.0
        )
        backcast, forecast = block(x)

        assert float(ops.max(ops.abs(forecast))) > 1e-6, (
            "trend forecast is identically zero: the polynomial basis did not survive "
            "the stateless build pass"
        )
        assert float(ops.max(ops.abs(backcast))) > 1e-6, (
            "trend backcast is identically zero, so the doubly-residual stack is a no-op"
        )

    def test_residual_is_not_bit_identical_to_the_input(self) -> None:
        block = _build_through_parent(
            TrendBlock(
                units=8,
                thetas_dim=3,
                backcast_length=BACKCAST_LENGTH,
                forecast_length=FORECAST_LENGTH,
            ),
            BACKCAST_LENGTH,
        )
        x = np.random.RandomState(1).randn(4, BACKCAST_LENGTH).astype("float32") * 3.0
        backcast, _ = block(ops.convert_to_tensor(x))
        residual = x - np.asarray(backcast)
        assert not np.array_equal(residual, x), (
            "the residual handed to the next block is the unmodified input: this stack "
            "explains nothing"
        )
