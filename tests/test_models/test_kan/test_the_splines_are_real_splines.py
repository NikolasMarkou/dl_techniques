"""KAN's central claim: learnable B-spline edge functions, not a Dense layer.

Why this file exists
--------------------
Substituting `keras.layers.Dense` for `KANLinear` passes every test in
`test_kan/test_model.py`: the only thing tying the package to splines is an
`isinstance` count and a `grid_size` attribute echo. Shapes, config round trips,
gradient flow and "the model trains" are all properties a Dense layer has too.

What actually distinguishes a B-spline basis from any dense/global basis:

1. **Compact support.** Each basis function is non-zero on exactly
   ``spline_order + 1`` knot intervals and EXACTLY zero elsewhere. MEASURED
   2026-08-18 (grid_size=5, spline_order=3, grid_range=(-2, 2), so h = 0.8):
   basis 4 responds only on x in [-1.175, 1.975], a window of 3.15 against the
   predicted ``(3 + 1) * 0.8 = 3.2``; the boundary bases 0 and 7 run off the
   respective ends, as they must. A Dense layer's response to a coefficient is
   non-zero for every x.
2. **Partition of unity.** With every spline coefficient set to 1, the summed
   basis is identically 1.0 inside the grid -- MEASURED min = max = 1.000000
   over x in [-2, 2]. An affine map gives a ramp, never a constant.

Both claims are checked against a synthetic GLOBAL response in the same test
(`test_the_locality_contract_rejects_a_global_basis`), which is the
dead-component control: a probe that only ever sees the real layer cannot show
it would reject the Dense substitution.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.ffn.kan_linear import KANLinear


GRID_SIZE = 5
SPLINE_ORDER = 3
GRID_RANGE = (-2.0, 2.0)
KNOT_SPACING = (GRID_RANGE[1] - GRID_RANGE[0]) / GRID_SIZE  # 0.8
X = np.linspace(-3.0, 3.0, 241, dtype="float32").reshape(-1, 1)
INSIDE = (X.ravel() >= GRID_RANGE[0]) & (X.ravel() <= GRID_RANGE[1])


def _spline_only_layer() -> KANLinear:
    """A built 1->1 layer with the BASE path switched off, so only splines act."""
    keras.utils.set_random_seed(0)
    layer = KANLinear(
        features=1,
        grid_size=GRID_SIZE,
        spline_order=SPLINE_ORDER,
        grid_range=GRID_RANGE,
    )
    layer(keras.ops.convert_to_tensor(X))
    layer.base_scaler.assign(keras.ops.zeros_like(layer.base_scaler))
    layer.spline_scaler.assign(keras.ops.ones_like(layer.spline_scaler))
    return layer


def _response(layer: KANLinear, coefficients: np.ndarray) -> np.ndarray:
    layer.spline_weight.assign(keras.ops.convert_to_tensor(coefficients))
    out = layer(keras.ops.convert_to_tensor(X))
    return np.asarray(keras.ops.convert_to_numpy(out)).ravel()


def _assert_locally_supported(response: np.ndarray, *, name: str) -> float:
    """The contract: a single basis must be zero outside a bounded window.

    Returns the measured support width.
    """
    live = np.where(np.abs(response) > 1e-6)[0]
    assert live.size > 0, f"{name}: the basis is identically zero everywhere"
    width = float(X[live[-1], 0] - X[live[0], 0])
    expected = (SPLINE_ORDER + 1) * KNOT_SPACING  # 3.2
    assert width <= 1.2 * expected, (
        f"{name}: support spans {width:.3f} in x, more than "
        f"{1.2 * expected:.3f} = 1.2 * (spline_order + 1) * h. A basis with "
        f"unbounded support is not a B-spline -- a Dense layer scores "
        f"{float(X[-1, 0] - X[0, 0]):.3f} here."
    )
    # ...and it must be EXACTLY zero outside, not merely small.
    outside = np.ones_like(response, dtype=bool)
    outside[live[0]: live[-1] + 1] = False
    assert np.all(response[outside] == 0.0), (
        f"{name}: the response outside the support window is small but not "
        f"exactly zero (max {float(np.max(np.abs(response[outside]))):.3e})"
    )
    return width


class TestBSplineBasisIsCompactlySupported:
    def test_an_interior_basis_has_bounded_support(self):
        layer = _spline_only_layer()
        shape = tuple(layer.spline_weight.shape)
        assert shape == (1, 1, GRID_SIZE + SPLINE_ORDER), shape

        coefficients = np.zeros(shape, dtype="float32")
        coefficients[0, 0, 4] = 1.0
        width = _assert_locally_supported(
            _response(layer, coefficients), name="basis 4"
        )
        # Measured 3.15 against the predicted 3.20.
        assert width == pytest.approx(3.15, abs=0.1)

    def test_the_locality_contract_rejects_a_global_basis(self):
        """RED proof: the same contract applied to a Dense-like global response.

        `KANLinear` -> `Dense` is the substitution that passes the whole
        existing suite; a Dense unit's response to one coefficient is
        ``w * x``, non-zero for every x. Run the contract on exactly that.
        """
        dense_like = (2.0 * X.ravel()).astype("float32")
        with pytest.raises(AssertionError, match="support spans"):
            _assert_locally_supported(dense_like, name="dense unit")

        # A smooth global bump (Gaussian) is the subtler mutant: it LOOKS local
        # but never reaches exactly zero. The contract must reject it too.
        gaussian = np.exp(-(X.ravel() ** 2)).astype("float32")
        with pytest.raises(AssertionError):
            _assert_locally_supported(gaussian, name="gaussian")

    def test_each_basis_covers_a_different_window(self):
        """The bases must TILE the grid, not all sit on top of one another."""
        layer = _spline_only_layer()
        shape = tuple(layer.spline_weight.shape)
        centres = []
        for k in range(shape[-1]):
            coefficients = np.zeros(shape, dtype="float32")
            coefficients[0, 0, k] = 1.0
            response = _response(layer, coefficients)
            live = np.where(np.abs(response) > 1e-6)[0]
            centres.append(float(X[live, 0].mean()))
        assert centres == sorted(centres), (
            f"basis centres are not ordered along x: {np.round(centres, 3)}"
        )
        assert centres[-1] - centres[0] > (GRID_RANGE[1] - GRID_RANGE[0]) / 2, (
            f"all {shape[-1]} bases are bunched together: {np.round(centres, 3)}"
        )


class TestBSplineBasisIsAPartitionOfUnity:
    def test_the_summed_basis_is_identically_one_inside_the_grid(self):
        layer = _spline_only_layer()
        shape = tuple(layer.spline_weight.shape)
        response = _response(layer, np.ones(shape, dtype="float32"))
        # MEASURED: min = max = 1.000000 over [-2, 2].
        np.testing.assert_allclose(
            response[INSIDE], np.ones(int(INSIDE.sum()), dtype="float32"),
            rtol=0, atol=1e-5,
        )

    def test_an_affine_map_would_fail_that(self):
        """RED proof for the partition-of-unity claim."""
        affine = (0.5 * X.ravel() + 1.0).astype("float32")
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                affine[INSIDE], np.ones(int(INSIDE.sum()), dtype="float32"),
                rtol=0, atol=1e-5,
            )
