"""The closed-form fit of HKAN is the regression it claims to be.

Guards, in this order:

1. **scikit-learn parity, per block and per output, in float64.** Every block
   of ``HKANLayer.solve_closed_form`` is compared with
   ``sklearn.linear_model.Ridge(alpha=l2)`` (``l2 > 0``) or
   ``LinearRegression`` (``l2 == 0``; never ``Ridge(alpha=0)``, which returns
   garbage on duplicated centers) fitted on features this file computes from
   the documented formula with scipy closed forms. The connecting stage is
   compared with ``LinearRegression`` on scikit-learn's own block predictions.
   48 cells: six basis names, both intercept settings, ``l2`` 0 and 0.01,
   distinct and duplicated centers.
2. The same chain through ``HKAN.fit_closed_form`` on a two-layer model: the
   float64 train predictions, the per-layer RMSE, the weights the model ends
   up holding, and ``l2_mix``.
3. The float32 forward pass of a fitted model against the float64 fit.
4. Determinism under one seed, difference under another, ``seed=0`` being a
   seed; the result not depending on ``chunk_size``.
5. The block R^2 matrix and the per-input importance (paper equation 14)
   against ``sklearn.metrics.r2_score``.
6. ``centers="data"``: every layer draws from ITS OWN input.

NAMED DEVIATION (decisions.md D-024). In the eight cells ``sigmoid`` and
``softplus`` at ``l2 == 0`` (both intercept settings, distinct and duplicated
centers) the block coefficients are NOT asserted at the tolerance of the other
40 cells. An unregularized block of those two functions at slope 5 is
ill-conditioned: the centered design has an effective condition number of
4e6 to 3.4e8 on this fixture (at most 3.6e5 in every other zero-ridge cell),
and its coefficients reach 8.5e6. A change in the last bit of the features or
of the target then moves the solution by up to ``cond * eps = 7.5e-8``
relative. It is measured twice: the package writes the sigmoid as
``exp(-logaddexp(0, -z))`` and this file as ``scipy.special.expit``, and the
sigmoid cells differ by 5.2e-10 of the largest coefficient and 4.7e-10 in the
block predictions (7.0e-12 and 7.3e-13 in the regular cells); and an
arithmetically equivalent rewrite of the target centering moved the softplus
block predictions by 3.7e-9 (decisions.md D-025, mutation x). For those eight
cells the coefficients are asserted at a relative 1e-6, and the claim is
carried by the block predictions and by the minimum-norm property.
``test_the_named_deviation_cells_are_the_ill_conditioned_ones`` keeps the list
honest. No cell is dropped.

Tolerances were measured on exactly these fixtures (seed 11, 96 rows) with
``<scratchpad>/step2/measure_cells.py`` on 2026-09-30; each constant states
the measured worst case beside it. The float64 solve runs in numpy on the CPU
whatever ``CUDA_VISIBLE_DEVICES`` says, so those numbers have one regime; the
float32 tolerance is measured on the CPU and on GPU 1.
"""

import itertools
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from dl_techniques.models.general_purpose.hkan import HKAN, HKANLayer

from . import CLOSED_FORM, HIDDEN, N_IN, N_ROWS, NUM_BASIS, SLOPE, block_features, make_data

L2_POSITIVE = 0.01

# --- float64 tolerances, all with rtol=0 -----------------------------------
#
# A least-squares solution is only determined to about cond * eps, relative,
# where cond is the effective condition number of the (centered) design. Both
# tolerances of a class are about 12 times that bound for the worst cell of
# the class, measured on this fixture (`<scratchpad>/step2/scales.py`):
#   regular cells   worst cond 3.6e5 (tanh, l2 == 0)      cond * eps = 8.0e-11
#   deviation cells worst cond 3.4e8 (softplus, l2 == 0)  cond * eps = 7.5e-8
#
# Coefficients are compared at COEF_REL times max(1, largest |scikit-learn
# coefficient| of the cell): an absolute bound alone means nothing across
# cells whose coefficients run from 0.13 (identity) to 8.5e6 (softplus at
# l2 == 0), where one float64 ulp is already 2e-9.
#   measured worst relative difference, 40 regular cells: 7.04e-12
#   measured worst in the eight deviation cells: 5.22e-10
# Block predictions, connecting weights and layer outputs are O(1) numbers.
#   measured worst, regular cells: block predictions 7.3e-13, connecting
#   weights 9.7e-14, outputs 3.5e-13
#   measured worst, deviation cells: 4.7e-10, 4.4e-10, 3.1e-10
# The first version of these constants was 1e-10 / 1e-7, set from the measured
# differences alone; mutation x of D-025 (an equivalent rewrite that changes
# only rounding) then failed four softplus nodes at 3.7e-9. A tolerance taken
# from a bit-identical reading is a statement about this machine's rounding.
# What the constants must still see: ignoring l2, dropping the centering,
# solving the normal equations at l2 == 0 or solving in float32 move a
# coefficient by 1e-3 relative or more and a prediction by 3.6e-5 or more
# (RED proofs b, c, d, k in decisions.md D-025).
COEF_REL = 1e-9
COEF_REL_DEVIATION = 1e-6
VALUE_ATOL = 1e-9
VALUE_ATOL_DEVIATION = 1e-6
#: Effective condition number above which a zero-ridge cell is a deviation
#: cell: between the two measured groups (at most 3.6e5, at least 4.0e6).
ILL_CONDITIONED = 1e6

CELLS = [
    pytest.param(
        (basis, intercept, l2, duplicated),
        id=f"{basis}-{'intercept' if intercept else 'no_intercept'}-"
           f"l2_{l2:g}-{'duplicated' if duplicated else 'distinct'}",
    )
    for basis, intercept, l2, duplicated in itertools.product(
        CLOSED_FORM, (True, False), (0.0, L2_POSITIVE), (False, True))
]


def _is_deviation(basis: str, l2: float) -> bool:
    return basis in ("sigmoid", "softplus") and l2 == 0.0


def sklearn_layer(
        basis: str, slope: float, x: np.ndarray, y: np.ndarray, centers: np.ndarray,
        l2_block: float, l2_mix: float, block_intercept: bool, mix_intercept: bool,
) -> Dict[str, np.ndarray]:
    """One HKAN layer fitted by scikit-learn, block by block, output by output.

    :return: ``coef (n_out, n_in, m)``, ``block_bias (n_out, n_in)``,
        ``phi (N, n_out, n_in)``, ``mix (n_out, n_in)``, ``bias (n_out,)``,
        ``output (N, n_out)``, all float64.
    """
    n_out, n_in, m = centers.shape
    coef = np.empty((n_out, n_in, m))
    block_bias = np.empty((n_out, n_in))
    phi = np.empty((x.shape[0], n_out, n_in))
    mix = np.empty((n_out, n_in))
    bias = np.empty((n_out,))
    output = np.empty((x.shape[0], n_out))

    def regressor(l2: float, intercept: bool) -> Any:
        if l2 > 0:
            return Ridge(alpha=l2, fit_intercept=intercept)
        return LinearRegression(fit_intercept=intercept)

    for q in range(n_out):
        for p in range(n_in):
            features = block_features(basis, slope, x[:, p], centers[q, p])
            block = regressor(l2_block, block_intercept).fit(features, y)
            coef[q, p] = block.coef_
            block_bias[q, p] = float(block.intercept_)
            phi[:, q, p] = block.predict(features)
        connecting = regressor(l2_mix, mix_intercept).fit(phi[:, q, :], y)
        mix[q] = connecting.coef_
        bias[q] = float(connecting.intercept_)
        output[:, q] = connecting.predict(phi[:, q, :])
    return {"coef": coef, "block_bias": block_bias, "phi": phi,
            "mix": mix, "bias": bias, "output": output}


def cell_centers(duplicated: bool) -> np.ndarray:
    """Centers of one parity cell, ``(HIDDEN, N_IN, NUM_BASIS[0])``."""
    centers = np.random.default_rng(5).uniform(
        0.0, 1.0, size=(HIDDEN, N_IN, NUM_BASIS[0]))
    if duplicated:
        centers[..., 1] = centers[..., 0]
        centers[..., 5] = centers[..., 0]
    return centers


@pytest.fixture(scope="module")
def data() -> Tuple[np.ndarray, np.ndarray]:
    return make_data()


@pytest.fixture(scope="module", params=CELLS)
def cell(request, data) -> Dict[str, Any]:
    """One parity cell: the port's solve and scikit-learn's, on the same centers."""
    basis, intercept, l2, duplicated = request.param
    x, y = data
    centers = cell_centers(duplicated)
    layer = HKANLayer(
        units=HIDDEN, num_basis=NUM_BASIS[0], basis=basis, slope=SLOPE,
        use_block_bias=intercept, use_bias=intercept)
    # chunk_size=2 does not divide HIDDEN=5: the last chunk is a partial one.
    port = layer.solve_closed_form(x, y, l2, 0.0, centers, chunk_size=2)
    oracle = sklearn_layer(basis, SLOPE, x, y, centers, l2, 0.0, intercept, intercept)
    deviation = _is_deviation(basis, l2)
    return {
        "basis": basis, "intercept": intercept, "l2": l2, "duplicated": duplicated,
        "x": x, "y": y, "centers": centers, "port": port, "oracle": oracle,
        "coef_atol": (COEF_REL_DEVIATION if deviation else COEF_REL)
        * max(1.0, float(np.abs(oracle["coef"]).max())),
        "value_atol": VALUE_ATOL_DEVIATION if deviation else VALUE_ATOL,
    }


def port_block_predictions(cell: Dict[str, Any]) -> np.ndarray:
    """The port's coefficients applied to this file's features, ``(N, n_out, n_in)``."""
    port = cell["port"]
    phi = np.empty_like(cell["oracle"]["phi"])
    for q in range(HIDDEN):
        for p in range(N_IN):
            features = block_features(
                cell["basis"], SLOPE, cell["x"][:, p], cell["centers"][q, p])
            phi[:, q, p] = features @ port["coef"][q, p] + port["block_bias"][q, p]
    return phi


class TestSklearnParityPerBlock:
    """Expanding stage and connecting stage against scikit-learn, 48 cells."""

    def test_sklearn_block_coefficients(self, cell):
        np.testing.assert_allclose(
            cell["port"]["coef"], cell["oracle"]["coef"], rtol=0,
            atol=cell["coef_atol"],
            err_msg="block coefficients differ from scikit-learn's",
        )

    def test_sklearn_block_intercepts(self, cell):
        if not cell["intercept"]:
            np.testing.assert_array_equal(
                cell["port"]["block_bias"], 0.0,
                err_msg="a block intercept was fitted with use_block_bias=False")
            np.testing.assert_array_equal(cell["oracle"]["block_bias"], 0.0)
            return
        np.testing.assert_allclose(
            cell["port"]["block_bias"], cell["oracle"]["block_bias"], rtol=0,
            atol=cell["coef_atol"],
            err_msg="block intercepts differ from scikit-learn's",
        )

    def test_sklearn_block_predictions(self, cell):
        np.testing.assert_allclose(
            port_block_predictions(cell), cell["oracle"]["phi"], rtol=0,
            atol=cell["value_atol"],
            err_msg="block predictions differ from scikit-learn's",
        )

    def test_sklearn_minimum_norm(self, cell):
        """At ``l2 == 0`` the solution is the minimum-norm one.

        Two statements. The coefficient norm of every block equals
        scikit-learn's (whose ``LinearRegression`` returns the minimum-norm
        solution). And, where centers are duplicated, the duplicated columns
        are identical features, so the minimum-norm solution (and any ridge
        solution) gives them EQUAL coefficients: a solver that splits the
        weight unevenly between identical columns fits equally well and is not
        minimum-norm.
        """
        port, oracle = cell["port"]["coef"], cell["oracle"]["coef"]
        relative = COEF_REL_DEVIATION if _is_deviation(cell["basis"], cell["l2"]) else COEF_REL
        port_norm = np.linalg.norm(port, axis=-1)
        oracle_norm = np.linalg.norm(oracle, axis=-1)
        np.testing.assert_allclose(
            port_norm, oracle_norm, rtol=0,
            atol=relative * max(1.0, float(oracle_norm.max())),
            err_msg="the coefficient norm is not scikit-learn's minimum norm",
        )
        if cell["duplicated"]:
            for other in (1, 5):
                np.testing.assert_allclose(
                    port[..., other], port[..., 0], rtol=0, atol=cell["coef_atol"],
                    err_msg=f"duplicated centers 0 and {other} carry different "
                            "coefficients: not the minimum-norm solution",
                )
        if cell["basis"] == "identity" and cell["intercept"]:
            # Centered identity features are one column repeated m times.
            np.testing.assert_allclose(
                port, np.broadcast_to(port[..., :1], port.shape), rtol=0,
                atol=cell["coef_atol"],
                err_msg="an identity block's coefficients are not all equal",
            )

    def test_sklearn_connecting_stage(self, cell):
        port, oracle = cell["port"], cell["oracle"]
        np.testing.assert_allclose(
            port["mix"], oracle["mix"], rtol=0, atol=cell["value_atol"],
            err_msg="connecting weights differ from LinearRegression's")
        if cell["intercept"]:
            np.testing.assert_allclose(
                port["bias"], oracle["bias"], rtol=0, atol=cell["value_atol"],
                err_msg="connecting intercepts differ from LinearRegression's")
        else:
            np.testing.assert_array_equal(port["bias"], 0.0)

    def test_sklearn_layer_output(self, cell):
        assert cell["port"]["output"].shape == (N_ROWS, HIDDEN)
        assert cell["port"]["output"].dtype == np.float64
        np.testing.assert_allclose(
            cell["port"]["output"], cell["oracle"]["output"], rtol=0,
            atol=cell["value_atol"],
            err_msg="the layer output differs from scikit-learn's",
        )

    def test_sklearn_block_r2(self, cell):
        """``block_r2[q, p]`` is ``r2_score`` of block ``(q, p)`` against ``y``."""
        expected = np.array([
            [r2_score(cell["y"], cell["oracle"]["phi"][:, q, p]) for p in range(N_IN)]
            for q in range(HIDDEN)
        ])
        assert cell["port"]["block_r2"].shape == (HIDDEN, N_IN)
        np.testing.assert_allclose(
            cell["port"]["block_r2"], expected, rtol=0, atol=cell["value_atol"])


def test_the_named_deviation_cells_are_the_ill_conditioned_ones(cell):
    """The deviation list is a measurement, not a convenience.

    For a zero-ridge cell, the worst effective condition number over its 15
    blocks (largest singular value of the centered design over the smallest
    one scikit-learn's rank cutoff keeps) is above ``ILL_CONDITIONED`` exactly
    in the cells named as deviations. A cell added to the list without being
    ill-conditioned, or an ill-conditioned cell left off it, fails here.
    """
    if cell["l2"] != 0.0:
        assert not _is_deviation(cell["basis"], cell["l2"])
        return
    worst = 0.0
    for q in range(HIDDEN):
        for p in range(N_IN):
            design = block_features(cell["basis"], SLOPE, cell["x"][:, p], cell["centers"][q, p])
            if cell["intercept"]:
                design = design - design.mean(axis=0)
            singular = np.linalg.svd(design, compute_uv=False)
            kept = singular[singular > singular[0] * max(design.shape) * np.finfo(np.float64).eps]
            worst = max(worst, float(singular[0] / kept.min()))
    assert (worst > ILL_CONDITIONED) == _is_deviation(cell["basis"], cell["l2"]), (
        f"effective condition number {worst:.2e}")


def test_the_parity_fixture_can_tell_its_cells_apart(data):
    """Non-vacuity: the four settings of a cell give four different fits.

    If ``l2``, the intercept or the duplication did not change scikit-learn's
    answer on this fixture, a port that ignored the setting would pass.
    """
    x, y = data

    def first_block(intercept: bool, l2: float, duplicated: bool) -> np.ndarray:
        oracle = sklearn_layer(
            "tanh", SLOPE, x, y, cell_centers(duplicated), l2, 0.0, intercept, intercept)
        return oracle["coef"][0, 0]

    base = first_block(True, L2_POSITIVE, False)
    assert np.abs(base - first_block(True, 0.0, False)).max() > 1e-3
    assert np.abs(base - first_block(False, L2_POSITIVE, False)).max() > 1e-3
    assert np.abs(base - first_block(True, L2_POSITIVE, True)).max() > 1e-3


# ---------------------------------------------------------------------------
# The model: a two-layer chain
# ---------------------------------------------------------------------------

#: Four fixtures: the paper-like stack (both intercepts, ridge, identity top
#: layer); zero ridge with data-driven first-layer centers; the paper's
#: bias-free equations; a ridge on the connecting stage.
MODEL_CONFIGS = {
    "intercepts_ridge": dict(
        hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis=("tanh", "identity"),
        slope=SLOPE, centers="random", l2_block=(L2_POSITIVE, 0.1), seed=0),
    "zero_ridge_data_centers": dict(
        hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis=("gaussian", "softplus"),
        slope=SLOPE, centers=("data", "random"), l2_block=0.0, seed=0),
    "bias_free_ridge": dict(
        hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis=("sigmoid", "tanh"),
        slope=SLOPE, centers="random", l2_block=L2_POSITIVE,
        use_block_bias=False, use_bias=False, seed=0),
    "ridge_on_the_connecting_stage": dict(
        hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis="tanh",
        slope=SLOPE, centers="equally_spaced", l2_block=L2_POSITIVE,
        l2_mix=(0.5, 0.0), seed=0),
}

# Float32 forward pass against the float64 fit, max abs over the 96 rows,
# measured on the four fixtures above (`measure_cells.py`), CPU
# (CUDA_VISIBLE_DEVICES="") then GPU 1 (CUDA_VISIBLE_DEVICES=1):
#   intercepts_ridge               1.47e-6   1.44e-6    largest |weight| 2.5
#   bias_free_ridge                1.54e-6   1.18e-6    largest |weight| 7.8
#   ridge_on_the_connecting_stage  1.16e-6   6.72e-7    largest |weight| 2.4
#   zero_ridge_data_centers        2.37e-4   3.07e-4    largest |weight| 2.1e3
# The zero-ridge fixture is 200 times worse because its unregularized
# coefficients are 800 times larger and float32 carries 6e-8 of each.
# Each tolerance is 6.5 times the worse of its two readings: 1e-5 for the
# three ridge fixtures, 2e-3 for the zero-ridge one. Both sit above float32
# resolution at the weight scale and far below what a defect in the forward
# pass does (a dropped connecting intercept moves the output by more than
# 1e-1, RED proof i in decisions.md D-025).
#: Float64 agreement of the model with the scikit-learn chain. The zero-ridge
#: fixture has an unregularized softplus top layer, a deviation-class solve.
#: Measured worst: 1.7e-13 (ridge fixtures), 7.0e-14 (zero ridge).
CHAIN_ATOL = {
    "intercepts_ridge": VALUE_ATOL,
    "bias_free_ridge": VALUE_ATOL,
    "ridge_on_the_connecting_stage": VALUE_ATOL,
    "zero_ridge_data_centers": VALUE_ATOL_DEVIATION,
}
FLOAT32_ATOL = {
    "intercepts_ridge": 1e-5,
    "bias_free_ridge": 1e-5,
    "ridge_on_the_connecting_stage": 1e-5,
    "zero_ridge_data_centers": 2e-3,
}


def stored_centers(model: HKAN) -> Tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(keras.ops.convert_to_numpy(layer.centers), dtype=np.float64)
        for layer in model.hkan_layers)


@pytest.fixture(scope="module", params=sorted(MODEL_CONFIGS))
def fitted(request, data) -> Dict[str, Any]:
    """A fitted two-layer model, its diagnostics and the scikit-learn chain."""
    config = MODEL_CONFIGS[request.param]
    x, y = data
    model = HKAN(**config)
    diagnostics = model.fit_closed_form(x, y, chunk_size=2)
    intercepts = (config.get("use_block_bias", True), config.get("use_bias", True))
    chain = []
    activations = x
    for index, centers in enumerate(stored_centers(model)):
        chain.append(sklearn_layer(
            model._basis[index], SLOPE, activations, y, centers,
            model._l2_block[index], model._l2_mix[index], *intercepts))
        activations = chain[-1]["output"]
    return {"name": request.param, "config": config, "model": model,
            "diagnostics": diagnostics, "chain": chain, "x": x, "y": y,
            "intercepts": intercepts}


class TestTheModelIsTheSklearnChain:
    """``fit_closed_form`` = scikit-learn layer 1, then layer 2 on ITS output."""

    def test_train_predictions(self, fitted):
        predictions = fitted["diagnostics"]["train_predictions"]
        assert predictions.shape == (N_ROWS,) and predictions.dtype == np.float64
        np.testing.assert_allclose(
            predictions, fitted["chain"][-1]["output"][:, 0], rtol=0,
            atol=CHAIN_ATOL[fitted["name"]])

    def test_layer_rmse(self, fitted):
        expected = [
            float(np.sqrt(np.mean((layer["output"] - fitted["y"][:, None]) ** 2)))
            for layer in fitted["chain"]
        ]
        assert len(fitted["diagnostics"]["layer_rmse"]) == 2
        np.testing.assert_allclose(
            fitted["diagnostics"]["layer_rmse"], expected, rtol=0,
            atol=CHAIN_ATOL[fitted["name"]])

    def test_the_weights_hold_the_solution(self, fitted):
        """Every weight of the model is scikit-learn's value, cast to float32."""
        block_intercept, mix_intercept = fitted["intercepts"]
        for layer, oracle in zip(fitted["model"].hkan_layers, fitted["chain"]):
            names = ["coef", "mix"]
            names += ["block_bias"] if block_intercept else []
            names += ["bias"] if mix_intercept else []
            for name in names:
                weight = keras.ops.convert_to_numpy(getattr(layer, name))
                assert weight.dtype == np.float32
                # The port's float64 value agrees with the oracle's to the
                # fixture's relative tolerance; the float32 cast adds half an
                # ulp of the largest value. Two float32 ulps cover both for
                # the ridge fixtures; the zero-ridge fixture adds its 1e-6.
                scale = max(1.0, float(np.abs(oracle[name]).max()))
                np.testing.assert_allclose(
                    weight, oracle[name], rtol=0,
                    atol=(2 * np.finfo(np.float32).eps + CHAIN_ATOL[fitted["name"]]) * scale,
                    err_msg=f"{layer.name}.{name} is not the fitted value",
                )

    def test_float32_forward_reproduces_the_float64_fit(self, fitted):
        forward = keras.ops.convert_to_numpy(
            fitted["model"](fitted["x"].astype("float32"), training=False))
        assert forward.shape == (N_ROWS, 1)
        np.testing.assert_allclose(
            forward[:, 0], fitted["diagnostics"]["train_predictions"],
            rtol=0, atol=FLOAT32_ATOL[fitted["name"]])
        np.testing.assert_allclose(
            forward[:, 0], fitted["chain"][-1]["output"][:, 0],
            rtol=0, atol=FLOAT32_ATOL[fitted["name"]],
            err_msg="the forward pass is not scikit-learn's chain prediction")

    def test_the_fit_is_not_trivial(self, fitted):
        """Non-vacuity: the fit explains the target and is not a constant."""
        predictions = fitted["diagnostics"]["train_predictions"]
        # measured: 0.75 (bias-free) to 0.95
        assert r2_score(fitted["y"], predictions) > 0.7
        assert np.std(predictions) > 0.1

    def test_importance_is_the_mean_block_r2(self, fitted):
        """Paper equation 14: mean over the first layer's outputs of block R^2."""
        phi = fitted["chain"][0]["phi"]
        block_r2 = np.array([
            [r2_score(fitted["y"], phi[:, q, p]) for p in range(N_IN)]
            for q in range(HIDDEN)
        ])
        diagnostics = fitted["diagnostics"]
        assert diagnostics["block_r2"].shape == (HIDDEN, N_IN)
        assert diagnostics["importance"].shape == (N_IN,)
        np.testing.assert_allclose(
            diagnostics["block_r2"], block_r2, rtol=0, atol=VALUE_ATOL)
        np.testing.assert_allclose(
            diagnostics["importance"], block_r2.mean(axis=0), rtol=0, atol=VALUE_ATOL)
        # The three inputs are not equally informative, so an average over the
        # wrong axis or a permuted column would show.
        assert np.ptp(diagnostics["importance"]) > 0.05

    def test_the_diagnostics_keys(self, fitted):
        assert set(fitted["diagnostics"]) == {
            "layer_rmse", "block_r2", "importance", "train_predictions"}


# ---------------------------------------------------------------------------
# Determinism, seeds, chunking
# ---------------------------------------------------------------------------

def _fit(data, chunk_size=None, **overrides) -> Tuple[HKAN, Dict[str, Any]]:
    config = dict(MODEL_CONFIGS["zero_ridge_data_centers"])
    config.update(overrides)
    model = HKAN(**config)
    return model, model.fit_closed_form(*data, chunk_size=chunk_size)


class TestSeedsAndChunks:

    @pytest.mark.parametrize("seed", [0, 7])
    def test_one_seed_gives_one_fit(self, data, seed):
        """Same seed: identical centers (both modes) and identical predictions.

        ``seed=0`` is a seed. A truthiness test on the seed would send it down
        the unseeded branch and the two models would differ.
        """
        first_model, first = _fit(data, seed=seed)
        second_model, second = _fit(data, seed=seed)
        for a, b in zip(stored_centers(first_model), stored_centers(second_model)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            first["train_predictions"], second["train_predictions"])

    def test_another_seed_gives_other_centers(self, data):
        first_model, first = _fit(data, seed=0)
        second_model, second = _fit(data, seed=1)
        for a, b in zip(stored_centers(first_model), stored_centers(second_model)):
            assert np.abs(a - b).max() > 1e-2
        assert np.abs(first["train_predictions"] - second["train_predictions"]).max() > 1e-6

    def test_the_two_layers_draw_different_centers(self):
        """Same shape, same seed, different ``layer_index``: different draws."""
        model = HKAN(hidden_units=(N_IN,), num_basis=4, seed=0)
        model.build((None, N_IN))
        first, second = stored_centers(model)
        assert first[0].shape == second[0].shape == (N_IN, 4)
        assert np.abs(first[0] - second[0]).max() > 1e-2

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, HIDDEN, 64])
    def test_the_result_does_not_depend_on_chunk_size(self, data, chunk_size):
        """1, two non-divisors of ``HIDDEN=5``, all at once, and more than all.

        Measured: every weight and the predictions are bit-identical across
        chunk sizes, so the comparison is exact.
        """
        reference_model, reference = _fit(data, chunk_size=None)
        model, result = _fit(data, chunk_size=chunk_size)
        np.testing.assert_array_equal(
            result["train_predictions"], reference["train_predictions"])
        np.testing.assert_array_equal(result["block_r2"], reference["block_r2"])
        assert len(model.get_weights()) == len(reference_model.get_weights()) == 10
        for ours, theirs in zip(model.get_weights(), reference_model.get_weights()):
            np.testing.assert_array_equal(ours, theirs)

    def test_chunk_size_below_one_is_refused(self, data):
        with pytest.raises(ValueError, match="chunk_size"):
            _fit(data, chunk_size=0)


# ---------------------------------------------------------------------------
# Data-driven centers
# ---------------------------------------------------------------------------

def _nearest(values: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Distance from each of ``values`` to the closest member of ``pool``."""
    return np.abs(values[:, None] - pool[None, :]).min(axis=1)


class TestDataCenters:
    """``centers="data"``: block ``(q, p)`` samples column ``p`` of the layer's input.

    The second layer is as wide as the input (3 -> 3 -> 1) ON PURPOSE, against
    this directory's non-square rule: with different widths, drawing a deeper
    layer's centers from ``x`` is a shape error and needs no guard; with equal
    widths it is silent. The target is shifted by +2 so that the hidden
    activations (1.6 to 3.7) and ``x`` (0 to 1) cannot be confused.
    """

    # Distance from a second-layer center to the nearest float32 activation of
    # the first layer. Measured max: 9.5e-7 (CPU), 7.2e-7 (GPU 1), on values
    # of 1.6 to 3.7. 1e-5 is 10 times the worse reading and about 40 float32
    # ulps at that scale; a center drawn from `x` instead is more than 1.1
    # away (measured).
    ACTIVATION_ATOL = 1e-5

    @pytest.fixture(scope="class")
    def data_fit(self):
        x, y = make_data(offset=2.0)
        model = HKAN(hidden_units=(N_IN,), num_basis=NUM_BASIS, basis="tanh",
                     slope=SLOPE, centers="data", l2_block=L2_POSITIVE, seed=0)
        model.build((None, N_IN))
        placeholder = stored_centers(model)
        diagnostics = model.fit_closed_form(x, y)
        return model, x, y, placeholder, diagnostics

    def test_first_layer_centers_are_rows_of_their_own_column(self, data_fit):
        model, x, _, _, _ = data_fit
        centers = stored_centers(model)[0]
        assert centers.shape == (N_IN, N_IN, NUM_BASIS[0])
        x32 = x.astype(np.float32).astype(np.float64)
        for p in range(N_IN):
            own = _nearest(centers[:, p, :].ravel(), x32[:, p])
            np.testing.assert_array_equal(
                own, 0.0, err_msg=f"a center of input {p} is not a value of column {p}")
            other = _nearest(centers[:, p, :].ravel(), x32[:, (p + 1) % N_IN])
            assert other.min() > 0.0, "the fixture cannot tell two columns apart"

    def test_blocks_sample_independently(self, data_fit):
        model, _, _, _, _ = data_fit
        centers = stored_centers(model)[0]
        assert np.abs(centers[0, 0] - centers[1, 0]).max() > 1e-3
        assert np.unique(centers[:, 0, :]).size > NUM_BASIS[0]

    def test_the_placeholder_was_replaced(self, data_fit):
        model, _, _, placeholder, _ = data_fit
        for before, after in zip(placeholder, stored_centers(model)):
            assert np.abs(before - after).max() > 1e-3

    def test_a_deeper_layer_draws_from_its_own_input(self, data_fit):
        model, x, _, _, _ = data_fit
        hidden = np.asarray(keras.ops.convert_to_numpy(
            model.hkan_layers[0](x.astype("float32"))), dtype=np.float64)
        assert hidden.min() > 1.2, "the fixture's activations overlap the inputs"
        centers = stored_centers(model)[1]
        assert centers.shape == (1, N_IN, NUM_BASIS[1])
        for p in range(N_IN):
            # The draw is from the float64 fit's activations; `hidden` is the
            # float32 forward pass of the same layer.
            distance = _nearest(centers[0, p], hidden[:, p])
            assert distance.max() <= self.ACTIVATION_ATOL, (
                f"second-layer centers of input {p} are up to {distance.max():.3e} "
                "from the nearest activation of the first layer: they were not "
                "drawn from this layer's input")
