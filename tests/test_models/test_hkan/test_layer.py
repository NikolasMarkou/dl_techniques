"""``HKANLayer``: its weights, its two basis tables, its forward arithmetic.

Guards:

* **Weights.** One stacked tensor per role, created in ``build()``, in a fixed
  order; ``block_bias`` and ``bias`` exist only when their flag is on and do
  not reorder the others; ``centers`` is non-trainable.
* **The two basis tables.** ``NUMPY_BASIS`` (used by the fit) and
  ``KERAS_BASIS`` (used by ``call()``) are each compared with the scipy/numpy
  closed form of the function, and with each other. If they drift apart the
  model that is fitted is not the model that predicts.
* **The forward arithmetic.** ``call()`` and ``block_outputs()`` against the
  module's two documented equations, transcribed here with explicit loops over
  ``q``, ``p`` and ``r`` in float64, on weights that are all different random
  numbers (so that a transposed axis, a flipped sign, an intercept added in
  the wrong place or a dropped intercept changes the numbers).
* The initial state (``mix = 1/n_in``, seeded centers, ``seed=0`` is a seed,
  one initializer instance is cloned per weight), the data-driven center draw,
  the config round trip and the argument validation.
* ``solve_linear_float64`` refuses anything that is not float64.
* **Seed streams.** No seeded stream of the layer is ``default_rng(seed)``
  (numpy drops trailing zeros of a seed list, so ``[seed, 0, 0]`` was), and
  the build stream, the data-draw stream and two layers' streams all differ.
* **Input width and dtype.** A built layer refuses an input of another width,
  width 1 included (which would broadcast silently); a float16 compute dtype
  is refused at build; numpy integers are accepted and stored as ``int``.
* A constant target has block R^2 0.0, and a non-finite solution is refused
  by the solve itself, before anything could be assigned.
"""

import inspect
from typing import Dict

import keras
import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import r2_score

from dl_techniques.models.general_purpose.hkan.hkan_layer import (
    BASIS_NAMES,
    CENTER_MODES,
    KERAS_BASIS,
    NUMPY_BASIS,
    SEED_DOMAIN_TAG,
    HKANLayer,
    solve_linear_float64,
)

from . import CLOSED_FORM, HIDDEN, N_IN, NUM_BASIS, SLOPE, block_features, make_data

N_BASIS = NUM_BASIS[0]
BATCH = 6


def _numpy(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _built(**overrides) -> HKANLayer:
    config = dict(units=HIDDEN, num_basis=N_BASIS, basis="tanh", slope=SLOPE, seed=0)
    config.update(overrides)
    layer = HKANLayer(**config)
    layer.build((None, N_IN))
    return layer


def _randomize(layer: HKANLayer, seed: int = 21) -> Dict[str, np.ndarray]:
    """Give every weight distinct random values; return them as float64."""
    rng = np.random.default_rng(seed)
    values = {}
    for weight in layer.weights:
        drawn = rng.uniform(-1.0, 1.0, size=tuple(weight.shape)).astype("float32")
        weight.assign(drawn)
        values[weight.name] = drawn.astype(np.float64)
    return values


def documented_forward(layer: HKANLayer, values: Dict[str, np.ndarray], x: np.ndarray):
    """The module docstring's two equations, one scalar at a time.

    ``phi[q, p] = sum_r coef[q, p, r] * g(slope * (x_p - centers[q, p, r])) + block_bias[q, p]``
    ``h[q]      = sum_p mix[q, p] * phi[q, p] + bias[q]``

    :return: ``(phi (B, n_out, n_in), h (B, n_out))`` in float64.
    """
    n_out, n_in, _ = values["centers"].shape
    phi = np.zeros((x.shape[0], n_out, n_in))
    h = np.zeros((x.shape[0], n_out))
    for q in range(n_out):
        for p in range(n_in):
            features = block_features(layer.basis, layer.slope, x[:, p], values["centers"][q, p])
            phi[:, q, p] = features @ values["coef"][q, p]
            if "block_bias" in values:
                phi[:, q, p] += values["block_bias"][q, p]
            h[:, q] += values["mix"][q, p] * phi[:, q, p]
        if "bias" in values:
            h[:, q] += values["bias"][q]
    return phi, h


# ---------------------------------------------------------------------------


class TestWeights:

    def test_weight_names_shapes_and_order(self):
        layer = _built()
        # Keras lists the trainable weights first, then the non-trainable one.
        assert [w.name for w in layer.weights] == [
            "coef", "block_bias", "mix", "bias", "centers"]
        assert [tuple(w.shape) for w in layer.weights] == [
            (HIDDEN, N_IN, N_BASIS), (HIDDEN, N_IN), (HIDDEN, N_IN),
            (HIDDEN,), (HIDDEN, N_IN, N_BASIS)]

    @pytest.mark.parametrize("block_bias,bias,expected", [
        (False, True, ["coef", "mix", "bias", "centers"]),
        (True, False, ["coef", "block_bias", "mix", "centers"]),
        (False, False, ["coef", "mix", "centers"]),
    ])
    def test_a_conditional_weight_does_not_reorder_the_others(
            self, block_bias, bias, expected):
        layer = _built(use_block_bias=block_bias, use_bias=bias)
        assert [w.name for w in layer.weights] == expected
        assert (layer.block_bias is None) == (not block_bias)
        assert (layer.bias is None) == (not bias)

    def test_centers_never_train(self):
        layer = _built()
        assert layer.centers.trainable is False
        assert [w.name for w in layer.trainable_weights] == [
            "coef", "block_bias", "mix", "bias"]
        assert [w.name for w in layer.non_trainable_weights] == ["centers"]

    def test_no_weight_before_build(self):
        layer = HKANLayer(units=HIDDEN)
        assert layer.weights == [] and layer.centers is None and layer.coef is None

    @pytest.mark.parametrize("shape", [(None,), (None, 4, N_IN), (None, None)])
    def test_build_refuses_a_bad_input_shape(self, shape):
        with pytest.raises(ValueError, match="n_in"):
            HKANLayer(units=HIDDEN).build(shape)


class TestInitialState:

    def test_mix_starts_at_one_over_n_in_and_coef_is_not_zero(self):
        """The gradient-trained modes need a live starting point (D-007)."""
        layer = _built()
        np.testing.assert_array_equal(_numpy(layer.mix), np.float32(1.0 / N_IN))
        assert np.abs(_numpy(layer.coef)).min() > 0.0
        assert np.std(_numpy(layer.coef)) > 1e-3

    def test_a_given_mix_initializer_is_used(self):
        layer = _built(mix_initializer=keras.initializers.Constant(0.25))
        np.testing.assert_array_equal(_numpy(layer.mix), np.float32(0.25))

    def test_random_centers_are_uniform_draws_on_the_unit_interval(self):
        centers = _numpy(_built(units=40).centers)
        assert centers.min() >= 0.0 and centers.max() <= 1.0
        assert centers.min() < 0.05 and centers.max() > 0.95
        assert np.unique(centers).size == centers.size

    def test_equally_spaced_centers_are_a_linspace_per_block(self):
        centers = _numpy(_built(centers="equally_spaced").centers)
        expected = np.linspace(0.0, 1.0, N_BASIS).astype("float32")
        np.testing.assert_array_equal(
            centers, np.broadcast_to(expected, (HIDDEN, N_IN, N_BASIS)))

    @pytest.mark.parametrize("seed", [0, 5])
    def test_one_seed_gives_the_same_centers(self, seed):
        """``seed=0`` included: it is a seed, not "unseeded"."""
        np.testing.assert_array_equal(
            _numpy(_built(seed=seed).centers), _numpy(_built(seed=seed).centers))

    def test_seed_and_layer_index_both_change_the_centers(self):
        base = _numpy(_built(seed=0).centers)
        assert np.abs(base - _numpy(_built(seed=1).centers)).max() > 1e-2
        assert np.abs(base - _numpy(_built(seed=0, layer_index=1).centers)).max() > 1e-2

    def test_unseeded_layers_differ(self):
        assert np.abs(
            _numpy(_built(seed=None).centers) - _numpy(_built(seed=None).centers)
        ).max() > 1e-2

    @pytest.mark.parametrize("initializer", [
        "random_normal", keras.initializers.RandomNormal(stddev=0.3)])
    def test_one_initializer_instance_is_cloned_per_weight(self, initializer):
        """Two layers given the SAME seedless instance draw different ``coef``.

        A Keras initializer instance fixes its seed at construction and
        replays its draw on every call, so handing one instance to two
        ``add_weight`` sites without cloning makes the two weights equal.
        """
        instance = keras.initializers.get(initializer)
        first = _built(coef_initializer=instance)
        second = _built(coef_initializer=instance)
        assert not np.array_equal(_numpy(first.coef), _numpy(second.coef))


class TestSeedStreams:
    """The seed list is ``[seed, layer_index, stream, SEED_DOMAIN_TAG]``.

    The expected generators are written out here, not taken from the layer,
    so the layout is pinned: the tag is last and is not zero.
    """

    SHAPE = (HIDDEN, N_IN, N_BASIS)

    @staticmethod
    def _expected(seed: int, layer_index: int, stream: int) -> np.random.Generator:
        return np.random.default_rng([seed, layer_index, stream, 0x484B414E])

    def test_the_tag_is_not_zero(self):
        assert SEED_DOMAIN_TAG == 0x484B414E != 0

    @pytest.mark.parametrize("seed", [0, 7, 123])
    def test_layer_zero_centers_are_not_the_callers_own_generator(self, seed):
        """``default_rng(seed)`` is what a caller draws the data from."""
        centers = _numpy(_built(seed=seed).centers)
        callers = np.random.default_rng(seed).uniform(0.0, 1.0, size=self.SHAPE)
        assert np.abs(centers - callers.astype("float32")).max() > 1e-2, (
            "layer 0's centers are the first values of default_rng(seed)")
        np.testing.assert_array_equal(
            centers,
            self._expected(seed, 0, 0).uniform(0.0, 1.0, size=self.SHAPE).astype("float32"))

    @pytest.mark.parametrize("seed", [0, 7, 123])
    @pytest.mark.parametrize("layer_index", [0, 1])
    @pytest.mark.parametrize("stream", [0, 1])
    def test_no_stream_is_the_callers_own_generator(self, seed, layer_index, stream):
        layer = HKANLayer(units=HIDDEN, seed=seed, layer_index=layer_index)
        ours = layer._generator(stream).uniform(size=16)
        assert np.abs(ours - np.random.default_rng(seed).uniform(size=16)).max() > 1e-2
        np.testing.assert_array_equal(
            ours, self._expected(seed, layer_index, stream).uniform(size=16))

    @pytest.mark.parametrize("seed", [0, 7])
    def test_the_data_draw_is_not_the_build_stream(self, seed):
        """The rows drawn for ``data`` centers against the build stream's.

        ``x[i, p] = i``, so the drawn centers ARE the drawn row indices.
        """
        n_rows = 1000
        x = np.broadcast_to(np.arange(n_rows, dtype=np.float64)[:, None], (n_rows, N_IN))
        layer = HKANLayer(units=HIDDEN, num_basis=N_BASIS, centers="data", seed=seed)
        rows = layer.sample_data_centers(x)
        build_stream = self._expected(seed, 0, 0).integers(0, n_rows, size=self.SHAPE)
        data_stream = self._expected(seed, 0, 1).integers(0, n_rows, size=self.SHAPE)
        assert (rows != build_stream).mean() > 0.9, (
            "the data draw replays the build-time stream")
        np.testing.assert_array_equal(rows, data_stream)

    @pytest.mark.parametrize("stream", [0, 1])
    def test_two_layers_have_different_streams(self, stream):
        first = HKANLayer(units=HIDDEN, seed=0, layer_index=0)._generator(stream)
        second = HKANLayer(units=HIDDEN, seed=0, layer_index=1)._generator(stream)
        assert np.abs(first.uniform(size=16) - second.uniform(size=16)).max() > 1e-2


class TestBasisTables:
    """Both tables against scipy/numpy closed forms, and against each other."""

    #: Arguments the hidden layers really see: |slope * (x - c)| reaches
    #: thousands at slope 50 on activations outside [0, 1].
    WIDE = np.concatenate([
        np.linspace(-6.0, 6.0, 49), [-5e3, -750.0, -40.0, 40.0, 750.0, 5e3]])

    def test_the_tables_cover_exactly_the_documented_names(self):
        assert set(NUMPY_BASIS) == set(KERAS_BASIS) == set(BASIS_NAMES) == set(CLOSED_FORM)
        assert CENTER_MODES == ("random", "equally_spaced", "data")

    @pytest.mark.parametrize("basis", sorted(CLOSED_FORM))
    def test_numpy_table_is_the_closed_form(self, basis):
        """Float64, tight: 4 ulps relative plus 1e-300 for the underflow tail.

        Measured worst relative difference over the six names: 4.3e-16
        (sigmoid, ``exp(-logaddexp(0, -z))`` against ``expit``), two ulps.
        """
        z = self.WIDE if basis != "gaussian" else np.linspace(-6.0, 6.0, 49)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            ours = NUMPY_BASIS[basis](z)
        expected = CLOSED_FORM[basis](z)
        assert ours.dtype == np.float64 and np.all(np.isfinite(ours))
        np.testing.assert_allclose(ours, expected, rtol=4 * np.finfo(np.float64).eps, atol=1e-300)

    @pytest.mark.parametrize("basis", sorted(CLOSED_FORM))
    def test_numpy_table_does_not_overflow(self, basis):
        z = np.array([-1e5, -5e3, 5e3, 1e5])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            assert np.all(np.isfinite(NUMPY_BASIS[basis](z)))

    @pytest.mark.parametrize("basis", sorted(CLOSED_FORM))
    def test_keras_table_is_the_closed_form(self, basis):
        """Float32 ops against the float64 closed form of the float32 argument.

        ``rtol=0``. The values are at most 6 in magnitude on ``[-6, 6]``
        (one float32 ulp there is 4.8e-7); measured worst difference over the
        six names 3.7e-7 on the CPU and 1.3e-7 on GPU 1.
        ``atol = 2e-6`` is 5 times the worse reading. The wide arguments are
        checked for finiteness and saturation separately, below.
        """
        z = np.linspace(-6.0, 6.0, 49).astype("float32")
        ours = _numpy(KERAS_BASIS[basis](keras.ops.convert_to_tensor(z)))
        np.testing.assert_allclose(
            ours, CLOSED_FORM[basis](z.astype(np.float64)), rtol=0, atol=2e-6)

    @pytest.mark.parametrize("basis", sorted(CLOSED_FORM))
    def test_the_two_tables_are_the_same_function(self, basis):
        """The fit's table against the forward pass's, on the same arguments."""
        z = np.linspace(-6.0, 6.0, 49).astype("float32")
        forward = _numpy(KERAS_BASIS[basis](keras.ops.convert_to_tensor(z)))
        fit = NUMPY_BASIS[basis](z.astype(np.float64))
        np.testing.assert_allclose(forward, fit, rtol=0, atol=2e-6)
        # The functions are not all the same one: the table is not six
        # aliases of one entry.
        assert np.abs(fit - z).max() > 0.5 or basis == "identity"

    @pytest.mark.parametrize("basis", sorted(CLOSED_FORM))
    def test_keras_table_is_finite_far_from_zero(self, basis):
        z = np.array([-5e3, -750.0, 750.0, 5e3], dtype="float32")
        ours = _numpy(KERAS_BASIS[basis](keras.ops.convert_to_tensor(z)))
        assert np.all(np.isfinite(ours))
        # Relative, because softplus, relu and identity return 5e3 here.
        np.testing.assert_allclose(
            ours, CLOSED_FORM[basis](z.astype(np.float64)), rtol=1e-6, atol=1e-6)

    def test_the_closed_forms_at_known_points(self):
        """The oracle itself, at values known by hand."""
        assert CLOSED_FORM["sigmoid"](np.array(0.0)) == 0.5 == expit(0.0)
        assert CLOSED_FORM["gaussian"](np.array(1.0)) == np.exp(-1.0)
        assert CLOSED_FORM["softplus"](np.array(0.0)) == np.log(2.0)
        assert CLOSED_FORM["relu"](np.array(-2.0)) == 0.0
        assert CLOSED_FORM["identity"](np.array(-2.0)) == -2.0


class TestForwardArithmetic:
    """``call()`` is the documented pair of equations.

    Tolerance, ``rtol=0``: with weights and inputs uniform on ``[-1, 1]`` /
    ``[0, 1]`` the outputs are sums of at most ``N_IN * N_BASIS = 21`` products
    of magnitude below 1 (up to 5 for softplus and relu at slope 5). Measured
    worst difference between the float32 layer and the float64 loops over the
    six basis names and four intercept settings: 2.5e-6 on the CPU, 2.2e-6 on
    GPU 1. ``atol = 1e-5`` is 4 times the worse reading. A flipped sign, a dropped or
    misplaced intercept, or a sum over the wrong axis moves the output by more
    than 1e-2 (RED proofs i, j, n, o, p in decisions.md D-025).
    """

    ATOL = 1e-5

    @pytest.fixture(scope="class")
    def x(self):
        return np.random.default_rng(3).uniform(0.0, 1.0, size=(BATCH, N_IN))

    @pytest.mark.parametrize("basis", sorted(CLOSED_FORM))
    @pytest.mark.parametrize("block_bias", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    def test_call_is_the_documented_equations(self, x, basis, block_bias, bias):
        layer = _built(basis=basis, use_block_bias=block_bias, use_bias=bias)
        values = _randomize(layer)
        phi, h = documented_forward(layer, values, x)

        ours_phi = _numpy(layer.block_outputs(keras.ops.convert_to_tensor(x.astype("float32"))))
        ours = _numpy(layer(x.astype("float32")))
        assert ours_phi.shape == (BATCH, HIDDEN, N_IN)
        assert ours.shape == (BATCH, HIDDEN), (
            f"the layer output has shape {ours.shape}, expected {(BATCH, HIDDEN)}")
        np.testing.assert_allclose(ours_phi, phi, rtol=0, atol=self.ATOL)
        np.testing.assert_allclose(ours, h, rtol=0, atol=self.ATOL)

    def test_the_fixture_separates_the_terms(self, x):
        """Non-vacuity: each term of the equations is visible at ``ATOL``."""
        layer = _built()
        values = _randomize(layer)
        _, h = documented_forward(layer, values, x)
        for name in ("block_bias", "bias"):
            without = {k: v for k, v in values.items() if k != name}
            assert np.abs(documented_forward(layer, without, x)[1] - h).max() > 1e-2
        swapped = dict(values, mix=values["mix"][::-1])
        assert np.abs(documented_forward(layer, swapped, x)[1] - h).max() > 1e-2

    def test_identity_ignores_the_slope(self, x):
        """``identity`` is ``g(d) = d`` on the unscaled difference (D-018)."""
        outputs = []
        for slope in (1.0, 50.0):
            layer = _built(basis="identity", slope=slope)
            _randomize(layer)
            outputs.append(_numpy(layer(x.astype("float32"))))
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_the_slope_reaches_a_non_linear_basis(self, x):
        outputs = []
        for slope in (1.0, 50.0):
            layer = _built(basis="tanh", slope=slope)
            _randomize(layer)
            outputs.append(_numpy(layer(x.astype("float32"))))
        assert np.abs(outputs[0] - outputs[1]).max() > 1e-2

    def test_compute_output_shape(self):
        assert _built().compute_output_shape((None, N_IN)) == (None, HIDDEN)
        assert _built().compute_output_shape((4, N_IN)) == (4, HIDDEN)

    @pytest.mark.parametrize("basis", sorted(CLOSED_FORM))
    def test_inputs_far_outside_the_unit_interval_stay_finite(self, basis):
        layer = _built(basis=basis, slope=50.0)
        x = np.array([[-100.0, 0.5, 100.0], [30.0, -30.0, 2.0]], dtype="float32")
        assert np.all(np.isfinite(_numpy(layer(x))))


class TestDataCenterDraw:

    def test_each_block_samples_its_own_column(self):
        x, _ = make_data()
        layer = HKANLayer(units=HIDDEN, num_basis=N_BASIS, centers="data", seed=0)
        centers = layer.sample_data_centers(x)
        assert centers.shape == (HIDDEN, N_IN, N_BASIS) and centers.dtype == x.dtype
        for p in range(N_IN):
            assert np.isin(centers[:, p, :], x[:, p]).all(), (
                f"a center of input {p} is not a value of column {p}")
            assert not np.isin(centers[:, p, :], x[:, (p + 1) % N_IN]).any()
        assert np.unique(centers[:, 0, :]).size > N_BASIS, "every block drew the same rows"

    def test_the_draw_is_seeded_and_assigns_nothing(self):
        x, _ = make_data()
        layer = _built(centers="data", seed=0)
        before = _numpy(layer.centers).copy()
        np.testing.assert_array_equal(
            layer.sample_data_centers(x), layer.sample_data_centers(x))
        np.testing.assert_array_equal(_numpy(layer.centers), before)

    @pytest.mark.parametrize("bad", [np.zeros((0, N_IN)), np.zeros((4,))])
    def test_a_bad_array_is_refused(self, bad):
        with pytest.raises(ValueError, match="draw centers"):
            HKANLayer(units=HIDDEN).sample_data_centers(bad)


class TestSolveGuards:

    def test_the_solve_refuses_float32(self):
        features = np.ones((4, 2), dtype=np.float32)
        target = np.ones((4,), dtype=np.float64)
        with pytest.raises(TypeError, match="float64"):
            solve_linear_float64(features, target, 0.0, True)
        with pytest.raises(TypeError, match="float64"):
            solve_linear_float64(features.astype(np.float64), target.astype(np.float32), 0.0, True)

    def test_the_layer_solve_refuses_float32_and_bad_shapes(self):
        x, y = make_data()
        layer = HKANLayer(units=HIDDEN, num_basis=N_BASIS)
        centers = np.zeros((HIDDEN, N_IN, N_BASIS))
        with pytest.raises(TypeError, match="float64"):
            layer.solve_closed_form(x.astype(np.float32), y, 0.0, 0.0, centers)
        with pytest.raises(ValueError, match="centers must have shape"):
            layer.solve_closed_form(x, y, 0.0, 0.0, centers[:, :, :-1])
        with pytest.raises(ValueError, match="expected x"):
            layer.solve_closed_form(x, y[:-1], 0.0, 0.0, centers)

    def test_assign_needs_a_built_layer(self):
        with pytest.raises(ValueError, match="must be built"):
            HKANLayer(units=HIDDEN).assign_solution({})

    def test_a_solution_that_overflows_float32_is_not_assigned(self):
        layer = _built()
        before = [w.numpy().copy() for w in layer.weights]
        solution = {
            "coef": np.full((HIDDEN, N_IN, N_BASIS), 1e300),
            "block_bias": np.zeros((HIDDEN, N_IN)),
            "mix": np.zeros((HIDDEN, N_IN)), "bias": np.zeros((HIDDEN,)),
        }
        with pytest.raises(ValueError, match="non-finite"):
            layer.assign_solution(solution)
        for weight, value in zip(layer.weights, before):
            np.testing.assert_array_equal(weight.numpy(), value)


    def test_a_non_finite_solution_is_refused_by_the_solve_itself(self):
        """Finite inputs of magnitude 1e160: the Gram matrix overflows, the
        ridge solve returns NaN without raising, and ``solve_closed_form``
        must refuse it before returning. (With ``l2_mix == 0`` numpy's
        ``lstsq`` raises ``LinAlgError`` first, so both stages are ridged.)"""
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=(40, N_IN)) * 1e160
        y = rng.normal(size=40)
        centers = rng.uniform(0.0, 1.0, size=(2, N_IN, 4))
        layer = HKANLayer(units=2, num_basis=4, basis="relu", slope=SLOPE)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        with np.errstate(all="ignore"):
            with pytest.raises(ValueError, match="non-finite values in 'coef'"):
                layer.solve_closed_form(x, y, 0.01, 0.5, centers)

    @pytest.mark.parametrize("block_bias", [True, False])
    def test_a_constant_target_has_block_r2_zero(self, block_bias):
        """No variance to explain: 0.0, where ``r2_score`` says 1.0.

        Two cells, because the two wrong answers hide in different ones. With
        a block intercept every block reproduces the constant exactly
        (measured residual 1e-30), where ``r2_score``'s convention is 1.0. Without one a
        ridged block cannot (measured residual sum of squares 0.10 to 2.5),
        where an "exact fit or not" switch the other way round gives 1.0.
        """
        x, _ = make_data()
        y = np.full((x.shape[0],), 0.7)
        layer = HKANLayer(units=HIDDEN, num_basis=N_BASIS, basis="tanh", slope=SLOPE,
                          use_block_bias=block_bias)
        centers = np.random.default_rng(2).uniform(0.0, 1.0, size=(HIDDEN, N_IN, N_BASIS))
        solution = layer.solve_closed_form(x, y, 0.01, 0.0, centers)
        residual = np.array([[
            np.sum((block_features("tanh", SLOPE, x[:, p], centers[q, p])
                    @ solution["coef"][q, p] + solution["block_bias"][q, p] - y) ** 2)
            for p in range(N_IN)] for q in range(HIDDEN)])
        if block_bias:
            assert residual.max() < 1e-24
        else:
            assert residual.min() > 1e-3
        np.testing.assert_array_equal(solution["block_r2"], np.zeros((HIDDEN, N_IN)))
        # The convention this departs from, stated by the oracle itself.
        assert r2_score(y, y) == 1.0


class TestInputWidth:
    """A built layer refuses another width; width 1 would broadcast."""

    @pytest.mark.parametrize("width", [1, N_IN + 2])
    def test_call_refuses_a_wrong_width(self, width):
        layer = _built()
        with pytest.raises(ValueError, match="incompatible with the layer"):
            layer(np.zeros((2, width), dtype="float32"))

    def test_call_refuses_a_wrong_rank(self):
        with pytest.raises(ValueError, match="incompatible with the layer"):
            _built()(np.zeros((2, 1, N_IN), dtype="float32"))

    def test_the_right_width_still_runs(self):
        assert _numpy(_built()(np.zeros((2, N_IN), dtype="float32"))).shape == (2, HIDDEN)

    @pytest.mark.parametrize("width", [1, N_IN + 2])
    def test_the_functional_api_carries_the_check(self, width):
        inputs = keras.Input((N_IN,))
        functional = keras.Model(inputs, HKANLayer(units=HIDDEN, seed=0)(inputs))
        assert functional(np.zeros((2, N_IN), dtype="float32")).shape == (2, HIDDEN)
        wrong = keras.Input((width,))
        with pytest.raises(ValueError, match="incompatible with the layer"):
            functional.layers[-1](wrong)


@pytest.fixture
def policy(request):
    """Set the global dtype policy for one test and always restore it."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous)


class TestDtypePolicy:

    @pytest.mark.parametrize("policy", ["mixed_float16", "float16"], indirect=True)
    def test_a_float16_compute_dtype_is_refused_at_build(self, policy):
        layer = HKANLayer(units=HIDDEN, seed=0)
        with pytest.raises(ValueError, match="'float32' and 'float64'") as raised:
            layer.build((None, N_IN))
        assert "float16" in str(raised.value) and policy in str(raised.value)
        assert layer.weights == []

    @pytest.mark.parametrize("policy", ["float32", "float64"], indirect=True)
    def test_float32_and_float64_build_and_run(self, policy):
        layer = _built()
        output = _numpy(layer(np.zeros((2, N_IN), dtype=policy)))
        assert output.dtype == np.dtype(policy) and output.shape == (2, HIDDEN)
        assert _numpy(layer.centers).dtype == np.dtype(policy)

    def test_a_per_layer_float16_dtype_is_refused_too(self):
        with pytest.raises(ValueError, match="'float32' and 'float64'"):
            HKANLayer(units=HIDDEN, dtype="float16").build((None, N_IN))

    def test_the_policy_was_restored(self):
        assert keras.mixed_precision.global_policy().name == "float32"


class TestConfig:

    def test_numpy_integers_are_accepted_and_stored_as_int(self):
        layer = HKANLayer(units=np.int64(HIDDEN), num_basis=np.int32(N_BASIS),
                          seed=np.int64(3), layer_index=np.uint8(1))
        config = layer.get_config()
        for name, value in (("units", HIDDEN), ("num_basis", N_BASIS),
                            ("seed", 3), ("layer_index", 1)):
            assert config[name] == value and type(config[name]) is int, name
        layer.build((None, N_IN))
        np.testing.assert_array_equal(
            _numpy(layer.centers),
            _numpy(_built(seed=3, layer_index=1).centers))

    @pytest.mark.parametrize("kwargs,match", [
        (dict(units=np.bool_(True)), "units"), (dict(units=np.float64(2.0)), "units"),
        (dict(units=2, num_basis=np.float32(3.0)), "num_basis"),
        (dict(units=2, seed=np.bool_(True)), "seed"),
        (dict(units=2, seed=np.int64(-1)), "seed"),
    ])
    def test_numpy_booleans_and_floats_are_still_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HKANLayer(**kwargs)

    def test_get_config_carries_every_constructor_argument(self):
        arguments = set(inspect.signature(HKANLayer.__init__).parameters) - {"self", "kwargs"}
        assert arguments <= set(_built().get_config())

    def test_config_round_trip(self):
        layer = _built(
            num_basis=4, basis="gaussian", slope=2.5, centers="equally_spaced",
            use_block_bias=False, use_bias=False,
            coef_initializer=keras.initializers.RandomNormal(stddev=0.2),
            mix_initializer="ones", seed=0, layer_index=2)
        config = layer.get_config()
        clone = HKANLayer.from_config(config)
        assert clone.get_config() == config
        assert config["seed"] == 0 and config["layer_index"] == 2
        assert config["centers"] == "equally_spaced" and config["basis"] == "gaussian"
        clone.build((None, N_IN))
        assert [w.name for w in clone.weights] == ["coef", "mix", "centers"]

    def test_seed_none_round_trips_as_none(self):
        assert HKANLayer.from_config(_built(seed=None).get_config()).seed is None

    def test_the_registered_name(self):
        assert keras.saving.get_registered_name(HKANLayer) == (
            "dl_techniques.models.hkan.hkan_layer>HKANLayer")

    @pytest.mark.parametrize("kwargs,match", [
        (dict(units=0), "units"), (dict(units=True), "units"), (dict(units=2.0), "units"),
        (dict(units=2, num_basis=0), "num_basis"),
        (dict(units=2, basis="spline"), "basis"),
        (dict(units=2, slope=0.0), "slope"), (dict(units=2, slope=float("inf")), "slope"),
        (dict(units=2, slope=-1.0), "slope"),
        (dict(units=2, centers="grid"), "centers"),
        (dict(units=2, seed=-1), "seed"), (dict(units=2, seed=1.5), "seed"),
        (dict(units=2, layer_index=-1), "layer_index"),
    ])
    def test_invalid_arguments_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HKANLayer(**kwargs)
