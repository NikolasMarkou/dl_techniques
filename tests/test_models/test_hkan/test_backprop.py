"""HKAN trains through stock ``compile()`` / ``fit()``, from either start.

The package has no custom training step, so everything here goes through
``keras.Model.fit`` and ``keras.Model.evaluate`` as shipped. Guards:

* stock ``fit`` lowers the loss from the initial state;
* the initial state is not a dead point: the shared gradient-flow oracle finds
  a non-zero gradient on every trainable weight BEFORE any update (where a
  zero ``mix`` would silence ``coef``), and again after one real optimizer
  step; every trainable weight moves in that step;
* ``centers`` is bit-identical before and after ``fit`` in every layer;
* ``fit_closed_form`` followed by ``fit`` starts from the closed-form
  solution: the loss Keras reports before any update equals the closed-form
  train MSE, whether the closed-form fit ran before ``compile()`` or after the
  evaluate function had already been traced;
* one initializer instance shared by the model's layers still gives every
  layer its own ``coef`` draw;
* ``initialize_centers`` (the gradient-only route to ``centers="data"``)
  draws each layer's centers from that layer's own forward input and leaves
  the other layers alone, and propagates EVERY row, also past the first
  forward batch of 256.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.general_purpose.hkan import HKAN

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from . import HIDDEN, N_IN, N_ROWS, NUM_BASIS, SLOPE, make_data

TRAINABLE_PATHS = [
    f"hkan_layer_{index}/{name}"
    for index in (0, 1) for name in ("coef", "block_bias", "mix", "bias")
]


def _numpy(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _model(**overrides) -> HKAN:
    keras.utils.set_random_seed(0)
    config = dict(hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis="tanh",
                  slope=SLOPE, l2_block=0.01, seed=3)
    config.update(overrides)
    return HKAN(**config)


def _compiled(learning_rate: float = 1e-2, **overrides) -> HKAN:
    model = _model(**overrides)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model


def _relative_paths(model: HKAN, weights) -> list:
    return [w.path.split("/", 1)[1] for w in weights]


def _mse_loss(y32: np.ndarray):
    target = keras.ops.convert_to_tensor(y32[:, None])
    return lambda outputs: keras.ops.mean(keras.ops.square(outputs - target))


@pytest.fixture(scope="module")
def data32():
    x, y = make_data()
    return x.astype("float32"), y.astype("float32")


class TestStockFit:

    def test_fit_lowers_the_loss(self, data32):
        """Measured on this fixture: 0.720 before, 0.0070 after 40 epochs of
        Adam at 1e-2 (CPU and GPU 1 agree to 6 digits). The bound is a tenth
        of the start, 10 times the measured end."""
        x, y = data32
        model = _compiled()
        before = float(model.evaluate(x, y, batch_size=N_ROWS, verbose=0))
        history = model.fit(x, y, epochs=40, batch_size=32, shuffle=False, verbose=0)
        after = float(model.evaluate(x, y, batch_size=N_ROWS, verbose=0))
        assert np.isfinite(after) and after < 0.1 * before, (before, after)
        assert history.history["loss"][-1] < history.history["loss"][0]

    def test_the_initial_state_is_not_a_dead_point(self, data32):
        """Every trainable weight has a non-zero gradient before any update."""
        x, y = data32
        model = _model()
        model.build((None, N_IN))
        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=_mse_loss(y))
        assert sorted(path.split("/", 1)[1] for path in report) == sorted(TRAINABLE_PATHS)

    def test_gradients_reach_every_weight_after_one_real_step(self, data32):
        x, y = data32
        model = _compiled()
        model.fit(x, y, epochs=1, batch_size=N_ROWS, verbose=0)
        assert int(model.optimizer.iterations) == 1
        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=_mse_loss(y))
        assert len(report) == 8

    def test_one_step_moves_every_trainable_weight_and_no_center(self, data32):
        x, y = data32
        model = _compiled()
        model.build((None, N_IN))
        assert _relative_paths(model, model.trainable_weights) == TRAINABLE_PATHS
        before = {w.path: w.numpy().copy() for w in model.weights}
        model.fit(x, y, epochs=1, batch_size=N_ROWS, verbose=0)
        moved = {w.path.split("/", 1)[1] for w in model.weights
                 if not np.array_equal(w.numpy(), before[w.path])}
        assert moved == set(TRAINABLE_PATHS)

    def test_centers_are_bit_identical_after_fit(self, data32):
        x, y = data32
        model = _compiled()
        model.build((None, N_IN))
        before = [layer.centers.numpy().copy() for layer in model.hkan_layers]
        model.fit(x, y, epochs=3, batch_size=32, verbose=0)
        for layer, centers in zip(model.hkan_layers, before):
            np.testing.assert_array_equal(
                layer.centers.numpy(), centers,
                err_msg=f"{layer.name}.centers changed during fit")
            assert layer.centers.trainable is False
        assert _relative_paths(model, model.non_trainable_weights) == [
            "hkan_layer_0/centers", "hkan_layer_1/centers"]


class TestClosedFormThenFit:
    """The fine-tune starts AT the closed-form solution, not from scratch.

    The closed-form train MSE is ``mean((p64 - y64)^2)`` from the fit's own
    float64 predictions; Keras reports ``mean((p32 - y32)^2)``. They differ by
    the float32 forward difference ``d`` AND the float32 cast of the target
    ``c``: ``|difference| <= 2 * rmse * rms(d - c) + rms(d - c)^2``. On this
    fixture ``rmse = 0.131``, ``max|d| = 8.4e-7`` (CPU; 6.6e-7 on GPU 1) and
    ``max|c| = 5e-8``, so the bound is 2.3e-7. Measured
    ``|Keras - closed form|`` over the three tests below: 1.8e-9 on the CPU,
    7.5e-9 to 1.1e-8 on GPU 1. ``atol = 1e-7`` is 9 times the worst reading
    and about 50 float32 ulps of the loss (1.72e-2), with ``rtol=0``; a fit
    the compiled model did not see would report the initial loss, 0.720.
    """

    ATOL = 1e-7

    @pytest.fixture(scope="class")
    def closed(self, data32):
        x, y = data32
        model = _compiled(learning_rate=1e-4)
        diagnostics = model.fit_closed_form(x, y)
        closed_mse = float(np.mean(
            (diagnostics["train_predictions"] - y.astype(np.float64)) ** 2))
        return model, closed_mse

    def test_evaluate_reports_the_closed_form_mse(self, data32, closed):
        x, y = data32
        model, closed_mse = closed
        loss = float(model.evaluate(x, y, batch_size=N_ROWS, verbose=0))
        assert closed_mse > 1e-3, "the fixture interpolates; the comparison is vacuous"
        np.testing.assert_allclose(loss, closed_mse, rtol=0, atol=self.ATOL)

    def test_a_fit_after_the_evaluate_function_was_traced_is_honoured(self, data32, closed):
        """compile, evaluate (traces), fit_closed_form, evaluate again."""
        x, y = data32
        _, closed_mse = closed
        model = _compiled()
        unfitted = float(model.evaluate(x, y, batch_size=N_ROWS, verbose=0))
        model.fit_closed_form(x, y)
        fitted = float(model.evaluate(x, y, batch_size=N_ROWS, verbose=0))
        assert unfitted > 10 * closed_mse, "the unfitted model is already at the solution"
        np.testing.assert_allclose(fitted, closed_mse, rtol=0, atol=self.ATOL)

    def test_the_first_epoch_starts_from_the_closed_form_loss(self, data32, closed):
        """With one full batch per epoch the first reported loss is the loss
        BEFORE any update: it must be the closed-form MSE, not a worse one."""
        x, y = data32
        _, closed_mse = closed
        model = _compiled(learning_rate=1e-4)
        model.fit_closed_form(x, y)
        history = model.fit(x, y, epochs=2, batch_size=N_ROWS, shuffle=False, verbose=0)
        first = history.history["loss"][0]
        assert first <= closed_mse + self.ATOL, (first, closed_mse)
        np.testing.assert_allclose(first, closed_mse, rtol=0, atol=self.ATOL)


class TestInitializerInstances:

    @pytest.mark.parametrize("initializer", [
        "random_normal", keras.initializers.RandomNormal(stddev=0.3)])
    def test_two_same_shaped_layers_start_with_different_coef(self, initializer):
        """4 -> 4 -> 4 -> 1 with one ``num_basis``: layers 0 and 1 have the
        same ``coef`` shape and receive the model's ONE initializer instance.
        (A caller-supplied SEEDED instance gives every layer the same draw by
        ``clone_initializer``'s contract; that is not what this guards.)"""
        model = HKAN(hidden_units=(4, 4), num_basis=6, coef_initializer=initializer, seed=0)
        model.build((None, 4))
        first, second = (_numpy(layer.coef) for layer in model.hkan_layers[:2])
        assert first.shape == second.shape == (4, 4, 6)
        assert model.hkan_layers[0].coef_initializer is model.hkan_layers[1].coef_initializer
        assert not np.array_equal(first, second), (
            "two layers replayed one initializer draw: coef is identical")

    def test_a_given_mix_initializer_reaches_every_layer(self):
        model = HKAN(hidden_units=(HIDDEN,), mix_initializer=keras.initializers.Constant(0.5))
        model.build((None, N_IN))
        for layer in model.hkan_layers:
            np.testing.assert_array_equal(_numpy(layer.mix), np.float32(0.5))

    def test_default_mix_is_one_over_each_layers_n_in(self):
        model = _model()
        model.build((None, N_IN))
        np.testing.assert_array_equal(_numpy(model.hkan_layers[0].mix), np.float32(1 / N_IN))
        np.testing.assert_array_equal(_numpy(model.hkan_layers[1].mix), np.float32(1 / HIDDEN))


class TestInitializeCenters:
    """The gradient-only route to data-driven centers.

    3 -> 3 -> 3 -> 1 with equal widths ON PURPOSE (see ``TestDataCenters`` in
    ``test_closed_form.py``): with unequal widths a draw from the wrong layer's
    input is a shape error. ``coef`` starts at ``random_normal(stddev=2)`` and
    ``mix`` at 1, so that the activations of successive layers are far apart.
    """

    @pytest.fixture(scope="class")
    def drawn(self, data32):
        x, _ = data32
        keras.utils.set_random_seed(1)
        model = HKAN(
            hidden_units=(N_IN, N_IN), num_basis=6, basis="tanh", slope=SLOPE,
            centers=("data", "random", "data"), seed=0,
            coef_initializer=keras.initializers.RandomNormal(stddev=2.0),
            mix_initializer="ones")
        model.build((None, N_IN))
        before = [layer.centers.numpy().copy() for layer in model.hkan_layers]
        model.initialize_centers(x)
        hidden_1 = _numpy(model.hkan_layers[0](x))
        hidden_2 = _numpy(model.hkan_layers[1](hidden_1))
        return model, x, before, hidden_1, hidden_2

    @staticmethod
    def _members(centers: np.ndarray, pool: np.ndarray) -> bool:
        return all(
            np.isin(centers[:, p, :], pool[:, p]).all() for p in range(pool.shape[1]))

    def test_the_first_layer_draws_from_x(self, drawn):
        model, x, _, _, _ = drawn
        assert self._members(model.hkan_layers[0].centers.numpy(), x)

    def test_a_deeper_layer_draws_from_its_own_forward_input(self, drawn):
        model, x, _, hidden_1, hidden_2 = drawn
        centers = model.hkan_layers[2].centers.numpy()
        assert centers.shape == (1, N_IN, 6)
        # Exact membership: the draw and `hidden_2` are the same float32
        # forward pass of the same 96 rows in one batch.
        assert self._members(centers, hidden_2), (
            "the third layer's centers are not activations of the second layer")
        assert not self._members(centers, x) and not self._members(centers, hidden_1)

    def test_a_layer_in_another_mode_is_left_alone(self, drawn):
        model, _, before, _, _ = drawn
        np.testing.assert_array_equal(model.hkan_layers[1].centers.numpy(), before[1])
        assert not np.array_equal(model.hkan_layers[0].centers.numpy(), before[0])
        assert not np.array_equal(model.hkan_layers[2].centers.numpy(), before[2])

    def test_it_builds_an_unbuilt_model_and_checks_the_width(self, data32):
        x, _ = data32
        model = HKAN(hidden_units=(HIDDEN,), centers="data", seed=0)
        model.initialize_centers(x)
        assert model.built
        with pytest.raises(ValueError, match="columns"):
            model.initialize_centers(np.zeros((4, N_IN + 1), dtype="float32"))

    def test_rows_past_the_first_forward_batch_reach_a_deeper_layer(self):
        """300 rows: the first 256 (one forward batch) lie in ``[0, 0.1]``,
        the last 44 in ``[0.9, 1]``. Layer 0 is set by hand to the identity
        map, so layer 1's input IS ``x`` and its ``data`` centers are values
        of ``x``. 192 draws per column: the chance that none comes from the
        last 44 rows is ``(256 / 300) ** 192 = 6e-14``. If the propagation
        dropped the rows after the last full batch, every center would be at
        most 0.1."""
        rng = np.random.default_rng(4)
        x = np.concatenate([
            rng.uniform(0.0, 0.1, size=(256, N_IN)),
            rng.uniform(0.9, 1.0, size=(44, N_IN)),
        ]).astype("float32")
        model = HKAN(hidden_units=(N_IN,), num_basis=(1, 64), basis=("identity", "tanh"),
                     centers=("random", "data"), use_block_bias=False, use_bias=False, seed=0)
        model.build((None, N_IN))
        first = model.hkan_layers[0]
        first.centers.assign(np.zeros((N_IN, N_IN, 1), dtype="float32"))
        first.coef.assign(np.ones((N_IN, N_IN, 1), dtype="float32"))
        first.mix.assign(np.eye(N_IN, dtype="float32"))
        np.testing.assert_array_equal(_numpy(first(x)), x)

        model.initialize_centers(x)
        centers = model.hkan_layers[1].centers.numpy()
        assert centers.shape == (1, N_IN, 64)
        assert self._members(centers, x), "a center is not a row of the layer's input"
        from_the_tail = (centers > 0.5).sum()
        # Expected share 44 / 300 of 192 draws = 28; measured 30.
        assert from_the_tail >= 10, (
            f"{from_the_tail} of {centers.size} centers come from the rows past "
            "index 256: the propagation dropped the tail")
        assert (centers < 0.5).sum() >= 100

    def test_the_drawn_model_trains(self, data32):
        x, y = data32
        model = _compiled(centers="data")
        model.initialize_centers(x)
        centers = [layer.centers.numpy().copy() for layer in model.hkan_layers]
        before = float(model.evaluate(x, y, batch_size=N_ROWS, verbose=0))
        model.fit(x, y, epochs=20, batch_size=32, shuffle=False, verbose=0)
        assert float(model.evaluate(x, y, batch_size=N_ROWS, verbose=0)) < 0.5 * before
        for layer, value in zip(model.hkan_layers, centers):
            np.testing.assert_array_equal(layer.centers.numpy(), value)
