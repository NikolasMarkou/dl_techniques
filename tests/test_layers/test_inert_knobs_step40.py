"""Five knobs that were validated, stored and serialized while nothing read them.

Each class here pins ONE knob with ONE isolating mutation: change only that argument,
hold everything else (including the seed) fixed, and require the observable behaviour to
move. Before the step-40 fix every assertion in this module was RED — the settings were
accepted, round-tripped, and had no effect at all.

The `graph_energy_transformer` knob is the exception and is pinned the other way: its
`target_index` is *deliberately* ignored for a documented XLA reason, so the test asserts
the ignoring is exact rather than that the knob works.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.nbeats_blocks import GenericBlock
from dl_techniques.layers.ffn.kan_linear import KANLinear
from dl_techniques.layers.memory.ntm_interface import AddressingMode
from dl_techniques.layers.memory.baseline_ntm import NTMReadHead, NTMWriteHead
from dl_techniques.models.time_series.prism.model import PRISMModel


# ---------------------------------------------------------------------
# Knob 1 — GenericBlock.basis_initializer
# ---------------------------------------------------------------------


def _generic_block(**kwargs):
    return GenericBlock(
        units=8,
        thetas_dim=4,
        backcast_length=6,
        forecast_length=3,
        **kwargs,
    )


class TestGenericBlockBasisInitializerIsRead:
    """`basis_initializer` was stored and serialized while both Dense basis layers
    hardcoded `Orthogonal(gain=0.1)`."""

    @staticmethod
    def _basis_kernels(block):
        block.build((None, 6))
        return (
            keras.ops.convert_to_numpy(block.backcast_basis.kernel),
            keras.ops.convert_to_numpy(block.forecast_basis.kernel),
        )

    def test_a_zeros_initializer_actually_zeroes_the_basis_kernels(self):
        back, fore = self._basis_kernels(_generic_block(basis_initializer="zeros"))
        # RED before the fix: the hardcoded Orthogonal(gain=0.1) made these non-zero.
        assert np.all(back == 0.0)
        assert np.all(fore == 0.0)

    def test_a_ones_initializer_actually_fills_the_basis_kernels(self):
        back, fore = self._basis_kernels(_generic_block(basis_initializer="ones"))
        assert np.all(back == 1.0)
        assert np.all(fore == 1.0)

    def test_the_knob_reaches_the_block_output_not_only_the_weights(self):
        """Anti-vacuity: a zeroed basis must annihilate the block's two outputs."""
        block = _generic_block(basis_initializer="zeros")
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(0).normal(size=(2, 6)).astype("float32")
        )
        backcast, forecast = block(x)
        assert np.all(keras.ops.convert_to_numpy(backcast) == 0.0)
        assert np.all(keras.ops.convert_to_numpy(forecast) == 0.0)

    def test_the_default_is_still_the_historical_small_gain_orthogonal(self):
        """Deleting the knob's effect must not have changed the default behaviour."""
        back, _ = self._basis_kernels(_generic_block())
        assert not np.all(back == 0.0)
        # Orthogonal(gain=0.1): columns are 0.1-scaled orthonormal, so every column norm
        # is 0.1 (the kernel is (thetas_dim=4, 6) -> tall dimension is the output).
        gram = back @ back.T
        np.testing.assert_allclose(
            gram, 0.01 * np.eye(gram.shape[0]), atol=1e-5
        )

    def test_get_config_round_trips_the_initializer(self):
        block = _generic_block(basis_initializer="ones")
        config = block.get_config()
        clone = GenericBlock.from_config(config)
        back, _ = self._basis_kernels(clone)
        assert np.all(back == 1.0)

    def test_the_default_survives_a_get_config_round_trip(self):
        """`None` resolves to Orthogonal(gain=0.1) and the *resolved* value serializes."""
        block = _generic_block()
        clone = GenericBlock.from_config(block.get_config())
        back, _ = self._basis_kernels(clone)
        gram = back @ back.T
        np.testing.assert_allclose(gram, 0.01 * np.eye(gram.shape[0]), atol=1e-5)


# ---------------------------------------------------------------------
# Knob 2 — PRISMModel.DEFAULT_QUANTILES
# ---------------------------------------------------------------------


def _prism(**kwargs):
    return PRISMModel(
        context_len=8,
        forecast_len=4,
        num_features=2,
        hidden_dim=8,
        num_layers=1,
        tree_depth=1,
        num_wavelet_levels=1,
        router_hidden_dim=8,
        ffn_expansion=1,
        **kwargs,
    )


class TestPrismDefaultQuantilesIsUsed:
    """`DEFAULT_QUANTILES = [0.1, 0.5, 0.9]` was dead: the model produced
    `np.linspace(0, 1, 5)[1:-1] = [0.25, 0.5, 0.75]`."""

    def test_the_named_constant_is_the_default_at_its_own_length(self):
        model = _prism(use_quantile_head=True, num_quantiles=3)
        # RED before the fix: this was [0.25, 0.5, 0.75].
        assert model.quantile_levels == [0.1, 0.5, 0.9]
        assert model.quantile_levels == PRISMModel.DEFAULT_QUANTILES

    def test_the_docstring_claim_about_the_percentiles_now_holds(self):
        model = _prism(use_quantile_head=True)
        assert model.quantile_levels[0] == pytest.approx(0.1)
        assert model.quantile_levels[-1] == pytest.approx(0.9)

    def test_another_length_still_falls_back_to_evenly_spaced_levels(self):
        """The isolating mutation for the OTHER branch: only `num_quantiles` moves."""
        model = _prism(use_quantile_head=True, num_quantiles=5)
        np.testing.assert_allclose(
            model.quantile_levels, np.linspace(0, 1, 7)[1:-1]
        )

    def test_an_explicit_list_still_wins_over_the_constant(self):
        model = _prism(
            use_quantile_head=True, num_quantiles=3, quantile_levels=[0.2, 0.5, 0.8]
        )
        assert model.quantile_levels == [0.2, 0.5, 0.8]

    def test_the_point_forecast_head_still_has_no_levels(self):
        assert _prism(use_quantile_head=False).quantile_levels is None


# ---------------------------------------------------------------------
# Knob 3 — AddressingMode
# ---------------------------------------------------------------------


HEAD_KWARGS = dict(memory_size=5, memory_dim=4, shift_range=3)
CONTROLLER_DIM = 6


def _memory_state(batch=2, memory_size=5, memory_dim=4, seed=0):
    from dl_techniques.layers.memory.ntm_interface import MemoryState

    rng = np.random.default_rng(seed)
    return MemoryState(
        memory=keras.ops.convert_to_tensor(
            rng.normal(size=(batch, memory_size, memory_dim)).astype("float32")
        ),
        usage=keras.ops.zeros((batch, memory_size)),
    )


@pytest.mark.parametrize("head_cls", [NTMReadHead, NTMWriteHead])
class TestAddressingModeIsBranchedOn:
    """`addressing_mode` was threaded into both heads, stored and serialized while
    `compute_addressing` ran content -> gate -> shift -> sharpen unconditionally."""

    @staticmethod
    def _run(head, seed=0):
        rng = np.random.default_rng(seed)
        controller_output = keras.ops.convert_to_tensor(
            rng.normal(size=(2, CONTROLLER_DIM)).astype("float32")
        )
        head.build((None, CONTROLLER_DIM))
        # A deliberately non-uniform previous-weight vector: under HYBRID the gate
        # interpolates towards it, under CONTENT it must be ignored entirely.
        prev = keras.ops.convert_to_tensor(
            np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (2, 1)).astype("float32")
        )
        weights, state = head.compute_addressing(
            controller_output, _memory_state(), prev
        )
        return keras.ops.convert_to_numpy(weights), state

    def test_content_and_hybrid_do_not_produce_the_same_weights(self, head_cls):
        keras.utils.set_random_seed(11)
        content = head_cls(addressing_mode=AddressingMode.CONTENT, **HEAD_KWARGS)
        keras.utils.set_random_seed(11)
        hybrid = head_cls(addressing_mode=AddressingMode.HYBRID, **HEAD_KWARGS)

        w_content, _ = self._run(content)
        w_hybrid, _ = self._run(hybrid)
        # RED before the fix: byte-identical, because the mode was never read.
        assert not np.allclose(w_content, w_hybrid, atol=1e-6)

    def test_content_mode_ignores_the_previous_weights(self, head_cls):
        """The defining property of content-only addressing."""
        keras.utils.set_random_seed(11)
        head = head_cls(addressing_mode=AddressingMode.CONTENT, **HEAD_KWARGS)
        head.build((None, CONTROLLER_DIM))
        rng = np.random.default_rng(3)
        controller_output = keras.ops.convert_to_tensor(
            rng.normal(size=(2, CONTROLLER_DIM)).astype("float32")
        )
        state = _memory_state()

        prev_a = keras.ops.convert_to_tensor(
            np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (2, 1)).astype("float32")
        )
        prev_b = keras.ops.convert_to_tensor(
            np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (2, 1)).astype("float32")
        )
        w_a, _ = head.compute_addressing(controller_output, state, prev_a)
        w_b, _ = head.compute_addressing(controller_output, state, prev_b)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(w_a),
            keras.ops.convert_to_numpy(w_b),
            atol=1e-7,
        )

    def test_hybrid_mode_does_read_the_previous_weights(self, head_cls):
        """Anti-vacuity for the test above: the same probe must MOVE under HYBRID."""
        keras.utils.set_random_seed(11)
        head = head_cls(addressing_mode=AddressingMode.HYBRID, **HEAD_KWARGS)
        head.build((None, CONTROLLER_DIM))
        rng = np.random.default_rng(3)
        controller_output = keras.ops.convert_to_tensor(
            rng.normal(size=(2, CONTROLLER_DIM)).astype("float32")
        )
        state = _memory_state()
        prev_a = keras.ops.convert_to_tensor(
            np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (2, 1)).astype("float32")
        )
        prev_b = keras.ops.convert_to_tensor(
            np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (2, 1)).astype("float32")
        )
        w_a, _ = head.compute_addressing(controller_output, state, prev_a)
        w_b, _ = head.compute_addressing(controller_output, state, prev_b)
        assert not np.allclose(
            keras.ops.convert_to_numpy(w_a),
            keras.ops.convert_to_numpy(w_b),
            atol=1e-6,
        )

    def test_content_mode_has_strictly_fewer_parameters(self, head_cls):
        content = head_cls(addressing_mode=AddressingMode.CONTENT, **HEAD_KWARGS)
        hybrid = head_cls(addressing_mode=AddressingMode.HYBRID, **HEAD_KWARGS)
        content.build((None, CONTROLLER_DIM))
        hybrid.build((None, CONTROLLER_DIM))
        assert content.count_params() < hybrid.count_params()

    def test_content_mode_leaves_the_location_state_fields_empty(self, head_cls):
        head = head_cls(addressing_mode=AddressingMode.CONTENT, **HEAD_KWARGS)
        _, state = self._run(head)
        assert state.gate is None
        assert state.shift is None
        assert state.gamma is None
        assert state.key is not None

    def test_the_weights_are_still_a_distribution_in_content_mode(self, head_cls):
        head = head_cls(addressing_mode=AddressingMode.CONTENT, **HEAD_KWARGS)
        weights, _ = self._run(head)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)
        assert np.all(weights >= 0.0)


# ---------------------------------------------------------------------
# Knob 4 — KANLinear.update_grid_from_samples
# ---------------------------------------------------------------------


class TestKanGridUpdateKeepsTheInteriorQuantiles:
    """Both docstrings promised quantile matching; the code kept only the min and the
    max and rebuilt a UNIFORM knot sequence."""

    @staticmethod
    def _skewed_batch(n=512, features=3, seed=7):
        """Heavily right-skewed data: the quantile grid and the uniform grid diverge."""
        rng = np.random.default_rng(seed)
        return (rng.random((n, features)) ** 4 * 8.0 - 4.0).astype("float32")

    @staticmethod
    def _layer():
        layer = KANLinear(features=2, grid_size=8, spline_order=3)
        layer.build((None, 3))
        return layer

    def test_the_adapted_grid_is_not_uniform_on_skewed_data(self):
        layer = self._layer()
        layer.update_grid_from_samples(self._skewed_batch())
        grid = keras.ops.convert_to_numpy(layer.grid)
        interior = grid[layer.spline_order: -layer.spline_order]
        spacings = np.diff(interior)
        # RED before the fix: every spacing was identical to ~1e-6.
        assert spacings.max() / spacings.min() > 2.0

    def test_the_interior_knots_match_the_empirical_quantiles(self):
        """The claim the docstring actually makes, checked against numpy."""
        x = self._skewed_batch()
        layer = self._layer()
        layer.update_grid_from_samples(x)
        grid = keras.ops.convert_to_numpy(layer.grid)
        interior = grid[layer.spline_order: -layer.spline_order]

        x_sorted = np.sort(x, axis=0)
        idx = np.linspace(0, x.shape[0] - 1, layer.grid_size + 1).astype("int32")
        expected = x_sorted[idx].mean(axis=1)
        np.testing.assert_allclose(interior, expected, rtol=1e-5, atol=1e-5)

    def test_the_knot_sequence_stays_monotone(self):
        layer = self._layer()
        layer.update_grid_from_samples(self._skewed_batch())
        grid = keras.ops.convert_to_numpy(layer.grid)
        assert np.all(np.diff(grid) > 0.0)

    def test_uniform_data_would_not_have_caught_this(self):
        """Why the skew matters: on uniform x the two spellings agree.

        Recorded so nobody 'simplifies' the fixture back to `rng.normal`.
        """
        rng = np.random.default_rng(1)
        x = rng.random((4096, 3)).astype("float32")
        layer = self._layer()
        layer.update_grid_from_samples(x)
        grid = keras.ops.convert_to_numpy(layer.grid)
        interior = grid[layer.spline_order: -layer.spline_order]
        spacings = np.diff(interior)
        assert spacings.max() / spacings.min() < 1.15

    def test_the_grid_change_reaches_the_layer_output(self):
        """Anti-vacuity: the adapted knots must move the forward pass."""
        x = self._skewed_batch()
        layer = self._layer()
        probe = keras.ops.convert_to_tensor(x[:4])
        before = keras.ops.convert_to_numpy(layer(probe))
        layer.update_grid_from_samples(x)
        after = keras.ops.convert_to_numpy(layer(probe))
        assert not np.allclose(before, after, atol=1e-6)


# ---------------------------------------------------------------------
# Knob 5 — GraphAnomalyDetector.target_index (deliberately inert)
# ---------------------------------------------------------------------


class TestGraphAnomalyTargetIndexIsExactlyIgnored:
    """`target_index` is accepted and NOT read — a documented XLA choice (D-003).

    This knob is not made to work; it is pinned as inert so the behaviour is a contract
    rather than an accident, and so the docstring claim is executable.
    """

    @staticmethod
    def _model_and_inputs(batch=2, n=6, f=4):
        from dl_techniques.models.graph_energy_transformer.model import (
            create_graph_anomaly_detector,
        )

        keras.utils.set_random_seed(5)
        model = create_graph_anomaly_detector(
            node_feature_dim=f,
            embed_dim=8,
            num_heads=2,
            head_dim=4,
            hopfield_dim=8,
            mlp_hidden_dim=8,
            num_steps=2,
        )
        rng = np.random.default_rng(2)
        inputs = {
            "node_features": keras.ops.convert_to_tensor(
                rng.normal(size=(batch, n, f)).astype("float32")
            ),
            "adjacency": keras.ops.convert_to_tensor(
                np.ones((batch, n, n), dtype="float32")
            ),
            "node_mask": keras.ops.convert_to_tensor(
                np.ones((batch, n), dtype="float32")
            ),
        }
        return model, inputs

    def test_a_nonzero_target_index_changes_nothing(self):
        model, inputs = self._model_and_inputs()
        zeros = dict(inputs)
        zeros["target_index"] = keras.ops.convert_to_tensor(
            np.zeros((2,), dtype="int32")
        )
        threes = dict(inputs)
        threes["target_index"] = keras.ops.convert_to_tensor(
            np.full((2,), 3, dtype="int32")
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(model(zeros, training=False)),
            keras.ops.convert_to_numpy(model(threes, training=False)),
            atol=0.0,
        )

    def test_the_readout_really_is_node_zero(self):
        """Anti-vacuity: perturbing node 0 must move the logit, and the model must be
        sensitive to the node dimension at all."""
        model, inputs = self._model_and_inputs()
        base = keras.ops.convert_to_numpy(model(inputs, training=False))

        nf = np.array(keras.ops.convert_to_numpy(inputs["node_features"]))
        bumped = nf.copy()
        bumped[:, 0, :] += 5.0
        moved = dict(inputs)
        moved["node_features"] = keras.ops.convert_to_tensor(bumped)
        assert not np.allclose(
            base, keras.ops.convert_to_numpy(model(moved, training=False)), atol=1e-5
        )
