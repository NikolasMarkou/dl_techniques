"""Test suite for BitLinear (BitNet b1.58 style quantization-aware linear layer).

Covers initialization and stored config, constructor validation (including the bit
specification, which defines the layer), forward shape over rank-2 and rank-3 inputs,
``compute_output_shape`` pre- and post-build, training-mode behaviour, gradient flow,
a ``.keras`` VALUE round trip, and the behavioural pins below. Every pin carries the
number measured for it, on CPU / float32 / keras 3.x / 2026-09-02, together with the
reading the injected defect produced.

  1. test_the_forward_value_does_not_depend_on_ste_lambda
     HEAD max|y(1.0) - y(2.5)| = 0.0; with the STE written as
     ``stop_gradient(q - t) + t * lambda`` the same reading is 8.726056.
  2. test_ste_lambda_scales_the_input_gradient        (the "something changed" twin of 1)
  3. test_the_output_does_not_depend_on_the_rest_of_the_batch
     HEAD max|full-batch row - row alone| = 0.0; with gamma reduced over the batch
     axis and the rescale applied before the matmul the reading is 6.3086e-03.
  4. test_the_rows_of_the_batch_are_not_all_equal      (anti-vacuity twin of 3)
  5. test_the_input_gradient_does_not_carry_a_gamma_term
     HEAD spread across rows = 1.192e-07 and across two unrelated inputs = 2.384e-07;
     dropping ``stop_gradient`` from the gamma statistic gives 8.040e-03 and 8.441e-03,
     a factor of ~6.7e4 above the head reading.
  6. test_one_bit_weights_are_binary / test_ternary_bit_specs_reach_zero
     bits=1 -> {-1, +1}; bits=1.58 and bits=2 -> {-1, 0, +1}. Removing the sign path
     turns the bits=1 reading into {-1, 0, +1}.
  7. test_an_explicit_range_survives_a_json_round_trip
     A tuple config is stored as a JSON list; before the fix ``from_config`` raised
     ``ValueError: Invalid bit specification: [-3.0, 3.0]``.
  8. test_stochastic_rounding_is_deterministic_at_inference / ..._varies_in_training
     HEAD inference run-to-run delta = 0.0; ignoring ``training`` gives 0.768210.
  9. test_weight_per_channel_changes_the_output       (dead-knob pin, §12.5)
 10. test_mixed_float16_stays_finite[0.0001]
     The scale is applied as ``x / gamma * Q_max``. Materializing ``Q_max / gamma``
     instead reads ``inf`` under mixed_float16 at input magnitude 1e-4 (gamma 1e-4,
     scale 1.27e6, float16 max 65504); this form reads a finite 1.025e-05. The
     ``magnitude=1.0`` arm is the control: there both forms are finite.

RED proofs, 2026-09-02. Each historical defect was re-injected over the shipped class
and the whole file re-run; the head run is 83 passed. Every injection reddens at least
one guard, and no injection leaves the file green:

  | injected defect                                          | tests reddened |
  |----------------------------------------------------------|----------------|
  | STE written as ``stop_gradient(q - t) + t * lambda``       | 5 |
  | gamma reduced over the batch axis, rescale before matmul   | 3 |
  | ``stop_gradient`` dropped from the gamma statistic         | 3 |
  | binary ``sign`` path removed (bits=1 falls back to round)  | 1 |
  | ``training`` ignored by stochastic rounding                | 1 |
  | pre-fix ``_bits_to_range`` (no validation, no list form)   | 12 |
  | ``weight_per_channel`` read then discarded in ``call``     | 2 |
  | scale materialized as ``Q_max / gamma``                    | 2 |
  | ``activation`` stored but never applied in ``call``        | 1 |
  | ``use_bias`` back to the pre-flip Dense default of True    | 2 |
"""

import os
import json

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.bitlinear_layer import BitLinear

# ---------------------------------------------------------------------

B, T, D, U = 4, 3, 6, 5

# Head readings for the pins that assert "no difference"; the injected defect
# sits three to five orders of magnitude above each of these (see module docstring).
EXACT = 0.0
GRADIENT_INVARIANCE_ATOL = 1e-5

# ---------------------------------------------------------------------


@pytest.fixture
def sample_input() -> np.ndarray:
    return np.random.default_rng(1234).standard_normal((B, D)).astype("float32")


@pytest.fixture
def other_input() -> np.ndarray:
    return np.random.default_rng(4321).standard_normal((B, D)).astype("float32")


@pytest.fixture
def basic_config() -> dict:
    return {"units": U}


def _layer(**kwargs) -> BitLinear:
    """Build a BitLinear on ``(None, D)`` from a fixed seed.

    ``use_bias`` is forced on here so the pins that match two layers weight for
    weight keep exercising the bias path; the constructor's own default is
    ``False`` and is pinned separately by
    ``TestBitLinearInitialization.test_the_bias_is_dropped_by_default``.
    """
    keras.utils.set_random_seed(1234)
    layer = BitLinear(**{"units": U, "use_bias": True, **kwargs})
    layer.build((None, D))
    return layer


def _input_gradient(layer: BitLinear, data: np.ndarray) -> np.ndarray:
    variable = tf.Variable(data)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(layer(variable))
    return tape.gradient(loss, variable).numpy()


def _weight_levels(layer: BitLinear) -> list:
    """The distinct values the quantized kernel actually takes."""
    kernel = keras.ops.cast(layer.kernel, "float32")
    gamma = layer._compute_gamma(kernel, layer.weight_scale_method, None)
    target_max = max(abs(v) for v in layer.weight_range)
    quantized = layer._quantize_tensor(
        kernel / gamma * target_max, layer.weight_range, layer._weight_is_binary
    )
    return sorted(set(keras.ops.convert_to_numpy(quantized).ravel().tolist()))


# ---------------------------------------------------------------------


class TestBitLinearInitialization:

    def test_construction_stores_config_without_building(self, basic_config):
        layer = BitLinear(**basic_config)
        assert layer.units == U
        assert layer.weight_scale_method == "abs_mean"
        assert layer.activation_scale_method == "abs_max"
        assert layer.weight_range == (-1.0, 1.0)
        assert layer.activation_range == (-127.0, 127.0)
        assert layer.supports_masking is True
        assert not layer.built

    def test_the_bias_is_dropped_by_default(self):
        """BitNet drops the bias in its quantized projections; keras.layers.Dense
        does not. This layer follows BitNet, so the default differs from Dense."""
        default = BitLinear(units=U)
        assert default.use_bias is False
        default.build((None, D))
        assert default.bias is None
        assert len(default.trainable_variables) == 1

        with_bias = BitLinear(units=U, use_bias=True)
        with_bias.build((None, D))
        assert with_bias.bias is not None
        assert len(with_bias.trainable_variables) == 2

    def test_the_sub_layer_is_created_in_init(self):
        assert BitLinear(units=U, use_input_norm=True).input_norm is not None
        assert BitLinear(units=U, use_input_norm=False).input_norm is None

    def test_the_norm_epsilon_reaches_the_sub_layer(self):
        layer = BitLinear(units=U, use_input_norm=True, norm_epsilon=3e-4)
        assert layer.input_norm.epsilon == pytest.approx(3e-4)

    def test_regularizers_are_resolved_like_initializers(self):
        layer = BitLinear(units=U, kernel_regularizer="l2", bias_regularizer="l1")
        assert isinstance(layer.kernel_regularizer, keras.regularizers.Regularizer)
        assert isinstance(layer.bias_regularizer, keras.regularizers.Regularizer)

    def test_the_kernel_constraint_is_applied_to_the_kernel(self):
        layer = _layer(kernel_constraint=keras.constraints.MaxNorm(1.0))
        assert layer.kernel.constraint is not None


class TestBitLinearValidation:

    @pytest.mark.parametrize("bad, match", [
        ({"units": 0}, "units must be a positive integer"),
        ({"units": -1}, "units must be a positive integer"),
        ({"units": True}, "units must be a positive integer"),
        ({"units": U, "weight_scale_method": "bogus"}, "weight_scale_method"),
        ({"units": U, "activation_scale_method": "bogus"}, "activation_scale_method"),
        ({"units": U, "quantization_method": "bogus"}, "quantization_method"),
        ({"units": U, "ste_lambda": 0}, "ste_lambda must be positive"),
        ({"units": U, "epsilon": 0}, "epsilon must be positive"),
        ({"units": U, "norm_epsilon": 0}, "norm_epsilon must be positive"),
        ({"units": U, "weight_bits": 0}, "weight_bits must be positive"),
        ({"units": U, "weight_bits": -4}, "weight_bits must be positive"),
        ({"units": U, "weight_bits": True}, "Invalid bit specification"),
        ({"units": U, "weight_bits": "8"}, "Invalid bit specification"),
        ({"units": U, "weight_bits": (1.0,)}, "exactly 2 elements"),
        ({"units": U, "weight_bits": (-1.0, -2.0, 3.0)}, "exactly 2 elements"),
        ({"units": U, "weight_bits": (5.0, -5.0)}, "min < max"),
        ({"units": U, "weight_bits": (2.0, 2.0)}, "min < max"),
        ({"units": U, "activation_bits": -8}, "activation_bits must be positive"),
    ])
    def test_invalid_args_raise(self, bad, match):
        with pytest.raises(ValueError, match=match):
            BitLinear(**bad)

    def test_an_undefined_input_dimension_raises_at_build(self):
        with pytest.raises(ValueError, match="last dimension"):
            BitLinear(units=U).build((None, None))


class TestBitLinearForward:

    @pytest.mark.parametrize("use_input_norm", [False, True])
    @pytest.mark.parametrize("units", [U, D])
    def test_forward_pass(self, sample_input, use_input_norm, units):
        layer = BitLinear(units=units, use_input_norm=use_input_norm)
        out = layer(sample_input)
        assert tuple(out.shape) == (B, units)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("weight_scale_method", ["abs_max", "abs_mean", "abs_median"])
    @pytest.mark.parametrize("activation_scale_method", ["abs_max", "abs_mean", "abs_median"])
    def test_every_scale_method_pair_runs_on_a_rank_3_input(
        self, weight_scale_method, activation_scale_method
    ):
        data = np.random.default_rng(1234).standard_normal((B, T, D)).astype("float32")
        layer = BitLinear(
            units=U,
            weight_scale_method=weight_scale_method,
            activation_scale_method=activation_scale_method,
        )
        out = keras.ops.convert_to_numpy(layer(data))
        assert out.shape == (B, T, U)
        assert np.all(np.isfinite(out))

    def test_compute_output_shape_before_build(self):
        layer = BitLinear(units=U)
        assert layer.compute_output_shape((B, D)) == (B, U)
        assert layer.compute_output_shape((B, T, D)) == (B, T, U)
        assert not layer.built

    def test_compute_output_shape_matches_call(self, sample_input):
        layer = BitLinear(units=U)
        out = layer(sample_input)
        assert tuple(out.shape) == tuple(layer.compute_output_shape(sample_input.shape))

    def test_a_zero_input_is_finite(self):
        """The gamma floor has to survive an all-zero token."""
        layer = BitLinear(units=U)
        out = keras.ops.convert_to_numpy(layer(np.zeros((B, D), "float32")))
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("use_bias, expected", [(False, 1), (True, 2)])
    def test_gradients_reach_every_trainable_weight(
        self, sample_input, use_bias, expected
    ):
        layer = _layer(use_bias=use_bias)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(sample_input))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == expected
        for variable, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, variable.name
            assert np.abs(grad.numpy()).max() > 0.0, variable.name


class TestBitLinearStraightThroughEstimator:
    """The STE must scale the gradient only; see pins 1 and 2."""

    def test_the_forward_value_does_not_depend_on_ste_lambda(self, sample_input):
        """HEAD 0.0; ``stop_gradient(q - t) + t * lambda`` reads 8.726056."""
        reference = _layer(ste_lambda=1.0)
        scaled = _layer(ste_lambda=2.5)
        scaled.kernel.assign(keras.ops.convert_to_numpy(reference.kernel))
        scaled.bias.assign(keras.ops.convert_to_numpy(reference.bias))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(reference(sample_input)),
            keras.ops.convert_to_numpy(scaled(sample_input)),
            atol=EXACT, rtol=0,
        )

    def test_ste_lambda_scales_the_input_gradient(self, sample_input):
        """The "something changed" twin: lambda=3 gives exactly 3x the gradient."""
        reference = _layer(ste_lambda=1.0, ste_clip_gradient=False, use_bias=False)
        scaled = _layer(ste_lambda=3.0, ste_clip_gradient=False, use_bias=False)
        scaled.kernel.assign(keras.ops.convert_to_numpy(reference.kernel))
        np.testing.assert_allclose(
            _input_gradient(scaled, sample_input),
            3.0 * _input_gradient(reference, sample_input),
            atol=1e-5, rtol=0,
        )

    def test_the_clipped_ste_zeroes_the_gradient_of_a_saturated_input(self):
        """A value far outside the representable range gets no pass-through."""
        layer = _layer(
            ste_clip_gradient=True, activation_bits=(-1.0, 1.0),
            activation_scale_method="abs_mean", use_bias=False,
        )
        # A kernel of ones quantizes to +1 everywhere, so the pass-through gradient
        # is U per element and cannot cancel to zero by accident. abs_mean scaling
        # pushes the outlier to |x| / mean|x| = 5.71, outside the range, while the
        # rest land at 0.057 inside it -- so exactly one element is masked. (Under
        # abs_max nothing ever saturates: the largest element maps to exactly Q_max.)
        layer.kernel.assign(np.ones((D, U), "float32"))
        data = np.array([[10.0, 0.1, 0.1, 0.1, 0.1, 0.1]], "float32")
        grad = _input_gradient(layer, data)
        assert grad[0, 0] == pytest.approx(0.0, abs=EXACT)
        assert np.abs(grad[0, 1:]).max() > 0.0

    def test_the_unclipped_ste_passes_the_gradient_of_a_saturated_input(self):
        """Anti-vacuity twin: the mask is what zeroes it, not the layer being dead."""
        layer = _layer(
            ste_clip_gradient=False, activation_bits=(-1.0, 1.0),
            activation_scale_method="abs_mean", use_bias=False,
        )
        layer.kernel.assign(np.ones((D, U), "float32"))
        data = np.array([[10.0, 0.1, 0.1, 0.1, 0.1, 0.1]], "float32")
        assert abs(_input_gradient(layer, data)[0, 0]) > 0.0

    def test_the_input_gradient_does_not_carry_a_gamma_term(
        self, sample_input, other_input
    ):
        """gamma is a constant of the backward pass.

        With gamma held constant the input gradient is ``ste_lambda * W_dequant``,
        the same for every row and for every input. HEAD spread across rows
        1.192e-07 / across inputs 2.384e-07; without ``stop_gradient`` on gamma the
        two readings become 8.040e-03 and 8.441e-03.
        """
        layer = _layer(
            ste_clip_gradient=False, use_bias=False, activation_scale_method="abs_max"
        )
        grad = _input_gradient(layer, sample_input)
        assert np.abs(grad).max() > 0.0, "vacuous: the gradient is dead"
        np.testing.assert_allclose(
            grad, np.repeat(grad[0:1], B, axis=0),
            atol=GRADIENT_INVARIANCE_ATOL, rtol=0,
        )
        np.testing.assert_allclose(
            grad, _input_gradient(layer, other_input),
            atol=GRADIENT_INVARIANCE_ATOL, rtol=0,
        )


class TestBitLinearBatchIndependence:
    """gamma never reduces over the batch axis; see pins 3 and 4."""

    def test_the_output_does_not_depend_on_the_rest_of_the_batch(self, sample_input):
        """HEAD 0.0; the pre-fix batch-axis reduction reads 6.3086e-03."""
        layer = _layer()
        full = keras.ops.convert_to_numpy(layer(sample_input))
        alone = np.concatenate([
            keras.ops.convert_to_numpy(layer(sample_input[i:i + 1])) for i in range(B)
        ])
        np.testing.assert_allclose(full, alone, atol=EXACT, rtol=0)

    def test_the_rows_of_the_batch_are_not_all_equal(self, sample_input):
        """Anti-vacuity twin of the pin above: a constant output would also pass it."""
        full = keras.ops.convert_to_numpy(_layer()(sample_input))
        assert np.abs(full - full[0:1]).max() > 1e-3

    def test_a_batch_of_one_does_not_saturate_every_element(self):
        """The pre-fix per-feature-across-batch scale mapped every element to +/-127."""
        layer = _layer(activation_scale_method="abs_max")
        data = np.random.default_rng(0).standard_normal((1, D)).astype("float32")
        gamma = layer._compute_gamma(keras.ops.convert_to_tensor(data), "abs_max", -1)
        scaled = keras.ops.convert_to_numpy(
            keras.ops.convert_to_tensor(data) / gamma * 127.0
        )
        assert np.sum(np.abs(np.abs(scaled) - 127.0) < 1e-3) == 1


class TestBitLinearBitSpecification:

    def test_one_bit_weights_are_binary(self):
        """bits=1 -> {-1, +1}. Removing the sign path reads {-1, 0, +1}."""
        keras.utils.set_random_seed(1234)
        layer = BitLinear(units=U, weight_bits=1)
        layer.build((None, 256))
        assert layer._weight_is_binary is True
        assert _weight_levels(layer) == [-1.0, 1.0]

    @pytest.mark.parametrize("bits", [1.58, 2])
    def test_ternary_bit_specs_reach_zero(self, bits):
        layer = BitLinear(units=U, weight_bits=bits)
        layer.build((None, 256))
        assert layer._weight_is_binary is False
        assert _weight_levels(layer) == [-1.0, 0.0, 1.0]

    @pytest.mark.parametrize("bits, expected", [
        (1, (-1.0, 1.0)), (1.58, (-1.0, 1.0)), (2, (-1.0, 1.0)),
        (3, (-3.0, 3.0)), (4, (-7.0, 7.0)), (8, (-127.0, 127.0)),
        ((-3.0, 5.0), (-3.0, 5.0)), ([-3.0, 5.0], (-3.0, 5.0)),
    ])
    def test_the_bit_to_range_table(self, bits, expected):
        assert BitLinear(units=U, weight_bits=bits).weight_range == expected

    def test_an_explicit_range_is_never_taken_as_binary(self):
        assert BitLinear(units=U, weight_bits=(-1.0, 1.0))._weight_is_binary is False


class TestBitLinearTrainingMode:

    def test_stochastic_rounding_is_deterministic_at_inference(self, sample_input):
        """HEAD 0.0; ignoring ``training`` reads 0.768210."""
        layer = _layer(quantization_method="stochastic", seed=7)
        first = keras.ops.convert_to_numpy(layer(sample_input, training=False))
        second = keras.ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(first, second, atol=EXACT, rtol=0)

    def test_stochastic_rounding_varies_in_training(self, sample_input):
        """Anti-vacuity twin: the knob is not simply dead."""
        layer = _layer(quantization_method="stochastic", seed=7)
        first = keras.ops.convert_to_numpy(layer(sample_input, training=True))
        second = keras.ops.convert_to_numpy(layer(sample_input, training=True))
        assert np.abs(first - second).max() > 0.0

    def test_round_clip_is_deterministic_in_training(self, sample_input):
        layer = _layer(quantization_method="round_clip")
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer(sample_input, training=True)),
            keras.ops.convert_to_numpy(layer(sample_input, training=True)),
            atol=EXACT, rtol=0,
        )


class TestBitLinearKnobsAreLive:
    """§12.5: every constructor parameter varies a measured output."""

    def test_weight_per_channel_changes_the_output(self, sample_input):
        per_tensor = _layer(weight_per_channel=False, weight_bits=4)
        per_channel = _layer(weight_per_channel=True, weight_bits=4)
        per_channel.kernel.assign(keras.ops.convert_to_numpy(per_tensor.kernel))
        per_channel.bias.assign(keras.ops.convert_to_numpy(per_tensor.bias))
        assert np.abs(
            keras.ops.convert_to_numpy(per_channel(sample_input))
            - keras.ops.convert_to_numpy(per_tensor(sample_input))
        ).max() > 1e-3

    @pytest.mark.parametrize("method", ["abs_mean", "abs_median"])
    def test_the_weight_scale_method_changes_the_output(self, sample_input, method):
        reference = _layer(weight_scale_method="abs_max", weight_bits=4)
        other = _layer(weight_scale_method=method, weight_bits=4)
        other.kernel.assign(keras.ops.convert_to_numpy(reference.kernel))
        other.bias.assign(keras.ops.convert_to_numpy(reference.bias))
        assert np.abs(
            keras.ops.convert_to_numpy(other(sample_input))
            - keras.ops.convert_to_numpy(reference(sample_input))
        ).max() > 1e-3

    def test_use_input_norm_changes_the_output(self, sample_input):
        with_norm = _layer(use_input_norm=True)
        without = _layer(use_input_norm=False)
        with_norm.kernel.assign(keras.ops.convert_to_numpy(without.kernel))
        with_norm.bias.assign(keras.ops.convert_to_numpy(without.bias))
        assert np.abs(
            keras.ops.convert_to_numpy(with_norm(sample_input))
            - keras.ops.convert_to_numpy(without(sample_input))
        ).max() > 1e-3

    def test_the_activation_is_applied_to_the_output(self, sample_input):
        """`activation` is the last step, so it must equal applying it by hand."""
        linear = _layer()
        activated = _layer(activation="relu")
        activated.kernel.assign(keras.ops.convert_to_numpy(linear.kernel))
        activated.bias.assign(keras.ops.convert_to_numpy(linear.bias))
        y_linear = keras.ops.convert_to_numpy(linear(sample_input))
        y_relu = keras.ops.convert_to_numpy(activated(sample_input))
        assert (y_linear < 0).any(), "vacuous: nothing for relu to clip"
        assert np.abs(y_linear - y_relu).max() > 1e-3
        np.testing.assert_allclose(np.maximum(y_linear, 0.0), y_relu, atol=EXACT, rtol=0)

    def test_the_default_activation_is_the_identity(self, sample_input):
        """`keras.activations.get(None)` resolves to `linear`, not to `None`."""
        default = _layer()
        explicit = _layer(activation="linear")
        explicit.kernel.assign(keras.ops.convert_to_numpy(default.kernel))
        explicit.bias.assign(keras.ops.convert_to_numpy(default.bias))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(default(sample_input)),
            keras.ops.convert_to_numpy(explicit(sample_input)),
            atol=EXACT, rtol=0,
        )

    def test_activation_bits_change_the_output(self, sample_input):
        coarse = _layer(activation_bits=2)
        fine = _layer(activation_bits=8)
        coarse.kernel.assign(keras.ops.convert_to_numpy(fine.kernel))
        coarse.bias.assign(keras.ops.convert_to_numpy(fine.bias))
        assert np.abs(
            keras.ops.convert_to_numpy(coarse(sample_input))
            - keras.ops.convert_to_numpy(fine(sample_input))
        ).max() > 1e-3


class TestBitLinearSerialization:

    @pytest.mark.parametrize("use_input_norm", [False, True])
    def test_serialization_round_trip(self, sample_input, use_input_norm, tmp_path):
        inp = keras.Input(shape=(D,))
        out = BitLinear(units=U, use_input_norm=use_input_norm, name="bl")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample_input)

        path = os.path.join(tmp_path, "bl.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"BitLinear": BitLinear}
        )
        y1 = loaded(sample_input)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_a_fully_configured_layer_round_trips_on_values(self, sample_input, tmp_path):
        inp = keras.Input(shape=(D,))
        out = BitLinear(
            units=U,
            weight_bits=(-3.0, 3.0),
            activation_bits=4,
            weight_scale_method="abs_median",
            activation_scale_method="abs_mean",
            weight_per_channel=True,
            quantization_method="stochastic",
            activation="gelu",
            use_input_norm=True,
            ste_lambda=2.0,
            ste_clip_gradient=False,
            epsilon=1e-4,
            norm_epsilon=1e-5,
            seed=11,
            kernel_regularizer="l2",
            bias_regularizer="l1",
            kernel_constraint=keras.constraints.MaxNorm(1.0),
            name="bl",
        )(inp)
        model = keras.Model(inp, out)
        y0 = keras.ops.convert_to_numpy(model(sample_input, training=False))

        path = os.path.join(tmp_path, "full.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"BitLinear": BitLinear})
        y1 = keras.ops.convert_to_numpy(loaded(sample_input, training=False))
        np.testing.assert_allclose(y0, y1, rtol=1e-5, atol=1e-5)

        restored = loaded.get_layer("bl")
        assert restored.weight_range == (-3.0, 3.0)
        assert restored.weight_per_channel is True
        assert keras.activations.serialize(restored.activation) == "gelu"
        assert restored.ste_clip_gradient is False
        assert restored.seed == 11
        assert restored.norm_epsilon == pytest.approx(1e-5)
        assert isinstance(restored.kernel_constraint, keras.constraints.MaxNorm)

    def test_get_config_covers_every_constructor_argument(self):
        import inspect

        config = BitLinear(units=U).get_config()
        parameters = [
            name for name in inspect.signature(BitLinear.__init__).parameters
            if name not in ("self", "kwargs")
        ]
        missing = [name for name in parameters if name not in config]
        assert not missing, f"get_config omits {missing}"

    def test_an_explicit_range_survives_a_json_round_trip(self):
        """Before the fix ``from_config`` raised on the JSON list form."""
        config = json.loads(json.dumps(
            BitLinear(units=U, weight_bits=(-3.0, 3.0)).get_config()
        ))
        assert isinstance(config["weight_bits"], list)
        assert BitLinear.from_config(config).weight_range == (-3.0, 3.0)

    def test_get_config_round_trip(self):
        layer = BitLinear(units=U, weight_bits=2, activation_bits=4, use_bias=True)
        rebuilt = BitLinear.from_config(layer.get_config())
        assert rebuilt.units == U
        assert rebuilt.use_bias is True
        assert rebuilt.weight_range == (-1.0, 1.0)
        assert rebuilt.activation_range == (-7.0, 7.0)

    def test_round_trip_with_none_options(self):
        layer = BitLinear(units=U)
        rebuilt = BitLinear.from_config(layer.get_config())
        assert rebuilt.kernel_regularizer is None
        assert rebuilt.kernel_constraint is None


class TestBitLinearPrecision:

    @pytest.mark.parametrize("magnitude", [1.0, 1e-4])
    def test_mixed_float16_stays_finite(self, sample_input, magnitude):
        """The scale is applied as ``x / gamma * Q_max``, never as ``x * (Q_max / gamma)``.

        float16 tops out at 65504, so the reciprocal form overflows for any gamma
        below ``127 / 65504 = 1.94e-03``. Measured at ``magnitude=1e-4``: the
        reciprocal scale reads ``inf`` and ``x * scale`` reads ``[inf, -inf, inf]``,
        while this form reads a finite ``1.025e-05``. At ``magnitude=1.0`` gamma is
        about 2 and both forms are finite -- that arm is the control showing the
        guard is not simply rejecting float16.
        """
        data = (sample_input * magnitude).astype("float32")
        policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            out = keras.ops.convert_to_numpy(_layer()(data))
            assert out.dtype == np.float16
            assert np.all(np.isfinite(out))
        finally:
            keras.mixed_precision.set_global_policy(policy)
        assert keras.mixed_precision.global_policy().name == policy.name

    def test_a_tiny_magnitude_input_stays_finite(self):
        """gamma = 1e-8 would give a scale of 1.27e10 in the reciprocal form."""
        data = (np.random.default_rng(0).standard_normal((B, D)) * 1e-8).astype("float32")
        out = keras.ops.convert_to_numpy(_layer()(data))
        assert np.all(np.isfinite(out))
