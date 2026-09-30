"""``HKAN``: construction, configuration, forward contract and edge cases.

Guards:

* the constructor's validation (every refusal is a ``ValueError`` naming the
  argument) and the scalar-or-per-layer expansion of the six layer settings;
* ``get_config`` carries every constructor argument and ``from_config``
  rebuilds the same model; ``seed=0`` and ``seed=None`` both round-trip;
* ``build()`` materializes every weight ``call()`` uses, 10 for a two-layer
  model with both intercepts and 6 without, in an order the conditional
  weights do not disturb;
* the forward output is ``(B, 1)`` and finite, also far outside ``[0, 1]`` and
  at slope 50;
* ``hidden_units`` and ``num_basis`` reach the parameterisation, ``slope`` and
  ``basis`` reach the forward pass (shared knob-sensitivity oracle);
* the package defines no ``train_step`` / ``test_step`` / ``predict_step`` and
  does not shadow ``keras.Model.fit``;
* the edge cases of the plan's problem statement: one basis function, a
  constant input column, the degenerate ``identity`` basis with several basis
  functions (``lambda_eff = lambda / m``), a hidden layer of width 1, no hidden
  layer, ``y`` as ``(N,)`` or ``(N, 1)``, a rejected multi-column ``y``, a fit
  on an unbuilt model, mismatched or non-finite inputs;
* numpy integers are accepted for the widths, ``num_basis`` and ``seed``, the
  config then holds plain Python numbers and a ``.keras`` save reloads;
* a built model refuses an input of another width (width 1 would broadcast
  silently) in ``call``, ``predict`` and ``fit``, also after a ``.keras``
  round trip, while the numpy-side methods keep their own error;
* a float16 or bfloat16 dtype policy is refused at build, set globally or
  given to the constructor; ``HKAN(dtype="float64")`` gives float64 layers,
  closes the float32 gap and reloads as float64.
"""

import inspect
import json
import os

import keras
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from dl_techniques.models.general_purpose.hkan import HKAN, HKANLayer, create_hkan
from dl_techniques.models.general_purpose.hkan import model as hkan_model_module

from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from . import HIDDEN, N_IN, N_ROWS, NUM_BASIS, SLOPE, make_data

TWO_LAYER_NAMES = [
    "coef", "block_bias", "mix", "bias", "centers",
    "coef", "block_bias", "mix", "bias", "centers",
]


def _numpy(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _model(**overrides) -> HKAN:
    config = dict(hidden_units=(HIDDEN,), num_basis=NUM_BASIS, basis="tanh",
                  slope=SLOPE, seed=0)
    config.update(overrides)
    return HKAN(**config)


@pytest.fixture(scope="module")
def data():
    return make_data()


class TestConstruction:

    def test_layers_and_their_settings(self):
        model = _model(basis=("tanh", "identity"), centers=("data", "random"))
        assert [type(layer) for layer in model.hkan_layers] == [HKANLayer, HKANLayer]
        assert [layer.units for layer in model.hkan_layers] == [HIDDEN, 1]
        assert [layer.num_basis for layer in model.hkan_layers] == list(NUM_BASIS)
        assert [layer.basis for layer in model.hkan_layers] == ["tanh", "identity"]
        assert [layer.centers_mode for layer in model.hkan_layers] == ["data", "random"]
        assert [layer.slope for layer in model.hkan_layers] == [SLOPE, SLOPE]
        assert [layer.layer_index for layer in model.hkan_layers] == [0, 1]
        assert [layer.seed for layer in model.hkan_layers] == [0, 0]

    def test_a_scalar_setting_applies_to_every_layer(self):
        model = HKAN(hidden_units=(HIDDEN, 2), num_basis=4, l2_block=0.5, l2_mix=0.25)
        assert [layer.num_basis for layer in model.hkan_layers] == [4, 4, 4]
        assert model._l2_block == [0.5, 0.5, 0.5] and model._l2_mix == [0.25, 0.25, 0.25]

    def test_default_intercepts_are_on(self):
        """The defaults of decisions.md D-039; the reference code's are slope
        1 and no ridge on either stage."""
        model = HKAN()
        assert model.use_block_bias is True and model.use_bias is True
        assert model.slope == 5.0 and model.l2_block == 0.01 and model.l2_mix == 0.01
        assert model.seed is None

    @pytest.mark.parametrize("kwargs,match", [
        (dict(hidden_units=5), "hidden_units"),
        (dict(hidden_units="5"), "hidden_units"),
        (dict(hidden_units=(5, 0)), "hidden_units"),
        (dict(hidden_units=(5.0,)), "hidden_units"),
        (dict(hidden_units=(True,)), "hidden_units"),
        (dict(hidden_units=(5,), num_basis=(7,)), "num_basis has 1 entries"),
        (dict(hidden_units=(5,), basis=("tanh", "tanh", "tanh")), "basis has 3 entries"),
        (dict(hidden_units=(5,), slope=(1.0,)), "slope has 1 entries"),
        (dict(hidden_units=(5,), centers=("random",)), "centers has 1 entries"),
        (dict(hidden_units=(5,), l2_block=(0.1,)), "l2_block has 1 entries"),
        (dict(hidden_units=(5,), l2_mix=(0.1, 0.1, 0.1)), "l2_mix has 3 entries"),
        (dict(l2_block=-0.1), "l2_block"),
        (dict(l2_mix=float("nan")), "l2_mix"),
        (dict(basis="spline"), "basis"),
        (dict(centers="grid"), "centers"),
        (dict(slope=0.0), "slope"),
        (dict(num_basis=0), "num_basis"),
        (dict(seed=-3), "seed"),
    ])
    def test_invalid_arguments_are_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HKAN(**kwargs)

    def test_no_pretrained_argument_and_no_variants_table(self):
        assert "pretrained" not in inspect.signature(HKAN.__init__).parameters
        assert not hasattr(hkan_model_module, "MODEL_VARIANTS")
        with pytest.raises(ValueError, match="pretrained"):
            create_hkan(pretrained=True)

    @pytest.mark.parametrize("name", ["train_step", "test_step", "predict_step", "fit"])
    def test_the_stock_training_loop_is_not_overridden(self, name):
        assert name not in vars(HKAN), f"HKAN overrides keras.Model.{name}"
        assert name not in vars(HKANLayer)
        assert getattr(HKAN, name) is getattr(keras.Model, name)

    def test_the_registered_name(self):
        assert keras.saving.get_registered_name(HKAN) == "dl_techniques.models.hkan.model>HKAN"


class TestConfig:

    FULL = dict(
        hidden_units=(HIDDEN, 2), num_basis=(7, 4, 3), basis=("tanh", "gaussian", "identity"),
        slope=(5.0, 2.0, 1.0), centers=("data", "equally_spaced", "random"),
        l2_block=(0.01, 0.0, 0.1), l2_mix=(0.0, 0.2, 0.0),
        use_block_bias=False, use_bias=False,
        coef_initializer=keras.initializers.RandomNormal(stddev=0.2),
        mix_initializer="ones", seed=0)

    def test_get_config_carries_every_constructor_argument(self):
        arguments = set(inspect.signature(HKAN.__init__).parameters) - {"self", "kwargs"}
        config = HKAN(**self.FULL).get_config()
        assert arguments <= set(config)
        for name in ("hidden_units", "num_basis", "basis", "slope", "centers",
                     "l2_block", "l2_mix"):
            assert config[name] == list(self.FULL[name])
        assert config["use_block_bias"] is False and config["use_bias"] is False
        assert config["seed"] == 0 and config["seed"] is not None
        assert config["mix_initializer"] is not None

    def test_from_config_rebuilds_the_same_model(self):
        model = HKAN(**self.FULL)
        clone = HKAN.from_config(model.get_config())
        assert clone.get_config() == model.get_config()
        model.build((None, N_IN))
        clone.build((None, N_IN))
        assert [tuple(w.shape) for w in clone.weights] == [tuple(w.shape) for w in model.weights]
        # Same seed, so the same centers, in every layer and every mode.
        for ours, theirs in zip(clone.hkan_layers, model.hkan_layers):
            np.testing.assert_array_equal(_numpy(ours.centers), _numpy(theirs.centers))

    def test_scalar_settings_stay_scalars(self):
        config = _model(num_basis=4, l2_block=0.5).get_config()
        assert config["num_basis"] == 4 and config["l2_block"] == 0.5
        assert config["hidden_units"] == [HIDDEN]

    def test_seed_none_round_trips_as_none(self):
        config = HKAN(seed=None).get_config()
        assert config["seed"] is None
        assert HKAN.from_config(config).seed is None


class TestNumpyIntegers:

    CONFIG = dict(
        hidden_units=[np.int64(HIDDEN), np.int32(2)],
        num_basis=(np.int64(7), np.int64(4), np.int64(3)),
        slope=np.float32(5.0), l2_block=np.float64(0.01), seed=np.int64(3))

    def test_they_are_accepted_and_the_config_is_plain(self):
        model = HKAN(**self.CONFIG)
        assert [layer.units for layer in model.hkan_layers] == [HIDDEN, 2, 1]
        config = model.get_config()
        assert config["hidden_units"] == [HIDDEN, 2]
        assert config["num_basis"] == [7, 4, 3] and config["seed"] == 3
        for value in (*config["hidden_units"], *config["num_basis"], config["seed"]):
            assert type(value) is int
        assert type(config["slope"]) is float and type(config["l2_block"]) is float
        json.dumps({name: config[name] for name in (
            "hidden_units", "num_basis", "slope", "l2_block", "l2_mix", "seed")})

    def test_a_scalar_numpy_num_basis(self):
        config = HKAN(num_basis=np.int64(5)).get_config()
        assert config["num_basis"] == 5 and type(config["num_basis"]) is int

    def test_the_model_is_the_one_built_from_python_integers(self):
        plain = HKAN(hidden_units=[HIDDEN, 2], num_basis=(7, 4, 3), slope=5.0,
                     l2_block=0.01, seed=3)
        model = HKAN(**self.CONFIG)
        ours, theirs = model.get_config(), plain.get_config()
        del ours["name"], theirs["name"]   # auto-generated, one per instance
        assert ours == theirs
        model.build((None, N_IN))
        plain.build((None, N_IN))
        for ours, theirs in zip(model.hkan_layers, plain.hkan_layers):
            np.testing.assert_array_equal(_numpy(ours.centers), _numpy(theirs.centers))

    def test_a_keras_archive_of_it_reloads(self, data, tmp_path):
        x, y = data
        model = HKAN(**self.CONFIG)
        model.fit_closed_form(x, y)
        path = os.path.join(str(tmp_path), "numpy_integers.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        assert loaded.get_config() == model.get_config()
        for ours, theirs in zip(model.get_weights(), loaded.get_weights()):
            np.testing.assert_array_equal(theirs, ours)

    @pytest.mark.parametrize("kwargs,match", [
        (dict(hidden_units=(np.bool_(True),)), "hidden_units"),
        (dict(hidden_units=(np.float64(5.0),)), "hidden_units"),
        (dict(hidden_units=(np.int64(0),)), "hidden_units"),
        (dict(num_basis=np.float32(4.0)), "num_basis"),
        (dict(seed=np.int64(-3)), "seed"),
    ])
    def test_numpy_booleans_floats_and_bad_values_are_still_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HKAN(**kwargs)


class TestInputWidth:
    """Built for ``N_IN = 3`` columns. One column would broadcast against the
    centers and return a ``(B, 1)`` answer; ``N_IN + 2`` was an opaque
    backend shape error."""

    WIDTHS = [1, N_IN + 2]

    @pytest.fixture(scope="class")
    def fitted_model(self, data):
        model = _model(l2_block=0.01)
        model.fit_closed_form(*data)
        return model

    @pytest.mark.parametrize("width", WIDTHS)
    def test_call_and_predict_refuse_it(self, fitted_model, width):
        bad = np.zeros((2, width), dtype="float32")
        with pytest.raises(ValueError, match="incompatible with the layer"):
            fitted_model(bad)
        with pytest.raises(ValueError, match="incompatible with the layer"):
            fitted_model.predict(bad, verbose=0)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_fit_refuses_it_and_accepts_the_right_width(self, data, width):
        x, y = data
        model = _model()
        model.build((None, N_IN))
        model.compile(optimizer="sgd", loss="mse")
        with pytest.raises(ValueError, match="incompatible with the layer"):
            model.fit(np.zeros((8, width), dtype="float32"), y[:8], epochs=1, verbose=0)
        history = model.fit(x.astype("float32"), y.astype("float32"), epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][0])

    @pytest.mark.parametrize("width", WIDTHS)
    def test_it_survives_a_keras_round_trip(self, fitted_model, data, tmp_path, width):
        path = os.path.join(str(tmp_path), "width.keras")
        fitted_model.save(path)
        loaded = keras.models.load_model(path)
        x = data[0].astype("float32")
        np.testing.assert_array_equal(_numpy(loaded(x)), _numpy(fitted_model(x)))
        with pytest.raises(ValueError, match="incompatible with the layer"):
            loaded(np.zeros((2, width), dtype="float32"))

    @pytest.mark.parametrize("width", WIDTHS)
    def test_inside_a_functional_model(self, width):
        inputs = keras.Input((N_IN,))
        inner = _model()
        functional = keras.Model(inputs, inner(inputs))
        assert functional(np.zeros((2, N_IN), dtype="float32")).shape == (2, 1)
        with pytest.raises(ValueError, match="incompatible with the layer"):
            inner(np.zeros((2, width), dtype="float32"))

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_numpy_side_methods_keep_their_own_error(self, fitted_model, data, width):
        bad = np.zeros((N_ROWS, width))
        with pytest.raises(ValueError, match=f"x has {width} columns but the model was built for {N_IN}"):
            fitted_model.fit_closed_form(bad, data[1])
        with pytest.raises(ValueError, match=f"x has {width} columns but the model was built for {N_IN}"):
            fitted_model.initialize_centers(bad)


class TestDtypePolicy:

    @pytest.fixture
    def policy(self, request):
        """Set the global dtype policy for one test and always restore it."""
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(request.param)
        try:
            yield request.param
        finally:
            keras.mixed_precision.set_global_policy(previous)

    REFUSED = ["mixed_float16", "float16", "mixed_bfloat16", "bfloat16"]

    @pytest.mark.parametrize("policy", REFUSED, indirect=True)
    def test_a_float16_policy_is_refused(self, data, policy):
        """At ``build``, at the first call, and by ``fit_closed_form``."""
        x, y = data
        with pytest.raises(ValueError, match="'float32' and 'float64'"):
            _model().build((None, N_IN))
        with pytest.raises(ValueError, match="'float32' and 'float64'"):
            _model()(x.astype("float32"))
        with pytest.raises(ValueError, match="'float32' and 'float64'"):
            _model().fit_closed_form(x, y)
        with pytest.raises(ValueError, match="'float32' and 'float64'"):
            create_hkan(hidden_units=(HIDDEN,), input_dim=N_IN)

    @pytest.mark.parametrize("dtype", REFUSED)
    def test_the_constructor_dtype_is_refused_too(self, data, dtype):
        """``HKAN(dtype=...)``, the standard Keras way, under the float32
        global policy. It never reached the layers: ``mixed_float16`` built,
        fitted and predicted with float32 layers on float16-rounded inputs,
        forward deviation 1.2e-04 (bfloat16: 9.2e-04), no error (decisions.md
        D-049)."""
        x, y = data
        with pytest.raises(ValueError, match="'float32' and 'float64'") as raised:
            _model(dtype=dtype).build((None, N_IN))
        assert dtype in str(raised.value)
        with pytest.raises(ValueError, match="'float32' and 'float64'"):
            _model(dtype=dtype).fit_closed_form(x, y)
        with pytest.raises(ValueError, match="'float32' and 'float64'"):
            create_hkan(hidden_units=(HIDDEN,), input_dim=N_IN, dtype=dtype)
        assert keras.mixed_precision.global_policy().name == "float32"

    @pytest.mark.parametrize("policy", ["float32", "float64"], indirect=True)
    def test_float32_and_float64_build_fit_and_predict(self, data, policy):
        x, y = data
        model = _model(l2_block=0.01)
        diagnostics = model.fit_closed_form(x, y)
        output = _numpy(model(x.astype(policy)))
        assert output.dtype == np.dtype(policy)
        # Measured deviation at the default l2_mix: 1.7e-07 (float32, CPU),
        # 1.9e-07 (GPU 1), 2.9e-16 (float64).
        assert diagnostics["forward_deviation_rms"] < (1e-5 if policy == "float32" else 1e-13)

    #: A configuration whose float32 model is NOT its fit (connecting weights
    #: from ``l2_mix = 0`` at slope 1): README section 6, row seven.
    WIDE_GAP = dict(hidden_units=(16,), num_basis=10, basis="sigmoid", slope=1.0,
                    l2_block=0.01, l2_mix=0.0, seed=0)

    def test_the_constructor_float64_reaches_every_layer_and_closes_the_gap(self, data):
        """Measured on this fixture (CPU), deviation / std(y): 1.0e-02 with
        float32 layers (the fit warns), 2.0e-11 with float64 layers
        (decisions.md D-049)."""
        x, y = data
        ratios = {}
        for dtype in ("float32", "float64"):
            model = HKAN(dtype=dtype, **self.WIDE_GAP)
            diagnostics = model.fit_closed_form(x, y)
            assert [layer.compute_dtype for layer in model.hkan_layers] == [dtype, dtype]
            assert {w.dtype for w in model.weights} == {dtype}
            assert _numpy(model(x[:4])).dtype == np.dtype(dtype)
            ratios[dtype] = diagnostics["forward_deviation_rms"] / np.std(y)
        assert keras.mixed_precision.global_policy().name == "float32"
        assert ratios["float32"] > 1e-4, "the fixture has no float32 gap to close"
        assert ratios["float64"] < 1e-9, ratios

    def test_a_float64_model_reloads_as_float64(self, data, tmp_path):
        x, y = data
        model = HKAN(dtype="float64", **self.WIDE_GAP)
        model.fit_closed_form(x, y)
        path = os.path.join(str(tmp_path), "float64.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        assert loaded.dtype_policy.name == "float64"
        assert [layer.compute_dtype for layer in loaded.hkan_layers] == ["float64", "float64"]
        for ours, theirs in zip(model.weights, loaded.weights):
            assert theirs.dtype == "float64"
            np.testing.assert_array_equal(theirs.numpy(), ours.numpy())
        np.testing.assert_array_equal(_numpy(loaded(x)), _numpy(model(x)))

    def test_the_policy_was_restored(self):
        assert keras.mixed_precision.global_policy().name == "float32"


class TestBuildAndForward:

    def test_build_materializes_every_weight(self):
        model = _model()
        assert model.weights == []
        model.build((None, N_IN))
        assert model.built and all(layer.built for layer in model.hkan_layers)
        assert [w.name for w in model.weights] == TWO_LAYER_NAMES
        assert len(model.trainable_weights) == 8 and len(model.non_trainable_weights) == 2
        assert [tuple(w.shape) for w in model.weights] == [
            (HIDDEN, N_IN, 7), (HIDDEN, N_IN), (HIDDEN, N_IN), (HIDDEN,), (HIDDEN, N_IN, 7),
            (1, HIDDEN, 4), (1, HIDDEN), (1, HIDDEN), (1,), (1, HIDDEN, 4),
        ]

    def test_without_intercepts_there_are_six_weights_in_the_same_order(self):
        model = _model(use_block_bias=False, use_bias=False)
        model.build((None, N_IN))
        assert [w.name for w in model.weights] == [
            "coef", "mix", "centers", "coef", "mix", "centers"]

    def test_the_first_call_creates_no_weight_build_did_not(self, data):
        built = _model()
        built.build((None, N_IN))
        called = _model()
        called(data[0].astype("float32"))
        # The first path segment is the model's auto-generated name.
        def relative(model):
            return [(w.path.split("/", 1)[1], tuple(w.shape)) for w in model.weights]

        assert relative(built) == relative(called)
        assert relative(built)[0] == ("hkan_layer_0/coef", (HIDDEN, N_IN, 7))

    @pytest.mark.parametrize("shape", [(None,), (None, 4, 4), (None, None)])
    def test_build_refuses_a_bad_input_shape(self, shape):
        with pytest.raises(ValueError, match="n_in"):
            _model().build(shape)

    def test_forward_shape_and_finiteness(self, data):
        model = _model()
        output = _numpy(model(data[0].astype("float32"), training=False))
        assert output.shape == (N_ROWS, 1) and output.dtype == np.float32
        assert np.all(np.isfinite(output))
        assert np.std(output) > 0.0, "the unfitted model is a constant"
        assert model.compute_output_shape((None, N_IN)) == (None, 1)

    def test_training_flag_changes_nothing(self, data):
        model = _model()
        x = data[0].astype("float32")
        np.testing.assert_array_equal(
            _numpy(model(x, training=True)), _numpy(model(x, training=False)))

    @pytest.mark.parametrize("basis", ["sigmoid", "softplus", "tanh", "gaussian", "relu"])
    def test_slope_50_far_outside_the_unit_interval_is_finite(self, basis):
        model = _model(basis=basis, slope=50.0)
        x = np.array([[-100.0, 0.5, 100.0], [40.0, -40.0, 3.0]], dtype="float32")
        assert np.all(np.isfinite(_numpy(model(x))))

    def test_the_factory_builds_when_given_an_input_width(self):
        model = create_hkan(hidden_units=(HIDDEN,), input_dim=N_IN, num_basis=NUM_BASIS)
        assert isinstance(model, HKAN) and model.built and len(model.weights) == 10
        assert not create_hkan(hidden_units=(HIDDEN,)).built
        with pytest.raises(ValueError, match="no_such_argument"):
            create_hkan(hidden_units=(HIDDEN,), no_such_argument=1)


class TestKnobs:

    def test_hidden_units_is_structural(self):
        assert_structural_knob_changes_weights(
            {units: (lambda units=units: create_hkan(hidden_units=units, input_dim=N_IN))
             for units in ((), (HIDDEN,), (HIDDEN, 2), (4, 2))},
            knob="hidden_units")

    def test_num_basis_is_structural(self):
        signatures = assert_structural_knob_changes_weights(
            {m: (lambda m=m: create_hkan(hidden_units=(HIDDEN,), input_dim=N_IN, num_basis=m))
             for m in (1, 4, 7, (7, 4))},
            knob="num_basis")
        assert signatures[7][0] == (HIDDEN, N_IN, 7)

    def test_intercept_flags_are_structural(self):
        assert_structural_knob_changes_weights(
            {flags: (lambda flags=flags: create_hkan(
                hidden_units=(HIDDEN,), input_dim=N_IN,
                use_block_bias=flags[0], use_bias=flags[1]))
             for flags in ((True, True), (False, True), (False, False), (True, False))},
            knob="(use_block_bias, use_bias)")

    def test_slope_and_basis_reach_the_forward_pass(self, data):
        x = data[0].astype("float32")
        assert_value_knob_changes_output(
            {slope: (lambda slope=slope: _model(slope=slope)) for slope in (1.0, 5.0, 50.0)},
            x, knob="slope")
        assert_value_knob_changes_output(
            {basis: (lambda basis=basis: _model(basis=basis))
             for basis in ("sigmoid", "gaussian", "relu", "tanh", "softplus", "identity")},
            x, knob="basis")

    def test_the_knob_assertion_can_fail(self):
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(
                {key: (lambda: create_hkan(hidden_units=(HIDDEN,), input_dim=N_IN))
                 for key in ("a", "b")},
                knob="hidden_units")


class TestEdgeCases:

    def test_one_basis_function(self, data):
        model = _model(num_basis=1)
        diagnostics = model.fit_closed_form(*data)
        assert model.hkan_layers[0].coef.shape == (HIDDEN, N_IN, 1)
        assert np.all(np.isfinite(diagnostics["train_predictions"]))

    def test_a_constant_input_column_has_zero_importance(self, data):
        x, y = data
        x = x.copy()
        x[:, 1] = 0.5
        diagnostics = _model(l2_block=0.01).fit_closed_form(x, y)
        assert np.all(np.isfinite(diagnostics["train_predictions"]))
        # A block of a constant column can only predict the mean: R^2 = 0.
        # Measured |importance[1]| = 0.0 exactly; the other two are 0.51 and 0.24.
        assert abs(diagnostics["importance"][1]) <= 1e-12
        assert diagnostics["importance"][0] > 0.1 and diagnostics["importance"][2] > 0.1

    def test_identity_with_several_basis_functions_only_rescales_the_ridge(self, data):
        """``m`` identity features are one centered column repeated ``m`` times.

        The minimum-norm ridge solution spreads one weight ``s`` evenly, so
        the penalty is ``l2 * s^2 / m``: the block is a one-feature ridge on
        ``x_p`` with ``alpha = l2 / m``. Measured agreement 1.1e-16;
        ``atol = 1e-12``.
        """
        x, y = data
        l2, m = 0.3, NUM_BASIS[0]
        layer = HKANLayer(units=HIDDEN, num_basis=m, basis="identity", slope=50.0)
        centers = np.random.default_rng(2).uniform(0.0, 1.0, size=(HIDDEN, N_IN, m))
        solution = layer.solve_closed_form(x, y, l2, 0.0, centers)
        for p in range(N_IN):
            column = x[:, p:p + 1]
            expected = Ridge(alpha=l2 / m).fit(column, y).predict(column)
            unscaled = Ridge(alpha=l2).fit(column, y).predict(column)
            assert np.abs(expected - unscaled).max() > 1e-3, "the fixture cannot see l2 / m"
            for q in range(HIDDEN):
                ours = (
                    (column - centers[q, p][None, :]) @ solution["coef"][q, p]
                    + solution["block_bias"][q, p])
                np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-12)

    def test_a_hidden_layer_of_width_one(self, data):
        model = HKAN(hidden_units=(1,), num_basis=NUM_BASIS, basis="tanh", slope=SLOPE, seed=0)
        diagnostics = model.fit_closed_form(*data)
        assert [tuple(layer.mix.shape) for layer in model.hkan_layers] == [(1, N_IN), (1, 1)]
        assert diagnostics["block_r2"].shape == (1, N_IN)
        assert _numpy(model(data[0].astype("float32"))).shape == (N_ROWS, 1)

    def test_no_hidden_layer(self, data):
        model = HKAN(hidden_units=(), num_basis=7, basis="tanh", slope=SLOPE, seed=0)
        diagnostics = model.fit_closed_form(*data)
        assert len(model.hkan_layers) == 1 and len(model.weights) == 5
        assert len(diagnostics["layer_rmse"]) == 1
        assert diagnostics["block_r2"].shape == (1, N_IN)
        assert diagnostics["layer_rmse"][0] < np.std(data[1])

    def test_y_as_a_vector_or_a_column_is_the_same_fit(self, data):
        x, y = data
        flat = _model().fit_closed_form(x, y)
        column = _model().fit_closed_form(x, y[:, None])
        np.testing.assert_array_equal(flat["train_predictions"], column["train_predictions"])
        assert column["train_predictions"].shape == (N_ROWS,)

    def test_inputs_may_be_float32_or_lists(self, data):
        x, y = data
        reference = _model().fit_closed_form(x.astype("float32"), y.astype("float32"))
        listed = _model().fit_closed_form(
            x.astype("float32").tolist(), y.astype("float32").tolist())
        np.testing.assert_array_equal(
            reference["train_predictions"], listed["train_predictions"])
        assert reference["train_predictions"].dtype == np.float64

    def test_a_multi_column_target_is_refused(self, data):
        x, y = data
        with pytest.raises(ValueError, match="single target"):
            _model().fit_closed_form(x, np.stack([y, y], axis=1))

    def test_a_fit_builds_an_unbuilt_model(self, data):
        model = _model()
        assert not model.built
        model.fit_closed_form(*data)
        assert model.built and [w.name for w in model.weights] == TWO_LAYER_NAMES

    @pytest.mark.parametrize("bad_x,bad_y,match", [
        (np.zeros((N_ROWS, N_IN + 1)), None, "columns"),
        (None, np.zeros((N_ROWS - 1,)), "rows"),
        (np.zeros((N_ROWS,)), None, "non-empty"),
        (np.zeros((0, N_IN)), np.zeros((0,)), "non-empty"),
    ])
    def test_mismatched_shapes_are_refused(self, data, bad_x, bad_y, match):
        x, y = data
        model = _model()
        model.build((None, N_IN))
        with pytest.raises(ValueError, match=match):
            model.fit_closed_form(x if bad_x is None else bad_x, y if bad_y is None else bad_y)

    @pytest.mark.parametrize("target", ["x", "y"])
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_inputs_are_refused_and_nothing_is_assigned(self, data, target, value):
        x, y = (array.copy() for array in data)
        (x if target == "x" else y)[3] = value
        model = _model()
        model.build((None, N_IN))
        before = [w.copy() for w in model.get_weights()]
        with pytest.raises(ValueError, match="finite"):
            model.fit_closed_form(x, y)
        for ours, theirs in zip(model.get_weights(), before):
            np.testing.assert_array_equal(ours, theirs)
