"""The homogeneity instrument of the bias-free denoisers (``train.common.denoising_homogeneity``).

Two instruments, each blind to a case the other sees (D-015). The tests pin both sides of every claim:

* every control has a POSITIVE arm (a planted offender is listed statically AND reads above the tolerance
  on the probe clone) next to its negative arm (the clean models of the three families pass);
* the two blind spots are pinned as blind spots: GELU is listed statically while the numeric probe at the
  default amplitude reads under the tolerance, and a LayerNorm model reads under it at init and above it
  after the clone or after a brief training;
* the probe's defaults are pinned because they are what makes a reading mean something: non-power-of-two
  alphas (a power of two is exact in float32 and reads 0.0 for any network) and TF32 off in a restored scope.

The numbers behind the tolerance (clean models at most 1.03e-5 at tiny size, every planted case at least
2.3e-3) are in ``plans/plan-2026-11-10T095307-d483ffab/scratch_measure/measured.md`` item (f) and in
``scratch_step6/measured_tolerance.md`` for small and base size. Everything here is CPU and tiny.
"""
import inspect
import math
import random
from typing import Any, Dict, List, Sequence

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.norms.bias_free_batch_norm import BiasFreeBatchNorm
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.models.vision.bias_free_denoisers.bfcnn import create_bfcnn_variant
from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import create_convunext_variant
from dl_techniques.models.vision.bias_free_denoisers.bfunet import create_bfunet_variant
from tests.numerics import tf32_execution
from train.common import denoising_homogeneity as dh
from train.common.seed import seed_generators

SIZE = 32
TOLERANCE = dh.PROBE_TOLERANCE


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------


def _leaky() -> keras.layers.Layer:
    """The legacy activation: a LeakyReLU INSTANCE with slope 0.1 (the recipe builds it as an instance)."""
    return keras.layers.LeakyReLU(negative_slope=0.1)


def _images(batch: int = 2, size: int = SIZE, seed: int = 0) -> np.ndarray:
    """Default-amplitude inputs: the ``[0, 1]`` domain of the denoisers."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, (batch, size, size, 3)).astype("float32")


def _bfcnn() -> keras.Model:
    return create_bfcnn_variant(
        "tiny", input_shape=(None, None, 3), activation=_leaky(),
        normalization_type="batchnorm", final_activation="linear",
    )


def _bfunet() -> keras.Model:
    return create_bfunet_variant(
        "tiny", input_shape=(None, None, 3), activation=_leaky(), final_activation="linear",
        block_normalization="batchnorm", use_gabor_stem=True, final_projection_groups=1,
        enable_deep_supervision=False,
    )


def _bfconvunext(version: str = "v1", gabor: bool = True, norm: str = "batchnorm", gelu: bool = False) -> keras.Model:
    """The bias-free ConvUNext at tiny size; ``gelu=True`` leaves the activations at the library default (GELU)."""
    activations: Dict[str, Any] = {}
    if not gelu:
        act = _leaky()
        activations = dict(block_activation=act, stem_activation=act, supervision_activation=act)
    return create_convunext_variant(
        "tiny", input_shape=(None, None, 3), enable_deep_supervision=False, convnext_version=version,
        block_normalization=norm, use_gabor_stem=gabor, final_projection_groups=1,
        final_activation="linear", **activations,
    )


def _chain(*middle: keras.layers.Layer, first_bias: bool = False) -> keras.Model:
    """``conv -> middle... -> LeakyReLU -> conv``: a small functional model whose only suspect is ``middle``."""
    inputs = keras.Input((None, None, 3))
    x = keras.layers.Conv2D(8, 3, padding="same", use_bias=first_bias)(inputs)
    for layer in middle:
        x = layer(x)
    x = _leaky()(x)
    return keras.Model(inputs, keras.layers.Conv2D(3, 3, padding="same", use_bias=False)(x))


def _plant(model: keras.Model, leaves: Sequence[str] = ("bias", "beta", "moving_mean"), seed: int = 1) -> int:
    """Assign random non-zero values to every weight whose last path component is in ``leaves``; return how many."""
    rng = np.random.default_rng(seed)
    count = 0
    for weight in model.weights:
        if weight.path.rpartition("/")[2] in leaves:
            weight.assign(rng.normal(0.0, 0.3, tuple(weight.shape)).astype("float32"))
            count += 1
    return count


def _kinds(report: Dict[str, Any]) -> set:
    return {row["kind"] for row in report["offenders"]}


def _clone_read(model: keras.Model, x: np.ndarray = None) -> Dict[str, Any]:
    return dh.probe_numeric(dh.init_probe_clone(model), _images() if x is None else x)


def _brief_fit(model: keras.Model, steps: int = 20, lr: float = 3e-3, seed: int = 0) -> keras.Model:
    """A few AdamW steps of denoising on random 32 px images: enough to move layer scales off their init."""
    rng = np.random.default_rng(seed)
    clean = rng.uniform(0.0, 1.0, (8, SIZE, SIZE, 3)).astype("float32")
    noisy = clean + rng.normal(0.0, 0.1, clean.shape).astype("float32")
    model.compile(optimizer=keras.optimizers.AdamW(lr), loss="mse")
    model.fit(noisy, clean, batch_size=4, epochs=steps // 2, verbose=0)
    return model



# ---------------------------------------------------------------------
# The defaults are what make a reading mean something
# ---------------------------------------------------------------------


class TestDefaults:
    def test_default_alphas_are_not_powers_of_two(self) -> None:
        """A power-of-two scale is exact in float32, so it would read 0.0 for ANY network."""
        for alpha in dh.PROBE_ALPHAS:
            mantissa, _ = math.frexp(alpha)
            assert mantissa != 0.5, f"{alpha} is a power of two"

    def test_probe_defaults_are_the_module_constants(self) -> None:
        signature = inspect.signature(dh.probe_numeric)
        assert signature.parameters["alphas"].default == dh.PROBE_ALPHAS == (0.3, 1.3, 3.7, 6.1)
        assert signature.parameters["tolerance"].default == dh.PROBE_TOLERANCE == 1e-3

    def test_power_of_two_alphas_read_zero_where_the_default_alphas_read_the_rounding(self) -> None:
        """Why the defaults are off powers of two: the same clean model, two readings, 0.0 and above it."""
        model = _chain(keras.layers.Conv2D(8, 3, padding="same", use_bias=False),
                       BiasFreeBatchNorm(name="bf_bn"))
        x = _images()
        pow2 = dh.probe_numeric(model, x, alphas=(0.5, 2.0, 4.0, 8.0))
        default = dh.probe_numeric(model, x)
        assert pow2["max_rel"] == 0.0
        assert 0.0 < default["max_rel"] < TOLERANCE
        assert all(value > 0.0 for value in default["per_alpha"])

    def test_a_planted_bias_reads_above_the_tolerance_at_either_alpha_set(self) -> None:
        """Positive arm of the above: a real violation (a bias) is seen at power-of-two alphas too; they hide only rounding."""
        model = _chain(first_bias=True)
        _plant(model, ("bias",))
        x = _images()
        assert dh.probe_numeric(model, x, alphas=(0.5, 2.0, 4.0, 8.0))["max_rel"] > TOLERANCE
        assert dh.probe_numeric(model, x)["max_rel"] > TOLERANCE

    def test_bad_arguments_are_refused(self) -> None:
        model = _chain()
        x = _images()
        for alphas in ((), (0.0,), (-1.0,), (float("nan"),), (float("inf"),)):
            with pytest.raises(ValueError, match="alphas"):
                dh.probe_numeric(model, x, alphas=alphas)
        for tolerance in (0.0, -1e-3, float("nan")):
            with pytest.raises(ValueError, match="tolerance"):
                dh.probe_numeric(model, x, tolerance=tolerance)


# ---------------------------------------------------------------------
# The clean families pass both instruments
# ---------------------------------------------------------------------

CLEAN_FAMILIES = {
    "bfcnn": _bfcnn,
    "bfunet": _bfunet,
    "bfconvunext_v1_gabor": lambda: _bfconvunext("v1", True),
    "bfconvunext_v1_nogabor": lambda: _bfconvunext("v1", False),
    "bfconvunext_v2_gabor": lambda: _bfconvunext("v2", True),
    "bfconvunext_v2_nogabor": lambda: _bfconvunext("v2", False),
}


@pytest.fixture(scope="module", params=sorted(CLEAN_FAMILIES))
def clean_model(request) -> keras.Model:
    return CLEAN_FAMILIES[request.param]()


class TestCleanFamiliesPass:
    def test_static_scan_lists_nothing(self, clean_model) -> None:
        report = dh.scan_static(clean_model)
        assert report["offenders"] == [] and report["counts"] == {} and report["passed"] is True

    def test_numeric_probe_passes_at_init_and_on_the_clone(self, clean_model) -> None:
        x = _images()
        at_init = dh.probe_numeric(clean_model, x)
        on_clone = dh.probe_numeric(dh.init_probe_clone(clean_model), x)
        assert at_init["passed"] and on_clone["passed"]
        # the clean side sits far under the tolerance (measured at most 1.03e-5; 3e-4 is the stop line of D-019)
        assert at_init["max_rel"] < 3e-4 and on_clone["max_rel"] < 3e-4

    def test_numeric_probe_passes_after_a_brief_training(self) -> None:
        for build in (_bfcnn, lambda: _bfconvunext("v2", False)):
            model = _brief_fit(build())
            assert dh.scan_static(model)["passed"]
            assert dh.probe_numeric(model, _images())["max_rel"] < 3e-4
            assert dh.probe_numeric(dh.init_probe_clone(model), _images())["max_rel"] < 3e-4


# ---------------------------------------------------------------------
# Planted offenders: listed statically AND above the tolerance after the clone
# ---------------------------------------------------------------------


class TestPlantedOffenders:
    def test_planted_conv_bias(self) -> None:
        model = _chain(first_bias=True)
        assert _plant(model, ("bias",)) == 1
        report = dh.scan_static(model)
        assert "bias" in _kinds(report) and not report["passed"]
        assert _clone_read(model)["max_rel"] > TOLERANCE
        # negative arm: the same architecture without the bias is clean on both
        clean = _chain(first_bias=False)
        assert dh.scan_static(clean)["passed"] and _clone_read(clean)["passed"]

    def test_planted_dense_bias_in_a_nested_functional_model(self) -> None:
        """A nested functional model is walked: the bias sits two levels down and is still found."""
        inner_in = keras.Input((None, None, 3))
        inner = keras.Model(inner_in, keras.layers.Conv2D(3, 1, use_bias=True, name="deep_conv")(inner_in), name="inner")
        outer_in = keras.Input((None, None, 3))
        outer = keras.Model(outer_in, inner(outer_in), name="outer")
        _plant(outer, ("bias",))
        report = dh.scan_static(outer)
        assert [row["layer"] for row in report["offenders"] if row["kind"] == "bias"] == ["deep_conv"]
        assert _clone_read(outer)["max_rel"] > TOLERANCE

    def test_planted_grn_beta(self) -> None:
        model = _chain(GlobalResponseNormalization(use_beta=True, name="grn"))
        assert _plant(model, ("beta",)) == 1
        report = dh.scan_static(model)
        assert _kinds(report) == {"beta"}
        assert _clone_read(model)["max_rel"] > TOLERANCE
        # negative arm: a GRN built without beta is clean on both (the bias-free arm of the library)
        clean = _chain(GlobalResponseNormalization(use_beta=False, name="grn"))
        assert dh.scan_static(clean)["passed"] and _clone_read(clean)["passed"]

    def test_planted_stock_batchnorm(self) -> None:
        model = _chain(keras.layers.BatchNormalization(center=True, scale=True, name="stock_bn"))
        assert _plant(model, ("beta", "moving_mean")) == 2
        report = dh.scan_static(model)
        assert {"stock_batchnorm", "beta", "moving_mean"} <= _kinds(report)
        assert _clone_read(model)["max_rel"] > TOLERANCE

    def test_stock_batchnorm_without_center_is_still_listed(self) -> None:
        """``center=False`` drops beta but the moving mean is still subtracted at inference."""
        model = _chain(keras.layers.BatchNormalization(center=False, name="stock_bn"))
        assert "stock_batchnorm" in _kinds(dh.scan_static(model))

    def test_layernorm_rmsnorm_groupnorm_are_listed(self) -> None:
        for layer, kind in (
            (keras.layers.LayerNormalization(), "layernorm"),
            (RMSNorm(), "rmsnorm"),
            (keras.layers.GroupNormalization(groups=2), "groupnorm"),
        ):
            model = _chain(layer)
            assert kind in _kinds(dh.scan_static(model)), kind
            # degree 0, not 1: the probe sees them too (no layer scale mutes them in this model)
            assert dh.probe_numeric(model, _images())["max_rel"] > TOLERANCE, kind

    def test_a_planted_offender_is_found_after_a_real_family(self) -> None:
        """The scan and the probe work on a built bias-free family, not only on the hand-built models."""
        model = _bfcnn()
        assert dh.scan_static(model)["passed"]
        # a family has no bias weight to plant, so break it the way a user would: a biased layer spliced on
        spliced = keras.Model(
            model.inputs[0],
            keras.layers.Conv2D(3, 1, use_bias=True, name="spliced")(model.outputs[0]),
        )
        _plant(spliced, ("bias",))
        assert [row["layer"] for row in dh.scan_static(spliced)["offenders"]] == ["spliced"]
        assert dh.probe_numeric(spliced, _images())["max_rel"] > TOLERANCE


# ---------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------


class TestActivations:
    @pytest.mark.parametrize("layer", [
        lambda: keras.layers.LeakyReLU(negative_slope=0.1),
        lambda: keras.layers.Activation("linear"),
        lambda: keras.layers.Activation("relu"),
        lambda: keras.layers.Activation("leaky_relu"),
        lambda: keras.layers.Activation(keras.layers.LeakyReLU(negative_slope=0.1)),
        lambda: keras.layers.ReLU(),
        lambda: keras.layers.PReLU(shared_axes=[1, 2]),
        lambda: keras.layers.Conv2D(8, 3, padding="same", use_bias=False, activation="relu"),
        lambda: keras.layers.Conv2D(8, 3, padding="same", use_bias=False, activation=None),
    ])
    def test_allowlisted_activations_pass(self, layer) -> None:
        report = dh.scan_static(_chain(layer()))
        assert report["passed"], report["offenders"]

    @pytest.mark.parametrize("layer, kind", [
        (lambda: keras.layers.Activation("gelu"), "activation"),
        (lambda: keras.layers.Activation("tanh"), "activation"),
        (lambda: keras.layers.Activation("swish"), "activation"),
        (lambda: keras.layers.ELU(), "activation"),
        (lambda: keras.layers.Softmax(), "activation"),
        (lambda: keras.layers.ReLU(max_value=6.0), "activation"),
        (lambda: keras.layers.ReLU(threshold=0.1), "activation"),
        (lambda: keras.layers.Conv2D(8, 3, padding="same", use_bias=False, activation="gelu"), "activation"),
    ])
    def test_activations_outside_the_allowlist_are_listed(self, layer, kind) -> None:
        report = dh.scan_static(_chain(layer()))
        assert kind in _kinds(report) and not report["passed"]

    def test_a_callable_activation_is_reported_unknown_and_fails(self) -> None:
        for layer in (keras.layers.Activation(lambda x: x * 1.0), keras.layers.Activation(keras.activations.gelu)):
            report = dh.scan_static(_chain(layer))
            assert not report["passed"]
        unknown = dh.scan_static(_chain(keras.layers.Activation(lambda x: x * 1.0)))
        assert "activation_unknown" in _kinds(unknown)
        # a Keras function the registry knows is judged by its name, not "unknown"
        assert "activation_unknown" not in _kinds(dh.scan_static(_chain(keras.layers.Activation(keras.activations.gelu))))

    def test_gelu_is_listed_statically_while_the_numeric_probe_reads_under_the_tolerance(self) -> None:
        """Pins WHY both instruments exist: the library default GELU is invisible to the numeric probe."""
        model = _bfconvunext("v1", True, gelu=True)
        report = dh.scan_static(model)
        # the library default is the callable ``exact_gelu``, which no Keras registry knows: reported unknown, and
        # unknown fails. One row per layer even though all 14 block activations share the name "activation".
        assert not report["passed"] and report["counts"] == {"activation_unknown": 14}
        assert all("exact_gelu" in row["detail"] for row in report["offenders"])
        x = _images()
        assert dh.probe_numeric(model, x)["passed"]
        assert dh.probe_numeric(dh.init_probe_clone(model), x)["passed"]


# ---------------------------------------------------------------------
# LayerNorm: invisible at init, visible on the clone and after training
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def layernorm_model() -> keras.Model:
    return _bfconvunext("v1", True, norm="layernorm")


class TestLayerNorm:
    def test_listed_statically(self, layernorm_model) -> None:
        report = dh.scan_static(layernorm_model)
        assert report["counts"].get("layernorm", 0) >= 1 and not report["passed"]

    def test_invisible_at_init_visible_on_the_clone(self, layernorm_model) -> None:
        x = _images()
        assert dh.probe_numeric(layernorm_model, x)["max_rel"] < TOLERANCE      # the layer scale mutes it
        assert dh.probe_numeric(dh.init_probe_clone(layernorm_model), x)["max_rel"] > TOLERANCE

    def test_a_briefly_trained_model_reads_above_the_tolerance_where_its_init_read_under(self) -> None:
        model = _bfconvunext("v1", True, norm="layernorm")
        x = _images()
        before = dh.probe_numeric(model, x)["max_rel"]
        _brief_fit(model)
        after = dh.probe_numeric(model, x)["max_rel"]
        assert before < TOLERANCE < after, (before, after)


# ---------------------------------------------------------------------
# Layer scales
# ---------------------------------------------------------------------


class TestLayerScale:
    def test_identified_by_the_gamma_scale_path_and_not_an_offender(self) -> None:
        report = dh.scan_static(_bfconvunext("v1", True))
        assert len(report["layer_scales"]) == 14 and report["passed"]
        assert all(path.endswith("gamma_scale/gamma") for path in report["layer_scales"])
        assert dh.scan_static(_bfcnn())["layer_scales"] == []

    def test_clone_sets_layer_scales_to_one_and_redraws_running_var(self) -> None:
        model = _bfconvunext("v1", True)
        before = {w.path: np.array(w.numpy()) for w in model.weights}
        clone = dh.init_probe_clone(model)
        scales = [w for w in clone.weights if w.path.endswith("gamma_scale/gamma")]
        variances = [w for w in clone.weights if w.path.endswith("running_var")]
        assert len(scales) == 14 and len(variances) == 14
        assert all(np.all(w.numpy() == 1.0) for w in scales)
        assert all(np.all((w.numpy() >= 0.5) & (w.numpy() < 2.0)) for w in variances)
        # positive arm: the init values really were different (1e-4 scales, running_var 1)
        assert all(np.all(before[w.path] < 1e-2) for w in scales)
        assert all(np.all(before[w.path] == 1.0) for w in variances)
        # every other weight is copied, and the original is untouched
        touched = {w.path for w in scales + variances}
        assert all(np.array_equal(w.numpy(), before[w.path]) for w in clone.weights if w.path not in touched)
        assert all(np.array_equal(w.numpy(), before[w.path]) for w in model.weights)

    def test_the_clone_seed_makes_the_draw_reproducible(self) -> None:
        model = _chain(BiasFreeBatchNorm(name="bf_bn"))
        draw = lambda seed: [w.numpy().copy() for w in dh.init_probe_clone(model, seed).weights if w.path.endswith("running_var")]
        first, again, other = draw(7), draw(7), draw(8)
        assert all(np.array_equal(a, b) for a, b in zip(first, again))
        assert not all(np.array_equal(a, b) for a, b in zip(first, other))


# ---------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------


class TestReport:
    def test_single_output_report_has_exactly_its_keys(self) -> None:
        report = dh.probe_numeric(_chain(), _images())
        assert set(report) == {"alphas", "max_rel", "per_alpha", "tolerance", "passed", "tf32"}
        assert report["alphas"] == list(dh.PROBE_ALPHAS) and report["max_rel"] == max(report["per_alpha"])
        assert report["tolerance"] == TOLERANCE and report["tf32"] is False

    def test_a_list_output_is_reduced_to_index_zero_and_the_report_names_it(self) -> None:
        inputs = keras.Input((None, None, 3))
        full = keras.layers.Conv2D(3, 3, padding="same", use_bias=False, name="full")(inputs)
        aux = keras.layers.Conv2D(3, 3, padding="same", use_bias=True, name="aux")(inputs)
        model = keras.Model(inputs, [full, aux])
        _plant(model, ("bias",))
        report = dh.probe_numeric(model, _images())
        assert report["output_index"] == 0
        assert report["passed"], "the biased aux output at index 1 must not be read"
        assert "output_index" not in dh.probe_numeric(_chain(), _images())

    def test_a_dict_output_is_refused(self) -> None:
        inputs = keras.Input((None, None, 3))
        model = keras.Model(inputs, {"a": keras.layers.Conv2D(3, 1, use_bias=False)(inputs)})
        with pytest.raises(ValueError, match="dict"):
            dh.probe_numeric(model, _images())

    def test_the_tolerance_decides_passed(self) -> None:
        model = _chain(first_bias=True)
        _plant(model, ("bias",))
        x = _images()
        value = dh.probe_numeric(model, x)["max_rel"]
        assert value > TOLERANCE
        assert dh.probe_numeric(model, x, tolerance=value * 2)["passed"] is True
        assert dh.probe_numeric(model, x, tolerance=value / 2)["passed"] is False

    def test_a_dead_model_never_reads_as_homogeneous(self) -> None:
        class Zero:
            def __call__(self, x, training=False):
                return np.zeros_like(x)

        report = dh.probe_numeric(Zero(), _images())
        assert report["max_rel"] == math.inf and report["passed"] is False

    def test_a_non_finite_output_fails(self) -> None:
        class Nan:
            def __call__(self, x, training=False):
                return np.full_like(x, np.nan)

        assert dh.probe_numeric(Nan(), _images())["passed"] is False

    def test_the_model_is_called_at_inference(self) -> None:
        seen: List[Any] = []

        class Spy:
            def __call__(self, x, training=False):
                seen.append(training)
                return x

        dh.probe_numeric(Spy(), _images())
        assert seen and all(flag is False for flag in seen)

    def test_an_unbuilt_model_is_refused_by_the_scan(self) -> None:
        with pytest.raises(ValueError, match="built"):
            dh.scan_static(keras.Sequential([keras.layers.Dense(2)]))


# ---------------------------------------------------------------------
# TF32 and random state
# ---------------------------------------------------------------------


class TestTf32:
    @pytest.mark.parametrize("prior", [True, False])
    def test_tf32_is_off_inside_the_probe_and_restored_after(self, prior) -> None:
        inside: List[bool] = []

        class Spy:
            def __call__(self, x, training=False):
                inside.append(bool(tf.config.experimental.tensor_float_32_execution_enabled()))
                return x

        with tf32_execution(prior):
            report = dh.probe_numeric(Spy(), _images())
            assert tf.config.experimental.tensor_float_32_execution_enabled() is prior
        assert inside and not any(inside)
        assert report["tf32"] is False

    @pytest.mark.parametrize("prior", [True, False])
    def test_tf32_is_restored_after_a_raising_probe(self, prior) -> None:
        class Boom:
            def __call__(self, x, training=False):
                raise RuntimeError("model failed")

        with tf32_execution(prior):
            with pytest.raises(RuntimeError, match="model failed"):
                dh.probe_numeric(Boom(), _images())
            assert tf.config.experimental.tensor_float_32_execution_enabled() is prior

    def test_the_context_manager_restores_after_a_raise(self) -> None:
        with tf32_execution(True):
            with pytest.raises(KeyError):
                with dh.tf32_off():
                    assert tf.config.experimental.tensor_float_32_execution_enabled() is False
                    raise KeyError("x")
            assert tf.config.experimental.tensor_float_32_execution_enabled() is True


def _dropout_model() -> keras.Model:
    inputs = keras.Input((8, 8, 3))
    x = keras.layers.Conv2D(4, 3, use_bias=False, padding="same")(inputs)
    x = keras.layers.Dropout(0.5)(x)
    return keras.Model(inputs, keras.layers.Conv2D(3, 1, use_bias=False)(x))


def _random_state(model: keras.Model) -> Dict[str, Any]:
    global_generator = keras.src.backend.common.global_state.get_global_attribute("global_seed_generator")
    return {
        "generators": [np.array(g.state.numpy()) for g in seed_generators(model)],
        "global": None if global_generator is None else np.array(global_generator.state.numpy()),
        "python": random.getstate(),
        "numpy_keys": np.random.get_state()[1].copy(),
        "numpy_pos": np.random.get_state()[2],
    }


def _assert_state_equal(left: Dict[str, Any], right: Dict[str, Any]) -> None:
    assert len(left["generators"]) == len(right["generators"])
    assert all(np.array_equal(a, b) for a, b in zip(left["generators"], right["generators"]))
    assert (left["global"] is None) == (right["global"] is None)
    if left["global"] is not None:
        assert np.array_equal(left["global"], right["global"])
    assert left["python"] == right["python"]
    assert np.array_equal(left["numpy_keys"], right["numpy_keys"]) and left["numpy_pos"] == right["numpy_pos"]


class TestRandomState:
    def _armed_model(self) -> keras.Model:
        """A model whose generators and the global generator are NOT at their initial state."""
        model = _dropout_model()
        x = np.ones((2, 8, 8, 3), "float32")
        model(x, training=True)
        model(x, training=True)
        keras.random.normal((1,))   # creates and advances Keras' global generator
        return model

    def test_the_clone_work_does_move_the_python_random_state(self) -> None:
        """Positive control: without the restore the clone WOULD move state, so the guard below is not vacuous."""
        model = self._armed_model()
        assert len(seed_generators(model)) >= 1
        before = random.getstate()
        dh._build_probe_clone(model, 0)
        assert random.getstate() != before

    def test_init_probe_clone_leaves_every_random_state_where_it_found_it(self) -> None:
        model = self._armed_model()
        before = _random_state(model)
        assert before["global"] is not None and any(state[1] > 0 for state in before["generators"])
        dh.init_probe_clone(model)
        _assert_state_equal(before, _random_state(model))

    def test_the_restore_also_runs_when_the_clone_raises(self) -> None:
        model = self._armed_model()
        before = _random_state(model)
        original = dh._build_probe_clone

        def failing(m: keras.Model, seed: int):
            random.random()              # move a state, then fail
            raise RuntimeError("clone failed")

        dh._build_probe_clone = failing
        try:
            with pytest.raises(RuntimeError, match="clone failed"):
                dh.init_probe_clone(model)
        finally:
            dh._build_probe_clone = original
        _assert_state_equal(before, _random_state(model))

    def test_probing_the_clone_leaves_the_model_and_its_generators_untouched(self) -> None:
        model = self._armed_model()
        weights = [np.array(w) for w in model.get_weights()]
        before = _random_state(model)
        dh.probe_numeric(dh.init_probe_clone(model), np.ones((2, 8, 8, 3), "float32"))
        _assert_state_equal(before, _random_state(model))
        assert all(np.array_equal(a, b) for a, b in zip(weights, model.get_weights()))
