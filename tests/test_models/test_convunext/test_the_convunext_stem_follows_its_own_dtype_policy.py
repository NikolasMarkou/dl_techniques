"""``ConvUNextStem(dtype=...)`` reaches the ``ConvStem`` it owns.

Pass-2 review NOTE 7 of plan-2026-10-26T125959-7af851e5. ``ConvStem`` carries a
per-layer dtype policy to its convolutions, normalization, activation and pool
(its own suite, ``TestAPerLayerDtypePolicyReachesEverySubLayer``), but
``ConvUNextStem`` built that ``ConvStem`` without passing its own policy on:
``ConvUNextStem(dtype="float64")`` reported ``compute_dtype`` float64 while the
inner stem and every sub-layer under it computed in float32, so the output came
back float32. The same defect one level up from the one fixed in ``ConvStem``.

``float32`` is the control: it is the ambient policy, so it passes with or
without the fix and shows the assertions do not reject everything. The global
policy cannot see this defect because every layer reads it by default, so each
arm sets the policy on the layer only.

The second half pins what ``ConvUNextStem`` documents about its ``activation``
argument. It is handed to ``ConvStem``, whose rules apply unchanged: a string, a
callable, or a Layer that is unbuilt (holds no weights) and whose class Keras can
locate; a built Layer or an unlocatable class is a ``ValueError`` at
construction. The messages were measured through ``ConvUNextStem``, not only
through ``ConvStem``.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.activations.exact_activations import exact_gelu
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.convunext.model import ConvUNextStem

POLICIES = ["float32", "mixed_float16", "float64"]
#: ``(attribute, ...)`` of the inner ConvStem a ConvUNextStem configures.
SUBLAYERS = ("conv1", "norm1", "activation1")


def _stem(policy: str, **kwargs) -> ConvUNextStem:
    return ConvUNextStem(filters=6, kernel_size=3, dtype=policy, **kwargs)


def _x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 8, 8, 3)).astype("float32")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("norm", ["global_response_norm", "layer_norm"])
def test_the_inner_stem_and_every_sublayer_follow_the_policy(policy, norm):
    outer = _stem(policy, stem_normalization=norm)
    assert isinstance(outer.stem, ConvStem)
    assert outer.stem.compute_dtype == outer.compute_dtype, (
        f"the inner ConvStem computes in {outer.stem.compute_dtype}, the "
        f"ConvUNextStem in {outer.compute_dtype} under dtype={policy!r}"
    )
    assert outer.stem.variable_dtype == outer.variable_dtype
    for attribute in SUBLAYERS:
        layer = getattr(outer.stem, attribute)
        assert layer is not None, attribute
        assert layer.compute_dtype == outer.compute_dtype, (
            f"{attribute}: compute {layer.compute_dtype} != "
            f"{outer.compute_dtype} under dtype={policy!r}"
        )
        assert layer.variable_dtype == outer.variable_dtype, attribute


@pytest.mark.parametrize("policy", POLICIES)
def test_the_output_is_in_the_layers_compute_dtype(policy):
    outer = _stem(policy)
    out = outer(_x(), training=False)
    assert keras.backend.standardize_dtype(out.dtype) == outer.compute_dtype
    assert np.isfinite(keras.ops.convert_to_numpy(out).astype("float64")).all()


@pytest.mark.parametrize("activation", ["gelu", "leaky_relu"])
def test_the_exact_activation_forms_follow_the_policy(activation):
    """``substitute_exact`` turns these strings into the exact callables
    (D-277); the callable path of ``ConvStem`` must carry the policy too."""
    outer = _stem("float64", activation=activation)
    assert outer.stem.activation1.compute_dtype == "float64"
    out = outer(_x(), training=False)
    assert keras.backend.standardize_dtype(out.dtype) == "float64"


def test_the_policy_survives_a_config_round_trip():
    outer = _stem("mixed_float16")
    rebuilt = ConvUNextStem.from_config(json.loads(json.dumps(outer.get_config())))
    assert rebuilt.compute_dtype == "float16"
    for attribute in SUBLAYERS:
        assert getattr(rebuilt.stem, attribute).compute_dtype == "float16", attribute


# ---------------------------------------------------------------------
# The documented refusals of the ``activation`` argument
# ---------------------------------------------------------------------


def _unlocatable_activation() -> keras.layers.Layer:
    """A custom activation Layer Keras cannot locate by name.

    Defined inside a function on purpose: Keras resolves a serialized class by
    ``module`` + ``class_name``, so a module-level class IS found again and
    would not exercise the failure.
    """

    class UnlocatableActivation(keras.layers.Layer):
        def call(self, inputs):
            return keras.ops.relu(inputs)

    return UnlocatableActivation()


def test_a_built_layer_holding_weights_is_refused():
    built = keras.layers.PReLU()
    built.build((None, 4))
    assert built.weights, "control: the PReLU holds an alpha weight"
    with pytest.raises(ValueError, match="PReLU.*already holds weights"):
        ConvUNextStem(filters=4, activation=built)


def test_a_layer_whose_class_cannot_be_located_is_refused_by_name():
    with pytest.raises(ValueError, match="UnlocatableActivation.*cannot clone") as caught:
        ConvUNextStem(filters=4, activation=_unlocatable_activation())
    assert "register" in str(caught.value)


def test_a_registered_unbuilt_layer_is_accepted_and_rebuilt_per_site():
    """Control for the two refusals: the same PReLU, unbuilt, is fine."""
    outer = ConvUNextStem(filters=4, activation=keras.layers.PReLU())
    assert isinstance(outer.stem.activation1, keras.layers.PReLU)
    out = outer(_x(), training=False)
    assert tuple(out.shape) == (2, 8, 8, 4)


def test_a_registered_callable_activation_is_accepted():
    outer = ConvUNextStem(filters=4, activation=exact_gelu)
    assert isinstance(outer.stem.activation1, keras.layers.Activation)
    assert tuple(outer(_x(), training=False).shape) == (2, 8, 8, 4)


@pytest.mark.parametrize("activation", ["gelu", "relu", "linear"])
def test_a_string_activation_is_accepted(activation):
    outer = ConvUNextStem(filters=4, activation=activation)
    assert tuple(outer(_x(), training=False).shape) == (2, 8, 8, 4)
