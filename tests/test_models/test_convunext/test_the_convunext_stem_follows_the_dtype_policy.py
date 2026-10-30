"""The level-0 ``ConvStem`` of ``create_convunext`` computes in the dtype policy.

``create_convunext`` builds the level-0 stem as a ``ConvStem`` named
``encoder_level_0_stem`` and states no ``dtype`` of its own, so it reads the
policy in force when the graph is built. ``ConvStem`` carries that policy to its
convolution, normalization and activation
(``tests/test_layers/test_conv_blocks/test_the_stem_follows_its_own_dtype_policy.py``).
What this file pins is the builder's side of it: the stem it creates, and every
sub-layer under it, computes in the policy and the graph's output comes back in
it.

History: until 2026-10-30 the stem was a ``ConvUNextStem`` wrapper around a
``ConvStem``, and the wrapper once built its ``ConvStem`` without passing its own
policy on, so ``ConvUNextStem(dtype="float64")`` reported float64 while the stem
under it computed in float32. The wrapper is deleted. A builder that pinned a
dtype on the ``ConvStem`` (``dtype="float32"``) would fail the ``mixed_float16``
and ``float64`` arms below the same way.

``float32`` is the control: it is the ambient policy, so it passes with or
without a dtype leak and shows the assertions do not reject everything.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.convunext.model import create_convunext

POLICIES = ["float32", "mixed_float16", "float64"]
#: The sub-layers of the level-0 ConvStem.
SUBLAYERS = ("conv1", "norm1", "act1")
#: ``(variable_dtype, compute_dtype)`` each policy must give the stem.
EXPECTED = {
    "float32": ("float32", "float32"),
    "mixed_float16": ("float32", "float16"),
    "float64": ("float64", "float64"),
}


@pytest.fixture
def policy_scope():
    """Set the global dtype policy for one test and restore it afterwards."""
    before = keras.config.dtype_policy()

    def _set(policy: str) -> None:
        keras.config.set_dtype_policy(policy)

    yield _set
    keras.config.set_dtype_policy(before)


def _model(**kwargs) -> keras.Model:
    return create_convunext(
        input_shape=(8, 8, 3), depth=2, initial_filters=6, blocks_per_level=1,
        stem_kernel_size=3, drop_path_rate=0.0, **kwargs)


def _x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 8, 8, 3)).astype("float32")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("norm", ["global_response_norm", "layer_norm"])
def test_the_stem_and_every_sublayer_follow_the_policy(policy_scope, policy, norm):
    policy_scope(policy)
    stem = _model(stem_normalization=norm).get_layer("encoder_level_0_stem")
    variable_dtype, compute_dtype = EXPECTED[policy]
    assert isinstance(stem, ConvStem)
    assert (stem.variable_dtype, stem.compute_dtype) == (variable_dtype, compute_dtype)
    for attribute in SUBLAYERS:
        layer = getattr(stem, attribute)
        assert layer is not None, attribute
        assert layer.compute_dtype == compute_dtype, (
            f"{attribute}: compute {layer.compute_dtype} != {compute_dtype} "
            f"under policy {policy!r}"
        )
        assert layer.variable_dtype == variable_dtype, attribute


@pytest.mark.parametrize("policy", POLICIES)
def test_the_output_is_in_the_policys_compute_dtype(policy_scope, policy):
    policy_scope(policy)
    out = _model()(_x(), training=False)
    assert keras.backend.standardize_dtype(out.dtype) == EXPECTED[policy][1]
    assert np.isfinite(keras.ops.convert_to_numpy(out).astype("float64")).all()


@pytest.mark.parametrize("activation", ["gelu", "leaky_relu"])
def test_the_exact_activation_forms_follow_the_policy(policy_scope, activation):
    """``substitute_exact`` turns these strings into the exact callables
    (D-277); the callable path of ``ConvStem`` must carry the policy too."""
    policy_scope("float64")
    model = _model(stem_activation=activation)
    stem = model.get_layer("encoder_level_0_stem")
    assert stem.act1.compute_dtype == "float64"
    out = model(_x(), training=False)
    assert keras.backend.standardize_dtype(out.dtype) == "float64"


def test_the_policy_survives_a_keras_round_trip(policy_scope, tmp_path):
    policy_scope("mixed_float16")
    path = str(tmp_path / "mixed.keras")
    _model().save(path)
    loaded = keras.models.load_model(path)
    stem = loaded.get_layer("encoder_level_0_stem")
    assert stem.compute_dtype == "float16"
    for attribute in SUBLAYERS:
        assert getattr(stem, attribute).compute_dtype == "float16", attribute
