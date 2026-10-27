"""``ConvUNextStem(dtype=...)`` reaches the ``ConvStem`` it owns.

``ConvStem`` carries a per-layer dtype policy to its convolutions, normalization,
activation and pool (``test_the_stem_follows_its_own_dtype_policy.py`` under
``tests/test_layers/test_conv_blocks/``), but ``ConvUNextStem`` built that
``ConvStem`` without passing its own policy on:
``ConvUNextStem(dtype="float64")`` reported ``compute_dtype`` float64 while the
inner stem and every sub-layer under it computed in float32, so the output came
back float32. The same defect one level up from the one fixed in ``ConvStem``.

``float32`` is the control: it is the ambient policy, so it passes with or
without the fix and shows the assertions do not reject everything. The global
policy cannot see this defect because every layer reads it by default, so each
arm sets the policy on the layer only.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.convunext.model import ConvUNextStem

POLICIES = ["float32", "mixed_float16", "float64"]
#: ``(attribute, ...)`` of the inner ConvStem a ConvUNextStem configures.
SUBLAYERS = ("conv1", "norm1", "act1")


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
    assert outer.stem.act1.compute_dtype == "float64"
    out = outer(_x(), training=False)
    assert keras.backend.standardize_dtype(out.dtype) == "float64"


def test_the_policy_survives_a_config_round_trip():
    outer = _stem("mixed_float16")
    rebuilt = ConvUNextStem.from_config(json.loads(json.dumps(outer.get_config())))
    assert rebuilt.compute_dtype == "float16"
    for attribute in SUBLAYERS:
        assert getattr(rebuilt.stem, attribute).compute_dtype == "float16", attribute
