"""A ``ConvStem`` computes in its dtype policy, whichever way the policy is set.

Two ways to set a policy are covered, because they fail differently:

* the GLOBAL policy (``dtype_policy`` fixture), which every sub-layer reads by
  default, so it cannot see a stem that forgets to hand its own policy on;
* the stem's own ``dtype=`` argument (a PER-LAYER policy), which only reaches a
  sub-layer if the stem passes it. Measured before it was fixed:
  ``ConvStem(4, dtype="mixed_float16")`` reported ``compute_dtype`` float16
  while ``conv1``, ``norm1``, the activation and the pool computed in float32,
  so the output was float32.

The second class sweeps every normalization type the factory can build at the
stem's shape, because the per-layer rule is only as good as the least careful
norm: a norm that builds an inner layer must hand it the same policy.

``float32`` is the control in every per-layer arm: it is the ambient policy, so
it passes with or without the fix and shows the assertions do not reject
everything.
"""

import inspect
import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.activations.exact_activations import exact_gelu
from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.layers.norms.factory import (
    _TYPE_TO_CLASS, create_normalization_layer)

from . import stem_instruments
from .stem_instruments import B, random_input, to_numpy

# The ``rng`` fixture is owned by ``stem_instruments``; binding it here is what
# makes pytest find it in this module.
rng = stem_instruments.rng

# The module scopes TF32 off so the exact-equality comparisons are made in true
# float32 on a TF32-capable GPU as well as on CPU.
pytestmark = pytest.mark.usefixtures("tf32_disabled")


# ---------------------------------------------------------------------
# The global policy
# ---------------------------------------------------------------------

class TestTheGlobalPolicyReachesTheForwardPass:
    def test_forward_pass_is_finite_under_every_policy(self, dtype_policy, rng):
        layer = ConvStem(filters=6, num_convs=2, use_pooling=True, pool_size=3,
                         pool_strides=2, pool_padding="same",
                         name=f"dtype_{dtype_policy}")
        out = layer(random_input(rng, (B, 8, 8, 4)), training=False)
        assert np.isfinite(to_numpy(out).astype("float64")).all()
        mixed = dtype_policy == "mixed_float16"
        assert (layer.compute_dtype != layer.variable_dtype) == mixed
        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype



# ---------------------------------------------------------------------
# A per-layer dtype policy reaches every sub-layer
# ---------------------------------------------------------------------

def _every_sublayer(stem):
    """``(attribute, layer)`` for each sub-layer a fully configured stem owns."""
    names = ("conv1", "norm1", "act1", "conv2", "norm2", "act2", "pool")
    return [(name, getattr(stem, name)) for name in names]


class TestAPerLayerDtypePolicyReachesEverySubLayer:
    """``ConvStem(dtype=...)`` is a per-layer policy, not the global one.

    Measured before the fix: ``ConvStem(4, dtype="mixed_float16")`` reported
    ``compute_dtype`` float16 while ``conv1``, ``norm1``, the activation and the
    pool computed in float32, so the output was float32 (and ``dtype="float64"``
    behaved the same way). ``TestTheGlobalPolicyReachesTheForwardPass`` sets the GLOBAL policy, which every
    sub-layer reads by default, so it cannot see this. ``float32`` is the control:
    it is the ambient policy, so it passes with or without the fix and shows the
    assertions are not rejecting everything.
    """

    @staticmethod
    def _stem(policy, **kwargs):
        return ConvStem(filters=6, num_convs=2, use_pooling=True, pool_size=3,
                        pool_strides=2, pool_padding="same", dtype=policy,
                        name=f"per_layer_{policy}", **kwargs)

    @pytest.mark.parametrize("policy", ["float32", "mixed_float16", "float64"])
    def test_every_sublayer_follows_the_stems_policy(self, policy):
        stem = self._stem(policy)
        for attribute, layer in _every_sublayer(stem):
            assert layer is not None, attribute
            assert layer.compute_dtype == stem.compute_dtype, (
                f"{attribute}: compute {layer.compute_dtype} != stem "
                f"{stem.compute_dtype} under dtype={policy!r}")
            assert layer.variable_dtype == stem.variable_dtype, attribute

    @pytest.mark.parametrize("policy", ["float32", "mixed_float16", "float64"])
    def test_the_output_is_in_the_stems_compute_dtype(self, policy, rng):
        stem = self._stem(policy)
        out = stem(random_input(rng, (B, 8, 8, 4)), training=False)
        assert keras.backend.standardize_dtype(out.dtype) == stem.compute_dtype
        assert np.isfinite(to_numpy(out).astype("float64")).all()

    @pytest.mark.parametrize("activation, kwargs", [
        ("mish", {}),
        ("relu", {"negative_slope": 0.25}),
        ("tanh", {}),
        (exact_gelu, {}),
    ], ids=["registry_key", "registry_key_with_kwargs", "keras_name", "callable"])
    def test_every_activation_form_follows_the_policy(self, activation, kwargs):
        stem = self._stem("mixed_float16", activation_type=activation,
                          activation_kwargs=kwargs)
        assert stem.act1.compute_dtype == "float16"
        assert stem.act2.compute_dtype == "float16"

    def test_a_non_default_norm_type_follows_the_policy(self):
        stem = self._stem("float64", normalization_type="rms_norm")
        assert stem.norm1.compute_dtype == "float64"
        assert stem.norm2.compute_dtype == "float64"

    def test_an_explicit_dtype_in_the_norm_kwargs_wins(self):
        stem = self._stem("mixed_float16", normalization_kwargs={"dtype": "float32"})
        assert stem.norm1.compute_dtype == "float32"
        assert stem.conv1.compute_dtype == "float16"

    def test_the_policy_survives_a_config_round_trip(self):
        stem = self._stem("mixed_float16")
        rebuilt = ConvStem.from_config(json.loads(json.dumps(stem.get_config())))
        for attribute, layer in _every_sublayer(rebuilt):
            assert layer.compute_dtype == "float16", attribute



# ---------------------------------------------------------------------
# Every normalization type the stem can build, at every policy
# ---------------------------------------------------------------------

# Types the stem cannot use. The stem needs its norm to preserve the shape of
# its input (module docstring); these three REDUCE the normalized axis away
# (norms/factory.py, "Three types are not shape-preserving"). The census test
# below measures each one on a rank-4 input, so a listed type that became
# shape-preserving fails there instead of staying excluded unnoticed.
_NORMS_THE_STEM_CANNOT_BUILD = {
    "decoupled_max_logit":
        "returns a 3-tuple (score, MaxCosine, MaxNorm), each (B, H, W): the "
        "channel axis is reduced away",
    "dml_plus_focal":
        "returns one (B, H, W) tensor: the channel axis is reduced away",
    "dml_plus_center":
        "returns a 2-tuple, (B, H, W) and (B, H, W, 1): the channel axis is "
        "reduced away",
}

# Extra constructor arguments a swept type needs beyond its name. EMPTY by
# measurement (2026-10-26): all 16 swept types build from the name alone. The
# only registry types with a required constructor argument are the two
# ``dml_plus_*`` ones (``model_type``, which the factory forces), and both are
# in the exclusion table above. A new type that needs an argument fails
# ``test_the_table_supplies_every_required_argument`` until it is listed here.
_NORM_EXTRA_KWARGS = {}

_SWEPT_NORMS = sorted(set(_TYPE_TO_CLASS) - set(_NORMS_THE_STEM_CANNOT_BUILD))


class TestEveryNormTypeTheStemCanBuild:
    """``ConvStem(dtype=...)`` with each factory norm, at float32 / mixed / float64.

    Measured before the fix (CPU, ``ConvStem(8, num_convs=2, dtype=policy)``):
    ``band_logit_norm`` raised ``InvalidArgumentError`` (Mul dtype mismatch) at
    ``float64`` and ``mixed_float16``, and ``adaptive_band_rms`` /
    ``zero_centered_adaptive_band_rms_norm`` ran but built their inner ``Dense``
    at float32. The other 14 swept types were already correct. The earlier
    per-layer test covered ``rms_norm`` only, so it could not see any of them.
    ``float32`` is the control: it is the ambient policy and passes with or
    without the fix.
    """

    def test_the_sweep_covers_the_registry_and_is_not_vacuous(self):
        assert set(_NORMS_THE_STEM_CANNOT_BUILD) <= set(_TYPE_TO_CLASS)
        assert set(_SWEPT_NORMS) | set(_NORMS_THE_STEM_CANNOT_BUILD) == set(
            _TYPE_TO_CLASS)
        assert not set(_SWEPT_NORMS) & set(_NORMS_THE_STEM_CANNOT_BUILD)
        # Derived from the registry, but pinned: a registry change must be read
        # against the exclusion table instead of being swept (or skipped) blind.
        assert len(_SWEPT_NORMS) == 16, _SWEPT_NORMS
        assert set(_NORM_EXTRA_KWARGS) <= set(_SWEPT_NORMS)

    @pytest.mark.parametrize("norm_type", sorted(_NORMS_THE_STEM_CANNOT_BUILD))
    def test_each_excluded_type_really_leaves_the_shape_contract(self, norm_type):
        """The exclusion is measured, not asserted by name."""
        assert _NORMS_THE_STEM_CANNOT_BUILD[norm_type]
        out = create_normalization_layer(norm_type)(
            np.ones((B, 8, 8, 4), "float32"))
        if isinstance(out, (tuple, list)):
            return
        assert tuple(out.shape) != (B, 8, 8, 4), (
            f"{norm_type} now preserves the shape: sweep it instead")

    @pytest.mark.parametrize("norm_type", _SWEPT_NORMS)
    def test_the_table_supplies_every_required_argument(self, norm_type):
        parameters = inspect.signature(_TYPE_TO_CLASS[norm_type].__init__).parameters
        required = {
            name for name, parameter in parameters.items()
            if name != "self" and parameter.default is parameter.empty
            and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD,
                                   parameter.KEYWORD_ONLY)}
        assert required <= set(_NORM_EXTRA_KWARGS.get(norm_type, {})), (
            f"{norm_type} needs {required}: add it to _NORM_EXTRA_KWARGS")

    @pytest.mark.parametrize("policy", ["float32", "mixed_float16", "float64"])
    @pytest.mark.parametrize("norm_type", _SWEPT_NORMS)
    def test_the_stem_computes_in_its_policy(self, norm_type, policy, rng):
        stem = ConvStem(
            filters=6, num_convs=2, use_pooling=True, pool_size=3,
            pool_strides=2, pool_padding="same", normalization_type=norm_type,
            dtype=policy, name=f"sweep_{norm_type}_{policy}",
            normalization_kwargs=dict(_NORM_EXTRA_KWARGS.get(norm_type, {})))
        x = random_input(rng, (B, 8, 8, 4))
        out = stem(x, training=False)
        assert not isinstance(out, (tuple, list)), type(out)
        assert tuple(out.shape) == tuple(stem.compute_output_shape(x.shape))
        assert keras.backend.standardize_dtype(out.dtype) == stem.compute_dtype
        assert np.isfinite(to_numpy(out).astype("float64")).all()
        # Every layer under the stem, nested ones included: an inner layer a
        # norm builds for itself is one of them.
        walked = list(stem._flatten_layers(include_self=False))
        assert len(walked) >= 7, [layer.name for layer in walked]
        for layer in walked:
            assert (layer.compute_dtype, layer.variable_dtype) == (
                stem.compute_dtype, stem.variable_dtype), (
                f"{layer.name} ({type(layer).__name__}): compute "
                f"{layer.compute_dtype} / variable {layer.variable_dtype} under "
                f"ConvStem(dtype={policy!r}, normalization_type={norm_type!r})")
