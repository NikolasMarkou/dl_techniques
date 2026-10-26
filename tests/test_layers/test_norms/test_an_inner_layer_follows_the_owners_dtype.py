"""Guard: a norm's INNER layer follows the dtype policy of the norm that owns it.

Three norms build a ``keras.layers`` sub-layer inside themselves:

* ``BandLogitNorm`` creates a ``LayerNormalization`` in ``__init__``;
* ``AdaptiveBandRMS`` and ``ZeroCenteredAdaptiveBandRMS`` create a ``Dense``
  (``band_dense``) in ``build()``.

None of them passed ``dtype=`` to that sub-layer, so the sub-layer read the
GLOBAL policy instead of its owner's. That is invisible whenever the owner's
policy IS the global one (the guide's ``dtype_policy`` fixture sets only the
global policy, so every global-policy arm passes) and wrong as soon as the owner
carries its own: ``BandLogitNorm(dtype="float64")`` then multiplies a float64
unit-normalized tensor by a float32 scale and raises ``InvalidArgumentError``
(Mul dtype mismatch). ``ConvStem(8, dtype="float64",
normalization_type="band_logit_norm")`` hit exactly this (review pass 2, NOTE 4).

Measured before the fix (CPU, keras 3.8, input ``(2, 5, 8)``):

* ``BandLogitNorm`` with ``dtype`` ``float64`` and ``mixed_float16``: forward
  raises ``InvalidArgumentError``; the inner ``LayerNormalization`` reports
  ``compute_dtype == "float32"``.
* ``AdaptiveBandRMS`` / ``ZeroCenteredAdaptiveBandRMS`` with the same two
  policies: the forward pass runs, but ``band_dense`` reports ``float32``, so
  under ``float64`` the band scale is produced at float32 precision.

Each arm has its control: the ``float32`` policy passes with or without the fix
(it is the ambient policy), and the global-policy arm passes with or without it
(the inner layer reads the global policy), so neither can be what turns red.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS
from dl_techniques.layers.norms.band_logit_norm import BandLogitNorm
from dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm import (
    ZeroCenteredAdaptiveBandRMS,
)

POLICIES = ("float32", "mixed_float16", "float64")

# norm class -> attribute that holds the sub-layer it creates.
_INNER = {
    BandLogitNorm: "norm",
    AdaptiveBandRMS: "dense_layer",
    ZeroCenteredAdaptiveBandRMS: "dense_layer",
}
_IDS = [cls.__name__ for cls in _INNER]

_SHAPE = (2, 5, 8)


def _input():
    return np.random.default_rng(0).standard_normal(_SHAPE).astype("float32")


def _dtype_name(tensor):
    return keras.backend.standardize_dtype(tensor.dtype)


class TestAnInnerLayerFollowsAPerLayerPolicy:
    """``Norm(dtype=policy)``: the owner's policy, not the global one."""

    @pytest.mark.parametrize("policy", POLICIES)
    @pytest.mark.parametrize("cls", list(_INNER), ids=_IDS)
    def test_the_inner_layer_has_the_owners_policy(self, cls, policy):
        norm = cls(dtype=policy)
        norm.build(_SHAPE)
        inner = getattr(norm, _INNER[cls])
        assert inner is not None
        assert (inner.compute_dtype, inner.variable_dtype) == (
            norm.compute_dtype, norm.variable_dtype), (
            f"{cls.__name__}(dtype={policy!r}): inner {type(inner).__name__} has "
            f"compute {inner.compute_dtype} / variable {inner.variable_dtype}, "
            f"owner has {norm.compute_dtype} / {norm.variable_dtype}")

    @pytest.mark.parametrize("policy", POLICIES)
    @pytest.mark.parametrize("cls", list(_INNER), ids=_IDS)
    def test_the_forward_pass_runs_and_returns_the_compute_dtype(self, cls, policy):
        norm = cls(dtype=policy)
        out = norm(_input(), training=False)
        assert _dtype_name(out) == norm.compute_dtype
        assert np.isfinite(keras.ops.convert_to_numpy(out).astype("float64")).all()

    def test_the_hazard_is_real(self):
        """The two non-float32 policies really are different from the ambient one.

        Without this the arms above could pass because every policy collapsed to
        float32 (13.2.6, part 1: prove the hazard before testing the defence).
        """
        assert keras.mixed_precision.global_policy().name == "float32"
        mixed = BandLogitNorm(dtype="mixed_float16")
        assert (mixed.compute_dtype, mixed.variable_dtype) == ("float16", "float32")
        assert BandLogitNorm(dtype="float64").compute_dtype == "float64"


class TestAnInnerLayerFollowsTheGlobalPolicy:
    """Control: under the GLOBAL policy the inner layer already agreed."""

    @pytest.mark.parametrize("cls", list(_INNER), ids=_IDS)
    def test_the_global_policy_reaches_the_inner_layer(self, cls, dtype_policy):
        norm = cls()
        out = norm(_input(), training=False)
        inner = getattr(norm, _INNER[cls])
        assert inner.compute_dtype == norm.compute_dtype
        assert _dtype_name(out) == norm.compute_dtype
        assert np.isfinite(keras.ops.convert_to_numpy(out).astype("float64")).all()
