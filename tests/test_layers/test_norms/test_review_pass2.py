"""Regression tests pinning the second-pass norms review fixes.

Covers (plan_2026-06-15_3028e33c):
- A1: DMLPlus(model_type='center').compute_output_shape matches call output.
- A2/A3: factory validate <-> create agreement (dynamic_tanh params, max_band_width bounds).
- B1: AdaptiveBandRMS / ZeroCenteredAdaptiveBandRMS fail-loud on a None normalized axis.
- B3: BandRMS / AdaptiveBandRMS reject an invalid axis TYPE at construction.
"""

import numpy as np
import pytest

from dl_techniques.layers.norms.max_logit_norm import DMLPlus
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS
from dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm import (
    ZeroCenteredAdaptiveBandRMS,
)
from dl_techniques.layers.norms.factory import (
    validate_normalization_config,
    create_normalization_layer,
)


class TestDMLPlusOutputShape:
    """A1: center model norm_factor is the keepdims norm, not the full input shape."""

    @pytest.mark.parametrize("shape,axis", [
        ((8, 5), -1),
        ((4, 3, 6), -1),
        ((4, 6, 3), 1),
    ])
    def test_center_compute_output_shape_matches_call(self, shape, axis):
        layer = DMLPlus(model_type="center", axis=axis)
        outputs = layer(np.random.randn(*shape).astype("float32"))
        actual = tuple(tuple(o.shape) for o in outputs)
        declared = tuple(tuple(d) for d in layer.compute_output_shape(shape))
        assert actual == declared


class TestFactoryValidateCreateAgreement:
    """A2/A3: validate must accept exactly what create accepts."""

    def _create_accepts(self, ntype, **kw):
        try:
            create_normalization_layer(ntype, **kw)
            return True
        except Exception:
            return False

    def _validate_accepts(self, ntype, **kw):
        try:
            validate_normalization_config(ntype, **kw)
            return True
        except ValueError:
            return False

    @pytest.mark.parametrize("kwargs", [
        {"bias_initializer": "zeros"},
        {"kernel_regularizer": "l2"},
        {"bias_regularizer": "l2"},
        {"kernel_constraint": "non_neg"},
        {"bias_constraint": "non_neg"},
    ])
    def test_dynamic_tanh_extra_params_agree(self, kwargs):
        assert (self._validate_accepts("dynamic_tanh", **kwargs)
                == self._create_accepts("dynamic_tanh", **kwargs) is True)

    @pytest.mark.parametrize("ntype", [
        "band_rms", "adaptive_band_rms", "band_logit_norm",
        "zero_centered_band_rms_norm", "zero_centered_adaptive_band_rms_norm",
    ])
    @pytest.mark.parametrize("mbw,ok", [(0.5, True), (1.5, False), (1.0, False), (0.0, False)])
    def test_max_band_width_bounds_agree(self, ntype, mbw, ok):
        v = self._validate_accepts(ntype, max_band_width=mbw)
        c = self._create_accepts(ntype, max_band_width=mbw)
        assert v == c == ok


class TestAdaptiveNoneDimFailLoud:
    """B1: a None normalized-axis size must raise, not silently mis-size the Dense."""

    @pytest.mark.parametrize("cls", [AdaptiveBandRMS, ZeroCenteredAdaptiveBandRMS])
    def test_none_normalized_axis_raises(self, cls):
        layer = cls(axis=-1)
        with pytest.raises(ValueError):
            layer.build((None, None))

    @pytest.mark.parametrize("cls", [AdaptiveBandRMS, ZeroCenteredAdaptiveBandRMS])
    def test_static_dim_builds_fine(self, cls):
        layer = cls(axis=-1)
        layer.build((None, 8))  # batch dynamic, feature static -> OK
        assert layer.built


class TestAxisTypeValidation:
    """B3: invalid axis TYPE rejected at construction (parity across the family)."""

    @pytest.mark.parametrize("cls", [BandRMS, AdaptiveBandRMS])
    def test_bad_axis_type_raises(self, cls):
        with pytest.raises(TypeError):
            cls(axis="last")

    @pytest.mark.parametrize("cls", [BandRMS, AdaptiveBandRMS])
    def test_valid_axis_accepted(self, cls):
        cls(axis=-1)
        cls(axis=[1, 2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
