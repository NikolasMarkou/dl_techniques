"""Unit tests for shared CLIP dual-encoder helpers.

Covers the pure head-mixing dispatch ``apply_clifford_head`` used by both
CliffordCLIP towers. The tensor work is stubbed with trivial callables so the
tests pin the *dispatch* semantics exactly against the literal expressions the
helper replaces.
"""

import numpy as np
from numpy.testing import assert_allclose

from dl_techniques.utils.clip_utils import apply_clifford_head


# Trivial deterministic dummies standing in for the real layers.
def _geo_layer(z_det, z_ctx):
    """Stand-in for SparseRollingGeometricProduct: elementwise product."""
    return z_det * z_ctx


def _scale_layer(geo):
    """Stand-in for the LayerScale/LearnableMultiplier residual gate."""
    return 0.5 * geo


class TestApplyCliffordHead:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.anchor = rng.standard_normal((4, 8)).astype("float32")
        self.z_det = rng.standard_normal((4, 8)).astype("float32")
        self.z_ctx = rng.standard_normal((4, 8)).astype("float32")

    def _call(self, head_kind):
        return apply_clifford_head(
            head_kind,
            self.anchor,
            self.z_det,
            self.z_ctx,
            _geo_layer,
            _scale_layer,
        )

    def test_plain_returns_anchor_identity(self):
        out = self._call("plain")
        # plain must return the anchor UNCHANGED (same object semantics / exact values).
        assert_allclose(out, self.anchor, rtol=0, atol=0)

    def test_mean_max_returns_geo(self):
        out = self._call("mean_max")
        assert_allclose(out, _geo_layer(self.z_det, self.z_ctx), rtol=0, atol=0)

    def test_learned_query_returns_geo(self):
        out = self._call("learned_query")
        assert_allclose(out, _geo_layer(self.z_det, self.z_ctx), rtol=0, atol=0)

    def test_learned_query_residual_returns_anchor_plus_scaled_geo(self):
        out = self._call("learned_query_residual")
        expected = self.anchor + _scale_layer(_geo_layer(self.z_det, self.z_ctx))
        assert_allclose(out, expected, rtol=0, atol=0)

    def test_geo_and_scale_not_called_for_plain(self):
        """plain must short-circuit before touching either layer."""
        calls = {"geo": 0, "scale": 0}

        def geo(a, b):
            calls["geo"] += 1
            return a * b

        def scale(g):
            calls["scale"] += 1
            return g

        apply_clifford_head(
            "plain", self.anchor, self.z_det, self.z_ctx, geo, scale
        )
        assert calls == {"geo": 0, "scale": 0}

    def test_scale_not_called_for_non_residual(self):
        """Only learned_query_residual invokes scale_layer."""
        calls = {"scale": 0}

        def scale(g):
            calls["scale"] += 1
            return g

        apply_clifford_head(
            "mean_max", self.anchor, self.z_det, self.z_ctx, _geo_layer, scale
        )
        apply_clifford_head(
            "learned_query", self.anchor, self.z_det, self.z_ctx, _geo_layer, scale
        )
        assert calls["scale"] == 0
