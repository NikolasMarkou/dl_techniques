"""Tests for KANInitializer (Rigas et al. variance-controlled KAN init).

Covers all 9 Success Criteria of plan_2026-06-12_6cc7c378:
    1. Construction / validation
    2. Shape dispatch (2D residual / 3D spline) + finite
    3. N-consistency with KANLinear (N = grid_size + spline_order)
    4. Seed reproducibility
    5. Three schemes + power_law magnitude ordering (residual std > spline std)
    6. get_config / from_config round-trip
    7. KANLinear integration (forward + gradient flow)
    8. .keras model save/load round-trip
    9. Backward-compat (default base_scaler all-ones; factory registers param)

Plus single-claim guards for defects found by review and verified by
measurement against the previous implementation:

* ``mu_R_0`` / ``mu_R_1`` were Monte-Carlo estimates over 10,000 draws seeded by
  the initializer's own ``seed``, so the variance rule moved with the RNG:
  7.90% spread in ``mu_R_0`` and 4.44% in ``mu_R_1`` over 200 seeds. Exact
  values are 0.094493 and 0.319078 on ``(-1, 1)``.
* ``mu_B_0`` and ``mu_B_1`` were the proxies ``1/(G+1)`` and ``1.0``. Measured
  against the real Cox-de Boor basis: ``mu_B_0`` was 2.3x-2.8x high, and the
  true ``mu_B_1`` is 0.521 / 1.282 / 2.899 for ``G`` = 5 / 10 / 20, so a
  hard-coded 1.0 was blind to the single most important KAN hyperparameter.
* ``np.random.default_rng(None)`` bypassed ``keras.utils.set_random_seed``, and
  a ``create_kan_initializers`` pair shared one stream -- the residual matrix
  and the leading block of the spline tensor correlated at exactly 1.0000.
* ``alpha``, ``beta`` and ``baseline_noise`` were unvalidated, the spline branch
  never checked its last dim against ``grid_size + spline_order`` (the D-001
  invariant it exists to protect), and ``.astype("float32")`` before the cast
  made ``dtype='float64'`` return float32-quantized values.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from unittest import mock
from keras import ops

from dl_techniques.initializers import KANInitializer, create_kan_initializers
from dl_techniques.initializers.kan_initializer import (
    _basis_moments,
    _silu_moments,
)
from dl_techniques.layers.ffn.kan_linear import KANLinear
from dl_techniques.layers.ffn.factory import FFN_REGISTRY


class TestKANInitializerConstruction:
    """SC1: construction defaults / custom params / validation."""

    def test_initialization_defaults(self):
        init = KANInitializer()
        assert init.scheme == "power_law"
        assert init.target == "residual"
        assert init.grid_size == 5
        assert init.spline_order == 3
        assert init.alpha == 0.25
        assert init.beta == 1.75
        assert init.baseline_noise == 0.1
        assert init.grid_range == (-1.0, 1.0)

    def test_initialization_custom(self):
        init = KANInitializer(
            scheme="glorot_inspired",
            target="spline",
            grid_size=8,
            spline_order=4,
            alpha=0.3,
            beta=2.0,
            baseline_noise=0.05,
            seed=42,
        )
        assert init.scheme == "glorot_inspired"
        assert init.target == "spline"
        assert init.grid_size == 8
        assert init.spline_order == 4
        assert init.alpha == 0.3
        assert init.beta == 2.0
        assert init.baseline_noise == 0.05
        assert init.seed == 42

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            KANInitializer(scheme="not_a_scheme")

    def test_invalid_target(self):
        with pytest.raises(ValueError):
            KANInitializer(target="not_a_target")

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError):
            KANInitializer(grid_size=0)

    def test_invalid_spline_order(self):
        with pytest.raises(ValueError):
            KANInitializer(spline_order=-1)


    @pytest.mark.parametrize("kwargs,match", [
        ({"alpha": -0.5}, "alpha must be >= 0"),
        ({"beta": -1.0}, "beta must be >= 0"),
        ({"alpha": float("nan")}, "alpha must be finite"),
        ({"baseline_noise": 0.0}, "baseline_noise must be positive"),
        ({"baseline_noise": -0.5}, "baseline_noise must be positive"),
        ({"grid_range": (1.0, -1.0)}, "lo < hi"),
        ({"grid_range": (0.0, float("inf"))}, "grid_range must be finite"),
        ({"grid_range": (0.0,)}, r"\(lo, hi\) pair"),
    ])
    def test_invalid_scheme_parameters(self, kwargs, match):
        """Unvalidated knobs each had a silent failure mode.

        A negative exponent inverts the power law into GROWTH with width;
        baseline_noise=0 gives a dead spline path; a negative baseline_noise
        silently acts as its absolute value (a Gaussian is symmetric).
        """
        with pytest.raises(ValueError, match=match):
            KANInitializer(**kwargs)

    def test_beta_not_above_alpha_warns(self, caplog):
        """beta > alpha is the whole point of the power-law bias; say so loudly."""
        with mock.patch(
            "dl_techniques.initializers.kan_initializer.logger"
        ) as mock_logger:
            KANInitializer(scheme="power_law", alpha=1.0, beta=0.5)
        mock_logger.warning.assert_called()
        assert "beta > alpha" in mock_logger.warning.call_args[0][0]

        with mock.patch(
            "dl_techniques.initializers.kan_initializer.logger"
        ) as mock_logger:
            KANInitializer(scheme="power_law", alpha=0.25, beta=1.75)
        mock_logger.warning.assert_not_called()


class TestKANInitializerConstants:
    """The expectation constants: exact, deterministic, grid-aware."""

    def test_the_silu_moments_are_exact(self):
        """mu_R_0 / mu_R_1 are quadrature values, not Monte-Carlo estimates.

        The exact integrals over U(-1, 1) are 0.094493 and 0.319078. The
        previous 10,000-sample estimate ranged over 0.090412-0.097873 and
        0.311370-0.325542 across 200 seeds.
        """
        init = KANInitializer()
        assert init.mu_R_0 == pytest.approx(0.094493, abs=1e-6)
        assert init.mu_R_1 == pytest.approx(0.319078, abs=1e-6)

    def test_the_constants_do_not_depend_on_the_seed(self):
        """The variance rule must not move with the RNG draw."""
        constants = [
            (k.mu_R_0, k.mu_R_1, k.mu_B_0, k.mu_B_1)
            for k in (
                KANInitializer(seed=None),
                KANInitializer(seed=0),
                KANInitializer(seed=12345),
            )
        ]
        assert constants[0] == constants[1] == constants[2]

    @pytest.mark.parametrize("grid_size,mu_b0,mu_b1", [
        (5, 0.059921, 0.520833),
        (10, 0.036874, 1.282051),
        (20, 0.020842, 2.898551),
    ])
    def test_the_basis_moments_are_the_real_basis(self, grid_size, mu_b0, mu_b1):
        """mu_B_* are the true Cox-de Boor expectations, not 1/(G+1) and 1.0.

        Values are pinned as literals rather than recomputed from the module's
        own helpers: an oracle that imports the thing it checks cannot fail.
        """
        init = KANInitializer(grid_size=grid_size, spline_order=3)

        assert init.mu_B_0 == pytest.approx(mu_b0, abs=1e-5)
        assert init.mu_B_1 == pytest.approx(mu_b1, abs=1e-5)

        # The retired proxies, for contrast.
        assert init.mu_B_0 != pytest.approx(1.0 / (grid_size + 1), abs=1e-3)
        assert init.mu_B_1 != pytest.approx(1.0, abs=1e-3)

    def test_the_basis_derivative_moment_tracks_grid_resolution(self):
        """A B-spline derivative scales as 1/h, so mu_B_1 must grow with G.

        The hard-coded 1.0 was constant in G, making the backward-path balance
        blind to the single most important KAN hyperparameter.
        """
        values = [KANInitializer(grid_size=g).mu_B_1 for g in (5, 10, 20)]
        assert values[0] < values[1] < values[2]
        assert values[2] / values[0] > 4.0

    def test_the_constants_follow_grid_range(self):
        """mu_B_0 is scale invariant; mu_B_1 scales as 1/width^2; SiLU moves."""
        narrow = KANInitializer(grid_range=(-1.0, 1.0))
        wide = KANInitializer(grid_range=(-2.0, 2.0))

        assert wide.mu_B_0 == pytest.approx(narrow.mu_B_0, rel=1e-6)
        assert wide.mu_B_1 == pytest.approx(narrow.mu_B_1 / 4.0, rel=1e-6)
        assert wide.mu_R_0 == pytest.approx(0.467474, abs=1e-6)
        assert wide.mu_R_1 == pytest.approx(0.426391, abs=1e-6)

    def test_the_quadrature_helpers_agree_with_monte_carlo(self):
        """Independent check that the exact rules integrate what they claim to."""
        rng = np.random.default_rng(0)
        x = rng.uniform(-1.0, 1.0, 400_000)
        sigmoid = 1.0 / (1.0 + np.exp(-x))

        mu_r0, mu_r1 = _silu_moments((-1.0, 1.0))
        assert mu_r0 == pytest.approx(np.mean((x * sigmoid) ** 2), rel=0.02)
        assert mu_r1 == pytest.approx(
            np.mean((sigmoid + x * sigmoid * (1 - sigmoid)) ** 2), rel=0.02
        )

        mu_b0, _ = _basis_moments(5, 3, (-1.0, 1.0))
        assert mu_b0 == pytest.approx(0.059921, abs=1e-5)


class TestKANInitializerVarianceRules:
    """The per-scheme std formulas and the forward gain they imply."""

    def test_the_glorot_rule_matches_the_paper_formula(self):
        """sigma^2 = (1/N) * 2/(n_in*mu_0 + n_out*mu_1) for BOTH roles.

        The 1/N factor is on the residual too, as printed in the paper: it
        apportions the edge's variance budget across the edge's additive terms
        (G+k basis terms plus the residual), not across N copies of a residual
        weight.
        """
        init = KANInitializer(scheme="glorot_inspired", grid_size=5, spline_order=3)
        n_in, n_out, N = 64, 32, 8

        sigma_r, sigma_b = init._compute_std_glorot(n_in, n_out, N)

        expected_r = np.sqrt(
            (1.0 / N) * 2.0 / (n_in * init.mu_R_0 + n_out * init.mu_R_1)
        )
        expected_b = np.sqrt(
            (1.0 / N) * 2.0 / (n_in * init.mu_B_0 + n_out * init.mu_B_1)
        )
        assert sigma_r == pytest.approx(expected_r, rel=1e-9)
        assert sigma_b == pytest.approx(expected_b, rel=1e-9)

    def test_the_power_law_has_no_backward_term(self):
        """n_out is accepted for dispatch uniformity and deliberately unused."""
        init = KANInitializer(scheme="power_law")
        assert init._compute_std_power_law(64, 8, 8) == init._compute_std_power_law(
            64, 4096, 8
        )

    @pytest.mark.parametrize("scheme,width_dependent", [
        ("power_law", True),
        ("glorot_inspired", False),
        ("baseline", True),
    ])
    def test_which_schemes_are_width_independent(self, scheme, width_dependent):
        """Only glorot_inspired holds its gain across width; the module says so."""
        gains = [
            sum(KANInitializer(scheme=scheme).expected_forward_gain(n, n))
            for n in (16, 256, 4096)
        ]
        spread = max(gains) / min(gains)
        assert (spread > 2.0) is width_dependent, gains

    @pytest.mark.parametrize("scheme,width,gain", [
        ("power_law", 16, 0.1336),
        ("power_law", 256, 0.5345),
        ("power_law", 4096, 2.138),
        ("glorot_inspired", 16, 0.2635),
        ("glorot_inspired", 4096, 0.2635),
        ("baseline", 16, 0.1712),
        ("baseline", 4096, 19.73),
    ])
    def test_the_documented_forward_gains(self, scheme, width, gain):
        """Pin the measured table in the module docstring.

        None of the three schemes is unit gain; the point of this test is that
        the numbers the docstring quotes stay true, not that they are 1.0.
        """
        total = sum(KANInitializer(scheme=scheme).expected_forward_gain(width, width))
        assert total == pytest.approx(gain, rel=2e-3)

    def test_the_analytic_gain_predicts_the_actual_draw(self):
        """expected_forward_gain is not a second, unrelated formula.

        Check it against the residual path realized by the sampled weights:
        gain = n_in * Var(r) * E[SiLU(x)^2].
        """
        init = KANInitializer(scheme="glorot_inspired", target="residual", seed=0)
        n_in, n_out = 128, 64

        weights = ops.convert_to_numpy(init((n_in, n_out)))
        measured = n_in * float(np.var(weights)) * init.mu_R_0
        predicted, _ = init.expected_forward_gain(n_in, n_out)

        assert measured == pytest.approx(predicted, rel=0.05)

    def test_the_power_law_spline_path_is_numerically_absent(self):
        """beta=1.75 is ten orders down, not 'somewhat smaller'.

        Documented as recoverable because dL/db_m = (dL/dy) * B_m(x) does not
        depend on b, so the coefficients still get full-strength gradients.
        """
        _, spline_gain = KANInitializer(scheme="power_law").expected_forward_gain(
            256, 256
        )
        assert spline_gain < 1e-8


class TestKANInitializerShapeDispatch:
    """SC2: shape dispatch + finite output + dimensionality guards."""

    def test_shape_residual_2d(self):
        init = KANInitializer(target="residual", seed=0)
        w = init((4, 8))
        assert tuple(w.shape) == (4, 8)
        w_np = ops.convert_to_numpy(w)
        assert not np.any(np.isnan(w_np))
        assert not np.any(np.isinf(w_np))
        assert not ops.any(ops.isnan(w))
        assert not ops.any(ops.isinf(w))

    def test_shape_spline_3d(self):
        init = KANInitializer(target="spline", seed=0)
        w = init((4, 8, 8))
        assert tuple(w.shape) == (4, 8, 8)
        assert not ops.any(ops.isnan(w))
        assert not ops.any(ops.isinf(w))

    def test_residual_rejects_3d(self):
        init = KANInitializer(target="residual", seed=0)
        with pytest.raises(ValueError):
            init((4, 8, 8))

    def test_spline_rejects_2d(self):
        init = KANInitializer(target="spline", seed=0)
        with pytest.raises(ValueError):
            init((4, 8))


class TestKANInitializerNConsistency:
    """SC3: N == grid_size + spline_order, matched to KANLinear.spline_weight."""

    def test_n_matches_kan_linear(self):
        grid_size, spline_order = 5, 3
        expected_n = grid_size + spline_order  # == 8

        # A real KANLinear: its spline_weight last-dim must equal N.
        layer = KANLinear(
            features=8, grid_size=grid_size, spline_order=spline_order
        )
        layer.build((None, 4))
        spline_shape = tuple(layer.spline_weight.shape)
        assert spline_shape[-1] == expected_n
        assert spline_shape == (4, 8, expected_n)

        # The spline-target initializer, called on that exact shape, returns a
        # matching shape (N derived directly from shape[-1] == grid_size+order).
        _, spline_init = create_kan_initializers(
            grid_size=grid_size, spline_order=spline_order, seed=0
        )
        out = spline_init(spline_shape)
        assert tuple(out.shape) == spline_shape
        assert tuple(out.shape)[-1] == grid_size + spline_order


    def test_a_spline_shape_desync_is_rejected(self):
        """The check that makes the D-001 invariant enforceable.

        Previously a spline initializer configured with grid_size=10 happily
        filled an (n_in, n_out, 8) tensor: the spline branch read N from the
        shape while the residual branch reconstructed N = grid_size +
        spline_order = 13, so the two roles' variance scales desynced silently
        -- the exact failure D-001 exists to prevent.
        """
        init = KANInitializer(target="spline", grid_size=10, spline_order=3)
        with pytest.raises(ValueError, match="desync"):
            init((4, 4, 8))

        # The matching configuration still builds.
        assert tuple(init((4, 4, 13)).shape) == (4, 4, 13)

    def test_the_residual_branch_uses_g_plus_k(self):
        """The 2D branch reconstructs N as grid_size + spline_order, not G+k+1.

        Never asserted numerically before. Read it back through the power law,
        where sigma_r = (n_in * N) ** -alpha is a strictly decreasing function
        of N, so the two candidates are distinguishable.
        """
        init = KANInitializer(scheme="power_law", target="residual",
                              grid_size=5, spline_order=3, seed=0)
        n_in, n_out = 32, 16

        realized = float(np.std(ops.convert_to_numpy(init((n_in, n_out)))))
        assert realized == pytest.approx((n_in * 8) ** -0.25, rel=0.1)
        assert realized != pytest.approx((n_in * 9) ** -0.25, rel=1e-3)


class TestKANInitializerSeed:
    """SC4: same seed -> identical; different seed -> different (both targets)."""

    @pytest.mark.parametrize(
        "target,shape",
        [("residual", (32, 16)), ("spline", (32, 16, 8))],
    )
    def test_seed_reproducibility(self, target, shape):
        a = ops.convert_to_numpy(KANInitializer(target=target, seed=7)(shape))
        b = ops.convert_to_numpy(KANInitializer(target=target, seed=7)(shape))
        assert np.allclose(a, b, atol=1e-7)

    @pytest.mark.parametrize(
        "target,shape",
        [("residual", (32, 16)), ("spline", (32, 16, 8))],
    )
    def test_different_seeds_differ(self, target, shape):
        a = ops.convert_to_numpy(KANInitializer(target=target, seed=1)(shape))
        b = ops.convert_to_numpy(KANInitializer(target=target, seed=2)(shape))
        assert not np.allclose(a, b)


    def test_a_seedless_instance_honours_the_global_seed(self):
        """keras.utils.set_random_seed controls the draw.

        np.random.default_rng(None) drew from OS entropy and ignored it, so a
        KAN built from these initializers was irreproducible under a global
        seed.
        """
        keras.utils.set_random_seed(1234)
        a = ops.convert_to_numpy(KANInitializer(seed=None)((16, 8)))
        keras.utils.set_random_seed(1234)
        b = ops.convert_to_numpy(KANInitializer(seed=None)((16, 8)))
        np.testing.assert_array_equal(a, b)

        keras.utils.set_random_seed(4321)
        c = ops.convert_to_numpy(KANInitializer(seed=None)((16, 8)))
        assert not np.allclose(a, c)

    def test_a_matched_pair_does_not_share_a_random_stream(self):
        """One `seed` must give two INDEPENDENT draws, not two views of one.

        Measured on the previous implementation: the residual matrix was the
        leading block of the same standard-normal stream as the spline tensor,
        with a Pearson correlation of exactly 1.0000.
        """
        residual_init, spline_init = create_kan_initializers(
            grid_size=5, spline_order=3, seed=0
        )
        residual = ops.convert_to_numpy(residual_init((8, 16))).ravel()
        spline = ops.convert_to_numpy(spline_init((8, 16, 8))).ravel()

        correlation = abs(np.corrcoef(residual, spline[: residual.size])[0, 1])
        assert correlation < 0.3, f"correlated draws: r = {correlation:.4f}"


class TestKANInitializerDtype:
    """dtype handling -- entirely untested before."""

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_dtype_is_honored(self, dtype):
        w = KANInitializer(seed=0)((16, 8), dtype=dtype)
        assert keras.backend.standardize_dtype(w.dtype) == dtype

    def test_float64_is_not_a_float32_upcast(self):
        """.astype("float32") before the cast quantized a float64 request."""
        w = np.asarray(
            ops.convert_to_numpy(KANInitializer(seed=0)((64, 32), dtype="float64"))
        )
        assert w.dtype == np.float64
        assert not np.array_equal(w, w.astype(np.float32).astype(np.float64))

    def test_dtype_none_follows_floatx(self):
        original = keras.config.floatx()
        try:
            for floatx in ("float32", "float64"):
                keras.config.set_floatx(floatx)
                w = KANInitializer(seed=0)((16, 8), dtype=None)
                assert keras.backend.standardize_dtype(w.dtype) == floatx
        finally:
            keras.config.set_floatx(original)

    def test_call_accepts_extra_kwargs(self):
        """Both sibling initializers accept **kwargs; this one now does too."""
        w = KANInitializer(seed=0)((4, 4), None, partition_shape=None)
        assert tuple(w.shape) == (4, 4)


class TestKANInitializerSchemes:
    """SC5: all schemes finite; power_law residual std > spline std."""

    @pytest.mark.parametrize(
        "scheme", ["power_law", "glorot_inspired", "baseline"]
    )
    def test_all_schemes_finite(self, scheme):
        res_w = ops.convert_to_numpy(
            KANInitializer(scheme=scheme, target="residual", seed=0)((64, 64))
        )
        spl_w = ops.convert_to_numpy(
            KANInitializer(scheme=scheme, target="spline", seed=0)((64, 64, 8))
        )
        assert np.all(np.isfinite(res_w))
        assert np.all(np.isfinite(spl_w))

    def test_power_law_magnitude_ordering(self):
        # beta (1.75) > alpha (0.25) -> residual std > spline std.
        res_w = ops.convert_to_numpy(
            KANInitializer(scheme="power_law", target="residual", seed=0)(
                (64, 64)
            )
        )
        spl_w = ops.convert_to_numpy(
            KANInitializer(scheme="power_law", target="spline", seed=0)(
                (64, 64, 8)
            )
        )
        assert np.std(res_w) > np.std(spl_w)


class TestKANInitializerSerialization:
    """SC6: get_config returns all 8 keys; from_config reproduces output."""

    def test_serialization_roundtrip(self):
        init = KANInitializer(
            scheme="glorot_inspired",
            target="spline",
            grid_size=6,
            spline_order=2,
            alpha=0.3,
            beta=1.9,
            baseline_noise=0.2,
            seed=11,
        )
        cfg = init.get_config()
        expected_keys = {
            "scheme",
            "target",
            "grid_size",
            "spline_order",
            "alpha",
            "beta",
            "baseline_noise",
            "seed",
        }
        assert expected_keys.issubset(set(cfg.keys()))

        restored = KANInitializer.from_config(cfg)
        assert restored.scheme == init.scheme
        assert restored.target == init.target
        assert restored.grid_size == init.grid_size
        assert restored.spline_order == init.spline_order
        assert restored.alpha == init.alpha
        assert restored.beta == init.beta
        assert restored.baseline_noise == init.baseline_noise
        assert restored.seed == init.seed

        np.testing.assert_allclose(
            ops.convert_to_numpy(init((16, 8, 8))),
            ops.convert_to_numpy(restored((16, 8, 8))),
            atol=1e-7,
        )

    def test_keras_serialize_deserialize(self):
        init = KANInitializer(scheme="power_law", target="residual", seed=3)
        restored = keras.initializers.deserialize(
            keras.initializers.serialize(init)
        )
        assert isinstance(restored, KANInitializer)
        np.testing.assert_allclose(
            ops.convert_to_numpy(init((8, 8))),
            ops.convert_to_numpy(restored((8, 8))),
            atol=1e-7,
        )


    def test_grid_range_roundtrips(self):
        """The new field survives a config round trip and changes the constants."""
        init = KANInitializer(grid_range=(-2.0, 2.0), seed=1)
        restored = KANInitializer.from_config(init.get_config())

        assert tuple(restored.grid_range) == (-2.0, 2.0)
        assert restored.mu_R_0 == pytest.approx(init.mu_R_0)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(init((16, 8))),
            ops.convert_to_numpy(restored((16, 8))),
        )

    def test_a_legacy_config_without_grid_range_still_loads(self):
        """Configs written before grid_range existed must deserialize."""
        restored = KANInitializer.from_config({
            "scheme": "power_law", "target": "residual", "grid_size": 5,
            "spline_order": 3, "alpha": 0.25, "beta": 1.75,
            "baseline_noise": 0.1, "seed": 3,
        })
        assert restored.grid_range == (-1.0, 1.0)

    def test_the_config_keeps_the_seed_the_caller_passed(self):
        """A seedless initializer stays seedless across a round trip."""
        assert KANInitializer().get_config()["seed"] is None
        assert KANInitializer(seed=11).get_config()["seed"] == 11

    def test_the_config_omits_the_derived_constants(self):
        """mu_* are exact functions of the serialized fields, so they are not fields."""
        config = KANInitializer().get_config()
        for name in ("mu_R_0", "mu_R_1", "mu_B_0", "mu_B_1"):
            assert name not in config


class TestKANInitializerFactory:
    """create_kan_initializers must be able to configure every scheme it offers."""

    def test_the_factory_forwards_every_scheme_parameter(self):
        """It could select 'baseline' but not configure baseline_noise, nor grid_range."""
        residual_init, spline_init = create_kan_initializers(
            grid_size=7, spline_order=2, scheme="baseline",
            grid_range=(-2.0, 2.0), alpha=0.5, beta=1.5,
            baseline_noise=0.25, seed=4,
        )
        for init in (residual_init, spline_init):
            assert init.scheme == "baseline"
            assert init.grid_size == 7
            assert init.spline_order == 2
            assert tuple(init.grid_range) == (-2.0, 2.0)
            assert init.alpha == 0.5
            assert init.beta == 1.5
            assert init.baseline_noise == 0.25

        assert residual_init.target == "residual"
        assert spline_init.target == "spline"

        # baseline_noise actually reaches the spline draw.
        spline = ops.convert_to_numpy(spline_init((32, 32, 9)))
        assert float(np.std(spline)) == pytest.approx(0.25, rel=0.1)


class TestKANInitializerKANLinearIntegration:
    """SC7: KANLinear wired with both initializers; forward + gradients."""

    def test_kan_linear_integration_forward_grad(self):
        res_init, spline_init = create_kan_initializers(
            grid_size=5, spline_order=3, scheme="power_law", seed=0
        )
        layer = KANLinear(
            features=8,
            grid_size=5,
            spline_order=3,
            kernel_initializer=spline_init,
            base_scaler_initializer=res_init,
        )

        x = keras.Variable(keras.random.normal((2, 4)))
        with tf.GradientTape() as tape:
            out = layer(x)
            loss = ops.mean(ops.square(out))

        out_np = ops.convert_to_numpy(out)
        assert np.all(np.isfinite(out_np))

        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == 3
        assert all(g is not None for g in grads)
        for g in grads:
            g_np = ops.convert_to_numpy(g)
            assert not np.any(np.isnan(g_np)), "gradient contains NaN"
            assert np.any(g_np != 0.0), "gradient is all-zero"


class TestKANInitializerSaveLoad:
    """SC8: .keras model save/load round-trip + initializer identity."""

    def test_model_save_load_with_kan_initializer(self, tmp_path):
        res_init, spline_init = create_kan_initializers(
            grid_size=5, spline_order=3, scheme="power_law", seed=5
        )
        inputs = keras.Input(shape=(4,))
        outputs = KANLinear(
            features=8,
            grid_size=5,
            spline_order=3,
            kernel_initializer=spline_init,
            base_scaler_initializer=res_init,
        )(inputs)
        model = keras.Model(inputs, outputs)

        x = np.random.randn(3, 4).astype("float32")
        before = ops.convert_to_numpy(model(x))

        path = os.path.join(str(tmp_path), "kan_model.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = ops.convert_to_numpy(loaded(x))

        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

        loaded_layer = None
        for lyr in loaded.layers:
            if isinstance(lyr, KANLinear):
                loaded_layer = lyr
                break
        assert loaded_layer is not None

        assert isinstance(loaded_layer.base_scaler_initializer, KANInitializer)
        assert isinstance(loaded_layer.kernel_initializer, KANInitializer)
        assert loaded_layer.base_scaler_initializer.seed == 5
        assert loaded_layer.kernel_initializer.seed == 5
        assert loaded_layer.base_scaler_initializer.target == "residual"
        assert loaded_layer.kernel_initializer.target == "spline"


class TestKANInitializerBackwardCompat:
    """SC9: default KANLinear base_scaler all-ones; factory registers param."""

    def test_default_base_scaler_is_ones(self):
        layer = KANLinear(features=8)
        layer.build((None, 4))
        assert bool(ops.all(layer.base_scaler == 1.0))

    def test_factory_registers_base_scaler_initializer(self):
        assert "base_scaler_initializer" in FFN_REGISTRY["kan"]["optional_params"]


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
