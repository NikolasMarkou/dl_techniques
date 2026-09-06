"""
Test suite for the optional TRAINABLE Gabor warm-start stem of the bias-free ConvUNext
denoiser (models/vision/bias_free_denoisers/bfconvunext.py,
`# DECISION standalone-2026-09-06-gabor-warm-start/D-001`).

The stem is the construction of Ozbulak & Ekenel (SIU 2018): a bias-free CROSS-CHANNEL
`keras.layers.Conv2D` whose kernel is initialized from a Gabor bank and then TRAINED.
It replaced a frozen `DepthwiseConv2D` bank, so two things changed at once and both are
pinned here:

  * the layer class and its trainability (Conv2D / trainable, not DepthwiseConv2D /
    frozen), and
  * `gabor_filters`, which is now the stem's OUTPUT CHANNEL COUNT (a Conv2D `filters`)
    rather than a depthwise `depth_multiplier` emitting `in_ch * gabor_filters`.

Covers: build with use_gabor_stem=True (full-resolution output preserved), the stem's
class / trainability / exact Gabor-initialized kernel values, the mandatory bias-free
1x1 projection, the no-projection path's `gabor_filters == initial_filters` contract,
end-to-end positive homogeneity of the bias-free arm, default use_gabor_stem=False
producing the unchanged architecture, and a full .keras round-trip.
"""

import os
import keras
import numpy as np
from typing import Tuple

from dl_techniques.initializers import GaborFiltersInitializer
from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    create_convunext_variant,
)

# Positive-homogeneity tolerance, DERIVED rather than pasted.
#
# float32 eps is 2**-23 = 1.1920929e-07. A denoiser forward is a long chain of
# float32 reductions, and rescaling the input reassociates every one of them, so the
# relative error of D(a*x) vs a*D(x) grows with the number of sequential roundings.
# The bound below is 64 * eps = 7.6294e-06: the smallest power-of-two multiple of eps
# that clears the MEASURED worst case on the configurations exercised here by more
# than 4x.
#
# MEASURED over 6 random model initializations, block_normalization='batchnorm',
# a in HOMOGENEITY_ALPHAS, 32x32x3 input (min .. max relative error):
#   convunext, bias-free stem   -> 2.54e-07 .. 5.08e-07
#   bfunet,    bias-free stem   -> 8.12e-07 .. 1.23e-06   (6.2x under the bound)
#   convunext, 0.3 bias on stem -> 1.75e-01 .. 1.02e+00   (>= 22900x the bound)
#   bfunet,    0.3 bias on stem -> 7.52e-01 .. 1.29e+00
# The gap between the two regimes is five orders of magnitude, so the bound's exact
# value is not load-bearing -- but it is still not a free parameter:
# `test_the_homogeneity_guard_can_fail` re-runs the same probe against a deliberately
# bias-carrying stem and asserts it exceeds the bound.
FLOAT32_EPS = float(np.finfo(np.float32).eps)
HOMOGENEITY_RTOL = 64.0 * FLOAT32_EPS
HOMOGENEITY_ALPHAS = (0.25, 0.5, 2.0, 7.0)


def max_homogeneity_relative_error(model, input_shape, alphas=HOMOGENEITY_ALPHAS,
                                   seed: int = 0) -> float:
    """Worst relative violation of D(a*x) == a*D(x) over `alphas`.

    Interface contract: pure measurement, no assertions, no I/O. `model` must be a
    single-output denoiser accepting `(batch, *input_shape)` float32. Returns a
    non-negative float; a positively homogeneous model returns ~float32 rounding noise.
    Shared by the convunext and bfunet homogeneity guards AND by their RED proofs, so
    the guard and its falsification use one instrument.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(2, *input_shape)).astype("float32")
    base = np.asarray(model(x, training=False))
    worst = 0.0
    for a in alphas:
        got = np.asarray(model(np.float32(a) * x, training=False))
        want = np.float32(a) * base
        scale = max(float(np.max(np.abs(want))), 1e-30)
        worst = max(worst, float(np.max(np.abs(got - want))) / scale)
    return worst


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------

INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)


def _build_gabor(input_shape=INPUT_SHAPE, **overrides):
    cfg = dict(
        input_shape=input_shape,
        depth=3,
        initial_filters=16,
        blocks_per_level=1,
        convnext_version='v1',
        use_gabor_stem=True,
        gabor_filters=8,
        gabor_kernel_size=7,
    )
    cfg.update(overrides)
    return create_convunext_denoiser(**cfg)


# ---------------------------------------------------------------------
# Gabor stem
# ---------------------------------------------------------------------

class TestGaborStem:
    """Trainable cross-channel Gabor warm-start stem on create_convunext_denoiser."""

    def test_build_and_full_res_output(self) -> None:
        model = _build_gabor()
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        y = model(x)
        # Denoiser output must preserve full spatial resolution and channel count.
        assert tuple(y.shape) == (2, *INPUT_SHAPE)

    def test_gabor_layer_is_a_trainable_cross_channel_conv2d(self) -> None:
        """The stem is the paper's TRAINABLE Conv2D warm start, not a frozen bank.

        Both halves matter and neither implies the other: a `Conv2D` could still be
        shipped frozen, and a `DepthwiseConv2D` could be shipped trainable. The
        `type(...) is` check is deliberate -- `DepthwiseConv2D` is NOT a subclass of
        `Conv2D` in Keras 3.8, but `isinstance` would also admit `SeparableConv2D`,
        which is a different construction again.
        """
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        assert type(gabor) is keras.layers.Conv2D, (
            f"stem must be a plain Conv2D (the Ozbulak & Ekenel warm start), got "
            f"{type(gabor).__name__}"
        )
        assert not isinstance(gabor, keras.layers.DepthwiseConv2D)
        assert gabor.trainable is True
        # Exactly one trainable weight: the kernel. use_bias is hardcoded False.
        assert len(gabor.trainable_weights) == 1
        assert gabor.use_bias is False
        assert gabor.bias is None
        n_trainable = int(sum(np.prod(w.shape) for w in gabor.trainable_weights))
        assert n_trainable == 7 * 7 * 3 * 8

    def test_gabor_kernel_holds_the_gabor_bank_values(self) -> None:
        """The stem's kernel IS the Gabor bank, not merely a Conv2D of the right shape.

        Follows the precedent in
        `tests/test_initializers/test_gabor_filters_initializer.py::test_kernel_matches_initializer`.
        `create_gabor_conv2d` passes only its Gabor-shaping defaults through, and the
        model builder threads none of them, so the expected bank is
        `GaborFiltersInitializer()` at the stem's own kernel shape.
        """
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        kernel = np.asarray(gabor.kernel)
        assert kernel.shape == (7, 7, 3, 8), (
            "cross-channel Conv2D kernel is (kh, kw, in_ch, filters)"
        )
        expected = np.asarray(GaborFiltersInitializer()((7, 7, 3, 8)))
        np.testing.assert_allclose(kernel, expected, atol=1e-6, rtol=0)
        # The same 2D bank is replicated across input channels, which is what makes
        # the stem colour-blind at initialization (D-001's accepted cost).
        for c in range(1, 3):
            np.testing.assert_allclose(
                kernel[:, :, 0, :], kernel[:, :, c, :], atol=1e-6, rtol=0
            )

    def test_gabor_stem_kernel_receives_gradient(self) -> None:
        """`trainable=True` is not decoration: the kernel is in the model's trainables."""
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        names = {w.path for w in model.trainable_weights}
        assert gabor.kernel.path in names, (
            "the Gabor kernel must reach model.trainable_weights, or the 'warm start' "
            "is just a frozen bank with a misleading flag"
        )

    def test_projection_is_bias_free(self) -> None:
        model = _build_gabor()
        proj = model.get_layer('gabor_stem_projection')
        # Mandatory 1x1 projection (gabor_filters -> initial_filters), bias-free.
        assert proj.use_bias is False
        assert proj.filters == 16

    def test_gabor_output_channels(self) -> None:
        """`gabor_filters` IS the output channel count -- NOT in_ch * gabor_filters.

        This is the semantic change to the public knob. With 3-channel input and
        gabor_filters=8 the old depthwise stem emitted 24 channels; the Conv2D stem
        emits 8.
        """
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        out_shape = gabor.compute_output_shape((None, 64, 64, 3))
        assert out_shape[-1] == 8
        assert out_shape[-1] != 3 * 8, "the old per-channel-multiplier semantics are gone"
        assert gabor.output.shape[-1] == 8

    def test_default_has_no_gabor(self) -> None:
        model = create_convunext_denoiser(
            input_shape=INPUT_SHAPE,
            depth=3,
            initial_filters=16,
            blocks_per_level=1,
            convnext_version='v1',
        )
        names = [l.name for l in model.layers]
        assert 'gabor_stem' not in names
        assert 'gabor_stem_projection' not in names

    def test_variant_forwards_gabor_kwargs(self) -> None:
        model = create_convunext_variant(
            'tiny', INPUT_SHAPE, enable_deep_supervision=False,
            convnext_version='v1', use_gabor_stem=True, gabor_filters=8,
        )
        assert 'gabor_stem' in [l.name for l in model.layers]

    def test_keras_round_trip(self, tmp_path) -> None:
        model = _build_gabor()
        x = np.random.rand(2, *INPUT_SHAPE).astype(np.float32)
        # training=False: StochasticDepth (drop-path) is identity at inference, so
        # the forward is deterministic and round-trip equality is meaningful.
        y_before = model(x, training=False)

        save_path = os.path.join(str(tmp_path), 'bfconvunext_gabor.keras')
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x, training=False)

        # Gabor stem must survive serialization as a TRAINABLE Conv2D with its
        # learned/initialized kernel intact.
        gabor = loaded.get_layer('gabor_stem')
        assert type(gabor) is keras.layers.Conv2D
        assert gabor.trainable is True
        assert len(gabor.trainable_weights) == 1
        np.testing.assert_allclose(
            np.asarray(gabor.kernel),
            np.asarray(model.get_layer('gabor_stem').kernel),
            atol=1e-6, rtol=0,
        )

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant).
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            atol=1e-4,
            err_msg="Outputs differ after .keras round-trip with Gabor stem",
        )


class TestGaborStemNoProjection:
    """No-projection Gabor stem (gabor_stem_projection=False): the Conv2D stem feeds
    the encoder directly, valid only when `gabor_filters == initial_filters`.

    The old contract was `input_channels * gabor_filters == initial_filters`. It is
    gone: the stem no longer multiplies by the input channel count.
    """

    def test_no_projection_layer_and_full_res_output(self) -> None:
        # gabor_filters(24) == initial_filters(24); input channels are irrelevant now.
        model = _build_gabor(gabor_stem_projection=False,
                             gabor_filters=24, initial_filters=24)
        names = [l.name for l in model.layers]
        assert 'gabor_stem' in names, "Gabor stem layer should still be present"
        assert 'gabor_stem_projection' not in names, "projection must be dropped"
        x = np.random.uniform(-0.5, 0.5, size=(2, *INPUT_SHAPE)).astype("float32")
        y = model(x, training=False)
        assert tuple(y.shape) == (2, *INPUT_SHAPE)

    def test_gabor_feeds_initial_filters_channels(self) -> None:
        model = _build_gabor(gabor_stem_projection=False,
                             gabor_filters=24, initial_filters=24)
        gabor = model.get_layer('gabor_stem')
        # Conv2D stem output == gabor_filters == initial_filters, INDEPENDENT of in_ch.
        assert gabor.output.shape[-1] == 24
        assert tuple(gabor.kernel.shape) == (7, 7, 3, 24)

    def test_no_projection_is_bias_free(self) -> None:
        # Removing the projection must not introduce any bias / centering anywhere.
        model = _build_gabor(gabor_stem_projection=False,
                             gabor_filters=24, initial_filters=24)
        offenders = [
            l.name for l in model._flatten_layers()
            if getattr(l, "use_bias", False)
            or (isinstance(l, keras.layers.LayerNormalization) and getattr(l, "center", False))
        ]
        assert offenders == [], f"bias/centering survived: {offenders}"

    def test_channel_mismatch_raises(self) -> None:
        # gabor_filters(8) != initial_filters(16) -> fail loudly, never pad/slice.
        import pytest
        with pytest.raises(ValueError, match="initial_filters"):
            _build_gabor(gabor_stem_projection=False, initial_filters=16)

    def test_the_old_per_channel_match_no_longer_satisfies_the_contract(self) -> None:
        """A config legal under the OLD rule must now RAISE, not silently build.

        `in_ch(3) * gabor_filters(8) == 24 == initial_filters` satisfied the depthwise
        contract exactly. Under the Conv2D stem it is a mismatch (8 != 24), and the
        builder must say so rather than quietly emitting an 8-channel stem into a
        24-channel encoder level.
        """
        import pytest
        with pytest.raises(ValueError, match="gabor_filters"):
            _build_gabor(gabor_stem_projection=False, initial_filters=24)

    def test_default_keeps_projection(self) -> None:
        # gabor_stem_projection defaults True -> projection present, 8 -> 16.
        model = _build_gabor()  # initial_filters=16, projection maps 8 -> 16
        names = [l.name for l in model.layers]
        assert 'gabor_stem_projection' in names


class TestGaborStemHomogeneity:
    """The bias-free arm stays positively homogeneous with the Conv2D Gabor stem.

    This is the guarantee the depthwise->cross-channel swap was accused of breaking.
    It does not: `D(a*x) == a*D(x)` comes from the ABSENCE OF BIAS, and channel
    summing was never the mechanism. `block_normalization='batchnorm'` is required --
    the builder's historical `'layernorm'` default is scale-INVARIANT (degree 0) and
    breaks homogeneity on its own, which the builder itself warns about.
    """

    SHAPE: Tuple[int, int, int] = (32, 32, 3)

    def _build(self, **overrides):
        cfg = dict(
            input_shape=self.SHAPE, depth=2, initial_filters=16, blocks_per_level=1,
            convnext_version='v1', block_normalization='batchnorm',
            use_gabor_stem=True, gabor_filters=8, gabor_kernel_size=5,
        )
        cfg.update(overrides)
        return create_convunext_denoiser(**cfg)

    def test_bias_free_arm_is_positively_homogeneous(self) -> None:
        model = self._build()
        err = max_homogeneity_relative_error(model, self.SHAPE)
        assert err <= HOMOGENEITY_RTOL, (
            f"D(a*x) != a*D(x): max relative error {err:.3e} exceeds the derived "
            f"float32 bound {HOMOGENEITY_RTOL:.3e} (= 64 * eps)"
        )

    def test_the_homogeneity_guard_can_fail(self) -> None:
        """RED proof: put a bias on the Gabor stem and the SAME probe must exceed the bound.

        A homogeneity assertion that cannot fail is worthless, and the obvious way for
        this one to be vacuous is a bound so loose that a genuinely bias-carrying stack
        still slips under it. The injection is exactly the thing the stem's hardcoded
        `use_bias=False` prevents, applied through the builder's own module namespace so
        the whole graph is rebuilt around it -- not a post-hoc attribute poke that the
        already-traced functional graph would ignore.
        """
        import dl_techniques.models.vision.convunext.model as cvx_model

        real = cvx_model.create_gabor_conv2d

        def biased(**kwargs):
            kwargs['use_bias'] = True
            return real(**kwargs)

        cvx_model.create_gabor_conv2d = biased
        try:
            broken = self._build()
            stem = broken.get_layer('gabor_stem')
            assert stem.use_bias is True and stem.bias is not None, (
                "injection did not take -- the RED proof would be vacuous"
            )
            # Keras zero-initializes a Conv2D bias, and a ZERO bias is still
            # homogeneous, so the injection must assign a non-zero value or it proves
            # nothing (this repo's recorded vacuous-negative-control defect).
            stem.bias.assign(np.full(stem.bias.shape, 0.3, dtype="float32"))
            err = max_homogeneity_relative_error(broken, self.SHAPE)
        finally:
            cvx_model.create_gabor_conv2d = real

        assert err > HOMOGENEITY_RTOL, (
            f"the homogeneity guard CANNOT FAIL: a stem carrying a 0.3 bias still "
            f"measured {err:.3e} <= {HOMOGENEITY_RTOL:.3e}"
        )

    def test_the_homogeneity_injection_is_restored(self) -> None:
        """The RED proof must not leak its monkeypatch into the rest of the session."""
        import dl_techniques.models.vision.convunext.model as cvx_model
        from dl_techniques.initializers import create_gabor_conv2d as real
        assert cvx_model.create_gabor_conv2d is real
        assert self._build().get_layer('gabor_stem').use_bias is False


class TestFinalProjectionGroups:
    """Grouped final 1x1 output projection (final_projection_groups)."""

    def _build(self, **overrides):
        # initial_filters=18 is divisible by 3 (output channels) so groups=3 is valid.
        cfg = dict(
            input_shape=(32, 32, 3), depth=3, initial_filters=18,
            blocks_per_level=1, convnext_version='v1',
        )
        cfg.update(overrides)
        return create_convunext_denoiser(**cfg)

    def test_default_groups_one(self) -> None:
        # Default == standard dense 1x1: groups=1, full (1,1,18,3) kernel.
        model = self._build()
        proj = model.get_layer('final_output')
        assert proj.groups == 1
        assert tuple(proj.kernel.shape) == (1, 1, 18, 3)

    def test_grouped_projection_kernel_and_output(self) -> None:
        # groups == output_channels: each output channel reads a disjoint 18/3=6 group.
        model = self._build(final_projection_groups=3)
        proj = model.get_layer('final_output')
        assert proj.groups == 3
        assert tuple(proj.kernel.shape) == (1, 1, 6, 3)  # (kh,kw,Cin/groups,Cout)
        x = np.random.uniform(-0.5, 0.5, size=(2, 32, 32, 3)).astype("float32")
        y = model(x, training=False)
        assert tuple(y.shape) == (2, 32, 32, 3)

    def test_grouped_projection_is_bias_free(self) -> None:
        model = self._build(final_projection_groups=3)
        proj = model.get_layer('final_output')
        assert proj.use_bias is False
        offenders = [
            l.name for l in model._flatten_layers()
            if getattr(l, "use_bias", False)
            or (isinstance(l, keras.layers.LayerNormalization) and getattr(l, "center", False))
        ]
        assert offenders == [], f"bias/centering survived: {offenders}"

    def test_indivisible_groups_raise(self) -> None:
        import pytest
        # 4 divides neither 18 nor 3.
        with pytest.raises(ValueError, match="final_projection_groups"):
            self._build(final_projection_groups=4)

    def test_groups_with_extra_zero_raise(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="extra_zero_output_channels"):
            self._build(final_projection_groups=3, extra_zero_output_channels=True)

    def test_grouped_round_trip(self, tmp_path) -> None:
        model = self._build(final_projection_groups=3)
        x = np.random.uniform(-0.5, 0.5, size=(2, 32, 32, 3)).astype("float32")
        y0 = keras.ops.convert_to_numpy(model(x, training=False))
        path = os.path.join(tmp_path, "grouped.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        assert reloaded.get_layer('final_output').groups == 3
        y1 = keras.ops.convert_to_numpy(reloaded(x, training=False))
        np.testing.assert_allclose(y0, y1, atol=1e-5)


class TestResidualStochasticDepth:
    """Regression guard: ConvNeXt blocks MUST be wired as residual branches with
    stochastic depth (the factory bug where blocks were chained sequentially with
    no residual / no drop-path must not recur)."""

    def test_residual_and_stochastic_depth_present(self):
        from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
        model = create_convunext_denoiser(
            input_shape=(64, 64, 3), depth=3, initial_filters=16,
            blocks_per_level=2, convnext_version="v1", drop_path_rate=0.2,
        )
        n_add = sum(1 for l in model.layers if isinstance(l, keras.layers.Add))
        n_sd = sum(1 for l in model.layers if isinstance(l, StochasticDepth))
        # depth=3 -> 3 enc + 1 bottleneck + 3 dec block-groups, 2 blocks each = 14 blocks.
        assert n_add == 14, f"expected 14 residual adds, got {n_add}"
        assert n_sd >= 1, "no StochasticDepth (drop-path) layers found"

    def test_no_residual_when_blocks_chained_would_change_shape_is_safe(self):
        # Residual add requires matching channels; the factory channel-adjusts before
        # blocks, so output must still equal the full-res input.
        model = create_convunext_denoiser(
            input_shape=(64, 64, 3), depth=3, initial_filters=16,
            blocks_per_level=1, convnext_version="v2", drop_path_rate=0.1,
        )
        y = model(np.zeros((2, 64, 64, 3), "float32"), training=False)
        assert tuple(y.shape) == (2, 64, 64, 3)

    def test_drop_path_zero_has_no_stochastic_depth(self):
        from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
        model = create_convunext_denoiser(
            input_shape=(64, 64, 3), depth=3, initial_filters=16,
            blocks_per_level=1, convnext_version="v1", drop_path_rate=0.0,
        )
        n_sd = sum(1 for l in model.layers if isinstance(l, StochasticDepth))
        n_add = sum(1 for l in model.layers if isinstance(l, keras.layers.Add))
        assert n_sd == 0, "drop_path_rate=0 should add no StochasticDepth layers"
        assert n_add >= 1, "residual connections must exist even at drop_path_rate=0"
