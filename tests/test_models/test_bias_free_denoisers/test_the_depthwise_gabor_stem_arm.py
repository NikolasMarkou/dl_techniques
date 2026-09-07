"""Guard: `gabor_filters_per_channel` builds a REAL per-channel Gabor stem, opt-in.

`# DECISION standalone-2026-09-07-depthwise-gabor-stem/D-001` re-opened the depthwise
Gabor bank that `standalone-2026-09-06-gabor-warm-start/D-001` had given up, as a second
arm of `create_convunext`'s stem keyed on `gabor_filters_per_channel`. Re-adding a
construction that was deliberately deleted is exactly the change that rots into a flag
which parses and does nothing, so this module pins the mechanism rather than the flag:

  1. the DEFAULT arm is untouched -- `None` still builds the cross-channel `Conv2D`;
  2. the depthwise arm builds a `DepthwiseConv2D` whose kernel IS a Gabor bank;
  3. it is genuinely PER-CHANNEL -- output channel `(c, j)` depends on input channel `c`
     ALONE. This is the property the `Conv2D` stem gave up and the only reason the arm
     exists, so it is tested by ablation, not by reading the layer class;
  4. `gabor_filters` is NOT read on this arm (the two counts are alternatives, and the
     dangerous failure is one of them silently winning);
  5. bias-freedom and degree-1 homogeneity survive -- they come from `use_bias=False`,
     never from depthwise-vs-cross-channel; and
  6. every stated-intent conflict RAISES rather than no-ops.

The homogeneity instrument (`max_homogeneity_relative_error`, `HOMOGENEITY_RTOL`) is
imported from `test_bfconvunext_gabor`, so this arm's homogeneity claim is measured on
the same instrument -- and against the same derived bound -- as the `Conv2D` arm's.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_models/test_bias_free_denoisers/test_the_depthwise_gabor_stem_arm.py -q
"""

import keras
import numpy as np
import pytest
from typing import Tuple

from dl_techniques.initializers import GaborFiltersInitializer
from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
)

from .test_bfconvunext_gabor import (
    HOMOGENEITY_RTOL,
    max_homogeneity_relative_error,
)

# Small but REAL topology. `initial_filters` is deliberately NOT a multiple of
# CHANNELS * PER_CHANNEL, so the projection is doing actual work and a test that
# accidentally dropped it would change the width.
CHANNELS: int = 3
PER_CHANNEL: int = 4
KERNEL: int = 7
INPUT_SHAPE: Tuple[int, int, int] = (32, 32, CHANNELS)
STEM_WIDTH: int = CHANNELS * PER_CHANNEL  # 12


def _build(**overrides):
    cfg = dict(
        input_shape=INPUT_SHAPE,
        depth=2,
        initial_filters=16,
        blocks_per_level=1,
        convnext_version='v1',
        block_normalization='batchnorm',
        use_gabor_stem=True,
        gabor_filters_per_channel=PER_CHANNEL,
        gabor_kernel_size=KERNEL,
    )
    cfg.update(overrides)
    return create_convunext_denoiser(**cfg)


class TestTheArmIsOptIn:
    """The default must be the 2026-09-06 cross-channel warm start, unchanged."""

    def test_none_still_builds_the_cross_channel_conv2d(self) -> None:
        model = _build(gabor_filters_per_channel=None, gabor_filters=8)
        stem = model.get_layer('gabor_stem')
        assert type(stem) is keras.layers.Conv2D
        # `gabor_filters` is still an OUTPUT channel count on this arm, not a multiplier.
        assert stem.output.shape[-1] == 8

    def test_the_parameter_defaults_to_none(self) -> None:
        """Omitting it entirely must be identical to passing None -- otherwise every
        caller written before this argument existed silently changed shape."""
        stem = create_convunext_denoiser(
            input_shape=INPUT_SHAPE, depth=2, initial_filters=16, blocks_per_level=1,
            block_normalization='batchnorm', use_gabor_stem=True, gabor_filters=8,
            gabor_kernel_size=KERNEL,
        ).get_layer('gabor_stem')
        assert type(stem) is keras.layers.Conv2D
        assert stem.kernel.shape == (KERNEL, KERNEL, CHANNELS, 8)


class TestTheDepthwiseStem:
    def test_the_stem_is_a_depthwise_conv2d(self) -> None:
        """`type(...) is` deliberately: `DepthwiseConv2D` is not a `Conv2D` subclass in
        Keras 3.8, and `SeparableConv2D` is a third construction again."""
        stem = _build().get_layer('gabor_stem')
        assert type(stem) is keras.layers.DepthwiseConv2D, (
            f"depthwise arm must build a DepthwiseConv2D, got {type(stem).__name__}"
        )
        assert stem.depth_multiplier == PER_CHANNEL

    def test_the_widths_multiply(self) -> None:
        """A depthwise convolution does not sum across input channels, so the stem
        emits `channels * per_channel`, NOT `per_channel`."""
        stem = _build().get_layer('gabor_stem')
        assert stem.output.shape[-1] == STEM_WIDTH
        assert stem.output.shape[-1] != PER_CHANNEL
        assert stem.kernel.shape == (KERNEL, KERNEL, CHANNELS, PER_CHANNEL), (
            "depthwise kernel is (kh, kw, in_ch, depth_multiplier)"
        )

    def test_the_kernel_holds_the_gabor_bank_values(self) -> None:
        """The stem's kernel IS the Gabor bank, not merely a depthwise conv of the
        right shape. Mirrors `test_bfconvunext_gabor`'s check on the Conv2D arm: the
        builder threads none of the Gabor shaping parameters, so the expected bank is
        `GaborFiltersInitializer()` evaluated at the stem's own kernel shape."""
        kernel = np.asarray(_build().get_layer('gabor_stem').kernel)
        expected = np.asarray(
            GaborFiltersInitializer()((KERNEL, KERNEL, CHANNELS, PER_CHANNEL))
        )
        np.testing.assert_allclose(kernel, expected, atol=1e-6, rtol=0)

    def test_the_stem_is_trainable_and_receives_gradient(self) -> None:
        """Built `trainable=True`, the OPPOSITE of `create_gabor_depthwise_conv2d`'s own
        default -- so that `--freeze-gabor-stem` stays the ONE knob that freezes a stem.
        Both halves are checked: the flag, and that the kernel actually reaches the
        model's trainable weights."""
        model = _build()
        stem = model.get_layer('gabor_stem')
        assert stem.trainable is True
        assert stem.kernel.path in {w.path for w in model.trainable_weights}

    def test_the_stem_and_its_projection_are_bias_free(self) -> None:
        model = _build()
        stem = model.get_layer('gabor_stem')
        assert stem.use_bias is False
        assert stem.bias is None
        assert len(stem.trainable_weights) == 1  # the kernel, nothing else
        assert model.get_layer('gabor_stem_projection').use_bias is False


class TestItIsGenuinelyPerChannel:
    """The reason the arm exists, tested by ablation rather than by layer class.

    A `DepthwiseConv2D` of the right shape would still pass every structural check
    above if the builder wired it to the wrong tensor. What cannot be faked is the
    RESPONSE: with only input channel `c` non-zero, exactly the `PER_CHANNEL` output
    channels belonging to `c` may be non-zero. The cross-channel `Conv2D` arm is run
    through the same probe as a negative control, because a probe that both arms pass
    measures nothing.
    """

    @staticmethod
    def _stem_response(model, active_channel: int) -> np.ndarray:
        """Max |stem output| per output channel, with only `active_channel` fed."""
        rng = np.random.default_rng(7)
        x = np.zeros((1, *INPUT_SHAPE), dtype="float32")
        x[..., active_channel] = rng.uniform(
            0.5, 1.0, size=(1, INPUT_SHAPE[0], INPUT_SHAPE[1])
        ).astype("float32")
        stem = model.get_layer('gabor_stem')
        # `model.inputs[0]`, not `model.inputs`: a 1-element LIST input structure makes
        # Keras warn "structure of `inputs` doesn't match" on every bare-tensor call
        # (the same trap `train.bfunet.common.train` documents for expose_bottleneck),
        # and this suite escalates warnings to errors.
        probe = keras.Model(model.inputs[0], stem.output)
        return np.max(np.abs(np.asarray(probe(x, training=False))), axis=(0, 1, 2))

    @pytest.mark.parametrize("active", range(CHANNELS))
    def test_only_one_channels_block_responds(self, active: int) -> None:
        response = self._stem_response(_build(), active)
        assert response.shape == (STEM_WIDTH,)
        # Keras lays a depthwise output out as channel-major:
        # [(c0,j0)..(c0,jM-1), (c1,j0)..]. Block `active` is the live one.
        lo, hi = active * PER_CHANNEL, (active + 1) * PER_CHANNEL
        live, dead = response[lo:hi], np.delete(response, np.s_[lo:hi])
        assert live.max() > 1e-3, (
            "the stem produced no response to the one channel that was fed"
        )
        np.testing.assert_allclose(dead, 0.0, atol=1e-6, rtol=0)

    def test_the_cross_channel_arm_fails_this_probe(self) -> None:
        """Negative control. The `Conv2D` stem sums over input channels, so EVERY
        output channel responds to a single live input channel. If this ever passes,
        the probe above is not measuring per-channel isolation."""
        response = self._stem_response(
            _build(gabor_filters_per_channel=None, gabor_filters=STEM_WIDTH), 0
        )
        assert response.shape == (STEM_WIDTH,)
        assert float(response[PER_CHANNEL:].max()) > 1e-3, (
            "cross-channel Conv2D left the other output channels silent -- the "
            "isolation probe cannot distinguish the two arms"
        )


class TestGaborFiltersIsNotReadOnThisArm:
    """The two counts are alternatives. The dangerous failure is one silently winning."""

    def test_the_graph_is_independent_of_gabor_filters(self) -> None:
        a = _build(gabor_filters=8)
        b = _build(gabor_filters=999)
        assert [w.shape for w in a.weights] == [w.shape for w in b.weights]
        assert a.count_params() == b.count_params()
        assert (a.get_layer('gabor_stem').output.shape[-1]
                == b.get_layer('gabor_stem').output.shape[-1] == STEM_WIDTH)

    def test_the_stem_width_is_not_gabor_filters(self) -> None:
        """A build whose `gabor_filters` happens to be a plausible width must still
        produce the depthwise width, not that one."""
        stem = _build(gabor_filters=STEM_WIDTH + 1).get_layer('gabor_stem')
        assert stem.output.shape[-1] == STEM_WIDTH


class TestBiasFreeGuaranteesSurvive:
    def test_the_depthwise_arm_is_positively_homogeneous(self) -> None:
        """D(a*x) == a*D(x). Homogeneity comes from `use_bias=False`, so the depthwise
        arm must clear the SAME derived bound the `Conv2D` arm does."""
        worst = max_homogeneity_relative_error(_build(), INPUT_SHAPE)
        assert worst <= HOMOGENEITY_RTOL, (
            f"depthwise-stem denoiser is not degree-1 homogeneous: {worst:.3e} > "
            f"{HOMOGENEITY_RTOL:.3e}"
        )

    def test_the_homogeneity_guard_can_fail(self) -> None:
        """RED proof for the assertion above, on THIS arm's OWN stem builder.

        The bound is shared with the `Conv2D` arm, so it is already known to be
        falsifiable there -- but a bound that a DEPTHWISE bias-carrying stack slips
        under would still make the assertion above vacuous, and depthwise convolution
        has fewer accumulations per output. Injected through the builder's module
        namespace so the graph is rebuilt around it (a post-hoc attribute poke is
        ignored by the already-traced functional graph, and `add_weight` on a built
        layer raises).
        """
        import dl_techniques.models.vision.convunext.model as cvx_model

        real = cvx_model.create_gabor_depthwise_conv2d

        def biased(**kwargs):
            kwargs['use_bias'] = True
            return real(**kwargs)

        cvx_model.create_gabor_depthwise_conv2d = biased
        try:
            broken = _build()
            stem = broken.get_layer('gabor_stem')
            assert stem.use_bias is True and stem.bias is not None, (
                "injection did not take -- the RED proof would be vacuous"
            )
            # Keras zero-initializes the bias, and a ZERO bias is still homogeneous, so
            # it must be assigned a non-zero value or this proves nothing (this repo's
            # recorded vacuous-negative-control defect).
            stem.bias.assign(np.full(stem.bias.shape, 0.3, dtype="float32"))
            err = max_homogeneity_relative_error(broken, INPUT_SHAPE)
        finally:
            cvx_model.create_gabor_depthwise_conv2d = real

        assert err > HOMOGENEITY_RTOL, (
            f"the homogeneity guard CANNOT FAIL: a depthwise stem carrying a 0.3 bias "
            f"still measured {err:.3e} <= {HOMOGENEITY_RTOL:.3e}"
        )

    def test_the_homogeneity_injection_is_restored(self) -> None:
        """The RED proof must not leak its monkeypatch into the rest of the session."""
        import dl_techniques.models.vision.convunext.model as cvx_model
        from dl_techniques.initializers import create_gabor_depthwise_conv2d as real
        assert cvx_model.create_gabor_depthwise_conv2d is real
        assert _build().get_layer('gabor_stem').use_bias is False

    def test_keras_round_trip(self, tmp_path) -> None:
        model = _build()
        x = np.random.default_rng(3).uniform(
            0.0, 1.0, size=(2, *INPUT_SHAPE)
        ).astype("float32")
        before = np.asarray(model(x, training=False))
        path = tmp_path / "depthwise_gabor.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)
        stem = reloaded.get_layer('gabor_stem')
        # The stem must survive as a DepthwiseConv2D -- a graph that reloads as a
        # Conv2D of the same width would be a different model with the same PSNR.
        assert type(stem) is keras.layers.DepthwiseConv2D
        assert stem.kernel.shape == (KERNEL, KERNEL, CHANNELS, PER_CHANNEL)
        np.testing.assert_allclose(
            np.asarray(reloaded(x, training=False)), before, atol=1e-6, rtol=0
        )


class TestNoProjection:
    """With the 1x1 dropped, the depthwise rule is `channels * per_channel ==
    initial_filters` -- the pre-2026-09-06 rule, back for THIS arm only."""

    def test_matching_width_builds_and_drops_the_projection(self) -> None:
        model = _build(initial_filters=STEM_WIDTH, gabor_stem_projection=False)
        assert model.get_layer('gabor_stem').output.shape[-1] == STEM_WIDTH
        assert not any(l.name == 'gabor_stem_projection' for l in model.layers)

    def test_mismatched_width_raises(self) -> None:
        with pytest.raises(ValueError, match="gabor_filters_per_channel"):
            _build(initial_filters=STEM_WIDTH + 4, gabor_stem_projection=False)

    def test_the_cross_channel_rule_does_not_leak_into_this_arm(self) -> None:
        """`initial_filters == gabor_filters_per_channel` satisfies the OTHER arm's
        shape of rule and must NOT be accepted here."""
        with pytest.raises(ValueError, match="gabor_filters_per_channel"):
            _build(initial_filters=PER_CHANNEL, gabor_stem_projection=False)


class TestStatedIntentConflictsRaise:
    """A flag that reaches no layer must raise, never no-op."""

    def test_without_a_stem_it_raises(self) -> None:
        with pytest.raises(ValueError, match="use_gabor_stem=False"):
            _build(use_gabor_stem=False)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_multiplier_raises(self, bad: int) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            _build(gabor_filters_per_channel=bad)
