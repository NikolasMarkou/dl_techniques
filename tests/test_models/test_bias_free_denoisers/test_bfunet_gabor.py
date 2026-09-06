"""
Test suite for the optional TRAINABLE Gabor warm-start stem of the bias-free U-Net
denoiser (models/vision/bias_free_denoisers/bfunet.py,
`# DECISION standalone-2026-09-06-gabor-warm-start/D-002`).

`bfunet.py` carries its OWN `_build_gabor_stem`, independent of the one in
`models/vision/convunext/model.py`. Two independent implementations of one pattern is
exactly how the two arms drift apart, so this module is the deliberate mirror of
`test_bfconvunext_gabor.py`: the same claims, asserted against the other builder, plus
one cross-builder agreement test that fails if either side moves alone.

The stem is the construction of Ozbulak & Ekenel (SIU 2018): a bias-free CROSS-CHANNEL
`keras.layers.Conv2D` whose kernel is initialized from a Gabor bank and then TRAINED.
It replaced a frozen `DepthwiseConv2D` bank, so two things changed at once:

  * the layer class and its trainability (Conv2D / trainable, not DepthwiseConv2D /
    frozen), and
  * `gabor_filters`, which is now the stem's OUTPUT CHANNEL COUNT (a Conv2D `filters`)
    rather than a depthwise `depth_multiplier` emitting `in_ch * gabor_filters`.
"""

import os
from typing import Tuple

import keras
import numpy as np
import pytest

from dl_techniques.initializers import GaborFiltersInitializer
from dl_techniques.models.vision.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
)
from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
)

# The homogeneity bound and its probe are OWNED by the convunext module and imported,
# not re-spelled. One rule, one home: a second copy of a numeric bound kept in lockstep
# by hand is this repo's recorded defect class, and it is precisely the drift this
# module exists to prevent.
from .test_bfconvunext_gabor import (  # noqa: F401
    HOMOGENEITY_ALPHAS,
    HOMOGENEITY_RTOL,
    max_homogeneity_relative_error,
)

INPUT_SHAPE: Tuple[int, int, int] = (64, 64, 3)


def _build_gabor(input_shape=INPUT_SHAPE, **overrides):
    cfg = dict(
        input_shape=input_shape,
        depth=3,
        initial_filters=16,
        blocks_per_level=1,
        use_gabor_stem=True,
        gabor_filters=8,
        gabor_kernel_size=7,
    )
    cfg.update(overrides)
    return create_bfunet_denoiser(**cfg)


class TestBfunetGaborStem:
    """Trainable cross-channel Gabor warm-start stem on create_bfunet_denoiser."""

    def test_build_and_full_res_output(self) -> None:
        model = _build_gabor()
        x = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
        y = model(x, training=False)
        assert tuple(y.shape) == (2, *INPUT_SHAPE)

    def test_gabor_layer_is_a_trainable_cross_channel_conv2d(self) -> None:
        """The stem is the paper's TRAINABLE Conv2D warm start, not a frozen bank.

        Both halves matter and neither implies the other: a `Conv2D` could still be
        shipped frozen, and a `DepthwiseConv2D` could be shipped trainable.
        """
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        assert type(gabor) is keras.layers.Conv2D, (
            f"stem must be a plain Conv2D (the Ozbulak & Ekenel warm start), got "
            f"{type(gabor).__name__}"
        )
        assert not isinstance(gabor, keras.layers.DepthwiseConv2D)
        assert gabor.trainable is True
        assert len(gabor.trainable_weights) == 1
        assert gabor.use_bias is False
        assert gabor.bias is None
        n_trainable = int(sum(np.prod(w.shape) for w in gabor.trainable_weights))
        assert n_trainable == 7 * 7 * 3 * 8

    def test_gabor_kernel_holds_the_gabor_bank_values(self) -> None:
        """The stem's kernel IS the Gabor bank, not merely a Conv2D of the right shape."""
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        kernel = np.asarray(gabor.kernel)
        assert kernel.shape == (7, 7, 3, 8)
        expected = np.asarray(GaborFiltersInitializer()((7, 7, 3, 8)))
        np.testing.assert_allclose(kernel, expected, atol=1e-6, rtol=0)
        for c in range(1, 3):
            np.testing.assert_allclose(
                kernel[:, :, 0, :], kernel[:, :, c, :], atol=1e-6, rtol=0
            )

    def test_gabor_stem_kernel_receives_gradient(self) -> None:
        """`trainable=True` is not decoration: the kernel is in the model's trainables."""
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        assert gabor.kernel.path in {w.path for w in model.trainable_weights}

    def test_gabor_output_channels(self) -> None:
        """`gabor_filters` IS the output channel count -- NOT in_ch * gabor_filters."""
        model = _build_gabor()
        gabor = model.get_layer('gabor_stem')
        assert gabor.output.shape[-1] == 8
        assert gabor.output.shape[-1] != 3 * 8

    def test_projection_is_bias_free(self) -> None:
        model = _build_gabor()
        proj = model.get_layer('gabor_stem_projection')
        assert proj.use_bias is False
        assert proj.filters == 16

    def test_default_has_no_gabor(self) -> None:
        model = create_bfunet_denoiser(
            input_shape=INPUT_SHAPE, depth=3, initial_filters=16, blocks_per_level=1,
        )
        names = [l.name for l in model.layers]
        assert 'gabor_stem' not in names
        assert 'gabor_stem_projection' not in names

    def test_keras_round_trip(self, tmp_path) -> None:
        model = _build_gabor()
        x = np.random.rand(2, *INPUT_SHAPE).astype(np.float32)
        y_before = model(x, training=False)

        save_path = os.path.join(str(tmp_path), 'bfunet_gabor.keras')
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x, training=False)

        gabor = loaded.get_layer('gabor_stem')
        assert type(gabor) is keras.layers.Conv2D
        assert gabor.trainable is True
        assert len(gabor.trainable_weights) == 1
        np.testing.assert_allclose(
            np.asarray(gabor.kernel),
            np.asarray(model.get_layer('gabor_stem').kernel),
            atol=1e-6, rtol=0,
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            atol=1e-4,
            err_msg="Outputs differ after .keras round-trip with Gabor stem",
        )


class TestBfunetGaborStemNoProjection:
    """`gabor_stem_projection=False` requires `gabor_filters == initial_filters`.

    The old contract was `input_channels * gabor_filters == initial_filters`. It is
    gone: the stem no longer multiplies by the input channel count.
    """

    def test_no_projection_layer_and_full_res_output(self) -> None:
        model = _build_gabor(gabor_stem_projection=False,
                             gabor_filters=24, initial_filters=24)
        names = [l.name for l in model.layers]
        assert 'gabor_stem' in names
        assert 'gabor_stem_projection' not in names
        x = np.random.uniform(0.0, 1.0, size=(2, *INPUT_SHAPE)).astype("float32")
        y = model(x, training=False)
        assert tuple(y.shape) == (2, *INPUT_SHAPE)

    def test_gabor_feeds_initial_filters_channels(self) -> None:
        model = _build_gabor(gabor_stem_projection=False,
                             gabor_filters=24, initial_filters=24)
        gabor = model.get_layer('gabor_stem')
        assert gabor.output.shape[-1] == 24
        assert tuple(gabor.kernel.shape) == (7, 7, 3, 24)

    def test_no_projection_is_bias_free(self) -> None:
        model = _build_gabor(gabor_stem_projection=False,
                             gabor_filters=24, initial_filters=24)
        offenders = [
            l.name for l in model._flatten_layers()
            if getattr(l, "use_bias", False)
            or (isinstance(l, keras.layers.LayerNormalization)
                and getattr(l, "center", False))
        ]
        assert offenders == [], f"bias/centering survived: {offenders}"

    def test_channel_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_filters"):
            _build_gabor(gabor_stem_projection=False, initial_filters=16)

    def test_the_old_per_channel_match_no_longer_satisfies_the_contract(self) -> None:
        """A config legal under the OLD rule must now RAISE, not silently build."""
        with pytest.raises(ValueError, match="gabor_filters"):
            _build_gabor(gabor_stem_projection=False, initial_filters=24)


class TestBfunetGaborStemHomogeneity:
    """The bias-free arm stays positively homogeneous with the Conv2D Gabor stem.

    `block_normalization='batchnorm'` is required: bfunet's `'layernorm'` option is
    scale-INVARIANT (degree 0) and breaks homogeneity on its own -- measured once on
    this configuration at a relative error of 2.98e+00, i.e. the norm, not the stem,
    is the offender there.
    """

    SHAPE: Tuple[int, int, int] = (32, 32, 3)

    def _build(self, **overrides):
        cfg = dict(
            input_shape=self.SHAPE, depth=2, initial_filters=16, blocks_per_level=1,
            block_normalization='batchnorm',
            use_gabor_stem=True, gabor_filters=8, gabor_kernel_size=5,
        )
        cfg.update(overrides)
        return create_bfunet_denoiser(**cfg)

    def test_bias_free_arm_is_positively_homogeneous(self) -> None:
        model = self._build()
        err = max_homogeneity_relative_error(model, self.SHAPE)
        assert err <= HOMOGENEITY_RTOL, (
            f"D(a*x) != a*D(x): max relative error {err:.3e} exceeds the derived "
            f"float32 bound {HOMOGENEITY_RTOL:.3e} (= 64 * eps)"
        )

    def test_the_homogeneity_guard_can_fail(self) -> None:
        """RED proof: put a bias on the Gabor stem and the SAME probe must exceed the bound.

        The injection is exactly what the stem's hardcoded `use_bias=False` prevents,
        applied through bfunet's own module namespace so the whole functional graph is
        rebuilt around it. A post-hoc attribute poke would be ignored by the already
        traced graph and would prove nothing.
        """
        import dl_techniques.models.vision.bias_free_denoisers.bfunet as bfunet_mod

        real = bfunet_mod.create_gabor_conv2d

        def biased(**kwargs):
            kwargs['use_bias'] = True
            return real(**kwargs)

        bfunet_mod.create_gabor_conv2d = biased
        try:
            broken = self._build()
            stem = broken.get_layer('gabor_stem')
            assert stem.use_bias is True and stem.bias is not None, (
                "injection did not take -- the RED proof would be vacuous"
            )
            # Keras zero-initializes a Conv2D bias, and a ZERO bias is still
            # homogeneous, so the injection must assign a non-zero value.
            stem.bias.assign(np.full(stem.bias.shape, 0.3, dtype="float32"))
            err = max_homogeneity_relative_error(broken, self.SHAPE)
        finally:
            bfunet_mod.create_gabor_conv2d = real

        assert err > HOMOGENEITY_RTOL, (
            f"the homogeneity guard CANNOT FAIL: a stem carrying a 0.3 bias still "
            f"measured {err:.3e} <= {HOMOGENEITY_RTOL:.3e}"
        )

    def test_the_homogeneity_injection_is_restored(self) -> None:
        """The RED proof must not leak its monkeypatch into the rest of the session."""
        import dl_techniques.models.vision.bias_free_denoisers.bfunet as bfunet_mod
        from dl_techniques.initializers import create_gabor_conv2d as real
        assert bfunet_mod.create_gabor_conv2d is real
        assert self._build().get_layer('gabor_stem').use_bias is False


class TestTheTwoGaborStemsAgree:
    """convunext and bfunet must build the SAME stem from the same knobs.

    They are two independent `_build_gabor_stem` implementations. Nothing in the import
    graph forces them to agree, so this is the guard that notices when one is changed
    and the other is not.
    """

    SHAPE: Tuple[int, int, int] = (32, 32, 3)
    KWARGS = dict(depth=2, initial_filters=16, blocks_per_level=1,
                  block_normalization='batchnorm', use_gabor_stem=True,
                  gabor_filters=8, gabor_kernel_size=5)

    def _stems(self):
        a = create_convunext_denoiser(input_shape=self.SHAPE, convnext_version='v1',
                                      **self.KWARGS).get_layer('gabor_stem')
        b = create_bfunet_denoiser(input_shape=self.SHAPE,
                                   **self.KWARGS).get_layer('gabor_stem')
        return a, b

    def test_same_class_shape_trainability_and_kernel(self) -> None:
        a, b = self._stems()
        assert type(a) is type(b) is keras.layers.Conv2D
        assert a.trainable is b.trainable is True
        assert a.use_bias is b.use_bias is False
        assert tuple(a.kernel.shape) == tuple(b.kernel.shape) == (5, 5, 3, 8)
        assert a.strides == b.strides == (1, 1)
        assert a.padding == b.padding == 'same'
        assert a.filters == b.filters == 8
        # Same deterministic Gabor bank in both, to the initializer's own tolerance.
        np.testing.assert_allclose(
            np.asarray(a.kernel), np.asarray(b.kernel), atol=1e-6, rtol=0
        )

    def test_both_reject_the_same_no_projection_mismatch(self) -> None:
        """One rule, two builders: the projection contract must be identical."""
        kwargs = dict(self.KWARGS)
        kwargs['gabor_stem_projection'] = False  # gabor_filters(8) != initial(16)
        with pytest.raises(ValueError, match="gabor_filters"):
            create_convunext_denoiser(input_shape=self.SHAPE, convnext_version='v1',
                                      **kwargs)
        with pytest.raises(ValueError, match="gabor_filters"):
            create_bfunet_denoiser(input_shape=self.SHAPE, **kwargs)
