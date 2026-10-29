"""Tests for the UniversalInvertedBottleneck layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import UniversalInvertedBottleneck

B, H, W, C = 2, 8, 8, 8


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestUniversalInvertedBottleneck:

    def test_construction(self):
        layer = UniversalInvertedBottleneck(filters=C)
        assert layer.filters == C

    @pytest.mark.parametrize("bad", [
        {"filters": 0},
        {"filters": C, "stride": 0},
        {"filters": C, "kernel_size": 0},
        {"filters": C, "dropout_rate": 2.0},
        {"filters": C, "expanded_channels": -1},
        {"filters": C, "padding": "bogus"},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            UniversalInvertedBottleneck(**bad)

    def test_causal_padding_is_refused_by_the_layer_itself(self):
        """Keras Conv2D and DepthwiseConv2D have no 'causal'; the layer's own
        check must say so instead of passing it on to a Keras error."""
        with pytest.raises(ValueError, match=r"padding must be 'same' or 'valid'"):
            UniversalInvertedBottleneck(filters=C, padding="causal")

    @pytest.mark.parametrize("kw,out_hw", [
        ({"use_dw1": True, "use_dw2": False}, H),
        ({"use_dw1": True, "use_dw2": True}, H),
        ({"use_dw1": False, "use_dw2": False}, H),         # FFN variant
        ({"use_dw1": True, "stride": 2}, H // 2),
        ({"use_squeeze_excitation": True}, H),
    ])
    def test_forward_pass(self, sample, kw, out_hw):
        layer = UniversalInvertedBottleneck(filters=C, **kw)
        out = layer(sample)
        assert tuple(out.shape) == (B, out_hw, out_hw, C)

    def test_compute_output_shape(self):
        layer = UniversalInvertedBottleneck(filters=16, stride=2)
        assert layer.compute_output_shape((B, H, W, C)) == (B, 4, 4, 16)

    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("use_dw2", [True, False])
    @pytest.mark.parametrize("use_dw1", [True, False])
    @pytest.mark.parametrize("use_start_dw", [True, False])
    def test_stride_is_applied_for_every_depthwise_combination(
        self, sample, use_start_dw, use_dw1, use_dw2, stride
    ):
        """The stride must not depend on which optional depthwise convs exist.

        This crosses ``use_dw1=False`` with ``stride>1`` — a crossing no other
        test in this file makes, which is why the block used to return the
        UNSTRIDED map while ``compute_output_shape`` claimed the strided one.
        Both assertions run on every arm on purpose: the first pins the two
        surfaces against each other, the second pins them to the independently
        computed value, so an arm where both are wrong the same way still fails.

        ``use_start_dw`` is crossed in here rather than pinned by a second test
        because the pre-expansion depthwise took over the HEAD of the stride
        precedence: it is a new opportunity for exactly the same defect, on a
        sub-layer that is built on a different shape than every other one.
        """
        layer = UniversalInvertedBottleneck(
            filters=16, stride=stride, use_start_dw=use_start_dw,
            use_dw1=use_dw1, use_dw2=use_dw2,
        )
        declared = tuple(layer.compute_output_shape((B, H, W, C)))
        actual = tuple(layer(sample).shape)

        assert actual == declared, (
            f"call() returned {actual} but compute_output_shape() declared "
            f"{declared} for use_dw1={use_dw1}, use_dw2={use_dw2}, "
            f"stride={stride}"
        )

        expected_hw = (H + stride - 1) // stride
        assert actual == (B, expected_hw, expected_hw, 16)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = UniversalInvertedBottleneck(
            filters=C, use_dw1=True, use_squeeze_excitation=True, name="uib"
        )(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "uib.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"UniversalInvertedBottleneck": UniversalInvertedBottleneck}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = UniversalInvertedBottleneck(filters=C, expansion_factor=6, use_dw2=True)
        rebuilt = UniversalInvertedBottleneck.from_config(layer.get_config())
        assert rebuilt.filters == C and rebuilt.use_dw2 is True

    def test_start_dw_params_survive_the_config_round_trip(self):
        """A parameter missing from ``get_config`` is restored at its DEFAULT.

        That is a silent pass, not a failure: ``from_config`` would happily
        return a block with ``use_start_dw=False`` and the reader would see an
        IB where an ExtraDW was saved. Both new values are therefore asserted
        against non-default values, so neither can be satisfied by the default.
        """
        layer = UniversalInvertedBottleneck(
            filters=C, use_start_dw=True, start_dw_kernel_size=7, use_dw1=False
        )
        config = layer.get_config()
        assert config["use_start_dw"] is True
        assert config["start_dw_kernel_size"] == 7

        rebuilt = UniversalInvertedBottleneck.from_config(config)
        assert rebuilt.use_start_dw is True
        assert rebuilt.start_dw_kernel_size == 7
        assert rebuilt.start_dw is not None

    def test_start_dw_is_pre_expansion_and_middle_dw_is_post_expansion(self):
        """The two positions differ ONLY in the channel count they see.

        A count of depthwise layers cannot tell a ConvNext block from an IB —
        both own exactly one. The built kernel's input-channel dimension can:
        the start DW is built on the layer input, the middle DW on the expanded
        tensor. That is a structural fact no parameter-count coincidence fakes.
        """
        expansion = 4
        start_only = UniversalInvertedBottleneck(
            filters=16, expansion_factor=expansion,
            use_start_dw=True, use_dw1=False,
        )
        middle_only = UniversalInvertedBottleneck(
            filters=16, expansion_factor=expansion,
            use_start_dw=False, use_dw1=True,
        )
        start_only.build((B, H, W, C))
        middle_only.build((B, H, W, C))

        assert start_only.start_dw.kernel.shape[2] == C
        assert middle_only.dw1.kernel.shape[2] == C * expansion


class TestDropPath:
    """Stochastic-depth rate on the residual branch (MetaFormer MBConv need).

    The reference blocks this layer is being used by (MetaFormer's conv stages)
    schedule a drop-connect rate per block and apply it to the residual BRANCH
    before the add. These tests pin the three load-bearing facts: no sublayer
    at rate zero, inference identity, and training drops the branch — not the
    shortcut, and not a sole output.
    """

    def test_zero_rate_creates_no_sublayer(self):
        layer = UniversalInvertedBottleneck(filters=C, drop_path_rate=0.0)
        with_drop = UniversalInvertedBottleneck(filters=C, drop_path_rate=0.25)
        assert layer.drop_path is None
        assert with_drop.drop_path is not None
        assert with_drop.drop_path.drop_path_rate == 0.25

    def test_invalid_rates_raise(self):
        with pytest.raises(ValueError):
            UniversalInvertedBottleneck(filters=C, drop_path_rate=1.0)
        with pytest.raises(ValueError):
            UniversalInvertedBottleneck(filters=C, drop_path_rate=-0.1)
        with pytest.raises(TypeError):
            UniversalInvertedBottleneck(filters=C, drop_path_rate="0.1")

    def test_rate_survives_get_config_round_trip(self):
        layer = UniversalInvertedBottleneck(
            filters=C, dropout_rate=0.1, drop_path_rate=0.3
        )
        config = layer.get_config()
        assert config["drop_path_rate"] == 0.3
        rebuilt = UniversalInvertedBottleneck.from_config(config)
        assert rebuilt.drop_path_rate == 0.3
        assert rebuilt.drop_path is not None

    def test_inference_with_rate_is_weight_identical_to_rate_zero(self, sample):
        """At inference the layer must be a pure identity, not a rescale.

        StochasticDepth's ``1 / keep_prob`` correction is training-only, so a
        rate-0.5 block run with ``training=False`` must return exactly what a
        rate-0 block with the SAME weights returns.
        """
        plain = UniversalInvertedBottleneck(filters=C, drop_path_rate=0.0)
        steep = UniversalInvertedBottleneck(filters=C, drop_path_rate=0.5)
        plain(sample)
        steep(sample)
        steep.set_weights(plain.get_weights())

        y_plain = plain(sample, training=False)
        y_steep = steep(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_plain),
            keras.ops.convert_to_numpy(y_steep),
            rtol=0, atol=0,
        )

    def test_the_drop_path_drops_the_branch_not_the_shortcut(self, sample):
        """A red proof recording: the guard below was RED against a mutant.

        The mutant applied the drop-path to the shortcut instead of the branch
        (``shortcut = drop_path(shortcut)``). Asserting only that ``some``
        outputs changed cannot see that defect — with a rate of 0.99 most
        samples change either way. What sees it, exactly, is the dropped
        sample's output: with the branch dropped it is the identity
        ``shortcut + 0``; with the shortcut dropped it is the branch term
        ``0 + projection(x)``, equal to the input only by coincidence of the
        projection being zero, which a block owned by this suite's B_*xH*W
        forward pass never is.
        """
        keras.utils.set_random_seed(7)
        batch = 128
        rate = 0.99
        inp = np.random.default_rng(0).standard_normal(
            (batch, H, W, C)).astype("float32")

        layer = UniversalInvertedBottleneck(
            filters=C, drop_path_rate=rate
        )
        out = keras.ops.convert_to_numpy(layer(inp, training=True))
        inp_np = keras.ops.convert_to_numpy(inp)

        dropped = np.all(out == inp_np, axis=(1, 2, 3))
        n_dropped = int(dropped.sum())
        # rate 0.99 on 128 samples: the all-kept event has probability
        # ~1e-258, so a floor FAR below the expectation is deterministic-safe
        # in every direction — higher and the mask is being misread; lower,
        # the layer is not dropping whole samples at this rate.
        assert n_dropped >= 100, (
            f"expected >= 100 of 128 fully-identity samples at rate {rate}, "
            f"got {n_dropped}"
        )
        # And the kept minority must differ, i.e. the branch is live when not
        # dropped. If the mutant dropped instead the SHORTCUT, kept samples
        # (mask=0 on the addend) would read 0 + branch, and no sample would be
        # the plain input at all — the exact failure mode the name records.
        assert n_dropped < batch, "no sample kept: drop-path is not the branch"

    def test_stride_two_block_never_enters_the_drop_path(self, sample):
        """A stride-2 block has no residual, so rate must have zero effect."""
        keras.utils.set_random_seed(11)
        plain = UniversalInvertedBottleneck(
            filters=C, stride=2, drop_path_rate=0.0
        )
        steep = UniversalInvertedBottleneck(
            filters=C, stride=2, drop_path_rate=0.9
        )
        plain(sample)
        steep(sample)
        steep.set_weights(plain.get_weights())

        y_plain = plain(sample, training=True)
        y_steep = steep(sample, training=True)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_plain),
            keras.ops.convert_to_numpy(y_steep),
            rtol=0, atol=0,
        )
