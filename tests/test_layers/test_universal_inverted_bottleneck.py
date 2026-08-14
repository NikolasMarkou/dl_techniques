"""Tests for the UniversalInvertedBottleneck layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck

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
