"""``SqueezeExcitation(divisor=...)``: the bottleneck width rule.

``divisor=1`` (the default) keeps the rule every consumer was built with,
``max(1, round(C * reduction_ratio))``. A divisor above 1 applies the
MobileNet reference rounding, Keras Applications ``mobilenet_v3._depth`` and
torchvision ``models/_utils._make_divisible`` (the same formula): the nearest
multiple of ``divisor``, at least ``divisor``, bumped one step when rounding
fell more than 10% below the value. Both rules are re-typed here as oracles,
so the test does not read the formula it checks.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.squeeze_excitation import SqueezeExcitation

CHANNELS = range(1, 1025)


def _reference_make_divisible(v: float, divisor: int) -> int:
    """torchvision ``_make_divisible`` / Keras Applications ``_depth``."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _width(channels: int, **kwargs) -> int:
    layer = SqueezeExcitation(**kwargs)
    layer.build((None, channels))
    return layer.bottleneck_channels


@pytest.mark.parametrize("ratio", [0.25, 0.0625, 0.125, 0.5, 1.0, 0.01])
def test_the_default_keeps_the_nearest_integer_rule(ratio):
    widths = [_width(c, reduction_ratio=ratio) for c in CHANNELS]
    assert widths == [max(1, round(c * ratio)) for c in CHANNELS]


@pytest.mark.parametrize("divisor", [8, 4])
def test_a_divisor_applies_the_mobilenet_reference_rounding(divisor):
    widths = [_width(c, reduction_ratio=0.25, divisor=divisor) for c in CHANNELS]
    assert widths == [_reference_make_divisible(c * 0.25, divisor) for c in CHANNELS]
    # The two rules really differ (anti-vacuity): 16 channels -> 4 vs 8.
    assert _width(16, reduction_ratio=0.25) == 4
    assert _width(16, reduction_ratio=0.25, divisor=8) == 8


@pytest.mark.parametrize("bad", [0, -8, 2.5, True, "8", None])
def test_a_non_positive_or_non_int_divisor_is_refused(bad):
    with pytest.raises(ValueError, match="divisor"):
        SqueezeExcitation(divisor=bad)


def test_the_divisor_round_trips_through_the_config():
    layer = SqueezeExcitation(reduction_ratio=0.25, divisor=8)
    assert layer.get_config()["divisor"] == 8
    clone = SqueezeExcitation.from_config(layer.get_config())
    x = np.random.default_rng(0).standard_normal((2, 4, 4, 16)).astype("float32")
    clone(x)
    assert clone.bottleneck_channels == 8
    assert SqueezeExcitation().get_config()["divisor"] == 1


def test_the_uib_squeeze_excite_is_the_shared_layer_with_its_divisor():
    """UniversalInvertedBottleneck (MobileNetV3's block) builds its SE as this
    layer, not a private copy, and ``se_divisor`` reaches it and its config."""
    from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import (
        UniversalInvertedBottleneck,
    )
    block = UniversalInvertedBottleneck(
        filters=16, expanded_channels=72, use_squeeze_excitation=True,
        se_divisor=8)
    block(np.zeros((1, 8, 8, 16), "float32"))
    assert isinstance(block.se, SqueezeExcitation)
    assert block.se.bottleneck_channels == _reference_make_divisible(72 * 0.25, 8) == 24
    clone = UniversalInvertedBottleneck.from_config(block.get_config())
    clone(np.zeros((1, 8, 8, 16), "float32"))
    assert clone.se.divisor == 8 and clone.se.bottleneck_channels == 24


def test_the_gate_activation_defaults_to_sigmoid_and_hard_sigmoid_is_relu6_over_6(
        golden_reference_device):
    """``gate_activation`` (step 9.3.1): the default keeps the sigmoid gate
    every consumer was built with; ``'hard_sigmoid'`` is the MobileNetV3
    reference ``relu6(z + 3) / 6``, checked by value against the layer's own
    weights with both saturated ends reached. On the CPU golden device: the
    GPU runs the 1x1 convs in TF32 (measured 1.3e-4 off the float32 oracle)."""
    with keras.device(golden_reference_device):
        _check_the_two_gates()


def _check_the_two_gates():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3, 3, 16)).astype("float32")
    for gate, reference in (
            ("sigmoid", lambda z: 1.0 / (1.0 + np.exp(-z))),
            ("hard_sigmoid", lambda z: np.clip(z + 3.0, 0.0, 6.0) / 6.0)):
        kwargs = {} if gate == "sigmoid" else {"gate_activation": gate}
        layer = SqueezeExcitation(reduction_ratio=0.5, divisor=8, use_bias=True, **kwargs)
        layer(x)
        w1, b1 = (np.asarray(v) for v in layer.conv_reduce.weights)
        w2, b2 = (np.asarray(v) for v in layer.conv_restore.weights)
        b2 = 4.0 * rng.standard_normal(b2.shape).astype("float32")
        layer.conv_restore.bias.assign(b2)
        hidden = np.maximum(x.mean(axis=(1, 2), keepdims=True) @ w1[0, 0] + b1, 0.0)
        z = hidden @ w2[0, 0] + b2
        if gate == "hard_sigmoid":
            assert (z <= -3).any() and (z >= 3).any()
        np.testing.assert_allclose(np.asarray(layer(x)), x * reference(z),
                                   rtol=1e-5, atol=1e-6)
        assert layer.get_config()["gate_activation"] == gate
        clone = SqueezeExcitation.from_config(layer.get_config())
        clone(x)
        clone.set_weights(layer.get_weights())
        np.testing.assert_array_equal(np.asarray(clone(x)), np.asarray(layer(x)))


def test_the_uib_passes_its_se_bias_and_gate():
    from dl_techniques.layers.conv_blocks.universal_inverted_bottleneck import (
        UniversalInvertedBottleneck,
    )
    block = UniversalInvertedBottleneck(
        filters=16, expanded_channels=72, use_squeeze_excitation=True,
        se_divisor=8, se_use_bias=True, se_gate_activation="hard_sigmoid")
    block(np.zeros((1, 8, 8, 16), "float32"))
    assert block.se.use_bias and block.se.get_config()["gate_activation"] == "hard_sigmoid"
    config = block.get_config()
    assert (config["se_use_bias"], config["se_gate_activation"]) == (True, "hard_sigmoid")
    # Independent of the block's own use_bias (the references' SE convs are
    # biased while every other conv is followed by a BatchNorm).
    assert block.use_bias is False
