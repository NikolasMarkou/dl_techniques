"""``EnhancementHead`` can express a restoration network's tail conv exactly.

The restoration models end in ONE 3x3 conv to the image channels: SCUNet's
``m_tail`` (no bias) and DarkIR's ``ending`` (bias), with the long skip or the
global residual applied by the model. Before plan-2026-10-14T062616-81938e81
step 6.1 the head always built three ``ConvBlock``s before its output conv, so
it could not be that tail, and both models' default output was a different
network (D-021, D-022). The head gained two arguments, following the
``ClassificationHead.block_dims`` precedent (D-013):

* ``block_dims``: ``None`` keeps the three ``hidden_dim`` blocks, ``()`` builds
  none, a tuple sets each width; validated by the helper ``ClassificationHead``
  shares;
* ``output_use_bias``: whether the output conv has a bias.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision import ClassificationHead, EnhancementHead


def _x(channels=8):
    return np.random.default_rng(1).standard_normal((2, 8, 8, channels)).astype("float32")


@pytest.mark.parametrize("use_bias", [False, True])
def test_no_block_dims_is_exactly_one_conv(use_bias, golden_reference_device):
    x = _x()
    with keras.device(golden_reference_device):
        head = EnhancementHead(output_channels=3, block_dims=(),
                               output_use_bias=use_bias, hidden_dim=8)
        y = keras.ops.convert_to_numpy(head(x)['enhanced'])
        conv = head.output_conv
        ref = keras.ops.conv(x, conv.kernel, padding="same")
        if use_bias:
            conv.bias.assign(np.linspace(-1, 1, 3).astype("float32"))
            y = keras.ops.convert_to_numpy(head(x)['enhanced'])
            ref = ref + conv.bias
        ref = keras.ops.convert_to_numpy(ref)
    assert head.enhance_blocks == []
    expected = ["output_conv/kernel"] + (["output_conv/bias"] if use_bias else [])
    assert [w.path.split("/", 1)[1] for w in head.weights] == expected
    np.testing.assert_array_equal(y, ref)


def test_block_dims_sets_each_refinement_width():
    head = EnhancementHead(output_channels=3, block_dims=(16, 4), hidden_dim=8)
    head(_x())
    assert [b.filters for b in head.enhance_blocks] == [16, 4]
    assert head.output_conv.kernel.shape == (3, 3, 4, 3)


def test_the_default_is_still_three_hidden_dim_blocks():
    head = EnhancementHead(output_channels=3, hidden_dim=8)
    head(_x())
    assert [b.filters for b in head.enhance_blocks] == [8, 8, 8]
    assert head.output_conv.use_bias is True
    assert len(head.weights) == 14


def test_the_transposed_conv_path_reads_the_bias_flag():
    head = EnhancementHead(output_channels=3, scale_factor=2, block_dims=(),
                           output_use_bias=False, hidden_dim=8)
    out = head(_x())['enhanced']
    assert tuple(out.shape) == (2, 16, 16, 3)
    assert [w.path.split("/", 1)[1] for w in head.weights] == ["upsample/kernel"]


@pytest.mark.parametrize("bad", [(0,), (-4,), (True,), (2.5,), (8, 0)])
@pytest.mark.parametrize("make_head", [
    lambda bad: EnhancementHead(block_dims=bad, hidden_dim=8),
    lambda bad: ClassificationHead(num_classes=4, block_dims=bad, hidden_dim=8),
], ids=["EnhancementHead", "ClassificationHead"])
def test_a_bad_width_is_refused_by_both_heads(make_head, bad):
    with pytest.raises(ValueError, match="block_dims must hold positive integers"):
        make_head(bad)


@pytest.mark.parametrize("block_dims, use_bias", [((), False), ((4,), True), (None, False)])
def test_the_config_round_trips(block_dims, use_bias):
    head = EnhancementHead(output_channels=3, block_dims=block_dims,
                           output_use_bias=use_bias, hidden_dim=8)
    clone = EnhancementHead.from_config(head.get_config())
    head(_x())
    clone(_x())
    assert clone.block_dims == head.block_dims
    assert clone.output_use_bias is use_bias
    assert [tuple(w.shape) for w in clone.weights] == [tuple(w.shape) for w in head.weights]
