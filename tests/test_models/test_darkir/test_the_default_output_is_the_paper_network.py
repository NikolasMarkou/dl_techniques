"""DarkIR's default output is the official network, tail included.

cidautai/DarkIR ``archs/DarkIR.py`` ends ``DarkIR.forward`` with
``x = self.ending(x); x = x + input``, ``ending = nn.Conv2d(width, img_channel,
3, padding=1, bias=True)``. From 2026-10-10 (3b281907a) to 2026-10-15 the
default ``include_top`` head was a 3-ConvBlock ``EnhancementHead`` appended
AFTER that global residual, so the default model was the paper's network plus
an extra one. Since plan-2026-10-14T062616-81938e81 step 6.1 (D-022) the head
IS ``ending`` (``block_dims=()``, bias kept), the model applies the global
residual after it, and ``include_top=False`` returns the last decoder stage's
``width``-channel features.

The same step found a backbone divergence (D-023): the official
``EBlock(chan, extra_depth_wise=...)`` takes its default ``dilations=[1]``
(only the ``DBlock``s receive ``dilations``), while this port gave every
encoder block the model's ``[1, 4, 9]``. That is 42,240 extra parameters at
width 32 (two depthwise branches of ``20 * c`` each, ``c`` summed over the 8
encoder blocks: 1056).

Evidence pinned here:

* the parameter count equals a hand sum of the official module list,
  transcribed below from ``archs/DarkIR.py`` and ``archs/arch_model.py``, for
  DarkIR-m (width 32) and DarkIR-l (width 64), headed, headed with the side
  output, and headless;
* with the head conv zeroed, the output IS the input (the global residual);
* the headed output is the head conv on the headless features plus the input.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.image_restoration.darkir.model import DarkIR


# ---------------------------------------------------------------------
# The official module list, transcribed (bias=True unless stated)
# ---------------------------------------------------------------------

def _eblock(c, dilations=(1,)):
    n = 9 * c + c                           # extra_conv: depthwise 3x3 on c
    n += 2 * c * c + 2 * c                  # conv1: c -> 2c
    n += len(dilations) * (9 * 2 * c + 2 * c)   # Branch(c, DW_Expand=2)
    n += c * c + c                          # sca conv: c -> c
    n += c * c + c                          # conv3: c -> c
    n += 4 * c                              # norm1, norm2 (LayerNorm2d)
    n += (2 * c * c + 2 * c) + (2 * c * c + c)  # FreMLP: c -> 2c -> c
    n += 2 * c                              # gamma, beta
    return n


def _dblock(c, dilations):
    n = 2 * c * c + 2 * c                   # conv1: c -> 2c
    n += 2 * c * 2 * 9 + 2 * c              # extra_conv: Conv2d(2c, 2c, 3, groups=c)
    n += len(dilations) * (9 * 2 * c + 2 * c)   # Branch(2c, DW_Expand=1)
    n += c * c + c                          # sca conv
    n += c * c + c                          # conv3
    n += 2 * c * c + 2 * c                  # conv4: c -> 2c
    n += c * c + c                          # conv5: c -> c
    n += 4 * c + 2 * c                      # norms, gamma, beta
    return n


def _official_darkir_params(width, side_out, ending, img=3, middle_enc=2,
                            middle_dec=2, enc=(1, 2, 3), dec=(3, 1, 1),
                            dilations=(1, 4, 9)):
    n = 9 * img * width + width             # intro
    c = width
    for blocks in enc:
        n += blocks * _eblock(c)            # EBlock(chan): dilations=[1]
        n += 4 * c * 2 * c + 2 * c          # downs: Conv2d(c, 2c, 2, 2)
        c *= 2
    n += middle_enc * _eblock(c)
    n += middle_dec * _dblock(c, dilations)
    if side_out:
        n += 9 * c * img + img              # side_out
    for blocks in dec:
        n += c * 2 * c                      # ups: Conv2d(c, 2c, 1, bias=False)
        c //= 2
        n += blocks * _dblock(c, dilations)
    if ending:
        n += 9 * width * img + img          # ending
    return n


def _count(model):
    return sum(int(np.prod(w.shape)) for w in model.weights)


@pytest.mark.parametrize("width, include_top, use_side_loss, expected", [
    (32, True, False, 3_314_723),
    (32, True, True, 3_321_638),
    (32, False, False, 3_313_856),
    (64, True, False, 12_943_427),
])
def test_the_parameter_count_is_the_official_hand_sum(
        width, include_top, use_side_loss, expected):
    hand_sum = _official_darkir_params(width, side_out=use_side_loss,
                                       ending=include_top)
    assert hand_sum == expected
    model = DarkIR(width=width, include_top=include_top, use_side_loss=use_side_loss)
    model.build((None, 64, 64, 3))
    assert _count(model) == hand_sum


_TINY = dict(width=8, middle_blk_num_enc=1, middle_blk_num_dec=1,
             enc_blk_nums=[1, 1], dec_blk_nums=[1, 1], dilations=[1, 2])


def _x():
    return np.random.default_rng(2).random((1, 32, 32, 3)).astype("float32")


def test_with_the_head_conv_zeroed_the_output_is_the_input(golden_reference_device):
    with keras.device(golden_reference_device):
        model = DarkIR(**_TINY)
        x = _x()
        model(x)
        for w in model.enhancement_head.weights:
            w.assign(np.zeros(w.shape, dtype=w.dtype))
        y = keras.ops.convert_to_numpy(model(x, training=False))
    np.testing.assert_array_equal(y, x)


def test_the_head_conv_and_the_residual_follow_the_headless_features(golden_reference_device):
    x = _x()
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(0)
        headed = DarkIR(**_TINY)
        headless = DarkIR(include_top=False, **_TINY)
        headed(x)
        headless(x)
        # Positional, not by path: the blocks' LayerNorms are auto-named
        # (``layer_normalization_<n>``, pinned in the roundtrip family's
        # AUTONAME_STEMS), so two instances disagree by path. The head is the
        # headed model's last two weights; everything before pairs 1:1.
        n_head = len(headed.weights) - len(headless.weights)
        assert n_head == 2
        for src, dst in zip(headed.weights[:-n_head], headless.weights):
            assert tuple(src.shape) == tuple(dst.shape)
            dst.assign(src)
        y = keras.ops.convert_to_numpy(headed(x, training=False))
        f = headless(x, training=False)
        conv = headed.enhancement_head.output_conv
        ref = keras.ops.convert_to_numpy(
            keras.ops.conv(f, conv.kernel, padding="same") + conv.bias + x)
    assert tuple(f.shape) == (1, 32, 32, _TINY["width"])
    assert [w.path.split("/", 1)[1] for w in headed.enhancement_head.weights] == [
        "enhancement_head/output_conv/kernel", "enhancement_head/output_conv/bias"]
    np.testing.assert_array_equal(y, ref)


def test_the_encoder_blocks_run_one_undilated_branch():
    model = DarkIR(**_TINY)
    encoders = [b for stage in model.encoder_stages for b in stage] + list(
        model.middle_encoder_blocks)
    decoders = [b for stage in model.decoder_stages for b in stage] + list(
        model.middle_decoder_blocks)
    assert encoders and all(list(b.dilations) == [1] for b in encoders)
    assert decoders and all(list(b.dilations) == _TINY["dilations"] for b in decoders)


def test_a_side_output_needs_the_head():
    with pytest.raises(ValueError, match="include_top"):
        DarkIR(include_top=False, use_side_loss=True, **_TINY)
