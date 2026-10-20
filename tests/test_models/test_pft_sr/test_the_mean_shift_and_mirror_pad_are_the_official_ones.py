"""The RGB mean shift is applied and removed by value, and the input is padded
to a multiple of the window by the official mirror and cropped back.

Official ``PFT.forward`` (:1102-1147):

- ``x = cat([x, flip(x, H)], H)[:, :, :h]`` and the same along W, ``h`` the
  next multiple of ``window_size``: symmetric padding, valid only while the pad
  is not longer than the image (otherwise the concatenation is too short and
  the official code fails).
- ``x = (x - mean) * img_range``, ``mean = (0.4488, 0.4371, 0.4040)`` when
  ``in_chans == 3`` and 0 otherwise; at the end ``x / img_range + mean``.
- the output is cropped to ``h_ori * upscale, w_ori * upscale``.

Mean shift: zero every weight except a ``conv_first`` that copies the three
input channels and ``conv_after_body``, which stays zero. Then the headless
output is ``conv_first`` of the shifted input, i.e. ``(x - mean) * img_range``
in channels 0-2 exactly, and a headed model whose tail is zero returns exactly
``mean``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.super_resolution.pft_sr import PFTSR
from dl_techniques.models.vision.super_resolution.pft_sr.blocks import mirror_pad

MEAN = np.array([0.4488, 0.4371, 0.4040], dtype="float32")


def _zeroed(img_range=1.0, include_top=True, in_channels=3):
    model = PFTSR(scale=2, in_channels=in_channels, embed_dim=8, num_blocks=[1],
                  num_heads=2, window_size=4, img_range=img_range,
                  include_top=include_top)
    model.build((None, None, None, in_channels))
    for v in model.weights:
        v.assign(np.zeros(v.shape, dtype="float32"))
    kernel = np.zeros(model.conv_first.kernel.shape, dtype="float32")
    for c in range(in_channels):
        kernel[1, 1, c, c] = 1.0
    model.conv_first.kernel.assign(kernel)
    return model


@pytest.mark.parametrize("img_range", [1.0, 255.0])
def test_the_mean_is_subtracted_and_scaled_at_the_input(img_range):
    x = np.random.default_rng(0).random((2, 8, 12, 3)).astype("float32")
    y = np.asarray(_zeroed(img_range, include_top=False)(x))
    np.testing.assert_allclose(y[..., :3], (x - MEAN) * img_range, rtol=1e-6, atol=1e-6)
    assert np.all(y[..., 3:] == 0.0)


@pytest.mark.parametrize("img_range", [1.0, 255.0])
def test_the_mean_is_added_back_at_the_output(img_range):
    x = np.random.default_rng(1).random((1, 8, 12, 3)).astype("float32")
    y = np.asarray(_zeroed(img_range)(x))
    assert y.shape == (1, 16, 24, 3)
    np.testing.assert_array_equal(y, np.broadcast_to(MEAN, y.shape))


def test_a_one_channel_model_shifts_nothing():
    x = np.random.default_rng(2).random((1, 8, 12, 1)).astype("float32")
    np.testing.assert_array_equal(
        np.asarray(_zeroed(include_top=False, in_channels=1)(x))[..., :1], x)
    assert np.all(np.asarray(_zeroed(in_channels=1)(x)) == 0.0)


@pytest.mark.parametrize("h, w", [(7, 10), (5, 8), (8, 12), (2, 4), (9, 9)])
def test_the_pad_is_the_official_mirror(h, w):
    x = np.random.default_rng(3).random((2, h, w, 3)).astype("float32")
    hp, wp = -(-h // 4) * 4, -(-w // 4) * 4
    official = np.concatenate([x, np.flip(x, 1)], 1)[:, :hp]
    official = np.concatenate([official, np.flip(official, 2)], 2)[:, :, :wp]
    got = np.asarray(mirror_pad(x, 4))
    np.testing.assert_array_equal(got, official)
    np.testing.assert_array_equal(
        got, np.pad(x, ((0, 0), (0, hp - h), (0, wp - w), (0, 0)), mode="symmetric"))


@pytest.mark.parametrize("h, w", [(7, 10), (5, 6)])
def test_the_output_is_cropped_to_the_input_size(h, w):
    model = PFTSR(scale=3, embed_dim=8, num_blocks=[1], num_heads=2, window_size=4,
                  upsampler="pixelshuffledirect")
    x = np.random.default_rng(4).random((1, h, w, 3)).astype("float32")
    assert tuple(model(x).shape) == (1, 3 * h, 3 * w, 3)
    headless = PFTSR(scale=3, embed_dim=8, num_blocks=[1], num_heads=2, window_size=4,
                     include_top=False)
    assert tuple(headless(x).shape) == (1, h, w, 8)


def test_a_pad_longer_than_the_image_is_refused():
    """``h_ori = 3``, window 8: the official ``cat([x, flip(x)])`` has 6 rows,
    fewer than the 8 it slices, and fails; the port raises and says why."""
    model = PFTSR(scale=2, embed_dim=8, num_blocks=[1], num_heads=2, window_size=8)
    with pytest.raises(ValueError, match="mirror"):
        model(np.zeros((1, 3, 16, 3), "float32"))
