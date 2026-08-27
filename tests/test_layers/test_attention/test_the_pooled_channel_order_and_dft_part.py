"""Guards for two silently-swappable choices: CBAM's pooling order, FNet's DFT part.

Why this file exists
--------------------
Two one-token swaps change what the layer computes and are caught by no test:

* ``spatial_attention.py`` concatenates ``[avg_pool, max_pool]`` along the channel
  axis. Reversing it to ``[max_pool, avg_pool]`` passes all 47 tests. It is not a
  harmless relabeling: the following convolution learns a separate weight per
  input channel, so the order decides which learned slice sees which statistic --
  and for a checkpoint trained under one order, loading it under the other feeds
  every kernel the wrong pooled input.
* ``fnet_fourier_transform.py`` takes the REAL part of its 2-D DFT. Taking the
  imaginary part instead passes all 25 tests, because no test compares the output
  against an independent DFT.

Both guards use oracles built outside the implementation: a conv kernel forced to
select one channel, and ``numpy.fft.fft2``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.fnet_fourier_transform import (
    FNetFourierTransform,
)
from dl_techniques.layers.attention.spatial_attention import SpatialAttention

pytestmark = pytest.mark.usefixtures("tf32_disabled")

BATCH, HEIGHT, WIDTH, CHANNELS = 2, 8, 8, 4


@pytest.fixture(name="feature_map")
def _feature_map():
    return (
        np.random.default_rng(0)
        .normal(size=(BATCH, HEIGHT, WIDTH, CHANNELS))
        .astype("float32")
    )


def test_the_first_pooled_channel_is_the_average_not_the_max(feature_map):
    """Pin WHICH pooled statistic occupies channel 0.

    A 1x1 convolution whose kernel is ``[[1], [0]]`` selects the first
    concatenated channel and ignores the second, so the pre-activation output is
    exactly that channel. Comparing it against an independently computed mean --
    and confirming it is NOT the max -- fixes the order without reference to how
    the concatenation is written.

    Why this can fail if the implementation is wrong: with the order reversed the
    selected channel becomes the max-pool, which differs from the mean on any
    input with channel variation.
    """
    layer = SpatialAttention(kernel_size=1, use_bias=False)
    layer(feature_map)  # build

    # Select concatenated channel 0, ignore channel 1.
    selector = np.zeros_like(np.asarray(layer.conv.get_weights()[0]))
    selector[..., 0, 0] = 1.0
    layer.conv.set_weights([selector])

    # The gate activation is monotonic, so invert it by comparing gate(x) values
    # rather than raw logits: gate(avg) vs gate(max).
    actual = np.asarray(layer(feature_map))
    expected_avg = np.asarray(
        layer.gate_activation(
            keras.ops.mean(feature_map, axis=-1, keepdims=True)
        )
    )
    expected_max = np.asarray(
        layer.gate_activation(
            keras.ops.max(feature_map, axis=-1, keepdims=True)
        )
    )

    assert not np.allclose(expected_avg, expected_max), (
        "this input cannot distinguish mean from max, so the assertion below "
        "would pass either way"
    )
    np.testing.assert_allclose(
        actual,
        expected_avg,
        atol=1e-5,
        rtol=0,
        err_msg=(
            "concatenated channel 0 is not the AVERAGE pool. The order is "
            "[avg_pool, max_pool]; reversing it feeds every learned conv weight "
            "the other statistic, which silently invalidates any checkpoint "
            "trained under the original order."
        ),
    )


def test_the_transform_returns_the_real_part_of_the_dft(feature_map):
    """Compare FNet against an independent ``numpy.fft.fft2``.

    Why this can fail if the implementation is wrong: taking the imaginary part
    instead is caught by none of the layer's 25 tests, because nothing in that
    file compares the output to a DFT computed outside the layer.
    """
    tokens = (
        np.random.default_rng(1)
        .normal(size=(BATCH, HEIGHT, CHANNELS))
        .astype("float32")
    )

    spectrum = np.fft.fft2(tokens.astype("float64"), axes=(1, 2))
    expected_real = np.real(spectrum)
    expected_imag = np.imag(spectrum)

    assert not np.allclose(expected_real, expected_imag, atol=1e-3), (
        "real and imaginary parts coincide on this input, so the assertion "
        "below could not tell them apart"
    )

    # `normalize_dft=False` is the unscaled transform, so it compares directly.
    unscaled = np.asarray(
        FNetFourierTransform(normalize_dft=False)(tokens, training=False)
    )
    np.testing.assert_allclose(
        unscaled,
        expected_real,
        atol=1e-4,
        rtol=0,
        err_msg=(
            "FNet output does not match the REAL part of a 2-D DFT computed by "
            "numpy; the most likely cause is the imaginary part being taken, or "
            "the transform being applied over the wrong axis pair"
        ),
    )


def test_the_default_normalization_is_the_orthonormal_scale(feature_map):
    """Pin `normalize_dft=True` to `1/sqrt(H * W)`.

    Measured: the default output is the unscaled transform times 0.17678, and
    `1/sqrt(8 * 4) = 0.176777`. Without this, the normalization factor is a free
    parameter no test constrains.
    """
    tokens = (
        np.random.default_rng(1)
        .normal(size=(BATCH, HEIGHT, CHANNELS))
        .astype("float32")
    )
    normalized = np.asarray(
        FNetFourierTransform(normalize_dft=True)(tokens, training=False)
    )
    unscaled = np.asarray(
        FNetFourierTransform(normalize_dft=False)(tokens, training=False)
    )
    expected = unscaled / np.sqrt(HEIGHT * CHANNELS)
    np.testing.assert_allclose(normalized, expected, atol=1e-5, rtol=0)
