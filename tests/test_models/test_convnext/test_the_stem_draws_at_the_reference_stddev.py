"""The ConvNeXt stem and downsample convs draw at the reference stddev (D-076).

The official ``_init_weights`` (facebookresearch/ConvNeXt and ConvNeXt-V2)
draws every ``nn.Conv2d`` from ``trunc_normal_(std=.02)``. Before D-076
``STEM_INITIALIZER`` was the string ``"truncated_normal"``, whose Keras default
stddev is 0.05. A ``TruncatedNormal`` cut at 2 stddev has a sample standard
deviation of 0.8796 x stddev: 0.0176 at the reference, 0.044 at the old value.
The guard reads the DRAWN kernels, not only the config.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2

#: Sample std of a TruncatedNormal(0, s) cut at +-2 s, divided by s.
_TRUNCATED_STD_RATIO = 0.8796
_REFERENCE_STDDEV = 0.02


def _convs(model):
    yield "stem_conv", model.stem.conv1
    for downsampler in model.downsamplers:
        yield downsampler["conv"].name, downsampler["conv"]


@pytest.mark.parametrize("cls", [ConvNeXtV1, ConvNeXtV2], ids=["v1", "v2"])
def test_every_stem_and_downsample_kernel_draws_at_std_002(cls):
    keras.utils.set_random_seed(1234)
    model = cls(
        depths=[1, 1, 1], dims=[64, 96, 128], strides=2, num_classes=4,
        include_top=False,
    )
    model.build((None, 32, 32, 3))
    expected = _TRUNCATED_STD_RATIO * _REFERENCE_STDDEV
    seen = 0
    for name, conv in _convs(model):
        kernel = np.asarray(conv.kernel)
        assert kernel.size >= 768, (name, kernel.shape)
        std = float(kernel.std())
        # 768+ values: the sample std's relative error is under 3%; 15% is
        # loose for the reference and far from the old 0.044.
        assert abs(std - expected) < 0.15 * expected, (
            f"{cls.__name__}.{name} kernel std {std:.5f}, expected "
            f"{expected:.5f} (TruncatedNormal(stddev=0.02)); Keras' bare "
            "'truncated_normal' draws at stddev 0.05 (0.044 sample std)"
        )
        config = conv.kernel_initializer.get_config()
        assert config["stddev"] == _REFERENCE_STDDEV, (name, config)
        seen += 1
    assert seen == 3
