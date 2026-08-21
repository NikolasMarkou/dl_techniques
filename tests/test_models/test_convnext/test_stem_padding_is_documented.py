"""F-60: the stem's padding rule now matches what the module docstring claims.

**Ruling: narrow the DOCSTRING, do not change the stem.** MEASURED at
kernel == stride == 4: `"same"` and `"valid"` agree exactly on a divisible input
(`(32,32,3)` -> 8x8 either way) and diverge only on a non-divisible one
(`(30,30,3)` -> 7x7 valid, 8x8 same). Flipping the stem to unconditional
`"same"` would silently move the spatial geometry of any checkpoint trained at a
non-divisible size -- weight-shape-compatible, activation-value different -- and
the 0x0-collapse-to-NaN failure that the DOWNSAMPLE layer's `"same"` exists for
cannot reach the stem, whose input is the image. The docstring was the thing
that was provably wrong. See decisions.md D-125.

This file pins the CURRENT behaviour so that a future "consistency" edit to the
stem has to argue with a test rather than with a paragraph.
"""

import numpy as np
import pytest

from dl_techniques.models.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2

TINY = dict(depths=[1, 1], dims=[8, 16], num_classes=4, include_top=False)


@pytest.mark.parametrize("cls", [ConvNeXtV1, ConvNeXtV2], ids=["v1", "v2"])
class TestStemPadding:
    def test_stride_gt_1_uses_valid(self, cls):
        model = cls(strides=4, **TINY)
        model.build((None, 30, 30, 3))
        assert model.stem_conv.padding == "valid"

    def test_stride_1_uses_same(self, cls):
        model = cls(strides=1, **TINY)
        model.build((None, 30, 30, 3))
        assert model.stem_conv.padding == "same"

    def test_the_two_paddings_agree_on_a_divisible_input(self, cls):
        """Why the rule is harmless where it is exercised: 32 is divisible by 4."""
        model = cls(strides=4, **TINY)
        out = model(np.zeros((1, 32, 32, 3), dtype="float32"))
        assert np.all(np.isfinite(np.asarray(out)))

    def test_the_downsample_layers_are_unconditionally_same(self, cls):
        """The half of the rule that IS unconditional, and must stay so."""
        model = cls(strides=4, **TINY)
        model.build((None, 32, 32, 3))
        downsamples = [
            sub
            for group in model.downsample_layers_list
            for sub in (group if isinstance(group, (list, tuple)) else [group])
            if getattr(sub, "padding", None) is not None
        ]
        assert downsamples, "no downsample conv found; the sweep stopped seeing them"
        assert all(layer.padding == "same" for layer in downsamples)
