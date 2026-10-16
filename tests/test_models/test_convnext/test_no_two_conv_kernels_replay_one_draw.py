"""No two kernels of a ConvNeXt block or model replay one initializer draw.

Guard for plan-2026-10-14T062616-81938e81 step 12 Part F item 15 (D-091).
``ConvNextV1Block`` and ``ConvNextV2Block`` built ONE seedless
``TruncatedNormal`` instance into ``conv_params`` and handed it to both
pointwise convs, so every block's ``expand_conv`` kernel and its
``reduce_conv`` kernel were the same random numbers (the reduce kernel the
expand draw's prefix). MEASURED at 36dbd9938 with the shared instrument
(``initializer_replay_oracle``; D-076 first saw it): r = 1.000000 for every
block, 4 of 105 pairs in a ``depths=[2, 2]`` V1 and V2 build. Each pointwise
conv now gets its own instance. The block is shared by convunext,
bfconvunext, superpoint, sam2, thera, the embedding experiments and the
bias-free denoiser application, so their fixed-seed outputs moved with it.
"""

import keras
import numpy as np
import pytest

from ..initializer_replay_oracle import replayed_pairs
from dl_techniques.layers.conv_blocks.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.conv_blocks.convnext_v2_block import ConvNextV2Block
from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2

X = np.zeros((2, 32, 32, 3), dtype="float32")


def _two_blocks(block_cls, **kwargs) -> keras.Model:
    keras.utils.set_random_seed(1234)
    inputs = keras.Input((8, 8, 16))
    x = block_cls(kernel_size=3, filters=16, name="block_a", **kwargs)(inputs)
    x = block_cls(kernel_size=3, filters=16, name="block_b", **kwargs)(x)
    return keras.Model(inputs, x)


_BLOCKS = [
    ("v1", lambda: _two_blocks(ConvNextV1Block)),
    ("v1_softorthonormal", lambda: _two_blocks(
        ConvNextV1Block, use_softorthonormal_regularizer=True)),
    ("v2", lambda: _two_blocks(ConvNextV2Block)),
]


@pytest.mark.parametrize("name,build", _BLOCKS, ids=[n for n, _ in _BLOCKS])
def test_no_two_block_kernels_replay_one_draw(name, build):
    model = build()
    replayed, compared = replayed_pairs(model)
    # Two blocks: depthwise, expand and reduce kernels each -> 15 pairs.
    assert compared >= 10, f"only {compared} kernel pairs compared"
    assert not replayed, f"{name}: {replayed}"


@pytest.mark.parametrize("cls", [ConvNeXtV1, ConvNeXtV2], ids=lambda c: c.__name__)
def test_no_two_model_kernels_replay_one_draw(cls):
    keras.utils.set_random_seed(1234)
    model = cls.from_variant("cifar10", num_classes=10, input_shape=(32, 32, 3))
    model(X)
    replayed, compared = replayed_pairs(model)
    assert compared >= 100, f"only {compared} kernel pairs compared"
    assert not replayed, (
        f"{cls.__name__}: {len(replayed)} of {compared} kernel pairs replay one "
        f"initializer draw (D-091). Pairs: {replayed[:10]}")
