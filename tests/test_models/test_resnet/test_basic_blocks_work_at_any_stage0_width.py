"""`ResNet(block_type='basic')` must run for any `filters_per_stage[0]`.

Rationale
---------
The stem convolution was hard-coded to 64 channels, while a `basic` block's
stage-0 first block is given ``use_projection=False`` unconditionally --
correctly, because stage 0 does not stride, so its shortcut is an identity and
requires the stem to emit exactly ``filters_per_stage[0]`` channels. MEASURED at
HEAD:

    basic       f0=8    RAISED ValueError    bottleneck  f0=8    OK
    basic       f0=16   RAISED ValueError    bottleneck  f0=16   OK
    basic       f0=32   RAISED ValueError    bottleneck  f0=32   OK
    basic       f0=64   OK                   bottleneck  f0=64   OK
    basic       f0=128  RAISED ValueError    bottleneck  f0=128  OK
    from_variant resnet18 OK

Every shipped variant uses 64, which is why `from_variant` and the whole suite
were green. The error surfaced two frames deep in `layers/standard_blocks.py`
and named neither `filters_per_stage` nor 64.

See decisions.md D-041 (plan-2026-08-19T163559-499b6f0e).
"""

import hashlib

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.resnet.model import ResNet

# ---------------------------------------------------------------------

INPUT_SHAPE = (32, 32, 3)
STAGE0_WIDTHS = [8, 16, 32, 64, 128]


@pytest.fixture(scope="module")
def images():
    return np.random.RandomState(0).randn(2, *INPUT_SHAPE).astype("float32")


def _model(block_type: str, stage0: int) -> ResNet:
    return ResNet(
        num_classes=3,
        blocks_per_stage=[1, 1, 1, 1],
        filters_per_stage=[stage0, stage0 * 2, stage0 * 4, stage0 * 8],
        block_type=block_type,
        input_shape=INPUT_SHAPE,
    )


@pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
@pytest.mark.parametrize("stage0", STAGE0_WIDTHS)
def test_the_forward_pass_runs(block_type, stage0, images):
    """The 2x5 grid; four of the five `basic` cells raised at HEAD."""
    output = np.array(_model(block_type, stage0)(images, training=False))
    assert output.shape == (2, 3)
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
@pytest.mark.parametrize("stage0", STAGE0_WIDTHS)
def test_the_stem_emits_the_requested_stage0_width(block_type, stage0):
    """The isolating assertion: the stem must honour the request, not 64.

    This is what makes the identity shortcut in stage 0 valid, and it is the
    property a "fix" that merely forced `use_projection=True` would NOT have.
    """
    model = _model(block_type, stage0)
    assert model.stem.conv1.filters == stage0, (
        f"stem emits {model.stem.conv1.filters} channels for "
        f"filters_per_stage[0]={stage0}"
    )


class TestShippedVariantsAreUnaffected:
    """The change must be invisible to everything that already worked.

    NOTE ON THE INSTRUMENT: these digests are taken on CPU deliberately. The
    same comparison on GPU 1 reported `resnet50` DIFFERING -- and re-running the
    IDENTICAL code twice on GPU gave two more different digests
    (721f022d4acb8b07, da6a233ebcc4b9f9), i.e. the GPU forward is not
    reproducible run-to-run at this size, so it cannot answer an inertness
    question at all.
    """

    @pytest.mark.parametrize("variant", ["resnet18", "resnet34", "resnet50"])
    def test_every_shipped_variant_uses_a_64_channel_stem(self, variant):
        model = ResNet.from_variant(variant, num_classes=3,
                                    input_shape=INPUT_SHAPE)
        assert model.filters_per_stage[0] == 64
        assert model.stem.conv1.filters == 64, (
            "a shipped variant's stem width has moved; every existing "
            "checkpoint's stem kernel shape depends on this"
        )

    # count_params() at num_classes=3 with the paper's GAP + FC head
    # (plan-2026-10-14T062616-81938e81 step 3, D-013): the 1000-class figures
    # of README section 17 (11,699,112 / 21,814,696 / 25,610,152) minus the
    # 1000-wide fc (513,000 basic, 2,049,000 bottleneck) plus the 3-wide one
    # (1,539 / 6,147). The two-DenseBlock head of 2026-10-09..14 read
    # 11,352,259 / 21,467,843 / 24,120,515.
    @pytest.mark.parametrize("variant,params", [
        ("resnet18", 11_187_651), ("resnet34", 21_303_235), ("resnet50", 23_567_299),
    ])
    def test_the_parameter_count_is_unchanged(self, variant, params, images):
        model = ResNet.from_variant(variant, num_classes=3,
                                    input_shape=INPUT_SHAPE)
        model(images, training=False)
        assert model.count_params() == params
