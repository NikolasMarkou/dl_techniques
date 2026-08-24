"""`basic` blocks + a non-64 stage-0 width + deep supervision: it WORKS.

Rationale
---------
``tests/test_models/test_resnet/test_deep_supervision.py`` pinned
``filters_per_stage[0] == 64`` and recorded the reason in a comment: a narrower
stage 0 under ``block_type="basic"`` "is a pre-existing ResNet defect ... it is
dodged here, not fixed". Nothing in the suite pinned that combination in EITHER
direction, so "dodged" was a hypothesis about the current tree, not a reading of
it.

MEASURED at HEAD, before this file existed -- ``block_type='basic'``,
``blocks_per_stage=[1,1,1,1]``, ``filters_per_stage=[w, 2w, 4w, 8w]``,
``input_shape=(32, 32, 3)``, a forward pass on a real ``(2, 32, 32, 3)`` batch,
across both ``stem_type`` values (the knob landed one commit earlier, so the
question is two-dimensional now):

    stem_type   deep_supervision   w=8   w=16   w=32   w=64   w=128
    imagenet    True                OK     OK     OK     OK      OK
    imagenet    False               OK     OK     OK     OK      OK
    cifar       True                OK     OK     OK     OK      OK
    cifar       False               OK     OK     OK     OK      OK

All 20 cells run. With ``enable_deep_supervision=True`` every cell returns a
list of FOUR ``(2, 3)`` tensors; with it False, a single ``(2, 3)`` tensor.

So the pin is the WORKING direction, and the comment quoted above is stale: the
defect it describes was closed by ``D-041`` (plan-2026-08-19T163559-499b6f0e),
which made the stem width follow ``filters_per_stage[0]`` instead of a literal
64 -- see ``test_basic_blocks_work_at_any_stage0_width.py``, which pins the
``enable_deep_supervision=False`` base case of the same grid. Deep supervision
adds auxiliary heads over the per-stage features; it never touches the stage-0
shortcut that was the actual defect, which is why fixing the base case fixed
this case too. That reasoning is *why* it holds; the table above is the
evidence, and this file is what keeps it true.

RED proofs (a "it works" pin is only meaningful if it can fail)
--------------------------------------------------------------
One injection per assertion, with the ACTUAL observed text.

**Injection 1 -- restore the pre-D-041 stem**: ``filters=64`` in
``_build_stem`` instead of ``filters=self.filters_per_stage[0]``. Result:
**24 failed, 6 passed** of 30. The 6 survivors are exactly the ``w=64`` cells
(3 tests x 2 stems), where the requested width coincides with the literal. Two
distinct failure texts:

    ValueError: Inputs have incompatible shapes. Received shapes (8, 8, 8) and
    (8, 8, 64)

    AssertionError: stage-0 width 8 requested, stem emits 64 -- the pre-D-041
    literal is back and the identity shortcut cannot hold
    assert 64 == 8

**Injection 2 -- ``x = keras.ops.stop_gradient(x)`` after ``stem_act`` in
``call``**: forward shapes and the stem width are untouched, only the backward
graph moves. Result: **10 failed, 20 passed** -- exactly the ten
``test_gradients_reach_every_trainable_weight`` cells, i.e. the gradient arm is
not carried by the two arms above:

    AssertionError: gradient flow is incomplete in ResNet: 41/44 trainable
    weights receive a live gradient.
    3 weight(s) received NO gradient (not on the backward graph -- built,
    saved, and never executed):
      res_net_20/stem_conv/kernel
      res_net_20/stem_bn/gamma
      res_net_20/stem_bn/beta
"""

import numpy as np
import pytest

from dl_techniques.models.resnet.model import ResNet
from tests.test_models.gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
)

# ---------------------------------------------------------------------

INPUT_SHAPE = (32, 32, 3)
STAGE0_WIDTHS = [8, 16, 32, 64, 128]
STEM_TYPES = ["imagenet", "cifar"]
NUM_CLASSES = 3
NUM_STAGES = 4


@pytest.fixture(scope="module")
def images():
    return np.random.RandomState(0).randn(2, *INPUT_SHAPE).astype("float32")


def _model(stage0: int, stem_type: str) -> ResNet:
    return ResNet(
        num_classes=NUM_CLASSES,
        blocks_per_stage=[1, 1, 1, 1],
        filters_per_stage=[stage0, stage0 * 2, stage0 * 4, stage0 * 8],
        block_type="basic",
        enable_deep_supervision=True,
        input_shape=INPUT_SHAPE,
        stem_type=stem_type,
    )


@pytest.mark.parametrize("stem_type", STEM_TYPES)
@pytest.mark.parametrize("stage0", STAGE0_WIDTHS)
def test_the_forward_pass_returns_one_output_per_stage(stage0, stem_type, images):
    """The 2x5 grid, deep supervision ON. All ten cells measured OK."""
    outputs = _model(stage0, stem_type)(images, training=False)

    assert isinstance(outputs, list), (
        f"enable_deep_supervision=True must return a list, got {type(outputs)}"
    )
    assert len(outputs) == NUM_STAGES, (
        f"expected one output per stage ({NUM_STAGES}), got {len(outputs)}"
    )
    for index, output in enumerate(outputs):
        array = np.array(output)
        assert array.shape == (2, NUM_CLASSES), (
            f"output {index} has shape {array.shape}, expected (2, {NUM_CLASSES})"
        )
        assert np.all(np.isfinite(array)), f"output {index} is not finite"


@pytest.mark.parametrize("stem_type", STEM_TYPES)
@pytest.mark.parametrize("stage0", STAGE0_WIDTHS)
def test_the_stem_still_honours_the_requested_stage0_width(stage0, stem_type):
    """The isolating assertion -- this is the property D-041 established.

    It is restated here rather than delegated to
    ``test_basic_blocks_work_at_any_stage0_width.py`` because that file never
    constructs a deep-supervised model, so nothing there would notice a
    regression that fired only when the supervision heads exist.
    """
    model = _model(stage0, stem_type)
    assert model.stem_conv.filters == stage0, (
        f"stage-0 width {stage0} requested, stem emits "
        f"{model.stem_conv.filters} -- the pre-D-041 literal is back and the "
        "identity shortcut cannot hold"
    )


@pytest.mark.parametrize("stem_type", STEM_TYPES)
@pytest.mark.parametrize("stage0", STAGE0_WIDTHS)
def test_gradients_reach_every_trainable_weight(stage0, stem_type, images):
    """Running is not training: every weight must be on the backward graph.

    The blocks are subclassed layers built lazily inside ``call``, so
    ``trainable_weights`` is EMPTY on a fresh instance -- the oracle raises on
    an empty weight set rather than passing vacuously, so the forward pass
    below is load-bearing, not warm-up.
    """
    model = _model(stage0, stem_type)
    model(images, training=False)

    report = assert_gradients_reach_every_trainable_weight(model, images)
    assert len(report) > 0
