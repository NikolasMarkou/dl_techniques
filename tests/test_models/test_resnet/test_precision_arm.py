"""
R-088 / R-141 mixed-precision arm for ``models/vision/resnet`` -- the DEEP-SUPERVISED
regime, which the shared family does not reach.

Why this file exists at all, re-derived rather than carried
-----------------------------------------------------------
The plan's F-13 recorded "zero adoption of ``precision_arm_oracle`` in either
package". That premise is FALSE at the family level and true only at the
package level: ``resnet`` IS a registered subject in
``precision_arm_subjects.py`` (``_sub("resnet", ...)``, line 115) and its arm
runs green from ``test_precision_arm_family.py``. Duplicating that subject here
would be a second copy of an assertion the repo deliberately centralises.

What the family subject does NOT cover is the configuration this package's
production trainer actually runs and this plan is about to change:

======================  ================================  ======================
                        family subject                    here
======================  ================================  ======================
``block_type``          ``basic``                         ``bottleneck``
stages                  2                                 4
outputs                 1                                 4 (deep supervision)
======================  ================================  ======================

A single-output arm cannot see a dtype defect confined to the deep-supervision
heads, and ``src/train/resnet/train_resnet.py`` -- repaired at step 1 of this
plan -- is a MULTI-OUTPUT trainer. That is the gap.

MEASURED, GPU 1 (RTX 4070), TF32 on by default -- the regime this gate runs in
------------------------------------------------------------------------------
``ResNet(num_classes=4, blocks_per_stage=[1,1,1,1], filters_per_stage=[8,16,32,64],
block_type='bottleneck', enable_deep_supervision=True, input_shape=(32,32,3))``:

RE-MEASURED 2026-10-14 (plan-2026-10-14T062616-81938e81 step 2), batch 2,
after the heads became the shared ``ClassificationHead`` emitting logits:

* ``mixed_float16``: 4 output tensors, dtypes
  ``['float16','float16','float16','float16']``, ``absmax``
  ``[9.023438, 2.457031, 1.481445, 0.902344]``, all finite;
* ``float32`` control: same arity, ``absmax``
  ``[9.016292, 2.457358, 1.483411, 0.902183]``;
* the largest fp16-vs-float32 relative disagreement across the four heads is
  **1.3e-3** (head 2), so ``rtol_against_float32=1e-2`` carries a 7.5x margin;
  the bound was chosen from the first measurement (1.0e-3) and not widened;
* backward under ``mixed_float16``: 0 ``None`` gradients, 0 non-finite,
  ``grad_norm_sum`` 32740.2 (float32 37021.7).

Batch 2, not 1: each head's two ``DenseBlock`` layers carry a BatchNorm, and
the backward part runs in training mode, where a batch of one normalizes every
feature to exactly 0. At batch 1 every head output is its constant bias and
the loss and every gradient are exactly 0 (measured: loss 0.0,
``grad_norm_sum`` 0 under both policies). That passed while the heads ended
in a softmax only because a uniform softmax still has a ramp-loss gradient.

GREEN on every one of the four parts. Nothing here is xfailed and nothing was
relaxed after the numbers were seen.

Anti-vacuity
------------
``assert_precision_arm`` carries its own policy controls (the model must be
BUILT under ``mixed_float16``, not merely called under it). This file adds the
one control the oracle cannot: that the subject really is multi-output, so a
future edit that quietly drops ``enable_deep_supervision`` turns this file into
a duplicate of the family arm instead of silently still passing.
"""

from typing import Any, Dict

import numpy as np

from dl_techniques.models.vision.resnet import ResNet

from ..precision_arm_oracle import assert_precision_arm, run_forward

#: Deep supervision on ResNet emits one head per stage.
N_STAGES = 4

#: Measured on GPU 1; see the module docstring. The bound is 10x the observed
#: worst-case relative disagreement, and it is NOT to be widened -- a wider
#: tolerance than the signal asserts nothing.
RTOL_AGAINST_FLOAT32 = 1e-2


def _build() -> ResNet:
    """The deep-supervised bottleneck ResNet this arm judges.

    Kept tiny on purpose: the arm measures dtypes and finiteness, which are
    width-independent, and a real variant would put four fp16/float32/backward
    passes into a gate that already runs on a shared GPU.
    """
    return ResNet(
        num_classes=4,
        blocks_per_stage=[1, 1, 1, 1],
        filters_per_stage=[8, 16, 32, 64],
        block_type="bottleneck",
        enable_deep_supervision=True,
        input_shape=(32, 32, 3),
    )


def _images() -> np.ndarray:
    # Batch 2: see the module docstring (a batch of one zeroes every
    # BatchNorm-headed output in training mode).
    return np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")


def test_the_subject_is_actually_multi_output() -> None:
    """Anti-vacuity: without this, the arm below could silently become the family's.

    ``enable_deep_supervision`` is the ONLY reason this file is not a duplicate
    of ``test_precision_arm_family.py``'s ``resnet`` case. If it stops producing
    one head per stage, this fails here rather than passing as a redundant
    single-output arm.
    """
    report = run_forward(_build, _images, "float32", training=False)
    assert report["n_tensors"] == N_STAGES, (
        f"the deep-supervised subject returned {report['n_tensors']} tensors, "
        f"expected one head per stage ({N_STAGES}); this arm is no longer "
        f"measuring the multi-output path it exists for"
    )


def test_the_deep_supervised_resnet_runs_under_mixed_float16() -> None:
    """All four parts of the arm on the multi-output bottleneck configuration.

    MEASURED (GPU 1): fp16 ``absmax`` ``[17.515625, 18.96875, 6.109375,
    4.2734375]`` against float32 ``[17.498203, 18.970032, 6.109206, 4.271974]``;
    backward 0 ``None`` / 0 non-finite, ``grad_norm_sum`` 217.27.
    """
    reports: Dict[str, Any] = assert_precision_arm(
        build=_build,
        make_inputs=_images,
        rtol_against_float32=RTOL_AGAINST_FLOAT32,
    )

    fp16 = reports["mixed_float16"]
    assert fp16["model_policy"] == "mixed_float16"
    assert reports["float32"]["model_policy"] == "float32"
    assert fp16["n_tensors"] == N_STAGES, (
        f"the fp16 arm judged {fp16['n_tensors']} tensors, not {N_STAGES}"
    )
    assert fp16["dtypes"] == ["float16"] * N_STAGES, (
        f"one of the deep-supervision heads did not reach the compute dtype: "
        f"{fp16['dtypes']}. A float32 auxiliary head opts the whole "
        f"multi-output trainer out of mixed precision."
    )

    backward = reports["backward_mixed_float16"]
    assert backward["n_none"] == 0, (
        f"{backward['n_none']} gradient(s) came back None under mixed_float16; "
        f"a deep-supervision head with no gradient is a head that does not train"
    )
    assert backward["n_nonfinite"] == 0
    assert backward["grad_norm_sum"] > 0.0, (
        "the backward pass produced an all-zero gradient sum, so parts 4's "
        "finiteness assertion would be vacuous"
    )
