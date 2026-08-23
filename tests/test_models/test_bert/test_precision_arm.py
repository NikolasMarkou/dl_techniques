"""
R-088 / R-141 mixed-precision arm for ``models/bert`` -- and the defect it found.

Why this file exists at all, re-derived rather than carried
-----------------------------------------------------------
The plan's F-13 recorded "zero adoption of ``precision_arm_oracle`` in either
package". That premise is FALSE at the family level and true only at the
package level: ``bert`` IS a registered subject in
``precision_arm_subjects.py`` (``_sub("bert", ...)``, line 314) and its arm --
the bare ``create_bert`` ENCODER -- runs green from
``test_precision_arm_family.py``. Re-registering that subject here would be a
second copy of an assertion the repo deliberately centralises.

What the family subject does not reach is ``create_bert_with_head`` -- the
package's flagship factory, the one every README example uses and the one this
plan's step 4 is about to change. Pointing the arm at it found a real defect.

THE FINDING (MEASURED, GPU 1 / RTX 4070, TF32 on by default)
-------------------------------------------------------------
``create_bert_with_head("tiny", NLPTaskConfig(SENTIMENT_ANALYSIS, num_classes=3))``
under ``mixed_float16`` RAISES, verbatim::

    TypeError: Exception encountered when calling GELU.call().

    `x` and `y` must have the same dtype, got tf.float16 != tf.float32.

    Arguments received by GELU.call():
      • inputs=tf.Tensor(shape=(2, 32), dtype=float16)

Its float32 control is green, and the bare encoder at the SAME config is green
under ``mixed_float16`` too (``test_the_encoder_is_green_which_localises_the_defect``),
so the fault is in the head path and not in ``models/bert``'s own encoder.

Root cause, isolated at the layer::

    # layers/activations/expanded_activations.py:174
    return 0.5 * inputs * (1 + keras.ops.erf(inputs / keras.ops.sqrt(2.0)))

``keras.ops.sqrt(2.0)`` returns **float32 regardless of the active dtype
policy** (MEASURED below), so it meets a float16 autocast tensor and TensorFlow
refuses the divide. This is the same shape as the ``qwen`` MoE defect
(``ops.one_hot`` ignoring the policy, decisions.md D-064 of
``plan-2026-08-19T163559-499b6f0e``), and it hits exactly 2 of the 6 concrete
activations in that module -- ``GELU`` and ``xGELU``, both of which divide by
``ops.sqrt(2.0)``. ``SiLU``, ``xSiLU``, ``xATLU`` and ``EluPlusOne`` are all
green at float16.

DISPOSITION -- pre-registered in decisions.md D-011, BEFORE any number was seen
-------------------------------------------------------------------------------
The cause is ``src/dl_techniques/layers/activations/expanded_activations.py``,
which is OUTSIDE ``models/bert/`` and ``models/resnet/`` and is reached by every
consumer of the activation factory's ``'gelu'`` / ``'xgelu'`` keys, not just
BERT. The pre-registered rule for a RED whose cause is outside these two
packages is: do NOT fix it here, pin it ``xfail(strict=True)`` with the measured
cause, record the measurement in ``decisions.md``, and report it as a scope
question. That is what this file does. The bound was not relaxed, the oracle
call was not deleted, and the arm is ``strict=True`` so the pin FAILS the moment
the shared layer is repaired -- which is what stops it becoming cargo.

See ``decisions.md`` D-016.
"""

from typing import Any, Dict

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.activations.expanded_activations import GELU, xGELU
from dl_techniques.models.bert import create_bert, create_bert_with_head
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

from ..precision_arm_oracle import (
    assert_precision_arm,
    precision_policy,
    run_forward,
)

#: The same encoder geometry on both sides of the localisation pair, so the
#: only difference between the green control and the red arm is the head.
ENCODER_KWARGS: Dict[str, Any] = {
    "vocab_size": 64,
    "max_position_embeddings": 32,
    "hidden_size": 32,
    "num_layers": 1,
    "num_heads": 2,
    "intermediate_size": 64,
}


def _inputs() -> Dict[str, np.ndarray]:
    """The three inputs ``create_bert_with_head``'s Functional wrapper requires."""
    rng = np.random.RandomState(0)
    return {
        "input_ids": rng.randint(0, ENCODER_KWARGS["vocab_size"], (2, 16)).astype("int32"),
        "attention_mask": np.ones((2, 16), dtype="int32"),
        "token_type_ids": np.zeros((2, 16), dtype="int32"),
    }


def _build_encoder() -> keras.Model:
    return create_bert("tiny", **ENCODER_KWARGS)


def _build_with_head() -> keras.Model:
    return create_bert_with_head(
        "tiny",
        NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3,
        ),
        bert_config_overrides=dict(ENCODER_KWARGS),
    )


# ---------------------------------------------------------------------
# the framework behaviour that made the defect -- pinned at the layer
# ---------------------------------------------------------------------


@pytest.mark.parametrize("activation_class", [GELU, xGELU])
def test_the_gelu_layers_divide_by_a_float32_constant(activation_class) -> None:
    """The RED half: the mechanism, pinned where it lives.

    If ``keras.ops.sqrt`` ever follows the compute dtype, this fails and the
    ``xfail`` below becomes removable -- which is the point of pinning the
    mechanism separately from the symptom.

    MEASURED (GPU 1): ``ops.sqrt(2.0).dtype`` is ``float32`` inside a
    ``mixed_float16`` policy; both ``GELU`` and ``xGELU`` raise ``TypeError``;
    casting the constant to the input's dtype returns a float16 tensor.
    """
    with precision_policy("mixed_float16"):
        root_two = ops.sqrt(2.0)
        assert "float32" in str(root_two.dtype), (
            f"ops.sqrt(2.0) now returns {root_two.dtype} under mixed_float16; "
            f"the float32-constant mechanism this file pins is gone"
        )

        half = ops.cast(
            ops.convert_to_tensor(np.random.RandomState(0).randn(2, 4).astype("float32")),
            "float16",
        )
        with pytest.raises(TypeError, match="same dtype"):
            _ = activation_class()(half)

        repaired = half / ops.cast(root_two, half.dtype)
        assert "float16" in str(repaired.dtype)


def test_the_other_expanded_activations_are_green_at_float16() -> None:
    """Anti-vacuity twin: the module is not uniformly broken, so the pin is specific.

    MEASURED (GPU 1): of the six concrete activations in
    ``expanded_activations.py``, exactly ``GELU`` and ``xGELU`` raise; ``SiLU``,
    ``xSiLU``, ``xATLU`` and ``EluPlusOne`` all return float16. Without this
    arm, "GELU raises under fp16" would be indistinguishable from "nothing in
    this module runs under fp16", and the diagnosis above would be unearned.
    """
    from dl_techniques.layers.activations.expanded_activations import (
        EluPlusOne, SiLU, xATLU, xSiLU,
    )

    with precision_policy("mixed_float16"):
        x = ops.convert_to_tensor(np.random.RandomState(0).randn(2, 4).astype("float32"))
        for activation_class in (SiLU, xSiLU, xATLU, EluPlusOne):
            out = activation_class()(x)
            assert "float16" in str(out.dtype), (
                f"{activation_class.__name__} returned {out.dtype} under "
                f"mixed_float16; the float32-constant defect is wider than the "
                f"two activations this file names"
            )


# ---------------------------------------------------------------------
# localisation: the encoder is green, so the head is the fault
# ---------------------------------------------------------------------


def test_the_encoder_is_green_which_localises_the_defect() -> None:
    """The bare encoder, at the head factory's exact geometry, under fp16.

    Deliberately a single ``run_forward`` and NOT a second
    ``assert_precision_arm``: the full four-part arm on ``create_bert`` already
    runs in ``test_precision_arm_family.py`` and duplicating it here would be a
    copy, not a measurement. What this arm adds is the CONTRAST -- same encoder
    config, no head, no raise -- which is the evidence that
    ``models/bert``'s own code is not what fails below.

    MEASURED (GPU 1): 2 output tensors, the float one at ``float16``, finite.
    """
    report = run_forward(_build_encoder, _inputs, "mixed_float16", training=False)

    assert report["model_policy"] == "mixed_float16"
    float_dtypes = [d for d in report["dtypes"] if d.startswith("float")]
    assert float_dtypes, f"the encoder returned no float tensor: {report['dtypes']}"
    assert all(d == "float16" for d in float_dtypes), (
        f"the encoder's float outputs are {float_dtypes} under mixed_float16"
    )
    assert sum(report["n_nan"]) == 0 and sum(report["n_inf"]) == 0, (
        f"the encoder's fp16 output is not finite: {report['n_nan']} NaN / "
        f"{report['n_inf']} Inf"
    )


def test_the_head_model_is_green_under_float32() -> None:
    """Control for the xfail below: the head model is not simply broken.

    Without this, ``xfail(strict=True)`` on the fp16 arm would be satisfied by
    a model that does not run under ANY policy, and the finding would be
    misattributed to mixed precision.
    """
    report = run_forward(_build_with_head, _inputs, "float32", training=False)

    assert report["model_policy"] == "float32"
    assert report["n_tensors"] > 0
    assert sum(report["n_nan"]) == 0 and sum(report["n_inf"]) == 0
    assert all(d == "float32" for d in report["dtypes"] if d.startswith("float"))


# ---------------------------------------------------------------------
# the arm itself -- pinned RED, cause outside these two packages
# ---------------------------------------------------------------------


# DECISION plan-2026-08-23T203721-009b7ccf/D-016
# This arm is RED and is PINNED, not fixed. Do NOT "just fix it" here.
#
# WHAT NOT TO DO:
#   * Do NOT delete the xfail and edit
#     `layers/activations/expanded_activations.py` from this plan. That file is
#     outside `models/bert/` and `models/resnet/` and is shared by every
#     consumer of the activation factory's 'gelu'/'xgelu' keys; this plan's
#     declared blast radius does not include it, and the fix needs its own
#     measured consumer survey.
#   * Do NOT drop `strict=True`. A non-strict xfail goes quietly green when the
#     shared layer is repaired and the pin becomes cargo.
#   * Do NOT widen the arm's tolerances or pass `expected_compute_dtype=None`
#     to make it pass. The failure is a RAISE, not a numeric disagreement, and
#     part 2 of the arm is the part most often silently dropped.
#
# The diagnosis is confirmed by repair, not asserted: casting `ops.sqrt(2.0)`
# to `inputs.dtype` at expanded_activations.py:174 makes this arm XPASS(strict)
# on all four parts. See decisions.md D-016.
@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEASURED GPU 1: create_bert_with_head under mixed_float16 raises "
        "TypeError '`x` and `y` must have the same dtype, got tf.float16 != "
        "tf.float32' inside GELU.call(). Cause is "
        "layers/activations/expanded_activations.py:174 -- keras.ops.sqrt(2.0) "
        "is float32 under every policy -- which is OUTSIDE models/bert/ and "
        "models/resnet/ and is shared by every 'gelu'/'xgelu' activation-factory "
        "consumer. Pre-registered rule (decisions.md D-011/D-016): pin, do not "
        "fix, do not relax. strict=True so this FAILS when the shared layer is "
        "repaired and the pin must be removed."
    ),
)
def test_create_bert_with_head_runs_under_mixed_float16() -> None:
    """All four parts of the arm on the flagship factory. Currently RED."""
    reports = assert_precision_arm(
        build=_build_with_head,
        make_inputs=_inputs,
    )
    assert reports["mixed_float16"]["model_policy"] == "mixed_float16"
    assert reports["float32"]["model_policy"] == "float32"
