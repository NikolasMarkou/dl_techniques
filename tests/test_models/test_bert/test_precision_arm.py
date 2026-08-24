"""
R-088 / R-141 mixed-precision arm for ``models/language/bert`` -- and the defect it found and fixed.

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

THE FINDING (MEASURED, GPU 1 / RTX 4070, TF32 on by default) -- NOW FIXED
--------------------------------------------------------------------------
``create_bert_with_head("tiny", NLPTaskConfig(SENTIMENT_ANALYSIS, num_classes=3))``
under ``mixed_float16`` RAISED, verbatim::

    TypeError: Exception encountered when calling GELU.call().

    `x` and `y` must have the same dtype, got tf.float16 != tf.float32.

    Arguments received by GELU.call():
      • inputs=tf.Tensor(shape=(2, 32), dtype=float16)

Its float32 control is green, and the bare encoder at the SAME config is green
under ``mixed_float16`` too (``test_the_encoder_is_green_which_localises_the_defect``),
so the fault is in the head path and not in ``models/language/bert``'s own encoder.

Root cause, isolated at the layer -- the PRE-FIX line::

    # layers/activations/expanded_activations.py (pre-fix)
    return 0.5 * inputs * (1 + keras.ops.erf(inputs / keras.ops.sqrt(2.0)))

and the repaired one, which is what ships::

    root_two = keras.ops.cast(keras.ops.sqrt(2.0), inputs.dtype)
    return 0.5 * inputs * (1 + keras.ops.erf(inputs / root_two))

``keras.ops.sqrt(2.0)`` returns **float32 regardless of the active dtype
policy** (MEASURED below), so it meets a float16 autocast tensor and TensorFlow
refuses the divide. This is the same shape as the ``qwen`` MoE defect
(``ops.one_hot`` ignoring the policy, decisions.md D-064 of
``plan-2026-08-19T163559-499b6f0e``), and it hits exactly 2 of the 6 concrete
activations in that module -- ``GELU`` and ``xGELU``, both of which divide by
``ops.sqrt(2.0)``. ``SiLU``, ``xSiLU``, ``xATLU`` and ``EluPlusOne`` were all
green at float16 even before the repair.

DISPOSITION -- escalated under D-011/D-016, then RESOLVED as step 3.1
---------------------------------------------------------------------
Step 3 pinned this ``xfail(strict=True)`` and escalated it, because the cause
lives in ``src/dl_techniques/layers/activations/expanded_activations.py``, which
is outside ``models/language/bert/`` and ``models/vision/resnet/`` and is reached by every
consumer of the activation factory's ``'gelu'`` / ``'xgelu'`` keys. The
orchestrator resolved the escalation: BERT's flagship factory failing under
``mixed_float16`` is not a shippable state, so step 3.1 REPAIRED the shared
layer and this arm is now a plain, green, four-part assertion.

The affected set was RE-DERIVED at step 3.1 rather than carried from step 3's
"2 of 6": every class in that module was swept under BOTH ``mixed_float16`` and
``mixed_bfloat16``, with a float32 and a half-precision input each. The measured
set MATCHED -- exactly ``GELU`` and ``xGELU``, the only two that divide by the
float32 tensor ``keras.ops.sqrt(2.0)``. ``SiLU``, ``xATLU``, ``xSiLU`` and
``EluPlusOne`` were green in all four cells, and the float32 forward through
``GELU``/``xGELU`` is BITWISE identical before and after the repair
(``max|delta| = 0.0``, and the ``uint32`` views compare equal element-wise).

RED proof of the repaired assertions (fix reverted, then restored) --
ACTUAL observed text, recorded rather than predicted::

    reverted `expanded_activations.py` -> 4 failed, 2 passed:

      FAILED test_the_gelu_layers_carry_the_input_dtype_onto_their_sqrt2[GELU]
      FAILED test_the_gelu_layers_carry_the_input_dtype_onto_their_sqrt2[xGELU]
      FAILED test_every_expanded_activation_is_green_at_float16
      FAILED test_create_bert_with_head_runs_under_mixed_float16

    with, in all four, the SAME exception rather than an assertion message --
    recorded because it is not what was predicted: the layer RAISES before
    arm 2's `assert "float16" in str(out.dtype)` can evaluate, so the message
    that fires is the framework's, not this file's::

      TypeError: Exception encountered when calling GELU.call().

      `x` and `y` must have the same dtype, got tf.float16 != tf.float32.

      Arguments received by GELU.call():
        - inputs=tf.Tensor(shape=(2, 4), dtype=float16)

      src/dl_techniques/layers/activations/expanded_activations.py:176: TypeError

    (`xGELU.call()` at :444 gives the identical text with its own class name;
    `test_every_expanded_activation_is_green_at_float16` dies in its FIRST cell,
    `GELU` on a float32 input, which is why it names all six rather than
    trusting the loop to reach them.)

See ``decisions.md`` D-016 (the pin) and D-017 (the repair).
"""

from typing import Any, Dict

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.layers.activations.expanded_activations import GELU, xGELU
from dl_techniques.models.language.bert import create_bert, create_bert_with_head
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
def test_the_gelu_layers_carry_the_input_dtype_onto_their_sqrt2(activation_class) -> None:
    """The repaired half: the mechanism is still live, the layers no longer trip on it.

    Three arms, because the middle one alone would be satisfied by a layer that
    stopped using ``sqrt(2)`` at all, and the first alone proves nothing about
    this repository's code:

    1. ``keras.ops.sqrt(2.0)`` STILL returns a float32 tensor under
       ``mixed_float16`` -- the framework behaviour that caused the defect is a
       fact about Keras 3.8, not something the fix removed. If Keras ever makes
       it follow the compute dtype this arm fails, and the cast at
       ``expanded_activations.py`` becomes removable.
    2. The layer now returns a ``float16`` tensor on a ``float16`` input. This
       is the arm that goes RED the moment the cast is reverted.
    3. The naive expression -- a float16 tensor divided by that raw float32
       tensor -- STILL raises. This is the anti-vacuity arm: without it, arm 2
       would be indistinguishable from "float16 and float32 mix freely here",
       and the repair would be unearned.

    MEASURED (GPU 1): ``ops.sqrt(2.0).dtype`` is ``float32`` inside a
    ``mixed_float16`` policy; both ``GELU`` and ``xGELU`` return ``float16``;
    the raw divide raises ``TypeError``.
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

        out = activation_class()(half)
        assert "float16" in str(out.dtype), (
            f"{activation_class.__name__} returned {out.dtype} on a float16 "
            f"input under mixed_float16; the sqrt(2) constant is not carrying "
            f"the input's dtype"
        )

        with pytest.raises(TypeError, match="same dtype"):
            _ = half / root_two


def test_every_expanded_activation_is_green_at_float16() -> None:
    """After the repair the correct assertion is that ALL SIX are green.

    Before step 3.1 this arm named four and existed to prove the module was not
    uniformly broken. That framing is now the wrong one: the pin is gone, so the
    discriminating question is whether any activation in the module still fails
    to carry the compute dtype. Naming all six keeps it RED against a partial
    repair -- fixing ``GELU`` and forgetting ``xGELU``, say.

    It does not become vacuous, because it is paired with arm 3 of
    ``test_the_gelu_layers_carry_the_input_dtype_onto_their_sqrt2`` above, which
    pins that a float32 constant genuinely DOES still raise in this regime.

    MEASURED (GPU 1, step 3.1): all six return ``float16`` under
    ``mixed_float16`` and ``bfloat16`` under ``mixed_bfloat16``, for both a
    float32 and a half-precision input. Before the repair, ``GELU`` and
    ``xGELU`` raised in all four of those cells.
    """
    from dl_techniques.layers.activations.expanded_activations import (
        EluPlusOne, SiLU, xATLU, xSiLU,
    )

    every_activation = (GELU, SiLU, xATLU, xGELU, xSiLU, EluPlusOne)
    assert len(every_activation) == 6, (
        "this arm claims to cover every concrete activation in "
        "expanded_activations.py; update it when one is added"
    )

    with precision_policy("mixed_float16"):
        x32 = ops.convert_to_tensor(np.random.RandomState(0).randn(2, 4).astype("float32"))
        for activation_class in every_activation:
            for label, x in (("float32", x32), ("float16", ops.cast(x32, "float16"))):
                out = activation_class()(x)
                assert "float16" in str(out.dtype), (
                    f"{activation_class.__name__} returned {out.dtype} on a "
                    f"{label} input under mixed_float16; a float32 constant in "
                    f"its call() is not carrying the compute dtype"
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
    ``models/language/bert``'s own code is not what fails below.

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
# the arm itself -- was pinned RED at step 3, repaired at step 3.1
# ---------------------------------------------------------------------


# DECISION plan-2026-08-23T203721-009b7ccf/D-016
# DECISION plan-2026-08-23T203721-009b7ccf/D-017
# This arm WAS `xfail(strict=True)` under D-016, escalated as a scope question,
# and the escalation was resolved by repairing the shared layer (D-017). The
# `xfail` mark is deliberately GONE, not merely satisfied.
#
# WHAT NOT TO DO:
#   * Do NOT restore the `xfail`. It was `strict=True` precisely so it would
#     FAIL as XPASS once the shared layer was repaired; re-adding it now would
#     turn a green four-part arm into a permanent red one.
#   * Do NOT revert `keras.ops.cast(keras.ops.sqrt(2.0), inputs.dtype)` in
#     `layers/activations/expanded_activations.py`. That is the whole repair,
#     and this arm is its model-level detector.
#   * Do NOT widen the arm's tolerances or pass `expected_compute_dtype=None`.
#     Part 2 of the arm is the part most often silently dropped, and the
#     original failure was a RAISE, not a numeric disagreement.
# See decisions.md D-016 (the pin) and D-017 (the repair).
def test_create_bert_with_head_runs_under_mixed_float16() -> None:
    """All four parts of the arm on the flagship factory. GREEN since step 3.1."""
    reports = assert_precision_arm(
        build=_build_with_head,
        make_inputs=_inputs,
    )
    assert reports["mixed_float16"]["model_policy"] == "mixed_float16"
    assert reports["float32"]["model_policy"] == "float32"
