"""
R-088 / R-141 regression pin for ``qwen``: the float32 MoE routing mask.

The four-part arm itself runs in
``tests/test_models/test_precision_arm_family.py`` (subject ``qwen``). This
file pins the DEFECT that writing the arm found -- which lives in
``layers/moe/layer.py``, so EVERY hard-routed MoE consumer carried it, not
just Qwen.

MEASURED at HEAD:

* ``Qwen3Next`` under ``mixed_float16`` RAISED
  ``InvalidArgumentError: cannot compute Mul as input #1(zero-based) was
  expected to be a half tensor but is a float tensor`` at
  ``layers/moe/layer.py:323``;
* the float32 control was green.

Root cause (decisions.md D-064): ``ops.one_hot`` returns float32 regardless of
the active dtype policy, so the routing mask met a float16 expert output.
"""

import numpy as np
import pytest
from keras import ops

from ..precision_arm_oracle import precision_policy, run_backward
from ..precision_arm_subjects import SUBJECTS


def test_one_hot_ignores_the_policy_which_is_what_made_the_defect():
    """The RED half: the framework behaviour, pinned.

    If ``ops.one_hot`` ever follows the compute dtype, the casts in
    ``_process_hard_routing`` become removable rather than load-bearing.
    """
    with precision_policy("mixed_float16"):
        one_hot = ops.one_hot(ops.convert_to_tensor(np.array([[0, 1]])), 4)
        assert "float32" in str(one_hot.dtype), one_hot.dtype
        half = ops.convert_to_tensor(np.zeros((1, 2, 4), dtype="float16"))
        with pytest.raises(Exception, match="half tensor|same dtype"):
            _ = half * one_hot
        repaired = half * ops.cast(one_hot, half.dtype)
        assert "float16" in str(repaired.dtype)


def test_a_hard_routed_moe_runs_under_mixed_float16():
    """GREEN at the LAYER, not only at the model that happened to find it."""
    import keras
    from dl_techniques.layers.moe import MixtureOfExperts
    from dl_techniques.layers.moe.config import (
        ExpertConfig, GatingConfig, MoEConfig,
    )
    with precision_policy("mixed_float16"):
        keras.utils.set_random_seed(0)
        layer = MixtureOfExperts(config=MoEConfig(
            num_experts=4,
            expert_config=ExpertConfig(
                ffn_config={"type": "mlp", "hidden_dim": 16, "output_dim": 8}),
            gating_config=GatingConfig(top_k=2),
        ))
        x = ops.convert_to_tensor(
            np.random.RandomState(0).randn(2, 5, 8).astype("float32"))
        out = layer(x, training=False)
    tensor = out[0] if isinstance(out, (list, tuple)) else out
    assert "float16" in str(tensor.dtype), tensor.dtype
    arr = np.asarray(ops.convert_to_numpy(ops.cast(tensor, "float32")))
    assert np.isfinite(arr).all()
    assert np.abs(arr).max() > 0.0, "the MoE returned an all-zero tensor"


def test_the_qwen_backward_is_clean_under_mixed_float16():
    """The package-level statement, with its float32 control."""
    build, make_inputs, _kwargs = SUBJECTS["qwen"]
    fp16 = run_backward(build, make_inputs, "mixed_float16")
    f32 = run_backward(build, make_inputs, "float32")
    assert fp16["n_none"] == f32["n_none"]
    assert fp16["n_nonfinite"] == 0
    assert fp16["grad_norm_sum"] > 0.0
