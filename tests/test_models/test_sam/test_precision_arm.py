"""
R-088 / R-141 regression pin for ``SAM``: the float32 normalization constants.

The four-part arm itself runs in
``tests/test_models/test_precision_arm_family.py`` (subject ``SAM``,
``allowed_none_grads=12``, a count MEASURED IDENTICAL under float32). This
file pins the DEFECT that writing the arm found.

MEASURED at HEAD, on the reduced fixture of ``test_correctness.py``:

* any ``mixed_float16`` forward RAISED
  ``InvalidArgumentError: cannot compute Sub as input #1(zero-based) was
  expected to be a half tensor but is a float tensor`` at
  ``models/SAM/SAM1/model.py:572``, i.e. in ``preprocess``, before the image
  encoder ran at all;
* the float32 control was green.

Root cause (decisions.md D-063): ``pixel_mean`` / ``pixel_std`` are built once
in ``__init__`` as HARD float32 constants and were used uncast.
"""

import numpy as np
import pytest
from keras import ops

from ..precision_arm_oracle import precision_policy, run_backward
from ..precision_arm_subjects import SUBJECTS


def test_the_pre_fix_expression_raises_and_the_cast_repairs_it():
    """RED then GREEN, on the exact arithmetic of ``preprocess``."""
    with precision_policy("mixed_float16"):
        image = ops.convert_to_tensor(
            np.full((1, 4, 4, 3), 128.0, dtype="float16"))
        pixel_mean = ops.convert_to_tensor(
            np.array([123.675, 116.28, 103.53], dtype="float32"))
        with pytest.raises(Exception, match="half tensor|same dtype"):
            _ = image - pixel_mean
        repaired = image - ops.cast(pixel_mean, image.dtype)
        assert str(repaired.dtype).endswith("float16'>") or \
            "float16" in str(repaired.dtype)


def test_the_constants_are_still_float32_so_the_cast_is_load_bearing():
    """If ``__init__`` ever built them in the compute dtype, this fix is dead
    weight and should be removed rather than left unexplained."""
    from dl_techniques.models.SAM.SAM1.model import SAM
    from .test_correctness import build_reduced_sam
    with precision_policy("mixed_float16"):
        model = build_reduced_sam()
    assert isinstance(model, SAM)
    assert "float32" in str(model.pixel_mean.dtype), model.pixel_mean.dtype
    assert "float32" in str(model.pixel_std.dtype), model.pixel_std.dtype


def test_the_forward_runs_under_mixed_float16():
    """GREEN: the arm's part 1, stated at the package."""
    build, make_inputs, _kwargs = SUBJECTS["SAM"]
    with precision_policy("mixed_float16"):
        import keras
        keras.utils.set_random_seed(0)
        model = build()
        out = model(make_inputs(), training=False)
    tensors = list(out.values()) if isinstance(out, dict) else [out]
    floats = [t for t in tensors if "float" in str(t.dtype)]
    assert floats, "SAM returned no float tensor to judge"
    for t in floats:
        assert "float16" in str(t.dtype), (t.dtype, "expected the compute dtype")
        arr = np.asarray(ops.convert_to_numpy(ops.cast(t, "float32")))
        assert np.isfinite(arr).all()


def test_the_backward_none_count_matches_the_float32_control():
    """The 12 unreachable variables are a MODEL property, not an fp16 one."""
    build, make_inputs, _kwargs = SUBJECTS["SAM"]
    fp16 = run_backward(build, make_inputs, "mixed_float16")
    f32 = run_backward(build, make_inputs, "float32")
    assert fp16["n_none"] == f32["n_none"] == 12, (fp16["n_none"], f32["n_none"])
    assert fp16["n_vars"] == f32["n_vars"] == 201
    assert fp16["n_nonfinite"] == 0
    assert fp16["grad_norm_sum"] > 0.0
