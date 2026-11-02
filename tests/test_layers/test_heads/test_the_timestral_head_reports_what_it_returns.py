"""``TimestralQuantileHead`` reports its shapes and refuses levels with no median.

Measured on 2026-11-02 before the fixes (audit of ``layers/heads``):
``compute_output_shape`` raised ``NotImplementedError``, and levels without 0.5
passed the constructor and failed only in ``call``. The symbolic dtype is pinned
as well: this head returns at least float32 under ``mixed_float16``, and a plain
``compute_output_shape`` would make Keras report the float16 compute dtype.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.time_series import TimestralQuantileHead
from dl_techniques.layers.heads.vision_language import VLMTaskConfig, create_vlm_head

LEVELS = [0.1, 0.5, 0.9]


def _head():
    return TimestralQuantileHead(8, 2, LEVELS, 1.0)


@pytest.mark.parametrize("shape,expected", [
    ((None, 4, 8), (None, 8, 3)),
    ((None, None, 8), (None, None, 3)),
    ((2, 6, 4, 8), (2, 6, 8, 3)),
])
def test_compute_output_shape_matches_call(shape, expected) -> None:
    assert _head().compute_output_shape(shape) == expected


def test_compute_output_shape_agrees_with_an_eager_call() -> None:
    head = _head()
    outputs = head(np.zeros((2, 4, 8), "float32"))
    assert tuple(outputs.shape) == head.compute_output_shape((2, 4, 8))


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
def test_the_symbolic_dtype_is_the_returned_dtype(policy) -> None:
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy(policy)
    try:
        head = _head()
        symbolic = head(keras.Input((4, 8)))
        eager = head(np.zeros((1, 4, 8), "float32"))
    finally:
        keras.config.set_dtype_policy(previous)
    assert symbolic.dtype == "float32"
    assert symbolic.dtype == keras.backend.standardize_dtype(eager.dtype)
    assert symbolic.shape == (None, 8, 3)


def test_levels_with_no_median_are_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="0.5"):
        TimestralQuantileHead(8, 2, [0.1, 0.9], 1.0)


@pytest.mark.parametrize("num_heads", [0, -2, True])
def test_a_captioning_head_refuses_an_unusable_head_count(num_heads) -> None:
    config = VLMTaskConfig(name="c", task_type="image_captioning", vocab_size=8,
                           hidden_size=24)
    with pytest.raises(ValueError, match="num_heads"):
        create_vlm_head(config, num_heads=num_heads)
