"""The SAM prompt samplers take a REQUIRED seed (D-263).

Step 12.8 (D-259) moved every prompt draw of the SAM 1 and SAM 2 pipelines to a
stateless op on a per-element, per-pass seed, and left ``seed=None`` as an
optional path that fell back to the stateful ``tf.random`` op. That op, under
the global seed every trainer sets, restarts its sequence on every new iterator,
so a caller that forgot the seed got the same prompts every epoch with no error.
The optional path is gone: each sampler and both record mappers refuse a call
without a seed.

Subjects: ``train.sam.data`` (``split_prompt_seed``, ``sample_point_in_mask``,
``sample_point_outside_mask``, ``jitter_box``, ``to_training_record``) and
``train.sam2.data.to_video_training_record``.
"""

import inspect

import numpy as np
import pytest
import tensorflow as tf

from train.sam import data as sam_data
from train.sam2 import data as sam2_data

IMAGE_SIZE = 64
MASK = tf.constant(np.pad(np.ones((4, 4), np.float32), 6))
BOX = tf.constant([8.0, 8.0, 40.0, 40.0])
SEED = tf.constant([3, 7], tf.int64)

SUBJECTS = {
    "split_prompt_seed": (sam_data.split_prompt_seed, lambda f, **kw: f(count=2, **kw)),
    "sample_point_in_mask": (
        sam_data.sample_point_in_mask, lambda f, **kw: f(MASK, IMAGE_SIZE, **kw)),
    "sample_point_outside_mask": (
        sam_data.sample_point_outside_mask, lambda f, **kw: f(MASK, IMAGE_SIZE, **kw)),
    "jitter_box": (sam_data.jitter_box, lambda f, **kw: f(BOX, IMAGE_SIZE, **kw)),
    "to_training_record": (
        sam_data.to_training_record,
        lambda f, **kw: f(
            {sam_data.RECORD_IMAGE: tf.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)),
             sam_data.RECORD_MASK: MASK, sam_data.RECORD_BOX: BOX},
            IMAGE_SIZE, include_box=True, **kw)),
    "to_video_training_record": (
        sam2_data.to_video_training_record,
        lambda f, **kw: f(
            {sam2_data.RECORD_IMAGE: tf.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 3)),
             sam2_data.RECORD_MASK: tf.stack([MASK, MASK]),
             sam2_data.RECORD_BOX: BOX},
            IMAGE_SIZE, include_box=True, **kw)),
}


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_the_seed_parameter_has_no_default(name):
    function, _ = SUBJECTS[name]
    parameter = inspect.signature(function).parameters["seed"]
    assert parameter.default is inspect.Parameter.empty, (
        f"{name}(seed=...) still has a default ({parameter.default!r}): a call "
        f"without a seed would draw from a stateful op that repeats every epoch"
    )


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_a_call_without_a_seed_is_refused(name):
    function, call = SUBJECTS[name]
    with pytest.raises(TypeError, match="seed"):
        call(function)


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_a_none_seed_is_refused(name):
    """``seed=None`` was the old opt-in to the stateful op; it is an error now."""
    function, call = SUBJECTS[name]
    with pytest.raises(TypeError, match="seed"):
        call(function, seed=None)


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_a_call_with_a_seed_is_reproducible(name):
    function, call = SUBJECTS[name]
    first = tf.nest.flatten(call(function, seed=SEED))
    second = tf.nest.flatten(call(function, seed=SEED))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
