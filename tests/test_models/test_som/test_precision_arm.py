"""
R-088 / R-141 regression pin for ``som``: the autocast-vs-grid dtype split.

The four-part arm itself runs in
``tests/test_models/test_precision_arm_family.py`` (subject ``som``, with
``check_backward=False`` because a SOM has no trainable variables at all).
This file pins the DEFECT that writing the arm found.

MEASURED at HEAD:

* ``SOMModel(map_size=(4, 4), input_dim=8)`` under ``mixed_float16`` with
  ``training=True`` RAISED ``TypeError: x and y must have the same dtype, got
  tf.float32 != tf.float16`` at ``som_nd_layer.py:445``;
* the same model with ``training=False`` was GREEN, which is why no existing
  test saw it -- the whole competitive-learning path is training-only.

Root cause (decisions.md D-062): Keras AUTOCASTS a float32 variable to the
compute dtype when it is read inside ``call``, so ``self.iterations`` and
``self.max_iterations`` came back float16, while ``self.grid_positions`` -- a
plain tensor, not a variable -- stayed float32.
"""

import numpy as np
import pytest
from keras import ops

from ..precision_arm_oracle import precision_policy
from ..precision_arm_subjects import SUBJECTS


def test_keras_autocasts_a_float32_variable_read_inside_call():
    """The RED half: the framework behaviour the defect was made of.

    If Keras ever stops autocasting non-trainable float32 variables, this test
    fails and the explicit casts in ``_update_weights`` become removable.
    """
    import keras

    seen = {}

    class _Probe(keras.layers.Layer):
        def build(self, input_shape):
            self.counter = self.add_weight(
                name="counter", shape=(), dtype="float32",
                initializer="zeros", trainable=False)
            super().build(input_shape)

        def call(self, x):
            seen["inside_call_dtype"] = str(self.counter.dtype)
            seen["inside_call_read_dtype"] = str((self.counter + 0.0).dtype)
            return x

    with precision_policy("mixed_float16"):
        layer = _Probe()
        layer(ops.zeros((2, 3)))
        outside_call_dtype = str(layer.counter.dtype)

    # The variable is declared `dtype="float32"` and really IS float32 when
    # read outside `call` -- which is why a probe written outside `call` (the
    # first one this step wrote) reports no problem at all.
    assert "float32" in outside_call_dtype, outside_call_dtype
    assert "float16" in seen["inside_call_dtype"], (
        "a float32 variable read inside `call` is no longer autocast; "
        f"measured {seen['inside_call_dtype']!r}"
    )
    assert "float16" in seen["inside_call_read_dtype"]


def test_the_pre_fix_expression_raises_and_the_cast_repairs_it():
    """RED then GREEN, on the exact arithmetic of ``som_nd_layer.py:445``."""
    with precision_policy("mixed_float16"):
        squared_distance = ops.convert_to_tensor(
            np.arange(16, dtype="float32").reshape(4, 4))   # the float32 grid
        sigma_autocast = ops.convert_to_tensor(
            np.array(1.0, dtype="float16"))                 # the autocast read
        with pytest.raises(TypeError):
            ops.exp(-squared_distance / (2 * ops.square(sigma_autocast)))
        repaired = ops.exp(
            -squared_distance
            / (2 * ops.square(ops.cast(sigma_autocast, "float32")))
        )
        assert np.isfinite(
            np.asarray(ops.convert_to_numpy(repaired))).all()


@pytest.mark.parametrize("training", [False, True])
def test_the_som_forward_runs_under_mixed_float16_in_both_modes(training):
    """GREEN: the training path -- where the defect lived -- now runs."""
    build, make_inputs, _kwargs = SUBJECTS["som"]
    with precision_policy("mixed_float16"):
        import keras
        keras.utils.set_random_seed(0)
        model = build()
        bmu, error = model(make_inputs(), training=training)
    assert str(bmu.dtype) == "<dtype: 'int32'>" or "int" in str(bmu.dtype)
    err = np.asarray(ops.convert_to_numpy(ops.cast(error, "float32")))
    assert np.isfinite(err).all()


def test_the_competitive_update_actually_moved_the_map():
    """Anti-vacuity: a training call that changed NOTHING would pass above."""
    build, make_inputs, _kwargs = SUBJECTS["som"]
    with precision_policy("mixed_float16"):
        import keras
        keras.utils.set_random_seed(0)
        model = build()
        x = make_inputs()
        model(x, training=False)
        before = np.asarray(ops.convert_to_numpy(
            ops.cast(model.som_layer.weights_map, "float32"))).copy()
        for _ in range(3):
            model(x, training=True)
        after = np.asarray(ops.convert_to_numpy(
            ops.cast(model.som_layer.weights_map, "float32")))
    delta = float(np.abs(after - before).max())
    assert delta > 0.0, (
        "three training calls under mixed_float16 left the weight map "
        "bit-identical -- the update did not run")
