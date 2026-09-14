"""Tests for dl_techniques.utils.masking.factory.create_causal_attend_mask.

Direct unit coverage for the shared rank-3 causal attend-mask helper
consolidated from duplicated implementations in
`dl_techniques.layers.blt.blt_blocks` and
`dl_techniques.models.vision_language.clip.model`.
"""

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.masking import create_causal_attend_mask


class TestCreateCausalAttendMask:

    def test_shape_and_dtype(self):
        hidden_states = keras.ops.zeros((3, 5, 8))
        mask = create_causal_attend_mask(hidden_states)

        assert tuple(mask.shape) == (3, 5, 5)
        assert mask.dtype == "bool"

    def test_lower_triangular_attend_pattern(self):
        hidden_states = keras.ops.zeros((1, 4, 2))
        mask = create_causal_attend_mask(hidden_states)
        mask_np = keras.ops.convert_to_numpy(mask)[0]

        expected = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ])
        np.testing.assert_array_equal(mask_np, expected)

    def test_batch_broadcasting_is_identical_per_example(self):
        hidden_states = keras.ops.zeros((4, 6, 3))
        mask = create_causal_attend_mask(hidden_states)
        mask_np = keras.ops.convert_to_numpy(mask)

        for b in range(1, 4):
            np.testing.assert_array_equal(mask_np[b], mask_np[0])

    def test_ignores_values_and_dtype_of_input(self):
        random_values = keras.random.normal((2, 5, 4))
        int_values = keras.ops.ones((2, 5, 4), dtype="int32")

        mask_a = create_causal_attend_mask(random_values)
        mask_b = create_causal_attend_mask(int_values)

        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(mask_a),
            keras.ops.convert_to_numpy(mask_b),
        )

    def test_correct_under_tf_function_tracing(self):
        @tf.function
        def build_mask(x):
            return create_causal_attend_mask(x)

        hidden_states = tf.zeros((2, 4, 3))
        mask = build_mask(hidden_states)
        mask_np = mask.numpy()[0]

        expected = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ])
        np.testing.assert_array_equal(mask_np, expected)
