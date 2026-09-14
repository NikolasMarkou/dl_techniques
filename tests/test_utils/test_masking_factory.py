"""Tests for dl_techniques.utils.masking.factory.create_causal_attend_mask.

Direct unit coverage for the shared rank-3 causal (+ optional padding)
attend-mask helper consolidated from duplicated implementations in
`dl_techniques.layers.blt.blt_blocks`,
`dl_techniques.models.vision_language.clip.model` (pure-causal), and
`dl_techniques.models.language.qwen.components`,
`dl_techniques.layers.transformers.text_decoder` (causal + optional padding).
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

    def test_none_attention_mask_matches_pre_extension_behavior(self):
        hidden_states = keras.ops.zeros((2, 5, 4))

        mask_explicit_none = create_causal_attend_mask(hidden_states, None)
        mask_omitted = create_causal_attend_mask(hidden_states)

        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(mask_explicit_none),
            keras.ops.convert_to_numpy(mask_omitted),
        )

    def test_padding_mask_suppresses_padded_query_and_key_positions(self):
        hidden_states = keras.ops.zeros((1, 4, 2))
        attention_mask = keras.ops.convert_to_tensor([[1, 1, 1, 0]])

        mask = create_causal_attend_mask(hidden_states, attention_mask)
        mask_np = keras.ops.convert_to_numpy(mask)[0]

        # Causal triangle further suppressed: key col 3 blocked for every
        # query, and query row 3 (itself padded) attends to nothing.
        expected = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [False, False, False, False],
        ])
        np.testing.assert_array_equal(mask_np, expected)

    def test_padding_mask_batch_broadcasting(self):
        hidden_states = keras.ops.zeros((2, 4, 2))
        attention_mask = keras.ops.convert_to_tensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ])

        mask = create_causal_attend_mask(hidden_states, attention_mask)
        mask_np = keras.ops.convert_to_numpy(mask)

        # Example 0: no padding, plain causal triangle.
        expected_0 = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ])
        np.testing.assert_array_equal(mask_np[0], expected_0)

        # Example 1: last two positions padded (both as query and key).
        expected_1 = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ])
        np.testing.assert_array_equal(mask_np[1], expected_1)
