"""Tests for dl_techniques.utils.masking.factory shared attend-mask helpers.

Direct unit coverage for:
- `create_causal_attend_mask`: shared rank-3 causal (+ optional padding)
  attend-mask helper consolidated from duplicated implementations in
  `dl_techniques.layers.blt.entropy_model`, `dl_techniques.layers.blt.local_encoder`,
  `dl_techniques.layers.blt.global_transformer`, `dl_techniques.layers.blt.local_decoder`,
  `dl_techniques.models.vision_language.clip.model` (pure-causal), and
  `dl_techniques.models.language.qwen.components`,
  `dl_techniques.layers.transformers.text_decoder` (causal + optional padding).
- `create_banded_attend_mask`: shared rank-3 symmetric-band attend-mask
  helper consolidated from
  `dl_techniques.layers.attention.window_attention.WindowAttention._call_band`.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils.masking import create_banded_attend_mask, create_causal_attend_mask


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


class TestCreateBandedAttendMask:

    def test_shape_and_dtype(self):
        hidden_states = keras.ops.zeros((3, 6, 4))
        mask = create_banded_attend_mask(hidden_states, window_size=2)

        assert tuple(mask.shape) == (1, 6, 6)
        assert mask.dtype == "int32"

    def test_band_pattern_window_size_2(self):
        hidden_states = keras.ops.zeros((1, 6, 4))
        mask = create_banded_attend_mask(hidden_states, window_size=2)
        mask_np = keras.ops.convert_to_numpy(mask)[0]

        positions = np.arange(6)
        expected = (np.abs(positions[:, None] - positions[None, :]) <= 2).astype(np.int32)
        np.testing.assert_array_equal(mask_np, expected)

    def test_window_size_zero_raises(self):
        # window_size=0 -> band_width=0, rejected by the underlying
        # MaskFactory.create_banded_mask guard (band_width must be positive).
        # Not a real regression: no caller in this codebase constructs
        # WindowAttention(partition_mode='band') with window_size=0 (unvalidated
        # in the original hand-rolled code, but never exercised in practice).
        hidden_states = keras.ops.zeros((1, 4, 2))
        with pytest.raises(ValueError, match="band_width must be positive"):
            create_banded_attend_mask(hidden_states, window_size=0)

    def test_none_attention_mask_returns_band_alone(self):
        hidden_states = keras.ops.zeros((2, 5, 3))
        band_alone = create_banded_attend_mask(hidden_states, window_size=1)
        band_explicit_none = create_banded_attend_mask(hidden_states, window_size=1, attention_mask=None)

        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(band_alone),
            keras.ops.convert_to_numpy(band_explicit_none),
        )

    def test_rank3_attention_mask_composes_by_and(self):
        hidden_states = keras.ops.zeros((1, 4, 2))
        attention_mask = keras.ops.convert_to_tensor(
            [[[1, 1, 0, 0],
              [1, 1, 1, 0],
              [0, 1, 1, 1],
              [0, 0, 1, 1]]], dtype="int32"
        )
        mask = keras.ops.convert_to_numpy(
            create_banded_attend_mask(hidden_states, window_size=1, attention_mask=attention_mask)
        )[0]

        positions = np.arange(4)
        band = (np.abs(positions[:, None] - positions[None, :]) <= 1).astype(np.int32)
        expected = band * keras.ops.convert_to_numpy(attention_mask)[0]
        np.testing.assert_array_equal(mask, expected)

    def test_rank2_attention_mask_composes_by_and(self):
        hidden_states = keras.ops.zeros((1, 4, 2))
        attention_mask = keras.ops.convert_to_tensor([[1, 1, 0, 1]], dtype="int32")
        mask = keras.ops.convert_to_numpy(
            create_banded_attend_mask(hidden_states, window_size=1, attention_mask=attention_mask)
        )[0]

        positions = np.arange(4)
        band = (np.abs(positions[:, None] - positions[None, :]) <= 1).astype(np.int32)
        key_mask = np.array([1, 1, 0, 1])[None, :]
        expected = band * key_mask
        np.testing.assert_array_equal(mask, expected)

    def test_invalid_attention_mask_rank_raises(self):
        hidden_states = keras.ops.zeros((1, 4, 2))
        bad_mask = keras.ops.zeros((1, 4, 4, 4))
        with pytest.raises(ValueError, match="rank-2.*rank-3"):
            create_banded_attend_mask(hidden_states, window_size=1, attention_mask=bad_mask)
