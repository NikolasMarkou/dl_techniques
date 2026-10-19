"""Tests for Timestral attention masks."""

import keras
import numpy as np
import pytest

from dl_techniques.models.time_series.timestral.masks import (
    compute_enc_key_validity,
    enc_causal_mask,
    build_forecast_time_mask,
    build_forecast_positions,
    compute_block_layout,
    build_two_stream_time_mask,
    variate_firewall,
    build_two_stream_variate_mask,
    build_two_stream_positions,
    first_observed_patch,
)


class TestMasks:
    """Test DHF attention mask construction."""

    def test_compute_enc_key_validity_no_attend_missing(self):
        """Test key validity without attend_missing."""
        obs = keras.ops.convert_to_tensor([[True, True, False, True]])
        valid = compute_enc_key_validity(obs, attend_missing=False)
        assert valid.shape == (1, 4)
        assert keras.ops.all(valid == obs)

    def test_compute_enc_key_validity_attend_missing(self):
        """Test key validity with attend_missing."""
        # First patch unobserved (padding), then observed, then missing
        obs = keras.ops.convert_to_tensor([[False, True, False, True]])
        valid = compute_enc_key_validity(obs, attend_missing=True)
        # First False stays False (padding), then True propagates
        expected = [[False, True, True, True]]
        assert keras.ops.all(valid == keras.ops.convert_to_tensor(expected))

    def test_enc_causal_mask(self):
        """Test causal encoder mask."""
        key_valid = keras.ops.convert_to_tensor([[True, True, False, True]])
        mask = enc_causal_mask(key_valid)
        assert mask.shape == (1, 4, 4)
        # Causal: i >= j
        # Plus diagonal always open
        # Plus invalid keys masked out
        for i in range(4):
            for j in range(4):
                if i >= j and key_valid[0, j]:
                    assert mask[0, i, j]
                elif i == j:
                    assert mask[0, i, j]  # diagonal always open
                else:
                    assert not mask[0, i, j]

    def test_build_forecast_time_mask(self):
        """Test forecast time mask without covariates."""
        key_valid = keras.ops.convert_to_tensor([[True, True, True, True]])
        mask = build_forecast_time_mask(4, 2, key_valid)
        assert mask.shape == (1, 1, 6, 6)
        # Encoder part (0:4, 0:4) should be causal
        # Decoder part (4:6, 0:4) should attend to all valid encoder keys
        # Decoder part (4:6, 4:6) should be causal

    def test_build_forecast_positions(self):
        """Test forecast position computation."""
        first_obs = keras.ops.convert_to_tensor([1, 0])
        pos = build_forecast_positions(4, 2, first_obs)
        assert pos.shape == (2, 6)
        # Row 0: first_obs=1 -> positions -1,0,1,2,3,4
        # Row 1: first_obs=0 -> positions 0,1,2,3,4,5

    def test_compute_block_layout(self):
        """Test block anchor computation."""
        min_context = keras.ops.convert_to_tensor([2, 0])
        anchor, active = compute_block_layout(min_context, 8, 4)
        assert anchor.shape == (2, 8)
        assert active.shape == (2, 8)
        # Row 0: min_context=2, block_size=4
        # p=0,1: inactive; p=2,3,4,5: block 2; p=6,7: block 6
        assert not active[0, 0] and not active[0, 1]
        assert active[0, 2] and active[0, 5]
        assert anchor[0, 2] == 2 and anchor[0, 5] == 2
        assert anchor[0, 6] == 6

    def test_build_two_stream_time_mask(self):
        """Test two-stream time mask with covariates."""
        min_context = keras.ops.convert_to_tensor([2, 0])
        key_valid = keras.ops.convert_to_tensor([[True]*8, [True]*8])
        bidirectional = keras.ops.convert_to_tensor([False, True])

        mask = build_two_stream_time_mask(min_context, 8, 4, key_valid, bidirectional)
        assert mask.shape == (2, 1, 16, 16)

    def test_variate_firewall(self):
        """Test variate attention firewall."""
        # 3 rows: 0=target, 1=past_cov, 2=known_cov
        target_rows = keras.ops.convert_to_tensor([True, False, False])
        bidirectional_rows = keras.ops.convert_to_tensor([False, False, True])

        forbid = variate_firewall(3, target_rows, bidirectional_rows)
        assert forbid.shape == (3, 3)
        # Covariate rows (1,2) cannot read target row (0)
        assert forbid[1, 0] and forbid[2, 0]
        # Known-future (2) cannot read causal rows (0,1)
        assert forbid[2, 0] and forbid[2, 1]
        # Diagonal should be open
        assert not forbid[0, 0] and not forbid[1, 1] and not forbid[2, 2]

    def test_build_two_stream_variate_mask(self):
        """Test two-stream variate mask."""
        group_ids = keras.ops.convert_to_tensor([0, 0, 1, 1])  # 2 series, 2 rows each
        enc_key_valid = keras.ops.convert_to_tensor([[True]*4, [True]*4, [True]*4, [True]*4])
        target_rows = keras.ops.convert_to_tensor([True, False, True, False])
        bidirectional_rows = keras.ops.convert_to_tensor([False, False, False, True])

        mask = build_two_stream_variate_mask(
            group_ids, enc_key_valid, 2, target_rows, bidirectional_rows
        )
        assert mask.shape == (6, 1, 4, 4)  # 4+2 patches, 4 rows

    def test_build_two_stream_positions(self):
        """Test two-stream positions."""
        first_obs = keras.ops.convert_to_tensor([1, 0])
        pos = build_two_stream_positions(4, first_obs)
        assert pos.shape == (2, 8)
        # Encoder and decoder positions same per patch
        assert np.array_equal(pos[0, :4], pos[0, 4:])

    def test_group_first_observed_patch(self):
        """Test group first observed patch."""
        obs = keras.ops.convert_to_tensor([[False, True, False], [False, False, True]])
        group_ids = keras.ops.convert_to_tensor(np.array([0, 0], dtype=np.int32))
        first = first_observed_patch(obs, group_ids)
        assert first.shape == (2,)
        # Group 0: row 0 first obs at 1, row 1 at 2 -> group first is 1
        # (argmax of group max [0,1,1] is index 1)
        assert int(first[0]) == 1
        assert int(first[1]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])