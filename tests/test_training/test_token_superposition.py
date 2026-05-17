"""Tests for ``dl_techniques.training.token_superposition`` (TST).

Covers the seven correctness invariants from the plan plus a small
integration smoke test. Test names mirror the plan's success-criterion IDs
(``inv1_...``, ``inv2_...``, etc.) so the verification table maps 1-to-1.
"""

import dataclasses

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.training.token_superposition import (
    TSTConfig,
    TSTState,
)


# =====================================================================
# Step 2 — TSTConfig + TSTState
# =====================================================================


class TestTSTConfig:
    def test_defaults_match_documented(self):
        cfg = TSTConfig()
        assert cfg.bag_size == 6
        assert cfg.phase1_step_ratio == 0.25
        assert cfg.within_bag_weighting == "uniform"
        assert cfg.within_bag_alpha == 0.6

    def test_is_frozen(self):
        cfg = TSTConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.bag_size = 7  # type: ignore[misc]

    def test_validation_bag_size_zero(self):
        with pytest.raises(ValueError, match="bag_size"):
            TSTConfig(bag_size=0)

    def test_validation_bag_size_negative(self):
        with pytest.raises(ValueError, match="bag_size"):
            TSTConfig(bag_size=-3)

    def test_validation_phase1_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="phase1_step_ratio"):
            TSTConfig(phase1_step_ratio=1.5)
        with pytest.raises(ValueError, match="phase1_step_ratio"):
            TSTConfig(phase1_step_ratio=-0.01)

    def test_validation_alpha_non_positive(self):
        with pytest.raises(ValueError, match="within_bag_alpha"):
            TSTConfig(within_bag_alpha=0.0)
        with pytest.raises(ValueError, match="within_bag_alpha"):
            TSTConfig(within_bag_alpha=-0.5)

    def test_validation_unknown_weighting(self):
        with pytest.raises(ValueError, match="within_bag_weighting"):
            TSTConfig(within_bag_weighting="exotic")  # type: ignore[arg-type]


class TestTSTState:
    def test_phase_active_is_tf_variable_bool(self):
        st = TSTState(bag_size=4)
        assert isinstance(st.phase_active, tf.Variable)
        assert st.phase_active.dtype == tf.bool
        assert bool(st.phase_active.numpy()) is True

    def test_global_step_is_tf_variable_int64(self):
        st = TSTState(bag_size=4)
        assert isinstance(st.global_step, tf.Variable)
        assert st.global_step.dtype == tf.int64
        assert int(st.global_step.numpy()) == 0

    def test_phase_active_is_not_trainable(self):
        st = TSTState(bag_size=4)
        assert st.phase_active.trainable is False
        assert st.global_step.trainable is False

    def test_phase_active_init_false(self):
        st = TSTState(bag_size=2, phase_active_init=False)
        assert bool(st.phase_active.numpy()) is False

    def test_reset(self):
        st = TSTState(bag_size=2)
        st.phase_active.assign(False)
        st.global_step.assign(100)
        st.reset()
        assert bool(st.phase_active.numpy()) is True
        assert int(st.global_step.numpy()) == 0

    def test_invalid_bag_size(self):
        with pytest.raises(ValueError, match="bag_size"):
            TSTState(bag_size=0)
