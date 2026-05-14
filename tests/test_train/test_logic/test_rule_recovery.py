"""
Unit tests for ``train.logic.rule_recovery`` (load-bearing scorer).

Plan: plan_2026-05-14_e26eede2 step 3.

These tests are gate-blocking: if any fails, the rule_recovery scorer is wrong
and no E4 benchmark result can be trusted.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import pytest

from train.logic.rule_recovery import (
    MONKS_DOMAINS,
    MONKS_RULES,
    MONKS_TOTAL_CONFIGS,
    MONKS_TOTAL_ONEHOT_BITS,
    decode_onehot_to_categorical,
    enumerate_categorical_configs,
    monks_1_rule,
    monks_2_rule,
    monks_3_rule,
    one_hot_encode_categorical,
    rule_equivalence_score,
)


class TestConstants:
    def test_domains_canonical(self):
        assert MONKS_DOMAINS == [3, 3, 2, 3, 4, 2]

    def test_total_configs_432(self):
        assert MONKS_TOTAL_CONFIGS == 432

    def test_total_onehot_17(self):
        assert MONKS_TOTAL_ONEHOT_BITS == 17


class TestEnumeration:
    def test_shape(self):
        x = enumerate_categorical_configs()
        assert x.shape == (432, 6)
        assert x.dtype == np.int64

    def test_all_unique(self):
        x = enumerate_categorical_configs()
        unique = {tuple(r) for r in x}
        assert len(unique) == 432

    def test_per_attr_in_range(self):
        x = enumerate_categorical_configs()
        for i, d in enumerate(MONKS_DOMAINS):
            assert x[:, i].min() == 0
            assert x[:, i].max() == d - 1

    def test_canonical_order_first_row_all_zeros(self):
        # itertools.product yields the all-zeros row first.
        x = enumerate_categorical_configs()
        assert (x[0] == 0).all()

    def test_canonical_order_last_row_max_indices(self):
        x = enumerate_categorical_configs()
        last_expected = np.array([d - 1 for d in MONKS_DOMAINS])
        assert (x[-1] == last_expected).all()


class TestOneHotEncoding:
    def test_shape(self):
        x_cat = enumerate_categorical_configs()
        oh = one_hot_encode_categorical(x_cat)
        assert oh.shape == (432, 17)
        assert oh.dtype == np.float32

    def test_row_sum_equals_num_attrs(self):
        x_cat = enumerate_categorical_configs()
        oh = one_hot_encode_categorical(x_cat)
        # Exactly one bit hot per attribute, 6 attrs total.
        assert (oh.sum(axis=1) == 6.0).all()

    def test_values_binary(self):
        x_cat = enumerate_categorical_configs()
        oh = one_hot_encode_categorical(x_cat)
        assert ((oh == 0.0) | (oh == 1.0)).all()

    def test_roundtrip_identity(self):
        x_cat = enumerate_categorical_configs()
        oh = one_hot_encode_categorical(x_cat)
        dec = decode_onehot_to_categorical(oh)
        assert (dec == x_cat).all()

    def test_raises_on_out_of_range(self):
        bad = np.array([[3, 0, 0, 0, 0, 0]], dtype=np.int64)  # a1=3 is out of range (domain 3 -> max 2)
        with pytest.raises(ValueError):
            one_hot_encode_categorical(bad)

    def test_raises_on_wrong_shape(self):
        bad = np.array([[0, 0, 0]], dtype=np.int64)  # only 3 attrs
        with pytest.raises(ValueError):
            one_hot_encode_categorical(bad)


class TestPublishedRuleCounts:
    """Counts of class-1 examples on the 432-config enumeration, derived
    analytically — these are the gold-standard sanity checks.
    """

    def test_monks_1_has_216_positives(self):
        # (a1==a2) OR (a5==0)
        # |a1==a2|: 3 * (2*3*4*2) = 144
        # |a5==0|:  (3*3*2*3*1*2) = 108
        # |both|:   (3*1*2*3*1*2) = 36
        # Union: 144 + 108 - 36 = 216
        x = enumerate_categorical_configs()
        y = monks_1_rule(x)
        assert int(y.sum()) == 216

    def test_monks_2_has_142_positives(self):
        # Exactly 2 of 6 attrs at value 0.
        # Sum over (i,j) pairs of prod_{k not in (i,j)} (d_k - 1)
        x = enumerate_categorical_configs()
        y = monks_2_rule(x)
        expected = 0
        for i, j in combinations(range(6), 2):
            p = 1
            for k in range(6):
                if k != i and k != j:
                    p *= (MONKS_DOMAINS[k] - 1)
            expected += p
        assert int(y.sum()) == expected == 142

    def test_monks_3_has_228_positives(self):
        # Manual computation: (a5==2 AND a4==0) OR (a5!=3 AND a2!=2)
        # |a5==2 AND a4==0|: 3*3*2*1*1*2 = 36
        # |a5!=3 AND a2!=2|: 3*2*2*3*3*2 = 216
        # |both|: a5==2 AND a4==0 AND a2!=2 (a5==2 implies a5!=3): 3*2*2*1*1*2 = 24
        # Union: 36 + 216 - 24 = 228
        x = enumerate_categorical_configs()
        y = monks_3_rule(x)
        assert int(y.sum()) == 228


class TestSelfRoundtrip:
    """If we use a rule as the predictor it MUST score exact_match=True on
    itself. This is the load-bearing correctness test.
    """

    @pytest.mark.parametrize("problem_id", [1, 2, 3])
    def test_self_roundtrip_exact(self, problem_id):
        rule = MONKS_RULES[problem_id]

        def predict_fn(x_oh: np.ndarray) -> np.ndarray:
            x_cat = decode_onehot_to_categorical(x_oh)
            return rule(x_cat).astype(np.float32)

        score = rule_equivalence_score(predict_fn, rule)
        assert score["exact_match"] is True, f"Monks-{problem_id} failed self-roundtrip"
        assert score["accuracy"] == 1.0
        assert score["hamming_distance"] == 0
        assert score["num_configs"] == 432

    @pytest.mark.parametrize("problem_id", [1, 2, 3])
    def test_self_roundtrip_confusion_matrix(self, problem_id):
        rule = MONKS_RULES[problem_id]

        def predict_fn(x_oh: np.ndarray) -> np.ndarray:
            x_cat = decode_onehot_to_categorical(x_oh)
            return rule(x_cat).astype(np.float32)

        s = rule_equivalence_score(predict_fn, rule)
        # FP and FN must be zero for a perfect predictor.
        assert s["false_positive"] == 0
        assert s["false_negative"] == 0
        # TP + TN must sum to 432.
        assert s["true_positive"] + s["true_negative"] == 432


class TestEdgeCasePredictors:
    """Constant and trivial predictors must give expected non-trivial scores."""

    def test_constant_zero_on_monks_1(self):
        pred = lambda xo: np.zeros(xo.shape[0], dtype=np.float32)
        s = rule_equivalence_score(pred, MONKS_RULES[1])
        # Monks-1 has 216 class-0 examples. Const-0 hits all of them only.
        assert s["accuracy"] == pytest.approx(216 / 432, abs=1e-9)
        assert s["true_negative"] == 216
        assert s["false_negative"] == 216
        assert s["true_positive"] == 0
        assert s["false_positive"] == 0

    def test_constant_one_on_monks_1(self):
        pred = lambda xo: np.ones(xo.shape[0], dtype=np.float32)
        s = rule_equivalence_score(pred, MONKS_RULES[1])
        assert s["accuracy"] == pytest.approx(216 / 432, abs=1e-9)

    def test_constant_zero_on_monks_2(self):
        pred = lambda xo: np.zeros(xo.shape[0], dtype=np.float32)
        s = rule_equivalence_score(pred, MONKS_RULES[2])
        # Monks-2 has 142 positives / 290 negatives. Const-0 hits 290/432.
        assert s["accuracy"] == pytest.approx(290 / 432, abs=1e-9)


class TestPredictorInputContract:
    """Verify predict_fn signature handling — 1-D vs 2-D outputs, threshold."""

    def test_handles_2d_output(self):
        # predict_fn that returns shape (M, 1)
        rule = MONKS_RULES[1]

        def predict_fn(x_oh: np.ndarray) -> np.ndarray:
            x_cat = decode_onehot_to_categorical(x_oh)
            return rule(x_cat).astype(np.float32).reshape(-1, 1)

        s = rule_equivalence_score(predict_fn, rule)
        assert s["exact_match"] is True

    def test_threshold_respected(self):
        # Predictions all at 0.4. With threshold=0.5 (default) -> all zeros.
        rule = MONKS_RULES[1]
        pred = lambda xo: np.full(xo.shape[0], 0.4, dtype=np.float32)
        s = rule_equivalence_score(pred, rule, threshold=0.5)
        # Same as const-0 above.
        assert s["true_positive"] == 0

        # With threshold=0.3 (lower) -> all ones.
        s2 = rule_equivalence_score(pred, rule, threshold=0.3)
        assert s2["false_positive"] == 216  # all class-0 mispredicted as class-1
