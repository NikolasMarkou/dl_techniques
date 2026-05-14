"""
Tests for ``train.logic.monks_data`` UCI loader.

Plan: plan_2026-05-14_e26eede2 step 4.

Network test marked — runs only if UCI is reachable. The first run downloads
all 6 UCI files (~5 KB each) into ``~/.cache/dl_techniques/monks/``; subsequent
runs hit the cache.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from train.logic.monks_data import (
    EXPECTED_TEST_SIZE,
    EXPECTED_TRAIN_SIZES,
    load_monks,
)
from train.logic.rule_recovery import (
    MONKS_DOMAINS,
    MONKS_RULES,
    decode_onehot_to_categorical,
)


def _network_available() -> bool:
    try:
        socket.create_connection(("archive.ics.uci.edu", 443), timeout=5).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _network_available(),
    reason="UCI mirror not reachable; loader requires network on first call",
)


@pytest.mark.parametrize("problem_id", [1, 2, 3])
class TestLoadMonks:
    def test_shapes(self, problem_id):
        d = load_monks(problem_id)
        n_tr = EXPECTED_TRAIN_SIZES[problem_id]
        assert d["x_train_cat"].shape == (n_tr, 6)
        assert d["x_train_onehot"].shape == (n_tr, 17)
        assert d["y_train"].shape == (n_tr,)
        assert d["x_test_cat"].shape == (EXPECTED_TEST_SIZE, 6)
        assert d["x_test_onehot"].shape == (EXPECTED_TEST_SIZE, 17)
        assert d["y_test"].shape == (EXPECTED_TEST_SIZE,)
        assert d["domains"] == [3, 3, 2, 3, 4, 2]
        assert d["source"] == "uci"

    def test_dtypes(self, problem_id):
        d = load_monks(problem_id)
        assert d["x_train_cat"].dtype == np.int64
        assert d["x_train_onehot"].dtype == np.float32
        assert d["y_train"].dtype == np.int64

    def test_label_values_binary(self, problem_id):
        d = load_monks(problem_id)
        assert set(np.unique(d["y_train"]).tolist()).issubset({0, 1})
        assert set(np.unique(d["y_test"]).tolist()).issubset({0, 1})

    def test_attr_value_ranges(self, problem_id):
        d = load_monks(problem_id)
        for col, dom in enumerate(MONKS_DOMAINS):
            assert d["x_train_cat"][:, col].min() >= 0
            assert d["x_train_cat"][:, col].max() < dom
            assert d["x_test_cat"][:, col].min() >= 0
            assert d["x_test_cat"][:, col].max() < dom

    def test_onehot_row_sum_is_6(self, problem_id):
        d = load_monks(problem_id)
        assert (d["x_train_onehot"].sum(axis=1) == 6.0).all()
        assert (d["x_test_onehot"].sum(axis=1) == 6.0).all()

    def test_onehot_decodes_back_to_cat(self, problem_id):
        d = load_monks(problem_id)
        dec_tr = decode_onehot_to_categorical(d["x_train_onehot"])
        dec_te = decode_onehot_to_categorical(d["x_test_onehot"])
        assert (dec_tr == d["x_train_cat"]).all()
        assert (dec_te == d["x_test_cat"]).all()


class TestMonks1TestSetMatchesPublishedRule:
    """The Monks-1 test set IS the full 432-config enumeration. Its labels
    must exactly match the published rule applied to those configs.
    """

    def test_test_labels_match_rule(self):
        d = load_monks(1)
        y_pred = MONKS_RULES[1](d["x_test_cat"])
        assert (y_pred == d["y_test"]).all(), (
            "UCI Monks-1 test labels disagree with the published rule "
            "applied to the test inputs — encoding mismatch."
        )


class TestMonks2TestSetMatchesPublishedRule:
    def test_test_labels_match_rule(self):
        d = load_monks(2)
        y_pred = MONKS_RULES[2](d["x_test_cat"])
        assert (y_pred == d["y_test"]).all(), (
            "UCI Monks-2 test labels disagree with the published rule."
        )


class TestMonks3TestSetMatchesPublishedRule:
    """Monks-3 has 5 % noise on TRAIN. Test set is noise-free and should
    match the published rule exactly.
    """

    def test_test_labels_match_rule(self):
        d = load_monks(3)
        y_pred = MONKS_RULES[3](d["x_test_cat"])
        assert (y_pred == d["y_test"]).all(), (
            "UCI Monks-3 test labels disagree with the published rule."
        )


class TestMonks3TrainHasLabelNoise:
    """Sanity: the canonical Monks-3 train set should have some examples
    whose UCI label disagrees with the noise-free published rule (the
    canonical noise fraction is 5 % = ~6 rows of 122).
    """

    def test_some_train_labels_disagree_with_rule(self):
        d = load_monks(3)
        y_pred = MONKS_RULES[3](d["x_train_cat"])
        disagree = (y_pred != d["y_train"]).sum()
        # 5% of 122 = ~6. Accept any non-zero amount up to 15% as plausible.
        assert disagree >= 1
        assert disagree <= 20


class TestInvalidProblemId:
    def test_problem_id_0_raises(self):
        with pytest.raises(ValueError):
            load_monks(0)

    def test_problem_id_4_raises(self):
        with pytest.raises(ValueError):
            load_monks(4)
