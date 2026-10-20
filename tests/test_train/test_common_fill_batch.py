"""``train.common.datasets.fill_batch`` reads a file at most once per batch.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.4 (D-199). At
52d875f4c the attempts ran to ``3 * batch_size`` with the position taken modulo
the number of files, so a data set with fewer files than that read the same file
two or more times in one batch (3 files, batch 4: files 0, 1, 2, 0). A small
validation set is the case that reaches it: after D-197 it holds only the files
that qualify.
"""

import pytest

from train.common.datasets import fill_batch


def _fill(n_files, batch_size, idx=0, reject=()):
    read = []

    def load(i):
        read.append(i)
        return None if i in reject else i

    kept = fill_batch("Probe", idx, batch_size, list(range(n_files)), load, str, "file kept")
    return kept, read


def test_fewer_files_than_a_batch_are_each_read_once():
    kept, read = _fill(3, 4)
    assert kept == [0, 1, 2]
    assert read == [0, 1, 2]


def test_a_rejected_file_is_refilled_from_the_next_window():
    """Control: the D-126 refill is unchanged."""
    kept, read = _fill(12, 4, idx=1, reject={5})
    assert kept == [4, 6, 7, 8]
    assert read == [4, 5, 6, 7, 8]


def test_no_file_kept_raises_naming_each_file_once():
    with pytest.raises(RuntimeError, match=r"Probe batch 0: no file kept in 3 attempts; "
                                           r"tried \['0', '1', '2'\]"):
        _fill(3, 4, reject={0, 1, 2})
