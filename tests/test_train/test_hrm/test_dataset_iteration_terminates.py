"""`HRMTrainer`'s epoch loops must be bounded by `len(dataset)`.

`keras.utils.Sequence` IS `keras.utils.PyDataset` in Keras 3, and `PyDataset`
defines **no** `__iter__`. So `for batch in <a Sequence>` falls back to Python's
legacy `__getitem__` protocol, which walks index 0, 1, 2, ... until an
`IndexError` and **ignores `__len__` entirely**. `HRMTrainer` annotates both its
datasets as `keras.utils.Sequence`, and this package's own `SampleDataset`
generates fresh random data for any index and never raises -- so both epoch
loops ran forever. Nothing raised; it presented as a hang with no failure text,
which is why a grep was the only cheap instrument for finding it.

These tests drive the loops directly with a counting stub, bypassing
`HRMTrainer.__init__` (which would build a full HRM model) via
`object.__new__`. That is deliberate: the property under test is the loop bound,
not the model.
"""

import keras
import pytest

from train.hrm.train_hrm import HRMTrainer


N_BATCHES = 4
# A bound comfortably above `len()` but far below "forever", so a regression
# fails fast instead of hanging the suite.
RUNAWAY_LIMIT = 50


class CountingSequence(keras.utils.Sequence):
    """A `Sequence` that never raises `IndexError` -- like the real one.

    `SampleDataset.__getitem__` in `train_hrm.py` synthesizes random arrays for
    any index, so it has no natural terminator either. Reproducing that here is
    the whole point: a stub that *did* raise would hide the defect.
    """

    def __init__(self, n: int = N_BATCHES):
        self.n = n
        self.accessed = []

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.accessed.append(idx)
        if len(self.accessed) > RUNAWAY_LIMIT:
            raise AssertionError(
                f"dataset walked past index {idx} with len()=={self.n}: the "
                "epoch loop is unbounded (PyDataset has no __iter__)"
            )
        return {"batch": idx}


class _NoOpMetrics:
    def reset_state(self):
        pass


def _bare_trainer(train_ds=None, val_ds=None):
    """An `HRMTrainer` with only the attributes the epoch loops touch."""
    t = object.__new__(HRMTrainer)
    t.train_dataset = train_ds
    t.val_dataset = val_ds
    t.metrics = _NoOpMetrics()
    t.current_epoch = 0
    return t


class TestEpochLoopsAreBounded:

    def test_train_epoch_visits_each_index_exactly_once(self):
        ds = CountingSequence()
        t = _bare_trainer(train_ds=ds)
        t.train_step = lambda batch: {"loss": 0.0, "accuracy": 0.0,
                                      "exact_accuracy": 0.0}

        t.train_epoch()

        assert ds.accessed == list(range(N_BATCHES)), ds.accessed

    def test_evaluate_visits_each_index_exactly_once(self):
        ds = CountingSequence()
        t = _bare_trainer(val_ds=ds)
        t.evaluate_step = lambda batch: {"loss": 0.0}

        t.evaluate()

        assert ds.accessed == list(range(N_BATCHES)), ds.accessed

    def test_evaluate_with_no_val_dataset_is_a_noop(self):
        t = _bare_trainer(val_ds=None)
        assert t.evaluate() == {}


class TestTheFrameworkPremise:
    """Pin the Keras behaviour the fix is built on, so a version bump speaks."""

    def test_sequence_is_pydataset_and_has_no_iter(self):
        assert keras.utils.Sequence is keras.utils.PyDataset
        assert not hasattr(keras.utils.Sequence, "__iter__")

    def test_bare_iteration_ignores_len(self):
        import itertools

        ds = CountingSequence(n=3)
        walked = list(itertools.islice(iter(ds), 10))
        assert len(walked) == 10, (
            "iterating a PyDataset stopped at __len__ -- if Keras added "
            "__iter__, the range() workaround in train_hrm.py can be revisited"
        )
