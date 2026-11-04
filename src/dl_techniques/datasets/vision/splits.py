"""The seeded train / selection / test split of the small labelled-image datasets.

MNIST, CIFAR and Imagenette select on a held-out slice of TRAIN and keep the test split
for the final report. The arithmetic and the permutation live here once, so the in-memory
loader and the Imagenette loader cannot disagree about which rows a seed picks.
"""

from typing import Optional, Tuple

import numpy as np


def split_sizes(
        n_available: int, max_samples: Optional[int], validation_split: float
) -> Tuple[int, int]:
    """Sizes of the fit and selection splits: the ONE place the split arithmetic lives.

    The loaders (from the array or record count they loaded) and a trainer's config
    validation (from the known train size, before any run directory exists) both call it,
    so an early refusal cannot disagree with the split the run will get.

    :param n_available: Samples in the train set.
    :param max_samples: The ``--max-samples`` cap, or ``None``.
    :param validation_split: Fraction of the (capped) pool held out.
    :return: ``(n_fit, n_select)``, summing to the pool.
    :raises ValueError: If the selection split would hold out zero samples.
    """
    pool = n_available if max_samples is None else min(max_samples, n_available)
    n_select = int(pool * validation_split)
    if n_select == 0:
        raise ValueError(f"validation_split={validation_split} holds out 0 of {pool} samples")
    return pool - n_select, n_select


def check_validation_split(validation_split: float) -> None:
    """Refuse a validation split outside the open interval (0, 1), naming the value."""
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            f"validation_split must be strictly inside (0, 1), got {validation_split}. "
            "0 would silently validate on the test set."
        )


def seeded_test_rows(n_test: int, seed: int) -> np.ndarray:
    """EVERY test row, in a seeded order (never capped: see :func:`seeded_split_indices`)."""
    return np.random.default_rng([seed, 1]).permutation(n_test)


def seeded_split_indices(
        n_train: int, n_test: int, seed: int, validation_split: float,
        max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of the train and test sets that make up ``(fit, selection, test)``.

    A seeded permutation of the train rows is cut into the selection slice (first) and the
    fit rows (rest); the test rows are an independent seeded permutation of the WHOLE test
    split. ``max_samples`` caps the train pool only (fit and selection together), which is what
    bounds training cost; the held-out split every reported test number is computed on is never
    shrunk by it (plan 98b866fb 7.2, D-022).

    :return: ``(train_idx, select_idx, test_idx)``, disjoint within the train set.
    :raises ValueError: A bad ``validation_split``, or one that holds out zero samples.
    """
    # DECISION plan-2026-11-03T221936-98b866fb/D-022
    # The cap bounds the TRAIN pool only. Do NOT cut the test rows by it again: that made COCO's
    # held-out test 4,000 of 5,000 pairs and put every test number, figure and analyzer pass on a
    # subset nobody asked for. A bound on evaluation cost is its own cap (Hugging Face max_eval_samples,
    # Lightning limit_test_batches), never a side effect of the training cap.
    check_validation_split(validation_split)
    train_order = np.random.default_rng(seed).permutation(n_train)
    if max_samples is not None:
        train_order = train_order[:max_samples]
    _, n_select = split_sizes(n_train, max_samples, validation_split)
    return train_order[n_select:], train_order[:n_select], seeded_test_rows(n_test, seed)
