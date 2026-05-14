"""
Rule-recovery scorer for UCI Monks-1/2/3.

Plan: plan_2026-05-14_e26eede2 (E4 — UCI Monks rule recovery).

The scientific question: given a trained ``LearnableNeuralCircuit`` model that
classifies one-hot-encoded Monks examples, does it recover the *published
ground-truth rule* that generated the data?

Approach (D-001 of plan_2026-05-14_e26eede2): truth-table equivalence on the
finite categorical domain of Monks (3·3·2·3·4·2 = 432 valid configurations).
We enumerate all 432 configurations, one-hot encode each, query the model for a
prediction, and compare to the published rule's ground-truth label on the same
configurations.

Published rules (Thrun et al. 1991, "The MONK's Problems"):

    Monks-1: (a1 == a2) OR (a5 == "red")
    Monks-2: exactly two of {a1, ..., a6} are at their first value
    Monks-3: (a5 == "green" AND a4 == "sword") OR
             (a5 != "blue" AND a2 != "octagon")
             with 5 % label noise on the training set only.

In the UCI files the attribute values are 1-indexed integers. We work
exclusively in 0-indexed integers within this module (subtract 1 from any
1-indexed input). The encoders below operate on ``x_cat (N, 6)`` arrays of
0-indexed integers.

Module surface
--------------
- ``MONKS_DOMAINS``                 — list of attribute cardinalities (length 6)
- ``MONKS_TOTAL_ONEHOT_BITS``       — sum of MONKS_DOMAINS (17)
- ``MONKS_TOTAL_CONFIGS``           — product of MONKS_DOMAINS (432)
- ``enumerate_categorical_configs`` — all 432 valid (a1..a6) tuples as (432, 6)
- ``one_hot_encode_categorical``    — (N, 6) int → (N, 17) float32 one-hot
- ``monks_1_rule(x_cat)``           — (N, 6) → (N,) {0, 1} ground-truth
- ``monks_2_rule(x_cat)``
- ``monks_3_rule(x_cat)``
- ``MONKS_RULES``                   — dict {1: monks_1_rule, 2: ..., 3: ...}
- ``rule_equivalence_score(predict_fn, rule_fn, threshold=0.5) -> dict``
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List

import numpy as np

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Attribute cardinalities in canonical order a1..a6.
# a1 head_shape ∈ {round, square, octagon} (3)
# a2 body_shape ∈ {round, square, octagon} (3)
# a3 is_smiling ∈ {yes, no} (2)
# a4 holding   ∈ {sword, balloon, flag} (3)
# a5 jacket    ∈ {red, yellow, green, blue} (4)
# a6 has_tie   ∈ {yes, no} (2)
MONKS_DOMAINS: List[int] = [3, 3, 2, 3, 4, 2]
MONKS_TOTAL_ONEHOT_BITS: int = sum(MONKS_DOMAINS)  # 17
MONKS_TOTAL_CONFIGS: int = int(np.prod(MONKS_DOMAINS))  # 432

# Per-attr starting index in the concatenated one-hot vector.
_ONEHOT_OFFSETS: List[int] = [0]
for _d in MONKS_DOMAINS[:-1]:
    _ONEHOT_OFFSETS.append(_ONEHOT_OFFSETS[-1] + _d)


# ---------------------------------------------------------------------
# Enumeration + encoding
# ---------------------------------------------------------------------

def enumerate_categorical_configs() -> np.ndarray:
    """Return every valid (a1, a2, a3, a4, a5, a6) tuple as a (432, 6) int array.

    Values are 0-indexed.
    """
    rows = list(itertools.product(*[range(d) for d in MONKS_DOMAINS]))
    return np.asarray(rows, dtype=np.int64)


def one_hot_encode_categorical(x_cat: np.ndarray, domains: List[int] = None) -> np.ndarray:
    """One-hot encode a (N, 6) categorical-int array to (N, 17) float32.

    Parameters
    ----------
    x_cat : (N, 6) int array, 0-indexed values.
    domains : per-attr cardinality list. Defaults to MONKS_DOMAINS.

    Returns
    -------
    (N, sum(domains)) float32 one-hot matrix; each attr's one-hot slice is
    contiguous along the last axis, in canonical attr order.
    """
    if domains is None:
        domains = MONKS_DOMAINS
    if x_cat.ndim != 2 or x_cat.shape[1] != len(domains):
        raise ValueError(
            f"x_cat shape {x_cat.shape} incompatible with domains {domains}"
        )
    n = x_cat.shape[0]
    total = sum(domains)
    out = np.zeros((n, total), dtype=np.float32)
    offset = 0
    for i, d in enumerate(domains):
        col = x_cat[:, i].astype(np.int64)
        if (col < 0).any() or (col >= d).any():
            raise ValueError(
                f"attr {i}: value out of range [0, {d}); got min={col.min()} max={col.max()}"
            )
        out[np.arange(n), offset + col] = 1.0
        offset += d
    return out


def decode_onehot_to_categorical(x_oh: np.ndarray, domains: List[int] = None) -> np.ndarray:
    """Inverse of one_hot_encode_categorical (per-attr argmax)."""
    if domains is None:
        domains = MONKS_DOMAINS
    n = x_oh.shape[0]
    out = np.zeros((n, len(domains)), dtype=np.int64)
    offset = 0
    for i, d in enumerate(domains):
        out[:, i] = np.argmax(x_oh[:, offset:offset + d], axis=1)
        offset += d
    return out


# ---------------------------------------------------------------------
# Published Monks rules (operate on 0-indexed x_cat)
# ---------------------------------------------------------------------

def monks_1_rule(x_cat: np.ndarray) -> np.ndarray:
    """Monks-1: (a1 == a2) OR (a5 == 'red').

    In 0-indexed: (x_cat[:, 0] == x_cat[:, 1]) | (x_cat[:, 4] == 0)
    (jacket_color 'red' is value 1 in the UCI 1-indexed encoding, so 0 in
    our 0-indexed encoding.)
    """
    return ((x_cat[:, 0] == x_cat[:, 1]) | (x_cat[:, 4] == 0)).astype(np.int64)


def monks_2_rule(x_cat: np.ndarray) -> np.ndarray:
    """Monks-2: exactly two of {a1..a6} are at their *first value*.

    In 0-indexed: exactly two of x_cat[:, i] == 0.
    """
    return ((x_cat == 0).sum(axis=1) == 2).astype(np.int64)


def monks_3_rule(x_cat: np.ndarray) -> np.ndarray:
    """Monks-3: (a5 == 'green' AND a4 == 'sword') OR
                (a5 != 'blue' AND a2 != 'octagon').

    In 0-indexed (mapping per UCI categories):
        a5 jacket: 1=red, 2=yellow, 3=green, 4=blue  (1-indexed)
                ↦ 0=red, 1=yellow, 2=green, 3=blue  (0-indexed)
        a4 holding: 1=sword, 2=balloon, 3=flag      (1-indexed)
                  ↦ 0=sword, 1=balloon, 2=flag       (0-indexed)
        a2 body_shape: 1=round, 2=square, 3=octagon (1-indexed)
                     ↦ 0, 1, 2                       (0-indexed)
    So:
        (x_cat[:, 4] == 2) & (x_cat[:, 3] == 0)
        |
        (x_cat[:, 4] != 3) & (x_cat[:, 1] != 2)

    The 5 % label noise applies to the TRAINING data only and is preserved by
    the UCI dataset files. The published rule itself is noise-free; that's the
    rule we score against.
    """
    left = (x_cat[:, 4] == 2) & (x_cat[:, 3] == 0)
    right = (x_cat[:, 4] != 3) & (x_cat[:, 1] != 2)
    return (left | right).astype(np.int64)


MONKS_RULES: Dict[int, Callable[[np.ndarray], np.ndarray]] = {
    1: monks_1_rule,
    2: monks_2_rule,
    3: monks_3_rule,
}


# ---------------------------------------------------------------------
# Equivalence scoring
# ---------------------------------------------------------------------

def rule_equivalence_score(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    rule_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float = 0.5,
    domains: List[int] = None,
) -> Dict[str, float]:
    """Compute rule-recovery score by truth-table enumeration.

    Parameters
    ----------
    predict_fn : callable taking (M, sum(domains)) float32 one-hot input and
        returning either (M,) or (M, 1) float predictions in [0, 1] (probability
        of class 1). The function will threshold at ``threshold`` to get a
        binary label.
    rule_fn : callable taking (M, len(domains)) int categorical and returning
        (M,) int ground-truth labels in {0, 1}.
    threshold : float — threshold to binarize predict_fn output.
    domains : per-attribute cardinalities; defaults to MONKS_DOMAINS.

    Returns
    -------
    dict with keys:
        - ``exact_match`` (bool): all M predictions match ground truth
        - ``accuracy`` (float): fraction of matching predictions
        - ``hamming_distance`` (int): number of mismatches (0..M)
        - ``num_configs`` (int): M, the enumeration size
        - ``true_positive`` / ``true_negative`` / ``false_positive``
          / ``false_negative`` (int): confusion-matrix counts
        - ``num_class_0_gt`` / ``num_class_1_gt`` (int): ground-truth class
          counts in the enumeration (for sanity).
    """
    if domains is None:
        domains = MONKS_DOMAINS
    x_cat = enumerate_categorical_configs() if domains == MONKS_DOMAINS else (
        np.asarray(list(itertools.product(*[range(d) for d in domains])), dtype=np.int64)
    )
    x_oh = one_hot_encode_categorical(x_cat, domains)

    y_true = np.asarray(rule_fn(x_cat), dtype=np.int64).reshape(-1)
    if y_true.shape[0] != x_cat.shape[0]:
        raise ValueError(
            f"rule_fn returned {y_true.shape[0]} labels for {x_cat.shape[0]} configs"
        )

    raw_pred = np.asarray(predict_fn(x_oh), dtype=np.float32).reshape(-1)
    if raw_pred.shape[0] != x_oh.shape[0]:
        raise ValueError(
            f"predict_fn returned {raw_pred.shape[0]} preds for {x_oh.shape[0]} inputs"
        )
    y_pred = (raw_pred > threshold).astype(np.int64)

    match = (y_pred == y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    return {
        "exact_match": bool(match.all()),
        "accuracy": float(match.mean()),
        "hamming_distance": int((~match).sum()),
        "num_configs": int(x_cat.shape[0]),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "num_class_0_gt": int((y_true == 0).sum()),
        "num_class_1_gt": int((y_true == 1).sum()),
    }
