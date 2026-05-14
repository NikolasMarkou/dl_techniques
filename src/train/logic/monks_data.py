"""
UCI Monks-1/2/3 dataset loader with one-hot encoding.

Plan: plan_2026-05-14_e26eede2 step 4 (D-002 — UCI primary, OpenML soft fallback).

Source: Thrun et al. 1991. UCI mirror at
https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/

File format (whitespace-separated, one example per line):
    class a1 a2 a3 a4 a5 a6 example_id

where ``class`` ∈ {"0", "1"} and ``a1..a6`` are 1-indexed integer values.

Standard splits (canonical Thrun et al.):
    Monks-1: 124 train / 432 test
    Monks-2: 169 train / 432 test
    Monks-3: 122 train / 432 test (5 % label noise on train only)

Module surface
--------------
- ``load_monks(problem_id) -> dict``
    Returns:
        {
            "x_train_cat": (N_tr, 6) int (0-indexed),
            "x_train_onehot": (N_tr, 17) float32,
            "y_train":  (N_tr,) int {0, 1},
            "x_test_cat":  (432, 6) int,
            "x_test_onehot": (432, 17) float32,
            "y_test":   (432,) int,
            "domains": [3,3,2,3,4,2],
            "source": "uci"|"cache",
        }

- ``CACHE_DIR`` — ``~/.cache/dl_techniques/monks/`` (auto-created on first call).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dl_techniques.utils.logger import logger
from train.logic.rule_recovery import (
    MONKS_DOMAINS,
    one_hot_encode_categorical,
)

CACHE_DIR = Path(os.path.expanduser("~/.cache/dl_techniques/monks"))
UCI_BASE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "monks-problems"
)

EXPECTED_TRAIN_SIZES = {1: 124, 2: 169, 3: 122}
EXPECTED_TEST_SIZE = 432


# ---------------------------------------------------------------------
# Low-level UCI file IO
# ---------------------------------------------------------------------

def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_uci_file(problem_id: int, split: str) -> str:
    """Download (or read from cache) the UCI monks-N.{train,test} file.

    Returns the path to the cached local copy as a string.
    """
    if problem_id not in (1, 2, 3):
        raise ValueError(f"problem_id must be 1, 2, or 3; got {problem_id}")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split}")
    _ensure_cache_dir()
    fname = f"monks-{problem_id}.{split}"
    local = CACHE_DIR / fname
    if not local.exists():
        url = f"{UCI_BASE}/monks-{problem_id}.{split}"
        logger.info(f"Downloading {url} -> {local}")
        try:
            urllib.request.urlretrieve(url, str(local))
        except Exception as e:
            # Clean up partial file if any.
            if local.exists():
                local.unlink()
            raise RuntimeError(
                f"Failed to fetch Monks-{problem_id}.{split} from UCI: {e}. "
                f"If offline, please pre-populate {local}."
            ) from e
    return str(local)


def _parse_uci_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a UCI monks-N.{train,test} file.

    Returns (x_cat, y) where:
        x_cat (N, 6) 0-indexed int (we subtract 1 from the UCI 1-indexed values)
        y     (N,)   {0, 1} int
    """
    rows_x = []
    rows_y = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 7:
                raise ValueError(
                    f"Malformed line in {path}: expected >=7 tokens, got {len(tokens)}"
                )
            cls = int(tokens[0])
            attrs = [int(t) for t in tokens[1:7]]
            rows_y.append(cls)
            rows_x.append(attrs)
    x_cat = np.asarray(rows_x, dtype=np.int64) - 1  # convert 1-indexed -> 0-indexed
    y = np.asarray(rows_y, dtype=np.int64)
    return x_cat, y


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_monks(problem_id: int) -> Dict[str, object]:
    """Load Monks-{1,2,3} train + test splits with one-hot encoding.

    Parameters
    ----------
    problem_id : int — one of {1, 2, 3}.

    Returns
    -------
    dict with keys (all numpy arrays except 'domains' and 'source'):
        x_train_cat   (N_tr, 6) int64 0-indexed
        x_train_onehot (N_tr, 17) float32
        y_train       (N_tr,) int64 in {0, 1}
        x_test_cat    (432, 6) int64 0-indexed
        x_test_onehot (432, 17) float32
        y_test        (432,) int64
        domains       [3,3,2,3,4,2]
        source        "uci" (always, currently — kept for future OpenML branch)
    """
    if problem_id not in (1, 2, 3):
        raise ValueError(f"problem_id must be 1, 2, or 3; got {problem_id}")

    train_path = _fetch_uci_file(problem_id, "train")
    test_path = _fetch_uci_file(problem_id, "test")

    x_train_cat, y_train = _parse_uci_file(train_path)
    x_test_cat, y_test = _parse_uci_file(test_path)

    expected_tr = EXPECTED_TRAIN_SIZES[problem_id]
    if x_train_cat.shape[0] != expected_tr:
        raise RuntimeError(
            f"Monks-{problem_id} train: expected {expected_tr} rows, "
            f"got {x_train_cat.shape[0]}"
        )
    if x_test_cat.shape[0] != EXPECTED_TEST_SIZE:
        raise RuntimeError(
            f"Monks-{problem_id} test: expected {EXPECTED_TEST_SIZE} rows, "
            f"got {x_test_cat.shape[0]}"
        )

    # Validate value ranges.
    for col, d in enumerate(MONKS_DOMAINS):
        for name, arr in (("train", x_train_cat), ("test", x_test_cat)):
            if arr[:, col].min() < 0 or arr[:, col].max() >= d:
                raise RuntimeError(
                    f"Monks-{problem_id} {name}: attr {col} out of range [0,{d}); "
                    f"min={arr[:, col].min()} max={arr[:, col].max()}"
                )

    x_train_oh = one_hot_encode_categorical(x_train_cat)
    x_test_oh = one_hot_encode_categorical(x_test_cat)

    return {
        "x_train_cat": x_train_cat,
        "x_train_onehot": x_train_oh,
        "y_train": y_train,
        "x_test_cat": x_test_cat,
        "x_test_onehot": x_test_oh,
        "y_test": y_test,
        "domains": list(MONKS_DOMAINS),
        "source": "uci",
    }
