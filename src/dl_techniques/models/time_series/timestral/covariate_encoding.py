"""Target encoding of categorical covariates.

Timestral was trained on numeric, target-scaled covariates, so a categorical covariate
is turned into numbers before it reaches the model. The default is a per-series
smoothed mean-target encoding: in series i, category c becomes

    (smooth * mean_i + sum_{i,c}) / (smooth + count_{i,c})

over the series' context, with smooth = 1. Missing values (NaN) form their own
category. A category that never occurs in the context (e.g., a new value in the
horizon of a known covariate) resolves to the series mean.
"""

import keras
import numpy as np
from typing import Optional, Tuple


def target_encode(
    target_context,
    past_codes,
    future_codes: Optional = None,
    smooth: float = 1.0,
) -> Tuple:
    """Per-series smoothed mean-target encoding.

    Args:
        target_context: (T,) target values. Non-finite entries left out of means.
        past_codes: (T,) category codes aligned with target_context; NaN = missing.
        future_codes: Optional (H,) codes over horizon (known covariates).
        smooth: Weight of series mean in each category mean.

    Returns:
        (encoded_past, encoded_future) float32 arrays; encoded_future None if future_codes None.
    """
    # Convert to numpy for the encoding logic (runs once at forecast time)
    target = np.asarray(target_context, dtype=np.float64)
    past_codes = np.asarray(past_codes, dtype=np.float64)

    if target.shape != past_codes.shape:
        raise ValueError(f"Target {target.shape} and codes {past_codes.shape} must align")

    observed = np.isfinite(target)
    target_clean = np.where(observed, target, 0.0)
    item_count = float(observed.sum())
    item_mean = float(target_clean.sum() / item_count) if item_count > 0 else 0.0

    # Slot per distinct finite code, plus final slot for NaN
    categories = np.unique(past_codes[np.isfinite(past_codes)])
    nan_slot = len(categories)

    past_slots = _slots(past_codes, categories, nan_slot)
    sums = np.bincount(past_slots, weights=target_clean * observed, minlength=nan_slot + 1)
    counts = np.bincount(past_slots, weights=observed.astype(np.float64), minlength=nan_slot + 1)
    lookup = np.append((smooth * item_mean + sums) / (smooth + counts), item_mean)  # last: unseen

    encoded_past = lookup[past_slots].astype(np.float32)

    if future_codes is None:
        return encoded_past, None

    future_slots = _slots(
        np.asarray(future_codes, dtype=np.float64),
        categories, nan_slot, unseen=nan_slot + 1
    )
    return encoded_past, lookup[future_slots].astype(np.float32)


def _slots(codes, categories, nan_slot: int, unseen: Optional[int] = None):
    """Slot of each code in categories; NaN -> nan_slot, unknown -> unseen."""
    idx = np.searchsorted(categories, np.nan_to_num(codes, nan=0.0))
    idx = np.minimum(idx, max(len(categories) - 1, 0))
    known = np.isfinite(codes) & (len(categories) > 0)
    if len(categories) > 0:
        known &= categories[idx] == codes
    assert unseen is not None or (known | np.isnan(codes)).all(), \
        "Every past code must be in categories"
    return np.where(
        np.isnan(codes), nan_slot,
        np.where(known, idx, unseen if unseen is not None else 0)
    ).astype(np.intp)