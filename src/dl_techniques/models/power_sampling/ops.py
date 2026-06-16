"""Pure-numpy sampling helpers (no TensorFlow/Keras dependency).

These are the numeric primitives shared across the power-sampling engine. They
are moved verbatim from the original CliffordNet implementation so that the
algorithm semantics (log-softmax normalization, nucleus cutoff) are preserved
exactly. The underscore-prefixed names are retained to match the internal call
sites in the rest of the package; they are also exported via ``__all__`` so the
test suite can import them directly.
"""

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    shifted = logits - logits.max()
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return shifted - log_sum_exp


def _nucleus_sample(logits: np.ndarray, top_p: float) -> int:
    """Sample a token using nucleus (top-p) sampling.

    :param logits: Logits for a single position (already temperature-scaled).
    :param top_p: Cumulative probability threshold.
    :return: Sampled token ID.
    """
    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]

    # Numerically stable softmax
    probs = np.exp(sorted_logits - sorted_logits[0])
    probs /= probs.sum()

    # Find nucleus cutoff
    cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
    top_idx = sorted_idx[:cutoff]
    top_probs = probs[:cutoff]
    top_probs /= top_probs.sum()

    return int(top_idx[np.random.choice(len(top_idx), p=top_probs)])


__all__ = ["_log_softmax", "_nucleus_sample"]
