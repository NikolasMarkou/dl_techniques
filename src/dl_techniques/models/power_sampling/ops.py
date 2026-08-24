"""Pure-numpy sampling helpers (no TensorFlow/Keras dependency).

These are the numeric primitives shared across the power-sampling engine. They
were moved from the original CliffordNet implementation with the algorithm
semantics (log-softmax normalization, nucleus cutoff) preserved exactly; the one
deliberate departure is that ``_nucleus_sample`` now also *returns* the log
probability of the token it drew, because the caller cannot reconstruct that
number from the full-vocabulary log-softmax once truncation has excluded mass
(see the ``# DECISION`` note on the function). The underscore-prefixed names are
retained to match the internal call sites in the rest of the package; they are
also exported via ``__all__`` so the test suite can import them directly.
"""

from typing import Tuple

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    shifted = logits - logits.max()
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return shifted - log_sum_exp


# DECISION plan-2026-08-14T233721-d4f9beb2/D-019: this returns the log
# probability of the drawn token UNDER THE TRUNCATED, RENORMALIZED
# distribution — the one it actually sampled from. Do NOT go back to returning
# a bare token id and recovering the density at the call site with
# `_log_softmax(scaled_logits)[token_id]`: that is the full-vocabulary
# distribution, which is not the proposal whenever top-p truncation excludes
# real mass, and it is the `q(x|x')/q(x'|x)` factor of the MH acceptance ratio.
# Measured at top_p=0.5 over a linear logit ramp: the two differ by 0.602 nats.
# See decisions.md D-019.
def _nucleus_sample(logits: np.ndarray, top_p: float) -> Tuple[int, float]:
    """Sample a token using nucleus (top-p) sampling.

    :param logits: Logits for a single position (already temperature-scaled).
    :param top_p: Cumulative probability threshold.
    :return: ``(token_id, log_prob)`` — the sampled token and its log
        probability under the truncated + renormalized distribution the draw
        was made from (0 for every token outside the nucleus, so only the
        drawn token's value is ever reported).
    """
    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]

    # Numerically stable softmax
    probs = np.exp(sorted_logits - sorted_logits[0])
    probs /= probs.sum()

    # Find nucleus cutoff
    cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
    top_idx = sorted_idx[:cutoff]
    top_probs = probs[:cutoff] / probs[:cutoff].sum()

    choice = int(np.random.choice(len(top_idx), p=top_probs))
    return int(top_idx[choice]), float(np.log(top_probs[choice]))


__all__ = ["_log_softmax", "_nucleus_sample"]
