"""Corpus-level generation metrics: BLEU, ROUGE, chrF, METEOR and CIDEr.

Every metric here answers "how much of what you produced matches what a human
wrote", and all five answer it through the same three steps -- tokenize, count
n-grams, score the overlap. What differs is the *reduction*: which order of
n-gram is counted, whether recall is weighted above precision, whether word
order costs anything, whether a match must be exact, and whether a common n-gram
counts the same as a rare one.

**Plain functions on lists of strings, not ``keras.metrics.Metric``
subclasses.** Deliberate, and structural rather than a shortcut. A streaming
``update_state`` sees one candidate at a time, and three of these five cannot be
computed that way at all: CIDEr's IDF weights are a functional of the whole
reference corpus, METEOR's score is per-sentence and may only be averaged once
each sentence's fragmentation penalty has been applied to it, and BLEU's brevity
penalty needs the corpus-wide candidate/reference length ratio rather than a
per-batch mean. The package already carries plain functions for exactly this
reason (``embedding_quality``, ``pass_at_k``, ``llm_metrics.self_bleu``,
``perplexity_metric.perplexity``).

Choosing between these five, stated once so nobody double-counts evidence:

- **BLEU** is *precision-only over 4-grams*, with recall admitted only through
  the brevity penalty. A candidate that is right but phrased differently scores
  near zero; a candidate that repeats the reference's n-grams in nonsense order
  scores well. It is the weakest signal here and the one most often reported.
- **chrF** is BLEU's answer to morphology: character n-grams instead of word
  n-grams, so inflectional variants and cognates share prefixes. It is
  tokenization-agnostic, which is most of why WMT now runs it beside BLEU.
- **ROUGE** is *recall-leaning* and is the summarization default. ROUGE-1/2 are
  recall-oriented bags of n-grams; ROUGE-L adds an LCS term so word order costs
  something.
- **METEOR** is the only metric here that scores word order through a
  fragmentation penalty, and the only one whose F-mean is a precision-weighted
  harmonic mean rather than a geometric one.
- **CIDEr** is the only metric here that weights rare n-grams above common ones
  (TF-IDF). It is an image-captioning metric; its IDF term is close to
  meaningless on a corpus whose references share a small vocabulary, and it will
  still return a confident number there.

They are not redundant, and they disagree. Report the one your field reports,
plus at most one orthogonal axis -- BLEU + chrF, or ROUGE + METEOR -- and say
which, and do not average them together.

Two deliberate limitations, both consequences of this repository having no NLP
dependency (``llm_metrics`` states the same constraint for the same reason --
neither NLTK nor SacreBLEU is in ``pyproject.toml``):

- **These are not SacreBLEU numbers.** SacreBLEU pins tokenization, casing and
  smoothing into a reproducible signature string
  (``BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+tok.13a+version.2.x``).
  :func:`corpus_bleu` pins smoothing only, and
  :func:`whitespace_tokenize` is not SacreBLEU's 13a normalizer. Quoting a
  corpus-BLEU from this module beside a published WMT figure compares two
  different measurements. ``research/benchmarks/METRICS.md`` section BLEU says
  to always quote the signature, for exactly this reason.
- **The default METEOR is exact-match.** The reference implementation matches
  through Porter stemming and WordNet synonyms. :func:`meteor` accepts a
  ``match_fn`` hook for that, but its default aligns identical tokens only, so
  it scores LOWER than a stemmed METEOR on the same input and is not comparable
  to published METEOR numbers. :func:`meteor`'s docstring quantifies the gap.

Scaling: every function returns an unscaled float in ``[0.0, 1.0]`` except
:func:`cider`, which returns the ``10 x mean_n`` form its reference defines and
so lives in ``[0.0, 10.0]``. The ``* 100`` / ``* 10`` reporting conventions live
in :func:`generation_report`, never in the metric functions -- matching
``distinct_n``, ``self_bleu`` and ``mrr_at_k``, which are all unscaled.

References:
    - Papineni, K. et al. (2002). BLEU: a Method for Automatic Evaluation of
      Machine Translation. (https://aclanthology.org/P02-1040/)
    - Chen, S. & Cherry, L. (2014). A Systematic Comparison of the BLEU, Smoothing
      Methods Including a Novel BLEU Implementation. (https://aclanthology.org/W14-3346/)
    - Post, M. (2018). A Call for Clarity in Reporting BLEU Scores.
      (https://arxiv.org/abs/1804.08771) (the smoothing and signature conventions
      followed here)
    - Lin, C.-Y. (2004). ROUGE: a Package for Automatic Evaluation of Summaries.
      (the LCS-based ROUGE-L formulation and the ``beta``-weighted F measure)
    - Popovic, M. (2015). chrF: character n-gram F-score for Automatic MT
      Evaluation. (https://aclanthology.org/W15-3047/)
    - Banerjee, S. & Lavie, A. (2005). METEOR: an Automatic Metric for MT
      Evaluation with Improved Correlation with Human Judgments.
      (https://aclanthology.org/W05-0310/) (the 9:1 precision/recall weighting
      and the fragmentation penalty)
    - Vedantam, K. et al. (2015). CIDEr: Consensus-based Image Description
      Evaluation. (https://arxiv.org/abs/1411.5726) (the TF-IDF n-gram cosine)
    - Vedantam, K. et al. (2016). Consensus Evaluation for Image Captioning.
      (https://arxiv.org/abs/1603.07259) (the CIDEr-D length-difference penalty)
"""

import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

#: A tokenizer maps one string to its token sequence. Nothing in this module
#: inspects a token beyond comparing it for equality or grouping it, so any
#: callable with this shape works -- including one backed by a real MT
#: tokenizer, which is how a caller recovers comparability with published
#: numbers.
Tokenizer = Callable[[str], List[str]]

#: A matcher maps one token to the SET of surface forms it is equivalent to. It
#: must include the token itself; :func:`meteor` intersects candidate and
#: reference forms, so a matcher that omits the identity form silently drops
#: every exact match it was supposed to keep.
Matcher = Callable[[str], frozenset]

#: Supported BLEU smoothing methods, transcribed from SacreBLEU's four. Their
#: mapping onto the numbered methods in ``research/benchmarks/METRICS.md``:
#: ``none`` is its method 0, ``floor`` its method 1, ``add_k`` its method 2, and
#: ``exp`` its method 3 -- which that document also calls "NIST geometric
#: sequence smoothing", the two names for the same ``mteval-v13a.pl`` rule. One
#: name per operation, so the alias is not a fifth method.
BLEU_SMOOTHING_METHODS: Tuple[str, ...] = ("none", "exp", "floor", "add_k")

#: Supported ROUGE-N score selectors.
ROUGE_SCORE_TYPES: Tuple[str, ...] = ("fmeasure", "recall", "precision")

#: Supported CIDEr variants.
CIDER_VARIANTS: Tuple[str, ...] = ("d", "c")

#: Default maximum n-gram order for BLEU (the metric's ``n = 1..4``).
DEFAULT_BLEU_MAX_ORDER: int = 4

#: Default maximum character n-gram order for chrF.
DEFAULT_CHRF_CHAR_ORDER: int = 6

#: Default maximum word n-gram order for chrF (``0`` disables the word half, so
#: plain chrF; ``2`` gives chrF++).
DEFAULT_CHRF_WORD_ORDER: int = 0

#: Default recall weight for chrF. WMT reports ``chrF`` at ``beta = 2``.
DEFAULT_CHRF_BETA: float = 2.0

#: Default recall weight for ROUGE-L. ``beta = 1`` is the plain harmonic mean;
#: Lin's original ROUGE-L used ``beta -> inf``, i.e. recall only.
DEFAULT_ROUGE_L_BETA: float = 1.0

#: Default METEOR recall weight. The reference's ``alpha = 0.9, beta = 3.0``
#: pair -- see the ``_METEOR`` decision note on why this module exposes the
#: weight and not ``alpha``.
DEFAULT_METEOR_BETA: float = 3.0

#: Default METEOR fragmentation-penalty coefficient.
DEFAULT_METEOR_GAMMA: float = 0.5

#: Default CIDEr standard deviation for the Gaussian length penalty.
DEFAULT_CIDER_SIGMA: float = 6.0

# DECISION <gen-metrics-2026-10-18>/D-001
# The floor-smoothing delta and the add-k value are BOTH 0.1 / 1.0 by the
# Chen & Cherry defaults, and they are named constants because they are
# DIFFERENT operations that a reader will otherwise merge into "an epsilon":
#   floor  -- p_n = delta / total_n            (a floor on the numerator)
#   add-k  -- correct_n += k and total_n += k  (Chen & Cherry method 2)
# Neither touches a precision that is already non-zero, which is the property
# that makes them smoothing rather than a different metric: a candidate with no
# missing 4-gram gets exactly the same score under every method.
FLOOR_SMOOTHING_DELTA: float = 0.1
ADD_K_SMOOTHING_K: float = 1.0

# DECISION <gen-metrics-2026-10-18>/D-002
# METEOR exposes `beta`, NOT the reference paper's `alpha`. The paper writes
#     F_mean = P*R / (alpha*P + (1-alpha)*R)     with alpha = 0.9, beta = 3.0
# where its `beta` is the FRAGMENTATION exponent, and the "10PR/(R + 9P)" form
# that everyone quotes is that same F_mean rewritten. The two numbers are not
# independent parameters: with F_beta = (1 + beta^2)PR / (beta^2 P + R), the
# reference's alpha IS this F measure at beta = 3, since
#     alpha = beta^2 / (1 + beta^2) = 9/10 = 0.9   at beta = 3.
# Exposing both would ship a second knob that cannot move the score without
# moving the first -- a dead knob in the section 12.5 sense. ONE `beta` is
# exposed, it is used in BOTH places the reference uses it (the F-mean
# weighting, squared, and the fragmentation exponent, cubed), and the identity
# above is pinned numerically in
# tests/test_metrics/test_generation_metrics.py rather than left as prose.

__all__ = [
    "BLEU_SMOOTHING_METHODS",
    "CIDER_VARIANTS",
    "DEFAULT_BLEU_MAX_ORDER",
    "DEFAULT_CHRF_BETA",
    "DEFAULT_CHRF_CHAR_ORDER",
    "DEFAULT_CHRF_WORD_ORDER",
    "DEFAULT_CIDER_SIGMA",
    "DEFAULT_METEOR_BETA",
    "DEFAULT_METEOR_GAMMA",
    "DEFAULT_ROUGE_L_BETA",
    "GENERATION_METRIC_REGISTRY",
    "ROUGE_SCORE_TYPES",
    "bleu",
    "chrf",
    "cider",
    "corpus_bleu",
    "corpus_rouge_l",
    "corpus_rouge_n",
    "create_generation_metric",
    "generation_report",
    "meteor",
    "ngrams",
    "rouge_l",
    "rouge_lsum",
    "rouge_n",
    "sentence_bleu",
    "whitespace_tokenize",
    "split_sentences",
]


# ---------------------------------------------------------------------
# Tokenization and n-gram counting -- the shared core
# ---------------------------------------------------------------------


def whitespace_tokenize(text: str) -> List[str]:
    """Split on whitespace. The default tokenizer.

    Deliberately the crudest tokenizer available. These metrics are all
    tokenizer-dependent, and the honest default is the one whose dependence is
    obvious rather than hidden inside a normalization rule set.

    Args:
        text: Any string. Empty or whitespace-only input returns ``[]``.

    Returns:
        The whitespace-separated tokens, in order.
    """
    if not text:
        return []
    return text.split()


#: Sentence splitter used by :func:`rouge_lsum`. Splits after ``.``, ``!`` or
#: ``?`` followed by whitespace. This is NOT ``rouge_score``'s tokenizer-based
#: splitter, and the two disagree on abbreviations and on text with no terminal
#: punctuation at all; :func:`rouge_lsum`'s docstring says so.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences on terminal punctuation.

    Used only by :func:`rouge_lsum`. Abbreviations are not handled -- "Dr. Smith
    arrived." is two sentences by this rule -- because handling them needs a
    tokenizer this module deliberately does not depend on. Pass your own
    splitter to :func:`rouge_lsum` if that matters for your corpus.

    Args:
        text: Any string.

    Returns:
        A list of non-empty stripped sentences; ``[]`` for empty input.
    """
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Every contiguous ``n``-gram of ``tokens``, as tuples, in order.

    Returns a *list* rather than a ``Counter`` because callers need the order --
    METEOR's chunk scan reads positions and CIDEr's length penalty reads a
    total. The aggregation callers build a counter with :func:`_ngram_counts`
    over this.

    Args:
        tokens: A token sequence.
        n: N-gram order. Must be a positive integer.

    Returns:
        A list of ``len(tokens) - n + 1`` tuples, or ``[]`` when the token
        sequence is shorter than ``n``.

    Raises:
        ValueError: If ``n`` is not a positive integer.
    """
    n_int = _validate_positive_int(n, "n")
    if len(tokens) < n_int:
        return []
    return [tuple(tokens[i:i + n_int]) for i in range(len(tokens) - n_int + 1)]


def _ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    """Multiset of the token sequence's ``n``-grams.

    Uses ``setdefault`` rather than ``collections.Counter`` to keep the
    module's import surface at numpy.

    Args:
        tokens: A token sequence.
        n: N-gram order. Must be a positive integer.

    Returns:
        A ``dict`` mapping each distinct ``n``-gram to its multiplicity.
    """
    counts: Dict[Tuple[str, ...], int] = {}
    for gram in ngrams(tokens, n):
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _order_counts(
        cand_tokens: Sequence[str],
        ref_token_lists: Sequence[Sequence[str]],
        order: int,
) -> Tuple[int, int, List[int]]:
    """Clipped overlap and totals for one n-gram order.

    The primitive every n-gram metric here reduces to. The clip is what makes
    these metrics precision-with-recall-admitted rather than raw intersection: a
    candidate repeating one reference 4-gram five times scores five against a
    reference containing it once.

    The overlap takes the **max** over references per n-gram, never the sum.
    Summing would let ``k`` references between them cover ``k`` times the
    candidate's n-grams and drive precision above 1.

    Args:
        cand_tokens: The candidate's tokens.
        ref_token_lists: One token list per reference.
        order: N-gram order.

    Returns:
        ``(clipped, candidate_total, reference_totals)`` -- the clipped overlap,
        the candidate's own n-gram count, and each reference's n-gram count.
    """
    cand_counts = _ngram_counts(cand_tokens, order)
    ref_counts = [_ngram_counts(ref_tokens, order) for ref_tokens in ref_token_lists]

    clipped = 0
    for gram, cand_count in cand_counts.items():
        best = 0
        for ref_count in ref_counts:
            count = ref_count.get(gram, 0)
            if count > best:
                best = count
        if best:
            clipped += min(cand_count, best)

    return clipped, sum(cand_counts.values()), [sum(c.values()) for c in ref_counts]


def _closest_ref_length(candidate_length: int, ref_lengths: Sequence[int]) -> int:
    """The reference length nearest the candidate's -- BLEU's effective reference.

    With multiple references there is no single right length and the choice moves
    the brevity penalty. The standard is the one closest to the candidate
    length, ties going to the SHORTER reference.

    Args:
        candidate_length: The candidate's token count.
        ref_lengths: Each reference's token count.

    Returns:
        The selected effective reference length.

    Raises:
        ValueError: If ``ref_lengths`` is empty.
    """
    if not ref_lengths:
        raise ValueError("ref_lengths must not be empty")
    return min(ref_lengths, key=lambda length: (abs(length - candidate_length), length))


def _brevity_penalty(candidate_length: int, reference_length: int) -> float:
    """BLEU's brevity penalty.

    .. math::

        BP = \\begin{cases} 1 & c > r \\\\ e^{1 - r/c} & c \\le r
        \\end{cases}

    The two branches agree at ``c == r``, which is why the comparison is ``>``
    and not ``>=`` -- an exactly-as-long candidate is not penalised.

    An empty candidate returns ``0.0`` rather than raising or producing a NaN
    from ``r / 0``: an empty output is a score of zero, not an error.

    Args:
        candidate_length: Candidate token count ``c``.
        reference_length: Effective reference token count ``r``.

    Returns:
        ``BP`` in ``[0.0, 1.0]``.
    """
    if candidate_length == 0:
        return 0.0
    if candidate_length > reference_length:
        return 1.0
    return float(np.exp(1.0 - reference_length / candidate_length))


def _smoothed_precisions(
        per_order: Sequence[Tuple[int, int]],
        method: str,
) -> List[float]:
    """Apply a BLEU smoothing method to per-order ``(clipped, total)`` counts.

    A modified precision of zero makes the geometric mean zero, which is the
    whole reason smoothing exists. Transcribed from SacreBLEU's
    ``compute_bleu``; the four methods differ in what they do to a zero
    precision, and they differ measurably:

    ==========  ============================================================
    ``none``    Leave zeros. The geometric mean is then exactly ``0.0``.
    ``exp``     NIST / ``mteval-v13a`` geometric smoothing: a running counter
                starts at 1 and DOUBLES at each zero, giving
                ``p_n = 1 / (2**k * total_n)`` where ``k`` counts the zeros
                seen so far. SacreBLEU's default.
    ``floor``   ``p_n = delta / total_n`` with
                :data:`FLOOR_SMOOTHING_DELTA`. A floor on the NUMERATOR -- the
                denominator keeps the unmatched n-gram's full weight.
    ``add_k``   Chen & Cherry method 2: for ``n > 1`` both the clipped count
                and the total gain :data:`ADD_K_SMOOTHING_K`. Unlike the other
                three this one leaves order 1 alone.
    ==========  ============================================================

    Two behaviours of the reference are reproduced because dropping either
    changes a published number:

    - **A corpus with no match at any order returns all-zero precisions, even
      under smoothing.** Smoothing rescues a zero *precision*; it does not
      rescue a candidate that matched nothing, which would otherwise come back
      as a small positive score and read as "almost right".
    - **Orders beyond the candidate's length stay zero.** A three-token
      candidate has no 4-grams, so order 4 is left at 0.0 and the geometric
      mean is 0. This is why sentence BLEU on short candidates is brittle, and
      why the reference offers an ``effective_order`` switch; this module does
      not, because changing it would change the corpus number too.

    ``add_k`` is the one exception to the second bullet, and it is worth being
    precise about WHICH case it rescues, because the two are very different
    numbers. It adds ``k`` to the clipped count and to the total *before* the
    zero test, so:

    - a **MISSING** order (``total == 0``, the candidate has no n-grams of that
      length) becomes ``(0 + k) / (0 + k)`` = **exactly 1.0** -- a perfect
      precision for an order that does not exist;
    - a **PRESENT but zero-overlap** order becomes ``k / (total + k)``, which is
      ``0.5`` at ``total == 1`` and ``1/3`` at ``total == 2``, not 1.0.

    MEASURED. A three-token exact match has no 4-grams at all::

        >>> corpus_bleu("a b c", [["a b c"]], smooth="none")["precisions"]
        [1.0, 1.0, 1.0, 0.0]
        >>> corpus_bleu("a b c", [["a b c"]], smooth="add_k")["precisions"]
        [1.0, 1.0, 1.0, 1.0]

    And a genuine partial match, where every order EXISTS but two of them match
    nothing (``"a b x y"`` against ``"a b q r"``: 4 unigrams, 3 bigrams, 2
    trigrams, 1 4-gram, of which only ``a``, ``a b`` and nothing higher match)::

        >>> [round(p, 6) for p in corpus_bleu("a b x y", [["a b q r"]], smooth="add_k")["precisions"]]
        [0.5, 0.5, 0.333333, 0.5]
        >>> [round(p, 6) for p in corpus_bleu("a b x y", [["a b q r"]], smooth="none")["precisions"]]
        [0.5, 0.333333, 0.0, 0.0]

    The two zeroes are ``0/(2+1) = 1/3`` and ``0/(1+1) = 1/2`` -- inflated, but
    nowhere near 1.0. So the manufactured perfect precision is confined to the
    missing-order case, which is the one that matters for short-sentence
    corpora.

    So "a candidate shorter than ``max_order`` scores 0.0" holds for ``none``,
    ``exp`` and ``floor`` and fails for ``add_k`` -- and the failure mode is a
    manufactured 1.0 on an absent order, which is a worse thing to report than
    a zero. :func:`sentence_bleu` states the same exception for callers.

    Args:
        per_order: ``(clipped, total)`` pairs, order 1 upward. Only an exact zero
            clipped count is smoothed.
        method: One of :data:`BLEU_SMOOTHING_METHODS`.

    Returns:
        A list of precisions, one per entry of ``per_order``.

    Raises:
        ValueError: If ``method`` is not one of
            :data:`BLEU_SMOOTHING_METHODS`. An unknown method never falls back
            to a default -- a silently ignored mode string is indistinguishable
            from a metric that ignored a real setting.
    """
    method_name = _check_choice(method, "smooth", BLEU_SMOOTHING_METHODS)

    if not any(clipped for clipped, _ in per_order):
        return [0.0] * len(per_order)

    precisions: List[float] = []
    smooth_mteval = 1.0
    for index, (clipped, total) in enumerate(per_order, start=1):
        if method_name == "add_k" and index > 1:
            clipped = clipped + ADD_K_SMOOTHING_K
            total = total + ADD_K_SMOOTHING_K

        if total == 0:
            # No n-grams of this order exist in the candidate; nothing to
            # smooth and the order is genuinely absent.
            precisions.append(0.0)
            continue

        if clipped == 0:
            smooth_mteval *= 2.0
            if method_name == "exp":
                precisions.append(1.0 / (smooth_mteval * float(total)))
            elif method_name == "floor":
                precisions.append(FLOOR_SMOOTHING_DELTA / float(total))
            else:  # none
                precisions.append(0.0)
        else:
            precisions.append(clipped / float(total))
    return precisions


def _geometric_mean(precisions: Sequence[float]) -> float:
    """The weighted geometric mean BLEU is built on, in log space.

    .. math::

        \\exp\\!\\left(\\frac{1}{N}\\sum_{n=1}^{N} \\log p_n\\right)

    Returns exactly ``0.0`` if any precision is zero or non-positive: that is
    the mathematical value, not a special case, and computing it any other way
    (a small epsilon floor here, say) would let ``smooth="none"`` return a
    non-zero score on a candidate with a missing 4-gram.

    Args:
        precisions: Per-order precisions, order 1 upward.

    Returns:
        The geometric mean in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``precisions`` is empty.
    """
    if not precisions:
        raise ValueError("precisions must not be empty")
    if any(p <= 0.0 for p in precisions):
        return 0.0
    return float(np.exp(np.mean(np.log(np.asarray(precisions, dtype=np.float64)))))


def _f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    """The ``beta``-weighted harmonic mean of precision and recall.

    .. math::

        F_\\beta = \\frac{(1 + \\beta^2) \\, P \\, R}
                         {R + \\beta^2 \\, P}

    Args:
        precision: ``P >= 0``.
        recall: ``R >= 0``.
        beta: Recall weight. ``beta = 1`` is the plain harmonic mean; ``> 1``
            weights recall more heavily, which is what chrF (``beta = 2``) and
            METEOR (``beta = 3``) do.

    Returns:
        ``F_beta``, or exactly ``0.0`` when either side is zero rather than
        dividing by a zero denominator.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1.0 + beta_sq) * precision * recall / (recall + beta_sq * precision)


# ---------------------------------------------------------------------
# Validation helpers (public names: the metrics are called directly by callers
# and by create_generation_metric, so a private validator is not an option)
# ---------------------------------------------------------------------


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a positive Python/NumPy integer.

    Args:
        value: The value to check.
        name: The parameter name, used in the error message.

    Returns:
        ``value`` as an ``int``.

    Raises:
        ValueError: If ``value`` is a bool, not an integer, or ``< 1``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer; got {value!r} of type {type(value).__name__}"
        )
    value_int = int(value)
    if value_int < 1:
        raise ValueError(f"{name} must be >= 1; got {value_int}")
    return value_int


def _validate_positive_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite float ``> 0``.

    Args:
        value: The value to check.
        name: The parameter name, used in the error message.

    Returns:
        ``value`` as a ``float``.

    Raises:
        ValueError: If ``value`` is not a finite number, or ``<= 0``.
    """
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating),
    ):
        raise ValueError(
            f"{name} must be a number; got {value!r} of type {type(value).__name__}"
        )
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{name} must be finite; got {value_f}")
    if value_f <= 0.0:
        raise ValueError(f"{name} must be > 0; got {value_f}")
    return value_f


def _validate_non_negative_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a Python/NumPy integer ``>= 0``.

    Args:
        value: The value to check.
        name: The parameter name, used in the error message.

    Returns:
        ``value`` as an ``int``.

    Raises:
        ValueError: If ``value`` is a bool, not an integer, or ``< 0``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer; got {value!r} of type {type(value).__name__}"
        )
    value_int = int(value)
    if value_int < 0:
        raise ValueError(f"{name} must be >= 0; got {value_int}")
    return value_int


def _validate_non_negative_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite float ``>= 0``.

    Args:
        value: The value to check.
        name: The parameter name, used in the error message.

    Returns:
        ``value`` as a ``float``.

    Raises:
        ValueError: If ``value`` is not a finite number, or ``< 0``.
    """
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating),
    ):
        raise ValueError(
            f"{name} must be a number; got {value!r} of type {type(value).__name__}"
        )
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"{name} must be finite; got {value_f}")
    if value_f < 0.0:
        raise ValueError(f"{name} must be >= 0; got {value_f}")
    return value_f


def _check_choice(value: Any, name: str, allowed: Sequence[str]) -> str:
    """Normalise and validate a mode-string argument.

    Args:
        value: The mode name; any case, surrounding whitespace tolerated.
        name: The parameter name, used in the error message.
        allowed: The accepted lower-case names.

    Returns:
        The lower-cased mode name.

    Raises:
        ValueError: If ``value`` is not a string, or is not one of ``allowed``.
    """
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string; got {type(value).__name__}")
    name_lower = value.strip().lower()
    if name_lower not in allowed:
        raise ValueError(f"Unknown {name} {value!r}. Supported: {list(allowed)}")
    return name_lower


def _check_tokenizer(tokenizer: Optional[Tokenizer]) -> Tokenizer:
    """Resolve the ``tokenizer`` argument to a callable.

    Args:
        tokenizer: ``None`` (use :func:`whitespace_tokenize`) or a callable
            mapping a string to a token list.

    Returns:
        The tokenizer to use.

    Raises:
        ValueError: If ``tokenizer`` is neither ``None`` nor callable. A
            non-callable tokenizer is caught here rather than as a
            ``TypeError`` deep inside a counting loop.
    """
    if tokenizer is None:
        return whitespace_tokenize
    if not callable(tokenizer):
        raise ValueError(
            f"tokenizer must be callable or None; got {type(tokenizer).__name__}"
        )
    return tokenizer


def _as_references(references: Union[str, Sequence[Any]]) -> List[str]:
    """Normalise a reference argument to a flat list of reference strings.

    Accepts a bare string, a list of strings, or a list of per-sentence
    reference groups -- the last is how the sentence-level variants
    (:func:`rouge_lsum`, and the reference side of every multi-reference
    metric) receive their input.

    Args:
        references: A string, a sequence of strings, or a sequence of sequences
            of strings, nested no deeper than two levels.

    Returns:
        A flat list of reference strings.

    Raises:
        ValueError: If the nesting is deeper than two levels, if any element is
            not a string, or if the result is empty.
    """
    if isinstance(references, str):
        out = [references]
    elif isinstance(references, Sequence):
        out = []
        for element in references:
            if isinstance(element, str):
                out.append(element)
            elif isinstance(element, Sequence):
                for inner in element:
                    if not isinstance(inner, str):
                        raise ValueError(
                            f"references must contain strings; got "
                            f"{type(inner).__name__} nested inside a reference group"
                        )
                    out.append(inner)
            else:
                raise ValueError(
                    f"references must contain strings; got {type(element).__name__}"
                )
    else:
        raise ValueError(
            f"references must be a string or a sequence of strings; got "
            f"{type(references).__name__}"
        )
    if not out:
        raise ValueError("references must not be empty; got an empty sequence")
    return out


def _as_candidates(candidates: Union[str, Sequence[str]]) -> List[str]:
    """Normalise a candidate argument to a list of candidate strings.

    A bare string is accepted as a ONE-element corpus: every function in this
    module is corpus-shaped, and a single-string candidate is unambiguous
    because it is ``references=``, not the first argument, that carries the
    references.

    Args:
        candidates: A string or a sequence of strings.

    Returns:
        A list of candidate strings.

    Raises:
        ValueError: If any element is not a string, or the result is empty.
    """
    if isinstance(candidates, str):
        return [candidates]
    if not isinstance(candidates, Sequence):
        raise ValueError(
            f"candidates must be a string or a sequence of strings; got "
            f"{type(candidates).__name__}"
        )
    out = list(candidates)
    if not out:
        raise ValueError("candidates must not be empty; got an empty sequence")
    for candidate in out:
        if not isinstance(candidate, str):
            raise ValueError(
                f"every candidate must be a string; got {type(candidate).__name__}"
            )
    return out


def _paired_references(
        candidates: Sequence[str],
        references: Sequence[Any],
) -> List[List[str]]:
    """Pair every candidate with its own reference list, requiring equal lengths.

    Multi-reference metrics need this pairing. Rather than silently recycling a
    single reference across a corpus -- a real bug in hand-rolled BLEU scripts --
    a length mismatch raises here.

    A bare STRING is broadcast: one reference shared by every candidate. That is
    only really meaningful for a one-candidate corpus, but it is what a
    single-example call means, and the alternative -- iterating the string into
    one group per CHARACTER -- turns ``chrf("abcde", "abcde")`` into a
    five-candidate call against five one-character references.

    Args:
        candidates: One string per example.
        references: One reference string per candidate, or one reference group
            per candidate (a sequence of strings). A bare string broadcasts.

    Returns:
        A list of ``len(candidates)`` reference lists.

    Raises:
        ValueError: If the two lengths disagree.
    """
    if isinstance(references, str):
        return [[references] for _ in candidates]
    if len(references) != len(candidates):
        raise ValueError(
            f"candidates and references must be the same length; got "
            f"{len(candidates)} candidate(s) and {len(references)} reference group(s)"
        )
    return [_as_references(ref) for ref in references]


# ---------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------


def corpus_bleu(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        max_order: int = DEFAULT_BLEU_MAX_ORDER,
        smooth: str = "exp",
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, Any]:
    """Corpus BLEU with its components, over the whole candidate set.

    .. math::

        \\mathrm{BLEU} = BP \\cdot \\exp\\!\\left(
            \\frac{1}{N} \\sum_{n=1}^{N} \\log p_n \\right)

    where ``p_n`` is the fraction of the corpus's ``n``-grams whose count is
    matched by a reference up to that count, and ``BP`` uses the CORPUS-WIDE
    candidate and reference lengths.

    **Corpus BLEU is not the mean of sentence BLEU**, and the difference is not
    a rounding artefact -- it is a different measurement, and averaging sentence
    scores is the most common way a BLEU number becomes wrong. Three effects
    push them apart: the brevity penalty is applied once to the summed lengths
    rather than per sentence; a smoothing method sees one pooled zero count
    rather than one per sentence; and the geometric mean pools ``log p_n`` over
    the whole corpus rather than averaging per-sentence geometric means. A
    candidate set in which half the sentences are perfect and half are empty
    scores strictly higher as corpus BLEU than as sentence BLEU, because the
    perfect sentences' n-grams are not diluted by the empty ones' zeros.
    ``tests/test_metrics/test_generation_metrics.py`` pins the two apart with a
    measured gap rather than asserting they agree.

    Args:
        candidates: One candidate string, or one per example.
        references: Either one reference string (shared by every candidate, and
            only sensible for a one-candidate corpus), or one reference group
            per candidate. A reference group is a string or a sequence of
            strings.
        max_order: Highest n-gram order, inclusive, from 1. Four is the
            metric's standard. Lowering it to 1 gives unigram precision, which
            is a different metric under this metric's name.
        smooth: One of :data:`BLEU_SMOOTHING_METHODS`.
        tokenizer: Tokenizer callable, or ``None`` for
            :func:`whitespace_tokenize`.

    Returns:
        A dict with ``score`` (the BLEU value), ``precisions`` (a list of the
        per-order ``p_n``, after smoothing), ``bp``, ``ratio`` (candidate
        length over effective reference length), ``candidate_length``,
        ``reference_length``, ``max_order`` and ``smooth``.

    Raises:
        ValueError: If the candidate and reference lengths disagree, either side
            is empty, an element is not a string, ``max_order < 1``, or
            ``smooth`` is unknown.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import corpus_bleu
        >>> out = corpus_bleu(
        ...     ["the cat is on the mat", "there is a cat here too"],
        ...     ["the cat sits on the mat", "a cat is right here"],
        ... )
        >>> round(out["score"], 4) > 0.0
        True
        >>> out["bp"]
        1.0
    """
    max_order_int = _validate_positive_int(max_order, "max_order")
    smooth_name = _check_choice(smooth, "smooth", BLEU_SMOOTHING_METHODS)
    tokenize = _check_tokenizer(tokenizer)

    cand_list = _as_candidates(candidates)
    ref_groups = _paired_references(cand_list, references)

    pooled: List[Tuple[int, int]] = []
    candidate_length = 0
    reference_length = 0
    for candidate, refs in zip(cand_list, ref_groups):
        cand_tokens = tokenize(candidate)
        ref_token_lists = [tokenize(ref) for ref in refs]
        for order in range(1, max_order_int + 1):
            clipped, total, _ = _order_counts(cand_tokens, ref_token_lists, order)
            if order > len(pooled):
                pooled.append((clipped, total))
            else:
                prev_clipped, prev_total = pooled[order - 1]
                pooled[order - 1] = (prev_clipped + clipped, prev_total + total)
        candidate_length += len(cand_tokens)
        reference_length += _closest_ref_length(
            len(cand_tokens), [len(t) for t in ref_token_lists],
        )

    precisions = _smoothed_precisions(pooled, smooth_name)
    bp = _brevity_penalty(candidate_length, reference_length)
    score = bp * _geometric_mean(precisions)
    ratio = (
        candidate_length / reference_length if reference_length else 0.0
    )

    return {
        "score": float(score),
        "precisions": precisions,
        "bp": bp,
        "ratio": ratio,
        "candidate_length": candidate_length,
        "reference_length": reference_length,
        "max_order": max_order_int,
        "smooth": smooth_name,
    }


def sentence_bleu(
        candidate: str,
        references: Union[str, Sequence[str]],
        max_order: int = DEFAULT_BLEU_MAX_ORDER,
        smooth: str = "exp",
        tokenizer: Optional[Tokenizer] = None,
) -> float:
    """BLEU for a single candidate against one or more references.

    Identical in formula to :func:`corpus_bleu` applied to a one-element corpus,
    and reachable as such -- but it is the *sentence* variant, so its brevity
    penalty and its smoothing both apply to one sentence. Do not average this
    across a corpus to stand in for :func:`corpus_bleu`; the corpus value is
    the one to report, and :func:`corpus_bleu`'s docstring says why.

    **A candidate shorter than ``max_order`` tokens scores exactly ``0.0``
    under the default ``smooth="exp"``, and under ``"none"`` and ``"floor"``
    too.** This surprises people, so it is stated rather than left to be
    discovered: a three-token candidate has no 4-grams at all, so order 4 has a
    *total* of zero rather than a *zero precision*, and the reference's loop
    stops there -- smoothing rescues a zero precision, not a missing order. The
    geometric mean over four orders then contains a zero and the whole score is
    zero::

        >>> round(sentence_bleu("a b c", "a b c"), 6)      # exact 3-token match
        0.0
        >>> round(sentence_bleu("a b c d", "a b c d"), 6)  # one token longer
        1.0

    ``smooth="add_k"`` is the ONE exception: it adds ``k`` to the clipped count
    and the total before the zero test, so it rescues a missing order too and
    returns 1.0 on this example (see :func:`_smoothed_precisions` -- and note
    why that 1.0 is a manufactured perfect precision rather than a real one).
    Nothing else does.

    This is the reference implementation's behaviour and it is why the reference
    offers an ``effective_order`` switch; this module deliberately does not,
    because enabling it changes the CORPUS number too and that is not a change
    to make quietly inside a function named after the sentence variant. The
    practical consequence for a short-sentence corpus: report
    :func:`corpus_bleu`, or lower ``max_order`` to the shortest sentence you
    expect to see.

    Args:
        candidate: The generated string.
        references: A reference string, or a sequence of reference strings.
        max_order: Highest n-gram order, inclusive, from 1.
        smooth: One of :data:`BLEU_SMOOTHING_METHODS`.
        tokenizer: Tokenizer callable, or ``None`` for
            :func:`whitespace_tokenize`.

    Returns:
        The sentence BLEU in ``[0.0, 1.0]``. Exactly ``0.0`` whenever the
        candidate has fewer than ``max_order`` tokens, under every ``smooth``
        except ``"add_k"``.

    Raises:
        ValueError: If ``references`` is empty, ``max_order < 1``, or ``smooth``
            is unknown.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import sentence_bleu
        >>> round(sentence_bleu("the cat is on the mat", "the cat sits on the mat"), 6)
        0.379918
    """
    return float(
        corpus_bleu(
            candidate, [references], max_order=max_order, smooth=smooth,
            tokenizer=tokenizer,
        )["score"]
    )


def bleu(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        max_order: int = DEFAULT_BLEU_MAX_ORDER,
        smooth: str = "exp",
        tokenizer: Optional[Tokenizer] = None,
) -> float:
    """Corpus BLEU as a bare float. :func:`corpus_bleu` without the components.

    Args:
        candidates: One candidate string, or one per example.
        references: One reference string per candidate, or one reference group
            per candidate.
        max_order: Highest n-gram order, inclusive, from 1.
        smooth: One of :data:`BLEU_SMOOTHING_METHODS`.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.

    Returns:
        The corpus BLEU in ``[0.0, 1.0]``.

    Raises:
        ValueError: On any of :func:`corpus_bleu`'s conditions.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import bleu
        >>> round(bleu(["a b c d"], ["a b c d"]), 6)
        1.0
    """
    return float(
        corpus_bleu(
            candidates, references, max_order=max_order, smooth=smooth,
            tokenizer=tokenizer,
        )["score"]
    )


# ---------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------


def rouge_n(
        candidate: str,
        reference: str,
        n: int,
        score: str = "fmeasure",
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, float]:
    """ROUGE-N: clipped n-gram overlap of one candidate against one reference.

    All three quantities come from the same clipped overlap ``C``, so they are
    three readouts of one measurement rather than three metrics:

    .. math::

        P = C / |c|_n, \\qquad R = C / |r|_n

    ROUGE-N is reported as a recall-leaning quantity in the summarization
    literature, but the F-measure is what most papers quote. Select which one
    you return with ``score`` -- the other two are always present in the result,
    so a reader can always recover the others.

    Args:
        candidate: The generated string.
        reference: The single reference string. This function takes ONE
            reference; for the multi-reference, sentence-aware summarization
            form use :func:`rouge_lsum`.
        n: N-gram order.
        score: One of :data:`ROUGE_SCORE_TYPES` -- which quantity ``"score"``
            carries. Precision and recall are in the result regardless.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.

    Returns:
        A dict with ``score`` (the selected quantity), ``precision``,
        ``recall``, ``fmeasure`` and ``n``, all in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``n < 1``, or ``score`` is unknown.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import rouge_n
        >>> out = rouge_n("a b c d", "a b c d", n=2)
        >>> out["fmeasure"], out["recall"], out["precision"]
        (1.0, 1.0, 1.0)
    """
    n_int = _validate_positive_int(n, "n")
    score_name = _check_choice(score, "score", ROUGE_SCORE_TYPES)
    tokenize = _check_tokenizer(tokenizer)

    clipped, cand_total, ref_totals = _order_counts(
        tokenize(candidate), [tokenize(reference)], n_int,
    )
    precision = clipped / cand_total if cand_total else 0.0
    recall = clipped / ref_totals[0] if ref_totals[0] else 0.0

    # ROUGE-N reports the plain harmonic mean. It has no `beta` parameter --
    # ROUGE-L does, and that is exactly why the two cannot share one readout.
    fmeasure = _f_score(precision, recall, 1.0)
    selected = {
        "fmeasure": fmeasure,
        "recall": recall,
        "precision": precision,
    }[score_name]

    return {
        "score": float(selected),
        "precision": float(precision),
        "recall": float(recall),
        "fmeasure": float(fmeasure),
        "n": n_int,
    }


def _lcs_alignment(
        cand: Sequence[str],
        ref: Sequence[str],
) -> Tuple[int, set]:
    """Length of the longest common subsequence, and where it lands in ``cand``.

    A rolling-row dynamic program: an ``(m + 1) x (n + 1)`` table of VALUES is
    unnecessary because row ``i`` depends only on row ``i - 1``, so only two
    rows of values and one table of DIRECTIONS are kept. The direction table is
    not optional -- ``rouge_lsum`` needs to know WHICH candidate positions the
    LCS uses, and recovering that from the values afterwards means walking the
    table backwards guessing which predecessor was maximal, which silently
    picks the wrong alignment on every tie.

    **Cost is ``O(len_cand * len_ref)`` in both time and the direction table's
    memory.** Measured on an exact match: 0.001 s / 0.01 MB at 100 tokens,
    0.33 s / 32.7 MB at 2000. Exact LCS admits no sub-quadratic algorithm, so a
    few thousand tokens is the practical ceiling; sentence-level inputs
    (summaries, captions, MT segments) sit one to two orders of magnitude below
    it.

    Args:
        cand: Candidate tokens.
        ref: Reference tokens.

    Returns:
        ``(length, positions)`` -- the LCS length and the set of indices into
        ``cand`` at which the LCS places its tokens. The positions are what
        :func:`rouge_lsum` unions across reference sentences.
    """
    n_ref = len(ref)
    prev_row = [0] * (n_ref + 1)
    # 0 = came from the left (a reference token was skipped), 1 = came from
    # above (a candidate token was skipped), 2 = diagonal (the tokens matched).
    _LEFT, _UP, _DIAGONAL = 0, 1, 2
    direction = [[_LEFT] * (n_ref + 1) for _ in range(len(cand) + 1)]

    for i, cand_token in enumerate(cand, start=1):
        curr_row = [0] * (n_ref + 1)
        for j, ref_token in enumerate(ref, start=1):
            if cand_token == ref_token:
                # A matching cell is always on some optimal path, so the
                # diagonal wins outright and no comparison is needed.
                curr_row[j] = prev_row[j - 1] + 1
                direction[i][j] = _DIAGONAL
            elif curr_row[j - 1] >= prev_row[j]:
                # Tie goes LEFT, which makes the reconstruction find the
                # leftmost LCS.
                curr_row[j] = curr_row[j - 1]
                direction[i][j] = _LEFT
            else:
                curr_row[j] = prev_row[j]
                direction[i][j] = _UP
        prev_row = curr_row

    positions: set = set()
    i, j = len(cand), n_ref
    while i > 0 and j > 0:
        step = direction[i][j]
        if step == _DIAGONAL:
            positions.add(i - 1)
            i -= 1
            j -= 1
        elif step == _LEFT:
            j -= 1
        else:
            i -= 1
    return len(positions), positions


def rouge_l(
        candidate: str,
        reference: str,
        beta: float = DEFAULT_ROUGE_L_BETA,
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, float]:
    """ROUGE-L: F-measure over the longest common subsequence.

    Unlike ROUGE-1 and ROUGE-2, ROUGE-L rewards order: matched tokens must be
    in the same relative order in both sequences, but need not be contiguous,
    which is what lets a long candidate carry a high recall without being
    penalised for reordering an inserted clause.

    Args:
        candidate: The generated string.
        reference: The single reference string.
        beta: Recall weight for the F-measure. ``beta = 1`` is the harmonic mean
            and what modern implementations report. Lin's original ROUGE-L used
            ``beta -> inf``, i.e. pure recall, so the two are different metrics
            under one name; this module reports the F-measure by default and
            ``beta`` is the knob that reaches the recall-only form.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.

    Returns:
        A dict with ``score``, ``precision``, ``recall``, ``fmeasure``, ``lcs``
        and ``beta``, all quantities in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``beta <= 0`` or is not a number.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import rouge_l
        >>> rouge_l("a b c", "a x b")["lcs"]
        2
        >>> rouge_l("a b c", "a x b")["fmeasure"]     # P = 2/3, R = 2/3
        0.6666666666666666
    """
    beta_f = _validate_positive_float(beta, "beta")
    tokenize = _check_tokenizer(tokenizer)

    lcs_len, _ = _lcs_alignment(tokenize(candidate), tokenize(reference))
    cand_len = len(tokenize(candidate))
    ref_len = len(tokenize(reference))

    precision = lcs_len / cand_len if cand_len else 0.0
    recall = lcs_len / ref_len if ref_len else 0.0

    return {
        "score": float(_f_score(precision, recall, beta_f)),
        "precision": float(precision),
        "recall": float(recall),
        "fmeasure": float(_f_score(precision, recall, 1.0)),
        "lcs": int(lcs_len),
        "beta": beta_f,
    }


def rouge_lsum(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        beta: float = DEFAULT_ROUGE_L_BETA,
        tokenizer: Optional[Tokenizer] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
) -> Dict[str, float]:
    """ROUGE-Lsum: LCS computed per sentence pair, then unioned.

    The multi-sentence form of :func:`rouge_l`, and the summarization metric to
    report when references span sentences. For each candidate sentence and each
    reference sentence an LCS alignment is computed, and the matched candidate
    positions are **unioned** across reference sentences. The union is
    load-bearing: summing instead lets one candidate token be counted once per
    reference sentence it appears in, and drives precision above 1.

    ROUGE-L and ROUGE-Lsum differ by several points on the same inputs, so a
    report that says "ROUGE-L" without saying which is not reproducible. This
    function is the sentence-aware one.

    Args:
        candidates: One candidate string, or one per example.
        references: One reference string per candidate, or one reference group
            per candidate.
        beta: Recall weight for the F-measure.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.
        sentence_splitter: Splitter callable, or ``None`` for
            :func:`split_sentences`. This module's default is a regular
            expression over terminal punctuation and does NOT handle
            abbreviations, so it disagrees with ``rouge_score``'s
            tokenizer-based splitter on text like "Dr. Smith arrived."

    Returns:
        A dict with ``score``, ``precision``, ``recall``, ``fmeasure``,
        ``lcs`` and ``beta``. Quantities are corpus-level: totals over every
        candidate sentence, not a mean of per-example scores.

    Raises:
        ValueError: If the candidate and reference lengths disagree, either
            side is empty, or ``beta <= 0``.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import rouge_lsum
        >>> # The union is observable: "b" matches BOTH reference sentences and
        >>> # is counted once, so the LCS is 3 and not 4.
        >>> out = rouge_lsum(["a b c"], [["a b", "b c"]])
        >>> out["lcs"]
        3
    """
    beta_f = _validate_positive_float(beta, "beta")
    tokenize = _check_tokenizer(tokenizer)
    split = sentence_splitter if sentence_splitter is not None else split_sentences
    if not callable(split):
        raise ValueError(
            f"sentence_splitter must be callable or None; got "
            f"{type(sentence_splitter).__name__}"
        )

    cand_list = _as_candidates(candidates)
    ref_groups = _paired_references(cand_list, references)

    total_lcs = 0
    total_cand = 0
    total_ref = 0
    for candidate, refs in zip(cand_list, ref_groups):
        cand_sentences = [tokenize(s) for s in split(candidate)]
        # ONE level of flattening, not two. Every reference in every reference
        # group is a peer sentence to align against; keeping the group nesting
        # would hand `_lcs_alignment` a list of token LISTS, which matches
        # nothing and silently reports lcs = 0 for a perfect summary.
        ref_sentences = [tokenize(s) for ref in refs for s in split(ref)]
        total_cand += sum(len(s) for s in cand_sentences)
        total_ref += sum(len(s) for s in ref_sentences)

        for cand_sentence in cand_sentences:
            matched: set = set()
            for ref_sentence in ref_sentences:
                _, positions = _lcs_alignment(cand_sentence, ref_sentence)
                matched |= positions
            total_lcs += len(matched)

    precision = total_lcs / total_cand if total_cand else 0.0
    recall = total_lcs / total_ref if total_ref else 0.0

    return {
        "score": float(_f_score(precision, recall, beta_f)),
        "precision": float(precision),
        "recall": float(recall),
        "fmeasure": float(_f_score(precision, recall, 1.0)),
        "lcs": int(total_lcs),
        "beta": beta_f,
    }


# ---------------------------------------------------------------------
# chrF
# ---------------------------------------------------------------------


def _char_sequence(text: str, remove_whitespace: bool) -> str:
    """The character sequence chrF counts n-grams over.

    Args:
        text: Any string.
        remove_whitespace: Strip whitespace before counting, which is what makes
            chrF insensitive to the tokenization decisions that make BLEU
            tokenizer-dependent. The reference's default (``whitespace=False``)
            removes them too.

    Returns:
        The text to count character n-grams over.
    """
    return "".join(text.split()) if remove_whitespace else text


def _chrf_from_stats(
        stats: Sequence[Tuple[int, int, int]],
        beta: float,
) -> Tuple[float, float, float]:
    """Fold per-order ``(hyp, ref, match)`` counts into one chrF score.

    The averaging is over the EFFECTIVE orders only -- those where both the
    hypothesis and the reference actually had n-grams of that order -- and that
    detail is the single largest behavioural difference from a naive
    implementation:

    - A plain mean over all six orders counts a five-character candidate's
      missing 3-, 4-, 5- and 6-grams as four zeros, so ``chrf("abcde",
      "abcde")`` scores near **0.33** instead of **1.0**. The reference scores
      it **1.0**.
    - Restricting to effective orders is what makes chrF usable on short
      strings at all, which is most captioning.

    Args:
        stats: ``(hyp_count, ref_count, match_count)`` per order, any number of
            orders.
        beta: Recall weight.

    Returns:
        ``(score, mean_precision, mean_recall)``, all in ``[0.0, 1.0]``. Every
        entry is exactly ``0.0`` when no order had both counts non-zero.
    """
    sum_precision = 0.0
    sum_recall = 0.0
    effective_order = 0
    for hyp_count, ref_count, match_count in stats:
        if hyp_count > 0 and ref_count > 0:
            sum_precision += match_count / float(hyp_count)
            sum_recall += match_count / float(ref_count)
            effective_order += 1

    if effective_order == 0:
        return 0.0, 0.0, 0.0

    mean_precision = sum_precision / effective_order
    mean_recall = sum_recall / effective_order
    if mean_precision + mean_recall == 0.0:
        return 0.0, mean_precision, mean_recall
    return _f_score(mean_precision, mean_recall, beta), mean_precision, mean_recall


def chrf(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        char_order: int = DEFAULT_CHRF_CHAR_ORDER,
        word_order: int = DEFAULT_CHRF_WORD_ORDER,
        beta: float = DEFAULT_CHRF_BETA,
        remove_whitespace: bool = True,
        lowercase: bool = False,
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, Any]:
    """chrF / chrF++: character n-gram F-score, optionally augmented with words.

    .. math::

        \\mathrm{chrF}_\\beta = \\frac{(1 + \\beta^2) \\, \\mathrm{chrP} \\, \\mathrm{chrR}}
                                        {\\beta^2 \\, \\mathrm{chrP} + \\mathrm{chrR}}

    where each of ``chrP`` and ``chrR`` is the mean, over orders, of that
    order's clipped precision (or recall). Character orders run ``1..char_order``
    and, when ``word_order > 0``, word orders ``1..word_order``; the word half
    is what ``chrF++`` adds.

    Two aggregation choices, both taken from the reference implementation, and
    neither the obvious one:

    - **Counts are pooled across the corpus** before the score is formed, so this
      is a corpus statistic and not the mean of per-sentence chrF.
    - **Several references are resolved per segment, BEFORE pooling**: the
      reference with the best chrF wins that segment, and only its counts reach
      the pool. This is chrF's policy and it is the OPPOSITE of BLEU's, which
      takes the per-n-gram maximum across references. The two are not
      interchangeable, and a metric that quietly applied BLEU's rule here would
      score a segment high for being a blend of two references that each match
      it only halfway.

    See :func:`_chrf_from_stats` for why orders the candidate cannot have are
    excluded rather than counted as zero -- it is the difference between a
    five-character string scoring 0.33 and 1.0.

    Args:
        candidates: One candidate string, or one per example.
        references: One reference string per candidate, or one reference group
            per candidate.
        char_order: Highest character n-gram order, inclusive, from 1.
        word_order: Highest word n-gram order, inclusive, from 1. ``0``
            disables the word half (plain chrF); ``2`` gives chrF++.
        beta: Recall weight. WMT reports chrF at ``beta = 2``; ``beta = 1`` is
            the plain harmonic mean and is what "F-score" in the name suggests,
            so the two are different metrics under one name.
        remove_whitespace: Strip whitespace before counting character n-grams.
            ``False`` makes whitespace part of the alphabet, which is not a
            thing anyone reports.
        lowercase: Lower-case both sequences first. The reference's default is
            case-SENSITIVE, so this is off by default; chrF's character half is
            already largely case-insensitive through shared prefixes, which is
            why it can afford to be.
        tokenizer: Tokenizer callable for the WORD half only, or ``None`` for
            whitespace splitting. The character half never uses it.

    Returns:
        A dict with ``score``, ``char_precision``, ``char_recall``,
        ``word_precision``, ``word_recall`` (the last two ``None`` when
        ``word_order == 0``), ``beta``, ``char_order`` and ``word_order``.
        Every quantity is in ``[0.0, 1.0]``.

    Raises:
        ValueError: If the candidate and reference lengths disagree, either
            side is empty, either order is out of range, ``beta <= 0``, or
            either flag is not a bool.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import chrf
        >>> chrf("abcde", "abcde", char_order=6)["score"]
        1.0
        >>> chrf("abcde", "abcde", char_order=6, word_order=2)["score"]
        1.0
    """
    char_order_int = _validate_positive_int(char_order, "char_order")
    word_order_int = _validate_non_negative_int(word_order, "word_order")
    beta_f = _validate_positive_float(beta, "beta")
    for flag_name, flag in (
            ("remove_whitespace", remove_whitespace), ("lowercase", lowercase),
    ):
        if not isinstance(flag, bool):
            raise ValueError(
                f"{flag_name} must be a bool; got {type(flag).__name__}"
            )
    tokenize = _check_tokenizer(tokenizer)

    cand_list = _as_candidates(candidates)
    ref_groups = _paired_references(cand_list, references)

    char_stats: List[Tuple[int, int, int]] = []
    word_stats: List[Tuple[int, int, int]] = []

    for candidate, refs in zip(cand_list, ref_groups):
        prepared = [
            (ref.lower() if lowercase else ref) for ref in refs
        ]
        cand_text = candidate.lower() if lowercase else candidate

        cand_chars = _char_sequence(cand_text, remove_whitespace)
        char_counters = [
            _ngram_counts(cand_chars, order) for order in range(1, char_order_int + 1)
        ]
        per_ref_char = [
            [
                _ngram_counts(_char_sequence(ref, remove_whitespace), order)
                for order in range(1, char_order_int + 1)
            ]
            for ref in prepared
        ]

        # Best reference for this segment, chosen on the CHARACTER half alone
        # -- the same ordering the reference implementation uses, so adding
        # chrF++ cannot change which reference was picked.
        best_ref = max(
            range(len(prepared)),
            key=lambda i: _chrf_from_stats(
                _order_stats(char_counters, per_ref_char[i]), beta_f,
            )[0],
        )
        for order in range(char_order_int):
            hyp_count, ref_count, match_count = _order_stats(
                char_counters, per_ref_char[best_ref],
            )[order]
            if order < len(char_stats):
                h, r, m = char_stats[order]
                char_stats[order] = (h + hyp_count, r + ref_count, m + match_count)
            else:
                char_stats.append((hyp_count, ref_count, match_count))

        if word_order_int > 0:
            cand_words = tokenize(cand_text)
            word_counters = [
                _ngram_counts(cand_words, order) for order in range(1, word_order_int + 1)
            ]
            per_ref_word = [
                [_ngram_counts(tokenize(ref), order) for order in range(1, word_order_int + 1)]
                for ref in prepared
            ]
            for order in range(word_order_int):
                hyp_count, ref_count, match_count = _order_stats(
                    word_counters, per_ref_word[best_ref],
                )[order]
                if order < len(word_stats):
                    h, r, m = word_stats[order]
                    word_stats[order] = (h + hyp_count, r + ref_count, m + match_count)
                else:
                    word_stats.append((hyp_count, ref_count, match_count))

    # chrF++ folds the word orders into the SAME effective-order average as the
    # character orders -- it does not compute a second F-score and combine two
    # finished numbers. Two orders that disagreed about which reference won, or
    # an average over 6 orders instead of 8, would both be a different metric
    # under this name; the concatenation below is the reference's rule and the
    # word half is otherwise inert, which is a dead knob wearing a parameter.
    combined_stats = char_stats + word_stats
    score, char_p, char_r = _chrf_from_stats(combined_stats, beta_f)
    if word_order_int > 0:
        _word_score, word_p, word_r = _chrf_from_stats(word_stats, beta_f)
    else:
        word_p, word_r = None, None

    return {
        "score": float(score),
        # DIAGNOSTIC, not a component of `score`: the character and word halves
        # are averaged together into one effective-order mean, so these are
        # reported for inspection and are not separately recombinable.
        "char_precision": float(char_p),
        "char_recall": float(char_r),
        "word_precision": None if word_p is None else float(word_p),
        "word_recall": None if word_r is None else float(word_r),
        "beta": beta_f,
        "char_order": char_order_int,
        "word_order": word_order_int,
    }


def _order_stats(
        cand_counters: Sequence[Dict[Tuple[str, ...], int]],
        ref_counters: Sequence[Dict[Tuple[str, ...], int]],
) -> List[Tuple[int, int, int]]:
    """Per-order ``(hyp, ref, match)`` counts for one candidate-reference pair.

    The clip is ``min(hyp_count, ref_count)`` per n-gram, and the hypothesis
    count is reported as zero when the reference has no n-grams at all, matching
    the reference implementation -- otherwise an empty reference would produce
    ``0 / 0`` rather than a zero.

    Args:
        cand_counters: The candidate's n-gram multiset, one per order.
        ref_counters: The reference's n-gram multiset, one per order.

    Returns:
        One ``(hyp_count, ref_count, match_count)`` triple per order.
    """
    stats: List[Tuple[int, int, int]] = []
    for cand_counts, ref_counts in zip(cand_counters, ref_counters):
        match_count = 0
        hyp_count = 0
        for gram, count in cand_counts.items():
            hyp_count += count
            ref_count = ref_counts.get(gram, 0)
            if ref_count:
                match_count += min(count, ref_count)
        stats.append((hyp_count if ref_counts else 0, sum(ref_counts.values()), match_count))
    return stats

# ---------------------------------------------------------------------
# METEOR
# ---------------------------------------------------------------------


def _match_forms(token: str, match_fn: Optional[Matcher]) -> frozenset:
    """The set of surface forms one token counts as.

    Args:
        token: A token.
        match_fn: Optional matcher. Its result is UNIONED with the token's own
            form, so a matcher that only ever returns stems still cannot
            delete an exact match. The union is load-bearing: NLTK applies
            exact matching first and stemmed/synonym matching only to the
            leftovers, so an exact match is always preferred, and a matcher
            that swallowed identity forms would silently reorder the
            preference.

    Returns:
        The equivalence set for the token.
    """
    if match_fn is None:
        return frozenset((token,))
    forms = set(match_fn(token))
    forms.add(token)
    return frozenset(forms)


def _meteor_alignment(
        cand: Sequence[str],
        ref: Sequence[str],
        match_fn: Optional[Matcher],
) -> List[Tuple[int, int]]:
    """Align candidate positions to reference positions, both orders accounted.

    Candidate positions are visited in REVERSE and each takes the HIGHEST still
    unaligned reference position it is equivalent to. This is a transcription
    of the reference implementation's scan, and its two details are not
    incidental:

    - **Reverse order**, so the last candidate token claims the last matching
      reference token rather than the first. Scanning forward instead pairs the
      same multiset of tokens differently whenever a form repeats, which
      changes the CHUNK COUNT and therefore the penalty.
    - **Highest available reference position**, per form, not the closest. The
      reference has no distance notion at all in this stage.

    With ``match_fn=None`` this is exactly NLTK's exact-match stage, and it is
    linear in the input because reference positions are indexed by form once
    rather than rescanned per candidate token.

    Args:
        cand: Candidate tokens.
        ref: Reference tokens.
        match_fn: Optional equivalence matcher, or ``None`` for exact identity.

    Returns:
        ``(candidate_index, reference_index)`` pairs sorted by increasing
        candidate index.
    """
# Each reference position is indexed under EVERY form it is equivalent to, not
    # only under its literal surface form. Indexing by the literal token alone
    # makes `match_fn` almost entirely inert: a stemmed match works only when
    # the stem happens to also BE a reference token, so a caller who passes a
    # matcher sees the same score as one who does not, with nothing raised.
    ref_positions: Dict[str, List[int]] = {}
    for j, ref_token in enumerate(ref):
        for form in _match_forms(ref_token, match_fn):
            ref_positions.setdefault(form, []).append(j)

    taken: set = set()
    matches: List[Tuple[int, int]] = []
    for i in range(len(cand) - 1, -1, -1):
        best_j = -1
        for form in _match_forms(cand[i], match_fn):
            for j in ref_positions.get(form, ()):
                if j > best_j and j not in taken:
                    best_j = j
        if best_j >= 0:
            taken.add(best_j)
            matches.append((i, best_j))

    matches.sort()
    return matches


def _count_chunks(matches: Sequence[Tuple[int, int]]) -> int:
    """Count the METEOR chunks a matched alignment falls into.

    Two consecutive matches are in the SAME chunk only when they are adjacent in
    BOTH sequences -- ``matches[i+1] == matches[i] + (1, 1)`` -- which is the
    reference implementation's rule and its own description of itself ("the
    fewest possible number of chunks such that matched unigrams of each chunk
    are adjacent to each other").

    The rule is STRICTER than it first looks, and that strictness is the metric:

    - ``"a x b"`` against ``"a b"`` is TWO chunks. The deletion sits in the
      candidate only, but the matches are not adjacent in the candidate, so
      they do not form one fragment.
    - ``"a b"`` against ``"a y b"`` is TWO chunks, for the mirror reason.
    - Only a run of diagonally adjacent matches stays a single chunk, which is
      what an uninterrupted translated phrase looks like.

    A more permissive reading -- "a gap in only one sequence is not a
    separation" -- gives FEWER chunks and therefore a strictly HIGHER score on
    every reordered or edited candidate. It is the single most likely way to
    write a METEOR that disagrees with every published METEOR number, so it is
    pinned directly by a hand-computed test.

    Args:
        matches: ``(candidate_index, reference_index)`` pairs, sorted by
            increasing candidate index.

    Returns:
        The chunk count. ``0`` for an empty alignment; :func:`meteor` handles
        that case before calling, because with no matches the penalty's
        denominator is zero and the score is 0 by definition.
    """
    if not matches:
        return 0
    chunks = 1
    for prev, cur in zip(matches, matches[1:]):
        if not (cur[0] == prev[0] + 1 and cur[1] == prev[1] + 1):
            chunks += 1
    return chunks


def meteor(
        candidate: str,
        references: Union[str, Sequence[str]],
        beta: float = DEFAULT_METEOR_BETA,
        gamma: float = DEFAULT_METEOR_GAMMA,
        tokenizer: Optional[Tokenizer] = None,
        match_fn: Optional[Matcher] = None,
        lowercase: bool = True,
) -> Dict[str, float]:
    """METEOR for one candidate against one or more references.

    .. math::

        F_{\\mathrm{mean}} &= \\frac{P R}{\\beta^2 P + R}, \\qquad
        P = \\frac{m}{|c|}, \\quad R = \\frac{m}{|r|}

        \\mathrm{pen} = \\gamma \\left(\\frac{ch}{m}\\right)^{\\beta}, \\qquad
        \\mathrm{METEOR} = F_{\\mathrm{mean}} \\,(1 - \\mathrm{pen})

    ``m`` is the number of aligned tokens and ``ch`` the number of chunks (see
    :func:`_count_chunks`). With several references the score is the MAX over
    them, matching the reference implementation -- not the mean, which would let
    one bad reference drag down a candidate the metric considers answered.

    The formula is written with ``beta^2`` because the reference's
    ``F_mean = PR / (alpha P + (1 - alpha) R)`` with ``alpha = 0.9`` IS this F
    measure at ``beta = 3``, exactly: ``PR / (0.9P + 0.1R) = 10PR / (9P + R)``.
    The module's D-002 decision note records why only ``beta`` is exposed, and
    that identity is pinned numerically in the test suite rather than left as
    prose.

    **A perfect candidate does not score 1.0.** For a three-token identity
    match ``m = 3``, ``ch = 1``, so the penalty is ``0.5 * (1/3)**3 =
    0.018518...`` and the score is ``0.981481...``, not 1.0. This is the
    reference behaviour, not a defect: the fragmentation ratio is not normalised
    by sentence length, so short sentences carry more penalty, and only an
    unboundedly long contiguous match would drive the penalty to zero. A test
    that pinned ``1.0`` here would be pinning the wrong value and would go RED
    on a correct implementation.

    **The default alignment is exact-match, and that is a real fidelity loss.**
    The reference matches through Porter stemming and WordNet synonyms, staged
    after exact matching; this module ships no stemmer because NLTK is not a
    project dependency. On text where inflection differs (``the dog barked``
    against ``a dog is barking``) the default scores markedly lower than a
    stemmed METEOR, and its numbers are not comparable to published METEOR
    figures. Pass a ``match_fn`` to recover the stemmed form::

        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        meteor(cand, ref, match_fn=lambda t: frozenset({stemmer.stem(t)}))

    Args:
        candidate: The generated string.
        references: A reference string, or a sequence of reference strings. The
            score is the best over them.
        beta: Recall weight ``>= 1``, used in BOTH the F-mean (``beta^2``) and
            the fragmentation exponent. The reference default, 3.0, IS its
            9:1 precision/recall weighting. Values below 1 weight PRECISION
            above recall, which is outside the metric's definition.
        gamma: Fragmentation-penalty coefficient. ``0.0`` removes the penalty
            and leaves a recall-weighted unigram F-measure, which is a
            DIFFERENT metric under this name.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.
        match_fn: Optional equivalence matcher; see above. ``None`` means exact
            identity.
        lowercase: Lower-case both sequences before matching, which is the
            reference implementation's default preprocessing and what makes
            ``"The cat"`` match ``"the cat"``. Case is a real signal in some
            tasks, so this is a knob rather than a hard-wired rule.

    Returns:
        A dict with ``score`` in ``[0.0, 1.0]``, plus ``precision``, ``recall``,
        ``f_mean``, ``matches``, ``chunks``, ``penalty``, ``beta``, ``gamma``
        and ``n_references``.

    Raises:
        ValueError: If ``references`` is empty or holds a non-string, ``beta``
            is ``<= 0``, ``gamma`` is negative or non-finite, or ``lowercase``
            is not a bool.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import meteor
        >>> out = meteor("a b c", "a b c")
        >>> out["matches"], out["chunks"], out["penalty"]
        (3, 1, 0.018518518518518514)
        >>> round(out["score"], 6)
        0.981481
        >>> meteor("a b c", "x y z")["score"]
        0.0
    """
    beta_f = _validate_positive_float(beta, "beta")
    gamma_f = _validate_non_negative_float(gamma, "gamma")
    tokenize = _check_tokenizer(tokenizer)
    if match_fn is not None and not callable(match_fn):
        raise ValueError(
            f"match_fn must be callable or None; got {type(match_fn).__name__}"
        )
    if not isinstance(lowercase, bool):
        raise ValueError(
            f"lowercase must be a bool; got {type(lowercase).__name__}"
        )

    ref_list = _as_references(references)
    cand_tokens = tokenize(candidate.lower() if lowercase else candidate)

    best: Dict[str, float] = {}
    for ref in ref_list:
        ref_tokens = tokenize(ref.lower() if lowercase else ref)
        matches = _meteor_alignment(cand_tokens, ref_tokens, match_fn)
        n_matches = len(matches)

        precision = n_matches / len(cand_tokens) if cand_tokens else 0.0
        recall = n_matches / len(ref_tokens) if ref_tokens else 0.0
        f_mean = _f_score(precision, recall, beta_f)

        if n_matches == 0:
            # The penalty divides by the match count, so with no match there is
            # no penalty to form and the metric is 0 by definition.
            score, chunks, penalty = 0.0, 0, 0.0
        else:
            chunks = _count_chunks(matches)
            penalty = min(gamma_f * (chunks / float(n_matches)) ** beta_f, 1.0)
            score = f_mean * (1.0 - penalty)

        result = {
            "score": float(score),
            "precision": float(precision),
            "recall": float(recall),
            "f_mean": float(f_mean),
            "matches": int(n_matches),
            "chunks": int(chunks),
            "penalty": float(penalty),
            "beta": beta_f,
            "gamma": gamma_f,
        }
        # `not best` is the FIRST-REFERENCE test: an empty dict has no "score" key to
        # read, and a populated one always does.
        if not best or result["score"] > best["score"]:
            best = result

    # `_as_references` already rejected an empty list, so `ref_list` is
    # non-empty and the loop above ran at least once -- `best` is populated.
    # An empty dict (rather than a `None` sentinel) makes the "is this the
    # first reference?" test an `if not best`, which no static analyser has to
    # reason about, and the guard below turns a hypothetical empty return into
    # a named failure instead of a bare `{}`.
    if not best:  # pragma: no cover -- unreachable given the validation
        raise ValueError("meteor: no reference was scored")
    best["n_references"] = len(ref_list)
    return best

# ---------------------------------------------------------------------
# CIDEr / CIDEr-D
# ---------------------------------------------------------------------


def _cider_counts(tokens: Sequence[str], n: int) -> List[Dict[Tuple[str, ...], int]]:
    """CIDEr's "precook": n-gram multisets for every order ``1..n`` in one pass.

    Args:
        tokens: A token sequence.
        n: Highest n-gram order, inclusive, from 1.

    Returns:
        A list of ``n`` dicts, entry ``k`` holding the ``k + 1``-gram multiset.
        The ORDER IS THE LIST INDEX rather than something recovered from the
        n-gram tuple's length, because the reference implementation indexes its
        vector array the same way and the two must agree to produce the same
        number.
    """
    counts: List[Dict[Tuple[str, ...], int]] = [{} for _ in range(n)]
    for order in range(1, n + 1):
        bucket = counts[order - 1]
        for gram in ngrams(tokens, order):
            bucket[gram] = bucket.get(gram, 0) + 1
    return counts


def _tfidf_vector(
        counts: Sequence[Dict[Tuple[str, ...], int]],
        doc_freq: Sequence[Dict[Tuple[str, ...], int]],
        log_num_examples: float,
) -> Tuple[List[Dict[Tuple[str, ...], float]], List[float]]:
    """CIDEr's TF-IDF weighting of one sequence's n-gram multisets.

    The weight of an n-gram ``g`` occurring ``c`` times is

    ``c * (log(num_examples) - log(max(1, df(g))))`` = ``c * log(N / max(1, df(g)))``

    which is TF-IDF, and the sign is the thing to get right: a COMMON n-gram
    has a large ``df``, hence a large ``log(df)`` subtracted, hence a weight
    near zero. An n-gram appearing in every reference gets weight exactly zero
    and contributes nothing at all -- which is the entire point, since CIDEr
    exists to reward n-grams that distinguish one image's captions from the
    corpus's background vocabulary.

    ``log(N)`` is the natural log of the number of EXAMPLES (images in the
    original), not the total n-gram count. Using the n-gram count instead makes
    every weight negative on a corpus larger than ``e``, which flips the sign of
    the clipped similarity below.

    Args:
        counts: This sequence's per-order n-gram multisets, from
            :func:`_cider_counts`.
        doc_freq: Corpus document frequency per order.
        log_num_examples: ``log(N)``, the corpus IDF ceiling.

    Returns:
        ``(vector, norm)`` -- the per-order weighted dicts and their L2 norms.
        The norms are returned rather than recomputed because the similarity
        needs them once per order per pair.
    """
    vectors: List[Dict[Tuple[str, ...], float]] = []
    norms: List[float] = []
    for order, bucket in enumerate(counts):
        vector: Dict[Tuple[str, ...], float] = {}
        squared = 0.0
        for gram, count in bucket.items():
            df = doc_freq[order].get(gram, 0)
            weight = float(count) * (log_num_examples - float(np.log(max(1.0, df))))
            vector[gram] = weight
            squared += weight * weight
        vectors.append(vector)
        norms.append(float(np.sqrt(squared)))
    return vectors, norms


def _clipped_similarity(
        hyp_vector: Dict[Tuple[str, ...], float],
        ref_vector: Dict[Tuple[str, ...], float],
        hyp_norm: float,
        ref_norm: float,
) -> float:
    """One order's clipped cosine similarity between two CIDEr vectors.

    .. math::

        \\frac{\\sum_{g \\in s} \\min(w_h(g), w_r(g)) \\cdot w_r(g)}
             {\\lVert v_h \\rVert \\, \\lVert v_r \\rVert}

    The ``min`` is the load-bearing word. This is a CLIPPED inner product, not a
    plain dot product: a candidate that over-produces an n-gram is credited only
    up to what the reference itself has, exactly like BLEU's and chrF's clipping
    elsewhere in this module. A plain dot product would reward repetition --
    tripling an n-gram's weight in the candidate and calling it a better match.
    The sum runs over the CANDIDATE's n-grams, so the expression is not
    symmetric, which is intentional and matches the reference.

    Args:
        hyp_vector: The candidate's weighted vector for this order.
        ref_vector: The reference's weighted vector for this order.
        hyp_norm: The candidate vector's L2 norm.
        ref_norm: The reference vector's L2 norm.

    Returns:
        The clipped cosine in ``[0.0, 1.0]``, or exactly ``0.0`` when either
        norm is zero -- which happens when a sequence has no n-grams at that
        order, and is common at high orders on short captions.
    """
    if hyp_norm == 0.0 or ref_norm == 0.0:
        return 0.0
    total = 0.0
    for gram, hyp_weight in hyp_vector.items():
        ref_weight = ref_vector.get(gram, 0.0)
        if ref_weight:
            total += min(hyp_weight, ref_weight) * ref_weight
    return float(total / (hyp_norm * ref_norm))


def _gaussian(delta: float, sigma: float) -> float:
    """The Gaussian length penalty ``exp(-(delta^2) / (2 sigma^2))``.

    Args:
        delta: The signed length difference.
        sigma: The penalty's standard deviation, ``> 0``.

    Returns:
        A value in ``(0.0, 1.0]``, exactly ``1.0`` when ``delta == 0``.
    """
    return float(np.exp(-(delta * delta) / (2.0 * sigma * sigma)))


def cider(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        n: int = DEFAULT_BLEU_MAX_ORDER,
        sigma: float = DEFAULT_CIDER_SIGMA,
        variant: str = "d",
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, Any]:
    """CIDEr / CIDEr-D: TF-IDF-weighted n-gram similarity, averaged over orders.

    For each candidate and each order, the score is the mean over its
    references of a clipped cosine similarity multiplied by a Gaussian length
    penalty:

    .. math::

        g_n(s) = \\frac{1}{m} \\sum_{j} \\cos^{+}\\!\\big(v_n(s), v_n(r_{ij})\\big)
                 \\cdot e^{-\\left(\\frac{|r_{ij}| - |s|}{2\\sigma^2}\\right)^2}

    and CIDEr-D multiplies that by a SECOND Gaussian centred on the corpus's
    mean reference length, which penalises a candidate for being longer than
    typical rather than only for differing from its own references:

    .. math::

        g^{D}_n(s) = g_n(s) \\cdot e^{-\\left(\\frac{|s| - \\delta}{2\\sigma^2}\\right)^2}

    The reported value is ``10 * mean_n g_n``, on the ``[0, 10]`` scale its
    reference defines -- the one function here not normalized to ``[0, 1]``.

    IDF weights are a functional of the whole reference corpus, which is why
    this cannot be a streaming metric: one candidate scored alone gets an IDF
    vector with no corpus behind it. That also makes CIDEr's MAGNITUDE
    corpus-dependent in a way the other four here are not. Scoring three
    near-identical references against each other gives every n-gram a document
    frequency of 3, and the weighting collapses toward uniformity -- the metric
    still returns a confident number, it is simply measuring the wrong thing on
    a corpus that small.

    **CIDEr is zero on a one-example corpus, and on any corpus where every
    n-gram is universal.** The weight is ``log(N / max(1, df))``, so a corpus
    of ``N = 1`` gives ``log(1/1) = 0`` on every n-gram and the score is exactly
    ``0.0`` -- not because the candidate is wrong, but because IDF has no
    contrast to measure. The same applies to any corpus where every candidate's
    n-grams appear in every reference. So the nearest thing to an identity check
    is a corpus of three examples with DISJOINT vocabularies, which drives the
    weight to its maximum.

    Plain CIDEr here is a transcription of the reference scorer. CIDEr-D's
    second penalty follows Vedantam et al. (2016), whose reference scorer is not
    distributed under the same package; the first penalty, the clipping, the IDF
    and the ``10 * mean`` scaling are identical between the two.

    Args:
        candidates: One candidate string, or one per example.
        references: One reference string per candidate, or one reference group
            per candidate.
        n: Highest n-gram order, inclusive, from 1. Four is the reference's
            default. Orders the candidates are too short to have contribute
            ``0.0`` and are still counted in the mean, so a corpus of 3-token
            sentences at the default ``n = 4`` tops out at ``7.5`` rather than
            ``10.0`` even on a perfect match.
        sigma: Standard deviation of the Gaussian length penalty, in tokens.
            ``6.0`` is the reference's value for English.
        variant: ``"d"`` for CIDEr-D (with the corpus-length penalty) or ``"c"``
            for plain CIDEr (without it). Plain CIDEr scores strictly higher on
            the same input, so the two are different metrics under one name.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.

    Returns:
        A dict with ``score`` in ``[0.0, 10.0]``, ``cider_n`` (the per-order
        corpus means ``g_n``, length ``n``), ``delta`` (the corpus mean
        reference length), ``n``, ``sigma`` and ``variant``.

    Raises:
        ValueError: If the candidate and reference lengths disagree, either
            side is empty, ``n < 1``, ``sigma <= 0``, or ``variant`` is unknown.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import cider
        >>> corpus = ["a b c", "d e f", "g h i"]
        >>> cider(corpus, corpus, n=2)["score"]
        10.0
        >>> cider(["a b c d"], ["a b c d"])["score"]     # N == 1: no IDF contrast
        0.0
    """
    n_int = _validate_positive_int(n, "n")
    sigma_f = _validate_positive_float(sigma, "sigma")
    variant_name = _check_choice(variant, "variant", CIDER_VARIANTS)
    tokenize = _check_tokenizer(tokenizer)

    cand_list = _as_candidates(candidates)
    ref_groups = _paired_references(cand_list, references)

    cand_token_lists = [tokenize(c) for c in cand_list]
    ref_token_lists = [[tokenize(ref) for ref in refs] for refs in ref_groups]

    # Document frequency counts EXAMPLES, not references: an n-gram present in
    # any reference of one example contributes exactly 1, so one image with
    # five captions cannot out-vote five images with one caption each.

    # The IDF term `log(N / max(1, df))` is non-negative only while
    # `df <= N`, and that holds STRUCTURALLY here: the loop below runs exactly
    # `len(cand_list)` times and increments each gram at most once per pass,
    # so `df` cannot exceed `N` for any input. There is therefore deliberately
    # NO runtime check for it, and an earlier version of this module carried
    # one -- a guard no input could reach, whose test then "proved" it fired by
    # asserting the opposite. The reference implementation asserts the same
    # invariant because its `CiderScorer` accumulates corpora through
    # `__iadd__` across separately-scored segments; this port has no such path,
    # so the assertion has nothing to guard and was removed rather than left as
    # decoration.
    doc_freq: List[Dict[Tuple[str, ...], int]] = [{} for _ in range(n_int)]
    for ref_lists in ref_token_lists:
        for order in range(1, n_int + 1):
            example_grams: set = set()
            for ref_tokens in ref_lists:
                example_grams.update(ngrams(ref_tokens, order))
            for gram in example_grams:
                doc_freq[order - 1][gram] = doc_freq[order - 1].get(gram, 0) + 1

    log_num_examples = float(np.log(len(cand_list)))
    # CIDEr-D's `delta`: the mean length of every reference in the corpus, which
    # is what makes the second penalty a statement about the DATA rather than
    # about this candidate's own references.
    total_ref_tokens = sum(len(t) for ref_lists in ref_token_lists for t in ref_lists)
    delta = total_ref_tokens / len(cand_list)

    per_order_totals = [0.0] * n_int
    for index, cand_tokens in enumerate(cand_token_lists):
        cand_len = len(cand_tokens)
        hyp_vectors, hyp_norms = _tfidf_vector(
            _cider_counts(cand_tokens, n_int), doc_freq, log_num_examples,
        )
        for order in range(n_int):
            order_scores = []
            for ref_tokens in ref_token_lists[index]:
                ref_vectors, ref_norms = _tfidf_vector(
                    _cider_counts(ref_tokens, n_int), doc_freq, log_num_examples,
                )
                similarity = _clipped_similarity(
                    hyp_vectors[order], ref_vectors[order],
                    hyp_norms[order], ref_norms[order],
                )
                penalty = _gaussian(len(ref_tokens) - cand_len, sigma_f)
                if variant_name == "d":
                    penalty *= _gaussian(cand_len - delta, sigma_f)
                order_scores.append(similarity * penalty)
            per_order_totals[order] += float(np.mean(order_scores)) if order_scores else 0.0

    cider_n = [total / len(cand_list) for total in per_order_totals]
    score = 10.0 * float(np.mean(cider_n)) if cider_n else 0.0

    return {
        "score": score,
        "cider_n": cider_n,
        "delta": float(delta),
        "n": n_int,
        "sigma": sigma_f,
        "variant": variant_name,
    }

# ---------------------------------------------------------------------
# Report and registry
# ---------------------------------------------------------------------


def generation_report(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        max_order: int = DEFAULT_BLEU_MAX_ORDER,
        smooth: str = "exp",
        char_order: int = DEFAULT_CHRF_CHAR_ORDER,
        word_order: int = DEFAULT_CHRF_WORD_ORDER,
        chrf_beta: float = DEFAULT_CHRF_BETA,
        rouge_beta: float = DEFAULT_ROUGE_L_BETA,
        meteor_beta: float = DEFAULT_METEOR_BETA,
        meteor_gamma: float = DEFAULT_METEOR_GAMMA,
        cider_n: int = DEFAULT_BLEU_MAX_ORDER,
        cider_sigma: float = DEFAULT_CIDER_SIGMA,
        cider_variant: str = "d",
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, Any]:
    """Every metric in this module over one candidate/reference corpus.

    Returns each metric's full component dict under its own key, plus a
    ``"scaled"`` block carrying the conventional reporting units -- BLEU and
    chrF as percentages, CIDEr as its native ``[0, 10]``, ROUGE and METEOR
    unscaled because that is already how they are reported.

    The metric keys are ``bleu``, ``chrf``, ``rouge1``, ``rouge2``, ``rougeL``,
    ``rougeLsum``, ``meteor`` and ``cider``.

    **Two axes differ across the eight, and both are documented rather than
    left to be inferred from the code.**

    *Which references are used.* ROUGE-1, ROUGE-2 and ROUGE-L are
    single-reference metrics and are computed against the FIRST reference only.
    The other five use every reference, each under its own rule: BLEU max-pools
    per n-gram, chrF takes the single best reference per segment, CIDEr averages
    over them, ``rougeLsum`` uses all of them, and METEOR takes the best.

    *How the corpus is combined.* Four metrics POOL their sufficient statistics
    and compute one score from the pooled totals -- BLEU over n-gram counts,
    chrF over match counts, ROUGE-1/2/L over overlap and length totals, CIDEr
    over per-order ``g_n``. METEOR is the exception: its fragmentation penalty
    is defined per sentence, so there is nothing to pool and the reported value
    is the MEAN of the per-sentence scores. That is a property of the metric,
    not an inconsistency, and it is why this report is not "one aggregation
    applied uniformly".

    Args:
        candidates: One candidate string, or one per example.
        references: One reference string per candidate, or one reference group
            per candidate.
        max_order: BLEU's highest n-gram order.
        smooth: BLEU smoothing method; see :data:`BLEU_SMOOTHING_METHODS`.
        char_order: chrF's highest character n-gram order.
        word_order: chrF's highest word n-gram order; ``2`` gives chrF++.
        chrf_beta: chrF's recall weight.
        rouge_beta: ROUGE-L / ROUGE-Lsum recall weight.
        meteor_beta: METEOR's recall weight and fragmentation exponent.
        meteor_gamma: METEOR's fragmentation-penalty coefficient.
        cider_n: CIDEr's highest n-gram order.
        cider_sigma: CIDEr's Gaussian length-penalty standard deviation.
        cider_variant: ``"d"`` or ``"c"``.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.

    Returns:
        A dict with one entry per metric plus ``"scaled"``. Every entry carries
        at least ``score``; the per-metric component keys are that metric's own
        (BLEU's ``precisions``/``bp``, CIDEr's ``cider_n``, and so on), so a
        caller reading ``report[key]["score"]`` works uniformly and a caller
        wanting the breakdown asks the metric's own docstring what it reports.

    Raises:
        ValueError: On any of the constituent metrics' conditions, and on a
            candidate/reference length mismatch.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import generation_report
        >>> report = generation_report(
        ...     ["the cat is on the mat", "there is a cat"],
        ...     ["the cat sits on the mat", "a cat is right here"],
        ... )
        >>> round(report["bleu"]["score"], 6)
        0.273385
        >>> round(report["scaled"]["chrf"], 3)      # chrF reported as a percentage
        52.672
        >>> 0.0 <= report["cider"]["score"] <= 10.0  # CIDEr keeps its own scale
        True
    """
    cand_list = _as_candidates(candidates)
    ref_groups = _paired_references(cand_list, references)
    first_refs = [refs[0] for refs in ref_groups]

    bleu_out = corpus_bleu(
        cand_list, ref_groups, max_order=max_order, smooth=smooth, tokenizer=tokenizer,
    )
    chrf_out = chrf(
        cand_list, ref_groups, char_order=char_order, word_order=word_order,
        beta=chrf_beta, tokenizer=tokenizer,
    )
    # ROUGE-1, ROUGE-2 and ROUGE-L all go through the corpus-pooled form
    # unconditionally. Dispatching on `len(cand_list) == 1` to use the sentence
    # function instead would make the reported number change discontinuously
    # with corpus size: a one-candidate report would be sentence-pooled and a
    # two-candidate report corpus-pooled, and for one candidate the two are the
    # same only by accident. `rouge_lsum` is already corpus-pooled; these three
    # must match it in kind.
    rouge1 = corpus_rouge_n(cand_list, first_refs, n=1, tokenizer=tokenizer)
    rouge2 = corpus_rouge_n(cand_list, first_refs, n=2, tokenizer=tokenizer)
    rouge_l_out = corpus_rouge_l(cand_list, first_refs, beta=rouge_beta, tokenizer=tokenizer)
    rouge_lsum_out = rouge_lsum(
        cand_list, ref_groups, beta=rouge_beta, tokenizer=tokenizer,
    )

    # METEOR gets EVERY reference, not just the first. This was `refs[0]` and
    # it was a silent wrong answer: `meteor` itself takes the best reference,
    # so on a corpus whose second reference is the matching one the report
    # returned 0.0 where `meteor` returns 0.996 -- and chrF and CIDEr in this
    # same call were using all their references, so the report was internally
    # inconsistent as well as wrong. Pinned by
    # `TestGenerationReport::test_meteor_uses_every_reference_not_just_the_first`.
    #
    # The score is the MEAN of the per-sentence scores, which is the correct
    # corpus aggregation for this one metric: METEOR is a sentence-level
    # measurement whose fragmentation penalty is applied per sentence, so there
    # is nothing to pool. Every other metric here pools instead -- BLEU over
    # n-gram counts, chrF over match counts, CIDEr over per-order g_n. The
    # difference is a property of the metrics, not an inconsistency, and the
    # report's docstring now says which is which.
    meteor_scores = [
        meteor(cand, refs, beta=meteor_beta, gamma=meteor_gamma, tokenizer=tokenizer)
        for cand, refs in zip(cand_list, ref_groups)
    ]
    meteor_out = {
        "score": float(np.mean([m["score"] for m in meteor_scores])),
        "matches": int(sum(m["matches"] for m in meteor_scores)),
        "chunks": int(sum(m["chunks"] for m in meteor_scores)),
    }

    cider_out = cider(
        cand_list, ref_groups, n=cider_n, sigma=cider_sigma,
        variant=cider_variant, tokenizer=tokenizer,
    )

    report: Dict[str, Any] = {
        "bleu": bleu_out,
        "chrf": chrf_out,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rouge_l_out,
        "rougeLsum": rouge_lsum_out,
        "meteor": meteor_out,
        "cider": cider_out,
        "scaled": {
            "bleu": bleu_out["score"] * 100.0,
            "chrf": chrf_out["score"] * 100.0,
            "rouge1": rouge1["score"] * 100.0,
            "rouge2": rouge2["score"] * 100.0,
            "rougeL": rouge_l_out["score"] * 100.0,
            "rougeLsum": rouge_lsum_out["score"] * 100.0,
            "meteor": meteor_out["score"],
            "cider": cider_out["score"],
        },
    }
    return report


def corpus_rouge_n(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        n: int,
        score: str = "fmeasure",
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, float]:
    """Pool ROUGE-N over a corpus, counting clipped overlaps and totals.

    :func:`rouge_n` scores ONE candidate against ONE reference. This pools the
    sufficient statistics instead of averaging per-example F-scores, for the
    same reason chrF pools: a mean of per-sentence scores weights a one-token
    candidate equally with a one-hundred-token one.

    Multiple references are resolved by taking the FIRST for each candidate, as
    ROUGE-N is a single-reference metric.

    Args:
        candidates: One candidate string, or one per example.
        references: One reference string per candidate, or one reference group
            per candidate.
        n: N-gram order.
        score: One of :data:`ROUGE_SCORE_TYPES` -- which quantity ``"score"``
            carries.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.

    Returns:
        The same keys :func:`rouge_n` returns, pooled over the corpus.

    Raises:
        ValueError: If the candidate and reference lengths disagree, either
            side is empty, ``n < 1``, or ``score`` is unknown.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import corpus_rouge_n
        >>> # POOLED, not averaged: the 4-token reference contributes four
        >>> # reference tokens to the recall denominator regardless of how the
        >>> # 1-token one scores.
        >>> out = corpus_rouge_n(["a b c d", "z"], [["a b c d"], ["y"]], n=1)
        >>> round(out["recall"], 6)
        0.8
    """
    n_int = _validate_positive_int(n, "n")
    score_name = _check_choice(score, "score", ROUGE_SCORE_TYPES)
    cand_list = _as_candidates(candidates)
    ref_groups = _paired_references(cand_list, references)
    first_refs = [group[0] for group in ref_groups]

    tokenize = _check_tokenizer(tokenizer)
    clipped = cand_total = ref_total = 0
    for candidate, reference in zip(cand_list, first_refs):
        c, ct, rt = _order_counts(
            tokenize(candidate), [tokenize(reference)], n_int,
        )
        clipped += c
        cand_total += ct
        ref_total += rt[0]
    precision = clipped / cand_total if cand_total else 0.0
    recall = clipped / ref_total if ref_total else 0.0
    fmeasure = _f_score(precision, recall, 1.0)
    selected = {
        "fmeasure": fmeasure, "recall": recall, "precision": precision,
    }[score_name]
    return {
        "score": float(selected),
        "precision": float(precision),
        "recall": float(recall),
        "fmeasure": float(fmeasure),
        "n": n_int,
    }


def corpus_rouge_l(
        candidates: Union[str, Sequence[str]],
        references: Union[str, Sequence[Any]],
        beta: float = DEFAULT_ROUGE_L_BETA,
        tokenizer: Optional[Tokenizer] = None,
) -> Dict[str, float]:
    """Pool ROUGE-L over a corpus, summing LCS lengths and token counts.

    :func:`rouge_l` scores ONE candidate against ONE reference, so it is the
    wrong shape for a corpus. This pools the sufficient statistics instead of
    averaging per-example F-scores, which is the same distinction chrF draws:
    a mean of per-example scores weights a two-token example equally with a
    two-hundred-token one. :func:`generation_report` uses this, and it is what
    the factory registry exposes as ``rouge_l`` so every registry entry takes
    the same ``(candidates, references)`` arguments.

    Multiple references are resolved by taking the FIRST for each candidate, as
    ROUGE-L is a single-reference metric; use :func:`rouge_lsum` to use them
    all.

    Args:
        candidates: One candidate string, or one per example.
        references: One reference string per candidate, or one reference group
            per candidate.
        beta: Recall weight.
        tokenizer: Tokenizer callable, or ``None`` for whitespace splitting.

    Returns:
        The same keys :func:`rouge_l` returns, pooled over the corpus.

    Raises:
        ValueError: If the candidate and reference lengths disagree, either
            side is empty, or ``beta <= 0``.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import corpus_rouge_l
        >>> # POOLED, not averaged: `lcs` sums across the whole corpus, so a
        >>> # three-example corpus of 4-token exact matches reports 12.
        >>> corpus_rouge_l(["a b c d", "e f g h"], [["a b c d"], ["e f g h"]])["lcs"]
        8
    """
    cand_list = _as_candidates(candidates)
    ref_groups = _paired_references(cand_list, references)
    first_refs = [group[0] for group in ref_groups]

    beta_f = _validate_positive_float(beta, "beta")
    tokenize = _check_tokenizer(tokenizer)
    total_lcs = total_cand = total_ref = 0
    # zip over the NORMALISED lists. The incoming `candidates`/`references` may
    # be a bare string, which zip would otherwise iterate character by
    # character -- and `_paired_references` above has already accepted it.
    for candidate, reference in zip(cand_list, first_refs):
        cand_tokens = tokenize(candidate)
        ref_tokens = tokenize(reference)
        lcs_len, _ = _lcs_alignment(cand_tokens, ref_tokens)
        total_lcs += lcs_len
        total_cand += len(cand_tokens)
        total_ref += len(ref_tokens)
    precision = total_lcs / total_cand if total_cand else 0.0
    recall = total_lcs / total_ref if total_ref else 0.0
    return {
        "score": float(_f_score(precision, recall, beta_f)),
        "precision": float(precision),
        "recall": float(recall),
        "fmeasure": float(_f_score(precision, recall, 1.0)),
        "lcs": int(total_lcs),
        "beta": beta_f,
    }


#: The registry behind :func:`create_generation_metric`. Its key set, its
#: aliases and each entry's declared parameters are PUBLIC SURFACE -- adding,
#: renaming or removing one is a breaking change, not a cleanup. Pinned by
#: ``tests/test_metrics/test_the_generation_metric_registry_has_not_drifted.py``.
GENERATION_METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "bleu": {
        # `corpus_bleu`, not the bare-float `bleu` shorthand: the registry's
        # contract is that every entry returns the same result SHAPE, and a
        # caller doing `create_generation_metric(name)(c, r)["score"]` would get
        # a `TypeError` on this one entry and a dict on every other.
        "function": corpus_bleu,
        "required_params": [],
        "optional_params": {
            "max_order": DEFAULT_BLEU_MAX_ORDER,
            "smooth": "exp",
            "tokenizer": None,
        },
    },
    "chrf": {
        "function": chrf,
        "required_params": [],
        "optional_params": {
            "char_order": DEFAULT_CHRF_CHAR_ORDER,
            "word_order": DEFAULT_CHRF_WORD_ORDER,
            "beta": DEFAULT_CHRF_BETA,
            "remove_whitespace": True,
            "tokenizer": None,
        },
    },
    "cider": {
        "function": cider,
        "required_params": [],
        "optional_params": {
            "n": DEFAULT_BLEU_MAX_ORDER,
            "sigma": DEFAULT_CIDER_SIGMA,
            "variant": "d",
            "tokenizer": None,
        },
    },
    "rouge_l": {
        # The CORPUS-pooled form, not the single-pair `rouge_l`. Every registry
        # entry takes `(candidates, references)`, so pointing this at the
        # single-pair function would give it a different signature from every
        # other entry and a corpus caller would hit a `TypeError` inside the
        # factory.
        "function": corpus_rouge_l,
        "required_params": [],
        "optional_params": {"beta": DEFAULT_ROUGE_L_BETA, "tokenizer": None},
    },
    "rouge_n": {
        "function": corpus_rouge_n,
        "required_params": ["n"],
        "optional_params": {"score": "fmeasure", "tokenizer": None},
    },
    "rouge_lsum": {
        "function": rouge_lsum,
        "required_params": [],
        "optional_params": {"beta": DEFAULT_ROUGE_L_BETA, "tokenizer": None},
    },
}


def create_generation_metric(name: str, **kwargs: Any) -> Callable[..., Any]:
    """Build a metric callable from its registry name.

    Mirrors the config-driven factories elsewhere in this library, and follows
    their one hard rule: **an undeclared parameter raises, it is never filtered
    out.** A factory that silently drops an unrecognised key hands back a
    perfectly valid metric carrying a default nobody asked for, and the call
    site is indistinguishable from one that was honoured.

    Args:
        name: A registry key from :data:`GENERATION_METRIC_REGISTRY`.
        **kwargs: Parameters for the underlying function. Every key must be
            declared by that entry, as required or optional.

    Returns:
        A callable with the same ``(candidates, references)`` signature as the
        underlying metric and the supplied parameters already bound, via
        :func:`functools.partial`. It is a partial rather than a wrapper class so
        its ``repr`` names the metric it dispatches to, which is what a log line
        or a debugger is actually read for.

        **Every entry returns the same result shape**: a dict carrying at least
        ``score`` (plus that metric's components). Two entries return the bare
        float shorthands -- ``bleu`` and ``sentence_bleu`` -- and are therefore
        NOT registered; the registry points ``"bleu"`` at :func:`corpus_bleu`
        instead. A caller may rely on ``["score"]`` working for any name it
        builds.

    Raises:
        ValueError: If ``name`` is unknown, if a parameter is not declared by
            that entry, or if a required parameter is missing. Every message
            lists what is accepted.

    Example:
        >>> from dl_techniques.metrics.generation_metrics import create_generation_metric
        >>> chrF2 = create_generation_metric("chrf", char_order=3, word_order=2)
        >>> chrF2(["a b c", "d e f", "g h i"], ["a b c", "d e f", "g h i"])["score"]
        1.0
    """
    if not isinstance(name, str):
        raise ValueError(
            f"generation metric name must be a string; got {type(name).__name__}"
        )
    key = name.strip().lower()
    if key not in GENERATION_METRIC_REGISTRY:
        raise ValueError(
            f"Unknown generation metric {name!r}. "
            f"Available: {sorted(GENERATION_METRIC_REGISTRY)}"
        )

    entry = GENERATION_METRIC_REGISTRY[key]
    required = list(entry["required_params"])
    optional = dict(entry["optional_params"])
    declared = set(required) | set(optional)

    unsupported = sorted(set(kwargs) - declared)
    if unsupported:
        raise ValueError(
            f"create_generation_metric({key!r}): {len(unsupported)} unsupported "
            f"parameter(s) {unsupported}. Accepted: {sorted(declared)}"
        )
    missing = sorted(set(required) - set(kwargs))
    if missing:
        raise ValueError(
            f"create_generation_metric({key!r}): requires {missing}. "
            f"Optional: {sorted(optional)}"
        )

    logger.info(
        f"create_generation_metric: {key} "
        f"({ {**optional, **kwargs} })",
    )
    params = {**optional, **kwargs}
    return partial(entry["function"], **params)