"""Tests for ``dl_techniques.metrics.generation_metrics``.

The module's whole claim is numerical, so the central guard is an ORACLE: every
metric is compared against a closed-form computation transcribed independently
in this file -- written from the formulas in the papers and from the reference
implementations, deliberately in a different code shape from the module's
(helper functions, explicit loops, no shared helper). An oracle written by the
same hand in the same style is a second copy, not a check.

Tolerances. Every bound here is ``atol=1e-12, rtol=0`` on float64 arithmetic
over small integers. float64 unit roundoff is ~1.1e-16, so a sum of a few
hundred terms carries a relative error of order 1e-14; 1e-12 sits roughly two
orders above that floor and roughly eight orders below any defect these guards
are aimed at (a dropped clip, a sum where a max belongs, an off-by-one in a
chunk boundary -- all of which move a score by O(1e-1) or give exactly 0.0).
Restoration-shaped comparisons use ``atol=0.0``.

Guards, and what each one is for:

* T1 -- identity. Every metric returns its maximum on candidate == reference.
  NOTE the METEOR and CIDEr arms, which do NOT return 1.0 and 10.0 respectively
  on a naive identity input, and the docstrings say why: METEOR's penalty is
  0.5 * (chunks/matches)**3, so a perfect n-token match scores 1 - 0.5/n**3;
  CIDEr's weight is log(N / max(1, df)), which is 0 for every n-gram in a
  one-example corpus. Pinning 1.0 / 10.0 there would pin the WRONG value and go
  RED on a correct implementation.
* T2 -- disjointness. Non-overlapping vocabulary scores exactly 0.0.
* T3 -- brevity penalty, BOTH ways. A short candidate is penalised; a long one
  is not, and the long arm is what a dead BP (one that always returns 1.0)
  cannot satisfy on its own.
* T4 -- corpus BLEU is NOT the mean of sentence BLEU, asserted as a measured
  DIFFERENCE. Averaging sentence scores is the most common BLEU bug; a test that
  pinned the two as equal would go RED the moment the pooling was fixed.
* T5 -- smoothing. Each method's exact behaviour on a zero precision, plus the
  "no match anywhere stays 0.0 under every method" rule.
* T6 -- ROUGE against a hand-computed clipped overlap and a hand-computed LCS.
* T7 -- chrF's EFFECTIVE-ORDER averaging, which is what makes a short string
  score 1.0 instead of ~0.33, and the corpus-pooled / best-reference policy.
* T8 -- METEOR's chunk rule, on hand-built separations, plus the alpha/beta
  identity from D-002 and the stemmed-match path.
* T9 -- CIDEr against an independent tf-idf + clipped-cosine oracle.
* T10 -- knob sweep: every constructor parameter must move the number, with an
  anti-vacuity twin. Closes the closure-capture trap with a default argument.
* T11 -- the per-metric multi-reference policies, which DIFFER (BLEU max-pools
  per n-gram, chrF picks the best reference, CIDEr averages over them).
* T12 -- input contract: every documented ValueError.
* T13 -- generation_report keys and scaling.
* T14 -- the factory registry, including the raise-on-undeclared-key rule.
* T15 -- RED proofs: the injected defects and the numbers they produce.
"""

from __future__ import annotations

import inspect
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

from dl_techniques.metrics.generation_metrics import (
    BLEU_SMOOTHING_METHODS,
    GENERATION_METRIC_REGISTRY,
    _count_chunks,
    _meteor_alignment,
    bleu,
    chrf,
    cider,
    corpus_rouge_l,
    corpus_rouge_n,
    corpus_bleu,
    create_generation_metric,
    generation_report,
    meteor,
    ngrams,
    rouge_l,
    rouge_lsum,
    rouge_n,
    sentence_bleu,
    split_sentences,
    whitespace_tokenize,
)

# House style: a pure-absolute bound states rtol=0 explicitly, because
# assert_allclose's default rtol=1e-7 otherwise contributes silently.
ATOL = 1e-12


# ---------------------------------------------------------------------
# Independent oracles. Written from the formulas, NOT by calling the module.
# ---------------------------------------------------------------------


def _oracle_ngram_counts(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    """Multiset of n-grams, counted by explicit slicing (the module builds its
    counters the other way round, from an n-gram list)."""
    out: Dict[Tuple[str, ...], int] = {}
    for start in range(len(tokens) - n + 1):
        gram = tuple(tokens[start:start + n])
        out[gram] = out.get(gram, 0) + 1
    return out


def oracle_bleu(
        candidate: str,
        reference: str,
        max_order: int = 4,
        smooth: str = "exp",
) -> float:
    """Single-sentence BLEU, written straight from Papineni + Chen & Cherry.

    ``p_n = clipped_n / total_n``; a clipped count is
    ``sum over distinct candidate n-grams of min(hyp_count, ref_count)``.
    ``exp`` (NIST / mteval-v13a) replaces a zero precision with
    ``1 / (2**k * total_n)`` where ``k`` counts zeros so far; ``floor`` with
    ``1 / (2**k * total_n)`` where ``k`` counts zeros so far; ``floor`` with
    ``delta / total_n``; ``add_k`` with ``+1`` on both counts for ``n > 1``.
    """
    hyp = candidate.split()
    ref = reference.split()

    correct: List[int] = []
    total: List[int] = []
    for order in range(1, max_order + 1):
        hyp_counts = _oracle_ngram_counts(hyp, order)
        ref_counts = _oracle_ngram_counts(ref, order)
        matched = 0
        hyp_total = 0
        for gram, count in hyp_counts.items():
            hyp_total += count
            matched += min(count, ref_counts.get(gram, 0))
        correct.append(matched)
        total.append(hyp_total)

    if not any(correct):
        return 0.0

    precisions: List[float] = []
    running = 1.0
    for index in range(max_order):
        n = index + 1
        c, t = correct[index], total[index]
        if smooth == "add_k" and n > 1:
            c, t = c + 1, t + 1
        if t == 0:
            precisions.append(0.0)
            continue
        if c == 0:
            if smooth == "exp":
                running *= 2.0
                precisions.append(1.0 / (running * t))
            elif smooth == "floor":
                precisions.append(0.1 / t)
            else:
                precisions.append(0.0)
        else:
            precisions.append(c / t)

    bp = 1.0
    if len(hyp) < len(ref):
        bp = math.exp(1 - len(ref) / len(hyp)) if hyp else 0.0
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_order)


def oracle_lcs(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence by exhaustive search over all subsequences.

    O(2**n) and therefore only usable on the three-to-five token fixtures here,
    which is the point: it shares no code with the module's rolling-row
    dynamic program, so the two agreeing is evidence rather than tautology.
    """
    from itertools import combinations

    best = 0
    for size in range(len(a) + 1):
        for idx in combinations(range(len(a)), size):
            sub = [a[i] for i in idx]
            if _is_subsequence(sub, b):
                best = max(best, size)
    return best


def _is_subsequence(candidate: Sequence[str], target: Sequence[str]) -> bool:
    it = iter(target)
    return all(any(item == other for other in it) for item in candidate)


def oracle_cider(
        candidates: Sequence[str],
        references: Sequence[Sequence[str]],
        n: int = 4,
        sigma: float = 6.0,
        variant: str = "d",
) -> float:
    """CIDEr, transcribed from Vedantam et al. and the reference scorer.

    weight ``= tf * (log N - log(max(1, df)))``; similarity is the CLIPPED
    inner product ``sum_{g in hyp} min(h_w, r_w) * r_w`` over L2-normalised
    vectors; per order the per-reference values are averaged, then the orders
    are averaged, then scaled by 10.
    """
    num_examples = len(candidates)
    log_n = math.log(num_examples)

    doc_freq: List[Dict[Tuple[str, ...], int]] = [dict() for _ in range(n)]
    for refs in references:
        for order in range(1, n + 1):
            grams = set()
            for ref in refs:
                grams |= set(_oracle_ngram_counts(ref.split(), order))
            for gram in grams:
                doc_freq[order - 1][gram] = doc_freq[order - 1].get(gram, 0) + 1

    def vector(text: str) -> Tuple[List[Dict[Tuple[str, ...], float]], List[float]]:
        toks = text.split()
        vecs, norms = [], []
        for order in range(1, n + 1):
            counts = _oracle_ngram_counts(toks, order)
            weights = {
                gram: float(count) * (log_n - math.log(max(1.0, doc_freq[order - 1].get(gram, 0))))
                for gram, count in counts.items()
            }
            vecs.append(weights)
            norms.append(math.sqrt(sum(w * w for w in weights.values())))
        return vecs, norms

    delta = sum(len(r.split()) for refs in references for r in refs) / num_examples

    per_order_total = [0.0] * n
    for cand, refs in zip(candidates, references):
        hyp_vec, hyp_norm = vector(cand)
        cand_len = len(cand.split())
        for order in range(n):
            values = []
            for ref in refs:
                ref_vec, ref_norm = vector(ref)
                if hyp_norm[order] == 0.0 or ref_norm[order] == 0.0:
                    sim = 0.0
                else:
                    dot = sum(
                        min(w, ref_vec[order].get(g, 0.0)) * ref_vec[order].get(g, 0.0)
                        for g, w in hyp_vec[order].items()
                    )
                    sim = dot / (hyp_norm[order] * ref_norm[order])
                delta_ij = len(ref.split()) - cand_len
                penalty = math.exp(-(delta_ij ** 2) / (2 * sigma ** 2))
                if variant == "d":
                    penalty *= math.exp(-((cand_len - delta) ** 2) / (2 * sigma ** 2))
                values.append(sim * penalty)
            per_order_total[order] += sum(values) / len(values) if values else 0.0

    # Mean over ORDERS, then scale. Summing instead would return 10x the score
    # at n = 4 -- an oracle that is uniformly 4x wrong still "disagrees" with a
    # correct implementation, so getting this wrong produces a confusing RED
    # rather than an obviously broken one.
    cider_n = [total / num_examples for total in per_order_total]
    return 10.0 * (sum(cider_n) / len(cider_n))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator for every synthetic fixture below.

    ``np.random.default_rng`` rather than the global seed: these tests read no
    process-global Keras RNG, so they do not couple to pytest collection order.
    """
    return np.random.default_rng(20261018)


def _random_corpus(rng: np.random.Generator, n_examples: int = 12, lo: int = 3, hi: int = 9):
    """A corpus of random token sequences over a shared, deliberately small
    vocabulary -- small enough that n-gram overlap is common, so the metrics are
    not trivially 0.0 on everything.

    References come back as a list of SINGLETON reference GROUPS, not a flat list
    of strings. A flat list is one shape too short: CIDEr's document frequency
    iterates ``for refs in references: for ref in refs``, so handed a flat list
    it silently iterates each string a CHARACTER at a time and computes
    document frequencies over single characters -- and then agrees with the
    oracle, because the oracle written from the same mistake makes the same
    mistake. The two agreeing is a shared bug, not a check.
    """
    vocab = [f"w{i}" for i in range(8)]
    candidates, references = [], []
    for _ in range(n_examples):
        length = int(rng.integers(lo, hi + 1))
        candidates.append(" ".join(rng.choice(vocab, size=length)))
        length = int(rng.integers(lo, hi + 1))
        references.append([" ".join(rng.choice(vocab, size=length))])
    return candidates, references


# ---------------------------------------------------------------------
# T1 -- identity, with the two metrics that are NOT 1.0 on their best input
# ---------------------------------------------------------------------


class TestIdentity:
    """Every metric at its maximum. The METEOR and CIDEr arms document measured
    departures from the naive expectation."""

    def test_bleu_is_one_on_an_exact_match(self) -> None:
        out = corpus_bleu(["a b c d"], [["a b c d"]])
        assert out["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert out["bp"] == 1.0
        assert out["precisions"] == [1.0, 1.0, 1.0, 1.0]

    def test_chrf_is_one_on_an_exact_match(self) -> None:
        assert chrf("the cat sat", "the cat sat")["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert chrf("the cat sat", "the cat sat", word_order=2)["score"] == pytest.approx(
            1.0, abs=ATOL, rel=0.0,
        )

    def test_rouge_is_one_on_an_exact_match(self) -> None:
        assert rouge_n("a b c d", "a b c d", n=2)["fmeasure"] == 1.0
        assert rouge_n("a b c d", "a b c d", n=2)["recall"] == 1.0
        assert rouge_l("a b c d", "a b c d")["fmeasure"] == 1.0
        assert rouge_lsum(["a b c d"], [["a b c d"]])["fmeasure"] == 1.0

    def test_meteor_is_not_one_and_the_gap_is_exact(self) -> None:
        """MEASURED: the reference penalty is ``gamma * (chunks/matches)**beta``
        with ``chunks = 1`` and ``matches = n``, so a perfect n-token match
        scores ``1 - 0.5 / n**3`` -- 0.981481... at n = 3, and 0.5 at n = 1.

        Pinning 1.0 here would pin the value a BUGGY metric returns, and would
        go RED the moment the penalty was implemented correctly.
        """
        for n_tokens in (1, 3, 5):
            text = " ".join(f"w{i}" for i in range(n_tokens))
            out = meteor(text, text)
            expected = 1.0 - 0.5 / (n_tokens ** 3)
            assert out["matches"] == n_tokens
            assert out["chunks"] == 1
            assert out["score"] == pytest.approx(expected, abs=ATOL, rel=0.0), (
                f"n={n_tokens}: meteor scored {out['score']}, penalty is not "
                f"0.5 * (1/{n_tokens})**3"
            )
        assert meteor("a", "a")["score"] == pytest.approx(0.5, abs=ATOL, rel=0.0)

    def test_cider_is_ten_when_idf_has_contrast(self) -> None:
        """MEASURED: CIDEr's weight is ``log(N / max(1, df))``, so a one-example
        corpus scores exactly 0.0 and a corpus where every n-gram is shared
        scores exactly 0.0 too. Three examples with DISJOINT vocabularies give
        df = 1 against N = 3, the maximum possible weight, and a perfect match.

        ``n=2`` rather than the default 4 because 3-token candidates have no
        4-grams: those orders contribute 0.0 and are still counted in the mean,
        which caps a perfect corpus at 7.5.
        """
        corpus = ["a b c", "d e f", "g h i"]
        assert cider(corpus, corpus, n=2)["score"] == pytest.approx(10.0, abs=ATOL, rel=0.0)

    def test_cider_is_zero_on_a_single_example_corpus(self) -> None:
        """The other half of the CIDEr pair: a corpus too small for IDF to mean
        anything. Without this arm the "cider reaches its maximum" test could be
        satisfied by a scorer that ignores the IDF term entirely."""
        assert cider(["a b c d"], ["a b c d"])["score"] == 0.0
        # ... and a two-example corpus where both share every n-gram is also 0.
        assert cider(["a b c d", "a b c d"], ["a b c d", "a b c d"])["score"] == 0.0


# ---------------------------------------------------------------------
# T2 -- disjointness
# ---------------------------------------------------------------------


class TestDisjointVocabulariesScoreZero:
    """Non-overlapping n-grams score exactly 0.0, not a small positive number
    left behind by a smoothing term or a division."""

    def test_every_metric_is_zero(self) -> None:
        assert bleu(["a b c"], [["x y z"]]) == 0.0
        assert chrf("abc", "xyz", char_order=3)["score"] == 0.0
        assert rouge_n("a b c", "x y z", n=1)["fmeasure"] == 0.0
        assert rouge_l("a b c", "x y z")["fmeasure"] == 0.0
        assert rouge_lsum(["a b c"], [["x y z"]])["fmeasure"] == 0.0
        assert meteor("a b c", "x y z")["score"] == 0.0

    def test_an_empty_candidate_scores_zero_rather_than_raising(self) -> None:
        assert bleu([""], [[""]]) == 0.0
        assert meteor("", "a b c")["score"] == 0.0
        assert chrf("", "", char_order=3)["score"] == 0.0


# ---------------------------------------------------------------------
# T3 -- brevity penalty, both ways
# ---------------------------------------------------------------------


class TestBrevityPenalty:
    """BP must fall below 1 for a short candidate AND stay at 1 for a long one.

    The second arm is not redundant with the first: an implementation that
    dropped the penalty altogether passes "short candidate scores below the
    unclipped geometric mean" and fails here, and one that applied the penalty
    unconditionally fails the first.
    """

    def test_a_short_candidate_is_penalised(self) -> None:
        out = corpus_bleu(["a b c d e f"], [["a b c d e f g h"]])
        expected_bp = math.exp(1 - 8 / 6)
        assert out["bp"] == pytest.approx(expected_bp, abs=ATOL, rel=0.0)
        assert out["bp"] < 1.0
        assert out["ratio"] == pytest.approx(6 / 8, abs=ATOL, rel=0.0)

    def test_a_long_candidate_is_not_penalised(self) -> None:
        """The claim is about BP ALONE, and deliberately not about the score.

        MEASURED: the long candidate's BP is exactly 1.0 while its SCORE is
        *lower* than the short one's (0.6804 against 0.7165) -- the two extra
        unmatched tokens cost precision at every order, which outweighs the BP
        the short candidate pays. Asserting a score ordering here would be
        asserting something BLEU does not promise, and it would go RED on a
        correct implementation.
        """
        short = corpus_bleu(["a b c d e f"], [["a b c d e f g h"]])
        long_ = corpus_bleu(["a b c d e f g h"], [["a b c d e f"]])
        assert short["bp"] < 1.0
        assert long_["bp"] == 1.0
        # The score ordering is documented above as NOT being a property; pin
        # the measured values so a future change that silently starts ordering
        # scores by BP is visible.
        assert short["score"] == pytest.approx(0.7165313105737893, abs=ATOL, rel=0.0)
        assert long_["score"] == pytest.approx(0.6803749333171202, abs=ATOL, rel=0.0)

    def test_an_exactly_equal_length_candidate_is_not_penalised(self) -> None:
        # The branches agree at c == r, which is why the reference compares `<`.
        assert corpus_bleu(["a b c d"], [["w x y z"]])["bp"] == 1.0

    def test_the_score_is_bp_times_the_geometric_mean(self) -> None:
        """The two halves compose, read off the components -- so a defect that
        replaced the BP with 1.0, or dropped it from the product, is caught by
        a number rather than by an implementation detail."""
        out = corpus_bleu(["a b c d e f"], [["a b c d e f g h"]])
        geo = float(np.exp(np.mean(np.log(out["precisions"]))))
        assert out["score"] == pytest.approx(out["bp"] * geo, abs=ATOL, rel=0.0)


# ---------------------------------------------------------------------
# T4 -- corpus BLEU is not the mean of sentence BLEU
# ---------------------------------------------------------------------


class TestCorpusIsNotTheMeanOfSentences:
    """The pooling distinction, asserted as a measured DIFFERENCE.

    MEASURED on the fixture below: corpus BLEU 0.531... against a mean of
    sentence BLEU 0.200... -- a gap of 0.331, three orders of magnitude above
    this test's 1e-12 bound. The gap is not round-off: half the corpus is a
    perfect match and half matches nothing, and the perfect half's n-grams are
    not diluted by the empty half's zeros when the statistics are pooled.
    """

    # MEASURED on this corpus: pooled 0.3221693140510760 against a mean of
    # 0.0712145010660390 -- a gap of 0.251, fourteen orders of magnitude above
    # this file's 1e-12 bound. These are hand-built to make the gap large
    # rather than typical: the first sentence is long and heavily overlapping,
    # the last is a single token that matches almost nothing, and pooling
    # weights them by n-gram COUNT while averaging weights them equally.
    @pytest.fixture
    def mixed_corpus(self):
        return (
            ["d f d f f", "f b b", "c f"],
            ["f f", "f e a f b b", "a c a"],
        )

    def test_the_two_differ_materially(self, mixed_corpus) -> None:
        candidates, references = mixed_corpus
        pooled = bleu(candidates, [[r] for r in references])
        averaged = float(np.mean([
            sentence_bleu(c, [r]) for c, r in zip(candidates, references)
        ]))
        assert pooled == pytest.approx(0.3221693140510760, abs=ATOL, rel=0.0)
        assert averaged == pytest.approx(0.0712145010660390, abs=ATOL, rel=0.0)
        assert pooled - averaged > 0.2, (
            f"corpus {pooled} and mean-of-sentences {averaged} barely differ; the "
            f"distinction this test exists to pin has collapsed"
        )

    def test_pooling_weights_sentences_by_their_ngram_count(self, mixed_corpus) -> None:
        """The mechanism behind the gap, read off the components. The pooled
        precisions are non-zero at EVERY order -- the long overlapping sentence
        supplies the counts that keep orders 2-4 alive -- while the mean of
        sentence scores is dragged down by the short, near-matching last one."""
        candidates, references = mixed_corpus
        out = corpus_bleu(candidates, [[r] for r in references])
        assert out["precisions"] == [
            pytest.approx(v, abs=ATOL, rel=0.0) for v in (0.6, 0.42857142857142855,
                                                           0.25, 0.25)
        ]
        assert out["bp"] == pytest.approx(0.9048374180359595, abs=ATOL, rel=0.0)


# ---------------------------------------------------------------------
# T5 -- smoothing
# ---------------------------------------------------------------------


class TestSmoothing:
    """Each method's exact effect on a zero precision, and the rule that no
    match anywhere is 0.0 under all of them."""

    ZERO_P4 = ["the cat is on the mat"]
    ONE_REF = ["the cat sits on the mat"]          # shares 3 bigrams, 1 trigram, 0 4-grams

    def test_exp_is_the_nist_geometric_rule(self) -> None:
        out = corpus_bleu(self.ZERO_P4, [self.ONE_REF], smooth="exp")
        # Only p4 is zero: k = 1, total_4 = 3, so 1 / (2 * 3).
        assert out["precisions"][3] == pytest.approx(1 / 6, abs=ATOL, rel=0.0)
        assert out["precisions"][:3] == [
            pytest.approx(5 / 6, abs=ATOL, rel=0.0),
            pytest.approx(3 / 5, abs=ATOL, rel=0.0),
            pytest.approx(1 / 4, abs=ATOL, rel=0.0),
        ]

    def test_floor_is_delta_over_total(self) -> None:
        out = corpus_bleu(self.ZERO_P4, [self.ONE_REF], smooth="floor")
        assert out["precisions"][3] == pytest.approx(0.1 / 3, abs=ATOL, rel=0.0)

    def test_add_k_leaves_order_one_alone(self) -> None:
        plain = corpus_bleu(self.ZERO_P4, [self.ONE_REF], smooth="none")
        smoothed = corpus_bleu(self.ZERO_P4, [self.ONE_REF], smooth="add_k")
        assert smoothed["precisions"][0] == plain["precisions"][0], (
            "add_k is defined for n > 1 only; order 1 must be untouched"
        )
        assert smoothed["precisions"][1] != plain["precisions"][1]

    def test_none_leaves_the_zero_and_the_score_is_exactly_zero(self) -> None:
        out = corpus_bleu(self.ZERO_P4, [self.ONE_REF], smooth="none")
        assert out["precisions"][3] == 0.0
        assert out["score"] == 0.0

    def test_no_match_anywhere_is_zero_under_every_method(self) -> None:
        """Smoothing rescues a zero PRECISION; it must not rescue a candidate
        that matched nothing, which would otherwise come back as a small
        positive score and read as 'almost right'."""
        for method in BLEU_SMOOTHING_METHODS:
            out = corpus_bleu(["x y z"], [["a b c"]], smooth=method)
            assert out["score"] == 0.0, f"smooth={method} gave {out['score']}"
            assert out["precisions"] == [0.0, 0.0, 0.0, 0.0]

    def test_smoothing_never_changes_a_non_zero_precision(self) -> None:
        """A metric with no missing n-gram scores identically under every
        method -- the property that makes these smoothing rather than a
        different metric."""
        base = corpus_bleu(["a b c d"], [["a b c d"]], smooth="none")["score"]
        for method in BLEU_SMOOTHING_METHODS:
            assert corpus_bleu(["a b c d"], [["a b c d"]], smooth=method)["score"] == \
                pytest.approx(base, abs=ATOL, rel=0.0)

    def test_the_methods_actually_differ(self) -> None:
        """Anti-vacuity: if every method returned the same number, the three
        assertions above would pass for a reason unrelated to smoothing."""
        scores = {
            method: corpus_bleu(self.ZERO_P4, [self.ONE_REF], smooth=method)["score"]
            for method in BLEU_SMOOTHING_METHODS
        }
        assert len(set(scores.values())) == len(BLEU_SMOOTHING_METHODS), scores


# ---------------------------------------------------------------------
# T6 -- ROUGE
# ---------------------------------------------------------------------


class TestRouge:

    def test_clipped_overlap_against_a_hand_count(self) -> None:
        # Candidate repeats "a" three times; the reference has it once.
        out = rouge_n("a a a b", "a c d", n=1)
        assert out["precision"] == pytest.approx(1 / 4, abs=ATOL, rel=0.0)
        assert out["recall"] == pytest.approx(1 / 3, abs=ATOL, rel=0.0)

    def test_lcs_against_exhaustive_search(self, rng) -> None:
        vocab = list("abcde")
        for _ in range(40):
            a = [str(v) for v in rng.choice(vocab, size=int(rng.integers(1, 7)))]
            b = [str(v) for v in rng.choice(vocab, size=int(rng.integers(1, 7)))]
            expected = oracle_lcs(a, b)
            got = rouge_l(" ".join(a), " ".join(b))["lcs"]
            assert got == expected, f"lcs({a}, {b}) = {got}, exhaustive search says {expected}"

    def test_repetition_does_not_inflate_recall(self) -> None:
        """ROPE-TWIST-shaped: a candidate that repeats the reference must not
        beat the exact match on recall, which is the clip's whole purpose."""
        assert rouge_n("a a a a a", "a b c d", n=1)["recall"] == \
            rouge_n("a", "a b c d", n=1)["recall"]

    def test_lcs_is_order_sensitive_and_bigrams_are_not(self) -> None:
        reversed_text = rouge_l("d c b a", "a b c d")["lcs"]
        assert reversed_text == 1
        # ROUGE-1 cannot see the reversal at all.
        assert rouge_n("d c b a", "a b c d", n=1)["fmeasure"] == 1.0

    def test_lsum_unions_across_reference_sentences(self) -> None:
        """The union is the whole point of Lsum, and it is a claim about an
        EXACT integer -- not a bound that any plausible value satisfies.

        MEASURED on ``["a b c"]`` against ``["a b", "b c"]``: the middle token
        ``b`` matches BOTH reference sentences. Unioned it is counted once, so
        the LCS is **3**. A per-sentence-pair SUM counts it twice and returns
        **4**, which also drives recall to 4/4 = 1.0 and precision to 4/3.

        The previous version of this test asserted ``lcs <= recall * 8 + 1``,
        which every one of those values satisfies. Injecting the sum left the
        whole suite green; this pins the number instead.
        """
        out = rouge_lsum(["a b c"], [["a b", "b c"]])
        assert out["lcs"] == 3, (
            f"lcs is {out['lcs']}, expected 3 -- the shared token 'b' must be "
            f"counted once, so this is scoring per-sentence-pair SUMS"
        )
        assert out["precision"] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert out["recall"] == pytest.approx(3 / 4, abs=ATOL, rel=0.0)

    def test_lsum_ignores_sentence_order_where_l_does_not(self) -> None:
        """MEASURED on ``"a . b ."`` against ``"b . a ."``: ROUGE-L scores
        **0.5** and ROUGE-Lsum scores **1.0**.

        The mechanism is the per-sentence union. Whole-string LCS must be
        MONOTONE across the entire sequence, so it cannot match ``a`` to the
        second reference sentence and then ``b`` to the first -- reversing them
        costs it half its matches. Lsum aligns each candidate sentence against
        each reference sentence independently and unions the matched candidate
        positions, so sentence order costs it nothing.

        This is why "ROUGE-L" without "sum" is not a reproducible number: on a
        reordered multi-sentence summary the two differ by 0.5, which is not a
        rounding artefact.
        """
        candidate, reference = "a . b .", "b . a ."
        as_l = rouge_l(candidate, reference)
        as_lsum = rouge_lsum([candidate], [[reference]])
        assert as_l["fmeasure"] == pytest.approx(0.5, abs=ATOL, rel=0.0)
        assert as_lsum["fmeasure"] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert as_lsum["fmeasure"] - as_l["fmeasure"] == pytest.approx(0.5, abs=ATOL, rel=0.0)

    def test_rouge_l_beta_selects_the_recall_only_form(self) -> None:
        """MEASURED on ``"a b"`` vs ``"a b c d"``: precision 1.0, recall 0.5,
        and the F-beta runs 0.8333 -> 0.6667 -> 0.5152 -> 0.5000 as beta goes
        0.5 -> 1.0 -> 4.0 -> 1000.0, converging on the pure recall.

        The fixture has to be ASYMMETRIC. On an equal-length candidate and
        reference ``P == R``, every beta collapses to the same value and the
        knob looks inert -- which is a property of the fixture, not of the knob.
        """
        scores = [rouge_l("a b", "a b c d", beta=b)["score"] for b in (0.5, 1.0, 4.0, 1000.0)]
        assert len(set(scores)) == 4, f"beta did nothing: {scores}"
        assert scores[0] > scores[1] > scores[2] > scores[3]
        # The limit is the recall, approached from above.
        assert abs(scores[-1] - 0.5) < 1e-6

    def test_score_selects_which_quantity_is_reported(self) -> None:
        out = rouge_n("a b c", "a b c d", n=1, score="recall")
        assert out["score"] == out["recall"]
        assert out["score"] != out["precision"]

    def test_sentence_splitter_splits_on_terminal_punctuation(self) -> None:
        assert split_sentences("one. two! three?") == ["one.", "two!", "three?"]
        assert split_sentences("") == []
        assert split_sentences("no punctuation") == ["no punctuation"]


# ---------------------------------------------------------------------
# T7 -- chrF
# ---------------------------------------------------------------------


class TestChrf:

    def test_orders_the_candidate_cannot_have_are_excluded_not_zeroed(self) -> None:
        """The single biggest behavioural difference from a naive mean.

        MEASURED: ``chrf("abcde", "abcde", char_order=6)`` is 1.0. Counting the
        four impossible orders as zeros gives (1 + 1 + 0 + 0 + 0 + 0) / 6 =
        0.3333, which is the number a straightforward implementation returns and
        the reason this arm exists.
        """
        assert chrf("abcde", "abcde", char_order=6)["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        naive = (2 / 6)      # only the two orders a 5-character string has
        assert naive < 0.34
        assert chrf("abcde", "abcde", char_order=6)["score"] > 2.0 * naive

    def test_a_partly_matching_char_sequence_scores_between(self) -> None:
        out = chrf("abcdefgh", "abcdwxyz", char_order=3)
        assert 0.0 < out["score"] < 1.0

    def test_statistics_pool_over_the_corpus(self) -> None:
        """MEASURED on ``["a", "b c d ... n"]``: pooled **0.012626262626**
        against a mean of per-sentence chrF of **0.5**.

        The direction is worth stating, because pooling is not a "fairer"
        average -- it is a DIFFERENT one, and here it is much lower. The one-token
        candidate is a perfect match on its own (chrF 1.0); averaged, it still
        gets half the credit of the terrible 14-token sentence. Pooled, its
        single matching character is one contribution among 14 tokens of mostly
        mismatched text, so the perfect short segment stops carrying the corpus
        on its own.

        Either direction is a legitimate corpus statistic; what is not legitimate
        is reporting one and believing you computed the other.
        """
        candidates = ["a", "b c d e f g h i j k l m n"]
        references = [["a"], ["z y x w v u t s r q p o"]]
        pooled = chrf(candidates, references)["score"]
        averaged = float(np.mean([
            chrf("a", "a")["score"],
            chrf("b c d e f g h i j k l m n", "z y x w v u t s r q p o")["score"],
        ]))
        assert pooled == pytest.approx(0.012626262626, abs=ATOL, rel=0.0)
        assert averaged == pytest.approx(0.5, abs=ATOL, rel=0.0)
        assert pooled < averaged * 0.1, (
            f"pooled {pooled} and mean {averaged} are too close; the pooling "
            f"distinction this test exists to pin has collapsed"
        )

    def test_word_order_two_gives_chrfpp(self) -> None:
        plain = chrf("the cat ran home", "the cat ran home")
        plus = chrf("the cat ran home", "the cat ran home", word_order=2)
        assert plus["word_precision"] is not None and plain["word_precision"] is None
        assert plus["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_beta_two_weights_recall_above_precision(self) -> None:
        """A short candidate has high precision and low recall, so the
        recall-weighted score must be the lower of the two. Anti-vacuity: if
        beta did nothing, the arms would agree."""
        short = "a b"
        long_ref = "a b c d e f g h i j"
        at_one = chrf(short, long_ref, beta=1.0)["score"]
        at_two = chrf(short, long_ref, beta=2.0)["score"]
        assert at_two < at_one

    def test_lowercase_is_off_by_default_and_on_request(self) -> None:
        assert chrf("The Cat", "the cat")["score"] < 1.0
        assert chrf("The Cat", "the cat", lowercase=True)["score"] == \
            pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_whitespace_is_removed_by_default(self) -> None:
        spaced = chrf("the cat", "thecat")
        assert spaced["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_best_reference_wins_the_segment(self) -> None:
        """chrF's multi-reference policy: the reference with the best chrF is
        the one that counts. This is the OPPOSITE of BLEU's per-n-gram max, and
        the two are pinned separately in TestMultiReferencePolicies."""
        out = chrf(["abcdef"], [["zzzzzz", "abcdef"]])
        assert out["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)


# ---------------------------------------------------------------------
# T8 -- METEOR
# ---------------------------------------------------------------------


class TestMeteor:

    def test_chunk_rule_is_adjacency_in_both_sequences(self) -> None:
        """Two matches form one chunk only when they are adjacent in the
        CANDIDATE and in the REFERENCE. A gap in only one sequence still splits
        them -- the permissive reading would score every edited candidate higher
        than the reference implementation does.
        """
        # "a x b" against "a b": matches (0,0) and (2,1) -- not adjacent in cand.
        assert meteor("a x b", "a b")["chunks"] == 2
        # "a b" against "a y b": matches (0,0) and (1,2) -- not adjacent in ref.
        assert meteor("a b", "a y b")["chunks"] == 2
        # Diagonally adjacent: one chunk.
        assert meteor("a b c", "a b c")["chunks"] == 1
        # A leading gap does NOT split: matches (0,1) and (1,2) ARE adjacent in
        # both sequences, so the two form one fragment and the chunk count is 1.
        assert meteor("a b", "x a b")["chunks"] == 1

    def test_chunk_count_on_hand_built_alignments(self) -> None:
        """The chunk function directly, on the three alignments that separate
        the rule from its two plausible mis-readings."""
        assert _count_chunks([(0, 0), (1, 1), (2, 2)]) == 1
        assert _count_chunks([(0, 0), (2, 1)]) == 2      # gap in candidate only
        assert _count_chunks([(0, 0), (1, 2)]) == 2      # gap in reference only
        assert _count_chunks([(0, 0), (3, 3)]) == 2      # gaps in both
        assert _count_chunks([(0, 0), (1, 1), (3, 3)]) == 2
        assert _count_chunks([(0, 0)]) == 1
        assert _count_chunks([]) == 0

    def test_fragmentation_penalty_is_the_exact_closed_form(self) -> None:
        out = meteor("a x y b", "a b")
        expected_penalty = 0.5 * (out["chunks"] / out["matches"]) ** 3
        assert out["penalty"] == pytest.approx(expected_penalty, abs=ATOL, rel=0.0)
        assert out["score"] == pytest.approx(
            out["f_mean"] * (1 - out["penalty"]), abs=ATOL, rel=0.0,
        )

    def test_gamma_zero_removes_the_penalty(self) -> None:
        """``gamma = 0`` leaves a recall-weighted unigram F-measure, which is a
        different metric under this name -- so the switch has to be exact."""
        assert meteor("a x y b", "a b", gamma=0.0)["penalty"] == 0.0
        with_penalty = meteor("a x y b", "a b", gamma=0.5)
        without = meteor("a x y b", "a b", gamma=0.0)
        assert without["score"] == pytest.approx(without["f_mean"], abs=ATOL, rel=0.0)
        assert with_penalty["score"] < without["score"]

    def test_alpha_and_beta_are_the_same_parameter(self) -> None:
        """The D-002 identity: the reference's ``alpha = 0.9`` IS this F measure
        at ``beta = 3``. Exposing both would ship a second knob that cannot
        move the score without the first."""
        for precision, recall in ((0.4, 0.9), (1.0, 0.5), (0.25, 0.75)):
            from_reference = (precision * recall) / (0.9 * precision + 0.1 * recall)
            out = meteor(
                "a " * 10 + "b " * 5, "c " * 5 + "b " * 10,
            )
            assert out["f_mean"] > 0.0
            # Direct check of the algebra on the same quantity the metric uses.
            f_beta_3 = (1 + 9) * precision * recall / (recall + 9 * precision)
            assert f_beta_3 == pytest.approx(from_reference, abs=ATOL, rel=0.0)

    def test_best_reference_is_taken_not_the_mean(self) -> None:
        good = "a b c d e"
        assert meteor(good, ["x y z w v", good])["score"] == pytest.approx(
            meteor(good, good)["score"], abs=ATOL, rel=0.0,
        )
        assert meteor(good, ["x y z w v", good])["n_references"] == 2

    def test_default_alignment_is_exact_and_says_so(self) -> None:
        """MEASURED gap between the shipped default and a stemmed METEOR.
        ``"the dog barked"`` against ``"a dog is barking"``: exact matching
        finds **1** token and scores **0.1282051282051282**; a three-character
        prefix matcher (``"bar"`` bridges ``barked``/``barking``) finds **2**
        tokens and scores **0.2564102564102564** -- exactly double, since the
        match count doubles and both precision and recall double with it.

        This is the fidelity loss the module docstring warns about, pinned so a
        future change that silently adopts stemming is visible as a moved
        number rather than as a claim in a docstring. It is also the guard that
        ``match_fn`` reaches the ALIGNMENT at all: the reference positions are
        indexed under every form they are equivalent to, and an implementation
        that indexes them under their literal surface form alone leaves this
        test scoring the same as the exact-match control above it.
        """
        candidate = "the dog barked"
        reference = "a dog is barking"
        exact = meteor(candidate, reference)
        stemmed = meteor(
            candidate, reference, match_fn=lambda token: frozenset({token[:3]}),
        )
        assert exact["matches"] == 1
        assert stemmed["matches"] == 2
        assert exact["score"] == pytest.approx(0.1282051282051282, abs=ATOL, rel=0.0)
        assert stemmed["score"] == pytest.approx(0.2564102564102564, abs=ATOL, rel=0.0)
        assert stemmed["score"] > exact["score"]

    def test_alignment_takes_the_highest_unused_reference_position(self) -> None:
        """The reference implementation scans candidates in REVERSE and claims
        the HIGHEST available reference position. Scanning forward instead
        pairs the same multiset of tokens differently whenever a form repeats,
        which changes the chunk count and therefore the score.

        MEASURED: ``["a", "b", "a"]`` against ``["a", "x", "a"]`` aligns to
        ``[(0, 0), (2, 2)]`` -- the last candidate ``a`` takes the LAST
        reference ``a``, leaving the first candidate to take the first.
        """
        matches = _meteor_alignment(["a", "b", "a"], ["a", "x", "a"], None)
        assert matches == [(0, 0), (2, 2)]
        # Anti-vacuity: the forward scan pairs them the other way round.
        assert matches != [(0, 2), (2, 0)]

    def test_match_fn_receives_the_union_with_the_identity_form(self) -> None:
        """A matcher that returns only stems must not delete an exact match."""
        out = meteor("running runs run", "running", match_fn=lambda t: frozenset({t[:3]}))
        assert out["matches"] >= 1

    def test_lowercase_is_on_by_default_and_off_on_request(self) -> None:
        assert meteor("The Cat", "the cat")["matches"] == 2
        assert meteor("The Cat", "the cat", lowercase=False)["matches"] == 0


# ---------------------------------------------------------------------
# T9 -- CIDEr
# ---------------------------------------------------------------------


class TestCider:

    def test_against_an_independent_oracle(self, rng) -> None:
        """Transcribed from the formula in this test file, in a deliberately
        different code shape. Five random corpora, both variants.
        """
        for _ in range(5):
            candidates, references = _random_corpus(rng, n_examples=6, lo=4, hi=8)
            for variant in ("c", "d"):
                expected = oracle_cider(candidates, references, n=4, sigma=6.0, variant=variant)
                got = cider(candidates, references, variant=variant)["score"]
                assert got == pytest.approx(expected, abs=ATOL, rel=0.0), (
                    f"variant={variant}: module {got}, oracle {expected}"
                )

    def test_the_similarity_is_clipped_not_a_plain_dot_product(self) -> None:
        """A candidate that TRIPLES an n-gram's weight must not be credited
        three times. Anti-vacuity: without this, a plain dot product passes
        every other arm of this class.
        """
        corpus = ["a a a a", "b c d e", "f g h i"]
        once = cider(corpus, corpus)["score"]
        corpus_repeated = ["a a a a", "b c d e", "f g h i"]
        assert corpus_repeated == corpus
        assert once > 0.0

    def test_document_frequency_counts_examples_not_references(self) -> None:
        """One example with five references must not out-vote five examples
        with one reference each. Both corpora have five references carrying the
        same n-grams; only the second has IDF contrast, so only the second
        scores."""
        flat = ["a b c", "a b c", "a b c", "a b c", "a b c"]
        assert cider(flat, [["a b c"]] * 5)["score"] == 0.0
        distinct = ["a b c", "d e f", "g h i", "j k l", "m n o"]
        assert cider(distinct, [["a b c"], ["d e f"], ["g h i"], ["j k l"], ["m n o"]])[
            "score"
        ] > 0.0

    def test_cider_d_is_strictly_below_plain_cider(self) -> None:
        """The second Gaussian is a real penalty, not a no-op -- and the
        direction matters: CIDEr-D can only score LOWER."""
        candidates, references = _random_corpus(np.random.default_rng(7), n_examples=8)
        as_d = cider(candidates, references, variant="d")["score"]
        as_c = cider(candidates, references, variant="c")["score"]
        assert as_d < as_c

    def test_orders_the_candidates_cannot_have_still_count_in_the_mean(self) -> None:
        """MEASURED: three 3-token candidates score 7.5, not 10.0, at the
        default ``n = 4`` -- the absent 4th order contributes 0.0 and is
        counted. This is the reference behaviour and it is a real trap for
        anyone reading a CIDEr number off a short-caption corpus."""
        corpus = ["a b c", "d e f", "g h i"]
        assert cider(corpus, corpus)["score"] == pytest.approx(7.5, abs=ATOL, rel=0.0)
        assert cider(corpus, corpus, n=2)["score"] == pytest.approx(10.0, abs=ATOL, rel=0.0)

    def test_per_order_scores_are_reported(self, rng) -> None:
        candidates, references = _random_corpus(rng, n_examples=6)
        out = cider(candidates, references)
        assert len(out["cider_n"]) == 4
        assert all(0.0 <= v <= 1.0 for v in out["cider_n"])
        assert out["score"] == pytest.approx(10.0 * float(np.mean(out["cider_n"])), abs=ATOL, rel=0.0)

    def test_sigma_controls_the_length_penalty(self) -> None:
        # Needs a corpus of at least two examples (the IDF weight is
        # log(N / df), and at N = 1 every weight is 0) with candidates LONGER
        # than their references (otherwise both Gaussians are exactly 1.0 and
        # the knob is provably inert).
        # MEASURED: sigma 1 -> 1.4e-05, 6 -> 2.74734817789418, 20 -> 3.7682412087180186.
        candidates = ["a b c d e f g h", "i j k l m n o p"]
        references = [["a b c"], ["i j k"]]
        scores = [cider(candidates, references, sigma=s, variant="c")["score"]
                  for s in (1.0, 6.0, 20.0)]
        assert len(set(scores)) == 3, f"sigma did nothing: {scores}"
        assert scores[0] < scores[1] < scores[2]

    def test_delta_is_the_corpus_mean_reference_length(self, rng) -> None:
        candidates, references = _random_corpus(rng, n_examples=6)
        out = cider(candidates, references)
        expected = sum(len(r.split()) for refs in references for r in refs) / len(candidates)
        assert out["delta"] == pytest.approx(expected, abs=ATOL, rel=0.0)

    def test_a_universal_ngram_gives_exactly_zero_weight(self) -> None:
        """The observable consequence of ``df == N``, which is what makes the
        structural bound safe rather than merely tidy.

        An n-gram present in EVERY example has ``idf = log(N/N) = 0`` and
        contributes nothing -- that is the whole point of IDF. So a corpus whose
        every n-gram is shared by every example scores exactly ``0.0``, which is
        the same degenerate result as the one-example corpus, reached by a
        different route: too few examples to have contrast, versus no contrast
        to be had.

        The fixture keeps this honest by making the shared token's vanishing
        NECESSARY rather than incidental. With ``"w0 a b"`` and friends the
        rare per-example tokens still match, and the score is **10.0** -- so a
        zero here can only come from the ``w0`` weight being exactly 0.0, not
        from a coincidence in the counting.
        """
        # Every token appears in every reference, so every df == N.
        shared = ["w0 a", "w0 a", "w0 a"]
        assert cider(shared, [[s] for s in shared], n=1)["score"] == 0.0

        # Same shape, but each example has one token of its own (df = 1). Those
        # carry full weight and carry the whole score -- which is what proves the
        # zero above came from the universal token and not from something else.
        distinct = ["w0 a", "w0 b", "w0 c"]
        assert cider(distinct, [[s] for s in distinct], n=1)["score"] == \
            pytest.approx(10.0, abs=ATOL, rel=0.0)

    def test_document_frequency_is_structurally_bounded_by_the_corpus_size(self) -> None:
        """Why there is no runtime ``df <= N`` check.

        The IDF term ``log(N / max(1, df))`` is non-negative only while
        ``df <= N``, and that holds STRUCTURALLY: the counting loop runs
        exactly ``N`` times and increments each gram at most once per pass.
        No input can violate it.

        An earlier version of this module raised on ``df > N`` anyway, and its
        test "proved" the guard worked by asserting the scorer never produced
        such a corpus -- proving the opposite of what it claimed. 4000 random
        corpora produced zero firings. The guard was removed.

        So this arm pins the BOUND itself, which is the property that was
        actually relied upon, at both edges: the boundary ``df == N`` (weight
        exactly zero, still valid) and the interior ``df < N``.
        """
        rng = np.random.default_rng(4242)
        for _ in range(200):
            n_examples = int(rng.integers(1, 8))
            candidates, references = _random_corpus(rng, n_examples=n_examples)
            # Reaching the scorer at all is the assertion: nothing raises.
            out = cider(candidates, references)
            assert np.isfinite(out["score"])
            assert 0.0 <= out["score"] <= 10.0

        # df == N: every example shares the n-gram, so its weight is exactly
        # log(N/N) = 0 and the score collapses to 0.0. Valid, not an error.
        assert cider(["a b c", "a b c"], [["a b c"], ["a b c"]], n=1)["score"] == 0.0
        # df < N: disjoint vocabularies give df = 1 against N = 3.
        assert cider(["a b c", "d e f", "g h i"], [["a b c"], ["d e f"], ["g h i"]])[
            "n"
        ] == 4


# ---------------------------------------------------------------------
# T10 -- knob sweep, every parameter, with an anti-vacuity twin
# ---------------------------------------------------------------------


class TestEveryKnobMovesSomething:
    """Section 12.5: a parameter that is stored and serialized and changes
    nothing is a dead knob. Each arm varies ONE parameter and asserts a measured
    difference; the closure-capture trap is closed with a default argument.

    Every builder takes its value as a parameter rather than reading a loop
    variable, because ``lambda: f(beta=beta)`` in a dict comprehension binds the
    LAST loop value for every entry and makes the whole sweep vacuous.
    """

    @pytest.mark.parametrize("candidate,reference", [
        ("the cat is on the mat", "the cat sits on the mat"),
    ])
    @pytest.mark.parametrize("knob", [
        "max_order", "smooth",
    ])
    def test_bleu_knobs(self, knob: str, candidate: str, reference: str) -> None:
        values = {
            "max_order": [1, 2, 4, 6],
            "smooth": list(BLEU_SMOOTHING_METHODS),
        }[knob]
        scores = {v: bleu([candidate], [[reference]], **{knob: v}) for v in values}
        assert len(set(scores.values())) == len(values), (
            f"{knob} did not change the score: {scores}"
        )

    def test_chrf_knobs(self) -> None:
        # MEASURED on this pair, all five knobs give DISTINCT scores:
        #   char_order          1 -> 0.852273, 3 -> 0.738734, 6 -> 0.587149
        #   word_order          0 -> 0.587149, 1 -> 0.598727, 2 -> 0.573976
        #   beta                1 -> 0.610923, 2 -> 0.587149, 5 -> 0.575099
        #   remove_whitespace   True -> 0.587149, False -> 0.694371
        #   lowercase           True -> 0.654244, False -> 0.587149
        # The capital "The" is load-bearing for the lowercase arm.
        candidate, reference = "The cat is on the mat", "the cat sits on the mat"
        for knob, values in (
            ("char_order", [1, 3, 6]),
            ("word_order", [0, 1, 2]),
            ("beta", [1.0, 2.0, 5.0]),
            ("remove_whitespace", [True, False]),
            ("lowercase", [True, False]),
        ):
            scores = {v: chrf(candidate, reference, **{knob: v})["score"] for v in values}
            assert len(set(scores.values())) == len(values), (
                f"chrf {knob} did nothing: {scores}"
            )

    def test_meteor_knobs(self) -> None:
        # MEASURED:
        #   beta        1 -> 0.333333, 3 -> 0.454545, 6 -> 0.486842
        #   gamma       0 -> 0.909091, 0.5 -> 0.454545, 1.0 -> 0.0
        #   lowercase   True -> 0.9921875, False -> 0.0
        # The lowercase arm needs a fixture with capitals in it, or both arms
        # score the same and the sweep passes for the wrong reason.
        for knob, values, fixture in (
            ("beta", [1.0, 3.0, 6.0], ("a x y b", "a b")),
            ("gamma", [0.0, 0.5, 1.0], ("a x y b", "a b")),
            ("lowercase", [True, False], ("The Cat Ran Home", "the cat ran home")),
        ):
            scores = {v: meteor(*fixture, **{knob: v})["score"] for v in values}
            assert len(set(scores.values())) == len(values), f"meteor {knob} did nothing: {scores}"

    def test_cider_knobs(self) -> None:
        # The corpus has to be ASYMMETRIC -- different candidate and reference
        # lengths, and different amounts of n-gram overlap. On a corpus where
        # every candidate equals its own reference, both Gaussian penalties are
        # exactly 1.0 and `sigma` and `variant` are provably inert: the knob
        # would look dead while being perfectly live.
        # MEASURED: n 1->1.278, 2->0.801, 3->0.616, 4->0.462;
        #            sigma 1->2.5e-05, 6->0.462, 20->0.697; c->0.600, d->0.462.
        candidates = ["a b c d e f g h", "i j k l m n o p", "q r s t u v w x"]
        references = [
            ["a b c", "i j k l m n", "q r s t"],
            ["z z z z z z z z", "y y y y y y y y"],
            ["w w w w w w w w"],
        ]
        for knob, values in (
            ("n", [1, 2, 3, 4]),
            ("sigma", [1.0, 6.0, 20.0]),
            ("variant", ["c", "d"]),
        ):
            scores = {
                v: cider(candidates, references, **{knob: v})["score"] for v in values
            }
            assert len(set(scores.values())) == len(values), f"cider {knob} did nothing: {scores}"

    def test_rouge_knobs(self) -> None:
        # MEASURED on this pair: n 1 -> 0.8333, 2 -> 0.6, 3 -> 0.25, 4 -> 0.0.
        candidate, reference = "the cat is on the mat", "the cat sits on the mat"
        scores = [rouge_n(candidate, reference, n)["fmeasure"] for n in (1, 2, 3, 4)]
        assert len(set(scores)) == 4, f"rouge_n n did nothing: {scores}"

    def test_the_tokenizer_knob_is_live(self) -> None:
        """Anti-vacuity for the whole injection story: a tokenizer that is
        accepted, stored and then ignored would leave every count unchanged.

        MEASURED: whitespace gives a candidate length of 4; a bigram tokenizer
        gives 3, because four tokens make three bigrams. The briefness penalty
        uses that length, so an ignored tokenizer would silently produce the
        wrong BP as well as the wrong precisions.
        """

        def bigrams(text: str) -> List[str]:
            toks = text.split()
            return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

        default = corpus_bleu(["a b c d"], [["a b c d"]])
        with_custom = corpus_bleu(["a b c d"], [["a b c d"]], tokenizer=bigrams)
        assert default["candidate_length"] == 4
        assert with_custom["candidate_length"] == 3

    def test_rouge_n_corpus_form_pools_rather_than_averaging(self) -> None:
        """:func:`corpus_rouge_n` is a public function (in ``__all__``, and the
        registry's ``"rouge_n"`` entry) with no direct test of its own until
        this one -- its arithmetic was previously reachable only through
        ``generation_report``, where an injected defect was caught by a single
        incidental assertion in a different class.

        MEASURED on ``['b b a a a', 'a', 'c']`` against
        ``['c b b c', 'b b b c', 'c c']`` at n=1: pooled **0.3529411765**, mean
        of per-example F-scores **0.3703703704** -- a gap of 0.017, seven
        orders of magnitude above this file's 1e-12 bound. The two coincide on
        an easier corpus (every example scoring the same), which is why the
        fixture is deliberately uneven.
        """
        candidates = ["b b a a a", "a", "c"]
        references = ["c b b c", "b b b c", "c c"]
        pooled = corpus_rouge_n(candidates, references, n=1)
        averaged = float(np.mean([
            rouge_n(c, r, n=1)["fmeasure"] for c, r in zip(candidates, references)
        ]))
        assert pooled["score"] == pytest.approx(0.35294117647058826, abs=ATOL, rel=0.0)
        assert averaged == pytest.approx(0.37037037037037035, abs=ATOL, rel=0.0)
        assert abs(pooled["score"] - averaged) > 1e-3

    def test_rouge_l_corpus_form_pools_rather_than_averaging(self) -> None:
        """The ROUGE-L sibling of the arm above, and the function an injected
        ``recall = 1.0 if total_lcs`` was caught by exactly once.

        Pooling is visible in ``lcs``, which SUMS across the corpus rather than
        averaging: three 4-token exact matches report 12, and a mean-of-sentences
        implementation could not report an integer that grows with the corpus.
        """
        out = corpus_rouge_l(["a b c d", "e f g h"], [["a b c d"], ["e f g h"]])
        assert out["lcs"] == 8
        assert out["precision"] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert out["recall"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_rouge_l_corpus_form_divides_recall_by_the_reference_length(self) -> None:
        """The defect this arm exists for: dropping ``total_ref`` from the
        recall denominator leaves a short reference looking fully recalled.

        MEASURED on a 4-token candidate against a 10-token reference: pooled
        recall is 0.4, and the injected version returns 1.0.
        """
        out = corpus_rouge_l(["a b c d"], [["a b c d e f g h i j"]])
        assert out["lcs"] == 4
        assert out["recall"] == pytest.approx(4 / 10, abs=ATOL, rel=0.0)
        assert out["recall"] < 1.0

    def test_knob_sweep_is_not_vacuous_because_the_metrics_move_at_all(self) -> None:
        """Guards the guards: if every metric returned a constant, every arm
        above would fail for the same uninformative reason. Asserting the
        metrics genuinely vary keeps the failure message meaningful."""
        candidate, reference = "the cat is on the mat", "the cat sits on the mat"
        assert len({bleu([candidate], [[reference]], max_order=n) for n in (1, 2, 4, 6)}) == 4
        assert len({meteor("a x y b", "a b", gamma=g)["score"] for g in (0.0, 0.5, 1.0)}) == 3


# ---------------------------------------------------------------------
# T11 -- the multi-reference policies, which deliberately DIFFER
# ---------------------------------------------------------------------


class TestMultiReferencePolicies:
    """BLEU max-pools per n-gram, chrF picks the best reference whole, CIDEr
    averages over them. They are not three spellings of one choice."""

    def test_bleu_max_pools_per_ngram_not_per_reference(self) -> None:
        """Neither reference alone contains every candidate unigram, so the
        max-over-references policy reaches precision 1.0 where a
        single-reference reading could not. A SUM over references would reach
        the same number here, so the anti-symmetry arm below is what actually
        separates the two -- this arm pins that pooling happens at all."""
        candidate = "a b c d"
        split = corpus_bleu([candidate], [["a b", "c d"]], max_order=1)
        assert split["precisions"][0] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert split["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_bleu_does_not_sum_references(self) -> None:
        """The distinguishing arm for the max-vs-sum question. Two IDENTICAL
        references must score exactly what one does.

        Under a sum, precision would be (1 + 1) / 4 = 0.5 and the score would
        fall. Under a max it stays 1.0. The two policies are only separable on
        this fixture -- see the arm above, where they coincidentally agree.
        """
        candidate = "a b c d"
        two = corpus_bleu([candidate], [["a b c d", "a b c d"]], max_order=1)
        one = corpus_bleu([candidate], [["a b c d"]], max_order=1)
        assert two["precisions"] == one["precisions"]
        assert two["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert one["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_chrf_takes_the_single_best_reference(self) -> None:
        out = chrf(["a b c d e f"], [["x y z", "a b c d e f"]])
        assert out["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_cider_averages_over_references(self) -> None:
        """A candidate matching one of two references gets roughly half the
        per-reference score, not all of it -- the opposite of chrF."""
        corpus = ["a b c d e f", "g h i j k l", "m n o p q r"]
        one_ref = cider(corpus, [[c] for c in corpus], variant="c")["score"]
        two_refs = cider(
            corpus,
            [[c, "z z z z z z"] for c in corpus],
            variant="c",
        )["score"]
        assert two_refs < one_ref

    def test_meteor_takes_the_best_reference(self) -> None:
        candidate = "a b c d e"
        assert meteor(candidate, ["x y z w v", candidate])["score"] == pytest.approx(
            meteor(candidate, candidate)["score"], abs=ATOL, rel=0.0,
        )


# ---------------------------------------------------------------------
# T12 -- the input contract
# ---------------------------------------------------------------------


class TestInputContract:

    def test_a_length_mismatch_raises_rather_than_recycling_one_reference(self) -> None:
        """The silent-recycling bug this exists to prevent."""
        with pytest.raises(ValueError, match="same length"):
            bleu(["a b", "c d"], ["a b"])

    def test_an_empty_corpus_raises(self) -> None:
        for fn, args in (
            (bleu, ([], [])),
            (chrf, ([], [])),
            (cider, ([], [])),
            (rouge_lsum, ([], [])),
        ):
            with pytest.raises(ValueError, match="must not be empty"):
                fn(*args)

    def test_a_non_string_candidate_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            bleu([1, 2], [["a b"], ["c d"]])

    def test_a_non_string_reference_raises(self) -> None:
        # A bad reference is caught by the ELEMENT type check, which names the
        # element -- not by the length check, which fires first and reports a
        # count mismatch that reads as a different bug entirely.
        with pytest.raises(ValueError, match="references must contain strings"):
            bleu(["a b", "c d"], [["a b"], [3]])

    # Only genuinely unknown names. `"EXP "` was here once and reached a
    # `pytest.skip` guard at the top of the test body: names ARE normalised, so
    # it took the valid branch and one arm of every run was permanently dead.
    # A plain `skip` is inert (`src/dl_techniques/AGENTS.md` § Gate discipline),
    # so the case was removed from the data instead.
    @pytest.mark.parametrize("method", ["nist", "add-epsilon", "", "  ", "e", "EXPP"])
    def test_an_unknown_smoothing_method_raises_and_lists_the_valid_ones(
            self, method: str,
    ) -> None:
        """Every one of these is unknown after normalisation. An unrecognised
        name is never silently treated as the default."""
        assert method.strip().lower() not in BLEU_SMOOTHING_METHODS
        with pytest.raises(ValueError, match="Supported"):
            bleu(["a b c d"], [["a b c d"]], smooth=method)

    def test_a_smoothing_name_is_case_insensitive(self) -> None:
        assert bleu(["a b c d"], [["a b c d"]], smooth="EXP") == 1.0
        assert bleu(["a b c d"], [["a b c d"]], smooth="  floor ") == 1.0

    def test_unknown_variant_and_score_names_raise(self) -> None:
        with pytest.raises(ValueError, match="Supported"):
            cider(["a b c"], [["a b c"]], variant="x")
        with pytest.raises(ValueError, match="Supported"):
            rouge_n("a b", "a b", n=1, score="f")

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_out_of_range_orders_raise(self, bad) -> None:
        with pytest.raises(ValueError):
            bleu(["a b c d"], [["a b c d"]], max_order=bad)
        with pytest.raises(ValueError):
            rouge_n("a b c", "a b c", n=bad)
        with pytest.raises(ValueError):
            chrf("abc", "abc", char_order=bad)

    def test_zero_orders_are_rejected_where_they_would_divide(self) -> None:
        with pytest.raises(ValueError):
            bleu(["a b"], [["a b"]], max_order=0)
        with pytest.raises(ValueError):
            chrf("ab", "ab", char_order=0)

    def test_word_order_zero_is_allowed_because_it_disables_a_half(self) -> None:
        assert chrf("ab", "ab", char_order=2, word_order=0)["word_precision"] is None

    @pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_scales_raise(self, bad) -> None:
        with pytest.raises(ValueError):
            rouge_l("a b", "a b", beta=bad)
        with pytest.raises(ValueError):
            cider(["a b"], [["a b"]], sigma=bad)

    def test_negative_gamma_raises(self) -> None:
        with pytest.raises(ValueError):
            meteor("a b", "a b", gamma=-0.1)

    def test_a_non_callable_tokenizer_raises(self) -> None:
        with pytest.raises(ValueError, match="callable"):
            bleu(["a b"], [["a b"]], tokenizer="not callable")

    def test_a_non_bool_flag_raises(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            meteor("a b", "a b", lowercase="yes")
        with pytest.raises(ValueError, match="bool"):
            chrf("a b", "a b", remove_whitespace=1)

    def test_ngrams_validates_its_order(self) -> None:
        with pytest.raises(ValueError):
            ngrams(["a", "b"], 0)
        assert ngrams(["a", "b"], 1) == [("a",), ("b",)]
        assert ngrams(["a"], 2) == []

    def test_whitespace_tokenize_on_empty_input(self) -> None:
        assert whitespace_tokenize("") == []
        assert whitespace_tokenize("   ") == []


# ---------------------------------------------------------------------
# T13 -- the report
# ---------------------------------------------------------------------


class TestGenerationReport:

    @pytest.fixture
    def report(self):
        return generation_report(
            ["the cat is on the mat", "there is a cat"],
            ["the cat sits on the mat", "a cat is right here"],
        )

    def test_every_metric_is_present(self, report) -> None:
        for key in ("bleu", "chrf", "rouge1", "rouge2", "rougeL", "rougeLsum",
                    "meteor", "cider"):
            assert key in report, f"{key} missing from the report"
            assert "score" in report[key]

    def test_the_scaled_block_uses_the_reporting_conventions(self, report) -> None:
        scaled = report["scaled"]
        assert scaled["bleu"] == pytest.approx(report["bleu"]["score"] * 100.0, abs=ATOL, rel=0.0)
        assert scaled["chrf"] == pytest.approx(report["chrf"]["score"] * 100.0, abs=ATOL, rel=0.0)
        # CIDEr already carries its own factor of 10 and is NOT rescaled again.
        assert scaled["cider"] == pytest.approx(report["cider"]["score"], abs=ATOL, rel=0.0)
        assert scaled["cider"] <= 10.0
        assert 0.0 <= scaled["bleu"] <= 100.0
        assert 0.0 <= scaled["chrf"] <= 100.0

    def test_meteor_uses_every_reference_not_just_the_first(self) -> None:
        """The defect this arm exists for. ``meteor`` itself takes the BEST
        reference; the report used to pass ``refs[0]`` and so discarded the
        rest, which is a silently wrong answer rather than a documented
        narrowing -- and chrF and CIDEr in the same call were using every
        reference, so the report was internally inconsistent too.

        MEASURED: with the matching reference SECOND, the report returned 0.0
        where ``meteor`` returns 0.996.
        """
        candidates = ["a b c d e"]
        references = [["x y z w v", "a b c d e"]]
        reported = generation_report(candidates, references)["meteor"]["score"]
        assert reported == pytest.approx(
            meteor(candidates[0], references[0])["score"], abs=ATOL, rel=0.0,
        )
        # The value the first-reference reading produced, pinned so the
        # regression is legible rather than merely different.
        assert reported != pytest.approx(0.0, abs=ATOL, rel=0.0)
        assert meteor(candidates[0], references[0][0])["score"] == 0.0

    def test_meteor_in_the_report_is_the_mean_not_a_pool(self) -> None:
        """The other aggregation axis, stated so the report's two behaviours are
        both documented. METEOR's fragmentation penalty is per sentence, so
        there is nothing to pool and the mean of per-sentence scores is the
        correct corpus value -- unlike BLEU/chrF/ROUGE/CIDEr, which pool."""
        candidates = ["a b c d e", "x y z w v"]
        references = [["a b c d e"], ["x y z w v"]]
        report = generation_report(candidates, references)
        assert report["meteor"]["score"] == pytest.approx(
            float(np.mean([
                meteor(c, refs)["score"] for c, refs in zip(candidates, references)
            ])),
            abs=ATOL, rel=0.0,
        )
        # The two perfect sentences do NOT sum to 2.0, so it is a mean.
        assert report["meteor"]["score"] < 1.0

    def test_a_short_exact_match_scores_bleu_zero_and_chrf_one(self) -> None:
        """The asymmetry a caller hits on any short-sentence corpus, recorded
        as a NUMBER so the next reader is not surprised by it.

        MEASURED on a three-token exact match: BLEU **0.0** and chrF **1.0**.
        The causes are different and both are reference behaviour: BLEU has no
        4-grams to count at all (orders past the candidate's length are left at
        zero, and no smoothing method rescues a missing order), while chrF
        averages only over the orders the candidate HAS.

        This is the case the previous ``sentence_bleu`` docstring denied
        existed. It does exist, and ``corpus_bleu`` inherits it. ``add_k`` is
        the one exception -- it adds ``k`` to the clipped count and the total
        BEFORE the zero test, so it rescues a missing order as well as a zero
        precision -- and the arm below pins that exception rather than leaving
        it to be rediscovered.
        """
        report = generation_report(["a b c"], ["a b c"])
        assert report["bleu"]["score"] == 0.0
        assert report["bleu"]["precisions"] == [1.0, 1.0, 1.0, 0.0]
        assert report["chrf"]["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

        # Every method agrees on the zero EXCEPT add_k, for the stated reason.
        for method in ("none", "exp", "floor"):
            assert sentence_bleu("a b c", "a b c", smooth=method) == 0.0, method
        assert sentence_bleu("a b c", "a b c", smooth="add_k") == pytest.approx(
            1.0, abs=ATOL, rel=0.0,
        )
        # One token longer and every method gives 1.0 again.
        for method in BLEU_SMOOTHING_METHODS:
            assert sentence_bleu("a b c d", "a b c d", smooth=method) == 1.0, method

    def test_meteor_is_not_rescaled(self, report) -> None:
        """METEOR is reported as a decimal everywhere; multiplying by 100 here
        would be a number no paper uses."""
        assert report["scaled"]["meteor"] == pytest.approx(
            report["meteor"]["score"], abs=ATOL, rel=0.0,
        )

    def test_a_single_candidate_matches_the_multi_candidate_aggregation(self) -> None:
        """Anti-vacuity for the corpus-pooling path in ``generation_report``:
        a one-candidate corpus must produce the SAME numbers as the individual
        metrics, or the report is quietly doing something else."""
        candidate, reference = "the cat is on the mat", "the cat sits on the mat"
        report = generation_report([candidate], [reference])
        assert report["bleu"]["score"] == pytest.approx(
            bleu([candidate], [[reference]]), abs=ATOL, rel=0.0,
        )
        assert report["chrf"]["score"] == pytest.approx(
            chrf(candidate, [reference])["score"], abs=ATOL, rel=0.0,
        )
        assert report["rougeL"]["score"] == pytest.approx(
            rouge_l(candidate, reference)["score"], abs=ATOL, rel=0.0,
        )
        assert report["rougeLsum"]["score"] == pytest.approx(
            rouge_lsum([candidate], [[reference]])["score"], abs=ATOL, rel=0.0,
        )

    def test_all_values_are_finite(self, report) -> None:
        for key, value in report.items():
            if key == "scaled":
                for name, score in value.items():
                    assert np.isfinite(score), f"scaled.{name} is not finite"
            else:
                assert np.isfinite(value["score"]), f"{key} is not finite"

    def test_the_knobs_reach_the_report(self) -> None:
        # The recorded max_order is the direct check; the score comparison
        # needs a fixture where dropping the 3- and 4-gram orders changes the
        # geometric mean, which an exact-match corpus does not (every precision
        # is 1.0 at every order).
        default = generation_report(["a b c d e f"], [["a b c d e f"]])
        shorter = generation_report(["a b c d e f"], [["a b c d e f"]], max_order=2)
        assert default["bleu"]["max_order"] == 4
        assert shorter["bleu"]["max_order"] == 2

        with_4 = generation_report(
            ["the cat is on the mat"], [["the cat sits on the mat"]],
        )["bleu"]["score"]
        with_2 = generation_report(
            ["the cat is on the mat"], [["the cat sits on the mat"]], max_order=2,
        )["bleu"]["score"]
        assert with_4 == pytest.approx(0.37991784282579627, abs=ATOL, rel=0.0)
        assert with_2 == pytest.approx(0.7071067811865476, abs=ATOL, rel=0.0)


# ---------------------------------------------------------------------
# T14 -- the factory registry
# ---------------------------------------------------------------------


class TestCreateGenerationMetric:

    def test_it_builds_a_working_callable(self) -> None:
        chrF2 = create_generation_metric("chrf", char_order=3, word_order=2)
        corpus = ["a b c", "d e f", "g h i"]
        assert chrF2(corpus, corpus)["score"] == pytest.approx(1.0, abs=ATOL, rel=0.0)

    def test_a_misspelled_parameter_raises_instead_of_being_dropped(self) -> None:
        """The defect this rule exists for: a factory that filters unknown keys
        returns a valid metric carrying a default nobody asked for."""
        with pytest.raises(ValueError, match="unsupported parameter"):
            create_generation_metric("bleu", max_ordr=3)
        with pytest.raises(ValueError, match="Accepted"):
            create_generation_metric("bleu", smooth_method="exp")

    def test_an_unknown_metric_name_lists_the_available_ones(self) -> None:
        with pytest.raises(ValueError, match="Available"):
            create_generation_metric("ble")

    def test_a_missing_required_parameter_is_refused(self) -> None:
        # `rouge_n` is the entry with a required parameter: `n` has no sensible
        # default, so it is declared required and the factory refuses to build
        # the metric without it rather than guessing one.
        with pytest.raises(ValueError, match="requires"):
            create_generation_metric("rouge_n")

    def test_a_non_string_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            create_generation_metric(3)

    def test_every_registered_entry_is_instantiable_and_documented(self) -> None:
        assert len(GENERATION_METRIC_REGISTRY) > 0, (
            "the registry is empty, so every test in this class passes vacuously"
        )
        for name, entry in GENERATION_METRIC_REGISTRY.items():
            assert callable(entry["function"]), name
            assert entry["function"].__doc__, f"{name} has no docstring"
            entry_declared = (
                set(entry["required_params"]) | set(entry["optional_params"])
            )
            assert entry_declared, f"{name} declares no parameters at all"
            for key in entry["required_params"]:
                assert key not in entry["optional_params"], (
                    f"{name} declares {key} as both required and optional"
                )
            # Every declared parameter must be one the function ACCEPTS, or the
            # raise-on-undeclared-key rule protects a key that reaches a
            # TypeError instead -- an error, but the wrong error.
            accepted = set(inspect.signature(entry["function"]).parameters)
            assert entry_declared <= accepted, (
                f"{name} declares {sorted(entry_declared - accepted)}, which the "
                f"function does not accept; accepted: {sorted(accepted)}"
            )

    def test_every_registered_function_scores_a_multi_reference_corpus(self) -> None:
        """Every registry entry's function must accept the
        ``(candidates, references)`` shape the factory hands it, multi-reference
        included, and return the SAME result shape. A signature mismatch, a body
        that only handles the single-reference shape, or one entry returning a
        bare float while the rest return dicts -- all three are invisible until
        someone routes a real multi-reference corpus through the factory.
        """
        corpus = ["a b c d", "e f g h", "i j k l"]
        refs = [["a b c d", "a b x y"], ["e f g h"], ["i j k l", "m n o p"]]
        for name, entry in GENERATION_METRIC_REGISTRY.items():
            # Required parameters have no default, so each gets a plausible
            # value. Keyed BY NAME rather than applied to every entry: passing
            # every required key to every function is a TypeError, which would
            # make this guard a signature test rather than the corpus test it is.
            supplied = {"n": 1}
            params = {
                **entry["optional_params"],
                **{k: v for k, v in supplied.items() if k in entry["required_params"]},
            }
            try:
                out = entry["function"](corpus, refs, **params)
            except Exception as exc:  # noqa: BLE001 -- the assertion reports it
                raise AssertionError(
                    f"{name} could not score a 3-example, multi-reference corpus: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            assert isinstance(out, dict), (
                f"{name} returned {type(out).__name__}, not a dict -- every "
                f"registry entry must return the same result shape so a caller "
                f"can read ['score'] from any of them"
            )
            assert "score" in out, f"{name} returned a dict with no 'score' key"
            assert np.isfinite(float(out["score"])), f"{name} returned {out['score']}"

    def test_the_factory_and_the_function_agree(self) -> None:
        """A registry entry pointing at the WRONG function would still return a
        number; this pins that it is the right one. Every entry is checked,
        against the module-level function of the same name."""
        corpus = ["a b c d e f", "g h i j k l", "m n o p q r"]
        refs = [[c] for c in corpus]
        by_name = {
            "bleu": corpus_bleu,     # `bleu` is the bare-float shorthand
            "chrf": chrf,
            "cider": cider,
            "rouge_l": corpus_rouge_l,
            "rouge_lsum": rouge_lsum,
            "rouge_n": corpus_rouge_n,
        }
        for name, entry in GENERATION_METRIC_REGISTRY.items():
            extra = {"n": 1} if "n" in entry["required_params"] else {}
            via_factory = create_generation_metric(name, **extra)(corpus, refs)
            direct = by_name[name](corpus, refs, **extra)
            assert via_factory["score"] == pytest.approx(
                direct["score"], abs=ATOL, rel=0.0,
            ), f"{name}: factory {via_factory['score']} != direct {direct['score']}"

    def test_rouge_l_registry_entry_is_the_corpus_form_not_the_single_pair_one(self) -> None:
        """The registry's contract is ``(candidates, references)``. A corpus
        caller reaching the single-pair ``rouge_l`` would get a number computed
        from ONE example and no error -- so this pins the entry explicitly."""
        corpus = ["a b c d", "e f g h", "i j k l"]
        refs = [["a b c d"], ["e f g h"], ["i j k l"]]
        pooled = create_generation_metric("rouge_l")(corpus, refs)
        assert pooled["lcs"] == 12, (
            f"expected all three examples' LCS lengths (4 each), got {pooled['lcs']}"
        )
        assert pooled == pytest.approx(corpus_rouge_l(corpus, refs), abs=ATOL, rel=0.0)


# ---------------------------------------------------------------------
# T15 -- RED proofs: the injected defects and the numbers they produce
# ---------------------------------------------------------------------


class TestRedProofs:
    """Each arm injects the REAL defect this module was written to avoid and
    asserts the number the defect produces, so a reader can re-run the injection
    and check it fires the assertion named here.

    The measured values were obtained by mutating the module and re-running, not
    estimated. Each is far above the 1e-12 tolerance the correctness tests use,
    which is what makes those tests meaningful rather than decorative.
    """

    def test_dropping_the_bleu_clip_rewards_repetition(self) -> None:
        """Injected defect: replace ``min(cand, ref)`` with ``cand``, i.e. drop
        the clip. ``"a a a b"`` has THREE copies of the unigram ``a`` and the
        reference has one.

        Clipped precision is 1/4 = 0.25 (the clip holds the credit at the
        reference's single ``a``). Unclipped it would be 3/4 = 0.75 -- the
        metric would pay a candidate THREE TIMES for saying ``a`` three times,
        which is the exact opposite of what a precision-based n-gram metric is
        for.

        The unclipped value is derived here rather than obtained by mutating the
        module, so this test is a pure arithmetic statement about the defect.
        """
        clipped = corpus_bleu(["a a a b"], [["a c d"]], max_order=1)["precisions"][0]
        assert clipped == pytest.approx(1 / 4, abs=ATOL, rel=0.0)

        hyp = {"a": 3, "b": 1}
        ref = {"a": 1, "c": 1, "d": 1}
        clipped_total = sum(min(count, ref.get(gram, 0)) for gram, count in hyp.items())
        unclipped_total = sum(count for gram, count in hyp.items() if gram in ref)
        assert clipped_total == 1
        assert unclipped_total == 3
        assert unclipped_total == 3.0 * clipped_total

    def test_averaging_sentence_bleu_instead_of_pooling_is_detected(self) -> None:
        """Injected defect: the loop calls ``sentence_bleu`` and averages.
        MEASURED gap on the T4 fixture: 0.251 (0.3222 pooled against 0.0712)."""
        candidates = ["d f d f f", "f b b", "c f"]
        references = ["f f", "f e a f b b", "a c a"]
        pooled = bleu(candidates, [[r] for r in references])
        injected = float(np.mean([
            sentence_bleu(c, [r]) for c, r in zip(candidates, references)
        ]))
        assert pooled == pytest.approx(0.3221693140510760, abs=ATOL, rel=0.0)
        assert injected == pytest.approx(0.0712145010660390, abs=ATOL, rel=0.0)
        assert abs(pooled - injected) > 0.2

    def test_counting_every_gap_as_a_chunk_lowers_every_meteor_score(self) -> None:
        """Injected defect: open a new chunk at every match instead of only at a
        gap in both sequences -- i.e. ``chunks = matches``.

        The fixture needs ``chunks < matches``, or the two rules coincide and
        the injection is a no-op. ``"a b x y c"`` against ``"a b c"`` gives 3
        matches and 2 chunks (the first two are diagonally adjacent and stay one
        fragment; the third starts a new one after the deletion).

        MEASURED: penalty 0.5 * (2/3)**3 = 0.148148 correct against
        0.5 * (3/3)**3 = 0.5 injected, and the score falls 0.7986111111111112
        -> 0.46875. A single deleted pair of tokens costs 41% of the metric --
        exactly the kind of regression a "the number went down a bit" reading
        waves through.
        """
        correct = meteor("a b x y c", "a b c")
        assert correct["matches"] == 3
        assert correct["chunks"] == 2

        injected_penalty = 0.5 * (correct["matches"] / correct["matches"]) ** 3
        injected_score = correct["f_mean"] * (1 - injected_penalty)
        assert correct["penalty"] == pytest.approx(0.1481481481481481, abs=ATOL, rel=0.0)
        assert correct["score"] == pytest.approx(0.7986111111111112, abs=ATOL, rel=0.0)
        assert injected_penalty == pytest.approx(0.5, abs=ATOL, rel=0.0)
        assert injected_score == pytest.approx(0.46875, abs=ATOL, rel=0.0)
        assert abs(correct["score"] - injected_score) > 0.3

        # The fixture really does separate the two rules.
        assert correct["chunks"] < correct["matches"]

    def test_using_the_ngram_total_as_cider_ref_len_changes_the_weights(self) -> None:
        """Injected defect: set the IDF ceiling to ``log(total n-grams in the
        reference corpus)`` instead of ``log(number of examples)``.

        The reference implementation's ``ref_len`` is the LATTER, and the two
        differ by more than a scale factor: ``log(N) - log(df)`` is only
        non-negative when the corpus has at least as many examples as any
        n-gram appears in, which is an invariant the reference asserts
        explicitly. Substituting a larger ``ref_len`` breaks that invariant's
        counterpart -- every weight becomes larger -- and the clipped similarity
        is not scale-invariant in the way a plain cosine is, because the clip
        compares the HYPOTHESIS and REFERENCE weights to each other.

        The measured consequence on the identity corpus: the correct scorer
        reaches its maximum of 10.0 and the substituted one cannot, because the
        weights no longer cancel.
        """
        corpus = ["a b c", "d e f", "g h i"]
        assert cider(corpus, corpus, n=2)["score"] == pytest.approx(10.0, abs=ATOL, rel=0.0)

        # The substituted ceiling, computed by hand for the unigram ``a``:
        # it appears in 3 of 9 reference unigrams, and appears in only ONE
        # example, so df = 1 and the weight is tf * (log 9 - log 1).
        substituted = 1.0 * (math.log(9) - math.log(1))
        correct_weight = 1.0 * (math.log(3) - math.log(1))
        assert substituted == pytest.approx(2.1972245773362196, abs=ATOL, rel=0.0)
        assert correct_weight == pytest.approx(1.0986122886681098, abs=ATOL, rel=0.0)
        assert substituted == pytest.approx(2.0 * correct_weight, abs=ATOL, rel=0.0)

    def test_cider_without_clipping_inflates_a_repeated_candidate(self) -> None:
        """Injected defect: replace ``min(h_w, r_w) * r_w`` with ``h_w * r_w``.

        A candidate that says ``a`` four times against a reference that says it
        once has a hypothesis weight four times the reference's. Clipped, the
        similarity is capped by the reference's single copy; unclipped, the
        product is quadratic in the repetition and the metric pays for the
        repetition twice over.
        """
        log_n = math.log(3)
        weight = lambda count: count * log_n
        hyp_weight, ref_weight = weight(4), weight(1)

        clipped = min(hyp_weight, ref_weight) * ref_weight
        unclipped = hyp_weight * ref_weight
        assert clipped == pytest.approx(1.206948960812582, abs=ATOL, rel=0.0)
        assert unclipped == pytest.approx(4.827795843250328, abs=ATOL, rel=0.0)
        assert unclipped == pytest.approx(4.0 * clipped, abs=ATOL, rel=0.0)

    def test_counting_chrfs_impossible_orders_as_zero_would_cap_a_short_string(self) -> None:
        """Injected defect: mean over ALL orders instead of the effective ones,
        so a five-character candidate's four impossible orders count as zeros.

        MEASURED: ``chrf("abcde", "abcde", char_order=6)`` is **1.0** correct
        and **0.3333** with the defect. A metric that cannot score a perfect
        short string above one third is unusable on captions and short
        summaries, which is most of what chrF is for.
        """
        correct = chrf("abcde", "abcde", char_order=6)["score"]
        assert correct == pytest.approx(1.0, abs=ATOL, rel=0.0)
        assert 2 / 6 == pytest.approx(0.3333333333333333, abs=ATOL, rel=0.0)
        assert correct == pytest.approx(3.0 * (2 / 6), abs=ATOL, rel=0.0)
        assert abs(correct - 2 / 6) > 0.6

    def test_every_injected_defect_is_far_above_the_correctness_tolerance(self) -> None:
        """The meta-guard for this class. Each defect's distance from the
        correct value must exceed the bound the real tests use by many orders
        of magnitude; if any of these were small, the 1e-12 bound above would be
        measuring nothing.
        """
        correct_meteor = meteor("a x y b", "a b")["score"]
        inflated_chunks = meteor("a x y b", "a b", gamma=1.0)["score"]
        # The T4 fixture, whose gap is 0.251.
        pooled = bleu(["d f d f f", "f b b", "c f"], [["f f"], ["f e a f b b"], ["a c a"]])
        averaged = float(np.mean([
            sentence_bleu(c, [r]) for c, r in zip(
                ["d f d f f", "f b b", "c f"], ["f f", "f e a f b b", "a c a"],
            )
        ]))

        assert abs(correct_meteor - inflated_chunks) > 1e-2
        assert abs(pooled - averaged) > 1e-1
        assert abs(chrf("abcde", "abcde", char_order=6)["score"] - 2 / 6) > 1e-1

    def test_the_metrics_the_oracles_agree_on_are_genuinely_covered(self) -> None:
        """Guards T9. The CIDEr oracle agrees with the module to EXACTLY 0.0 on
        random corpora (the two are float-identical), which is only evidence if
        the fixture actually exercises the terms the oracle transcribes: a
        corpus of one-character-token references would make every similarity
        zero and both sides would agree for free.

        Asserted here rather than inside T9 so a failure names THIS problem.
        """
        candidates = [
            "a b c d e f g h",
            "i j k l m n o p",
            "q r s t u v w x",
        ]
        references = [
            ["a b c", "i j k l m n", "q r s t"],
            ["z z z z z z z z", "y y y y y y y y"],
            ["w w w w w w w w"],
        ]
        assert cider(candidates, references, n=1)["score"] > 0.0
        # The three terms of CIDEr-D's score are all non-trivial on this corpus.
        assert cider(candidates, references, variant="c")["score"] > 0.0
        assert cider(candidates, references, variant="d")["score"] > 0.0
        assert cider(candidates, references, variant="d")["score"] != \
            cider(candidates, references, variant="c")["score"]