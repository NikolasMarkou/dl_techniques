"""The `>>>` examples in `calibration_metrics.py` must state what the code returns.

Why this module exists (F-061). `calibration_metrics.py` is the one file in the analyzer
package carrying eleven `>>>` examples across seven functions, and NOTHING executed them:
`pyproject.toml` sets `testpaths = ["tests"]` with no `--doctest-modules`, so the examples
were prose. Four of the eleven were WRONG, and three of those had drifted far enough to
mislead:

| function | claimed | actual |
|---|---|---|
| `compute_ece(n_bins=5)` | `0.16` | `0.24` |
| `compute_adaptive_ece(n_bins=2)` | `0.0` | `0.30` |
| `compute_brier_score` | `0.135` | `0.075` |
| `compute_prediction_entropy_stats` | `0.4578` | `0.4609` |

The AECE one is the worst: it advertised `0.0` — *perfectly calibrated* — for a probe that
is maximally OVERconfident (100% accurate at 0.9 confidence). A reader checking whether a
metric behaved sanely would have concluded the opposite of the truth.

`test_analyzer_docs.py` already verifies docstring PROSE by substring, which cannot catch a
wrong NUMBER. This module executes the values instead. Adding `--doctest-modules` for the
whole package was rejected: it would start executing every example in every analyzer module
at once, and the examples here are illustrative rather than doctest-shaped (several print
rather than return).
"""

import numpy as np
import pytest

from dl_techniques.analyzer import calibration_metrics as cm


class TestTheDocumentedExamplesAreTheRealValues:
    """Each entry is (callable, arguments, the value the docstring now claims)."""

    @pytest.mark.parametrize(
        "label,actual,claimed",
        [
            ("compute_ece",
             lambda: cm.compute_ece(
                 np.array([0, 1, 1, 0, 1]),
                 np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8],
                           [0.8, 0.2], [0.4, 0.6]]),
                 n_bins=5),
             0.24),
            ("compute_ece_binary",
             lambda: float(cm.compute_ece_binary(
                 np.array([0, 0, 1, 1]),
                 np.array([0.5, 0.5, 0.5, 0.5]),
                 n_bins=10)),
             0.0),
            ("compute_adaptive_ece",
             lambda: cm.compute_adaptive_ece(
                 np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
                 np.array([[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5),
                 n_bins=2),
             0.30),
            ("compute_brier_score",
             lambda: cm.compute_brier_score(
                 np.array([[1, 0], [0, 1], [0, 1], [1, 0]]),
                 np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.9, 0.1]])),
             0.075),
        ],
    )
    def test_the_claimed_value_is_the_computed_one(self, label, actual, claimed):
        got = float(actual())
        assert got == pytest.approx(claimed, abs=1e-9), (
            f"{label} returned {got!r}, but its docstring example claims {claimed!r}. "
            f"The docstring is the wrong one: these are executed here precisely because "
            f"no doctest runner covers this module."
        )

    def test_the_entropy_example_matches_its_four_digit_claim(self):
        """`compute_prediction_entropy_stats`'s example prints, so it is rounded to 4dp."""
        stats = cm.compute_prediction_entropy_stats(
            np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]]))
        printed = f"{float(stats['mean_entropy']):.4f}"
        assert printed == "0.4609", (
            f"the example prints {printed!r}, its docstring says '0.4609'"
        )

    def test_the_adaptive_ece_probe_is_overconfident_not_perfect(self):
        """The AECE claim of `0.0` was the most misleading of the four (F-061).

        Pinned as its own assertion because the number alone does not explain WHY it was
        wrong: a reader seeing `0.0` would conclude "perfectly calibrated", when the probe
        is 100% accurate at 0.9 confidence — the definition of overconfident.
        """
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5)
        accuracy = float((np.argmax(y_prob, axis=1) == y_true).mean())
        confidence = float(np.max(y_prob, axis=1).mean())
        # The probe is 60% accurate while asserting 0.9 on every prediction. Both halves
        # of the calibration error are present, which is why the answer is 0.30 rather
        # than 0.0 — but it is decisively OVERCONFIDENT, which is the point being pinned.
        assert accuracy == pytest.approx(0.6)
        assert confidence == pytest.approx(0.9)
        assert confidence > accuracy, (
            "the probe is overconfident, so an ECE/AECE of 0.0 cannot be its answer"
        )
        # Assert the DECOMPOSITION, not just the total, so a future change to the binning
        # that breaks this probe fails HERE rather than looking like a legitimate new
        # answer. `compute_adaptive_ece` sorts by confidence and splits into two
        # equal-mass bins of 5: the low-confidence bin holds mean conf 0.1 / mean acc 0.4
        # (gap 0.3) and the high-confidence bin mean conf 0.9 / mean acc 0.6 (gap 0.3).
        # Weighted 0.5 each, that is 0.30.
        outcomes = (np.argmax(y_prob, axis=1) == y_true).astype(float)
        scores = np.max(y_prob, axis=1)
        order = np.argsort(scores)
        gaps = []
        for chunk in (order[:5], order[5:]):
            gaps.append(abs(outcomes[chunk].mean() - scores[chunk].mean()))
        assert gaps == [pytest.approx(0.3), pytest.approx(0.3)]
        assert float(np.mean(gaps)) == pytest.approx(0.30)
        assert float(cm.compute_adaptive_ece(y_true, y_prob, n_bins=2)) == pytest.approx(0.30)


class TestTheDocstringsCarryNoOtherStaleNumber:
    """Guard the whole file, so a future edit cannot reintroduce a drifted example.

    These read the DOCSTRINGS via `ast`, not the raw source text. A substring search over
    source would have to guess the indentation, and the first version of this class did:
    it searched for eight leading spaces and matched nothing, because `inspect.getsource`
    dedents. Anchoring on the parsed docstring is also what makes the assertion mean what
    it says — "this function's documented output", not "this number appears somewhere".
    """

    #: Expected output line of each function's `Example:` block.
    _CLAIMED_OUTPUTS = {
        "compute_ece": "0.24",
        "compute_ece_binary": "0.0",
        "compute_adaptive_ece": "0.3",
        "compute_brier_score": "0.075",
        "compute_prediction_entropy_stats": "0.4609",
    }

    @staticmethod
    def _documented_outputs():
        """Map function name -> the last non-`>>>` line of its ``Example:`` block.

        A doctest example's expected output is the text following the final ``>>>``
        line. Returns only functions that actually HAVE an ``Example:`` block, so a
        function without one is not silently counted as passing.
        """
        import inspect
        import textwrap

        found = {}
        for name in dir(cm):
            obj = getattr(cm, name)
            if not inspect.isfunction(obj) or obj.__module__ != cm.__name__:
                continue
            doc = textwrap.dedent(obj.__doc__ or "")
            if "Example:" not in doc:
                continue
            example = doc.split("Example:", 1)[1]
            output_lines = [ln.strip() for ln in example.splitlines()
                            if ln.strip() and not ln.strip().startswith(">>>")
                            and not ln.strip().startswith("...")]
            if output_lines:
                found[name] = output_lines[-1]
        return found

    def test_the_parser_really_found_the_examples(self):
        """Anti-vacuity: a broken parser must not make the assertions below vacuous."""
        found = self._documented_outputs()
        assert set(self._CLAIMED_OUTPUTS) <= set(found), (
            f"the docstring parser found {sorted(found)}, which does not cover "
            f"{sorted(set(self._CLAIMED_OUTPUTS) - set(found))}; the parser, not the "
            f"docstrings, is what failed"
        )

    @pytest.mark.parametrize("name,claimed", sorted(_CLAIMED_OUTPUTS.items()))
    def test_each_docstring_shows_its_own_computed_value(self, name, claimed):
        found = self._documented_outputs()
        assert found[name] == claimed, (
            f"{name}'s docstring documents {found[name]!r}, but the value it computes is "
            f"{claimed!r}. Update the example and this table in the same commit."
        )

    @pytest.mark.parametrize("stale", ["0.16", "0.135", "0.4578"])
    def test_no_docstring_reverted_to_a_pre_audit_value(self, stale):
        found = self._documented_outputs()
        offenders = {n: out for n, out in found.items() if out == stale}
        assert not offenders, (
            f"{sorted(offenders)} document(s) reverted to {stale!r}, one of the four "
            f"values F-061 corrected"
        )
