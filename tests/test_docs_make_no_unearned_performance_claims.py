"""Model docs may credit the PAPER with measured results; they may not credit THIS port.

Nothing under ``src/dl_techniques/models/`` has been trained or benchmarked in this
repository. A README or module docstring that says *the architecture* reaches
state-of-the-art is accurate and belongs there. One that says *this implementation*
does is a claim the repo cannot back, and it is the shape this guard rejects.

The guard is pure text: it opens ``.md`` and ``.py`` files and runs regexes. It
imports no model, builds nothing, and costs milliseconds.

Two independent rules:

1. **UNEARNED** -- a single sentence that both refers to this codebase in the first
   person (``this implementation``, ``our port``, ...) *and* asserts a performance
   superlative. The ATTRIBUTED shape is pinned structurally, not by an allowlist of
   sentences: a sentence is exempt when it also carries an attribution marker
   (``et al.``, ``the paper``, ``the authors report``, ``as reported by``, ...) or an
   explicit disclaimer (``has not been trained``, ``no performance claim``, ...).
   Rewrite the claim onto the architecture; do not delete the fact.

2. **PUFFERY** -- an unverifiable maturity adjective (``production-ready``,
   ``battle-tested``, ``fully-featured``, ...). No deployment or benchmark evidence
   exists for any model package, so these words assert nothing checkable.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pytest

MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "dl_techniques" / "models"

# --- rule 1: self-reference + performance superlative in ONE sentence ----------

SELF_REF_RE = re.compile(
    r"\b(this|our)\s+(implementation|implementations|port|codebase|package|repo|"
    r"repository|library|module)\b",
    re.IGNORECASE,
)

PERFORMANCE_RE = re.compile(
    r"(state[-\s]of[-\s]the[-\s]art|\bSOTA\b|outperform\w*|surpass\w*|"
    r"best[-\s]in[-\s]class|superior\b|world[-\s]class)",
    re.IGNORECASE,
)

# DECISION plan-2026-08-23T091307-9a110062/D-520
# Do NOT "simplify" this by exempting any sentence that cites a paper. That was tried
# and the RED proof killed it: the pft_sr sentence
#   "This implementation is based on the CVPR 2025 paper by Long et al. and achieves
#    state-of-the-art performance on single image super-resolution benchmarks."
# cites the paper AND credits this port in the same breath -- a citation-based
# exemption passes the exact defect the guard exists to catch. Only a REPORTING
# construction ("who report", "as reported by") or an explicit disclaimer exempts.
# Do NOT replace either regex with an allowlist of approved sentences. See D-520.
#
# Shapes that make a performance sentence legitimate. Deliberately NOT satisfied by
# merely naming a paper: "based on the paper by Long et al. and achieves SOTA" names
# the source but still hangs the verb off *this implementation*, which is exactly the
# defect. Only an explicit REPORTING construction (someone else reports the result) or
# an explicit disclaimer exempts a sentence.
ATTRIBUTION_RE = re.compile(
    r"(\b(who|which|they|authors?|paper|work|team)\s+report\w*|"
    r"\bas\s+report\w*\s+(by|in)\b|\breported\s+(by|in)\b|"
    r"\bquoted\s+from\b|\baccording\s+to\b|"
    r"(has|have)\s+not\s+been\s+"
    r"(independently\s+)?(trained|benchmarked|reproduced|measured|evaluated)|"
    r"no\s+(performance|accuracy|quality|latency)\s+claim)",
    re.IGNORECASE,
)

# --- rule 2: unverifiable maturity adjectives ---------------------------------

PUFFERY_RE = re.compile(
    r"(production[-\s]ready|battle[-\s]tested|fully[-\s]featured|"
    r"best[-\s]in[-\s]class|enterprise[-\s]grade|industrial[-\s]strength)",
    re.IGNORECASE,
)

# Held by a concurrent editing session at the time this guard was written; the
# phrase is real and must go. Each entry is asserted to still be dirty, so the
# list self-liquidates: fix the file and this test tells you to delete the entry.
# It is NOT a general-purpose allowlist -- do not add to it to silence a failure.
PENDING_PUFFERY = (
    "swin_transformer/model.py",
    "vit/model.py",
)

# A period only ends a sentence when the next token starts one. Without the
# uppercase lookahead, "by Long et al. and achieves state-of-the-art performance"
# splits at "al." and the performance clause loses its "This implementation" subject
# -- the guard would then pass the exact sentence it exists to catch.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[*`_])|\n{2,}|\n(?=[-*|#>])")


def _iter_doc_files():
    for path in sorted(MODELS_DIR.rglob("*")):
        if path.suffix in (".md", ".py") and path.is_file():
            yield path


def _sentences(text: str):
    """Yield (line_number, sentence) pairs, cheaply and approximately."""
    offset_line = {}
    line = 1
    for i, ch in enumerate(text):
        offset_line[i] = line
        if ch == "\n":
            line += 1
    pos = 0
    for chunk in _SENTENCE_SPLIT_RE.split(text):
        if chunk is None:
            continue
        idx = text.find(chunk, pos)
        if idx == -1:
            idx = pos
        pos = idx + len(chunk)
        stripped = chunk.strip()
        if stripped:
            yield offset_line.get(idx, 1), stripped


def _unearned_hits(path: Path) -> List[Tuple[int, str]]:
    hits = []
    for lineno, sentence in _sentences(path.read_text(encoding="utf-8")):
        if not SELF_REF_RE.search(sentence):
            continue
        if not PERFORMANCE_RE.search(sentence):
            continue
        if ATTRIBUTION_RE.search(sentence):
            continue
        hits.append((lineno, " ".join(sentence.split())[:200]))
    return hits


def test_no_doc_credits_this_port_with_measured_performance():
    """No sentence may claim SOTA/superiority *for this implementation*."""
    offenders = []
    for path in _iter_doc_files():
        for lineno, sentence in _unearned_hits(path):
            offenders.append(f"{path}:{lineno}: {sentence}")
    assert not offenders, (
        "Unearned performance claim -- these sentences credit THIS port with results "
        "it never measured. Re-attribute the claim to the paper/architecture (and say "
        "the port is unbenchmarked); do not delete the fact:\n  "
        + "\n  ".join(offenders)
    )


def test_no_doc_uses_unverifiable_maturity_puffery():
    """'production-ready' and friends assert maturity nothing in the repo backs."""
    offenders = []
    for path in _iter_doc_files():
        rel = str(path.relative_to(MODELS_DIR))
        if rel in PENDING_PUFFERY:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = PUFFERY_RE.search(line)
            if match:
                offenders.append(f"{path}:{lineno}: {match.group(0)!r} in: {line.strip()[:160]}")
    assert not offenders, (
        "Unverifiable puffery -- no model package has deployment or benchmark evidence:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("rel", PENDING_PUFFERY)
def test_pending_puffery_entries_are_still_needed(rel):
    """When a PENDING entry is cleaned up, this fails so the entry gets deleted."""
    path = MODELS_DIR / rel
    assert PUFFERY_RE.search(path.read_text(encoding="utf-8")), (
        f"{rel} no longer contains maturity puffery -- remove it from PENDING_PUFFERY "
        "in this file so the main guard covers it."
    )
