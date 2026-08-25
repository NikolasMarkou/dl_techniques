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
    # `vision/vit/model.py` left this set on 2026-08-25. It was NOT cleaned up
    # here: its puffery was removed earlier by whoever rewrote its docstring,
    # and this table could not tell, because the restructure had left the key
    # spelled `vit/model.py` -- so `MODELS_DIR / rel` raised FileNotFoundError
    # and the self-liquidation arm errored instead of firing. Repointing the key
    # made the arm run and it immediately demanded the deletion it was built to
    # demand. `vision/swin_transformer/model.py` still matches and stays.
    "vision/swin_transformer/model.py",
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


# --- rule 3: a README may not advertise a weight-loading path that raises -----

# DECISION plan-2026-08-23T091307-9a110062/D-603
# `vision/resnet/README.md` section 9 quoted "70-80% accuracy from scratch" against
# "85-95% pretrained" and showed 14 runnable `pretrained=True` examples -- while
# `ResNet._download_weights` raises `NotImplementedError` by design (the
# fake-weight-URL family). It advertised the payoff of a path that cannot run.
# MEASURED 2026-08-23 across all 23 model READMEs: 18 carry a transfer-learning
# or fine-tuning section, 10 of those name `pretrained=True`, and NINE of the
# ten already carried a disclaimer in prose. `resnet` was the only offender --
# and its README did carry a disclaimer, 900 lines earlier in section 5, which
# is why a "does the file mention it anywhere" check would have called it clean.
# So the rule is PER-OCCURRENCE-CONTEXT: a `pretrained=True` in a fenced code
# block must be within DISCLAIMER_WINDOW lines of a disclaimer.
#
# `detr` is NOT an offender and must not be "fixed": its backbone is
# `keras.applications.ResNet50(weights="imagenet")`, a real download that really
# works, which is why the rule keys on the phrase, not on the package.
DISCLAIMER_RE = re.compile(
    # `[#>*\s]*` absorbs a line-wrap through a code comment or a blockquote:
    # `masked_autoencoder` wraps as "raises\n# `NotImplementedError`".
    r"(raises?\s+[#>*\s]*`?NotImplementedError|raises?\b[^\n]*\bsee\s+section|"
    r"\bis\s+a\s+hard\s+error|no\s+(public\s+)?pretrained\s+weights|"
    r"nothing\s+is\s+downloadable|no\s+weights\s+are\s+downloadable|"
    r"no\s+trained\s+weights\s+ship|does\s+not\s+work|no\s+download\s+exists)",
    re.IGNORECASE,
)

PRETRAINED_TRUE_RE = re.compile(r"pretrained\s*=\s*True")

#: How far from a `pretrained=True` a disclaimer may sit and still be read as
#: attached to it. 60 lines is roughly one screen of README plus a code block.
DISCLAIMER_WINDOW = 60

#: A top-of-file banner covers the whole document, and is how `bert`, `gpt2`,
#: `distilbert` and `wave_field` handle this -- each opens with a blockquote
#: saying the `pretrained=True` examples below will raise. That is honest and
#: must not be flagged, so a disclaimer in the first BANNER_LINES lines counts
#: for every occurrence in the file. `resnet`'s pre-fix disclaimer was neither:
#: it sat at line 647, inside section 5's variant table, ~900 lines above the
#: section-9 transfer-learning box that showed 14 `pretrained=True` examples and
#: quoted an accuracy benefit for using them.
BANNER_LINES = 40


def _pretrained_true_without_disclaimer(path: Path) -> List[Tuple[int, str]]:
    """Lines using ``pretrained=True`` with no disclaimer within the window."""
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = []
    for i, line in enumerate(lines):
        if not PRETRAINED_TRUE_RE.search(line):
            continue
        # Searched over JOINED text, not line by line, and that is load-bearing:
        # both `dino` and `masked_autoencoder` wrap their disclaimer exactly at
        # "`pretrained=True` raises / `NotImplementedError`", so a per-line search
        # reports two files that are already correct.
        if DISCLAIMER_RE.search(" ".join(lines[:BANNER_LINES])):
            continue
        window = lines[max(0, i - DISCLAIMER_WINDOW):i + DISCLAIMER_WINDOW]
        if DISCLAIMER_RE.search(" ".join(window)):
            continue
        hits.append((i + 1, line.strip()[:160]))
    return hits


def test_no_readme_advertises_a_pretrained_path_that_raises():
    """Every ``pretrained=True`` in the docs sits next to the fact that it raises.

    Not "somewhere in the file": ``resnet`` had a correct disclaimer in section 5
    and a section-9 transfer-learning box, 900 lines later, that showed 14
    ``pretrained=True`` examples and quoted an accuracy benefit for taking them.
    """
    offenders = []
    for path in _iter_doc_files():
        if path.suffix != ".md":
            continue
        for lineno, line in _pretrained_true_without_disclaimer(path):
            offenders.append(f"{path}:{lineno}: {line}")
    assert not offenders, (
        "A README shows `pretrained=True` with no nearby statement that it raises "
        "NotImplementedError. No model package in this repository ships or can "
        "download weights; say so beside the example, or use the working form "
        "`pretrained='/path/to/file.keras'`:\n  " + "\n  ".join(offenders)
    )


def test_the_pretrained_rule_is_not_vacuous():
    """Anti-vacuity: the regexes must actually match the shapes they name."""
    assert PRETRAINED_TRUE_RE.search("model = M.from_variant('x', pretrained=True)")
    assert PRETRAINED_TRUE_RE.search("    pretrained = True,")
    assert DISCLAIMER_RE.search("`pretrained=True` raises `NotImplementedError`")
    assert DISCLAIMER_RE.search("No public pretrained weights are distributed")
    assert not DISCLAIMER_RE.search("Load ResNet-50 with ImageNet pretrained weights")
    # The pre-fix resnet shape: an example with nothing nearby to qualify it.
    assert _pretrained_true_without_disclaimer.__doc__


def test_the_population_the_rule_was_derived_from_is_still_the_population():
    """MEASURED 2026-08-23 and pinned, so a new offender is a diff, not a surprise.

    Counts READMEs, not occurrences: an occurrence count would churn on every
    example added. If this fails because a package gained a transfer-learning
    section, re-derive the numbers here rather than widening the window above.

    `colbert` joined the population on 2026-08-25 (`plan-2026-08-25-c71fc3ad`): its
    README names `pretrained=True` only to say the call RAISES `NotImplementedError`,
    which is the disclaimed form the window above already accepts.
    """
    readmes = [p for p in _iter_doc_files() if p.name == "README.md"]
    assert len(readmes) >= 20, len(readmes)
    with_pretrained_true = [
        p for p in readmes if PRETRAINED_TRUE_RE.search(p.read_text(encoding="utf-8"))
    ]
    assert {p.parent.name for p in with_pretrained_true} == {
        "bert", "bias_free_denoisers", "colbert", "dino", "distilbert", "gpt2",
        "masked_autoencoder", "mobilenet", "modern_bert", "resnet",
        "tree_transformer", "vit", "wave_field",
    }, sorted(p.parent.name for p in with_pretrained_true)


@pytest.mark.parametrize("rel", PENDING_PUFFERY)
def test_pending_puffery_entries_are_still_needed(rel):
    """When a PENDING entry is cleaned up, this fails so the entry gets deleted."""
    path = MODELS_DIR / rel
    assert PUFFERY_RE.search(path.read_text(encoding="utf-8")), (
        f"{rel} no longer contains maturity puffery -- remove it from PENDING_PUFFERY "
        "in this file so the main guard covers it."
    )
