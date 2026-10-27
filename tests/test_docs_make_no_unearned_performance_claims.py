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
    # `vision/vit/model.py` left this set on 2026-08-25, and
    # `vision/swin_transformer/model.py` left it on 2026-09-04 -- both had
    # their puffery removed by whoever rewrote the docstring, caught here by
    # the self-liquidation arm this table exists to run.
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

# Since 2026-10-27 `ResNet._download_weights` no longer exists: `from_variant`
# refuses `pretrained=True` through `pretrained_path_or_raise`. The rule is
# unchanged, because it keys on the README's wording and never on that symbol.
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
    """Pinned, so a new offender arrives as a diff rather than a surprise.

    Counts READMEs, not occurrences: an occurrence count would churn on every
    example added. If this fails because a package gained a transfer-learning
    section, re-derive the set here rather than widening the window above --
    and check BOTH halves for the newcomer before pinning it (see the comment
    inside the set).
    """
    readmes = [p for p in _iter_doc_files() if p.name == "README.md"]
    assert len(readmes) >= 20, len(readmes)
    with_pretrained_true = [
        p for p in readmes if PRETRAINED_TRUE_RE.search(p.read_text(encoding="utf-8"))
    ]
    assert {p.parent.name for p in with_pretrained_true} == {
        # Membership is EARNED, not assumed: a package belongs here only when
        # BOTH halves were checked -- its README names `pretrained=True` solely
        # to say the call RAISES (the disclaimed form the window above accepts),
        # AND the package's own model module genuinely raises
        # `NotImplementedError` on that path. Verified for every name below
        # except `depth_anything`, whose two halves differ and are stated, and
        # checked, at its entry.
        # A README that advertises a working `pretrained=True` it does not have
        # is caught by `test_no_readme_advertises_a_pretrained_path_that_raises`,
        # not by this pin; this one only tracks WHO is in the conversation.
        "ascii_bert", "ascii_clifford_bert", "ascii_convnext_bert",
        "bert", "bias_free_denoisers",
        # bit_diffusion: README "Usage" names `pretrained=True` only to say
        # `create_ditxa` and `create_shared_token_decoder` raise (added at
        # plan-2026-10-14T062616-81938e81 step 10); both really raise
        # `NotImplementedError`, verified by EXECUTING them
        # (tests/test_models/test_bit_diffusion/
        # test_a_ditxa_path_loads_every_weight.py and
        # test_the_token_decoder.py) and by the behavioural arm of
        # test_package_api_contract.py. Both halves checked.
        "bit_diffusion",
        # cliffordnet: README section 2 names `pretrained=True` only to say it
        # raises `NotImplementedError` naming the variant; both
        # `CliffordNet.from_variant` and `HierarchicalCliffordNet.from_variant`
        # really do, through `pretrained_path_or_raise`, verified by EXECUTING
        # both with `pretrained=True` (tests/test_models/test_cliffordnet/
        # test_model.py and test_hierarchical.py) and by the runtime contract
        # (test_the_vision_classifiers_share_one_runtime_contract.py). Both
        # halves checked.
        "cliffordnet",
        "colbert",
        # depth_anything: the one package whose `pretrained=True` WORKS (plan
        # -2026-10-14T062616-81938e81 step 1.7): it loads the official DINOv2
        # encoder file the user downloads, strictly (tests/test_models/
        # test_depth_anything/test_the_encoder_is_the_references_dinov2.py
        # compares the loaded encoder against a numpy port of that file).
        # Its README banner (inside BANNER_LINES) says no trained weights ship
        # and a missing file is a hard error, and the latter really raises
        # `FileNotFoundError`, verified by EXECUTING it in the same test file
        # with the weights directory pointed at an empty path. The banner also
        # cites what the start was measured to buy (D-140: held-out loss 0.57
        # to 0.59 against a constant 1.0 and the average layout 1.045). Both
        # halves checked (step 2.5, D-157).
        "depth_anything",
        "dino", "distilbert", "dit",
        # doc_res: README:215 says `pretrained=True` raises; model.py:707
        # really does `raise NotImplementedError`. Both halves checked.
        "doc_res",
        # doc_scanner: README section 7.4 names `pretrained=True` only to say
        # it raises, and section 7.2 refers back to it; model.py really does
        # `raise NotImplementedError` on ALL THREE classes' `from_variant` --
        # verified by EXECUTING each of the three factories with
        # `pretrained=True` (tests/test_models/test_doc_scanner/
        # test_composite.py::TestPretrainedRaisesOnAllThree) and by the
        # behavioural arm of test_package_api_contract.py, not by grepping for
        # the word. Both halves checked.
        "doc_scanner",
        # fastvlm: README:228 names `pretrained=True` only to say it raises
        # (added at plan-2026-10-14T062616-81938e81 step 5, which never ran
        # this file); `FastVLM.from_variant` really does raise
        # `NotImplementedError` via `pretrained_path_or_raise`, verified by
        # EXECUTING `FastVLM.from_variant('nano', ..., pretrained=True)` and
        # `create_fastvlm('nano', ..., pretrained=True)` at step 6.1 and
        # reading the raise. Both halves checked.
        "fastvlm",
        "fnet",
        # hnet: README:14 and :185 both name `pretrained=True` only to say it
        # raises; model.py:702 really does `raise NotImplementedError`, verified
        # by EXECUTING `HNet.from_variant('hnet_1stage_L', pretrained=True)` and
        # `create_hnet('hnet_1stage_L', pretrained=True)` and reading the raise,
        # not by grepping for the word. Both halves checked (line numbers and
        # execution re-derived 2026-10-13; `pretrained='<path>'` now restores
        # the backbone and is the README's working form).
        "hnet",
        "gpt2", "kan", "masked_autoencoder", "mobile_clip", "mobilenet",
        "modern_bert",
        # omnipoint: README:99 names `pretrained=True` only to say it raises;
        # model.py:534 really does `raise NotImplementedError`. Both halves
        # checked (re-derived 2026-10-04; pre-existing drift, not a new port).
        "omnipoint",
        # rbf_protonet: README:23/111/154/245 name `pretrained=True` only to
        # say it raises; model.py:279/661/884 really do
        # `raise NotImplementedError`. Both halves checked (re-derived
        # 2026-10-04; pre-existing drift, not a new port).
        "rbf_protonet",
        "resnet",
        # clifford_clip, sam1, sam2, sam3: each README names `pretrained=True`
        # only to say it raises (clifford_clip README:224, sam1 :115, sam2
        # :125, sam3 :124; the refusals added at plan-2026-10-14T062616-81938e81
        # step 11, D-045, which never ran this file). Each factory really
        # raises `NotImplementedError`, verified at step 12 by EXECUTING
        # `create_clifford_clip(<variant>, vocab_size=100, image_size=64,
        # context_length=16, pretrained=True)`, `create_sam('vit_b', ...)`,
        # `create_sam2('tiny', ...)` and `create_sam3('tiny', ...)` and reading
        # the raise. Both halves checked.
        "clifford_clip", "sam1", "sam2", "sam3",
        # siamfc: README section 2 names `pretrained=True` only to say it
        # raises; model.py `create_siamfc` really does
        # `raise NotImplementedError`, verified by EXECUTING
        # `create_siamfc(pretrained=True)` (tests/test_models/test_siamfc/
        # test_siamfc.py::TestSiamFC::test_pretrained_true_raises). Both
        # halves checked.
        "siamfc",
        "tree_transformer", "vit", "wave_field",
        # dasiamrpn: README section 2 names `pretrained=True` only to say it
        # raises; model.py `from_variant` really does
        # `raise NotImplementedError`, verified by EXECUTING
        # `create_dasiamrpn("otb", pretrained=True)`
        # (tests/test_models/test_dasiamrpn/test_dasiamrpn.py). Both halves
        # checked.
        "dasiamrpn",
        # deepsort: README section 2 names `pretrained=True` only to say it
        # raises; appearance.py `create_deepsort_embedding` really does
        # `raise NotImplementedError`, verified by EXECUTING
        # `create_deepsort_embedding(pretrained=True)`
        # (tests/test_models/test_deepsort/test_deepsort.py). Both halves
        # checked.
        "deepsort",
        # zamba2: README:9-10/170/175 name `pretrained=True` only to say it
        # raises; `pretrained_path_or_raise` (utils/weight_transfer.py, called
        # from model.py:932) really does `raise NotImplementedError`, verified
        # by EXECUTING `create_zamba2('zamba2_mini', pretrained=True)` and
        # `Zamba2Model.from_variant('zamba2_mini', pretrained=True)`. Both
        # halves checked (re-derived 2026-10-13).
        "zamba2",
        # alexnet: README:254 names `pretrained=True` only to say it raises
        # ("None ship with this port"); model.py:556 really does
        # `raise NotImplementedError`, verified by EXECUTING
        # `create_alexnet(pretrained=True)`. Both halves checked (re-derived
        # 2026-10-13; pre-existing drift, already missing from this set at
        # 240f07513).
        "alexnet",
        # gemma: README:467 names `pretrained=True` only to say it raises
        # beside the working `pretrained="<path>.keras"` form;
        # `pretrained_path_or_raise` (called from gemma3.py:639) really does
        # `raise NotImplementedError`, verified by EXECUTING
        # `create_gemma3("270m", pretrained=True)`. Both halves checked (new
        # in plan-2026-10-11T170958-dc4d2202 step 8b, re-derived 2026-10-13).
        "gemma",
        # qwen: README:183 names `pretrained=True` only to say it raises;
        # `pretrained_path_or_raise` (called from qwen3.py:665 and
        # qwen3_next.py:630) really does `raise NotImplementedError`, verified
        # by EXECUTING `create_qwen3("tiny", pretrained=True)` and
        # `create_qwen3_next("tiny", pretrained=True)` (Qwen3 has no "0.6b"
        # variant: `create_qwen3("0.6b", ...)` raises "Unknown variant"; its
        # variants are 30b-coder, medium, small, tiny). Both halves checked
        # (new in plan-2026-10-11T170958-dc4d2202 step 8b, re-derived
        # 2026-10-13 after the factory normalization).
        "qwen",
        # topographic_vae: README:101 names `pretrained=True` only to say it
        # raises; model.py:1285 really does `raise NotImplementedError`,
        # verified by EXECUTING `create_topographic_vae("mnist",
        # pretrained=True)` and `TopographicVAE.from_variant("mnist",
        # pretrained=True)`. Both halves checked (re-derived 2026-10-13;
        # pre-existing drift, already missing from this set at 240f07513).
        "topographic_vae",
        # topolm: README:32 names `pretrained=True` only to say it raises;
        # model.py:881 really does `raise NotImplementedError`, verified by
        # EXECUTING `create_topolm("small", pretrained=True)` and
        # `TopoLM.from_variant("small", pretrained=True)`. Both halves checked
        # (re-derived 2026-10-13; pre-existing drift, already missing from
        # this set at 240f07513).
        "topolm",
    }, sorted(p.parent.name for p in with_pretrained_true)


@pytest.mark.parametrize("rel", PENDING_PUFFERY)
def test_pending_puffery_entries_are_still_needed(rel):
    """When a PENDING entry is cleaned up, this fails so the entry gets deleted."""
    path = MODELS_DIR / rel
    assert PUFFERY_RE.search(path.read_text(encoding="utf-8")), (
        f"{rel} no longer contains maturity puffery -- remove it from PENDING_PUFFERY "
        "in this file so the main guard covers it."
    )
