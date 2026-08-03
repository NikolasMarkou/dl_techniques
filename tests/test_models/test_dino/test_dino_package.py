"""Package-surface tests for ``src/dl_techniques/models/dino/``.

These pin the normalization done in plan-2026-08-01T105809-dc0c402e step 6 (F-06),
which is about the PACKAGE rather than any one model version:

1. ``src/dl_techniques/models/dino/__init__.py`` exports a non-empty ``__all__``
   whose every name resolves, and does not silently omit a public name.
2. ``create_dino_v1`` / ``create_dino_v2`` / ``create_dino_v3`` agree on their
   SHARED parameter names and kinds.
3. The ``patch_size``-vs-variant precedence rule (``None`` defers to the variant,
   an explicit value always wins) actually holds.
4. ``input_shape`` is refused by all three factories.
5. Every path and every import ``src/dl_techniques/models/dino/README.md`` names
   resolves on disk / imports cleanly.
6. No headline measurement endpoint from ``research/2026_dino_ssl_measurements.md``
   is restated anywhere in the DINO subtree (added by
   plan-2026-08-03T043010-cecf4357 step 8).

Each guard was RED-proven by injecting the corresponding dead component; the
injected component and the assertion that fired are recorded in the plan's
``verification.md``.
"""

import inspect
import re
from pathlib import Path

import pytest

# Repo root: tests/test_models/test_dino/<this file> -> up 3.
REPO_ROOT = Path(__file__).resolve().parents[3]
DINO_PKG = REPO_ROOT / "src" / "dl_techniques" / "models" / "dino"
README = DINO_PKG / "README.md"


# ---------------------------------------------------------------------
# 1. __all__ completeness, both directions
# ---------------------------------------------------------------------

def test_all_is_non_empty_and_every_name_resolves():
    import dl_techniques.models.dino as m

    assert m.__all__, "models/dino/__init__.py must export a non-empty __all__"
    missing = [n for n in m.__all__ if not hasattr(m, n)]
    assert not missing, (
        f"__all__ lists names the package does not bind: {missing}. "
        f"This is the exact class the guard exists for — a name can be dropped "
        f"from an import block and survive in __all__ indefinitely."
    )


def test_all_has_no_duplicates():
    import dl_techniques.models.dino as m

    assert len(m.__all__) == len(set(m.__all__)), (
        f"__all__ contains duplicates: {m.__all__}"
    )


# Names each submodule defines at module level that are deliberately NOT
# re-exported from the package, with the reason. If a new public name appears
# in a submodule it must either be exported or added here — the converse test
# below fails otherwise, so the omission cannot be silent.
DELIBERATELY_NOT_EXPORTED = {
    # `common.reject_input_shape` is the shared internals of the three factories,
    # not part of the DINO model API. Callers never invoke it directly.
    "reject_input_shape",
    # `common.sync_teacher_to_student` (D-034) is likewise internal: it is called
    # BY `create_dino_teacher_student_pair` and `DINOTrainingModel.__init__`, so
    # every supported way of building a DINO pair already gets it. Exporting it
    # would advertise a construction-time-only operation that DESTROYS a trained
    # teacher if a caller runs it later.
    "sync_teacher_to_student",
}


def test_every_public_submodule_name_is_exported_or_explicitly_excluded():
    """The CONVERSE of the __all__ check: nothing public is silently omitted."""
    import importlib

    import dl_techniques.models.dino as pkg

    exported = set(pkg.__all__)
    unaccounted = {}
    for sub in ("dino_v1", "dino_v2", "dino_v3", "common"):
        mod = importlib.import_module(f"dl_techniques.models.dino.{sub}")
        for name, value in vars(mod).items():
            if name.startswith("_"):
                continue
            # Only names DEFINED here — not `keras`, `np`, `layers`, or anything
            # else imported into the module namespace.
            if getattr(value, "__module__", mod.__name__) != mod.__name__:
                continue
            if inspect.ismodule(value):
                continue
            if name in exported or name in DELIBERATELY_NOT_EXPORTED:
                continue
            unaccounted.setdefault(sub, []).append(name)

    assert not unaccounted, (
        f"public submodule names neither exported nor explicitly excluded: "
        f"{unaccounted}. Add them to __init__.py's __all__, make them private "
        f"with a leading underscore, or list them in DELIBERATELY_NOT_EXPORTED "
        f"with a reason."
    )


def test_no_module_level_model_variants_alias_was_invented():
    """MODEL_VARIANTS is a per-CLASS attribute on three classes with genuinely
    different contents. A single module-level alias would have to pick one and
    misdescribe the other two, so the package deliberately binds none."""
    import dl_techniques.models.dino as m

    assert not hasattr(m, "MODEL_VARIANTS")
    # ...and the three real tables are reachable through the exported classes,
    # which is what makes the omission harmless rather than a gap.
    assert m.DINOv1.MODEL_VARIANTS
    assert m.DINOv2VisionTransformer.MODEL_VARIANTS
    assert m.DINOv3.MODEL_VARIANTS
    # Non-vacuity: they really are different objects with different contents.
    assert m.DINOv3.MODEL_VARIANTS["giant"]["patch_size"] == (14, 14)
    assert "patch_size" not in m.DINOv1.MODEL_VARIANTS["giant"]
    assert m.DINOv2VisionTransformer.MODEL_VARIANTS["giant"]["ffn_type"] == "swiglu"


# ---------------------------------------------------------------------
# 2. Factory-signature convergence
# ---------------------------------------------------------------------

# The SHARED surface all three factories must agree on, name -> parameter kind.
# Version-specific knobs (dino_out_dim, num_register_tokens, rope_theta, ...)
# are deliberately NOT here: convergence is about the shared surface, not about
# erasing real differences between the three papers (decisions.md D-003).
SHARED_PARAMETERS = {
    "variant": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "image_size": inspect.Parameter.KEYWORD_ONLY,
    "patch_size": inspect.Parameter.KEYWORD_ONLY,
    "num_classes": inspect.Parameter.KEYWORD_ONLY,
    "include_top": inspect.Parameter.KEYWORD_ONLY,
    "kwargs": inspect.Parameter.VAR_KEYWORD,
}


def _factories():
    from dl_techniques.models.dino import (
        create_dino_v1,
        create_dino_v2,
        create_dino_v3,
    )

    return {
        "create_dino_v1": create_dino_v1,
        "create_dino_v2": create_dino_v2,
        "create_dino_v3": create_dino_v3,
    }


def test_the_three_factories_agree_on_the_shared_parameter_surface():
    observed = {}
    for name, fn in _factories().items():
        params = inspect.signature(fn).parameters
        observed[name] = {
            p: params[p].kind for p in SHARED_PARAMETERS if p in params
        }

    for name, got in observed.items():
        assert got == SHARED_PARAMETERS, (
            f"{name} diverges from the shared factory surface.\n"
            f"  expected: {SHARED_PARAMETERS}\n"
            f"  observed: {got}\n"
            f"  missing:  {sorted(set(SHARED_PARAMETERS) - set(got))}"
        )

    # And they agree with EACH OTHER, not merely with the table (this is what
    # catches all three drifting together).
    assert observed["create_dino_v1"] == observed["create_dino_v2"]
    assert observed["create_dino_v2"] == observed["create_dino_v3"]


def test_patch_size_default_is_the_none_sentinel_on_all_three():
    """The precedence rule is only expressible with None. A concrete default
    (e.g. 16) would silently override DINOv3's giant /14 variant."""
    for name, fn in _factories().items():
        default = inspect.signature(fn).parameters["patch_size"].default
        assert default is None, (
            f"{name}'s patch_size default is {default!r}, not None. A concrete "
            f"default cannot express 'the caller said nothing', so it would "
            f"override a variant's own patch_size with no way to opt out."
        )


def test_image_size_accepts_an_int_on_all_three():
    """v3 used to be typed (and behave as) tuple-only."""
    from dl_techniques.models.dino import (
        create_dino_v1,
        create_dino_v2,
        create_dino_v3,
    )

    m1 = create_dino_v1("tiny", image_size=32, patch_size=16, num_classes=3)
    m2 = create_dino_v2("tiny", image_size=32, patch_size=16, num_classes=3)
    m3 = create_dino_v3("tiny", image_size=32, patch_size=16, num_classes=3)
    for model in (m1, m2, m3):
        assert tuple(model.image_size) == (32, 32)


# ---------------------------------------------------------------------
# 3. patch_size-vs-variant precedence
# ---------------------------------------------------------------------

def test_none_patch_size_defers_to_the_v3_giant_variants_own_value():
    """DINOv3.MODEL_VARIANTS['giant'] carries patch_size=(14, 14); every other
    v3 variant carries (16, 16). This is the ONLY place in the family where a
    variant defines its own patch size, so it is the only place the precedence
    rule is observable — which is exactly why it is tested here and not on a
    variant where both branches happen to agree."""
    from dl_techniques.models.dino import DINOv3, create_dino_v3

    # Non-vacuity control: the two variants must genuinely disagree, or this
    # test would pass under either precedence rule.
    assert DINOv3.MODEL_VARIANTS["giant"]["patch_size"] == (14, 14)
    assert DINOv3.MODEL_VARIANTS["small"]["patch_size"] == (16, 16)

    # image_size=112 is divisible by BOTH 14 and 16 on purpose: if the sentinel
    # were replaced by a concrete default the model still BUILDS (with 16), so
    # the assertion below fails rather than an unrelated divisibility ValueError.
    giant = create_dino_v3(
        "giant", image_size=112, num_classes=3,
        embed_dim=32, depth=1, num_heads=4,
    )
    assert tuple(giant.patch_size) == (14, 14), (
        "patch_size=None must defer to the variant's own (14, 14) on giant"
    )

    small = create_dino_v3(
        "small", image_size=112, num_classes=3,
        embed_dim=32, depth=1, num_heads=4,
    )
    assert tuple(small.patch_size) == (16, 16)


def test_an_explicit_patch_size_wins_over_the_variants_own_value():
    from dl_techniques.models.dino import create_dino_v3

    # image_size=112 is divisible by BOTH 14 and 16 on purpose: under an inverted
    # precedence the model still BUILDS (with the variant's 14), so the assertion
    # below is what fails, not an unrelated divisibility ValueError.
    model = create_dino_v3(
        "giant", image_size=112, patch_size=16, num_classes=3,
        embed_dim=32, depth=1, num_heads=4,
    )
    assert tuple(model.patch_size) == (16, 16), (
        "an explicitly passed patch_size must override the variant's (14, 14)"
    )


def test_none_ffn_type_defers_to_the_v2_giant_variant_but_explicit_mlp_wins():
    """Same rule, second instance: DINOv2's giant carries ffn_type='swiglu'.

    Before the None sentinel, 'mlp' was BOTH the default AND the promotion
    trigger (`if variant == 'giant' and ffn_type == 'mlp': ffn_type = 'swiglu'`),
    so an explicit ffn_type='mlp' on giant was silently upgraded with no way to
    opt out. Both branches are asserted so the test cannot pass under the old
    always-promote behaviour.
    """
    from dl_techniques.layers.ffn.swiglu_ffn import SwiGLUFFN
    from dl_techniques.models.dino import create_dino_v2

    common = dict(
        image_size=32, patch_size=16, num_classes=3,
        embed_dim=32, depth=1, num_heads=4, num_register_tokens=0,
    )

    deferred = create_dino_v2("giant", **common)
    explicit_mlp = create_dino_v2("giant", ffn_type="mlp", **common)

    def _ffn_types(model):
        return [type(b.ffn) for b in model.backbone.transformer_blocks]

    assert all(t is SwiGLUFFN for t in _ffn_types(deferred)), (
        "ffn_type=None on giant must defer to the variant's 'swiglu'"
    )
    assert not any(t is SwiGLUFFN for t in _ffn_types(explicit_mlp)), (
        "an explicit ffn_type='mlp' on giant must NOT be promoted to swiglu"
    )


def test_none_num_register_tokens_defers_but_explicit_zero_wins():
    """Third instance of the same rule."""
    from dl_techniques.models.dino import create_dino_v2

    common = dict(
        image_size=32, patch_size=16, num_classes=3,
        embed_dim=32, depth=1, num_heads=4, ffn_type="mlp",
    )

    deferred = create_dino_v2("large", **common)
    explicit_zero = create_dino_v2("large", num_register_tokens=0, **common)

    assert deferred.backbone.num_register_tokens == 4
    assert explicit_zero.backbone.num_register_tokens == 0, (
        "an explicit num_register_tokens=0 on large must not be promoted to 4"
    )


# ---------------------------------------------------------------------
# 4. input_shape refusal
# ---------------------------------------------------------------------

@pytest.mark.parametrize("factory_name", ["create_dino_v1", "create_dino_v2",
                                          "create_dino_v3"])
def test_every_factory_refuses_input_shape(factory_name):
    """Asserted on the MESSAGE, not the type: a bare TypeError is also what
    Python raises for an unrelated signature mistake."""
    fn = _factories()[factory_name]

    with pytest.raises(TypeError, match="use 'image_size' instead"):
        fn("tiny", image_size=32, patch_size=16, num_classes=3,
           input_shape=(32, 32, 3))


def test_input_shape_is_not_a_named_parameter_on_any_factory():
    for name, fn in _factories().items():
        assert "input_shape" not in inspect.signature(fn).parameters, (
            f"{name} still names input_shape; it is the removed spelling"
        )


# ---------------------------------------------------------------------
# 5. README path / import checker
# ---------------------------------------------------------------------

# A path-looking token: something with a '/' and no whitespace, as it appears in
# prose or inside a fenced command. Trailing punctuation is stripped.
_PATH_TOKEN = re.compile(r"(?<![\w/.])((?:[\w.\-]+/)+[\w.\-]*)")
_IMPORT_LINE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import\s+|import\s+([\w.]+))", re.MULTILINE
)

# The repo's own top-level directories. A token is treated as a repo path ONLY if
# it starts with one of these — which is precisely the discipline this README
# follows (`plans/LESSONS.md` I:3: "the natural shorthand is always a dead path",
# so every path in the README is written FULL and repo-relative). This also keeps
# English prose containing a slash ("teacher/student", "ViT-B/8", "save/load")
# out of the path set instead of failing on it.
_REPO_ROOTS = ("src/", "tests/", "research/", "plans/", "imgs/", "results/")


def _readme_path_tokens(text: str):
    tokens = set()
    for match in _PATH_TOKEN.finditer(text):
        token = match.group(1).rstrip(".,;:)`'\"")
        if not token or not token.startswith(_REPO_ROOTS):
            continue
        tokens.add(token)
    return tokens


def test_readme_exists():
    assert README.is_file(), f"missing {README}"


def test_every_repo_path_the_readme_names_resolves():
    """`plans/LESSONS.md`: 'a doc naming paths will eventually describe code that
    does not exist', and 'the natural shorthand is always a dead path'. So the
    README writes FULL repo-relative paths and this checks every one."""
    text = README.read_text(encoding="utf-8")
    tokens = _readme_path_tokens(text)

    # Non-vacuity: the checker must actually have found paths to check. A regex
    # that silently matches nothing is the classic way this guard goes vacuous.
    assert len(tokens) >= 15, (
        f"the path extractor found only {len(tokens)} path-looking tokens in "
        f"README.md — it is probably not matching. Found: {sorted(tokens)}"
    )

    dead = sorted(t for t in tokens if not (REPO_ROOT / t).exists())
    assert not dead, (
        f"README.md names {len(dead)} path(s) that do not resolve under "
        f"{REPO_ROOT}: {dead}"
    )


def test_every_import_the_readme_names_is_importable():
    import importlib

    text = README.read_text(encoding="utf-8")
    modules = set()
    for m in _IMPORT_LINE.finditer(text):
        modules.add(m.group(1) or m.group(2))

    dl_modules = sorted(m for m in modules if m.startswith("dl_techniques"))
    # Non-vacuity: name the modules the README is REQUIRED to import, so an
    # extractor that silently stops matching cannot leave this test green.
    assert "dl_techniques.models.dino" in dl_modules
    assert "dl_techniques.losses" in dl_modules

    for name in dl_modules:
        importlib.import_module(name)


def test_readme_full_signature_block_matches_inspect():
    """§ 3 "Full signatures" prints all three signatures verbatim. A printed
    signature is a claim that rots the instant someone edits a factory, so it is
    diffed against `inspect.signature` here rather than trusted.

    The block is fenced as ```text, not ```python: it is a signature listing, not
    executable code (no body). The README's four ```python blocks ARE executable
    and were run verbatim before shipping.
    """
    text = README.read_text(encoding="utf-8")
    start = text.index("### Full signatures")
    block = text[text.index("```text", start) + len("```text"):]
    block = block[:block.index("```")]

    declared = {}
    for match in re.finditer(
        r"(create_dino_v[123])\(\n(.*?)\n\) ->", block, re.DOTALL
    ):
        params = []
        for line in match.group(2).splitlines():
            line = line.strip().rstrip(",")
            if not line or line == "*":
                continue
            name = line.split(":")[0].split("=")[0].strip().lstrip("*")
            if name:
                params.append(name)
        declared[match.group(1)] = params

    assert set(declared) == set(_factories()), (
        f"the README signature block declares {sorted(declared)}, expected all "
        f"three factories — the extractor or the block shape changed"
    )

    for name, fn in _factories().items():
        actual = list(inspect.signature(fn).parameters)
        assert declared[name] == actual, (
            f"README.md's printed signature for {name} has drifted.\n"
            f"  README:  {declared[name]}\n"
            f"  actual:  {actual}"
        )


def test_readme_symbols_named_in_import_blocks_resolve():
    """`from X import A, B` -> assert X really binds A and B."""
    import importlib

    text = README.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^\s*from\s+(dl_techniques[\w.]*)\s+import\s+\(?([^\n()]+)\)?",
        re.MULTILINE,
    )
    checked = 0
    for match in pattern.finditer(text):
        module = importlib.import_module(match.group(1))
        for symbol in match.group(2).split(","):
            symbol = symbol.strip().rstrip("\\").strip()
            if not symbol or not symbol.isidentifier():
                continue
            assert hasattr(module, symbol), (
                f"README.md does `from {match.group(1)} import {symbol}` but "
                f"that module binds no such name"
            )
            checked += 1
    assert checked >= 5, f"only {checked} README symbols checked — extractor broken"


# ---------------------------------------------------------------------
# 6. The measurement-restatement guard
# ---------------------------------------------------------------------
#
# Stored as integer ten-thousandths and formatted at import time, so THIS FILE
# does not itself contain the literals it forbids. The alternative — scanning
# every file and then exempting the scanner's own source — is a per-file
# exemption, and a per-file exemption list is precisely the shape this guard
# must not grow (it would make it a value-agreement checker in disguise).
# Consequence, for whoever is tempted to widen this tuple: THIS FILE is inside
# the scan set and its docstring below names the § 6.2 effect sizes verbatim, so
# adding any of them here makes the guard fire on its own source.
_FORBIDDEN_TEN_THOUSANDTHS = (
    # section 1 endpoints (mean of last 3 evaluated epochs), improved / baseline
    4326, 4661, 3363, 3285,
    # section 1 deltas vs each arm's own zero-step control, k20
    1426, 1693, 462, 316,
    # the descriptive (NOT pre-registered) arm-to-arm differences
    964, 1377,
    # the secondary k10 endpoint's four deltas
    1364, 1517, 488, 296,
    # the `--seed 1337` zero-step control
    2969,
)
FORBIDDEN_LITERALS = tuple(f"0.{v:04d}" for v in _FORBIDDEN_TEN_THOUSANDTHS)

MEASUREMENT_RECORD = REPO_ROOT / "research" / "2026_dino_ssl_measurements.md"

# The DINO subtree. The record itself (the single home) and
# `research/dino_ssl_measurements_evidence/` (the raw per-epoch records, which
# are the source data rather than a citation of it) are deliberately outside it.
RESTATEMENT_SCAN_DIRS = (
    REPO_ROOT / "src" / "dl_techniques" / "models" / "dino",
    REPO_ROOT / "src" / "train" / "dino",
    REPO_ROOT / "tests" / "test_train" / "test_dino",
    REPO_ROOT / "tests" / "test_models" / "test_dino",
)
# Derived by enumerating the four directories at write time: 7 + 3 + 2 + 7.
MIN_SCANNED_FILES = 19
_SCANNED_SUFFIXES = {".py", ".md", ".json", ".txt", ".csv", ".yaml", ".yml"}


def _restatement_scan_files():
    files = []
    for root in RESTATEMENT_SCAN_DIRS:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in _SCANNED_SUFFIXES:
                files.append(path)
    return files


def test_no_headline_measurement_is_restated_inside_the_dino_subtree():
    """A section-1 HEADLINE measurement endpoint has exactly ONE home:
    ``research/2026_dino_ssl_measurements.md``. Four separate drift escapes were
    caused by a second copy of a number going stale, so this forbids the copy
    rather than trying to agree with it.

    SCOPE, stated exactly, because the claim above is narrower than "no measured
    number is ever restated in this repo" and must not be read as that:

    * COVERED: the 15 literals in ``FORBIDDEN_LITERALS`` — section 1's per-arm
      endpoints, its deltas-vs-own-control, the descriptive arm-to-arm
      differences, the k10 endpoint deltas, and the ``--seed 1337`` control.
    * NOT COVERED: the record's OTHER measured values. MEASURED by scanning
      every ``0.\\d{4,}`` literal in the record against these four directories:
      **16 distinct record values are restated here and are deliberately not
      forbidden** — the eleven § 6.2 per-flag effect sizes in the README
      (``0.0000``, ``0.0020``, ``0.0024``, ``0.0028``, ``0.0063``, ``0.0371``,
      ``0.0387``, ``0.0400``, ``0.0498``, ``0.0518``, ``0.0625``, four of which
      are also quoted in ``train_dino.py``), the control-band WIDTH ``0.0195``,
      and the four band draws listed as exceptions below. The § 6.2 sizes are
      kept ON PURPOSE: each sits inside one long markdown table row, so
      forbidding them would delete almost no lines while destroying the
      MEASURED / NO-DIFFERENCE / UNMEASURED reference a reader ACTS on, and
      exempting them file-by-file would grow the per-file exception list this
      guard must not have. Their drift risk is real and unguarded.
    * NOT COVERED, geographically: only ``RESTATEMENT_SCAN_DIRS`` (four
      directories) is walked. ``src/train/common/``, ``tests/test_losses/``,
      ``tests/test_datasets/`` and the rest of ``research/`` are NOT scanned, so
      a restatement landing there is invisible to this test.

    This is a FORBID-RESTATEMENT check, not a value-agreement check. It never
    asserts that two numbers are equal — only that a measurement endpoint does
    not appear at all outside its home. Absence of a literal is a far weaker and
    more stable claim than text equality (this house has already retired one
    ``inspect.getsource`` text-diff guard for asserting on TEXT, not behaviour),
    and it needs no per-home regex: the homes state their numbers in different
    textual forms, so a value checker would need one parser per site.

    MEASUREMENT ENDPOINTS ONLY. Configuration values are deliberately NOT
    forbidden — ``295`` steps/epoch, ``0.04`` teacher temp, ``1.0`` warmup
    epochs, bank ``64`` / query ``32``, ``60`` epochs are inputs, they legitimately
    appear in code, ``--help`` strings and tests, and forbidding them would make
    this a lint against the source rather than against restated results.

    Two measured values are knowingly OUTSIDE the forbidden set, both because
    they cannot be removed from the subtree without losing something real:

    * ``1518.6`` MiB (the shape-validation peak) lives inside an argparse
      ``help=`` string in ``train_dino.py`` — an executable literal, not prose.
    * The four zero-step control draws at ``--seed 42`` (the ``0.2754``-``0.2949``
      k20 band) are stated ONCE in ``knn_eval.py``'s reading rules, where the
      band WIDTH is the instruction a caller needs before quoting a delta, and
      once historically in ``test_knn_eval.py``.

    The k10 control triple is not forbidden either, for the opposite reason: it
    has no home in the record, so exactly one copy is kept in ``knn_eval.py``.
    Deleting data to satisfy an instrument is the inverse of the point.
    """
    files = _restatement_scan_files()

    # Anti-vacuity (i): the scan actually reached the subtree.
    assert len(files) >= MIN_SCANNED_FILES, (
        f"restatement scan opened only {len(files)} files, expected at least "
        f"{MIN_SCANNED_FILES} — a directory moved or the walk stopped matching, "
        f"and a guard that scans nothing passes forever"
    )

    # Anti-vacuity (ii): there is something to forbid.
    assert FORBIDDEN_LITERALS, "the forbidden-literal inventory is empty"

    # Anti-vacuity (iii): every forbidden literal still exists AT ITS HOME. If a
    # number is renamed or re-derived in the record, this fails loudly instead of
    # leaving the guard watching for a string that can no longer appear anywhere.
    record = MEASUREMENT_RECORD.read_text(encoding="utf-8")
    homeless = [lit for lit in FORBIDDEN_LITERALS if lit not in record]
    assert not homeless, (
        f"{homeless} are forbidden everywhere but no longer appear in "
        f"{MEASUREMENT_RECORD.name} — the inventory has drifted from the home it "
        f"protects and must be re-derived from section 1"
    )

    restatements = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for lit in FORBIDDEN_LITERALS:
            if lit in text:
                restatements.append(f"{path.relative_to(REPO_ROOT)}: {lit}")

    assert not restatements, (
        "a headline measurement endpoint is restated outside "
        f"{MEASUREMENT_RECORD.name}:\n  " + "\n  ".join(restatements) + "\n"
        "Cite the record instead of copying the number. If the value genuinely "
        "cannot leave the code, it is not a citation and belongs in the "
        "documented exception list in this test's docstring."
    )
