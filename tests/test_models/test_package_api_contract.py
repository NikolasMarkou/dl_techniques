"""Package-level API contract for every subpackage of ``dl_techniques.models``.

These tests are cheap (no model is built) and cover the class of defect that
per-model suites structurally cannot see: a package's *public surface*. They are
parameterized over the real directory listing, so a new model package is covered
the moment it is added — nobody has to remember to extend a hand-written list.

Motivating defect: ``models/convnext/__init__.py`` declared

    __all__ = [ConvNeXtV1, ConvNeXtV2, create_convnext_v1, create_convnext_v2]

with the *objects* rather than their names. Every convnext test passed — none of
them imported the package that way — while ``from dl_techniques.models.convnext
import *`` raised ``TypeError: Item in __all__ must be str, not type``. An AST
scan showed convnext was the only package affected; ``test_all_entries_are_strings``
is what keeps it that way.

See ``src/dl_techniques/models/CLAUDE.md`` § "House Model Module Shape" for the
convention these tests enforce.
"""

import ast
import functools
import importlib
import inspect
import re
import sys
import types
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "models"


def _package_names() -> List[str]:
    """Every top-level model subpackage, from the directory listing."""
    return sorted(
        p.name
        for p in MODELS_DIR.iterdir()
        if p.is_dir() and p.name != "__pycache__" and (p / "__init__.py").exists()
    )


PACKAGES = _package_names()


#: Variant-name fragments ordered smallest-first. Used only to pick which named
#: variant the behavioural ``pretrained`` arm probes with; an unranked name falls
#: back to sorted order. A wrong pick cannot make the probe *wrong* -- every
#: ``from_variant`` validates the name against the very table the key came from --
#: it can only make it more expensive, which is why smallest-first is preferred.
_VARIANT_SIZE_ORDER = (
    "nano", "pico", "micro", "tiny", "mini", "xs", "small", "light",
    "base", "medium", "large", "xl", "xlarge", "huge",
)


def _smallest_variant(keys) -> str:
    """The smallest-looking key of a MODEL_VARIANTS table, deterministically."""

    def rank(key: str):
        low = key.lower()
        for i, token in enumerate(_VARIANT_SIZE_ORDER):
            if token in low:
                return (i, key)
        return (len(_VARIANT_SIZE_ORDER), key)

    return sorted(keys, key=rank)[0]


def _variant_table_for(fn):
    """The ``MODEL_VARIANTS`` table a ``variant``-taking callable looks names up in.

    Contract: takes any callable; returns a non-empty ``dict`` or ``None``.
    Resolution order, and why:

    1. ``fn.__self__`` -- a bound ``from_variant`` classmethod carries its owning
       class, which is where the house rule (``models/CLAUDE.md`` Axis 2) puts the
       table. This is the exact channel step 6 was told to read, in place of a
       hand-maintained variant table that would rot the moment a package renames
       a size.
    2. otherwise the *defining module*'s first class (``dir()`` order, so
       deterministic) carrying a ``MODEL_VARIANTS``. This is what reaches the
       ``create_<x>_with_head(<x>_variant, task_config)`` factories, which are
       plain functions sitting beside their model class.

    If (2) ever picks the wrong class in a two-model module, the probe fails LOUDLY
    with the package's own ``ValueError: Unknown variant`` rather than silently
    skipping the site -- the failure mode a discovery arm is allowed to have.
    """
    owner = getattr(fn, "__self__", None)
    if inspect.isclass(owner):
        table = getattr(owner, "MODEL_VARIANTS", None)
        if isinstance(table, dict) and table:
            return table
    module = sys.modules.get(getattr(fn, "__module__", ""))
    if module is None:
        return None
    for name in dir(module):
        obj = getattr(module, name, None)
        if not (inspect.isclass(obj) and obj.__module__ == fn.__module__):
            continue
        table = getattr(obj, "MODEL_VARIANTS", None)
        if isinstance(table, dict) and table:
            return table
    return None


def _resolve_required_arg(param_name: str, fn) -> Tuple[Any, str]:
    """Return ``(value, "")`` for a required argument, or ``(None, reason)``.

    The table is keyed by ARGUMENT KIND, never by call site: ``variant`` comes from
    the model's own ``MODEL_VARIANTS``, and the handful of remaining kinds are
    sizes, which any value satisfies because the contract under test
    (``pretrained=True`` raises) fires before anything is built. A per-factory
    table of variant names is exactly what this replaces -- it would have to be
    hand-maintained, and a renamed variant would silently drop a site.
    """
    if param_name == "variant" or param_name.endswith("_variant"):
        table = _variant_table_for(fn)
        if table is None:
            return None, "no MODEL_VARIANTS table is reachable from this callable"
        return _smallest_variant(list(table)), ""
    if param_name == "task_config":
        from dl_techniques.layers.heads.nlp.task_types import (
            NLPTaskConfig,
            NLPTaskType,
        )

        return (
            NLPTaskConfig(
                name="pretrained-contract-probe",
                task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=2,
            ),
            "",
        )
    if param_name in ("num_classes", "num_labels", "input_features", "output_features"):
        return 2, ""
    if param_name == "vocab_size":
        return 32, ""
    if param_name == "input_shape":
        return (32, 32, 3), ""
    return None, f"required argument {param_name!r} has no generic resolution"


def _iter_pretrained_callables():
    """Yield ``(label, fn, signature)`` for every public ``pretrained``-taking
    callable reachable from a model package's own namespace, deduplicated.

    Contract: iterates ``PACKAGES``, importing each; a package that fails to import
    is skipped silently (``test_every_package_imports`` is what catches that). The
    label is ``"<pkg>.<name>"`` or ``"<pkg>.<Class>.from_variant"``; dedup key is
    ``(module, qualname)`` so a symbol re-exported by two packages is probed once.

    Shared by ``_pretrained_factories`` and by the coverage test below so the two
    cannot drift in what they walk.
    """
    seen = set()
    for pkg in PACKAGES:
        try:
            module = importlib.import_module(f"dl_techniques.models.{pkg}")
        except Exception:  # covered by test_package_imports
            continue
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            candidates = []
            if inspect.isfunction(obj):
                candidates.append((f"{pkg}.{name}", obj))
            elif inspect.isclass(obj) and callable(getattr(obj, "from_variant", None)):
                candidates.append((f"{pkg}.{name}.from_variant", obj.from_variant))
            for label, fn in candidates:
                try:
                    sig = inspect.signature(fn)
                except (TypeError, ValueError):
                    continue
                if "pretrained" not in sig.parameters:
                    continue
                key = (fn.__module__, getattr(fn, "__qualname__", label))
                if key in seen:
                    continue
                seen.add(key)
                yield label, fn, sig


def _pretrained_factories() -> Tuple[List[Tuple[str, Any, dict]], List[Tuple[str, str]]]:
    """Every public model factory taking ``pretrained``, with the arguments needed
    to call it.

    Returns ``(reached, unreached)``. ``reached`` is ``(label, fn, kwargs)`` where
    ``kwargs`` satisfies every *required* parameter other than ``pretrained``;
    ``unreached`` is ``(label, reason)``.

    **Widened 2026-08-20 (plan-2026-08-19T163559-499b6f0e step 6(iii)).** The arm
    previously kept only factories callable with NO other required argument, which
    measured **21 of the 45-site AST population** -- every ``from_variant`` and
    every ``create_<x>_with_head`` in the tree was outside it, i.e. exactly the
    entry points a user reaches for a *named* model. The resolver above lifts that
    to 42; the three it still cannot reach are named, with reasons, in
    ``_PRETRAINED_UNREACHABLE_EVIDENCE`` below.
    """
    reached: List[Tuple[str, Any, dict]] = []
    unreached: List[Tuple[str, str]] = []
    for label, fn, sig in _iter_pretrained_callables():
        kwargs = {}
        reason = ""
        for name, param in sig.parameters.items():
            if name == "pretrained":
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            value, reason = _resolve_required_arg(name, fn)
            if reason:
                break
            kwargs[name] = value
        if reason:
            unreached.append((label, reason))
        else:
            reached.append((label, fn, kwargs))
    return (
        sorted(reached, key=lambda t: t[0]),
        sorted(unreached, key=lambda t: t[0]),
    )


PRETRAINED_FACTORIES, PRETRAINED_UNREACHED = _pretrained_factories()

#: The AST-population sites the widened arm still cannot call, each keyed by
#: ``(relpath, function name)`` -- never by line, for the reason
#: ``_SCHEDULED_FIXES`` gives -- and each carrying the evidence that clears it.
#: Measured 2026-08-20: 3 of 45, so the arm reaches **42 of 45**.
#:
#: ``test_the_uncovered_sites_are_the_named_ones`` fails if this set stops
#: matching, in EITHER direction: a new uncovered site is a factory that silently
#: left behavioural coverage, and a recorded site that became reachable is a
#: waiver hiding nothing.
_PRETRAINED_UNREACHABLE_EVIDENCE = {
    # bfunet's variants live in a module-level dict consumed by a plain factory;
    # the class carries no MODEL_VARIANTS, so the variant resolver has nothing to
    # read. Its `pretrained` contract is still covered statically by
    # `test_no_pretrained_branch_only_logs`, and it was probed by hand at D-003
    # (raises NotImplementedError). DISCOVERY reaches it; the RESOLVER does not.
    (
        "models/bias_free_denoisers/bfunet.py",
        "create_bfunet_variant",
    ): "no MODEL_VARIANTS table is reachable from the callable",
    # These two are not exported by their package `__init__.py`, so no namespace
    # walk can see them at all -- their four siblings (bert, distilbert,
    # modern_bert, tree_transformer `create_*_with_head`) ARE exported and ARE
    # reached, with the same resolver. The gap is an EXPORT gap, not a resolver
    # gap: this arm reaches them the day they are exported, and the fix is in the
    # package, not in this file.
    (
        "models/fnet/model.py",
        "create_fnet_with_head",
    ): "not exported from dl_techniques.models.fnet",
    (
        "models/mamba/mamba_v1.py",
        "create_mamba_with_head",
    ): "not exported from dl_techniques.models.mamba",
}


def _population_key(fn) -> Tuple[str, str]:
    """The ``(relpath, name)`` an imported callable occupies in the AST population.

    Contract: ``fn.__module__`` must live under ``dl_techniques``; returns the path
    relative to ``src/dl_techniques`` plus the callable's own name (the last
    component of ``__qualname__``, so ``BERT.from_variant`` -> ``from_variant``).
    Measured 2026-08-20: this key is UNIQUE across all 45 population entries and
    every reached factory maps into it, which is what makes the coverage claim a
    set difference rather than a count comparison.
    """
    rel = fn.__module__.split("dl_techniques.", 1)[-1].replace(".", "/") + ".py"
    return rel, getattr(fn, "__qualname__", fn.__name__).split(".")[-1]


def _sweep_pretrained_population(roots=None, src_root=None):
    """The AST population: every public ``def`` under ``models/`` taking ``pretrained``.

    Contract: returns ``[(relpath, lineno, funcname)]``. This is the DENOMINATOR the
    behavioural arm's coverage is measured against, so the coverage claim is a
    ratio against the tree rather than against the arm's own discovery -- an arm
    that stopped seeing half the tree would otherwise report 100%.
    """
    roots = (MODELS_DIR,) if roots is None else roots
    src_root = MODELS_DIR.parent if src_root is None else src_root
    out = []
    for rel, tree in _iter_modules(roots, src_root):
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            args = node.args
            names = [
                a.arg for a in args.posonlyargs + args.args + args.kwonlyargs
            ]
            if "pretrained" in names:
                out.append((rel, node.lineno, node.name))
    return sorted(out)


def _all_node(init_path: Path):
    """Return the ``__all__`` assignment node, or None if the file has none."""
    tree = ast.parse(init_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "__all__" for t in node.targets
        ):
            return node.value
    return None


class TestPackageDiscovery:
    """The parameterization itself must not silently collapse to nothing."""

    def test_packages_were_found(self):
        assert MODELS_DIR.is_dir(), f"models dir not found at {MODELS_DIR}"
        assert len(PACKAGES) > 50, (
            f"expected the full model package set, found {len(PACKAGES)}: {PACKAGES}"
        )

    def test_parent_init_is_not_a_public_api(self):
        """``dl_techniques.models`` itself exports nothing; import from the subpackage.

        Pinned because the parent init being empty is a documented convention
        (``models/CLAUDE.md``), not an oversight, and a well-meaning "fix" that
        re-exports all 73 packages there would make importing any single model
        pull in every model in the library.
        """
        parent = importlib.import_module("dl_techniques.models")
        assert not getattr(parent, "__all__", []), (
            "dl_techniques.models must stay empty; import from the subpackage"
        )


@pytest.mark.parametrize("pkg", PACKAGES)
class TestAllDeclaration:
    """``__all__``, where declared, must be well-formed and honest."""

    def test_all_entries_are_strings(self, pkg: str):
        """``__all__`` must hold NAMES, not the objects themselves.

        A list of objects passes every ordinary import but breaks ``import *``
        with ``TypeError: Item in __all__ must be str``.
        """
        node = _all_node(MODELS_DIR / pkg / "__init__.py")
        if node is None:
            pytest.skip(f"{pkg} declares no __all__")
        assert isinstance(node, (ast.List, ast.Tuple)), (
            f"{pkg}: __all__ must be a list or tuple literal"
        )
        offenders = [
            ast.dump(e) for e in node.elts if not isinstance(e, ast.Constant)
        ]
        assert not offenders, (
            f"{pkg}: __all__ must contain string names, not objects. "
            f"Offending entries: {offenders}"
        )

    def test_all_entries_resolve(self, pkg: str):
        """Every name in ``__all__`` must actually be bound by the package."""
        module = importlib.import_module(f"dl_techniques.models.{pkg}")
        declared = getattr(module, "__all__", None)
        if not declared:
            pytest.skip(f"{pkg} declares no __all__")
        missing = [name for name in declared if not hasattr(module, name)]
        assert not missing, f"{pkg}: __all__ names not bound by the package: {missing}"

    def test_no_duplicate_entries(self, pkg: str):
        module = importlib.import_module(f"dl_techniques.models.{pkg}")
        declared = getattr(module, "__all__", None)
        if not declared:
            pytest.skip(f"{pkg} declares no __all__")
        duplicates = sorted({n for n in declared if declared.count(n) > 1})
        assert not duplicates, f"{pkg}: duplicate __all__ entries: {duplicates}"


@pytest.mark.parametrize("pkg", PACKAGES)
class TestPackageImports:
    """A package must import cleanly, and its submodules must not be dead."""

    def test_package_imports(self, pkg: str):
        importlib.import_module(f"dl_techniques.models.{pkg}")

    def test_star_import_succeeds(self, pkg: str):
        """``from dl_techniques.models.<pkg> import *`` must not raise.

        This is the exact call that the convnext ``__all__`` defect broke while
        the package's own 73-test suite stayed green.
        """
        namespace: dict = {}
        exec(f"from dl_techniques.models.{pkg} import *", namespace)  # noqa: S102


class TestDecisionAnchorsIntact:
    """``# DECISION <plan-id>/D-NNN`` comments are a tracked, append-only record.

    They resolve through ``plans/ANCHORS.md`` and are the only thing keeping a
    non-obvious code choice explicable after its plan directory is gone. A
    comment-tidying sweep that removes one destroys that link silently, so the
    count is pinned here rather than left to reviewer attention.

    This is deliberately a floor, not an equality: adding new anchors is normal
    and must not fail the suite; losing them is what must fail.

    The pattern must match the anchor FORM (``# DECISION <plan-id>/D-NNN``), not
    the bare phrase. A plain ``grep "# DECISION"`` reports 284 because
    ``bias_free_denoisers/bfconvunext.py`` mentions ``# DECISION`` inside a
    docstring while pointing at the real anchor below it. Counting that mention
    as an anchor is the same false positive this repo has hit repeatedly with
    mechanical scans, and it would make the floor un-holdable the moment the
    docstring were reworded.
    """

    #: True anchors under src/dl_techniques/models/, measured 2026-08-14 and
    #: confirmed identical at commit 4300b2f19 (pre-work) and at HEAD.
    MINIMUM_ANCHOR_COUNT = 283

    ANCHOR_RE = re.compile(r"# DECISION [A-Za-z0-9_.-]+/D-\d+")

    def test_anchor_count_has_not_regressed(self):
        total = 0
        for path in MODELS_DIR.rglob("*.py"):
            total += len(self.ANCHOR_RE.findall(path.read_text()))
        assert total >= self.MINIMUM_ANCHOR_COUNT, (
            f"DECISION anchor count fell to {total}, below the pinned floor of "
            f"{self.MINIMUM_ANCHOR_COUNT}. A comment cleanup has removed tracked "
            f"provenance anchors; restore them and see plans/ANCHORS.md."
        )


class TestNoPlaceholderWeightURLs:
    """No model may ship a pretrained-weights table of unreachable URLs.

    Until 2026-08-14, 83 ``https://example.com/...`` URLs sat across 12 files in
    ``PRETRAINED_WEIGHTS`` tables. Each was paired with a ``try/except`` in
    ``from_variant`` that logged a warning and continued, so ``pretrained=True``
    returned a randomly-initialized model and the caller was never told. The
    house contract is now that ``_download_weights`` raises
    ``NotImplementedError``; this test stops the placeholder pattern coming back.
    """

    def test_no_example_com_urls(self):
        offenders = []
        for path in MODELS_DIR.rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), start=1):
                if "example.com" in line:
                    offenders.append(f"{path.relative_to(MODELS_DIR)}:{i}")
        assert not offenders, (
            "placeholder weight URLs are forbidden; make _download_weights raise "
            f"NotImplementedError instead. Found: {offenders}"
        )


def _log_only(body: List[ast.stmt]) -> bool:
    """True if every statement in ``body`` is a bare ``logger.*(...)`` call."""
    if not body:
        return False
    for stmt in body:
        if not isinstance(stmt, ast.Expr):
            return False
        call = stmt.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logger"
        ):
            return False
    return True


class TestPretrainedNeverSilentlyRandom:
    """``pretrained`` must never resolve to "log something and hand back random weights".

    ``TestNoPlaceholderWeightURLs`` above only forbids the literal ``example.com``.
    That is the *symptom* the 2026-08-14 sweep happened to leave behind, not the
    contract: nine factories -- ``dino_v2``, ``dino_v3``, ``swin_transformer``,
    ``mobilenet_v1``-``v4``, ``mobile_clip_v1``, ``mobile_clip_v2`` -- had no URL
    table at all, wrote ``if pretrained: logger.warning(...)``, and passed the
    ``example.com`` test while returning an untrained model to a caller who asked
    for a trained one. ``mobile_clip_v2`` is recent code, so the contract was not
    reaching new work either.

    Two arms, deliberately:

    * a **static** arm that reads every ``models/`` module, so packages whose
      factories need variant names / vocab sizes are still covered;
    * a **behavioural** arm over the factories that are callable with no other
      argument, so the assertion is about what the code *does*, not what it looks
      like. Both are derived from the real directory listing.
    """

    def test_no_pretrained_branch_only_logs(self):
        """No ``if pretrained:`` branch may consist solely of logging.

        Matching on the AST shape rather than on the warning text is deliberate:
        the nine sites shared the string "Pretrained weights are not yet
        implemented", and a guard keyed to that string would be defeated by
        rewording it. A branch that only logs cannot be doing anything else --
        it must raise, or load something.
        """
        offenders = []
        for path in sorted(MODELS_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                if not any(
                    isinstance(n, ast.Name) and n.id == "pretrained"
                    for n in ast.walk(node.test)
                ):
                    continue
                if _log_only(node.body):
                    offenders.append(f"{path.relative_to(MODELS_DIR)}:{node.lineno}")
        assert not offenders, (
            "a `pretrained` branch that only logs returns a randomly initialized "
            "model to a caller who asked for a trained one. Raise "
            "NotImplementedError instead (see models/CLAUDE.md Axis 3 and "
            f"resnet/model.py). Found: {offenders}"
        )

    def test_the_behavioural_arm_reaches_almost_the_whole_population(self):
        """The arm must cover >= 40 of the 45-site AST population (step 6(iii)).

        Two separate assertions, deliberately:

        * a **ratio against the tree** -- at most five of the AST population may be
          unreached. Measuring coverage against the arm's own discovery would let a
          decayed walk report 100% of a shrinking denominator;
        * an **absolute floor**, derived as ``int(0.8 * 42)`` from the 2026-08-20
          measurement (42 reached, 45 in the population). It is deliberately NOT
          set a couple of sites under 42: Phase 4 of this plan edits these very
          packages, and a floor with 5% headroom trips on a legitimate refactor
          rather than on the decay it exists to catch.
        """
        population = _sweep_pretrained_population()
        assert len(population) >= 40, (
            f"the AST population collapsed to {len(population)}; the walk stopped "
            "seeing models/"
        )
        assert len(PRETRAINED_FACTORIES) >= 33, (
            "expected the widened behavioural arm to reach dozens of `pretrained` "
            f"factories; discovery found {len(PRETRAINED_FACTORIES)}: "
            f"{[label for label, _, _ in PRETRAINED_FACTORIES]}"
        )
        assert len(population) - len(PRETRAINED_FACTORIES) <= 5, (
            f"the behavioural arm reaches {len(PRETRAINED_FACTORIES)} of "
            f"{len(population)} AST sites; step 6(iii) requires >= 40 of 45. "
            f"Unreached: {PRETRAINED_UNREACHED}"
        )

    def test_the_uncovered_sites_are_the_named_ones(self):
        """A site may not drop out of behavioural coverage unnoticed.

        The waiver-liveness rule of this file, applied to a coverage list rather
        than to an offender list. Note the two DIFFERENT ways a site stays
        uncovered, both folded into one set difference on purpose: the resolver
        cannot satisfy its arguments (bfunet), or the namespace walk never sees it
        because the package does not export it (fnet, mamba). Reporting only the
        first -- which is all the resolver itself knows about -- would have hidden
        the two export gaps entirely.
        """
        population = {(rel, name) for rel, _, name in _sweep_pretrained_population()}
        covered = {_population_key(fn) for _, fn, _ in PRETRAINED_FACTORIES}
        uncovered = population - covered
        expected = set(_PRETRAINED_UNREACHABLE_EVIDENCE)
        assert uncovered == expected, (
            "the set of `pretrained` sites the behavioural arm does not call "
            f"changed.\n  newly uncovered (no evidence recorded): {sorted(uncovered - expected)}"
            f"\n  recorded but now covered (delete the entry): {sorted(expected - uncovered)}"
            f"\n  resolver-level reasons: {PRETRAINED_UNREACHED}"
        )

    @pytest.mark.parametrize(
        "label,factory,kwargs",
        PRETRAINED_FACTORIES,
        ids=[label for label, _, _ in PRETRAINED_FACTORIES],
    )
    def test_pretrained_true_raises(self, label: str, factory, kwargs):
        """``pretrained=True`` must raise; no public weights ship with this repo.

        Cheap by construction: the smallest named variant, and never a forward
        pass. The contract is that the raise happens BEFORE any weight is
        allocated, so a site that got expensive here is itself a finding.
        """
        with pytest.raises(NotImplementedError):
            factory(pretrained=True, **kwargs)


def _docstring_line_numbers(path: Path) -> set:
    """1-based line numbers covered by any module/class/function docstring."""
    lines: set = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # covered by the import tests
        return lines
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
            first.value.value, str
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


class TestNoKeras2Residues:
    """The forward path must use Keras 3 spellings.

    ``keras.backend.GradientTape`` does not exist in Keras 3 at all: it sat in
    ``latent_gmm_registration.train_step`` and made the model untrainable while
    its suite stayed green, because every test was forward-pass only.
    """

    def test_no_keras_backend_calls(self):
        offenders = []
        for path in MODELS_DIR.rglob("*.py"):
            docstring_lines = _docstring_line_numbers(path)
            for i, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Prose is not code. Before this exclusion, a module docstring
                # explaining *why* `keras.backend.` must not be used failed the
                # suite -- the guard could not be documented in the tree it guards.
                if i in docstring_lines:
                    continue
                if "keras.backend." in line:
                    offenders.append(f"{path.relative_to(MODELS_DIR)}:{i} {stripped}")
        assert not offenders, (
            "use keras.config.floatx()/epsilon() and tf.GradientTape; "
            f"keras.backend.* found at: {offenders}"
        )


# ---------------------------------------------------------------------------
# Silent-kwarg-drop guard (the inverse of the factories' strict-drop raise).
# ---------------------------------------------------------------------------

LAYERS_DIR = Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "layers"

#: Paths in every sweep below are reported relative to this, so a waiver key is
#: stable no matter which root the file was reached through.
SRC_ROOT = MODELS_DIR.parent

#: Root set of the registry sweep. Unchanged since the guard shipped: the three
#: registry-backed factories are called from ``models/`` and from the shared
#: transformer blocks, and widening it has never been measured to add a site.
_REGISTRY_SWEEP_ROOTS = (MODELS_DIR, LAYERS_DIR / "transformers")

#: Root set of the two NORM sweeps, deliberately WIDER than the registry sweep's.
#: ``grep -rn "TransformerLayer(" src/dl_techniques/`` (2026-08-19) found three
#: construction sites outside ``layers/transformers/`` -- ``layers/blt_blocks.py``,
#: ``layers/graphs/relational_graph_transformer_blocks.py`` and
#: ``layers/reasoning/hrm_reasoning_module.py`` -- so the narrower root set would
#: have been silently blind to them. Measured cost of widening: 0 extra hits, and
#: the sweep still runs in well under a second.
_NORM_SWEEP_ROOTS = (MODELS_DIR, LAYERS_DIR)

#: The three factories whose registries declare a per-type parameter set. Maps
#: the factory's function name to (its type-selecting parameter, its registry).
#: ``norms``/``activations``/``sampling`` are deliberately absent: they have no
#: ``required_params``/``optional_params`` registry, so there is no declared-param
#: ground truth to diff a call site against.
_REGISTRY_FACTORIES = {
    "create_attention_layer": "attention_type",
    "create_ffn_layer": "ffn_type",
    "create_embedding_layer": "embedding_type",
}

#: Attributes every Keras layer/model carries. A call site is never "demonstrably
#: had the value" because ``keras.layers.Layer`` set one of these.
_KERAS_BASE_ATTRS = frozenset(
    {
        "built",
        "name",
        "trainable",
        "dtype",
        "dtype_policy",
        "supports_masking",
        "activity_regularizer",
        "input_spec",
    }
)

#: Sites that are SCHEDULED WORK, not accepted exceptions.
#:
#: **EMPTY as of 2026-08-18.** All 13 entries were fixed at their call sites; the
#: constant stays so the guard's shape is unchanged and so the next measured
#: instance has somewhere to be waived while its fix lands. It must never be used
#: to park an omission that is actually deliberate -- that is what
#: ``_NAME_COLLISIONS`` below is for, and it carries the read that clears it.
#:
#: Twelve of the thirteen were forwarded verbatim. The thirteenth pair -- both
#: Qwen3 wrappers' ``positional_learned`` ``dropout_rate`` -- was a REFUTATION:
#: the classes already applied that rate through a second standalone ``Dropout``
#: on the same tensor, so forwarding alone would have stacked two dropouts
#: (effective ``1-(1-p)^2``). They are cleared by forwarding the kwarg AND
#: deleting the redundant layer; see
#: ``tests/test_models/test_qwen/test_embedding_dropout_applied_once.py``.
#:
#: Key is ``(path relative to src/dl_techniques, class, factory type, param)``.
#: Deliberately NOT keyed by line number: the fixes themselves move lines, and a
#: waiver that silently stops matching is a waiver that hides a live defect.
_SCHEDULED_FIXES: set = set()

#: Name collisions this static predicate cannot resolve, cleared by manual read.
#:
#: Unlike ``_SCHEDULED_FIXES`` these are permanent: the enclosing class stores an
#: attribute that merely *shares a name* with a declared factory parameter.
#:
#: Five of the six are the ViT-family ``scale``, which is the variant-size string
#: (``"base"``, ``"large"``, ...; e.g. ``vit/model.py`` ``self.scale = str(scale)``)
#: and has nothing to do with ``positional_learned``'s embedding-scale parameter.
#: No AST predicate can tell the two apart -- both are ``self.scale`` assigned
#: from a same-named ``__init__`` argument -- so the discrimination is recorded
#: here, with its evidence, rather than pretended at.
#:
#: The sixth is ``ViT.activation``, added 2026-08-18 when the four REAL drops at
#: the same call site were fixed. ``patch_2d`` declares an ``activation``
#: (default ``'linear'``) and ``ViT`` stores ``self.activation``, but ViT's is the
#: FFN activation -- documented as such in ``vit/model.py`` and passed to every
#: ``TransformerLayer`` -- and forwarding its ``'gelu'`` default into the patch
#: projection would make the stem nonlinear, which no ViT is. See the
#: ``D-022`` anchor at that call site.
_NAME_COLLISIONS = {
    ("models/vit/model.py", "ViT", "patch_2d", "activation"),
    ("models/vit/model.py", "ViT", "positional_learned", "scale"),
    ("models/vit_hmlp/model.py", "ViTHMLP", "positional_learned", "scale"),
    ("models/vit_siglip/model.py", "SigLIPVisionTransformer", "positional_learned", "scale"),
    ("models/beit/model.py", "BeitModel", "positional_learned", "scale"),
    ("models/energy_transformer/model.py", "EnergyTransformerBackbone", "positional_learned", "scale"),
}


def _declared_params(registry: dict, type_name: str):
    """Every parameter a registry entry declares, or None if the type is unknown."""
    entry = registry.get(type_name)
    if entry is None:
        return None
    return set(entry.get("required_params") or []) | set(entry.get("optional_params") or {})


def _init_stored_attrs(cls: ast.ClassDef) -> set:
    """``self.<name>`` targets assigned anywhere in the class's ``__init__``."""
    init = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    if init is None:
        return set()
    stored = set()
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                stored.add(target.attr)
    return stored - _KERAS_BASE_ATTRS


def _callee_name(node: ast.Call) -> str:
    """The bare name a call node invokes, for ``f(...)`` and ``mod.f(...)`` alike.

    Contract: takes an ``ast.Call``; returns the callee's last name component, or
    ``""`` when the callee is neither a ``Name`` nor an ``Attribute`` (a call on a
    subscript or a call result, which no sweep in this file matches). Shared by all
    three sweeps below -- do not re-derive it inline.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _iter_classes(roots, src_root):
    """Yield ``(relpath, ast.ClassDef)`` for every class under ``roots``.

    Contract: ``roots`` is any iterable of directories to ``rglob("*.py")``;
    ``src_root`` is the directory paths are reported relative to (it must be a
    parent of every root, or ``relative_to`` raises). Files that do not parse are
    skipped silently -- ``test_every_package_imports`` is what catches those.
    Shared by the three call-site sweeps so they cannot drift in what they walk.
    """
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            rel = path.relative_to(src_root).as_posix()
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:  # covered by the import tests
                continue
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                yield rel, cls


def _sweep_factory_call_sites() -> Tuple[List[tuple], List[str]]:
    """Return ``(dropped, dynamic)`` for every registry-backed factory call site.

    ``dropped`` holds ``(relpath, lineno, class, type, param)`` for each call that
    omits a parameter the registry declares *and* the enclosing class stores --
    the demonstrable "the caller had the value and did not forward it" signature.
    ``dynamic`` holds ``"path:line"`` for call sites whose type is a variable,
    which cannot be resolved statically.
    """
    from dl_techniques.layers.attention.factory import ATTENTION_REGISTRY
    from dl_techniques.layers.embedding.factory import EMBEDDING_REGISTRY
    from dl_techniques.layers.ffn.factory import FFN_REGISTRY

    registries = {
        "create_attention_layer": (_REGISTRY_FACTORIES["create_attention_layer"], ATTENTION_REGISTRY),
        "create_ffn_layer": (_REGISTRY_FACTORIES["create_ffn_layer"], FFN_REGISTRY),
        "create_embedding_layer": (_REGISTRY_FACTORIES["create_embedding_layer"], EMBEDDING_REGISTRY),
    }
    dropped: List[tuple] = []
    dynamic: List[str] = []

    for rel, cls in _iter_classes(_REGISTRY_SWEEP_ROOTS, SRC_ROOT):
        stored = _init_stored_attrs(cls)
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            fname = _callee_name(node)
            if fname not in registries:
                continue
            type_kw, registry = registries[fname]
            type_node = node.args[0] if node.args else None
            for kw in node.keywords:
                if kw.arg == type_kw:
                    type_node = kw.value
            if not (
                isinstance(type_node, ast.Constant)
                and isinstance(type_node.value, str)
            ):
                dynamic.append(f"{rel}:{node.lineno}")
                continue
            declared = _declared_params(registry, type_node.value)
            if declared is None:
                dynamic.append(f"{rel}:{node.lineno} (unknown type {type_node.value!r})")
                continue
            # A `**config` unpack can carry anything; only literal kwargs
            # are provably present, so an unpack clears the whole call.
            if any(kw.arg is None for kw in node.keywords):
                continue
            passed = {kw.arg for kw in node.keywords}
            for param in sorted((declared - passed) & stored):
                dropped.append(
                    (rel, node.lineno, cls.name, type_node.value, param)
                )
    return dropped, dynamic


class TestFactoryKnobsAreForwarded:
    """A parameter the caller stores and the factory declares must be passed.

    ``attention``/``ffn``/``embedding`` factories are STRICT about the *undeclared*
    direction -- passing a key the type does not accept raises
    ``STRICT_DROPPED_KEY_MARKER``. This is the inverse and until now unguarded
    direction: the call site hand-writes its kwarg list, the enclosing class holds
    the value in ``self.<name>``, ``get_config()`` faithfully serializes it, and
    the factory never sees it. Nothing raises, nothing warns, and every existing
    test passes because the built layer is *valid* -- just not the one that was
    asked for. Thirteen live instances across five call sites were found the day
    this guard was written, including a Swin/ViT patch embedding that trains at
    glorot no matter which initializer the caller supplies.

    The predicate is deliberately narrow, and each narrowing is a place it can
    miss a real defect:

    * only **literal** ``type=`` strings (a variable type has no static declared-
      param set; those sites are reported non-fatally by
      ``test_dynamic_call_sites_are_reported``);
    * only calls with **no** ``**unpack`` (an unpack may well carry the param);
    * only the three registry-backed factories -- ``create_normalization_layer``
      has no ``required_params``/``optional_params`` registry, so the norm-epsilon
      family of this same defect is out of reach here. **Superseded 2026-08-19**:
      the norm family is now covered by ``TestNormalizationKnobsAreForwarded`` and
      ``TestTransformerLayerNormArgsAreForwarded`` below, which reach it through
      ``inspect.signature`` and through the ``TransformerLayer(...)`` indirection
      respectively -- but THIS predicate still does not, and must not be widened
      to try. This is also why the two by-design omissions catalogued in the sweep
      that motivated this guard (``adaln_zero.py`` and ``text_encoder.py``'s
      ``**norm_config`` unpack) need no waiver here: both are
      ``create_normalization_layer`` calls and this predicate never reaches them.
      Waiving them anyway would have been a waiver guarding nothing;
    * only params the enclosing class stores in ``__init__``, which is what makes
      a hit *demonstrable* rather than merely suspicious.
    """

    def test_no_declared_and_stored_param_is_dropped(self):
        # DECISION plan-2026-08-18T140459-7991552f/D-015
        # Guards the SILENT-DROP direction of the factory contract. The opposite
        # direction -- a caller passing a key the type does not declare -- is
        # already guarded at runtime by the strict raise added in D-011/D-023
        # (`STRICT_DROPPED_KEY_MARKER`). That raise cannot see this direction at
        # all: an omitted kwarg is indistinguishable from a caller who wanted the
        # default. Do NOT "simplify" this into a runtime check inside the
        # factories -- the information needed (does the CALLER hold this value?)
        # exists only at the call site, statically. Do NOT widen it by dropping
        # the "enclosing class stores it" clause either: without that clause every
        # deliberate use of a factory default becomes a failure. See
        # plans/.../findings/dead-knob-systematic-sweep.md Part D, which rejected
        # a per-call-site test (a) and a `strict_forward` helper (b) in favour of
        # this AST guard (c).
        dropped, _ = _sweep_factory_call_sites()
        waived = _SCHEDULED_FIXES | _NAME_COLLISIONS
        offenders = [
            f"{rel}:{line} {cls}: create_*_layer({typ!r}) drops {param!r} "
            f"(self.{param} is stored in __init__)"
            for rel, line, cls, typ, param in dropped
            if (rel, cls, typ, param) not in waived
        ]
        assert not offenders, (
            "a factory call site drops a parameter its enclosing class stores. "
            "Forward it, or -- if the omission is deliberate -- add it to "
            "_NAME_COLLISIONS with the read that clears it. Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_waivers_still_match_a_real_site(self):
        """A waiver that no longer matches anything is a waiver hiding nothing.

        Both lists are keyed by (path, class, type, param) precisely so they keep
        matching as the fixes move lines. A stale entry means either the fix
        landed (delete the entry) or the code moved (re-key it) -- and a stale
        entry left in place would silently start waiving nothing while looking
        like it still guards something.
        """
        dropped, _ = _sweep_factory_call_sites()
        live = {(rel, cls, typ, param) for rel, _, cls, typ, param in dropped}
        stale = sorted((_SCHEDULED_FIXES | _NAME_COLLISIONS) - live)
        assert not stale, (
            "waiver entries no longer match any call site; delete them if the "
            f"fix landed, re-key them if the code moved: {stale}"
        )

    def test_the_sweep_found_call_sites(self):
        """The parameterization must not silently collapse to nothing."""
        dropped, dynamic = _sweep_factory_call_sites()
        assert len(dropped) + len(dynamic) >= 20, (
            "expected the factory call-site sweep to reach dozens of sites; it "
            f"found {len(dropped)} dropped-param hits and {len(dynamic)} dynamic "
            "sites, which means the AST walk stopped seeing the tree"
        )

    def test_dynamic_call_sites_are_reported(self):
        """Non-fatal: list the call sites this guard structurally cannot check.

        A variable ``type=`` has no static declared-param set. These are not
        unguarded in practice -- the factories' strict raise fires the moment such
        a path is actually built -- but they are invisible to the check above, and
        an inventory that shrinks silently is how a guard's coverage rots.
        """
        _, dynamic = _sweep_factory_call_sites()
        if dynamic:
            warnings.warn(
                "factory call sites with a non-literal type (not statically "
                f"checkable, {len(dynamic)}): {dynamic}",
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Norm-forwarding guards: the same silent-drop defect, one indirection away.
# ---------------------------------------------------------------------------

#: Stored-attribute names that mean "this class holds a normalization epsilon".
#:
#: A plain ``"eps" in name`` substring test matches ``mask_annealing_steps``
#: (``layers/transformers/eomt_transformer.py``) -- st-EPS. The token form below
#: matches ``eps``, ``epsilon``, ``norm_eps``, ``layer_norm_eps``,
#: ``norm_epsilon`` and misses ``steps``/``keeps``/``epsilon_decay``-style
#: near-misses. Measured 2026-08-19: the substring form found 8 candidate classes,
#: this form finds 7, and the one it drops is the ``steps`` collision.
_EPS_ATTR_RE = re.compile(r"(?:^|_)(?:eps|epsilon)(?:$|_)")

#: The two ``TransformerLayer`` parameters that carry a caller's norm config into
#: every in-block norm. ``TransformerLayer._create_normalization_layer`` calls
#: ``create_normalization_layer(..., **custom_args)`` where ``custom_args`` is one
#: of these (default ``{}``), so omitting both silently pins every in-block norm
#: to the FACTORY's ``epsilon`` default regardless of the model's own knob.
_TRANSFORMER_NORM_ARG_KWARGS = ("attention_norm_args", "ffn_norm_args")

#: Name collisions for the DIRECT ``create_normalization_layer`` predicate,
#: cleared by manual read. Sibling of ``_NAME_COLLISIONS``, deliberately a
#: separate constant: that one is keyed by a registry factory's declared params,
#: this one by ``_accepted_params``' ``inspect.signature``-derived set, and
#: conflating the two would hide which predicate a waiver actually clears.
#:
#: The single entry is the ViT-family ``scale`` shape already catalogued in
#: ``_NAME_COLLISIONS``, reached here through a different factory:
#: ``SigLIPVisionTransformer`` stores ``self.scale = str(scale)`` (the variant
#: size, ``vit_siglip/model.py:357``) while ``LayerNormalization.__init__``
#: declares ``scale: bool``. Same two words, unrelated meanings.
#:
#: Key is ``(path relative to src/dl_techniques, class, normalization type,
#: param)`` -- never a line number, for the reason ``_SCHEDULED_FIXES`` gives.
_NORM_NAME_COLLISIONS = {
    ("models/vit_siglip/model.py", "SigLIPVisionTransformer", "layer_norm", "scale"),
}


def _stored_eps_attrs(stored: set) -> set:
    """The subset of ``stored`` that names a normalization epsilon."""
    return {attr for attr in stored if _EPS_ATTR_RE.search(attr.lower())}


def _sweep_transformer_layer_norm_args(roots=None, src_root=None):
    """Find ``TransformerLayer(...)`` sites that cannot pass their own epsilon on.

    Contract: returns ``(hits, n_constructions, n_candidates)``.

    * ``hits`` -- ``(relpath, lineno, class, sorted eps attrs, sorted missing
      kwargs)`` for every construction inside a class that stores an epsilon
      attribute and passes NEITHER ``attention_norm_args`` NOR ``ffn_norm_args``
      (a call passing one of the two is not flagged: the caller demonstrably knows
      the channel exists, and which of the two norms a knob belongs to is a design
      choice this predicate has no standing to make).
    * ``n_constructions`` -- every ``TransformerLayer(...)`` seen, for the vacuity
      assertion.
    * ``n_candidates`` -- those whose enclosing class stores an epsilon attribute.

    ``roots``/``src_root`` default to the real tree; they exist so the predicate
    can be pointed at a synthetic fixture and proven to fire. A ``**unpack`` clears
    a call, exactly as in ``_sweep_factory_call_sites``.
    """
    roots = _NORM_SWEEP_ROOTS if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    hits: List[tuple] = []
    n_constructions = 0
    n_candidates = 0

    for rel, cls in _iter_classes(roots, src_root):
        eps_attrs = _stored_eps_attrs(_init_stored_attrs(cls))
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            if _callee_name(node) != "TransformerLayer":
                continue
            n_constructions += 1
            if not eps_attrs:
                continue
            n_candidates += 1
            if any(kw.arg is None for kw in node.keywords):
                continue
            passed = {kw.arg for kw in node.keywords}
            missing = [k for k in _TRANSFORMER_NORM_ARG_KWARGS if k not in passed]
            if len(missing) == len(_TRANSFORMER_NORM_ARG_KWARGS):
                hits.append((rel, node.lineno, cls.name, sorted(eps_attrs), missing))
    return hits, n_constructions, n_candidates


def _sweep_norm_factory_call_sites(roots=None, src_root=None):
    """The registry-shaped predicate for ``create_normalization_layer``.

    Contract: returns ``(dropped, dynamic, n_literal)`` with the same element
    shapes as ``_sweep_factory_call_sites``' ``(dropped, dynamic)``, plus the
    literal-type call count for the vacuity assertion.

    ``norms/factory.py`` has no ``required_params``/``optional_params`` registry,
    so the declared-param ground truth comes from ``_accepted_params(type)``, which
    derives it from ``inspect.signature`` of ``_TYPE_TO_CLASS[type].__init__``. Do
    NOT hand-maintain a second accepted-param list here: that list has drifted
    twice already, which is why the factory replaced it with the signature.
    """
    from dl_techniques.layers.norms.factory import _TYPE_TO_CLASS, _accepted_params

    roots = _NORM_SWEEP_ROOTS if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    dropped: List[tuple] = []
    dynamic: List[str] = []
    n_literal = 0

    for rel, cls in _iter_classes(roots, src_root):
        stored = _init_stored_attrs(cls)
        for node in ast.walk(cls):
            if not isinstance(node, ast.Call):
                continue
            if _callee_name(node) != "create_normalization_layer":
                continue
            type_node = node.args[0] if node.args else None
            for kw in node.keywords:
                if kw.arg in ("normalization_type", "type"):
                    type_node = kw.value
            if not (
                isinstance(type_node, ast.Constant)
                and isinstance(type_node.value, str)
            ):
                dynamic.append(f"{rel}:{node.lineno}")
                continue
            norm_type = type_node.value
            if norm_type not in _TYPE_TO_CLASS:
                dynamic.append(f"{rel}:{node.lineno} (unknown type {norm_type!r})")
                continue
            n_literal += 1
            if any(kw.arg is None for kw in node.keywords):
                continue
            declared = _accepted_params(norm_type)
            passed = {kw.arg for kw in node.keywords}
            for param in sorted((declared - passed) & stored):
                dropped.append((rel, node.lineno, cls.name, norm_type, param))
    return dropped, dynamic, n_literal


#: A synthetic module, never imported -- only parsed. Both predicates are proven
#: to fire on it, and proven NOT to fire on its fixed twin below, so neither can
#: pass by finding nothing. Kept as source text rather than as a real defect in
#: the tree for the obvious reason: the tree is supposed to be clean.
_INJECTED_DEFECT_SRC = '''
class InjectedTransformerUser(keras.layers.Layer):
    def __init__(self, layer_norm_eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_eps = layer_norm_eps
        self.blocks = [
            TransformerLayer(hidden_size=8, num_heads=2, intermediate_size=16)
            for _ in range(2)
        ]


class InjectedNormUser(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.norm = create_normalization_layer('layer_norm', name='n')
'''

#: The same two classes with the omission repaired. A predicate that flags this
#: too is shape-matching, not measuring.
_INJECTED_FIXED_SRC = '''
class InjectedTransformerUser(keras.layers.Layer):
    def __init__(self, layer_norm_eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_eps = layer_norm_eps
        norm_args = {'epsilon': layer_norm_eps}
        self.blocks = [
            TransformerLayer(
                hidden_size=8,
                num_heads=2,
                intermediate_size=16,
                attention_norm_args=dict(norm_args),
                ffn_norm_args=dict(norm_args),
            )
            for _ in range(2)
        ]


class InjectedNormUser(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.norm = create_normalization_layer(
            'layer_norm', epsilon=epsilon, name='n'
        )
'''


def _write_fixture(tmp_path: Path, source: str) -> Tuple[tuple, Path]:
    """Write ``source`` as a fake package file; return ``(roots, src_root)``."""
    pkg = tmp_path / "models" / "injected"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "model.py").write_text(source)
    return (tmp_path / "models",), tmp_path


class TestTransformerLayerNormArgsAreForwarded:
    """A model's epsilon knob must have a route into its own in-block norms.

    This is the predicate that finds real defects. The direct-call one below is
    the shape the norm factory's own registry gap suggests, but MEASURED against
    this tree it finds zero genuine drops (44 literal-type calls, one hit, and
    that hit is the ``vit_siglip`` ``scale`` name collision). The defect family
    actually lives one indirection out: a model stores ``self.layer_norm_eps``,
    forwards it to its embedding norm, and then builds its encoder stack with
    ``TransformerLayer(...)`` passing neither ``attention_norm_args`` nor
    ``ffn_norm_args`` -- so all ``2*num_layers`` in-block norms run at the
    factory's default instead. Five models did exactly that until 2026-08-19
    (BERT/DistilBERT/ModernBERT at ``1e-6`` instead of their own ``1e-12``, a
    ~1e5x mismatch against the embedding norm they sit beside).

    Narrowings, each a place this can miss a real defect:

    * only classes storing an epsilon-named attribute. ``MobileClipTextEncoder``
      (``models/mobile_clip/components.py``) is invisible to this predicate for
      that reason -- its epsilon is a module-level ``REFERENCE_NORM_EPSILON``
      constant, not a stored attribute -- and it is a *correct* site, so the
      narrowing costs nothing there today. Dropping the clause is not the fix:
      without it every ``TransformerLayer(...)`` that legitimately wants the
      default becomes a failure (14 such sites at the time of writing);
    * only calls with no ``**unpack``;
    * only ``TransformerLayer``. ``TransformerDecoderLayer`` takes the same two
      dicts and is not swept -- no measured instance, and adding it unmeasured
      would be a guard nobody has seen fire;
    * nothing here reaches a knob that has no channel at all. ``GroupAttention``
      (``models/tree_transformer/components.py``) had no ``layer_norm_eps``
      parameter until it was given one as a PRODUCT fix; no call-site predicate
      can see a parameter that does not exist. It is still out of reach today for
      a second reason -- it calls ``create_normalization_layer`` with a dynamic
      type -- and is therefore reported by ``test_dynamic_call_sites_are_reported``
      rather than checked.
    """

    def test_no_eps_storing_class_omits_both_norm_arg_dicts(self):
        # DECISION plan-2026-08-19T070627-a616f581/D-008
        # This guard is written against the TransformerLayer INDIRECTION, not
        # against `create_normalization_layer(...)` call sites, because the
        # direct-call form guards ZERO: measured over models/ + layers/, the
        # signature-derived direct predicate returns exactly one hit and it is a
        # known name collision. The epsilon a model stores never reaches a
        # `create_normalization_layer` call it writes itself -- it reaches (or
        # fails to reach) one that `TransformerLayer._create_normalization_layer`
        # writes, through the `attention_norm_args`/`ffn_norm_args` dicts.
        # WHAT NOT TO DO: do not "simplify" this into the direct-call predicate
        # below, and do not delete this class as redundant with it -- they have
        # disjoint reach, and this is the one with a demonstrated defect family
        # (5 models, fixed the same day this shipped). See decisions.md D-008.
        hits, _, _ = _sweep_transformer_layer_norm_args()
        offenders = [
            f"{rel}:{line} {cls}: TransformerLayer(...) passes neither "
            f"{' nor '.join(missing)} while the class stores {eps} -- every "
            "in-block norm runs at the factory's epsilon default"
            for rel, line, cls, eps, missing in hits
        ]
        assert not offenders, (
            "a model's epsilon knob has no route into its own in-block norms. "
            "Build a `{'epsilon': self.<knob>}` dict and pass it as BOTH "
            "attention_norm_args and ffn_norm_args. Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_sweep_found_call_sites(self):
        """The AST walk must not silently collapse to nothing.

        A guard that reports zero offenders because it looked in the wrong place
        is indistinguishable from a clean tree -- which is the entire failure mode
        of writing this guard the day after the last offender was fixed. Floors
        are set well under the 2026-08-19 measurement (28 constructions, 7
        candidates) so ordinary churn does not trip them.
        """
        _, n_constructions, n_candidates = _sweep_transformer_layer_norm_args()
        assert n_constructions >= 20, (
            f"expected dozens of TransformerLayer(...) sites, found "
            f"{n_constructions}: the AST walk stopped seeing the tree"
        )
        assert n_candidates >= 5, (
            f"expected several eps-storing classes to build TransformerLayers, "
            f"found {n_candidates}: the stored-attribute predicate stopped matching"
        )

    def test_predicate_fires_on_an_injected_defect(self, tmp_path):
        """Dead-component probe: the predicate must go RED on a real omission."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_DEFECT_SRC)
        hits, n_constructions, n_candidates = _sweep_transformer_layer_norm_args(
            roots, src_root
        )
        assert n_constructions == 1 and n_candidates == 1
        assert len(hits) == 1, hits
        rel, _, cls, eps, missing = hits[0]
        assert cls == "InjectedTransformerUser"
        assert eps == ["layer_norm_eps"]
        assert missing == ["attention_norm_args", "ffn_norm_args"]

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once the same site forwards the dicts."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_FIXED_SRC)
        hits, _, n_candidates = _sweep_transformer_layer_norm_args(roots, src_root)
        assert n_candidates == 1, "the fixture must still be reached"
        assert hits == [], hits


class TestNormalizationKnobsAreForwarded:
    """``create_normalization_layer`` call sites, checked against the signature.

    The registry-shaped sibling of ``TestFactoryKnobsAreForwarded``, for the one
    factory that has no registry: the declared-param set comes from
    ``norms/factory.py::_accepted_params``, i.e. from ``inspect.signature`` of the
    class each type instantiates.

    Its measured yield on this tree is zero genuine drops, and that is expected --
    it is here for the regression direction, not because it found anything. The
    reason is structural and worth stating so nobody "fixes" it by loosening the
    name match: an epsilon knob is almost always stored under a *different* name
    than the parameter that carries it (``self.norm_eps`` -> ``epsilon=``,
    ``self.norm_epsilon`` -> ``epsilon=``), so the stored-attribute clause that
    makes a hit demonstrable is also what makes this predicate quiet. Matching on
    value rather than name would need dataflow, not AST shape.

    Same four narrowings as ``TestFactoryKnobsAreForwarded``, and the first one
    bites much harder here: 138 of ~167 call sites pass a dynamic
    ``normalization_type`` (it is nearly always ``self.normalization_type``), so
    this predicate sees under a fifth of the tree's calls.
    """

    def test_no_declared_and_stored_param_is_dropped(self):
        dropped, _, _ = _sweep_norm_factory_call_sites()
        offenders = [
            f"{rel}:{line} {cls}: create_normalization_layer({typ!r}) drops "
            f"{param!r} (self.{param} is stored in __init__)"
            for rel, line, cls, typ, param in dropped
            if (rel, cls, typ, param) not in _NORM_NAME_COLLISIONS
        ]
        assert not offenders, (
            "a create_normalization_layer call site drops a parameter its "
            "enclosing class stores. Forward it, or -- if the names merely "
            "collide -- add it to _NORM_NAME_COLLISIONS with the read that "
            "clears it. Found:\n  " + "\n  ".join(offenders)
        )

    def test_waivers_still_match_a_real_site(self):
        """A waiver matching nothing is a waiver hiding nothing (see the sibling)."""
        dropped, _, _ = _sweep_norm_factory_call_sites()
        live = {(rel, cls, typ, param) for rel, _, cls, typ, param in dropped}
        stale = sorted(_NORM_NAME_COLLISIONS - live)
        assert not stale, (
            "norm waiver entries no longer match any call site; delete them if "
            f"the collision is gone, re-key them if the code moved: {stale}"
        )

    def test_the_sweep_found_call_sites(self):
        """The walk must reach a plausible share of the tree's call sites."""
        dropped, dynamic, n_literal = _sweep_norm_factory_call_sites()
        assert n_literal >= 15, (
            f"expected dozens of literal-type create_normalization_layer calls, "
            f"found {n_literal} (2026-08-19: 29)"
        )
        assert len(dynamic) >= 50, (
            f"expected the dynamic-type inventory to stay large, found "
            f"{len(dynamic)} (2026-08-19: 138); a collapse here means the AST "
            "walk stopped seeing the tree, not that the tree got more static"
        )
        assert len(dropped) + n_literal >= 15

    def test_predicate_fires_on_an_injected_defect(self, tmp_path):
        """Dead-component probe: a stored, declared, unpassed param must be seen."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_DEFECT_SRC)
        dropped, _, n_literal = _sweep_norm_factory_call_sites(roots, src_root)
        assert n_literal == 1
        assert len(dropped) == 1, dropped
        _, _, cls, typ, param = dropped[0]
        assert (cls, typ, param) == ("InjectedNormUser", "layer_norm", "epsilon")

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once the same call forwards the parameter."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_FIXED_SRC)
        dropped, _, n_literal = _sweep_norm_factory_call_sites(roots, src_root)
        assert n_literal == 1, "the fixture must still be reached"
        assert dropped == [], dropped

    def test_dynamic_call_sites_are_reported(self):
        """Non-fatal inventory of what this predicate structurally cannot check.

        ``GroupAttention``'s own norm lives in here, not in the checked set.
        """
        _, dynamic, _ = _sweep_norm_factory_call_sites()
        if dynamic:
            warnings.warn(
                "create_normalization_layer call sites with a non-literal type "
                f"(not statically checkable, {len(dynamic)}): {dynamic}",
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# MODEL_VARIANTS: the house rule's variant registry, previously unguarded.
# ---------------------------------------------------------------------------

#: Legacy spellings of the variant registry, per ``models/CLAUDE.md`` Axis 2:
#: "Packages that predate this spec also use ``VARIANT_CONFIGS``, ``NAM_VARIANTS``,
#: ``NTM_VARIANTS`` or ``MCI_VARIANTS`` for that same role; where one of those is
#: the package's *only* variant table, add ``MODEL_VARIANTS`` as a class-level
#: alias to the same dict."
#:
#: ``SCALE_CONFIGS`` is deliberately NOT here: the same section says in bold that
#: it "is NOT a stale spelling of MODEL_VARIANTS, and the two must not be merged
#: where both appear" -- it is the architecture table, MODEL_VARIANTS is the
#: public-name registry, and ``beit``/``vit``/``energy_transformer`` carry both.
_LEGACY_VARIANT_TABLE_RE = re.compile(r"^(?:[A-Z0-9]+_VARIANTS|VARIANT_CONFIGS)$")

#: Deliberate exceptions, keyed by ``(relpath, symbol, kind)`` -- never by line
#: number, for the reason ``_SCHEDULED_FIXES`` gives. Each carries the read that
#: clears it, and ``test_waivers_still_match_a_real_site`` fails if one stops
#: matching, so a waiver cannot outlive the thing it waives.
_MODEL_VARIANTS_WAIVERS = {
    # SD3VAE is not a keras.Model: it is a plain Python holder pairing an
    # ideogram4 ``AutoEncoder`` with SD3's latent-norm helpers (its own docstring
    # says so). Its ``from_variant`` delegates to ``create_sd3_vae`` ->
    # ``config.get_sd3_config(variant)``, and the sd3_mmdit family's variant
    # registry has ONE home there (``config.PRESETS``, shared by the transformer,
    # the VAE and the pipeline). Restating it as ``SD3VAE.MODEL_VARIANTS`` would
    # create the second home the house rule exists to prevent. models/CLAUDE.md
    # § "When the shape does not apply": multi-model families apply the shape per
    # inner architecture, and the inner architecture here is ``AutoEncoder``.
    (
        "models/sd3_mmdit/vae.py",
        "SD3VAE",
        "from_variant-without-table",
    ),
}


def _module_level_names(tree: ast.Module) -> set:
    """Names assigned at module scope (``Assign`` and ``AnnAssign`` alike)."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _class_body_names(cls: ast.ClassDef) -> set:
    """Names assigned in a class BODY (class attributes), not in its methods."""
    names = set()
    for node in cls.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _is_variant_table_literal(node: ast.expr) -> bool:
    """A ``{'name': {...}, ...}`` literal: string keys, dict values, non-empty."""
    if not isinstance(node, ast.Dict) or not node.keys:
        return False
    if not all(
        isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.keys
    ):
        return False
    return all(isinstance(v, ast.Dict) for v in node.values)


def _iter_modules(roots, src_root):
    """Yield ``(relpath, ast.Module)`` for every parseable file under ``roots``.

    Contract: same arguments and same silent-skip-on-SyntaxError policy as
    ``_iter_classes``; this one keeps the MODULE node because the MODEL_VARIANTS
    rule is satisfiable at module scope (``beit``/``energy_transformer`` define
    theirs there, deliberately) and because one of its three predicates walks
    module-level FUNCTIONS, which ``_iter_classes`` cannot reach.
    """
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            rel = path.relative_to(src_root).as_posix()
            try:
                yield rel, ast.parse(path.read_text())
            except SyntaxError:  # covered by the import tests
                continue


def _sweep_model_variants(roots=None, src_root=None):
    """Find named-variant registries that are not reachable as ``MODEL_VARIANTS``.

    Contract: returns ``(hits, counts)``.

    * ``hits`` -- ``(relpath, lineno, symbol, kind, detail)`` where ``kind`` is one
      of three predicates, each transcribed from a different sentence of
      ``models/CLAUDE.md`` § Axis 2:

      - ``"from_variant-without-table"``: *"``from_variant(cls, variant, ...)``
        looks the name up in ``MODEL_VARIANTS``"*. A class defining
        ``from_variant`` whose module and class body both lack ``MODEL_VARIANTS``
        **and** whose ``from_variant`` body never reads the name at all. The last
        clause is what clears ``DINOv2``, whose ``from_variant`` deliberately
        reads ``DINOv2VisionTransformer.MODEL_VARIANTS`` -- one table, one home,
        exactly what the rule wants.
      - ``"legacy-table-without-alias"``: *"where one of those is the package's
        only variant table, add ``MODEL_VARIANTS`` as a class-level alias"*.
      - ``"function-local-table"``: a function taking a ``variant`` parameter that
        builds its variant table as a LOCAL dict literal. Nothing external can
        reach it -- ``getattr(cls, "MODEL_VARIANTS")`` raises, which is the exact
        failure mode that got ``fastvit`` fixed on 2026-08-19.

    * ``counts`` -- vacuity denominators: ``n_classes``, ``n_from_variant``,
      ``n_legacy_tables``, ``n_variant_functions``.

    ``roots``/``src_root`` default to the real tree; they exist so the predicates
    can be pointed at a synthetic fixture and proven to fire, exactly as in
    ``_sweep_transformer_layer_norm_args``.
    """
    roots = (MODELS_DIR,) if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    hits: List[tuple] = []
    counts = dict(
        n_classes=0, n_from_variant=0, n_legacy_tables=0, n_variant_functions=0
    )

    for rel, tree in _iter_modules(roots, src_root):
        module_names = _module_level_names(tree)
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            counts["n_classes"] += 1
            body_names = _class_body_names(cls)
            has_alias = "MODEL_VARIANTS" in body_names or "MODEL_VARIANTS" in module_names
            legacy = sorted(
                n
                for n in body_names
                if _LEGACY_VARIANT_TABLE_RE.match(n) and n != "MODEL_VARIANTS"
            )
            from_variant = next(
                (
                    n
                    for n in cls.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == "from_variant"
                ),
                None,
            )
            if from_variant is not None:
                counts["n_from_variant"] += 1
                reads_table = any(
                    (isinstance(n, ast.Attribute) and n.attr == "MODEL_VARIANTS")
                    or (isinstance(n, ast.Name) and n.id == "MODEL_VARIANTS")
                    for n in ast.walk(from_variant)
                )
                if not (has_alias or reads_table):
                    hits.append(
                        (
                            rel,
                            from_variant.lineno,
                            cls.name,
                            "from_variant-without-table",
                            "from_variant resolves no MODEL_VARIANTS table",
                        )
                    )
            if legacy:
                counts["n_legacy_tables"] += 1
                if "MODEL_VARIANTS" not in body_names:
                    hits.append(
                        (
                            rel,
                            cls.lineno,
                            cls.name,
                            "legacy-table-without-alias",
                            f"only table is {'/'.join(legacy)}",
                        )
                    )

        for fn in [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            args = fn.args
            names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
            if "variant" not in names:
                continue
            counts["n_variant_functions"] += 1
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and _is_variant_table_literal(node.value):
                    local = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if not local:
                        continue
                    hits.append(
                        (
                            rel,
                            node.lineno,
                            fn.name,
                            "function-local-table",
                            f"local `{local[0]}` holds {len(node.value.keys)} named "
                            "variants that nothing can introspect",
                        )
                    )
    return hits, counts


#: Synthetic package-shaped source, never imported -- only parsed. All three
#: predicates must fire on it and be silent on its twin below.
_INJECTED_VARIANTS_DEFECT_SRC = '''
class InjectedFromVariantUser(keras.Model):
    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_variant(cls, variant, **kwargs):
        if variant == "small":
            return cls(width=8, **kwargs)
        return cls(width=16, **kwargs)


class InjectedLegacyTableUser(keras.Model):
    NAM_VARIANTS = {"small": {"width": 8}, "large": {"width": 16}}

    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)


def create_injected(variant="small", **kwargs):
    configs = {"small": {"width": 8}, "large": {"width": 16}}
    return InjectedFromVariantUser(**configs[variant], **kwargs)
'''

#: The same three sites, repaired the way the house rule asks.
_INJECTED_VARIANTS_FIXED_SRC = '''
class InjectedFromVariantUser(keras.Model):
    MODEL_VARIANTS = {"small": {"width": 8}, "large": {"width": 16}}

    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_variant(cls, variant, **kwargs):
        return cls(**cls.MODEL_VARIANTS[variant], **kwargs)


class InjectedLegacyTableUser(keras.Model):
    NAM_VARIANTS = {"small": {"width": 8}, "large": {"width": 16}}
    MODEL_VARIANTS = NAM_VARIANTS

    def __init__(self, width=8, **kwargs):
        super().__init__(**kwargs)


def create_injected(variant="small", **kwargs):
    return InjectedFromVariantUser.from_variant(variant, **kwargs)
'''


class TestModelVariantsArePresent:
    """Named variants must be reachable as ``MODEL_VARIANTS``, not just callable.

    ``models/CLAUDE.md`` § Axis 2 makes ``MODEL_VARIANTS`` the canonical name for
    the registry of publicly named variants, tells packages carrying a legacy
    spelling to add it as a class-level alias, and defines ``from_variant`` as the
    method that "looks the name up in ``MODEL_VARIANTS``". Until this class
    shipped **nothing enforced any of that** -- the convention was documented and
    unguarded, which is how ``fastvit`` reached 2026-08-19 with
    ``getattr(FastVitImageEncoder, "MODEL_VARIANTS")`` raising ``AttributeError``
    while this very file asserted the opposite in prose.

    Narrowings, each a place this can miss a real defect:

    * a package whose named sizes are exposed under a DIFFERENT parameter name is
      invisible here. ``depth_anything`` takes ``encoder_type='vit_s'|'vit_b'|
      'vit_l'`` -- three genuine published sizes -- and no predicate below fires
      on it, because none of them can tell that knob apart from any other
      validated string argument. It was given a ``MODEL_VARIANTS`` table anyway
      (see decisions.md D-009); the guard simply cannot be what keeps it;
    * the local-table predicate needs a ``variant``-named parameter AND a literal
      ``{str: dict}`` assignment. A table built by a loop, a comprehension, or
      ``dict(...)`` is not seen;
    * ``SCALE_CONFIGS`` is out of scope by explicit instruction of the rule.
    """

    def test_every_from_variant_class_resolves_a_model_variants_table(self):
        # DECISION plan-2026-08-19T070627-a616f581/D-009
        # The trigger for this guard is EVIDENCE OF NAMED VARIANTS (a
        # `from_variant`, a legacy table, or a hidden local table) -- NOT "is a
        # keras.Model". WHAT NOT TO DO: do not "strengthen" this into "every
        # model class must declare MODEL_VARIANTS". models/CLAUDE.md
        # § "When the shape does not apply" says in terms: "Do not invent a
        # MODEL_VARIANTS table to satisfy the template" -- a package with no
        # genuine named variants is compliant WITHOUT one, and the strengthened
        # form would demand fabricated tables from ~20 packages. See D-009.
        hits, _ = _sweep_model_variants()
        offenders = [
            f"{rel}:{line} {sym}: {detail}"
            for rel, line, sym, kind, detail in hits
            if kind == "from_variant-without-table"
            and (rel, sym, kind) not in _MODEL_VARIANTS_WAIVERS
        ]
        assert not offenders, (
            "a from_variant classmethod resolves no MODEL_VARIANTS table, so its "
            "variants exist only inside its own body. Hoist them to a class-level "
            "MODEL_VARIANTS dict (models/CLAUDE.md Axis 2). Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_every_legacy_variant_table_has_a_model_variants_alias(self):
        hits, _ = _sweep_model_variants()
        offenders = [
            f"{rel}:{line} {sym}: {detail}"
            for rel, line, sym, kind, detail in hits
            if kind == "legacy-table-without-alias"
            and (rel, sym, kind) not in _MODEL_VARIANTS_WAIVERS
        ]
        assert not offenders, (
            "a class's only variant table uses a legacy spelling with no "
            "MODEL_VARIANTS alias. Add `MODEL_VARIANTS = <existing name>` in the "
            "class body -- an ALIAS, never a rename: src/train/ and the test "
            "suites reference the old spelling. Found:\n  " + "\n  ".join(offenders)
        )

    def test_no_variant_table_hides_inside_a_factory_body(self):
        hits, _ = _sweep_model_variants()
        offenders = [
            f"{rel}:{line} {sym}(): {detail}"
            for rel, line, sym, kind, detail in hits
            if kind == "function-local-table"
            and (rel, sym, kind) not in _MODEL_VARIANTS_WAIVERS
        ]
        assert not offenders, (
            "a factory's variant table is a local variable, so no caller can "
            "enumerate the variants it accepts. Hoist it to the model class as "
            "MODEL_VARIANTS and read it from the factory. Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_sweep_found_variant_sites(self):
        """The AST walk must not silently collapse to nothing.

        Floors are set well under the 2026-08-19 measurement (252 classes, 64
        ``from_variant`` classes, 2 legacy tables, 150 functions taking
        ``variant``) so ordinary churn does not trip them. The legacy-table
        denominator is deliberately NOT floored above 1: there are only two such
        classes left in the tree, and fixing them by renaming is forbidden, so it
        should stay at two.
        """
        _, counts = _sweep_model_variants()
        assert counts["n_classes"] >= 150, counts
        assert counts["n_from_variant"] >= 40, counts
        assert counts["n_variant_functions"] >= 100, counts
        assert counts["n_legacy_tables"] >= 2, counts

    def test_predicate_fires_on_an_injected_defect(self, tmp_path):
        """Dead-component probe: all three predicates must go RED on a real gap."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_VARIANTS_DEFECT_SRC)
        hits, counts = _sweep_model_variants(roots, src_root)
        assert counts["n_from_variant"] == 1 and counts["n_legacy_tables"] == 1
        # 2, not 1: `from_variant(cls, variant, **kwargs)` takes a `variant`
        # parameter too, so it is counted alongside `create_injected`.
        assert counts["n_variant_functions"] == 2
        by_kind = {kind: (sym, detail) for _, _, sym, kind, detail in hits}
        assert set(by_kind) == {
            "from_variant-without-table",
            "legacy-table-without-alias",
            "function-local-table",
        }, hits
        assert by_kind["from_variant-without-table"][0] == "InjectedFromVariantUser"
        assert by_kind["legacy-table-without-alias"][0] == "InjectedLegacyTableUser"
        assert by_kind["function-local-table"][0] == "create_injected"

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once the same three sites are repaired."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_VARIANTS_FIXED_SRC)
        hits, counts = _sweep_model_variants(roots, src_root)
        assert counts["n_from_variant"] == 1, "the fixture must still be reached"
        assert counts["n_legacy_tables"] == 1, "the fixture must still be reached"
        assert counts["n_variant_functions"] == 2, "the fixture must still be reached"
        assert hits == [], hits

    @pytest.mark.parametrize(
        "module_path,class_name,expected_keys",
        [
            ("dl_techniques.models.SAM.SAM1.model", "SAM", ["vit_b", "vit_h", "vit_l"]),
            ("dl_techniques.models.kan.model", "KAN",
             ["large", "medium", "micro", "small", "xlarge"]),
            ("dl_techniques.models.ntm.model", "NTMModel", ["base", "large", "tiny"]),
            ("dl_techniques.models.nano_vlm.model", "NanoVLM",
             ["base", "large", "mini"]),
            ("dl_techniques.models.nano_vlm_world_model.model", "ScoreBasedNanoVLM",
             ["base", "large", "mini"]),
            ("dl_techniques.models.pft_sr.model", "PFTSR",
             ["base", "large", "light"]),
            ("dl_techniques.models.depth_anything.model", "DepthAnything",
             ["vit_b", "vit_l", "vit_s"]),
        ],
    )
    def test_repaired_packages_expose_their_variants_at_runtime(
        self, module_path, class_name, expected_keys
    ):
        """The seven classes repaired on 2026-08-19, pinned by RESOLVED value.

        The three sweeps above are static: they prove a class BODY assigns the
        name. This one proves ``getattr(cls, "MODEL_VARIANTS")`` actually
        resolves on the imported class and enumerates the variants the package
        already supported -- which is the property that was missing (the fastvit
        report was an ``AttributeError`` at runtime, not an AST observation), and
        the property an alias or a class-attribute hoist could get wrong.

        The key lists are DERIVED, not chosen: every one of them is the key set
        the package's own factory or ``from_variant`` already accepted before the
        repair. No variant was invented.
        """
        cls = getattr(importlib.import_module(module_path), class_name)
        table = getattr(cls, "MODEL_VARIANTS", None)
        assert isinstance(table, dict), (
            f"{class_name}.MODEL_VARIANTS must resolve to a dict, got {table!r}"
        )
        assert sorted(table) == sorted(expected_keys), (
            f"{class_name}.MODEL_VARIANTS keys changed: {sorted(table)} != "
            f"{sorted(expected_keys)}"
        )
        assert all(isinstance(v, dict) for v in table.values()), table

    def test_legacy_aliases_are_the_same_object_not_a_copy(self):
        """``MODEL_VARIANTS = <legacy>`` must ALIAS, so one edit reaches both.

        A copy would satisfy every static predicate above and still let the two
        spellings drift -- which is the whole reason the house rule says "alias to
        the same dict" and "prefer an alias over renaming in place".
        """
        from dl_techniques.models.kan.model import KAN
        from dl_techniques.models.ntm.model import NTMModel

        assert KAN.MODEL_VARIANTS is KAN.VARIANT_CONFIGS
        assert NTMModel.MODEL_VARIANTS is NTMModel.NTM_VARIANTS

    def test_waivers_still_match_a_real_site(self):
        """A waiver matching nothing is a waiver hiding nothing (see the siblings)."""
        hits, _ = _sweep_model_variants()
        live = {(rel, sym, kind) for rel, _, sym, kind, _ in hits}
        stale = sorted(_MODEL_VARIANTS_WAIVERS - live)
        assert not stale, (
            "MODEL_VARIANTS waiver entries no longer match any site; delete them "
            f"if the exception is gone, re-key them if the code moved: {stale}"
        )



# ---------------------------------------------------------------------------
# Registration and export guards (plan-2026-08-19T163559-499b6f0e step 6).
#
# Three families, all FREEZE-FORWARD: the measured offender count is 0 for every
# one of them, so none can be validated by "it finds the known defects" -- each
# ships with an injected-defect fixture proving it goes RED and a fixed twin
# proving it does not fire on the repair.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Memo for the sweeps below when they run over their DEFAULT roots.
#:
#: The registry-key sweep parses ``src/`` plus the whole of ``tests/`` -- ~2,900
#: files -- and five tests call it. Measured 2026-08-20: without this memo the
#: file's runtime went 40s -> 94s; with it, 44s. The memo is keyed by function
#: name and is bypassed entirely when explicit roots are passed, so every injected
#: fixture below still runs a real sweep and cannot be served a cached answer.
_SWEEP_MEMO: dict = {}


def _memo_default_roots(fn):
    """Cache ``fn()`` when, and only when, it is called with no explicit roots."""

    @functools.wraps(fn)
    def wrapper(roots=None, src_root=None):
        if roots is not None or src_root is not None:
            return fn(roots, src_root)
        if fn.__name__ not in _SWEEP_MEMO:
            _SWEEP_MEMO[fn.__name__] = fn(None, None)
        return _SWEEP_MEMO[fn.__name__]

    return wrapper


#: Root set of the registry-key sweep: **every registering path in the repo**.
#:
#: NOT ``models/`` (220 decorated classes) and NOT ``src/dl_techniques/`` alone
#: (726). A bare ``@register_keras_serializable()`` registers under the literal
#: key ``Custom><ClassName>``, which is MODULE-INDEPENDENT -- measured on Keras
#: 3.8.0, see decisions.md D-002 -- so the namespace those classes compete in is
#: flat and repo-global. Both narrower scopings are structurally blind to the
#: family this guard exists for: each of the four duplicate class-NAME near
#: misses in the tree (``ConvBlock``, ``MLPBlock``, ``Downsample``, ``Upsample``)
#: has at least one leg outside ``models/``, and 20+ classes under ``src/train/``
#: register into the same flat namespace as the library's.
_REGISTRY_KEY_ROOTS = (
    REPO_ROOT / "src" / "dl_techniques",
    REPO_ROOT / "src" / "train",
    REPO_ROOT / "src" / "applications",
    REPO_ROOT / "tests",
)


def _decorator_name(node: ast.expr) -> str:
    """The bare name a decorator expression names, called or not.

    Contract: takes any decorator node; returns the last name component for
    ``@f``, ``@mod.f``, ``@f(...)`` and ``@mod.f(...)`` alike, or ``""`` for
    anything else. The call form delegates to ``_callee_name`` -- do not
    re-implement that branch here.
    """
    if isinstance(node, ast.Call):
        return _callee_name(node)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


@_memo_default_roots
def _sweep_registry_keys(roots=None, src_root=None):
    """Every ``@register_keras_serializable`` class, by the key it claims.

    Contract: returns ``(keys, counts)``. ``keys`` maps ``"<package>><name>"`` to
    the list of ``"relpath:lineno"`` claiming it; ``counts`` carries the sweep's
    population and its three measured-empty blind spots.

    The key is computed the way Keras computes it: ``package`` defaults to the
    literal string ``"Custom"`` and ``name`` to the class's own name, and both
    are read from POSITIONAL as well as keyword decorator arguments (the Keras
    signature is ``register_keras_serializable(package="Custom", name=None)``, so
    ``@register_keras_serializable("dl_techniques")`` is a legal spelling this
    sweep must not miss).
    """
    roots = _REGISTRY_KEY_ROOTS if roots is None else roots
    src_root = REPO_ROOT if src_root is None else src_root
    keys: dict = {}
    counts = {
        "n_decorated": 0,
        "n_bare": 0,
        "n_package": 0,
        "n_named": 0,
        "n_aliased_import": 0,
        "n_function_decorated": 0,
        "n_dynamic_registration": 0,
    }
    for rel, tree in _iter_modules(roots, src_root):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if (
                        alias.name.rsplit(".", 1)[-1] == "register_keras_serializable"
                        and alias.asname
                        and alias.asname != "register_keras_serializable"
                    ):
                        counts["n_aliased_import"] += 1
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(
                    _decorator_name(d) == "register_keras_serializable"
                    for d in node.decorator_list
                ):
                    counts["n_function_decorated"] += 1
                continue
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
                if _callee_name(node.value) == "get_custom_objects":
                    counts["n_dynamic_registration"] += 1
                continue
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                if _decorator_name(dec) != "register_keras_serializable":
                    continue
                counts["n_decorated"] += 1
                package = name = None
                if isinstance(dec, ast.Call):
                    for i, arg in enumerate(dec.args):
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if i == 0:
                                package = arg.value
                            elif i == 1:
                                name = arg.value
                    for kw in dec.keywords:
                        if not (
                            isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            continue
                        if kw.arg == "package":
                            package = kw.value.value
                        elif kw.arg == "name":
                            name = kw.value.value
                counts["n_package" if package else "n_bare"] += 1
                if name:
                    counts["n_named"] += 1
                key = f"{package or 'Custom'}>{name or node.name}"
                keys.setdefault(key, []).append(f"{rel}:{node.lineno}")
    return keys, counts


#: Two classes claiming one registry key, in the exact shape the tree's four
#: duplicate class-NAME near-misses would take if one of them ever lost its
#: prefix. Parsed, never imported.
_INJECTED_REGISTRY_DEFECT_SRC = '''
@keras.saving.register_keras_serializable()
class Downsample(keras.layers.Layer):
    pass


@keras.saving.register_keras_serializable()
class Downsample(keras.layers.Layer):  # noqa: F811 -- a second FILE in reality
    pass
'''

#: The same two classes, one of them prefixed per R-011. A predicate that flags
#: this too is matching on the class name, not on the registry key.
_INJECTED_REGISTRY_FIXED_SRC = '''
@keras.saving.register_keras_serializable()
class Downsample(keras.layers.Layer):
    pass


@keras.saving.register_keras_serializable()
class ConvUNextDownsample(keras.layers.Layer):
    pass
'''


class TestRegistryKeysDoNotCollide:
    """No two registered classes may claim the same Keras registry key.

    A bare ``@register_keras_serializable()`` claims ``Custom><ClassName>``, and
    that key is module-independent: two classes with the same name in two
    different files silently occupy one slot, and whichever imports LAST wins
    every ``.keras`` deserialization of both. Measured 2026-08-20 over the whole
    repo: **761 decorated classes, 761 distinct keys, 0 collisions** -- so this is
    a freeze, and the only evidence it works is the injected pair below.

    Adding ``package=`` to a bare decorator is NOT a safe repair: it moves the key
    (measured on Keras 3.8.0 -- ``Custom>ProbeBare`` vs ``dl_techniques>ProbePkg``)
    and every ``.keras`` config storing the old ``registered_name`` stops
    resolving. The repair for a collision is to PREFIX the class name (R-011).

    Three blind spots, each measured EMPTY today and each therefore a shape this
    guard would silently skip if it ever appeared. They are reported non-fatally
    by ``test_registration_blind_spots_are_still_empty`` rather than left implicit,
    because a ``>=`` population floor cannot expose any of them:

    * an aliased decorator import (``register_keras_serializable as reg``) -- the
      predicate matches the literal name, so ``@reg()`` is invisible: **0 today**;
    * the decorator on a ``FunctionDef`` rather than a ``ClassDef``. Keras allows
      it, and such a function competes for the same key namespace: **0 today**;
    * a dynamic ``get_custom_objects()["Custom>X"] = X``, which registers with no
      decorator at all: **0 today**.
    """

    def test_no_two_registered_classes_claim_the_same_key(self):
        # DECISION plan-2026-08-19T163559-499b6f0e/D-012
        # The subject set is the WHOLE REPO -- src/dl_techniques + src/train +
        # src/applications + tests -- and that is load-bearing, not incidental.
        # WHAT NOT TO DO: do not narrow this to `models/` (220 classes) or to
        # `src/dl_techniques/` (726). The key a bare decorator claims is
        # module-independent, so `src/train/` and `tests/` compete in the SAME
        # flat `Custom>` namespace; 20+ src/train classes register there today,
        # and every one of the four duplicate class-name near-misses in the tree
        # has a leg outside `models/`. A narrower subject set is structurally
        # blind to the family this guard exists for. Also do NOT "fix" a future
        # collision by adding `package=`: that moves the key and breaks every
        # existing checkpoint that stored the old registered_name (D-002).
        # See decisions.md D-012.
        keys, _ = _sweep_registry_keys()
        offenders = [
            f"{key} claimed by {sites}"
            for key, sites in sorted(keys.items())
            if len(sites) > 1
        ]
        assert not offenders, (
            "two @register_keras_serializable classes claim one registry key; "
            "whichever module imports last wins every deserialization of both. "
            "PREFIX the class name (models/CLAUDE.md R-011) -- do NOT add "
            "`package=`, which moves the key and invalidates existing "
            ".keras checkpoints. Found:\n  " + "\n  ".join(offenders)
        )

    def test_the_sweep_found_registered_classes(self):
        """Anti-vacuity floor, DERIVED rather than set just under the population.

        Measured 2026-08-20: 761 decorated classes (725 bare, 36 with
        ``package=``). The floor is ``int(0.8 * 761) == 608``: a fifth of the
        repo's registered classes may disappear before this guard is allowed to
        call itself alive. A floor a few percent under 761 would trip on a
        legitimate refactor -- Phase 4 of this plan edits these very packages --
        and would say nothing more about whether the walk still sees the tree.
        """
        keys, counts = _sweep_registry_keys()
        assert counts["n_decorated"] >= 608, (
            f"expected ~761 @register_keras_serializable classes repo-wide, found "
            f"{counts['n_decorated']}: the AST walk stopped seeing the tree ({counts})"
        )
        assert len(keys) >= 608, counts
        assert counts["n_bare"] >= counts["n_package"], (
            "the bare decorator is this repo's convention (725 vs 36 measured); "
            f"an inversion means the sweep is reading something else: {counts}"
        )

    def test_registration_blind_spots_are_still_empty(self):
        """Non-fatal: the three registration shapes this predicate cannot key.

        All three measured 0 on 2026-08-20. They are reported rather than asserted
        at zero because appearing is not itself a defect -- it is a hole in THIS
        guard, and the inventory is what stops the hole opening silently.
        """
        _, counts = _sweep_registry_keys()
        blind = {
            k: counts[k]
            for k in (
                "n_aliased_import",
                "n_function_decorated",
                "n_dynamic_registration",
            )
            if counts[k]
        }
        if blind:
            warnings.warn(
                "registration shapes the registry-key guard cannot see have "
                f"appeared: {blind}. Extend _sweep_registry_keys before trusting "
                "its zero.",
                UserWarning,
                stacklevel=2,
            )

    def test_predicate_fires_on_an_injected_collision(self, tmp_path):
        """Dead-component probe: the predicate must go RED on a real collision."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_REGISTRY_DEFECT_SRC)
        keys, counts = _sweep_registry_keys(roots, src_root)
        assert counts["n_decorated"] == 2, counts
        collisions = {k: v for k, v in keys.items() if len(v) > 1}
        assert list(collisions) == ["Custom>Downsample"], keys
        assert len(collisions["Custom>Downsample"]) == 2, collisions

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once one of the two is prefixed."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_REGISTRY_FIXED_SRC)
        keys, counts = _sweep_registry_keys(roots, src_root)
        assert counts["n_decorated"] == 2, "the fixture must still be reached"
        assert sorted(keys) == ["Custom>ConvUNextDownsample", "Custom>Downsample"]
        assert not [k for k, v in keys.items() if len(v) > 1], keys


# ---------------------------------------------------------------------------
# R-013: an __init__.py may not bind a name over its own subpackage.
# ---------------------------------------------------------------------------


@_memo_default_roots
def _sweep_init_subpackage_shadows(roots=None, src_root=None):
    """Package ``__init__.py`` files binding a name over their own subpackage.

    Contract: returns ``(hits, counts)`` where ``hits`` is
    ``[(relpath, lineno, name)]``.

    ``from . import sub`` is deliberately NOT a hit: that binds the subpackage
    itself, which is the whole point of the form. Every other binding of the same
    name -- an assignment, a ``def``/``class``, ``from .other import sub``, or
    ``from .sub import sub`` -- puts a non-module object where ``import
    <pkg>.<sub>`` expects the submodule, and the break surfaces only at collection
    time, in a traceback that names the importer rather than the shadow.
    """
    roots = (SRC_ROOT,) if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    hits = []
    counts = {"n_packages": 0, "n_subpackage_relations": 0}
    for root in roots:
        for init in sorted(Path(root).rglob("__init__.py")):
            counts["n_packages"] += 1
            subs = {
                p.name
                for p in init.parent.iterdir()
                if p.is_dir()
                and p.name != "__pycache__"
                and (p / "__init__.py").exists()
            }
            if not subs:
                continue
            counts["n_subpackage_relations"] += len(subs)
            try:
                tree = ast.parse(init.read_text())
            except SyntaxError:  # covered by the import tests
                continue
            rel = init.relative_to(src_root).as_posix()
            bound: dict = {}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            bound.setdefault(t.id, node.lineno)
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    bound.setdefault(node.target.id, node.lineno)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.setdefault(node.name, node.lineno)
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        continue  # `from . import sub` binds the subpackage itself
                    for alias in node.names:
                        name = alias.asname or alias.name
                        if name != "*":
                            bound.setdefault(name, node.lineno)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[0]
                        bound.setdefault(name, node.lineno)
            for name, lineno in sorted(bound.items()):
                if name in subs:
                    hits.append((rel, lineno, name))
    return sorted(hits), counts


def _write_shadow_fixture(tmp_path: Path, init_source: str):
    """Write a two-level fake package; return ``(roots, src_root)``.

    Deliberately NOT ``_write_fixture``: this predicate's subject is a package
    __init__ *and the sibling directories beside it*, which a single-file fixture
    cannot express. Everything else about the shape -- parsed, never imported,
    RED source and fixed twin -- is the same.
    """
    pkg = tmp_path / "models" / "injected"
    (pkg / "components").mkdir(parents=True, exist_ok=True)
    (pkg / "components" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text(init_source)
    (pkg / "model.py").write_text("components = None\nInjected = None\n")
    return (tmp_path / "models",), tmp_path


#: A package whose ``__init__.py`` re-exports a name that happens to match its own
#: ``components/`` subpackage. Every test in the package still passes; the break
#: is ``import ...injected.components`` returning a non-module.
_INJECTED_SHADOW_DEFECT_SRC = '''from .model import Injected, components

__all__ = ["Injected", "components"]
'''

#: The same package, the shadow renamed, and -- deliberately -- the LEGAL
#: ``from . import components`` form added, so the fixed twin also proves the
#: exclusion is real rather than the predicate simply having stopped matching.
_INJECTED_SHADOW_FIXED_SRC = '''from . import components
from .model import Injected

__all__ = ["Injected", "components"]
'''


class TestNoInitShadowsItsOwnSubpackage:
    """R-013/R-052: a package ``__init__.py`` may not bind over its own subpackage.

    ``from .model import components`` in a package that also HAS a
    ``components/`` directory leaves ``<pkg>.components`` bound to whatever
    ``model.py`` exported. Nothing inside the package notices -- its own imports
    are relative and resolve to the module -- and the failure surfaces at
    collection time in some other suite, as an ``AttributeError`` or an
    ``ImportError`` naming the importer rather than the shadow.

    Measured 2026-08-20 over ``src/dl_techniques/``: 132 packages, 131
    package/subpackage relations, **0 shadows**. Freeze-forward; the injected pair
    below is the only evidence the predicate works.

    Narrowings, each a place this can miss a real shadow:

    * ``from .sub import *`` is not resolved, so a star-exported name matching a
      sibling subpackage is invisible;
    * bindings created at runtime (``globals()[name] = ...``, a ``for`` loop over
      ``__all__``) are not seen -- only module-level static binding forms.
    """

    def test_no_package_init_binds_over_a_subpackage(self):
        hits, _ = _sweep_init_subpackage_shadows()
        offenders = [
            f"{rel}:{line} binds {name!r}, which is also the name of the "
            f"{name}/ subpackage beside it"
            for rel, line, name in hits
        ]
        assert not offenders, (
            "a package __init__.py binds a name over one of its own subpackage "
            "directories, so `import <pkg>.<sub>` returns the bound object "
            "instead of the module. Rename the export. Found:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_sweep_found_packages(self):
        """Anti-vacuity floor, derived from the 2026-08-20 measurement.

        132 packages and 131 subpackage relations under ``src/dl_techniques/``;
        the floors are ``int(0.8 * n)`` of each. The subpackage-relation floor is
        the load-bearing one: a sweep that found the ``__init__.py`` files but
        stopped listing the directories beside them would report zero shadows
        forever while still passing a package count.
        """
        _, counts = _sweep_init_subpackage_shadows()
        assert counts["n_packages"] >= 105, counts
        assert counts["n_subpackage_relations"] >= 104, counts

    def test_predicate_fires_on_an_injected_shadow(self, tmp_path):
        """Dead-component probe: the predicate must go RED on a real shadow."""
        roots, src_root = _write_shadow_fixture(
            tmp_path, _INJECTED_SHADOW_DEFECT_SRC
        )
        hits, counts = _sweep_init_subpackage_shadows(roots, src_root)
        assert counts["n_subpackage_relations"] == 1, counts
        assert [(rel, name) for rel, _, name in hits] == [
            ("models/injected/__init__.py", "components")
        ], hits

    def test_predicate_is_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire on the rename, nor on `from . import <sub>`."""
        roots, src_root = _write_shadow_fixture(tmp_path, _INJECTED_SHADOW_FIXED_SRC)
        hits, counts = _sweep_init_subpackage_shadows(roots, src_root)
        assert counts["n_subpackage_relations"] == 1, "the fixture must be reached"
        assert hits == [], hits


# ---------------------------------------------------------------------------
# The three D-003 prevention guards: `by_name`, swallowed loads, `pretrained`.
# ---------------------------------------------------------------------------

#: Callees whose kwargs are searched for `by_name`.
_WEIGHT_LOAD_CALLEES = frozenset(
    {"load_weights", "load_own_variables", "set_weights"}
)

#: Callees whose presence inside a ``try`` body makes that ``try`` a weight load.
_WEIGHT_LOAD_TRY_CALLEES = frozenset(
    {
        "load_weights",
        "load_model",
        "load_weights_or_raise",
        "_download_weights",
        "_download_bfunet_weights",
        "get_file",
        "load_own_variables",
        "set_weights",
    }
)


@_memo_default_roots
def _sweep_by_name_kwargs(roots=None, src_root=None):
    """``load_weights(..., by_name=...)`` call sites. Contract: ``(hits, n_calls)``.

    ``hits`` is ``["relpath:lineno"]``; ``n_calls`` is the whole weight-load call
    population, which is the anti-vacuity denominator.
    """
    roots = (SRC_ROOT,) if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    hits = []
    n_calls = 0
    for rel, tree in _iter_modules(roots, src_root):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _callee_name(node) not in _WEIGHT_LOAD_CALLEES:
                continue
            n_calls += 1
            if any(kw.arg == "by_name" for kw in node.keywords):
                hits.append(f"{rel}:{node.lineno}")
    return sorted(hits), n_calls


def _enclosing_symbol_map(tree: ast.Module) -> dict:
    """Map every node to the name of the nearest enclosing def/class, or ``"<module>"``.

    Contract: takes a parsed module; returns a dict keyed by node identity. Used to
    key the R-049 waivers by ENCLOSING SYMBOL rather than by line number, so a
    waiver keeps matching while the fixes around it move lines.
    """
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    out = {}
    for node in ast.walk(tree):
        cur = parents.get(node)
        name = "<module>"
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = cur.name
                break
            cur = parents.get(cur)
        out[id(node)] = name
    return out


@_memo_default_roots
def _sweep_weight_load_handlers(roots=None, src_root=None):
    """``except`` handlers around a weight load that log without re-raising.

    Contract: returns ``(offenders, n_handlers)``. ``offenders`` is
    ``[(relpath, lineno, enclosing symbol, except-clause source)]`` for every
    handler that logs and does NOT re-raise; ``n_handlers`` counts every handler
    around a weight load, offending or not.
    """
    roots = (SRC_ROOT,) if roots is None else roots
    src_root = SRC_ROOT if src_root is None else src_root
    offenders = []
    n_handlers = 0
    for rel, tree in _iter_modules(roots, src_root):
        enclosing = _enclosing_symbol_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            if not any(
                isinstance(c, ast.Call) and _callee_name(c) in _WEIGHT_LOAD_TRY_CALLEES
                for stmt in node.body
                for c in ast.walk(stmt)
            ):
                continue
            for handler in node.handlers:
                n_handlers += 1
                logs = any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and isinstance(c.func.value, ast.Name)
                    and c.func.value.id == "logger"
                    for stmt in handler.body
                    for c in ast.walk(stmt)
                )
                reraises = any(isinstance(n, ast.Raise) for n in ast.walk(handler))
                if logs and not reraises:
                    clause = (
                        ast.unparse(handler.type) if handler.type is not None else "bare"
                    )
                    offenders.append(
                        (rel, handler.lineno, enclosing[id(handler)], clause)
                    )
    return sorted(offenders), n_handlers


#: The seven handlers that match "logs a weight-load failure and falls through"
#: today, keyed by ``(relpath, enclosing symbol, except-clause source)`` -- NEVER
#: by line number -- each with the read from D-003 §6 that clears it.
#:
#: This is a WAIVER, not a narrowed predicate, and the distinction is the whole
#: point. Narrowing the predicate to exclude these shapes (e.g. "only bare
#: ``except Exception`` counts", or "only handlers around ``load_weights``")
#: would stop matching the moment somebody broadened one of the five narrow
#: tuples to ``Exception`` -- which is the ONLY way this defect can appear here,
#: since all five are currently unreachable precisely because their tuple does
#: not catch the ``NotImplementedError`` their body can raise. A waiver keyed to
#: the clause TEXT fails loudly on exactly that edit.
_WEIGHT_LOAD_HANDLER_WAIVERS = {
    # 1-5: dead by construction. `_download_weights` raises only
    # NotImplementedError, which `(IOError, OSError, ValueError)` does not catch,
    # so the handler body never runs and the NIE reaches the caller (runtime-proved
    # at D-003 §3a: 45/45 sites raise). Each is one broadened `except` away from
    # being the live defect, which is what the clause-text key catches.
    (
        "models/bert/bert.py",
        "from_variant",
        "(IOError, OSError, ValueError)",
    ): "dead handler -- _download_weights raises only NotImplementedError",
    (
        "models/cliffordnet/model.py",
        "from_variant",
        "(IOError, OSError, ValueError)",
    ): "dead handler -- same proof as bert",
    (
        "models/tree_transformer/model.py",
        "from_variant",
        "(IOError, OSError, ValueError)",
    ): "dead handler -- same proof as bert",
    (
        "models/vit/model.py",
        "from_variant",
        "(IOError, OSError, ValueError)",
    ): "dead handler -- same proof as bert; scheduled for deletion (prior plan F-32)",
    (
        "models/wave_field/model.py",
        "from_variant",
        "(IOError, OSError, ValueError)",
    ): (
        "dead handler -- same proof as bert; carries an explicit "
        "`# DECISION plan-2026-08-13T091555-230c101d/D-005` forbidding broadening "
        "the tuple to Exception, i.e. the narrow tuple IS the guard there"
    ),
    # 6-7: matched by the shape, but not R-049 -- no weight FILE is loaded.
    (
        "models/depth_anything/model.py",
        "build",
        "Exception",
    ): (
        "not R-049 -- degrades a teacher clone and sets use_feature_alignment = "
        "False; a genuine silent-knob-disable, routed to Phase 3 batch 5 as an "
        "R-119/R-120 candidate rather than waived away"
    ),
    (
        "models/depth_anything/model.py",
        "from_pretrained_encoder",
        "Exception",
    ): (
        "not R-049 -- the pretrained load itself (load_weights_from_checkpoint) "
        "is OUTSIDE any try; this handler covers only the post-load teacher "
        "EMA re-sync"
    ),
}


#: A `by_name` call site and a swallowed load, in one parsed-never-imported file.
_INJECTED_LOAD_DEFECT_SRC = '''
class InjectedLoader(keras.Model):
    def restore(self, path):
        self.load_weights(path, by_name=True, skip_mismatch=True)

    def restore_pretrained(self, path):
        try:
            self.load_weights(path)
        except (IOError, OSError):
            logger.warning("could not load %s, continuing with random weights", path)
'''

#: The same two methods repaired: the kwarg dropped (Keras 3 removed it), and the
#: handler re-raising instead of falling through into a random-init model.
_INJECTED_LOAD_FIXED_SRC = '''
class InjectedLoader(keras.Model):
    def restore(self, path):
        self.load_weights(path, skip_mismatch=True)

    def restore_pretrained(self, path):
        try:
            self.load_weights(path)
        except (IOError, OSError) as exc:
            logger.error("could not load %s", path)
            raise ValueError(f"failed to load weights from {path}") from exc
'''


class TestWeightLoadingContract:
    """The two static halves of the ``pretrained`` contract, frozen at zero.

    ``TestPretrainedNeverSilentlyRandom`` above covers the ``if pretrained:``
    branch. These cover the two ways a load can go wrong AFTER that branch, both
    measured at **0 live instances** on 2026-08-20 (D-003) and therefore both
    freeze-forward:

    * ``by_name=`` on a weight load. Keras 3 removed the kwarg; a call passing it
      raises, and where such calls "survived" it was because an enclosing
      ``except`` swallowed the raise. **The instrument must be AST**: the naive
      text control ``grep -rn by_name src/dl_techniques/`` returns 10 hits and
      ALL TEN are false positives -- 4 in ``.md`` files and 6 in comments that
      exist precisely to record that the kwarg was removed;
    * an ``except`` around a weight load that logs and falls through, returning a
      randomly-initialized model to a caller who asked for a trained one.
    """

    def test_no_weight_load_passes_by_name(self):
        # DECISION plan-2026-08-19T163559-499b6f0e/D-012
        # AST, never a text grep. WHAT NOT TO DO: do not "simplify" this into
        # `grep -rn by_name`. Measured at D-003: the text form returns 10 hits
        # over src/dl_techniques/ and every one is a false positive -- 4 .md
        # files and 6 comments/docstrings written to record that Keras 3 REMOVED
        # the kwarg. A text guard here would be permanently red for reasons that
        # have nothing to do with the contract, and would be deleted. See
        # decisions.md D-012 and reference_grep_over_this_repo_finds_docstrings.
        hits, _ = _sweep_by_name_kwargs()
        assert not hits, (
            "`by_name` was removed from Model.load_weights in Keras 3; a call "
            "passing it raises at runtime. Use dl_techniques.utils.weight_transfer "
            f"for partial restores instead. Found: {hits}"
        )

    def test_no_weight_load_handler_logs_and_falls_through(self):
        # DECISION plan-2026-08-19T163559-499b6f0e/D-012
        # The seven known handlers are cleared by an EVIDENCE-KEYED WAIVER, not by
        # a narrowed predicate. WHAT NOT TO DO: do not narrow this to bare
        # `except Exception`, and do not restrict it to `load_weights` callees, to
        # make the seven go away. Five of the seven are dead ONLY because their
        # `(IOError, OSError, ValueError)` tuple fails to catch the
        # NotImplementedError their body can raise -- so broadening that tuple is
        # the single edit that turns them into the live defect, and a narrowed
        # predicate is exactly the thing that would stop seeing them at that
        # moment. The waiver is keyed by clause TEXT so that edit fails loudly.
        # See decisions.md D-012.
        offenders, _ = _sweep_weight_load_handlers()
        live = [
            f"{rel}:{line} in {sym}(): `except {clause}` logs and falls through"
            for rel, line, sym, clause in offenders
            if (rel, sym, clause) not in _WEIGHT_LOAD_HANDLER_WAIVERS
        ]
        assert not live, (
            "an except around a weight load logs and falls through, so a failed "
            "load returns a randomly-initialized model as if it were pretrained. "
            "Re-raise, or -- if the handler is provably not a weight-file load -- "
            "add it to _WEIGHT_LOAD_HANDLER_WAIVERS with the read that clears it. "
            "Found:\n  " + "\n  ".join(live)
        )

    def test_waivers_still_match_a_real_site(self):
        """A waiver matching nothing is a waiver hiding nothing (see the siblings).

        This one has teeth beyond the usual: step 25 of this plan DELETES the
        ``vit`` handler, and two of the five carry decision anchors. A path-keyed
        or line-keyed waiver would linger after either edit and silently permit a
        LATER, DIFFERENT handler at the same site.
        """
        offenders, _ = _sweep_weight_load_handlers()
        live = {(rel, sym, clause) for rel, _, sym, clause in offenders}
        stale = sorted(set(_WEIGHT_LOAD_HANDLER_WAIVERS) - live)
        assert not stale, (
            "weight-load handler waivers no longer match any site; delete them if "
            f"the handler is gone, re-key them if the code moved: {stale}"
        )

    def test_the_sweeps_found_load_sites(self):
        """Anti-vacuity floors, derived from the 2026-08-20 measurement.

        12 weight-load call sites and 9 handlers around a weight load, over
        ``src/dl_techniques/``. Floors are ``int(0.8 * n)`` of each -- 9 and 7.
        Note that this population is genuinely SMALL: the D-003 write-up proposed
        a floor of ">= 30 call sites" from a partial count, and re-deriving it at
        the moment the guard landed is what caught that. A floor above the
        population is a guard that fails for a reason unrelated to its contract.
        """
        _, n_calls = _sweep_by_name_kwargs()
        _, n_handlers = _sweep_weight_load_handlers()
        assert n_calls >= 9, (
            f"expected ~12 weight-load call sites, found {n_calls}: the AST walk "
            "stopped seeing the tree"
        )
        assert n_handlers >= 7, (
            f"expected ~9 handlers around a weight load, found {n_handlers}"
        )

    def test_predicates_fire_on_an_injected_defect(self, tmp_path):
        """Dead-component probe: both predicates must go RED on a real site."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_LOAD_DEFECT_SRC)
        hits, n_calls = _sweep_by_name_kwargs(roots, src_root)
        assert n_calls == 2, n_calls
        assert hits == ["models/injected/model.py:4"], hits
        offenders, n_handlers = _sweep_weight_load_handlers(roots, src_root)
        assert n_handlers == 1, n_handlers
        assert [(rel, sym, clause) for rel, _, sym, clause in offenders] == [
            ("models/injected/model.py", "restore_pretrained", "(IOError, OSError)")
        ], offenders

    def test_predicates_are_silent_on_the_fixed_twin(self, tmp_path):
        """...and must NOT fire once the kwarg is dropped and the handler re-raises."""
        roots, src_root = _write_fixture(tmp_path, _INJECTED_LOAD_FIXED_SRC)
        hits, n_calls = _sweep_by_name_kwargs(roots, src_root)
        assert n_calls == 2, "the fixture must still be reached"
        assert hits == [], hits
        offenders, n_handlers = _sweep_weight_load_handlers(roots, src_root)
        assert n_handlers == 1, "the fixture must still be reached"
        assert offenders == [], offenders


# ---------------------------------------------------------------------------
# The behavioural `pretrained` arm's own RED proof.
# ---------------------------------------------------------------------------


class _InjectedPretrainedDefect:
    """A factory that LOGS on ``pretrained=True`` and returns a random model.

    The exact nine-site defect D-003 was written about, in miniature: it has a
    real ``MODEL_VARIANTS`` table so the discovery arm reaches it, and it does not
    raise, so the behavioural predicate must fail on it.
    """

    MODEL_VARIANTS = {"small": {"width": 8}, "large": {"width": 64}}

    @classmethod
    def from_variant(cls, variant, pretrained=False, **kwargs):
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}: {list(cls.MODEL_VARIANTS)}")
        if pretrained:
            logger_message = "Pretrained weights are not yet implemented"
            del logger_message
        return cls.MODEL_VARIANTS[variant]


class _InjectedPretrainedFixed:
    """The same factory, repaired: ``pretrained=True`` raises before it builds."""

    MODEL_VARIANTS = {"small": {"width": 8}, "large": {"width": 64}}

    @classmethod
    def from_variant(cls, variant, pretrained=False, **kwargs):
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}: {list(cls.MODEL_VARIANTS)}")
        if pretrained:
            raise NotImplementedError("no public weights ship with dl_techniques")
        return cls.MODEL_VARIANTS[variant]


def _probe_pretrained_raises(fn, kwargs) -> str:
    """``""`` if ``fn(pretrained=True, **kwargs)`` raises ``NotImplementedError``.

    Contract: returns a one-line description of the violation otherwise. Factored
    out of ``test_pretrained_true_raises`` for one reason only -- so the predicate
    itself can be proven RED against the injected pair below. A predicate that
    exists only inside a parametrized test cannot be shown to fire.
    """
    try:
        fn(pretrained=True, **kwargs)
    except NotImplementedError:
        return ""
    except Exception as exc:  # noqa: BLE001 -- the wrong exception is also a failure
        return f"raised {type(exc).__name__} instead of NotImplementedError: {exc}"
    return "returned a model instead of raising"


class TestPretrainedBehaviouralArmIsProvenRed:
    """The widened behavioural arm, proven to fire.

    The arm reaches 42 of the 45-site AST population and every one of the 42
    raises today, so nothing in the tree can demonstrate that the arm would notice
    if one stopped. These four tests do, on both halves of it: the RESOLVER (does
    it read ``MODEL_VARIANTS`` and produce a callable argument set?) and the
    PREDICATE (does ``pretrained=True`` raise?).
    """

    def test_the_predicate_fires_on_an_injected_silent_random(self):
        """Dead-component probe: a log-and-return factory must be caught."""
        kwargs = {}
        value, reason = _resolve_required_arg(
            "variant", _InjectedPretrainedDefect.from_variant
        )
        assert reason == "", reason
        kwargs["variant"] = value
        assert value == "small", "the resolver must pick the smallest named variant"
        problem = _probe_pretrained_raises(
            _InjectedPretrainedDefect.from_variant, kwargs
        )
        assert problem == "returned a model instead of raising", problem

    def test_the_predicate_is_silent_on_the_fixed_twin(self):
        """...and must NOT fire once the same factory raises."""
        value, reason = _resolve_required_arg(
            "variant", _InjectedPretrainedFixed.from_variant
        )
        assert reason == ""
        assert _probe_pretrained_raises(
            _InjectedPretrainedFixed.from_variant, {"variant": value}
        ) == ""

    def test_the_resolver_reports_an_unreachable_from_variant(self):
        """The resolver must SAY it cannot reach a table, never guess one.

        Guessing (e.g. falling back to ``"base"``) would turn an unreachable site
        into one that silently passes on a ValueError path, which is how a
        coverage arm rots into a coverage claim.

        The injected class is built in a THROWAWAY MODULE, not beside its twins in
        this file, and that is load-bearing: ``_variant_table_for``'s second
        resolution step reads the defining module's classes, so a no-table class
        declared here would resolve ``_InjectedPretrainedDefect.MODEL_VARIANTS``
        and this test would prove the opposite of what it says. (It did, the first
        time it was run -- the resolver returned ``"small"``.)
        """
        module = types.ModuleType("_injected_no_variant_table")
        sys.modules[module.__name__] = module
        try:
            exec(  # noqa: S102 -- a 3-line fixture, parsed nowhere else
                "class NoTable:\n"
                "    @classmethod\n"
                "    def from_variant(cls, variant, pretrained=False, **kwargs):\n"
                "        return variant\n",
                module.__dict__,
            )
            value, reason = _resolve_required_arg(
                "variant", module.NoTable.from_variant
            )
        finally:
            del sys.modules[module.__name__]
        assert value is None
        assert "MODEL_VARIANTS" in reason, reason

    def test_every_reached_factory_raises(self):
        """The whole reached set, as a single non-parametrized assertion.

        Redundant with ``test_pretrained_true_raises`` by design: that one reports
        per-site (which is what a developer wants), this one reports the SET (which
        is what makes "42 of 45 raise" a single checkable statement, and what the
        RED proof above is a proof about).
        """
        problems = {
            label: problem
            for label, fn, kwargs in PRETRAINED_FACTORIES
            if (problem := _probe_pretrained_raises(fn, kwargs))
        }
        assert not problems, problems
