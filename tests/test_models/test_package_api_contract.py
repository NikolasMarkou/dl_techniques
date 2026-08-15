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
import importlib
import inspect
import re
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


def _pretrained_factories() -> List[Tuple[str, Any]]:
    """Every public model factory that (a) takes ``pretrained`` and (b) is callable
    with no other argument.

    Discovered from the real package listing, so a new model package is covered the
    moment it is added. The "no other required argument" filter is what keeps this
    test cheap and generic: it needs no hand-written table of variant names, and the
    contract under test (``pretrained=True`` raises) fires before anything is built.
    """
    out: List[Tuple[str, Any]] = []
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
                params = sig.parameters
                if "pretrained" not in params:
                    continue
                required = [
                    n
                    for n, p in params.items()
                    if n != "pretrained"
                    and p.default is inspect.Parameter.empty
                    and p.kind
                    not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                ]
                if required:
                    continue
                key = (fn.__module__, getattr(fn, "__qualname__", label))
                if key in seen:
                    continue
                seen.add(key)
                out.append((label, fn))
    return sorted(out, key=lambda t: t[0])


PRETRAINED_FACTORIES = _pretrained_factories()


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

    def test_the_behavioural_arm_found_factories(self):
        """The parameterization must not silently collapse to nothing."""
        assert len(PRETRAINED_FACTORIES) >= 20, (
            "expected at least 20 no-required-argument factories taking "
            f"`pretrained`; discovery found {len(PRETRAINED_FACTORIES)}: "
            f"{[label for label, _ in PRETRAINED_FACTORIES]}"
        )

    @pytest.mark.parametrize(
        "label,factory",
        PRETRAINED_FACTORIES,
        ids=[label for label, _ in PRETRAINED_FACTORIES],
    )
    def test_pretrained_true_raises(self, label: str, factory):
        """``pretrained=True`` must raise; no public weights ship with this repo."""
        with pytest.raises(NotImplementedError):
            factory(pretrained=True)


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
