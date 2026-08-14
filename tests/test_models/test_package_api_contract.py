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
import re
from pathlib import Path
from typing import List

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


class TestNoKeras2Residues:
    """The forward path must use Keras 3 spellings.

    ``keras.backend.GradientTape`` does not exist in Keras 3 at all: it sat in
    ``latent_gmm_registration.train_step`` and made the model untrainable while
    its suite stayed green, because every test was forward-pass only.
    """

    def test_no_keras_backend_calls(self):
        offenders = []
        for path in MODELS_DIR.rglob("*.py"):
            for i, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "keras.backend." in line:
                    offenders.append(f"{path.relative_to(MODELS_DIR)}:{i} {stripped}")
        assert not offenders, (
            "use keras.config.floatx()/epsilon() and tf.GradientTape; "
            f"keras.backend.* found at: {offenders}"
        )
