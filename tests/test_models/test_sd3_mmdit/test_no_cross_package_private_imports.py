"""``sd3_mmdit`` must not import an underscore-prefixed name from another package.

``plan-2026-08-31-a4e0c303`` de-duplicated two helpers by deleting the local
copies in ``sd3_mmdit`` and importing the surviving owners instead. The first
pass imported them under their original PRIVATE spellings
(``sd3_adaln._modulate``, ``ideogram4.config._validate_vae_groupnorm``), which
made both names lie: a leading underscore says "module-private", yet the name
had an out-of-package consumer, so a future reader tidying away an "unused
private helper" would silently break this package. Step ``2.1``/``3.2`` promoted
both to public names; this file is the guard that they stay public and that no
NEW cross-package private import appears here.

The sweep is AST-based over every module in the package, so it also covers
helpers this plan never touched. It carries an anti-vacuity floor: if the walk
stops seeing cross-package imports at all, the parametrized arms would pass
vacuously, so the census size is asserted separately.
"""

import ast
import pathlib

import pytest

# ---------------------------------------------------------------------

PACKAGE = "dl_techniques.models.vision_language.sd3_mmdit"
PACKAGE_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "dl_techniques"
    / "models"
    / "vision_language"
    / "sd3_mmdit"
)

# Anti-vacuity floor, set well below today's census and well above zero: it
# exists to make a broken walk (or a package emptied by a restructure) fail
# loudly rather than turn every arm below into a vacuous pass. Do NOT raise it
# to whatever the census currently reads -- that turns an import added anywhere
# in the package into a spurious failure.
_CENSUS_FLOOR = 40


def _cross_package_imported_names():
    """``(module_file, from_module, imported_name)`` for every out-of-package import."""
    found = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                # A relative import cannot leave the package here (every one is
                # `from .submodule import ...` in `__init__.py`).
                continue
            module = node.module or ""
            if module == PACKAGE or module.startswith(PACKAGE + "."):
                continue
            for alias in node.names:
                found.append((path.name, module, alias.name))
    return found


CROSS_PACKAGE_IMPORTS = _cross_package_imported_names()

# ---------------------------------------------------------------------


class TestTheSharedHelpersArePublicAtTheirOwners:
    """Both de-duplicated helpers are reachable under a public spelling."""

    def test_modulate_is_public_in_sd3_adaln(self):
        from dl_techniques.layers.transformers import sd3_adaln

        assert callable(sd3_adaln.modulate)
        assert not hasattr(sd3_adaln, "_modulate"), (
            "the private spelling is back (or a compatibility alias was left "
            "behind); `sd3_mmdit/blocks.py` imports this name across a package "
            "boundary, so the public name must be the only one"
        )

    def test_validate_vae_groupnorm_is_public_in_ideogram4(self):
        from dl_techniques.models.vision_language.ideogram4 import config

        assert callable(config.validate_vae_groupnorm)
        assert not hasattr(config, "_validate_vae_groupnorm"), (
            "the private spelling is back (or a compatibility alias was left "
            "behind); `sd3_mmdit/config.py` imports this name across a package "
            "boundary, so the public name must be the only one"
        )


class TestNoCrossPackagePrivateImport:
    """No module in this package imports an underscored name from outside it."""

    def test_the_census_is_not_empty_or_shrinking(self):
        assert len(CROSS_PACKAGE_IMPORTS) >= _CENSUS_FLOOR, (
            f"only {len(CROSS_PACKAGE_IMPORTS)} cross-package imported names "
            f"were found under {PACKAGE_DIR}; the walk is broken or the "
            f"package moved, and every arm below is passing vacuously"
        )

    @pytest.mark.parametrize(
        "entry",
        CROSS_PACKAGE_IMPORTS,
        ids=[f"{f}:{m.rsplit('.', 1)[-1]}.{n}" for f, m, n in CROSS_PACKAGE_IMPORTS],
    )
    def test_the_imported_name_is_public(self, entry):
        module_file, from_module, name = entry
        assert not name.startswith("_"), (
            f"{module_file} imports the private name `{name}` from "
            f"`{from_module}`, which is outside this package. A leading "
            f"underscore on a name with an out-of-package consumer is a false "
            f"privacy marker: promote the name at its owner instead"
        )
