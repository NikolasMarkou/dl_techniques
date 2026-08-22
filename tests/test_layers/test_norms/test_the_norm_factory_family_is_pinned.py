"""Ceiling gate + raise contract for ``create_normalization_layer``'s call-site family.

Why this file exists
====================

178 call sites across 83 files in ``src/`` reach ``create_normalization_layer``.
**149 of them pass a variable, not a literal** (94 an ``Attribute`` such as
``self.normalization_type``, 55 a bare ``Name``), so no static instrument can
know which of the 18 registry keys they will ask for. That population was
reported but never pinned: D-136 gave a ceiling gate to its *sibling* family
(``_DYNAMIC_CALL_SITE_CEILING = 15`` in
``tests/test_models/test_package_api_contract.py``) and explicitly recorded the
normalization sweep as "reported-but-unpinned". This file closes that gap.

The residual risk is **growth, not silent fallback.** The factory *raises* on an
unknown key -- it does not quietly fall back to any layer -- so each of the 149
dynamic sites is self-guarding at runtime the moment its branch is exercised.
``test_an_unknown_normalization_type_raises_and_names_the_valid_ones`` pins that
raise, because it is the entire reason a population ceiling is an adequate
ruling here rather than a per-value assertion.
"""

import ast
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import pytest

from dl_techniques.layers.norms.factory import (
    _TYPE_TO_CLASS,
    create_normalization_layer,
)

_TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(os.path.dirname(_TESTS_DIR), "src")

#: Ceiling on ``create_normalization_layer`` call sites whose ``normalization_type``
#: is a variable rather than a string literal. Such a site has no statically
#: knowable key, so no static sweep can validate it at all.
#:
#: MEASURED over all of ``src/`` (``dl_techniques`` + ``applications`` + ``train``)
#: 2026-08-23: **178 sites / 83 files**, split **94 Attribute + 55 Name = 149
#: uncheckable**, 29 literal. Reproduces D-140's table bit-for-bit, including the
#: file count. Note the scope: restricted to ``src/dl_techniques`` alone the same
#: sweep reads 172/78/143 -- six of the dynamic sites live outside the library
#: package, which is why the sweep root is ``src/``.
#:
#: A grep-based count is wrong in BOTH directions here (D-140 measured grep at
#: 198 total / 20 literal against the true 178 / 29). Use the AST.
_UNCHECKABLE_NORM_CALL_SITE_CEILING = 149

#: Ceiling on the total population, dynamic and literal together. Pinned as well
#: so that "a dynamic site became literal" and "a site disappeared" are
#: distinguishable from "the family shrank".
_TOTAL_NORM_CALL_SITE_CEILING = 178


def _sweep_norm_call_sites() -> Tuple[Counter, List[str], List[str]]:
    """Return ``(classification_counts, dynamic_sites, literal_sites)``."""
    counts: Counter = Counter()
    dynamic: List[str] = []
    literal: List[str] = []
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, SRC_ROOT)
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            # Deliberately no `except SyntaxError: continue` -- a silent skip
            # would silently shrink every number in this file (D-070).
            tree = ast.parse(source, filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                callee = getattr(node.func, "attr", getattr(node.func, "id", None))
                if callee != "create_normalization_layer":
                    continue
                arg = node.args[0] if node.args else None
                for keyword in node.keywords:
                    if keyword.arg == "normalization_type":
                        arg = keyword.value
                site = f"{rel}:{node.lineno}"
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    counts["LITERAL"] += 1
                    literal.append(site)
                elif isinstance(arg, ast.Attribute):
                    counts["DYNAMIC(Attribute)"] += 1
                    dynamic.append(site)
                elif isinstance(arg, ast.Name):
                    counts["DYNAMIC(Name)"] += 1
                    dynamic.append(site)
                else:
                    counts[f"OTHER({type(arg).__name__})"] += 1
                    dynamic.append(site)
    return counts, dynamic, literal


@pytest.fixture(scope="module")
def norm_call_sites() -> Tuple[Counter, List[str], List[str]]:
    return _sweep_norm_call_sites()


class TestTheNormFactoryCallSitePopulationIsPinned:
    """Growth is the only residual risk, so growth is what is pinned."""

    def test_the_statically_uncheckable_population_has_not_grown(self, norm_call_sites):
        counts, dynamic, _ = norm_call_sites
        if dynamic:
            warnings.warn(
                "create_normalization_layer call sites with a non-literal type "
                f"(not statically checkable, {len(dynamic)}): {dynamic}",
                UserWarning,
                stacklevel=2,
            )
        assert len(dynamic) <= _UNCHECKABLE_NORM_CALL_SITE_CEILING, (
            "statically uncheckable create_normalization_layer call sites grew "
            f"from {_UNCHECKABLE_NORM_CALL_SITE_CEILING} to {len(dynamic)} "
            f"(classification: {dict(counts)}). Each new one is a key no static "
            "sweep can resolve; it is only validated the moment that particular "
            "runtime branch executes. Either give the new site a literal "
            "`normalization_type=`, or re-open the ruling in decisions.md D-201. "
            "Do NOT raise this ceiling to make a red test green."
        )

    def test_the_total_population_has_not_grown(self, norm_call_sites):
        counts, dynamic, literal = norm_call_sites
        total = len(dynamic) + len(literal)
        assert total <= _TOTAL_NORM_CALL_SITE_CEILING, (
            f"create_normalization_layer call sites grew from "
            f"{_TOTAL_NORM_CALL_SITE_CEILING} to {total} "
            f"(classification: {dict(counts)}). See decisions.md D-201."
        )


class TestTheNormFactoryRaisesRatherThanFallingBack:
    """The reason a population ceiling is a sufficient ruling for this family."""

    def test_an_unknown_normalization_type_raises_and_names_the_valid_ones(self):
        with pytest.raises(ValueError) as excinfo:
            create_normalization_layer("not_a_real_normalization")
        message = str(excinfo.value)
        assert "not_a_real_normalization" in message, (
            "the raise must echo the offending key; a config-driven caller has "
            "no other way to find which of its 149 dynamic sites misfired"
        )
        for valid_key in sorted(_TYPE_TO_CLASS):
            assert valid_key in message, (
                f"the ValueError omits the supported type {valid_key!r}. The "
                "error text is the only inventory a caller sees at the moment "
                "it fails."
            )

    def test_the_raise_is_not_a_silent_fallback(self):
        """A returned layer -- of ANY class -- would defeat every dynamic site."""
        try:
            result = create_normalization_layer("definitely_not_a_norm_type")
        except ValueError:
            return
        pytest.fail(
            "create_normalization_layer returned "
            f"{type(result).__name__} for an unknown key instead of raising. "
            "That single behaviour change would turn all 149 statically "
            "uncheckable call sites from self-guarding into silently wrong, and "
            "the population ceiling above would no longer be an adequate ruling."
        )

    @pytest.mark.parametrize("valid_key", sorted(_TYPE_TO_CLASS))
    def test_every_registry_key_actually_constructs(self, valid_key):
        layer = create_normalization_layer(valid_key)
        assert isinstance(layer, type(layer))
        assert layer is not None
