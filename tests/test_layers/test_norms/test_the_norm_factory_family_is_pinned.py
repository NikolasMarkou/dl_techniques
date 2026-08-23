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

import keras
import pytest

from tests.norm_epsilon_oracle import _epsilon_of

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

    # R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251. This test
    # REPORTS a census through `warnings.warn` on purpose -- the inventory is
    # non-fatal by design, and turning it into an assertion would make a
    # growth report indistinguishable from the ceiling breach the sibling
    # test convicts. Under the repo-wide `error::UserWarning` that report
    # would become a failure, so this ONE test opts back out.
    @pytest.mark.filterwarnings("always::UserWarning")
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


# ---------------------------------------------------------------------------
# N-11: the factory is NOT a drop-in for constructing the layer directly.
# ---------------------------------------------------------------------------

#: What each registry key's epsilon is when the caller says nothing, MEASURED
#: 2026-08-23 on keras 3.8.0: ``(factory default, the class's OWN default)``.
#: ``None`` means the layer has no epsilon at all (``dynamic_tanh`` -- the factory
#: pops the key). Entries whose two values differ are the trap: a caller who
#: "just routes through the factory like resnet does" silently moves epsilon.
#:
#: **11 of the 16 bare-constructible types diverge.** The 1000x ``batch_norm`` /
#: ``layer_norm`` rows are
#: the ones that nearly shipped: step 18's brief proposed routing mobilenet and
#: cbam (189 BatchNormalization layers) through this factory, which would have
#: divided their epsilon by 1000 with no test noticing.
_EPSILON_DEFAULTS = {
    "adaptive_band_rms": (1e-06, 1e-07),
    "band_logit_norm": (1e-06, 1e-07),
    "band_rms": (1e-06, 1e-07),
    "batch_norm": (1e-06, 1e-03),
    "bias_free_batch_norm": (1e-06, 1e-06),
    "decoupled_max_logit": (1e-06, 1e-07),
    "dynamic_tanh": (None, None),
    "energy_layer_norm": (1e-06, 1e-05),
    "global_response_norm": (1e-06, 1e-06),
    "layer_norm": (1e-06, 1e-03),
    "logit_norm": (1e-06, 1e-07),
    "max_logit_norm": (1e-06, 1e-07),
    "rms_norm": (1e-06, 1e-06),
    "zero_centered_adaptive_band_rms_norm": (1e-06, 1e-07),
    "zero_centered_band_rms_norm": (1e-06, 1e-07),
    "zero_centered_rms_norm": (1e-06, 1e-06),
}

#: ``dml_plus_focal`` / ``dml_plus_center`` cannot be constructed bare (their
#: classes take a required positional argument), so their class-own default is
#: not observable this way. Named rather than silently absent, because "the dict
#: happens not to mention it" and "it was checked and excluded" must not look the
#: same (LESSONS: an all-skip module reads as a pass).
_EPSILON_UNCONSTRUCTIBLE_BARE = ("dml_plus_focal", "dml_plus_center")


class TestTheFactoryEpsilonIsNotTheLayerDefault:
    """N-11. Make the 1000x trap impossible to fall into a second time."""

    def test_the_keras_batchnorm_default_is_still_a_thousand_times_the_factorys(self):
        keras_default = keras.layers.BatchNormalization().epsilon
        factory_default = _epsilon_of(create_normalization_layer("batch_norm"))
        assert keras_default == pytest.approx(1e-3), (
            f"keras.layers.BatchNormalization default epsilon moved to "
            f"{keras_default}. Every claim in create_normalization_layer's "
            "not-a-drop-in warning is calibrated against 1e-3; re-measure the "
            "whole table in D-202 before touching this number."
        )
        assert factory_default == pytest.approx(1e-6), (
            f"create_normalization_layer's epsilon default moved to "
            f"{factory_default}. That silently re-tunes every one of the 178 "
            "call sites that does not pass epsilon explicitly."
        )
        assert keras_default / factory_default == pytest.approx(1000.0), (
            "the documented 1000x divergence between "
            "keras.layers.BatchNormalization() and "
            "create_normalization_layer('batch_norm') no longer holds. Either "
            "the trap is gone (update the factory warning and D-202) or one "
            "default drifted (find out which)."
        )

    @pytest.mark.parametrize("norm_type", sorted(_EPSILON_DEFAULTS))
    def test_the_factory_and_class_defaults_are_both_pinned(self, norm_type):
        expected_factory, expected_own = _EPSILON_DEFAULTS[norm_type]
        observed_factory = _epsilon_of(create_normalization_layer(norm_type))
        observed_own = _epsilon_of(_TYPE_TO_CLASS[norm_type]())
        matches_factory = (
            observed_factory is None
            if expected_factory is None
            else observed_factory == pytest.approx(expected_factory)
        )
        matches_own = (
            observed_own is None
            if expected_own is None
            else observed_own == pytest.approx(expected_own)
        )
        assert matches_factory, (
            f"{norm_type}: factory epsilon default moved from "
            f"{expected_factory} to {observed_factory}"
        )
        assert matches_own, (
            f"{norm_type}: the CLASS's own epsilon default moved from "
            f"{expected_own} to {observed_own}. The divergence table in "
            "create_normalization_layer's warning is now wrong."
        )

    def test_the_divergent_majority_is_still_the_majority(self):
        diverging = sorted(
            key
            for key, (factory, own) in _EPSILON_DEFAULTS.items()
            if factory != own
        )
        assert len(diverging) == 11, (
            f"the number of registry types whose factory epsilon differs from "
            f"their own class default changed from 11 to {len(diverging)}: "
            f"{diverging}. That is the headline number in the factory's "
            "not-a-drop-in warning."
        )

    @pytest.mark.parametrize("norm_type", _EPSILON_UNCONSTRUCTIBLE_BARE)
    def test_the_excluded_types_are_excluded_for_the_stated_reason(self, norm_type):
        with pytest.raises(TypeError):
            _TYPE_TO_CLASS[norm_type]()

    def test_the_factory_warns_in_its_own_docstring(self):
        doc = create_normalization_layer.__doc__ or ""
        assert "NOT A DROP-IN REPLACEMENT" in doc, (
            "create_normalization_layer's docstring no longer carries the "
            "not-a-drop-in warning. That warning is the only thing standing "
            "between a reviewer and the 1000x epsilon change that step 18's "
            "brief proposed for 189 mobilenet/cbam layers."
        )
        assert "1000x" in doc and "batch_norm" in doc
