"""The legacy ``Custom>`` alias namespace: no refusals, no duplicate names, no shadows.

Every class registered through ``dl_techniques.utils.keras_registration.register_dl_technique``
binds TWO keys: a package-qualified one (``dl_techniques.layers.ffn>MLPBlock``) and a legacy
alias under the bare prefix that a stock ``@keras.saving.register_keras_serializable()`` mints
by default. The legacy namespace is **flat and repo-global** -- ``src/train/`` classes and the
bare fixtures under ``tests/`` compete in the identical slot space -- so a census scoped to
``src/dl_techniques/`` would report a uniqueness it never measured. All four roots are walked.

**Why this module exists at all.** The contract it asserts had, before it, essentially no guard:

* ``tests/conftest.py``'s purpose-built ``assert_package_qualified_registration`` /
  ``registration_contract`` was written for exactly this audit and, measured 2026-09-01, had
  **16 call sites and zero** passing ``expect_legacy_alias`` -- and none targeting any class
  that carried a refusal. A fixture nobody calls is a guard-shaped hole. Arm (d) below is its
  first live wiring.
* ``tests/test_serialization_registry.py`` walks the package with ``pkgutil.walk_packages`` and
  wraps each import in ``except Exception: continue``. That swallows precisely the
  ``AliasCollisionError`` a bad flip raises, while its population floor keeps passing. Arm (e)
  is the repair.
* ``tests/test_models/test_package_api_contract.py``'s ``_sweep_registry_keys`` keys on the
  QUALIFIED key and deliberately ignores the alias -- structurally blind to the bare-alias
  family this module is about. The two answer different questions and are deliberately NOT
  merged into one predicate.

**The census is executable.** It re-derives its population from the tree with ``ast`` at test
time and never carries a hand-maintained roster of names, because in this repository a
read-derived population of 8 has measured 14, and the "four names carry a refusal" claim went
stale in five files at once while every one of them kept agreeing with the others.
"""

import ast
import importlib
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import keras

from dl_techniques.utils.keras_registration import LEGACY_ALIAS_PREFIX

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The whole namespace, not a slice of it. See the module docstring.
_ROOTS = (
    REPO_ROOT / "src" / "dl_techniques",
    REPO_ROOT / "src" / "train",
    REPO_ROOT / "src" / "applications",
    REPO_ROOT / "tests",
)

#: Both decorator spellings that can put an object into the legacy namespace.
_REGISTRATION_DECORATORS = ("register_keras_serializable", "register_dl_technique")

#: Measured at ``755f06a38`` (2026-09-01) over the four roots: 788 registration sites,
#: 772 of them alias-relevant. The anti-vacuity floor is ``int(0.8 * 788)``: a fifth of
#: the tree's registrations may legitimately disappear before this walk is allowed to
#: call itself alive. A floor a few percent under the population would trip on an
#: ordinary refactor and would say nothing more about whether the walk still sees a tree.
_POPULATION_AT_BASELINE = 788
_POPULATION_FLOOR = int(0.8 * _POPULATION_AT_BASELINE)


def _decorator_name(node: ast.expr) -> str:
    """The bare name a decorator expression names, called or not.

    Contract: returns the last name component for ``@f``, ``@mod.f``, ``@f(...)`` and
    ``@mod.f(...)``; ``""`` for anything else. Never raises.
    """
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _const_str(node: ast.expr):
    """The literal ``str`` a node holds, or ``None`` if it is not a string constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _sweep_legacy_alias_namespace(roots=None, src_root=None) -> Tuple[Dict, Dict, Dict]:
    """Every registration site, split by what it does to the legacy alias namespace.

    Contract -- returns ``(claims, groups, counts)``:

    * ``claims`` maps the bare legacy name an object CLAIMS to the list of
      ``"relpath:lineno"`` claiming it. Two entries under one name is a live shadow:
      no exception is raised for it, the loser is decided by import order, and every
      alias-resolved deserialization of both objects goes to whichever imported last.
    * ``groups`` maps a bare object ``__name__`` to every ALIAS-RELEVANT site defining
      it, whether that site currently claims the alias or refuses it. This is the
      LATENT population: a refusal is exactly how a duplicate name hides from
      ``claims``, so a census that looked only at claimants would report the tree
      clean while the duplicate that forced the refusal sat untouched.
    * ``counts`` carries the vacuity denominators and the blind-spot inventory.

    ``roots`` / ``src_root`` default to the real tree. They exist so the injected-defect
    controls run against a ``tmp_path`` corpus. **Do not** redirect a control through
    ``PYTHONPATH`` instead: ``pyproject.toml`` sets ``pythonpath = ["src"]``, which wins,
    and the mutant would silently be measured against the real tree -- a false GREEN.

    Alias-relevance, and why it is narrower than "every registration site":

    * every ``register_dl_technique`` site is alias-relevant -- it either binds the
      legacy key or explicitly refuses it, and both are facts about this namespace;
    * a stock ``register_keras_serializable`` site is alias-relevant only when it lets
      ``package`` default, because that default IS the legacy prefix. A site passing an
      explicit package never enters this namespace at all.

    # DECISION plan-2026-09-01T110541-dcc1574a/D-005
    # WHAT NOT TO DO: do not widen ``groups`` to every registration site regardless of
    # package. That looks simpler and is wrong -- ``tests/test_models/test_vit/`` and
    # ``tests/test_models/test_vit_hmlp/`` each register a function named
    # ``registered_scaled_relu`` under an explicit ``package="dl_techniques_test"``.
    # Those two share a bare name and collide with each other in a DIFFERENT namespace;
    # they cannot shadow anything here, and pulling them in would make this arm fail for
    # a reason it does not test and cannot fix.
    """
    roots = _ROOTS if roots is None else roots
    src_root = REPO_ROOT if src_root is None else src_root

    claims: Dict[str, List[str]] = defaultdict(list)
    groups: Dict[str, List[Tuple[str, int, object]]] = defaultdict(list)
    refusals: List[Tuple[str, int, str]] = []
    counts = {
        "n_sites": 0,
        "n_alias_relevant": 0,
        "n_claiming": 0,
        "n_refusing": 0,
        "n_explicitly_packaged_stock": 0,
        # Blind spots -- shapes this predicate cannot key. Reported, never asserted at
        # zero: their appearing is a hole in THIS guard, not itself a defect.
        "n_aliased_decorator_import": 0,
        "n_dynamic_registration": 0,
        "n_nonliteral_legacy_alias": 0,
    }

    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(src_root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        tail = alias.name.rsplit(".", 1)[-1]
                        if (
                            tail in _REGISTRATION_DECORATORS
                            and alias.asname
                            and alias.asname != tail
                        ):
                            counts["n_aliased_decorator_import"] += 1
                    continue
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
                    if _decorator_name(node.value) == "get_custom_objects":
                        counts["n_dynamic_registration"] += 1
                    continue
                if not isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                for dec in node.decorator_list:
                    decorator = _decorator_name(dec)
                    if decorator not in _REGISTRATION_DECORATORS:
                        continue
                    counts["n_sites"] += 1

                    package = registered_name = None
                    legacy_alias = True
                    if isinstance(dec, ast.Call):
                        for i, arg in enumerate(dec.args):
                            literal = _const_str(arg)
                            if literal is None:
                                continue
                            if i == 0:
                                package = literal
                            elif i == 1 and decorator == "register_keras_serializable":
                                registered_name = literal
                        for kw in dec.keywords:
                            if kw.arg == "package":
                                package = _const_str(kw.value) or package
                            elif (
                                kw.arg == "name"
                                and decorator == "register_keras_serializable"
                            ):
                                registered_name = _const_str(kw.value) or registered_name
                            elif kw.arg == "legacy_alias":
                                if isinstance(kw.value, ast.Constant):
                                    legacy_alias = kw.value.value
                                else:
                                    legacy_alias = True
                                    counts["n_nonliteral_legacy_alias"] += 1

                    site = f"{rel}:{node.lineno}"
                    if decorator == "register_dl_technique":
                        alias_relevant = True
                    else:
                        alias_relevant = package in (None, LEGACY_ALIAS_PREFIX)
                        if not alias_relevant:
                            counts["n_explicitly_packaged_stock"] += 1

                    if not alias_relevant:
                        continue

                    counts["n_alias_relevant"] += 1
                    groups[node.name].append((rel, node.lineno, legacy_alias))
                    if decorator == "register_dl_technique" and legacy_alias is False:
                        counts["n_refusing"] += 1
                        refusals.append((rel, node.lineno, node.name))
                    else:
                        counts["n_claiming"] += 1
                        claims[registered_name or node.name].append(site)

    counts["refusals"] = sorted(refusals)
    return dict(claims), dict(groups), counts


# --------------------------------------------------------------------------- #
# tmp_path corpora for the controls. Parsed, never imported.
# --------------------------------------------------------------------------- #

#: Two objects claiming ONE legacy key. Nothing raises for this shape -- the stock
#: decorator overwrites in silence -- so only arm (a) can see it.
_INJECTED_SHADOW_SRC = '''
@keras.saving.register_keras_serializable()
class Downsample(keras.layers.Layer):
    pass


@keras.saving.register_keras_serializable()
class Downsample(keras.layers.Layer):  # noqa: F811 -- a second FILE in reality
    pass
'''

#: A refusal. Arm (b) must convict this and arm (a) must NOT: a refused alias claims
#: nothing, which is exactly why the refusal population needs its own arm.
_INJECTED_REFUSAL_SRC = '''
@register_dl_technique("dl_techniques.probe", legacy_alias=False)
class Downsample(keras.layers.Layer):
    pass
'''

#: The shape this plan retires: a duplicate bare name kept invisible by a refusal on
#: BOTH sides. Arm (a) is silent here BY CONSTRUCTION -- neither side claims the key --
#: and arm (c) is the only thing in the repository that can see it.
_INJECTED_HIDDEN_DUPLICATE_SRC = '''
@register_dl_technique("dl_techniques.probe.a", legacy_alias=False)
class Downsample(keras.layers.Layer):
    pass


@register_dl_technique("dl_techniques.probe.b", legacy_alias=False)
class Downsample(keras.layers.Layer):  # noqa: F811 -- a second FILE in reality
    pass
'''

#: The repaired twin: one side prefixed per this plan's D-001, both aliases restored.
#: All three predicates must stay SILENT. A predicate that also flags this is matching
#: on "there exists a class called Downsample", not on the property under test.
_INJECTED_FIXED_SRC = '''
@register_dl_technique("dl_techniques.probe.a")
class Downsample(keras.layers.Layer):
    pass


@register_dl_technique("dl_techniques.probe.b")
class PWFNetDownsample(keras.layers.Layer):
    pass
'''

#: The narrowing arm (c) deliberately makes. Two same-named objects under an explicit
#: NON-legacy package cannot shadow anything in this namespace, and arm (c) must not
#: convict them. This is the real shape at
#: ``tests/test_models/test_vit{,_hmlp}/test_activation_serialization.py``.
_INJECTED_OUT_OF_NAMESPACE_SRC = '''
@keras.saving.register_keras_serializable(package="dl_techniques_test")
def registered_scaled_relu(x):
    return x


@keras.saving.register_keras_serializable(package="dl_techniques_test")
def registered_scaled_relu(x):  # noqa: F811 -- a second FILE in reality
    return x
'''


def _write_fixture(tmp_path: Path, source: str) -> Tuple[tuple, Path]:
    """Write ``source`` as a fake package module; return ``(roots, src_root)``."""
    pkg = tmp_path / "probe" / "injected"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "model.py").write_text(source, encoding="utf-8")
    return (tmp_path / "probe",), tmp_path


# --------------------------------------------------------------------------- #
# Arm (e): the non-swallowing import walk.
# --------------------------------------------------------------------------- #

#: Frozen inventory measured at ``755f06a38`` (2026-09-01) by importing all 795
#: ``dl_techniques`` modules with NO exception handling. One root cause:
#: ``utils/alignment/metrics.py`` annotates with ``keras.ops.Tensor``, which does not
#: exist in Keras 3. Pre-existing and unrelated to registration; frozen rather than
#: fixed here so that a NEWCOMER -- the shape an ``AliasCollisionError`` from a bad flip
#: takes -- is convicted by name.
_MODULES_THAT_RAISE_ON_IMPORT = frozenset(
    {
        "dl_techniques.utils.alignment",
        "dl_techniques.utils.alignment.alignment",
        "dl_techniques.utils.alignment.metrics",
        "dl_techniques.utils.alignment.utils",
    }
)

#: Module count at the same commit. Anti-vacuity for the walk itself.
_MODULE_COUNT_FLOOR = int(0.8 * 795)


def _enumerate_dl_techniques_modules() -> List[str]:
    """Every importable ``dl_techniques`` module name, derived from the FILESYSTEM.

    # DECISION plan-2026-09-01T110541-dcc1574a/D-005
    # WHAT NOT TO DO: do not enumerate with ``pkgutil.walk_packages``. Measured at
    # ``755f06a38``: it reports ONE failing module where the filesystem walk reports
    # FOUR. ``walk_packages`` imports each package to descend into it, so when a
    # package's ``__init__`` raises it never enumerates that package's children --
    # they are not "swallowed", they are never attempted. An import failure therefore
    # hides its own blast radius from the very walk meant to measure it, which is the
    # second half of the S-2 defect this arm repairs.
    """
    src = REPO_ROOT / "src"
    pkg = src / "dl_techniques"
    names = set()
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = os.path.relpath(dirpath, src).replace(os.sep, ".")
        if (Path(dirpath) / "__init__.py").exists():
            names.add(rel)
        for filename in filenames:
            if filename.endswith(".py") and filename != "__init__.py":
                names.add(f"{rel}.{filename[:-3]}")
    return sorted(names)


class TestTheLegacyAliasNamespace:
    """Arms (a)-(f). Only (b), (c) and (d) are expected to fail before the fix lands."""

    def test_no_legacy_key_has_two_claimants(self):
        """(a) One legacy key, one owner -- across all four roots.

        Nothing raises for a two-claimant legacy key: the stock decorator overwrites
        without complaint and the winner is decided by import order, so the suite stays
        green while every alias-resolved load of both objects goes to the wrong class.
        This arm is the only measurement behind assumption A5 -- that the eight aliases
        this plan restores collide with nothing in ``src/train/`` or ``tests/``.
        """
        claims, _, counts = _sweep_legacy_alias_namespace()
        shadows = [
            f"{LEGACY_ALIAS_PREFIX}>{name} claimed by {sites}"
            for name, sites in sorted(claims.items())
            if len(sites) > 1
        ]
        assert not shadows, (
            "two registered objects claim one legacy alias key. No exception is raised "
            "for this -- the loser is decided by import order. PREFIX one of the class "
            "names with its package (research/2026_keras_custom_models_instructions_v2.md"
            ":216); do NOT resolve it by refusing the alias on both sides, which only "
            "hides the duplicate from this arm. Found:\n  " + "\n  ".join(shadows)
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "measured RED at 755f06a38: 8 legacy_alias=False sites (5 distinct names). "
            "Retired by plan-2026-09-01-dcc1574a steps 3-5; XPASS is the signal that "
            "they landed."
        ),
    )
    def test_no_site_refuses_the_legacy_alias(self):
        """(b) The refusal population is EXACTLY empty.

        ``legacy_alias=False`` is an escape hatch for one situation only: two registered
        objects share a bare name, so neither may own the key. That situation is resolved
        by renaming, not by refusing -- a refusal leaves the duplicate in place and blinds
        arm (a) to it. The parameter stays in ``keras_registration.py`` for a future
        genuine collision; its POPULATION being empty is a different claim, and this is
        the arm that keeps it empty.
        """
        _, _, counts = _sweep_legacy_alias_namespace()
        offenders = [f"{rel}:{lineno} {name}" for rel, lineno, name in counts["refusals"]]
        assert not offenders, (
            f"{len(offenders)} registration site(s) refuse the legacy alias. A refusal is "
            "only correct while two registered objects share the bare name; resolve the "
            "duplicate by PREFIXING one class with its package and then take the house "
            "default. Found:\n  " + "\n  ".join(offenders)
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "measured RED at 755f06a38: 3 duplicate bare names (Downsample, MLPBlock, "
            "Upsample), each hidden behind a refusal on both sides. Retired by "
            "plan-2026-09-01-dcc1574a steps 3-4; XPASS is the signal."
        ),
    )
    def test_no_two_registered_objects_share_a_bare_name(self):
        """(c) The latent duplicates -- the ones a refusal hides from arm (a).

        Arm (a) sees only what is CLAIMED. Refuse the alias on both sides of a duplicate
        and arm (a) goes quiet while the duplicate is still there, still blocking both
        classes from the house default, and still one careless flip away from an
        ``AliasCollisionError`` that takes down a whole module's import. This arm keys on
        the object name rather than on the key, so the refusal cannot hide anything from
        it.
        """
        _, groups, _ = _sweep_legacy_alias_namespace()
        duplicates = [
            f"{name}: " + ", ".join(f"{rel}:{lineno}" for rel, lineno, _ in sorted(sites))
            for name, sites in sorted(groups.items())
            if len(sites) > 1
        ]
        assert not duplicates, (
            f"{len(duplicates)} bare name(s) are defined by two registered objects in the "
            "flat legacy namespace. Both sides are then forced to refuse the alias, which "
            "is not a fix -- it is the duplicate wearing a disguise. PREFIX the narrower "
            "consumer with its package. Found:\n  " + "\n  ".join(duplicates)
        )

    #: The eight classes that carried a refusal at ``755f06a38``, by module. Each entry
    #: lists the acceptable spellings, post-rename first, so this arm reads the same
    #: before and after this plan's renames and convicts a module that lost the symbol
    #: entirely. It is a ROSTER and it rots -- which is why arm (b) above, which cannot
    #: rot, is what actually pins the population; this arm exists to assert the RUNTIME
    #: consequence that no AST census can see.
    _REFUSAL_ROSTER = (
        ("dl_techniques.layers.standard_blocks", ("ConvBlock",)),
        ("dl_techniques.layers.attention.area_attention", ("AreaAttention",)),
        ("dl_techniques.layers.ffn.mlp", ("MLPBlock",)),
        ("dl_techniques.layers.tabm_blocks", ("TabMMLPBlock", "MLPBlock")),
        (
            "dl_techniques.models.vision.image_restoration.pw_fnet.model",
            ("PWFNetDownsample", "Downsample"),
        ),
        (
            "dl_techniques.models.vision.image_restoration.pw_fnet.model",
            ("PWFNetUpsample", "Upsample"),
        ),
        ("dl_techniques.models.vision_language.ideogram4.vae", ("Downsample",)),
        ("dl_techniques.models.vision_language.ideogram4.vae", ("Upsample",)),
    )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "measured RED at 755f06a38: all 8 classes resolve their legacy alias to None. "
            "Retired by plan-2026-09-01-dcc1574a steps 3-5; XPASS is the signal."
        ),
    )
    def test_the_eight_refusing_classes_satisfy_the_registration_contract(
        self, registration_contract
    ):
        """(d) The runtime half, through the fixture built for exactly this audit.

        ``registration_contract`` is ``tests/conftest.py``'s
        ``assert_package_qualified_registration``, called here at its DEFAULT
        ``expect_legacy_alias=True``. Measured 2026-09-01, it had 16 call sites and not
        one passed that argument or named any of these eight classes -- the instrument
        existed and had never been pointed at its subject. It asserts identity
        (``is cls``), not presence: an alias resolving to a DIFFERENT object is the
        silent collision, and ``is not None`` cannot see it.
        """
        failures = []
        resolved = 0
        for module_name, candidates in self._REFUSAL_ROSTER:
            module = importlib.import_module(module_name)
            found = [name for name in candidates if hasattr(module, name)]
            assert len(found) == 1, (
                f"{module_name}: expected exactly one of {candidates} to exist, found "
                f"{found}. Either the rename half-landed or the class was deleted."
            )
            resolved += 1
            cls = getattr(module, found[0])
            try:
                registration_contract(cls)
            except AssertionError as exc:
                failures.append(f"{module_name}.{found[0]}: {exc}")
        assert resolved == len(self._REFUSAL_ROSTER), resolved
        assert not failures, (
            f"{len(failures)} of {len(self._REFUSAL_ROSTER)} classes fail the shared "
            "registration contract at its default expect_legacy_alias=True:\n  "
            + "\n  ".join(failures)
        )

    def test_no_module_fails_to_import_that_did_not_already(self):
        """(e) The import walk that does NOT swallow.

        ``tests/test_serialization_registry.py`` wraps every per-module import in
        ``except Exception: continue``. An ``AliasCollisionError`` raised inside a
        decorator kills the whole defining module's import, and that walk cannot tell
        the difference between "imported fine" and "exploded"; its ~20% population
        headroom absorbs one module's registrations without noticing. This arm freezes
        the failing set instead, so a newcomer is convicted BY NAME.
        """
        module_names = _enumerate_dl_techniques_modules()
        assert len(module_names) >= _MODULE_COUNT_FLOOR, (
            f"the filesystem walk found only {len(module_names)} dl_techniques modules "
            f"(floor {_MODULE_COUNT_FLOOR}); it stopped seeing the package"
        )
        failing = {}
        for name in module_names:
            try:
                importlib.import_module(name)
            except BaseException as exc:  # noqa: BLE001 -- the point is to catch all
                failing[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"

        newcomers = sorted(set(failing) - _MODULES_THAT_RAISE_ON_IMPORT)
        repaired = sorted(_MODULES_THAT_RAISE_ON_IMPORT - set(failing))
        assert not newcomers, (
            "module(s) that used to import now raise. If one names an AliasCollisionError "
            "then two registered objects claimed one legacy key and the losing module is "
            "gone, together with every importer of it:\n  "
            + "\n  ".join(f"{name} | {failing[name]}" for name in newcomers)
        )
        assert not repaired, (
            "these modules no longer fail to import -- good, but the frozen inventory in "
            "_MODULES_THAT_RAISE_ON_IMPORT is now wrong and must shrink to match, or this "
            f"arm quietly stops discriminating: {repaired}"
        )
        assert len(keras.saving.get_custom_objects()) > int(0.8 * 1458), (
            "the registry lost more than a fifth of its keys after importing the whole "
            f"package: {len(keras.saving.get_custom_objects())} (1458 at 755f06a38)"
        )

    def test_the_census_still_sees_the_tree(self):
        """(f) Anti-vacuity: a floor derived from the population, not set just under it."""
        claims, groups, counts = _sweep_legacy_alias_namespace()
        assert counts["n_sites"] >= _POPULATION_FLOOR, (
            f"expected ~{_POPULATION_AT_BASELINE} registration sites across the four "
            f"roots, found {counts['n_sites']}: the AST walk stopped seeing the tree "
            f"({counts})"
        )
        assert counts["n_alias_relevant"] >= int(0.8 * 772), counts
        assert counts["n_claiming"] > 0 and len(claims) > 0, counts
        assert len(groups) >= int(0.8 * 769), counts

    # The blind-spot inventory is a REPORT, not an assertion: a shape appearing is a hole
    # in this guard rather than a defect in the tree, and asserting it at zero would make
    # a growth report indistinguishable from a real collision. Under the repo-wide
    # `error::UserWarning` that report would become a failure, so this ONE test opts out.
    @pytest.mark.filterwarnings("always::UserWarning")
    def test_the_blind_spots_are_still_empty(self):
        """(f, cont.) The three shapes this census cannot key. All measured 0."""
        _, _, counts = _sweep_legacy_alias_namespace()
        blind = {
            key: counts[key]
            for key in (
                "n_aliased_decorator_import",
                "n_dynamic_registration",
                "n_nonliteral_legacy_alias",
            )
            if counts[key]
        }
        if blind:
            warnings.warn(
                "registration shapes this legacy-alias census cannot see have appeared: "
                f"{blind}. Extend _sweep_legacy_alias_namespace before trusting its zero.",
                UserWarning,
                stacklevel=2,
            )


class TestTheCensusIsNotVacuous:
    """Injected-defect controls. Every corpus lives in ``tmp_path``, never the real tree.

    Each control asserts the injection was PARSED -- an exact site count -- before it
    trusts what the predicate says about it. A control that skips that step cannot
    distinguish "the predicate fired" from "the predicate ran against an empty corpus and
    the assertion happened to hold".
    """

    def test_arm_a_fires_on_an_injected_shadow(self, tmp_path):
        roots, src_root = _write_fixture(tmp_path, _INJECTED_SHADOW_SRC)
        claims, _, counts = _sweep_legacy_alias_namespace(roots, src_root)
        assert counts["n_sites"] == 2, counts  # the injection APPLIED
        assert counts["n_claiming"] == 2, counts
        shadowed = {name: sites for name, sites in claims.items() if len(sites) > 1}
        assert list(shadowed) == ["Downsample"], claims
        assert len(shadowed["Downsample"]) == 2, shadowed

    def test_arm_b_fires_on_an_injected_refusal(self, tmp_path):
        roots, src_root = _write_fixture(tmp_path, _INJECTED_REFUSAL_SRC)
        claims, _, counts = _sweep_legacy_alias_namespace(roots, src_root)
        assert counts["n_sites"] == 1, counts  # the injection APPLIED
        assert counts["n_refusing"] == 1, counts
        assert [name for _, _, name in counts["refusals"]] == ["Downsample"]
        # ...and arm (a) is silent, because a refused alias claims nothing. This is the
        # whole reason arm (b) cannot be folded into arm (a).
        assert not [sites for sites in claims.values() if len(sites) > 1], claims

    def test_arm_c_fires_on_a_duplicate_hidden_by_refusals(self, tmp_path):
        roots, src_root = _write_fixture(tmp_path, _INJECTED_HIDDEN_DUPLICATE_SRC)
        claims, groups, counts = _sweep_legacy_alias_namespace(roots, src_root)
        assert counts["n_sites"] == 2, counts  # the injection APPLIED
        assert counts["n_refusing"] == 2, counts
        assert not [sites for sites in claims.values() if len(sites) > 1], (
            "arm (a) must be BLIND here -- neither side claims the key. If it convicts, "
            "arm (c) is redundant and this control proves nothing."
        )
        assert [name for name, sites in groups.items() if len(sites) > 1] == ["Downsample"]

    def test_all_three_arms_stay_silent_on_the_prefixed_twin(self, tmp_path):
        roots, src_root = _write_fixture(tmp_path, _INJECTED_FIXED_SRC)
        claims, groups, counts = _sweep_legacy_alias_namespace(roots, src_root)
        assert counts["n_sites"] == 2, counts  # the injection APPLIED
        assert counts["n_claiming"] == 2 and counts["n_refusing"] == 0, counts
        assert not [sites for sites in claims.values() if len(sites) > 1], claims
        assert not [name for name, sites in groups.items() if len(sites) > 1], groups

    def test_arm_c_ignores_a_duplicate_outside_the_legacy_namespace(self, tmp_path):
        """The narrowing, proven rather than asserted in prose.

        Two same-named objects under an explicit non-legacy package collide with each
        other somewhere else; they cannot shadow anything HERE, and a census that
        convicted them would be matching on the name alone.
        """
        roots, src_root = _write_fixture(tmp_path, _INJECTED_OUT_OF_NAMESPACE_SRC)
        claims, groups, counts = _sweep_legacy_alias_namespace(roots, src_root)
        assert counts["n_sites"] == 2, counts  # the injection APPLIED
        assert counts["n_explicitly_packaged_stock"] == 2, counts
        assert counts["n_alias_relevant"] == 0, counts
        assert not claims and not groups, (claims, groups)
