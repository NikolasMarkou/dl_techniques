"""Step 9 -- the curated package surface, and a repo-wide registry sweep.

Two things are pinned here, both of which drift SILENTLY.

**The export surface.** ``models/vision_language/sam/sam2/__init__.py`` exports exactly three names
(S-2, mirroring ``models/vision_language/sam/sam1/__init__.py``). A surface widens one convenience
re-export at a time, and no other test in this suite would notice: every
component test imports from its own submodule, so the package init could export
all fifteen classes and stay green. :class:`TestExportSurface` asserts the exact
set, in both directions -- nothing missing, nothing extra.

**Registered-key uniqueness across the WHOLE repository.** ``test_model.py``'s
G8.5 probe asks a narrower question: were the SAM 2 keys absent from a fresh
interpreter's registry BEFORE ``dl_techniques.models.vision_language.sam.sam2.model`` was imported.
That catches a collision with anything on that module's own import path, which
includes SAM 1. It CANNOT see a collision with a module neither of them imports
-- some third package registering ``Custom>Hiera`` would only ever collide in a
process that happened to import both, and a bare
``@keras.saving.register_keras_serializable()`` OVERWRITES the earlier entry
without raising. :class:`TestRepoWideRegistryUniqueness` closes that gap
statically: it walks every ``.py`` file under ``src/`` with :mod:`ast`, derives
each decorated class's registry key the same way Keras does, and asserts no key
is claimed twice anywhere. Static rather than by import because importing all
~750 registering modules in one process is neither cheap nor side-effect-free.
"""

import ast
import collections
import importlib
import pathlib
from typing import Dict, List, Optional

import pytest

# The PACKAGE. It needs no alias: the subpackage is spelled ``sam2`` and the
# class on the next line ``SAM2``, and Python is case-sensitive, so the two
# names coexist. Before the 2026-08-24 restructure the subpackage was ``SAM2/``
# and this line read ``import SAM2 as sam2`` to keep them apart.
from dl_techniques.models.vision_language.sam import sam2
from dl_techniques.models.vision_language.sam.sam2 import SAM2, SAM2MemoryBank, create_sam2

# ---------------------------------------------------------------------
# The pinned surface. Changing this tuple is the deliberate act; the tests
# below make sure it cannot happen by accident.
# ---------------------------------------------------------------------

EXPECTED_ALL = ("SAM2", "SAM2MemoryBank", "create_sam2")

#: Public classes that stay behind their submodules (S-6). Not exhaustive of
#: the package, deliberately -- these are the ones a "helpful" re-export would
#: most plausibly reach for.
SUBMODULE_ONLY = (
    "Hiera",
    "HieraBlock",
    "HieraMultiScaleAttention",
    "HieraPatchEmbed",
    "SAM2FpnNeck",
    "SAM2Fuser",
    "SAM2ImageEncoder",
    "SAM2MaskDecoder",
    "SAM2MaskDownSampler",
    "SAM2MemoryAttention",
    "SAM2MemoryAttentionLayer",
    "SAM2MemoryEncoder",
)

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3] / "src"


class TestExportSurface:
    """S-2: exactly three exported names, mirroring ``models/vision_language/sam/sam1``."""

    def test_all_is_exactly_the_curated_three(self) -> None:
        assert tuple(sorted(sam2.__all__)) == EXPECTED_ALL, (
            f"the SAM 2 export surface changed: {sorted(sam2.__all__)!r} "
            f"!= {list(EXPECTED_ALL)!r}. Widening it is allowed but must be "
            f"deliberate -- update this test in the same commit."
        )

    def test_every_exported_name_resolves_to_the_right_object(self) -> None:
        from dl_techniques.models.vision_language.sam.sam2.memory_bank import (
            SAM2MemoryBank as BankFromSubmodule,
        )
        from dl_techniques.models.vision_language.sam.sam2.model import SAM2 as ModelFromSubmodule
        from dl_techniques.models.vision_language.sam.sam2.model import (
            create_sam2 as FactoryFromSubmodule,
        )

        assert SAM2 is ModelFromSubmodule
        assert SAM2MemoryBank is BankFromSubmodule
        assert create_sam2 is FactoryFromSubmodule

    def test_star_import_binds_exactly_the_curated_names(self) -> None:
        """``import *`` is the surface a user actually gets."""
        namespace: Dict[str, object] = {}
        exec("from dl_techniques.models.vision_language.sam.sam2 import *", namespace)  # noqa: S102
        bound = {k for k in namespace if not k.startswith("__")}
        assert bound == set(EXPECTED_ALL), (
            f"`import *` bound {sorted(bound)!r}, expected "
            f"{list(EXPECTED_ALL)!r}"
        )

    @pytest.mark.parametrize("name", SUBMODULE_ONLY)
    def test_implementation_classes_are_not_re_exported(
            self, name: str) -> None:
        """S-6: components are imported from their submodule, not the package.

        A bare ``hasattr`` on the package is not the assertion -- importing a
        submodule binds its NAME on the package object, so ``sam2.hiera`` is
        legitimately present. The question is whether the CLASS was hoisted.
        """
        assert name not in sam2.__all__
        hoisted = getattr(sam2, name, None)
        assert hoisted is None, (
            f"{name!r} is re-exported from the package init; S-2 says it "
            f"stays behind its submodule"
        )

    def test_the_surface_mirrors_sam_one_in_shape(self) -> None:
        """S-2 is a claim about SHAPE: model + factory-ish + one helper.

        Compared structurally, not by name: SAM 1 exports 3 names too, and if
        someone broadens SAM 1's surface this test says so rather than letting
        SAM 2 quietly follow.
        """
        sam1 = importlib.import_module("dl_techniques.models.vision_language.sam.sam1")
        assert len(sam1.__all__) == len(sam2.__all__) == 3


# ---------------------------------------------------------------------
# Repo-wide registered-key uniqueness (static AST sweep)
# ---------------------------------------------------------------------


#: The two decorator spellings that register a class in this tree, and the
#: positional slot each takes its package string from. ``register_dl_technique``
#: (``src/dl_techniques/utils/keras_registration.py``) wraps the Keras decorator
#: and REQUIRES its package positionally; the stock decorator defaults to
#: ``"Custom"``. A sweep that knows only the stock name went blind to 744 of the
#: tree's 744 sites the day the migration landed, and reported ZERO collisions
#: while seeing nothing at all -- so both names are keyed here, together.
_REGISTRATION_DECORATORS = ("register_keras_serializable", "register_dl_technique")


def _registered_key(node: ast.ClassDef) -> Optional[str]:
    """Derive the Keras registry key for a decorated class.

    Mirrors the rule the decorators themselves apply: the key is
    ``f"{package}>{name}"``. For the stock
    ``keras.saving.register_keras_serializable`` ``package`` defaults to
    ``"Custom"`` and ``name`` to the class's own name, and both may be given
    positionally or by keyword. For ``register_dl_technique`` the package is
    mandatory and positional and there is no ``name`` argument, so the key is
    always ``f"{package}>{node.name}"``.

    Note this returns the QUALIFIED key only. The legacy ``Custom>`` alias that
    ``register_dl_technique`` also binds is deliberately not returned: it is a
    second key for the SAME object, so counting it as a claimant would make every
    aliased class read as its own collision.

    :param node: The class definition to inspect.
    :type node: ast.ClassDef
    :return: The registry key, or ``None`` if the class is not registered.
    :rtype: Optional[str]
    """
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        target = call.func if call else decorator
        attr = getattr(target, "attr", getattr(target, "id", None))
        if attr not in _REGISTRATION_DECORATORS:
            continue
        package = "Custom" if attr == "register_keras_serializable" else None
        name = node.name
        if call is not None:
            if len(call.args) >= 1 and isinstance(call.args[0], ast.Constant):
                package = call.args[0].value
            if (attr == "register_keras_serializable"
                    and len(call.args) >= 2
                    and isinstance(call.args[1], ast.Constant)):
                name = call.args[1].value
            for keyword in call.keywords:
                if not isinstance(keyword.value, ast.Constant):
                    continue
                if keyword.arg == "package":
                    package = keyword.value.value
                elif keyword.arg == "name" and attr == "register_keras_serializable":
                    name = keyword.value.value
        if package is None:  # a non-literal package string; unkeyable, not absent
            continue
        return f"{package}>{name}"
    return None


def _sweep_registered_keys() -> Dict[str, List[str]]:
    """Map every registry key under ``src/`` to the sites that claim it."""
    keys: Dict[str, List[str]] = collections.defaultdict(list)
    for path in sorted(SRC_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                key = _registered_key(node)
                if key is not None:
                    keys[key].append(
                        f"{path.relative_to(SRC_ROOT)}:{node.lineno}")
    return keys


#: The 14 registered SAM 2 classes (``SAM2MemoryBank`` is deliberately NOT one
#: -- it is a plain-Python state container that owns no weights, see D-026).
#:
#: These were spelled ``Custom>AxialRoPE2D`` and so on until the 2026-08-29
#: registration migration. The package prefix is derived here
#: rather than pasted onto each of the fourteen: SAM 2 lives under
#: ``models/vision_language/sam/sam2/``, and the key strips the ``vision_language``
#: family and ``sam`` subfamily containers because those are a filing decision
#: that has already been reshuffled once. Deriving it means a fifteenth class
#: cannot be added with a subtly different prefix.
SAM2_KEY_PREFIX = "dl_techniques.models.sam2"
SAM2_REGISTERED_KEYS = (
    # NOT under the SAM 2 prefix: `AxialRoPE2D` is a shared embedding layer that
    # SAM 2 consumes, and it was ALREADY shared before the migration. Deriving
    # its key from SAM 2's prefix would have silently invented a key no class
    # claims, and the test below would then have read as a missing registration.
    "dl_techniques.layers.embedding.axial_rope_2d>AxialRoPE2D",
) + tuple(
    f"{SAM2_KEY_PREFIX}.{module}>{name}" for module, name in (
        ("hiera", "Hiera"),
        ("hiera", "HieraBlock"),
        ("hiera", "HieraMultiScaleAttention"),
        ("hiera", "HieraPatchEmbed"),
        ("model", "SAM2"),
        ("neck", "SAM2FpnNeck"),
        ("memory_encoder", "SAM2Fuser"),
        ("neck", "SAM2ImageEncoder"),
        ("mask_decoder", "SAM2MaskDecoder"),
        ("memory_encoder", "SAM2MaskDownSampler"),
        ("memory_attention", "SAM2MemoryAttention"),
        ("memory_attention", "SAM2MemoryAttentionLayer"),
        ("memory_encoder", "SAM2MemoryEncoder"),
    )
)


class TestRepoWideRegistryUniqueness:
    """Registry keys must be unique across the WHOLE of ``src/``.

    Wider than ``test_model.py``'s G8.5, which can only see collisions on
    ``models/vision_language/sam/sam2/model.py``'s own import path.
    """

    @pytest.fixture(scope="class")
    def keys(self) -> Dict[str, List[str]]:
        return _sweep_registered_keys()

    def test_the_sweep_actually_found_the_repo(
            self, keys: Dict[str, List[str]]) -> None:
        """Fixture-validity: a sweep that finds nothing is vacuously unique."""
        assert SRC_ROOT.is_dir(), f"src root not found at {SRC_ROOT}"
        total = sum(len(v) for v in keys.values())
        assert total > 500, (
            f"only {total} registered classes found under {SRC_ROOT} -- the "
            f"sweep is not reaching the repository, so its uniqueness result "
            f"means nothing"
        )

    def test_no_registry_key_is_claimed_twice_anywhere(
            self, keys: Dict[str, List[str]]) -> None:
        duplicates = {k: v for k, v in keys.items() if len(v) > 1}
        assert not duplicates, (
            f"{len(duplicates)} registry key(s) claimed more than once -- a "
            f"bare @register_keras_serializable() OVERWRITES silently, which "
            f"breaks checkpoint loading for whichever class loses: "
            f"{duplicates}"
        )

    @pytest.mark.parametrize("key", SAM2_REGISTERED_KEYS)
    def test_each_sam2_key_is_claimed_exactly_once(
            self, key: str, keys: Dict[str, List[str]]) -> None:
        sites = keys.get(key, [])
        assert len(sites) == 1, (
            f"{key!r} is claimed by {len(sites)} site(s): {sites}"
        )

    def test_the_memory_bank_is_deliberately_unregistered(
            self, keys: Dict[str, List[str]]) -> None:
        """D-026/step 6: the bank owns no weights, so it has no registry key.

        Registering it would mint public surface for a class that never
        appears in a serialized config.
        """
        claimants = [k for k in keys if k.endswith(">SAM2MemoryBank")]
        assert not claimants, claimants
