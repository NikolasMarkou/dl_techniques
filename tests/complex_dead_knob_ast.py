"""Shared AST oracle for the two documented-dead knobs of ``layers/complex/``.

``ComplexLayer.epsilon`` and ``ComplexLayer.kernel_initializer`` are both public,
validated, serialized constructor arguments that NO computation reads. Two guards
pin that state -- one in ``tests/test_models/test_the_two_documented_dead_knobs.py``
(epsilon, decisions.md D-053) and one in
``tests/test_layers/test_complex/test_complex_layers.py`` (kernel_initializer,
plan-2026-09-08T070501-528ded1a decisions.md D-002).

Both guards used to parse a SINGLE module, which was sound while all seven classes
lived in one file. The one-class-per-module split of
``plan-2026-09-08T070501-528ded1a`` step 8 moved six classes out and left both
guards parsing ``base.py`` alone -- i.e. BLIND. MEASURED at ``d148888a7``: adding a
live read ``self.kernel_initializer((2, 2))`` inside ``ComplexDense.call`` left
``tests/test_layers/test_complex/`` at 159 passed and
``tests/test_models/test_the_two_documented_dead_knobs.py`` at 10 passed. The
build-time spy oracle cannot see it either -- it calls ``build()``, never a
forward pass.

The fix is to walk EVERY module of the package, which is what this module exists
to do exactly once for both callers.

Interface contract
------------------
``complex_package_modules()``
    Returns ``{module_basename: module_object}`` for every module of
    ``dl_techniques.layers.complex``, enumerated from the package's own
    ``__path__`` via :func:`pkgutil.iter_modules` -- never from a hardcoded list,
    so an eighth module added tomorrow is covered the day it lands. The empty
    ``__init__.py`` is not yielded by ``iter_modules`` and so is absent. Raises
    ``ImportError`` if a module of the package cannot be imported.

``self_attribute_sites(module, attr)``
    Returns the list of ``ast.Attribute`` nodes in ``module``'s source that read
    or write ``self.<attr>``. AST, deliberately: the DECISION comments placed at
    these very sites name the attributes, so a ``source.count("self.epsilon")``
    cannot tell a consumer from a comment about the absence of consumers (the
    first draft of the epsilon guard read 4 for exactly that reason).

``self_attribute_site_counts(attr)``
    Returns ``{module_basename: len(self_attribute_sites(...))}`` over the whole
    package. Failure mode: propagates ``ImportError``/``OSError`` from import or
    ``inspect.getsource``; it never swallows a module it could not read, because a
    silently skipped module is precisely the blindness this module was written to
    remove.

``EXPECTED_DEAD_KNOB_SITES``
    The invariant both callers assert: 2 sites in ``base.py`` (the ``__init__``
    assignment and the ``get_config`` entry) and 0 in every leaf. Built at call
    time from the enumerated module set, so it grows with the package.
"""

import ast
import inspect
import pkgutil
from types import ModuleType
from typing import Dict, List

import dl_techniques.layers.complex as _complex_package

# The one module of the package that legitimately reads either knob.
BASE_MODULE_NAME = "base"

# The two knobs pinned as documented-dead.
DEAD_KNOBS = ("epsilon", "kernel_initializer")

# The count expected in `base.py`: the `__init__` assignment (a Store) and the
# `get_config` entry (a Load). A third site means the knob has acquired a
# consumer and the ruling must be re-decided, not patched.
EXPECTED_BASE_SITES = 2


def complex_package_modules() -> Dict[str, ModuleType]:
    """Import and return every module of ``dl_techniques.layers.complex``.

    :return: ``{module_basename: module_object}``, enumerated from the filesystem.
    """
    modules: Dict[str, ModuleType] = {}
    for module_info in pkgutil.iter_modules(_complex_package.__path__):
        name = module_info.name
        modules[name] = __import__(
            f"{_complex_package.__name__}.{name}", fromlist=[name]
        )
    return modules


def self_attribute_sites(module: ModuleType, attr: str) -> List[ast.Attribute]:
    """Every ``self.<attr>`` AST node in ``module``'s source."""
    tree = ast.parse(inspect.getsource(module))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ]


def self_attribute_site_counts(attr: str) -> Dict[str, int]:
    """``self.<attr>`` site count for every module of the package."""
    return {
        name: len(self_attribute_sites(module, attr))
        for name, module in complex_package_modules().items()
    }


def expected_dead_knob_sites() -> Dict[str, int]:
    """The invariant: ``EXPECTED_BASE_SITES`` in ``base``, ``0`` in every leaf."""
    return {
        name: (EXPECTED_BASE_SITES if name == BASE_MODULE_NAME else 0)
        for name in complex_package_modules()
    }


def describe_site_counts(counts: Dict[str, int]) -> str:
    """A stable, sorted rendering of a count map for a failure message."""
    return ", ".join(f"{name}.py={count}" for name, count in sorted(counts.items()))


# The attribute used as the liveness probe below. `self.strides` is read by both
# shape-owning leaves and by NO code in `base.py`, which is the exact opposite of
# the dead knobs' profile -- so a scan that reads it correctly cannot be one that
# is silently returning zeros.
LIVENESS_PROBE_ATTRIBUTE = "strides"
LIVENESS_PROBE_MODULES = ("complex_conv2d", "complex_average_pooling2d")

# Every module the package is known to contain today. Used ONLY as a floor for
# the liveness assertion (`>=`), never as the enumeration itself -- the counts
# always come from `pkgutil`, so an eighth module is scanned the day it lands.
KNOWN_MODULE_FLOOR = frozenset({
    "base",
    "complex_conv2d",
    "complex_dense",
    "complex_relu",
    "complex_average_pooling2d",
    "complex_dropout",
    "complex_global_average_pooling2d",
})


def assert_scan_reaches_the_leaves() -> None:
    """LIVENESS for the two dead-knob guards — without it, "0 in every leaf" is unfalsifiable.

    A scan that silently failed to reach the leaves would report ``0`` for each of
    them and both guards would pass forever — which is precisely the state
    MEASURED at ``d148888a7``, where both guards parsed ``base.py`` alone. Two
    things are asserted here that the failure mode cannot satisfy: the enumerated
    module set covers every module the package is known to hold, and an attribute
    that IS live in the leaves reads a positive count there.

    :raises AssertionError: if the package scan is blind to the leaf modules.
    """
    modules = complex_package_modules()
    assert KNOWN_MODULE_FLOOR <= set(modules), (
        f"the package scan enumerated only {sorted(modules)}, which is missing "
        f"{sorted(KNOWN_MODULE_FLOOR - set(modules))} — the dead-knob guards are "
        "blind to whatever it cannot see"
    )

    live = self_attribute_site_counts(LIVENESS_PROBE_ATTRIBUTE)
    blind = [name for name in LIVENESS_PROBE_MODULES if live.get(name, 0) <= 0]
    assert not blind, (
        f"`self.{LIVENESS_PROBE_ATTRIBUTE}` — an attribute that is demonstrably "
        f"live in {list(LIVENESS_PROBE_MODULES)} — read "
        f"[{describe_site_counts(live)}]; the AST scan is not reaching {blind}, "
        "so the dead-knob guards' '0 in every leaf' is vacuous"
    )
    assert live[BASE_MODULE_NAME] == 0, (
        f"`self.{LIVENESS_PROBE_ATTRIBUTE}` is read in {BASE_MODULE_NAME}.py, so "
        "it is no longer a leaf-only probe and this liveness arm has lost its "
        "contrast — pick another attribute"
    )
