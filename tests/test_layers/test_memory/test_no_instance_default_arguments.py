"""
Package-wide guard against instance-default arguments in `layers/memory/`.

A parameter whose default is a live object -- `def f(x=Thing())` -- evaluates
that object ONCE, when the module is imported. Every caller who omits the
argument then shares it. For a Keras initializer or regularizer that means
every layer built at defaults draws from the same object, which is the v2
guide's section 1.4 anti-pattern and the same family as the shared-initializer
defects already fixed on the kernel side (D-068) and the bias side (C-15).

Three sites were fixed under D-019: `NeuroGrid.grid_initializer`,
`NeuroGrid.grid_regularizer` and `SoftSOMLayer.kernel_regularizer`. This guard
exists so a fourth cannot be added silently. It sweeps by MECHANISM over every
module in the package rather than checking the three known line numbers, because
a defect list scoped to the sites someone already found cannot find a new one.

The sweep carries an anti-vacuity floor. A census that walks zero modules, or
inspects zero functions, reports zero hits for the wrong reason and would pass
while testing nothing.
"""

import ast
import pathlib

import pytest

# Node types that construct or allocate when the `def` statement executes.
# A `Call` covers `Thing()`; the containers cover `[]`, `{}` and `set()`
# literals plus comprehensions and `[1] * 3`, which are mutable and shared the
# same way; a `Lambda` allocates a new function object per `def` execution.
_INSTANCE_DEFAULT_NODES = (
    ast.Call,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.BinOp,
    ast.Lambda,
)

# A default that is a bare NAME hides whatever that name is bound to. The two
# fixed layers now use exactly this shape -- `x=_KERNEL_REGULARIZER_DEFAULT` --
# so it is the house pattern and therefore the MOST LIKELY shape of a fourth
# defect. Resolving the name is what stops this guard from being blind to the
# very thing it exists to prevent. Only these bindings are safe as a default.
_SAFE_SENTINEL_FACTORIES = frozenset({"object"})

_PACKAGE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src" / "dl_techniques" / "layers" / "memory"
)

# The floor. Both numbers are deliberately well below the real counts (8 modules
# and 123 functions when this guard was written) so ordinary growth does not trip
# them, while a sweep that silently walked nothing still fails.
_MIN_MODULES = 5
_MIN_FUNCTIONS = 40


def _module_level_bindings(tree):
    """
    Map every module-level name to the expression it is bound to.

    Only module-level assignments are collected. A name bound more than once
    keeps its LAST binding, which is what a `def` executed at import time sees.

    :param tree: The parsed module.
    :type tree: ast.Module
    :return: Mapping of name to the bound expression node.
    :rtype: dict[str, ast.AST]
    """
    bindings = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                bindings[node.target.id] = node.value
    return bindings


def _resolve(default, bindings):
    """
    Resolve a default expression to the node that actually allocates.

    A bare name default is followed to its module-level binding, because
    `_D = Thing()` followed by `def f(x=_D)` shares one object exactly as
    `def f(x=Thing())` does. The indirection changes nothing at runtime.

    :param default: The default expression from the signature.
    :type default: ast.AST
    :param bindings: Module-level name bindings.
    :type bindings: dict[str, ast.AST]
    :return: A tuple of (node_to_judge, how_it_was_reached).
    :rtype: tuple[ast.AST, str]
    """
    if isinstance(default, ast.Name) and default.id in bindings:
        return bindings[default.id], f"via module constant {default.id}"
    return default, "directly"


def _is_safe_sentinel(node):
    """
    Is this node an allocation that is safe to share as a default?

    `object()` is safe: it is an opaque identity marker carrying no state, which
    is the whole point of a sentinel. Anything else that constructs is not.

    :param node: The node the default resolves to.
    :type node: ast.AST
    :return: True when the node is a bare `object()` call.
    :rtype: bool
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SAFE_SENTINEL_FACTORIES
        and not node.args
        and not node.keywords
    )


def _sweep():
    """
    Walk every module in the package and collect its instance-default arguments.

    Recursive, so a future subpackage cannot hide from it.

    :return: A tuple of (hits, module_count, function_count, module_names).
        Each hit names the file, line, enclosing function, how the default was
        reached, and the offending expression.
    :rtype: tuple[list[str], int, int, set[str]]
    """
    hits = []
    modules = 0
    functions = 0
    seen = set()

    for path in sorted(_PACKAGE.rglob("*.py")):
        modules += 1
        seen.add(path.name)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bindings = _module_level_bindings(tree)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            functions += 1

            # `kw_defaults` holds None for keyword-only args that have no
            # default at all; those are not defaults and must not be inspected.
            defaults = list(node.args.defaults) + [
                d for d in node.args.kw_defaults if d is not None
            ]

            for default in defaults:
                resolved, how = _resolve(default, bindings)
                if not isinstance(resolved, _INSTANCE_DEFAULT_NODES):
                    continue
                if _is_safe_sentinel(resolved):
                    continue
                hits.append(
                    f"{path.name}:{default.lineno} in {node.name}() {how} -> "
                    f"{ast.unparse(resolved)}"
                )

    return hits, modules, functions, seen


class TestNoInstanceDefaultArguments:
    """
    Pins the package at zero instance-default arguments.

    The sweep is the guard. It is deliberately not parametrized over the three
    known sites: a guard that names the sites it was written for cannot catch
    the fourth one.
    """

    def test_the_package_directory_resolves(self):
        """
        Anti-vacuity: the sweep must be pointed at real source.

        A wrong relative path yields an empty glob and therefore zero hits, which
        would read as a pass.

        :raises AssertionError: If the package directory does not exist.
        """
        assert _PACKAGE.is_dir(), (
            f"the sweep is pointed at {_PACKAGE}, which does not exist; "
            "a zero-hit result from this path would be meaningless"
        )

    def test_the_sweep_inspects_a_credible_amount_of_code(self):
        """
        Anti-vacuity: the sweep must actually walk modules and functions.

        :raises AssertionError: If fewer than the floor's modules or functions
            were inspected.
        """
        _, modules, functions, _seen = _sweep()

        assert modules >= _MIN_MODULES, (
            f"the sweep walked only {modules} modules (floor {_MIN_MODULES}); "
            "a zero-hit result from this few is not evidence"
        )
        assert functions >= _MIN_FUNCTIONS, (
            f"the sweep inspected only {functions} functions "
            f"(floor {_MIN_FUNCTIONS}); a zero-hit result from this few is not "
            "evidence"
        )

        # Walking SOME real directory is not the same as walking the RIGHT one.
        # These two files are where the fixed defects lived; if the sweep is not
        # reading them it is not guarding anything that matters.
        for required in ("neuro_grid.py", "som_nd_soft_layer.py"):
            assert required in _seen, (
                f"{required} was not in the swept set {sorted(_seen)}; the sweep "
                "is pointed somewhere real but wrong"
            )

    def test_no_parameter_defaults_to_a_constructed_object(self):
        """
        The guard itself. No parameter anywhere in the package may default to a
        constructed object or a mutable literal.

        If this fails, do not silence it by moving the default into the
        docstring. Take a sentinel instead, and resolve it inside `__init__`
        BEFORE any `keras.*.get(...)` normalisation -- `get(None)` returns
        `None` and would drop the object entirely. See decisions.md D-019.

        :raises AssertionError: If any instance-default argument is found.
        """
        hits, modules, functions, _seen = _sweep()

        assert hits == [], (
            f"{len(hits)} instance-default argument(s) found across {modules} "
            f"modules / {functions} functions. Each is shared by every caller "
            f"who omits it:\n  " + "\n  ".join(hits)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
