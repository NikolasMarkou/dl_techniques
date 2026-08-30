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
# literals, which are mutable and shared the same way.
_INSTANCE_DEFAULT_NODES = (ast.Call, ast.List, ast.Dict, ast.Set)

_PACKAGE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src" / "dl_techniques" / "layers" / "memory"
)

# The floor. Both numbers are deliberately well below the real counts (8 modules
# and 123 functions when this guard was written) so ordinary growth does not trip
# them, while a sweep that silently walked nothing still fails.
_MIN_MODULES = 5
_MIN_FUNCTIONS = 40


def _sweep():
    """
    Walk every module in the package and collect its instance-default arguments.

    :return: A tuple of (hits, module_count, function_count). Each hit is a
        string naming the file, line, enclosing function and the offending
        default expression.
    :rtype: tuple[list[str], int, int]
    """
    hits = []
    modules = 0
    functions = 0

    for path in sorted(_PACKAGE.glob("*.py")):
        modules += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))

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
                if isinstance(default, _INSTANCE_DEFAULT_NODES):
                    hits.append(
                        f"{path.name}:{default.lineno} in {node.name}() -> "
                        f"{ast.unparse(default)}"
                    )

    return hits, modules, functions


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
        _, modules, functions = _sweep()

        assert modules >= _MIN_MODULES, (
            f"the sweep walked only {modules} modules (floor {_MIN_MODULES}); "
            "a zero-hit result from this few is not evidence"
        )
        assert functions >= _MIN_FUNCTIONS, (
            f"the sweep inspected only {functions} functions "
            f"(floor {_MIN_FUNCTIONS}); a zero-hit result from this few is not "
            "evidence"
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
        hits, modules, functions = _sweep()

        assert hits == [], (
            f"{len(hits)} instance-default argument(s) found across {modules} "
            f"modules / {functions} functions. Each is shared by every caller "
            f"who omits it:\n  " + "\n  ".join(hits)
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
