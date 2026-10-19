"""Census: TensorFlow's TF32 flag is written only by ``tests/numerics.py``.

``tf.config.experimental.enable_tensor_float_32_execution`` is PROCESS-GLOBAL. Until
plan-2026-10-14T062616-81938e81 step 2.6 (D-164) the suite wrote it from 14 files, each
with its own capture / restore / assert harness, and the copies had drifted: one restored
without a ``finally`` and never asserted the restoration, one asserted ``is`` against the
captured value. Every scope now routes through ``tests/numerics.py::tf32_execution``
(``true_float32_matmul`` is its "off" form), and this census keeps it that way.

The census is an AST walk, so a docstring or comment that NAMES the function is not a
write; any attribute or name reference to it outside ``tests/numerics.py`` is.
"""

import ast
import pathlib
import warnings

import pytest

from tests.numerics import (
    TF32_UNIT_ROUNDOFF,
    matmul_unit_roundoff,
    tf32_execution,
    true_float32_matmul,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
THE_WRITER = "tests/numerics.py"
FLAG_SETTER = "enable_tensor_float_32_execution"
FLOAT32_UNIT_ROUNDOFF = 2.0 ** -24


def flag_setter_references(tree: ast.AST) -> list:
    """Line numbers of every reference to the TF32 flag setter in ``tree``.

    :param tree: A parsed module.
    :return: Sorted line numbers of ``<x>.enable_tensor_float_32_execution`` attributes
        and bare ``enable_tensor_float_32_execution`` names (calls or not).
    """
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == FLAG_SETTER:
            lines.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == FLAG_SETTER:
            lines.append(node.lineno)
        elif isinstance(node, ast.alias) and node.name == FLAG_SETTER:
            lines.append(node.lineno)
    return sorted(lines)


def _census() -> dict:
    found = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
            tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = flag_setter_references(tree)
        if lines:
            found[path.relative_to(REPO_ROOT).as_posix()] = lines
    return found


def test_the_tf32_flag_is_set_only_in_numerics():
    census = _census()
    elsewhere = {path: lines for path, lines in census.items() if path != THE_WRITER}
    assert not elsewhere, (
        f"`{FLAG_SETTER}` is referenced outside {THE_WRITER}: {elsewhere}. Use "
        "`tests.numerics.true_float32_matmul()` (TF32 off) or "
        "`tests.numerics.tf32_execution(enabled)` instead (D-164)."
    )
    # Liveness: the census must see the one writer, or it is reading nothing.
    assert census.get(THE_WRITER), f"the census no longer sees {THE_WRITER}'s writer"


@pytest.mark.parametrize("source, expected", [
    ("import tensorflow as tf\ntf.config.experimental.enable_tensor_float_32_execution(False)\n", [2]),
    ("def f(tf):\n    setter = tf.config.experimental.enable_tensor_float_32_execution\n", [2]),
    ("from tensorflow.config.experimental import enable_tensor_float_32_execution\n", [1]),
    ('"""tf.config.experimental.enable_tensor_float_32_execution(False)"""\n', []),
    ("# tf.config.experimental.enable_tensor_float_32_execution(False)\n", []),
    ("tf.config.experimental.tensor_float_32_execution_enabled()\n", []),
])
def test_the_census_reads_writes_not_prose(source, expected):
    assert flag_setter_references(ast.parse(source)) == expected


def test_true_float32_matmul_runs_float32_and_restores():
    """Inside the scope the measured matmul precision is float32, on any device."""
    import tensorflow as tf

    before = tf.config.experimental.tensor_float_32_execution_enabled()
    with true_float32_matmul():
        assert tf.config.experimental.tensor_float_32_execution_enabled() is False
        assert matmul_unit_roundoff() == FLOAT32_UNIT_ROUNDOFF
    assert tf.config.experimental.tensor_float_32_execution_enabled() == before


def test_tf32_execution_on_is_tf32_on_a_tensor_core_gpu():
    """The "on" arm is live where TF32 exists: the probe reads the TF32 roundoff."""
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("TF32 is a GPU execution mode; the flag is numerically inert on CPU")
    before = tf.config.experimental.tensor_float_32_execution_enabled()
    with tf32_execution(True) as enabled:
        assert enabled is True
        assert matmul_unit_roundoff() == TF32_UNIT_ROUNDOFF
    assert tf.config.experimental.tensor_float_32_execution_enabled() == before


def test_the_scope_restores_after_a_raise():
    import tensorflow as tf

    before = tf.config.experimental.tensor_float_32_execution_enabled()
    with pytest.raises(RuntimeError, match="inside"):
        with tf32_execution(not before):
            raise RuntimeError("inside")
    assert tf.config.experimental.tensor_float_32_execution_enabled() == before
