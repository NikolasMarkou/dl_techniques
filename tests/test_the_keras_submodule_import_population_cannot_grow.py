"""Ratchet: the ``from keras import ...`` population under ``src/`` and ``tests/`` cannot grow.

The house rule is ``import keras`` and a qualified call site (``keras.ops.*``,
``keras.layers.*``). Two statement kinds break it, both found by an AST walk
(not a text grep, so docstrings and comments are never counted):

* ``from keras import X`` / ``from keras.sub import X`` (absolute only);
* ``import keras.sub`` (``import keras`` itself is the allowed form).

Measured at Step 12 of ``plan-2026-10-14T062616-81938e81``, after
``layers/pooling/pixel_shuffle.py``, ``layers/conv_blocks/universal_inverted_bottleneck.py``
and ``tests/test_train/test_cliffordnet/test_train_clip_textlm.py`` were
converted: **543 statements in 521 files**. The rest are not migrated in bulk:
the user stopped a mass migration of these files on 2026-08-25 ("just fix what
i broke, no need to change all those files"); a file is converted when it is
edited for another reason (decisions.md D-082).

So this is a ceiling, not a ban. Converting a file lowers the measured value;
lower the ceilings to match in the same commit. A new ``from keras import``
anywhere raises the count and fails here.

Re-measured at Step 2.6 of the same plan (D-162), after the user's
``timestral`` port (6d757c6a6, 10 statements in 8 files) was converted:
**538 statements in 516 files**. The old ceilings (543 / 521) were 5 above the
population, so the port's 10 only showed up as 548 > 543.
"""

import ast
import pathlib
import warnings

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("src", "tests")

#: Measured ceilings (statements, files). Lower them when a file is converted.
STATEMENT_CEILING = 538
FILE_CEILING = 516

#: Files converted by name; they must stay clean whatever the totals do.
CONVERTED = (
    "src/dl_techniques/layers/pooling/pixel_shuffle.py",
    "src/dl_techniques/layers/conv_blocks/universal_inverted_bottleneck.py",
    "tests/test_train/test_cliffordnet/test_train_clip_textlm.py",
    "src/dl_techniques/models/time_series/timestral/covariate_encoding.py",
    "src/dl_techniques/models/time_series/timestral/forecaster.py",
    "src/dl_techniques/models/time_series/timestral/layers.py",
    "src/dl_techniques/models/time_series/timestral/loading.py",
    "src/dl_techniques/models/time_series/timestral/masks.py",
    "src/dl_techniques/models/time_series/timestral/model.py",
    "src/dl_techniques/models/time_series/timestral/scaler.py",
    "tests/test_models/test_timestral/test_forecaster.py",
)


def keras_submodule_imports(tree: ast.AST) -> int:
    """Count the statements in ``tree`` that bind a keras submodule name.

    :param tree: A parsed module.
    :return: Number of ``from keras[.x] import ...`` (absolute) and
        ``import keras.x`` statements.
    """
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "keras" or module.startswith("keras.")):
                count += 1
        elif isinstance(node, ast.Import):
            if any(alias.name.startswith("keras.") for alias in node.names):
                count += 1
    return count


def _census():
    per_file = {}
    for root in ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            with warnings.catch_warnings():
                # Invalid escapes in other files' strings are not this census's business.
                warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
                tree = ast.parse(path.read_text(encoding="utf-8"))
            n = keras_submodule_imports(tree)
            if n:
                per_file[path.relative_to(REPO_ROOT).as_posix()] = n
    return per_file


@pytest.fixture(scope="module")
def census():
    return _census()


def test_the_population_does_not_grow(census):
    statements = sum(census.values())
    assert statements <= STATEMENT_CEILING, (
        f"{statements} keras-submodule import statements (ceiling {STATEMENT_CEILING}). "
        "Use `import keras` and qualify the call site (keras.ops.*, keras.layers.*)."
    )
    assert len(census) <= FILE_CEILING, (
        f"{len(census)} files with keras-submodule imports (ceiling {FILE_CEILING})."
    )


def test_the_converted_files_stay_converted(census):
    regressed = [path for path in CONVERTED if path in census]
    assert not regressed, f"converted files import a keras submodule again: {regressed}"
    for path in CONVERTED:
        assert (REPO_ROOT / path).is_file(), f"converted file moved or deleted: {path}"


@pytest.mark.parametrize("source, expected", [
    ("import keras\n", 0),
    ("from keras import ops\n", 1),
    ("from keras import ops, layers\n", 1),
    ("import keras.ops\n", 1),
    ("from keras.src import backend\n", 1),
    ("def f():\n    from keras import ops\n", 1),
    ('"""from keras import ops"""\n', 0),
    ("# from keras import ops\n", 0),
    ("from .keras import ops\n", 0),
])
def test_the_counter_on_planted_cases(source, expected):
    assert keras_submodule_imports(ast.parse(source)) == expected
