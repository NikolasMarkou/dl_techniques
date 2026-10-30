"""Source rules for ``src/train/__main__.py`` and ``src/train/api/**/*.py``.

Several census guards under ``tests/test_train/`` either cannot see the new
package or sit beside a pinned count that is already red at HEAD
(``test_the_census_counts_are_pinned``, exempt 48 != 50, which Step 1 of the
plan deselects by exact id). This file therefore carries the checks for the
API on its own, by AST:

1. no bare, ``Exception`` or ``BaseException`` handler;
2. no ``except KeyboardInterrupt``;
3. no ``if __name__ ...`` test (an entry-point block would make
   ``train.__main__`` a seed-census entry point that declares no ``--seed``);
4. every handler in ``api/cli.py`` names exactly ``DLTrainError`` (D-009);
5. no module-scope import of ``tensorflow``, ``keras``, ``train.common``,
   ``dl_techniques.models`` or ``yaml`` (D-012; the help path stays light);
6. no ``print(`` call (logging goes through the logger, output through ``sys.stdout.write``);
7. no write of ``CUDA_VISIBLE_DEVICES``.

Each rule is a function over a source text. A planted snippet proves the
rule fires, a clean snippet proves it stays quiet, and the sweep applies it to
the real files, with a floor on how many files it saw.
"""

import ast
from pathlib import Path
from typing import Iterator, List, Set

import pytest

SRC_TRAIN = Path(__file__).resolve().parents[3] / "src" / "train"
MAIN_FILE = SRC_TRAIN / "__main__.py"
API_DIR = SRC_TRAIN / "api"

#: ``__main__.py`` plus the seven modules Phase 1 ships (``__init__``, ``args``,
#: ``cli``, ``config``, ``errors``, ``registry``, ``run_dir``). Later phases only add files.
MIN_FILES = 8

BROAD = ("Exception", "BaseException")
HEAVY = ("tensorflow", "keras", "train.common", "dl_techniques.models", "yaml")
GPU_VARIABLE = "CUDA_VISIBLE_DEVICES"


def api_files() -> List[Path]:
    """Return ``__main__.py`` and every module under ``api/``, sorted."""
    found = [MAIN_FILE] + sorted(
        path for path in API_DIR.rglob("*.py") if "__pycache__" not in path.parts
    )
    return found


def package_of(path: Path) -> str:
    """Return the dotted package a module lives in, for relative imports."""
    relative = path.relative_to(SRC_TRAIN.parent).with_suffix("")
    return ".".join(relative.parts[:-1])


def _names(node: ast.AST) -> List[str]:
    """Return the exception names an ``except`` clause lists (dotted tails kept)."""
    if node is None:
        return []
    if isinstance(node, ast.Tuple):
        return [name for element in node.elts for name in _names(element)]
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    return [ast.unparse(node)]


def handlers(tree: ast.AST) -> Iterator[ast.ExceptHandler]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            yield node


# ---------------------------------------------------------------------
# Rules: each takes (source, path) and returns a list of problem strings.
# ---------------------------------------------------------------------


def rule_no_broad_handler(tree: ast.AST, path: Path) -> List[str]:
    problems = []
    for handler in handlers(tree):
        if handler.type is None:
            problems.append(f"{path.name}:{handler.lineno}: bare except")
        elif any(name in BROAD for name in _names(handler.type)):
            problems.append(f"{path.name}:{handler.lineno}: broad handler {ast.unparse(handler.type)}")
    return problems


def rule_no_keyboard_interrupt_handler(tree: ast.AST, path: Path) -> List[str]:
    return [
        f"{path.name}:{handler.lineno}: except KeyboardInterrupt"
        for handler in handlers(tree)
        if "KeyboardInterrupt" in _names(handler.type)
    ]


def _mentions_dunder_name(node: ast.AST) -> bool:
    return any(isinstance(inner, ast.Name) and inner.id == "__name__" for inner in ast.walk(node))


def rule_no_entry_point_block(tree: ast.AST, path: Path) -> List[str]:
    return [
        f"{path.name}:{node.lineno}: if __name__ ..."
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and _mentions_dunder_name(node.test)
    ]


def rule_cli_handlers_name_dltrainerror(tree: ast.AST, path: Path) -> List[str]:
    if path.name != "cli.py":
        return []
    problems = []
    for handler in handlers(tree):
        exact = isinstance(handler.type, ast.Name) and handler.type.id == "DLTrainError"
        if not exact:
            shown = "bare" if handler.type is None else ast.unparse(handler.type)
            problems.append(f"{path.name}:{handler.lineno}: handler names {shown}, not DLTrainError")
    return problems


def _heavy(name: str) -> bool:
    return any(name == bad or name.startswith(bad + ".") for bad in HEAVY)


def _module_scope_imports(node: ast.AST) -> Iterator[ast.AST]:
    """Yield import nodes that run when the module is imported.

    Function bodies are skipped; class bodies, ``if``, ``try`` and ``with``
    blocks are not, since they execute at import time.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child
        yield from _module_scope_imports(child)


def rule_no_heavy_module_scope_import(tree: ast.AST, path: Path) -> List[str]:
    package = package_of(path)
    problems = []
    for node in _module_scope_imports(tree):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        else:
            if node.level:
                base_parts = package.split(".")
                base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            candidates = [base] + [f"{base}.{alias.name}" for alias in node.names]
        for name in candidates:
            if _heavy(name):
                problems.append(f"{path.name}:{node.lineno}: module-scope import of {name}")
                break
    return problems


def rule_no_print(tree: ast.AST, path: Path) -> List[str]:
    return [
        f"{path.name}:{node.lineno}: print() call"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]


def _alias_names(tree: ast.AST) -> Set[str]:
    """Return names assigned the literal ``CUDA_VISIBLE_DEVICES``."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Constant) and value.value == GPU_VARIABLE:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                aliases.update(t.id for t in targets if isinstance(t, ast.Name))
    return aliases


def rule_no_gpu_variable_write(tree: ast.AST, path: Path) -> List[str]:
    aliases = _alias_names(tree)

    def is_variable(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return node.value == GPU_VARIABLE
        return isinstance(node, ast.Name) and node.id in aliases

    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
            if is_variable(node.slice):
                problems.append(f"{path.name}:{node.lineno}: subscript write of {GPU_VARIABLE}")
        elif isinstance(node, ast.Call):
            method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if method in ("setdefault", "pop", "putenv", "update") and any(
                is_variable(arg) for arg in node.args
            ):
                problems.append(f"{path.name}:{node.lineno}: {method}() on {GPU_VARIABLE}")
            if any(keyword.arg == GPU_VARIABLE for keyword in node.keywords):
                problems.append(f"{path.name}:{node.lineno}: keyword {GPU_VARIABLE}")
        elif isinstance(node, ast.Dict):
            if any(key is not None and is_variable(key) for key in node.keys):
                problems.append(f"{path.name}:{node.lineno}: dict key {GPU_VARIABLE}")
    return problems


RULES = {
    "no_broad_handler": rule_no_broad_handler,
    "no_keyboard_interrupt_handler": rule_no_keyboard_interrupt_handler,
    "no_entry_point_block": rule_no_entry_point_block,
    "cli_handlers_name_dltrainerror": rule_cli_handlers_name_dltrainerror,
    "no_heavy_module_scope_import": rule_no_heavy_module_scope_import,
    "no_print": rule_no_print,
    "no_gpu_variable_write": rule_no_gpu_variable_write,
}

CLI_PATH = API_DIR / "cli.py"
OTHER_PATH = API_DIR / "other.py"

# (rule, planted snippet, path it is judged as). Every rule has at least one.
PLANTED = [
    ("no_broad_handler", "try:\n    pass\nexcept Exception:\n    pass\n", OTHER_PATH),
    ("no_broad_handler", "try:\n    pass\nexcept BaseException:\n    pass\n", OTHER_PATH),
    ("no_broad_handler", "try:\n    pass\nexcept:\n    pass\n", OTHER_PATH),
    ("no_broad_handler", "try:\n    pass\nexcept (ValueError, Exception):\n    pass\n", OTHER_PATH),
    ("no_broad_handler", "try:\n    pass\nexcept builtins.Exception:\n    pass\n", OTHER_PATH),
    ("no_keyboard_interrupt_handler", "try:\n    pass\nexcept KeyboardInterrupt:\n    pass\n", OTHER_PATH),
    ("no_keyboard_interrupt_handler", "try:\n    pass\nexcept (OSError, KeyboardInterrupt):\n    pass\n", OTHER_PATH),
    ("no_entry_point_block", 'if __name__ == "__main__":\n    pass\n', OTHER_PATH),
    ("no_entry_point_block", 'try:\n    pass\nfinally:\n    if "__main__" == __name__:\n        pass\n', OTHER_PATH),
    ("cli_handlers_name_dltrainerror", "try:\n    pass\nexcept ConfigError:\n    pass\n", CLI_PATH),
    ("cli_handlers_name_dltrainerror", "try:\n    pass\nexcept (DLTrainError, OSError):\n    pass\n", CLI_PATH),
    ("cli_handlers_name_dltrainerror", "try:\n    pass\nexcept Exception:\n    pass\n", CLI_PATH),
    ("no_heavy_module_scope_import", "import yaml\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "import tensorflow as tf\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "import keras.layers\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "from keras import ops\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "from train.common import seed\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "from train import common\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "from dl_techniques.models.vision.vit import model\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "from dl_techniques import models\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "from ..common import seed\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "try:\n    import yaml\nexcept ImportError:\n    yaml = None\n", OTHER_PATH),
    ("no_heavy_module_scope_import", "class Holder:\n    import tensorflow\n", OTHER_PATH),
    ("no_print", 'print("x")\n', OTHER_PATH),
    ("no_print", 'def f():\n    print("x")\n', OTHER_PATH),
    ("no_gpu_variable_write", 'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n', OTHER_PATH),
    ("no_gpu_variable_write", 'import os\nos.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")\n', OTHER_PATH),
    ("no_gpu_variable_write", 'import os\nos.putenv("CUDA_VISIBLE_DEVICES", "0")\n', OTHER_PATH),
    ("no_gpu_variable_write", 'env = {"CUDA_VISIBLE_DEVICES": "0"}\n', OTHER_PATH),
    ("no_gpu_variable_write", 'import os\nNAME = "CUDA_VISIBLE_DEVICES"\nos.environ[NAME] = "0"\n', OTHER_PATH),
    ("no_gpu_variable_write", 'import subprocess\nsubprocess.run([], env=dict(CUDA_VISIBLE_DEVICES="0"))\n', OTHER_PATH),
]

# Source that every rule must leave alone.
CLEAN = [
    (
        "import sys\n"
        "from dl_techniques.utils.logger import logger\n"
        "from . import registry\n"
        "from .errors import ConfigError\n"
        "\n"
        "def f():\n"
        "    import yaml\n"
        "    from train.common import seed\n"
        "    import keras\n"
        "    try:\n"
        "        pass\n"
        "    except (ValueError, OSError):\n"
        "        pass\n"
        "    except DLTrainError:\n"
        "        pass\n"
        "    sys.stdout.write('ok')\n"
        "    return logger\n"
        "\n"
        "import os\n"
        "visible = os.environ.get('CUDA_VISIBLE_DEVICES')\n"
        "doc = 'CUDA_VISIBLE_DEVICES is read, never written'\n"
        "if sys.argv:\n"
        "    pass\n"
    ),
]


@pytest.mark.parametrize(
    "rule, source, path",
    PLANTED,
    ids=[f"{rule}-{index}" for index, (rule, _, _) in enumerate(PLANTED)],
)
def test_a_planted_violation_is_flagged(rule, source, path):
    problems = RULES[rule](ast.parse(source), path)
    assert problems, f"{rule} did not flag:\n{source}"


def test_every_rule_has_a_planted_control():
    assert {rule for rule, _, _ in PLANTED} == set(RULES)


@pytest.mark.parametrize("rule", sorted(RULES))
def test_clean_source_is_left_alone(rule):
    for source in CLEAN:
        for path in (OTHER_PATH, SRC_TRAIN / "__main__.py"):
            assert RULES[rule](ast.parse(source), path) == [], f"{rule} flagged clean source"
    # The cli rule is path-keyed: DLTrainError-only handlers pass there too.
    only = "try:\n    pass\nexcept DLTrainError as exc:\n    pass\n"
    assert rule_cli_handlers_name_dltrainerror(ast.parse(only), CLI_PATH) == []


def test_the_sweep_sees_the_package():
    files = api_files()
    assert len(files) >= MIN_FILES, [path.name for path in files]
    names = {path.name for path in files}
    assert {"__main__.py", "cli.py", "registry.py"} <= names
    # Liveness: cli.py holds the handler the rule about it is about.
    handler_count = sum(1 for _ in handlers(ast.parse(CLI_PATH.read_text())))
    assert handler_count >= 1


@pytest.mark.parametrize("rule", sorted(RULES))
def test_the_api_source_obeys_the_rule(rule):
    problems = []
    for path in api_files():
        problems += RULES[rule](ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), path)
    assert problems == [], "\n".join(problems)


def test_main_is_the_unconditional_exit():
    """``__main__.py`` is ``sys.exit(main(sys.argv[1:]))`` with no guard around it."""
    tree = ast.parse(MAIN_FILE.read_text(encoding="utf-8"))
    calls = [
        node for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "sys.exit"
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0].value.args[0]) == "main(sys.argv[1:])"
