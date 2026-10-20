"""Every NLP head helper reaches ``src/`` through one import surface.

The heads review's A2-2: the fifteen headed language models imported nine
helper names in five different subsets, partly from the package
``dl_techniques.layers.heads.nlp`` and partly from its submodule
``.model_head``, because the package did not export ``TASK_HEAD_NAME``,
``head_output_shape`` or ``ids_shape_of``. Measured at 3a89046f8: 15 model
files and ``src/train/common/nlp.py`` imported from ``.model_head``.

The check is an AST walk over every ``src/`` file outside
``layers/heads/nlp`` itself: no import names ``layers.heads.nlp.model_head``
(or any other submodule of the package), and every name imported from
``layers.heads.nlp`` is in its ``__all__``. A planted source proves the walk
sees both kinds of offence.
"""

import ast
from pathlib import Path
from typing import List

import pytest

import dl_techniques.layers.heads.nlp as nlp_package

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = "dl_techniques.layers.heads.nlp"
OWN_DIR = SRC / "dl_techniques" / "layers" / "heads" / "nlp"


def _offences(source: str) -> List[str]:
    """Return one line per offending import in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module.startswith(PACKAGE + "."):
            found.append(f"line {node.lineno}: imports from {node.module}")
        elif node.module == PACKAGE:
            for alias in node.names:
                if alias.name not in nlp_package.__all__:
                    found.append(
                        f"line {node.lineno}: {alias.name} is not in "
                        f"{PACKAGE}.__all__")
    return found


def _consumers() -> List[Path]:
    return sorted(
        p for p in SRC.rglob("*.py")
        if OWN_DIR not in p.parents and PACKAGE in p.read_text(encoding="utf-8")
    )


def test_the_walk_sees_both_offences():
    planted = (
        "from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME\n"
        "from dl_techniques.layers.heads.nlp import get_head_class\n"
        "from dl_techniques.layers.heads.nlp import NLPTaskType\n"
    )
    assert len(_offences(planted)) == 2, _offences(planted)


def test_the_census_finds_the_language_models():
    # Anti-vacuity: the fifteen headed language models plus the shared trainer.
    language = [p for p in _consumers() if "models/language" in p.as_posix()]
    assert len(language) >= 15, language


@pytest.mark.parametrize(
    "path", _consumers(), ids=lambda p: p.relative_to(SRC).as_posix())
def test_a_consumer_imports_only_the_package_surface(path):
    assert _offences(path.read_text(encoding="utf-8")) == []


def test_the_three_helpers_are_exported():
    for name in ("TASK_HEAD_NAME", "head_output_shape", "ids_shape_of"):
        assert name in nlp_package.__all__
        assert getattr(nlp_package, name) is not None


def test_the_heads_union_leaves_out_only_the_in_model_helpers():
    # The heads package docstring says so; this keeps it true.
    import dl_techniques.layers.heads as heads

    helpers = {
        "TASK_HEAD_NAME", "build_model_task_head", "finalize_head_outputs",
        "head_attention_mask", "head_output_shape", "ids_shape_of",
        "task_config_to_dict", "validate_task_head_knobs",
    }
    assert set(nlp_package.__all__) - set(heads.__all__) == helpers
