"""The head factories and the NLP fine-tuning scripts bind no name they never read.

Guard for plan-2026-10-14T062616-81938e81 step 11.4(d)/(e), leftovers of
Steps 11.2 and 11.3 (D-061, D-068). Measured at ``56b696fe2``:

* ``layers/heads/vision/factory.py:83`` imported ``Callable`` and never used
  it, and three ``build`` unpackings bound ``seq_len`` and never read it
  (``:936``, ``:1400``, ``:1405``);
* ``train/bert/finetune.py:14`` and ``train/fnet/finetune.py:14`` imported
  ``List``; ``train/bert/wikipedia/finetune.py`` imported ``numpy`` and
  ``Tuple`` (``:16-17``) and unpacked an ``info`` it never read (``:182``).

Two checks over each module's AST, no third-party linter:

* a module-level import whose bound name is never loaded (a line marked
  ``noqa`` is a deliberate registrar import and is skipped; names inside
  string annotations count as loads);
* a name bound by a tuple or list unpacking inside a function and never
  loaded in it (a leading ``_`` marks a deliberate discard).
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"

MODULES = [
    "dl_techniques/layers/heads/factory.py",
    "dl_techniques/layers/heads/validation.py",
    "dl_techniques/layers/heads/vision/base.py",
    "dl_techniques/layers/heads/vision/config.py",
    "dl_techniques/layers/heads/vision/decoder.py",
    "dl_techniques/layers/heads/vision/dense_vector.py",
    "dl_techniques/layers/heads/vision/depth.py",
    "dl_techniques/layers/heads/vision/detection.py",
    "dl_techniques/layers/heads/vision/dispatch.py",
    "dl_techniques/layers/heads/vision/enhancement.py",
    "dl_techniques/layers/heads/vision/instance_segmentation.py",
    "dl_techniques/layers/heads/vision/multitask.py",
    "dl_techniques/layers/heads/vision/segmentation.py",
    "dl_techniques/layers/heads/vision/utils.py",
    "dl_techniques/layers/heads/vision/classification.py",
    "dl_techniques/layers/heads/vision/task_types.py",
    "dl_techniques/layers/heads/nlp/factory.py",
    "dl_techniques/layers/heads/nlp/utils.py",
    "dl_techniques/layers/heads/nlp/base.py",
    "dl_techniques/layers/heads/nlp/dispatch.py",
    "dl_techniques/layers/heads/nlp/multitask.py",
    "dl_techniques/layers/heads/nlp/text_classification.py",
    "dl_techniques/layers/heads/nlp/token_classification.py",
    "dl_techniques/layers/heads/nlp/question_answering.py",
    "dl_techniques/layers/heads/nlp/text_similarity.py",
    "dl_techniques/layers/heads/nlp/text_generation.py",
    "dl_techniques/layers/heads/nlp/multiple_choice.py",
    "dl_techniques/layers/heads/nlp/task_types.py",
    "dl_techniques/layers/heads/nlp/model_head.py",
    "dl_techniques/layers/heads/vlm/factory.py",
    "dl_techniques/layers/heads/vlm/utils.py",
    "dl_techniques/layers/heads/vlm/base.py",
    "dl_techniques/layers/heads/vlm/image_captioning.py",
    "dl_techniques/layers/heads/vlm/vqa.py",
    "dl_techniques/layers/heads/vlm/visual_grounding.py",
    "dl_techniques/layers/heads/vlm/image_text_matching.py",
    "dl_techniques/layers/heads/vlm/multitask.py",
    "dl_techniques/layers/heads/vlm/dispatch.py",
    "dl_techniques/layers/heads/vlm/task_types.py",
    "train/bert/finetune.py",
    "train/fnet/finetune.py",
    "train/bert/wikipedia/finetune.py",
    "train/tree_transformer/finetune.py",
    "train/cliffordnet/train_clip.py",
    # Step 12 Part F items 7 and 17: an unread ``stage_idx`` and a dead
    # ``create_topographic_vae`` import.
    "dl_techniques/models/vision_language/clifford_clip/model.py",
    "train/topographic_vae/train_topographic_vae.py",
]


def _loads(tree):
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            annotations += [a.annotation for a in args if a.annotation is not None]
            annotations += [node.returns] if node.returns is not None else []
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    for annotation in annotations:
        for node in ast.walk(annotation):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                try:
                    parsed = ast.parse(node.value, mode="eval")
                except SyntaxError:
                    continue
                names |= {n.id for n in ast.walk(parsed) if isinstance(n, ast.Name)}
    return names


def unused_imports(source):
    """``[(line, name)]`` of module-level imports whose name is never loaded."""
    tree = ast.parse(source)
    lines = source.splitlines()
    bound = {}
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if "noqa" in lines[node.lineno - 1]:
            continue
        for alias in node.names:
            bound[alias.asname or alias.name.split(".")[0]] = node.lineno
    loaded = _loads(tree)
    return sorted((line, name) for name, line in bound.items() if name not in loaded)


def unread_unpacked_names(source):
    """``[(line, function, name)]`` of unpacked names a function never loads."""
    found = set()
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.For):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if not isinstance(target, (ast.Tuple, ast.List)):
                    continue
                for elt in target.elts:
                    if (isinstance(elt, ast.Name) and not elt.id.startswith("_")
                            and elt.id not in loaded):
                        found.add((elt.lineno, fn.name, elt.id))
    return sorted(found)


@pytest.mark.parametrize("module", MODULES)
def test_no_import_is_unused(module) -> None:
    assert unused_imports((ROOT / module).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("module", MODULES)
def test_no_unpacked_name_is_unread(module) -> None:
    assert unread_unpacked_names((ROOT / module).read_text(encoding="utf-8")) == []


def test_the_checks_see_what_they_look_for() -> None:
    """Both checks fire on a planted case and stay quiet on the legal forms."""
    planted = (
        "import os\nimport sys  # noqa: F401\nfrom typing import List, Tuple\n"
        "def f(x: 'Tuple[int, int]'):\n    a, b = x\n    _c, d = x\n    return a + d\n")
    assert unused_imports(planted) == [(1, "os"), (3, "List")]
    assert unread_unpacked_names(planted) == [(5, "f", "b")]
