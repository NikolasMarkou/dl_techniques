"""Every third-party module the library imports is a declared dependency.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 8.1 (D-141).
``models/vision/dino/official_weights.py`` imports ``safetensors`` inside the
function that reads the official DINOv2 checkpoints, and ``pyproject.toml``
declared it nowhere: an install from the manifest, with every extra, could not
load the weights that DINOv2 and Depth Anything take with ``pretrained=True``.

An AST census over ``src/dl_techniques`` collects the top-level name of every
absolute import that is neither the standard library nor ``dl_techniques``.
Each name must map to a distribution in ``[project].dependencies`` or one
``[project.optional-dependencies]`` extra, or be in ``UNDECLARED`` below. That
set is exact in both directions: a new undeclared import fails, and declaring
one of the listed modules fails until its entry is removed.
"""

import ast
import os
import re
import sys
import tomllib
from typing import Dict, Set

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src", "dl_techniques")

#: Import name -> distribution name, where the two differ.
DISTRIBUTION_OF = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "tensorflow_datasets": "tensorflow-datasets",
    "cv2": "opencv-python",
    "tensorflow_probability": "tensorflow-probability",
}

#: Imported by the library, declared nowhere in ``pyproject.toml``. Each one is
#: imported inside the function that needs it, so the module still imports
#: without it. Found by this census when it was written (D-141), reported, and
#: outside step 8.1's brief, which names ``safetensors`` only.
UNDECLARED = {
    "cv2": "datasets/bdd100k_video.py, datasets/synthetic_shapes.py",
    "ml_dtypes": "utils/dtype_policy.py; installed by TensorFlow as its own dependency",
    "psutil": "utils/inference.py; listed in requirements.txt only",
    "pycocotools": "callbacks/coco_map_callback.py, datasets/vision/coco_multitask_local.py",
    "tensorflow_probability": "utils/alignment/metrics.py; falls back without it",
}


def _imported_modules() -> Dict[str, Set[str]]:
    found: Dict[str, Set[str]] = {}
    stdlib = set(sys.stdlib_module_names)
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top in stdlib or top == "dl_techniques":
                        continue
                    found.setdefault(top, set()).add(os.path.relpath(path, SRC_ROOT))
    return found


def _declared_distributions() -> Set[str]:
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as handle:
        project = tomllib.load(handle)["project"]
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    return {
        re.split(r"[\s<>=!~\[;]", requirement.strip(), maxsplit=1)[0].lower()
        for requirement in requirements
    }


def _undeclared(imported: Dict[str, Set[str]], declared: Set[str]) -> Set[str]:
    return {
        name for name in imported
        if DISTRIBUTION_OF.get(name, name).lower().replace("_", "-") not in {
            d.replace("_", "-") for d in declared}
    }


def test_the_census_sees_the_core_dependencies():
    imported = _imported_modules()
    for name in ("keras", "numpy", "tensorflow", "safetensors"):
        assert name in imported, f"the import census no longer sees {name!r}"


def test_every_third_party_import_is_declared_or_named():
    undeclared = _undeclared(_imported_modules(), _declared_distributions())
    new = undeclared - set(UNDECLARED)
    assert not new, (
        f"imported by src/dl_techniques but declared nowhere in pyproject.toml: "
        f"{sorted(new)}. Add each to [project].dependencies or the extra that "
        f"matches its use (the 'data' extra for file and dataset readers).")
    stale = set(UNDECLARED) - undeclared
    assert not stale, f"declared or no longer imported, remove from UNDECLARED: {sorted(stale)}"


def test_a_planted_undeclared_import_is_found():
    imported = {"keras": {"x.py"}, "a_module_nobody_declares": {"y.py"}}
    assert _undeclared(imported, _declared_distributions()) == {"a_module_nobody_declares"}
