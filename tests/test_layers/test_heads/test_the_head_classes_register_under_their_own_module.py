"""Every head class registers under the module that defines it.

The repository rule (``src/dl_techniques/AGENTS.md`` § Registration) is that the
``package`` string given to ``register_dl_technique`` is the defining module's
dotted path. The heads were once all defined in one ``factory.py`` per domain and
registered under it. After each factory was split, a class that kept the old
string would mint a key naming a module that does not hold it.

Two instruments, because neither sees what the other does:

* a table of the 24 registered classes, checked through the live Keras registry
  (key, defining module, alias identity);
* an AST census of every ``@register_dl_technique("...")`` under ``heads/``, which
  also catches a class added later with a copied string.
"""

import ast
import importlib
from pathlib import Path

import keras
import pytest

import dl_techniques.layers.heads as heads_pkg

HEADS_DIR = Path(heads_pkg.__file__).parent
PACKAGE = "dl_techniques.layers.heads"

#: Public class name to ``<domain>.<module>`` that must define and register it.
_HOMES = {
    "BaseNLPHead": "nlp.base",
    "TextClassificationHead": "nlp.text_classification",
    "TokenClassificationHead": "nlp.token_classification",
    "QuestionAnsweringHead": "nlp.question_answering",
    "TextSimilarityHead": "nlp.text_similarity",
    "TextGenerationHead": "nlp.text_generation",
    "MultipleChoiceHead": "nlp.multiple_choice",
    "MultiTaskNLPHead": "nlp.multitask",
    "BaseVisionHead": "vision.base",
    "DetectionHead": "vision.detection",
    "SegmentationHead": "vision.segmentation",
    "ClassificationHead": "vision.classification",
    "InstanceSegmentationHead": "vision.instance_segmentation",
    "DepthEstimationHead": "vision.depth",
    "SurfaceNormalsHead": "vision.dense_vector",
    "OpticalFlowHead": "vision.dense_vector",
    "EnhancementHead": "vision.enhancement",
    "MultiTaskHead": "vision.multitask",
    "BaseVLMHead": "vlm.base",
    "ImageCaptioningHead": "vlm.image_captioning",
    "VQAHead": "vlm.vqa",
    "VisualGroundingHead": "vlm.visual_grounding",
    "ImageTextMatchingHead": "vlm.image_text_matching",
    "MultiTaskVLMHead": "vlm.multitask",
}


def _class(name: str):
    return getattr(heads_pkg, name)


def test_the_guard_has_subjects() -> None:
    """Without this, an empty table or census would pass every arm below."""
    assert len(_HOMES) == 24
    assert len(_decorated_sites()) >= 24


@pytest.mark.parametrize("name,home", sorted(_HOMES.items()))
def test_the_class_is_defined_in_its_own_module(name: str, home: str) -> None:
    cls = _class(name)
    assert cls.__module__ == f"{PACKAGE}.{home}"
    assert getattr(importlib.import_module(cls.__module__), name) is cls


@pytest.mark.parametrize("name,home", sorted(_HOMES.items()))
def test_the_registered_key_names_the_defining_module(name: str, home: str) -> None:
    assert keras.saving.get_registered_name(_class(name)) == f"{PACKAGE}.{home}>{name}"


def test_the_bare_alias_resolves_to_the_same_class() -> None:
    for name in _HOMES:
        assert keras.saving.get_registered_object(f"Custom>{name}") is _class(name)


def _decorated_sites():
    """``(path, class name, package string)`` for every registered class under heads/."""
    sites = []
    for path in sorted(HEADS_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for deco in node.decorator_list:
                if (isinstance(deco, ast.Call)
                        and getattr(deco.func, "id", None) == "register_dl_technique"
                        and deco.args and isinstance(deco.args[0], ast.Constant)):
                    sites.append((path, node.name, deco.args[0].value))
    return sites


def test_every_registration_string_is_its_own_files_dotted_path() -> None:
    wrong = []
    for path, cls, package in _decorated_sites():
        expected = ".".join(
            (PACKAGE, *path.relative_to(HEADS_DIR).with_suffix("").parts)
        )
        if package != expected:
            wrong.append(f"{path.relative_to(HEADS_DIR)}:{cls} says {package!r}, expected {expected!r}")
    assert not wrong, "\n".join(wrong)


def test_the_census_sees_exactly_the_tabled_classes() -> None:
    assert sorted(cls for _, cls, _ in _decorated_sites()) == sorted(_HOMES)
