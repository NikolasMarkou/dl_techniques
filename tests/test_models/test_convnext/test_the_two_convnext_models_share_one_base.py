"""ConvNeXt V1 and V2 are two leaves over one base class, ``_ConvNeXtBase``.

The two models differ in one thing, the block (``ConvNextV1Block`` against
``ConvNextV2Block``, which adds Global Response Normalization), and in their variant
tables. Everything else, the constructor, the builders, ``call``, ``build``,
``compute_output_shape``, the pretrained loader, ``from_variant``, ``get_config`` and
``summary``, is the same text, so it lives once in ``convnext/common.py``. A leaf file
keeps its registration key, its class docstring, ``MODEL_VARIANTS``, ``BLOCK_CLASS`` and
its ``create_*`` factory.

The guard reads the leaf source: a method defined in a leaf would be a second copy of the
base that the next edit fixes in one file only. It is a function of source text, so a
planted leaf with a method proves it can fail.
"""

import ast
import importlib
from pathlib import Path

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.conv_blocks.convnext_v2_block import ConvNextV2Block
from dl_techniques.models.vision.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2

CONVNEXT = Path(__file__).resolve().parents[3] / "src/dl_techniques/models/vision/convnext"
LEAVES = (
    pytest.param(ConvNeXtV1, "convnext_v1.py", ConvNextV1Block, id="v1"),
    pytest.param(ConvNeXtV2, "convnext_v2.py", ConvNextV2Block, id="v2"),
)
#: What a leaf may assign in its class body.
LEAF_CONSTANTS = frozenset({"MODEL_VARIANTS", "BLOCK_CLASS"})
TINY = dict(depths=[1, 1], dims=[8, 16], num_classes=4, input_shape=(32, 32, 3))


def base_class() -> type:
    """``_ConvNeXtBase``, imported when a test needs it so one missing module fails one test.

    Read as an attribute of its module, not bound to a local name: the underscore-name census
    (``tests/test_no_underscore_named_layer_takes_its_auto_name.py``) lets an abstract base
    be subclassed or type-checked but not passed around by name.
    """
    return importlib.import_module("dl_techniques.models.vision.convnext.common")._ConvNeXtBase


def leaf_body_problems(source: str, class_name: str) -> list:
    """What a leaf class defines beyond its docstring and ``LEAF_CONSTANTS``.

    Interface contract: ``source`` is the text of a leaf module and ``class_name`` the
    class in it. Returns one message per method or other constant in the class body,
    empty for a clean leaf. Raises ``AssertionError`` when the class is absent.
    """
    cls = next((n for n in ast.parse(source).body
                if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    assert cls is not None, f"{class_name} is not in the source"
    out = []
    for node in cls.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the docstring
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"`{node.name}` is defined in the leaf, so it is a second copy of the base")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if not names <= LEAF_CONSTANTS:
                out.append(f"constant {sorted(names - LEAF_CONSTANTS)} belongs in the base")
        else:
            out.append(f"unexpected {type(node).__name__} in the class body")
    return out


_CLEAN_LEAF = '''
class Leaf(_ConvNeXtBase):
    """A leaf."""

    MODEL_VARIANTS = {"tiny": {"depths": [1], "dims": [8]}}
    BLOCK_CLASS = SomeBlock
'''


def test_the_leaf_rule_passes_a_clean_leaf_and_sees_each_flaw() -> None:
    assert leaf_body_problems(_CLEAN_LEAF, "Leaf") == []
    with_method = _CLEAN_LEAF + "\n    def call(self, inputs):\n        return inputs\n"
    assert leaf_body_problems(with_method, "Leaf") == [
        "`call` is defined in the leaf, so it is a second copy of the base"]
    with_constant = _CLEAN_LEAF + "\n    LAYERNORM_EPSILON = 1e-6\n"
    assert leaf_body_problems(with_constant, "Leaf") == [
        "constant ['LAYERNORM_EPSILON'] belongs in the base"]
    with pytest.raises(AssertionError, match="Other is not in the source"):
        leaf_body_problems(_CLEAN_LEAF, "Other")


@pytest.mark.parametrize(("cls", "filename", "block"), LEAVES)
def test_a_leaf_class_defines_only_its_variants_and_its_block(cls, filename, block) -> None:
    source = (CONVNEXT / filename).read_text(encoding="utf-8")
    assert leaf_body_problems(source, cls.__name__) == []


@pytest.mark.parametrize(("cls", "filename", "block"), LEAVES)
def test_a_leaf_inherits_the_base_body_it_does_not_copy(cls, filename, block) -> None:
    assert cls.__mro__[1] is base_class()
    for name in ("__init__", "build", "call", "compute_output_shape", "load_pretrained_weights",
                 "from_variant", "get_config", "summary"):
        assert getattr(cls, name).__qualname__.startswith("_ConvNeXtBase."), name
    assert cls.BLOCK_CLASS is block


def test_the_two_leaves_share_one_constructor_and_differ_only_by_block_and_variants() -> None:
    assert ConvNeXtV1.__init__ is ConvNeXtV2.__init__
    assert ConvNeXtV1.get_config is ConvNeXtV2.get_config
    assert ConvNeXtV1.BLOCK_CLASS is not ConvNeXtV2.BLOCK_CLASS
    assert set(ConvNeXtV1.MODEL_VARIANTS) != set(ConvNeXtV2.MODEL_VARIANTS)


@pytest.mark.parametrize(("cls", "filename", "block"), LEAVES)
def test_every_block_of_a_model_is_its_leafs_block(cls, filename, block) -> None:
    model = cls(**TINY)
    model(np.zeros((1, 32, 32, 3), dtype="float32"), training=False)
    blocks = [entry["block"] for stage in model.stages for entry in stage]
    assert len(blocks) == 2
    assert all(type(b) is block for b in blocks)


@pytest.mark.parametrize(("cls", "filename", "block"), LEAVES)
def test_the_registration_key_is_the_leafs_own(cls, filename, block) -> None:
    stem = filename.removesuffix(".py")
    assert keras.saving.get_registered_name(cls) == f"dl_techniques.models.convnext.{stem}>{cls.__name__}"


def test_the_base_refuses_to_build_without_a_block_class() -> None:
    """The base is never built itself; a leaf that forgets ``BLOCK_CLASS`` fails by name."""

    class NoBlock(base_class()):
        MODEL_VARIANTS = {}

    with pytest.raises(TypeError, match="BLOCK_CLASS"):
        NoBlock(**TINY)
