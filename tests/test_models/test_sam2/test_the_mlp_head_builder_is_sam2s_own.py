"""SAM 2 builds its MLP heads with its own builder, never sam1's private one.

``sam2/mask_decoder.py`` and ``sam2/model.py`` imported
``sam1.mask_decoder._build_mlp_head``, a private name crossing a sibling
package (plan-2026-10-14T062616-81938e81 D-006). SAM 2 now has
``sam2.mask_decoder.build_mlp_head``, a copy that builds the identical
``keras.Sequential`` (D-046). Two guards:

* an AST scan of every ``sam2`` module: an import from a ``sam1`` module may
  only name public symbols;
* the builder's structure, pinned against the reference ``MLP``: ``N - 1``
  hidden Dense layers of ``hidden_dim`` with the activation, one linear
  ``output_dim`` layer, and the layer names the weight paths are keyed on.
"""

import ast
from pathlib import Path

import keras
import numpy as np
import pytest

import dl_techniques.models.vision_language.sam.sam2 as sam2_package
from dl_techniques.models.vision_language.sam.sam2.mask_decoder import build_mlp_head
from dl_techniques.models.vision_language.sam.sam2.model import create_sam2

SAM2_DIR = Path(sam2_package.__file__).parent
SAM1_PREFIX = "dl_techniques.models.vision_language.sam.sam1"


def _sam1_imports():
    """``(module file, imported module, name)`` for every sam1 import in sam2."""
    found = []
    for path in sorted(SAM2_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(SAM1_PREFIX):
                found.extend((path.name, node.module, alias.name) for alias in node.names)
    return found


def test_sam2_imports_no_private_name_from_sam1():
    imports = _sam1_imports()
    # Anti-vacuity: sam2 does reuse sam1's public classes, so the scan must see them.
    assert {name for _, _, name in imports} >= {"TwoWayTransformer", "PromptEncoder"}, imports
    private = [entry for entry in imports if entry[2].startswith("_")]
    assert not private, f"sam2 imports private sam1 names: {private}"


@pytest.mark.parametrize("num_layers", [1, 3])
def test_the_builder_is_the_reference_mlp(num_layers):
    head = build_mlp_head(
        num_layers=num_layers, hidden_dim=16, output_dim=4, activation="relu",
        dense_name_template="probe_dense{n}", name="probe_head")
    assert isinstance(head, keras.Sequential) and head.name == "probe_head"
    assert [layer.name for layer in head.layers] == [
        f"probe_dense{n}" for n in range(1, num_layers + 1)]
    assert all(isinstance(layer, keras.layers.Dense) for layer in head.layers)
    assert [layer.units for layer in head.layers] == [16] * (num_layers - 1) + [4]
    assert [keras.activations.serialize(layer.activation) for layer in head.layers] == (
        ["relu"] * (num_layers - 1) + ["linear"])
    out = head(np.ones((2, 8), dtype="float32"))
    assert tuple(out.shape) == (2, 4)


def test_every_sam2_mlp_head_keeps_its_weight_paths():
    """The decoder's and the model's heads carry the names the archive keys on."""
    model = create_sam2("tiny")
    model.build(None)
    decoder = model.mask_decoder
    assert [layer.name for layer in decoder.iou_prediction_head.layers] == [
        f"iou_dense{n}" for n in range(1, decoder.iou_head_depth + 1)]
    assert [layer.name for layer in decoder.output_hypernetworks_mlps[0].layers] == [
        "hyper_dense1_0", "hyper_dense2_0", "hyper_dense3_0"]
    assert [layer.name for layer in decoder.pred_obj_score_head.layers] == [
        "obj_score_dense1", "obj_score_dense2", "obj_score_dense3"]
    assert [layer.name for layer in model.obj_ptr_proj.layers][0] == "obj_ptr_proj_dense1"
