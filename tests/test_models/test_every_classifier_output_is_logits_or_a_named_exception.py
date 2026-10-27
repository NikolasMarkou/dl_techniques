"""Every classifier-like output in ``models/`` is logits, or a named exception.

Census guard for plan-2026-10-14T062616-81938e81 iteration 2 step 2.4 (D-122
item 5, D-153). An AST walk over ``src/dl_techniques/models`` collects every
place a softmax or a sigmoid can enter a model:

- ``activation='softmax'|'sigmoid'`` on any call;
- ``Softmax(...)`` and ``Activation('softmax'|'sigmoid')``;
- ``keras.ops`` / ``ops`` / ``keras.activations`` / ``tf`` ``sigmoid(...)`` and
  ``softmax(...)`` calls;
- the strings ``'softmax'`` / ``'sigmoid'`` anywhere else in code (not in a
  docstring), which is how a variable-valued activation enters
  (``final_activation = "softmax" if ...``).

Each site is keyed ``(path under models/, enclosing scope, token)`` and must be
classified exactly once:

- :data:`INTERNAL`: not a model output (attention weights, gates, sampling,
  smoothing coefficients, a companion key next to the logits);
- :data:`RANGE_MAPS`: an output that maps to a range, not a class score (boxes,
  pixels, an IoU estimate);
- :data:`CLASS_SCORE_EXCEPTIONS`: a class-score probability that is the model's
  documented product. This list must equal the exception list in
  ``models/GUIDE.md`` (the logits bullet), in both directions, and each entry's
  reason is checked against the built model.

A census site no table names fails, and so does a table entry the census no
longer finds.

A model can return a SHARED layer's output, from outside ``models/``:
``SHGCNLinkPredictor`` returns ``FermiDiracDecoder``'s edge score. Each such
layer module is listed in :data:`SHARED_OUTPUT_LAYERS` with the model module
that returns it, is scanned by the same census (keyed by its path under
``src/dl_techniques/``, prefixed ``src:``), and the model module must still
import the layer, so a stale row fails (D-159). Three exceptions enter without any softmax/sigmoid token (capsule
lengths, a normalized RBF), so their rows hold no census key; their reason is
checked by building the model.
"""

import ast
import os
import re
from typing import Callable, Dict, Set, Tuple

import keras
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(REPO_ROOT, "src", "dl_techniques")
MODELS_ROOT = os.path.join(SRC_ROOT, "models")
GUIDE = os.path.join(MODELS_ROOT, "GUIDE.md")

Key = Tuple[str, str, str]
_ACTIVATIONS = ("softmax", "sigmoid")
_OP_ROOTS = ("keras", "ops", "tf", "K", "activations")


#: Shared layers outside ``models/`` whose output a model returns as its own:
#: layer module (under ``src/dl_techniques/``) -> (model module under
#: ``models/`` that returns it, the layer class it imports).
SHARED_OUTPUT_LAYERS: Dict[str, Tuple[str, str]] = {
    "layers/graphs/fermi_diract_decoder.py": ("graph/shgcn/model.py", "FermiDiracDecoder"),
}


def census(root: str = MODELS_ROOT) -> Set[Key]:
    """Every softmax/sigmoid site under ``root``, keyed ``(relpath, scope, token)``.

    Interface contract (callers: the tests in this file, including the planted
    positive control): ``root`` is a directory of Python sources; returns the
    set of keys described in the module docstring. For the default root the
    :data:`SHARED_OUTPUT_LAYERS` modules are scanned too, keyed
    ``src:<path under src/dl_techniques>``. A file that does not parse raises
    ``SyntaxError`` (a silent skip would shrink the census).
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if filename.endswith(".py"):
                path = os.path.join(dirpath, filename)
                files.append((path, os.path.relpath(path, root).replace(os.sep, "/")))
    if root == MODELS_ROOT:
        files += [(os.path.join(SRC_ROOT, rel), f"src:{rel}") for rel in SHARED_OUTPUT_LAYERS]
    keys: Set[Key] = set()
    for path, rel in files:
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        def scope(node: ast.AST) -> str:
            names = []
            while id(node) in parents:
                node = parents[id(node)]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
            return ".".join(reversed(names)) or "<module>"

        counted = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)):
                counted.add(id(body[0].value))  # a docstring is prose, not code
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = ast.unparse(node.func)
            last = func.split(".")[-1]
            for keyword in node.keywords:
                value = keyword.value
                if (keyword.arg == "activation" and isinstance(value, ast.Constant)
                        and value.value in _ACTIVATIONS):
                    keys.add((rel, scope(node), f"activation={value.value}"))
                    counted.add(id(value))
            if last == "Softmax":
                keys.add((rel, scope(node), "Softmax()"))
            if (last == "Activation" and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in _ACTIVATIONS):
                keys.add((rel, scope(node), f"Activation({node.args[0].value})"))
                counted.add(id(node.args[0]))
            if last in _ACTIVATIONS and func.split(".")[0] in _OP_ROOTS:
                keys.add((rel, scope(node), f"{last}()"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in _ACTIVATIONS and id(node) not in counted:
                keys.add((rel, scope(node), f"'{node.value}'"))
    return keys


#: Not a model output.
INTERNAL: Dict[Key, str] = {
    ("general_purpose/hkan/hkan_layer.py", "<module>", "'sigmoid'"): "basis-function table key",
    ("general_purpose/hkan/hkan_layer.py", "<module>", "sigmoid()"): "sigmoid basis function of a hidden unit",
    ("general_purpose/hkan/hkan_layer.py", "HKANLayer.__init__", "'sigmoid'"): "default basis name",
    ("general_purpose/hkan/model.py", "HKAN.__init__", "'sigmoid'"): "default basis name",
    ("language/blt/model.py", "ByteLatentTransformer._top_p_filtering", "softmax()"): "nucleus sampling",
    ("language/blt/model.py", "ByteLatentTransformer.generate", "softmax()"): "token sampling",
    ("language/hrm/model.py", "HierarchicalReasoningModel._forward_step", "sigmoid()"): "Q-learning halt target",
    ("language/tree_transformer/components.py", "GroupAttention.call", "softmax()"): "attention weights",
    ("language/tree_transformer/components.py", "TreeMHA.call", "softmax()"): "attention weights",
    ("language/trm/model.py", "TRM.call", "sigmoid()"): "Q-learning continue target",
    ("neural_computer/nam/cell.py", "NAMCell.call", "softmax()"): "reduction and operator mixing weights",
    ("time_series/adaptive_ema/model.py", "AdaptiveEMASlopeFilterModel.call", "sigmoid()"): "soft slope gates",
    ("time_series/ets/model.py", "ETSModel._smoothing", "sigmoid()"): "smoothing coefficients in (0, 1)",
    ("time_series/ets/model.py", "ETSModel.alpha", "sigmoid()"): "smoothing coefficient read-out",
    ("time_series/ets/model.py", "ETSModel.beta", "sigmoid()"): "smoothing coefficient read-out",
    ("time_series/ets/model.py", "ETSModel.gamma", "sigmoid()"): "smoothing coefficient read-out",
    ("time_series/timestral/layers.py", "Attention.call", "sigmoid()"):
        "sdpa_sigmoid output gate on the attention (found by this census when Timestral landed)",
    ("time_series/xlstm/forecaster.py", "xLSTMForecaster.__init__", "'sigmoid'"): "sLSTM forget-gate kind",
    ("time_series/xlstm/model.py", "xLSTM.__init__", "'sigmoid'"): "sLSTM forget-gate kind",
    ("vision/beit/model.py",
     "BeitForImageClassification._build_classification_head._MeanPoolingHead.call", "softmax()"):
        "the head's 'probabilities' companion key; the model returns 'logits' (D-099)",
    ("vision/energy_transformer/model.py", "<module>", "'softmax'"): "Hopfield activation option",
    ("vision/image_restoration/doc_scanner/components.py", "SepConvGRU._gru_step", "sigmoid()"): "GRU gates",
    ("vision/image_restoration/doc_scanner/model.py", "DocScanner.call", "sigmoid()"):
        "the segmenter's logits become the confidence the composite thresholds at 0.5 (D-146)",
    ("vision/image_restoration/doc_scanner/warp.py", "convex_upsample", "softmax()"): "convex upsampling weights",
    ("world_models/levjepa/blocks.py", "LeVJEPABlock.call", "softmax()"): "attention weights",
    ("vision/super_resolution/pft_sr/blocks.py", "PFTLayer._attend", "softmax()"):
        "attention weights of the official PFT layer (D-212; the site arrived with the body port)",
    ("vision_language/bit_diffusion/bridge_process.py", "sample_timesteps_logit_normal", "sigmoid()"):
        "logit-normal timestep sampling",
    ("vision_language/ideogram4/vae.py", "AttnBlock.call", "softmax()"): "attention weights",
    ("vision_language/sam/sam1/image_encoder.py", "WindowedAttentionWithRelPos.call", "softmax()"): "attention weights",
    ("vision_language/sam/sam2/hiera.py", "HieraMultiScaleAttention.call", "softmax()"): "attention weights",
    ("vision_language/sam/sam2/memory_attention.py", "_SAM2RoPEAttention.call", "softmax()"): "attention weights",
    ("vision_language/sam/sam2/memory_encoder.py", "SAM2MemoryEncoder._affine_sigmoid", "sigmoid()"):
        "mask-to-memory scaling (reference sigmoid_scale_for_mem_enc)",
    ("vision_language/sam/sam2/model.py", "SAM2._blend_object_pointer", "sigmoid()"): "soft object-pointer gate",
    ("vision_language/sam/sam3/decoder.py", "Sam3DecoderLayer.__init__", "'softmax'"): "attention probability type",
    ("vision_language/sam/sam3/decoder.py", "_Sam3DecoderAttention.call", "softmax()"): "attention weights",
    ("vision_language/sam/sam3/maskformer_segmentation.py", "Sam3SegmentationHead.__init__", "'softmax'"):
        "attention probability type",
    ("vision_language/sam/sam3/sam3_image.py", "Sam3Image._fuse", "sigmoid()"):
        "presence fusion; the result is returned as logits (inverse sigmoid, clipped)",
    ("vision_language/sam/sam3/vitdet.py", "_Sam3ViTDetAttention.call", "softmax()"): "attention weights",
    ("vision_language/sd3_mmdit/text_encoders.py", "T5Encoder.call", "softmax()"): "attention weights",
    ("vision_language/sd3_mmdit/text_encoders.py", "_quick_gelu", "sigmoid()"): "QuickGELU activation",
}

#: An output that maps to a range, not a class score.
RANGE_MAPS: Dict[Key, str] = {
    ("vision/capsnet/model.py", "CapsNet._build_decoder", "activation=sigmoid"): "reconstruction pixels in [0, 1]",
    ("vision/capsnet/model_v2.py", "CapsNetV2._build_decoder", "activation=sigmoid"): "reconstruction pixels in [0, 1]",
    ("vision/detr/model.py", "DETR.call", "sigmoid()"): "normalized box coordinates (reference DETR)",
    ("vision/mambalct/model.py", "MambaLCT.call", "sigmoid()"): "normalized box coordinates",
    ("vision/topographic_vae/model.py", "TopographicVAE.__init__", "'sigmoid'"): "Bernoulli pixel decoder",
    ("vision/vae/model.py", "VAE", "'sigmoid'"): "pixel decoder default",
    ("vision/vae/model.py", "VAE.__init__", "'sigmoid'"): "pixel decoder default",
    ("vision/vq_vae_rotation/model.py", "_build_auto_decoder", "activation=sigmoid"): "pixel decoder",
    ("vision_language/sam/sam2/mask_decoder.py", "SAM2MaskDecoder.predict_masks", "sigmoid()"):
        "IoU estimate in [0, 1] (reference iou_prediction_use_sigmoid)",
    ("vision_language/sam/sam3/decoder.py", "Sam3TransformerDecoder.call", "sigmoid()"): "box refinement",
    ("vision_language/sam/sam3/query_selection.py", "Sam3EncoderQuerySelection.call", "sigmoid()"): "anchor boxes",
    ("vision_language/sam/sam3/sam3_image.py", "Sam3Image._forward_all", "sigmoid()"): "box coordinates",
}


def _probe_capsnet():
    from dl_techniques.models.vision.capsnet import create_capsnet
    model = create_capsnet(num_classes=4, input_shape=(28, 28, 1))
    lengths = keras.ops.convert_to_numpy(model(_images((2, 28, 28, 1)), training=False)["length"])
    assert lengths.shape == (2, 4) and (lengths >= 0).all() and (lengths < 1).all(), lengths


def _probe_capsnet_v2():
    from dl_techniques.models.vision.capsnet.model_v2 import CapsNetV2
    model = CapsNetV2(num_classes=4, input_shape=(28, 28, 1), stem="legacy")
    lengths = keras.ops.convert_to_numpy(model(_images((2, 28, 28, 1)), training=False))
    assert lengths.shape == (2, 4) and (lengths > 0).all() and (lengths < 1).all(), lengths


def _probe_rbf_protonet():
    from dl_techniques.models.vision.rbf_protonet import create_rbf_protonet
    model = create_rbf_protonet(num_classes=4, input_shape=(32, 32, 3), stem_filters=8,
                                filters_per_stage=[8, 16], feature_dim=16)
    rows = keras.ops.convert_to_numpy(model(_images((2, 32, 32, 3)), training=False)).sum(-1)
    np.testing.assert_allclose(rows, 1.0, rtol=0, atol=1e-5)


def _probe_clifford_rbf_protonet():
    from dl_techniques.models.vision.rbf_protonet.model import CliffordRBFProtoNet
    model = CliffordRBFProtoNet(input_shape=(32, 32, 3), num_classes=4, depth=2)
    rows = keras.ops.convert_to_numpy(model(_images((2, 32, 32, 3)), training=False)).sum(-1)
    np.testing.assert_allclose(rows, 1.0, rtol=0, atol=1e-5)


def _probe_qwen3_reranker():
    from dl_techniques.models.language.qwen.qwen3_embeddings import Qwen3RerankerModel
    model = Qwen3RerankerModel(vocab_size=64, hidden_size=32, num_layers=1, num_heads=4,
                               intermediate_size=64, max_seq_len=16, yes_token_id=11, no_token_id=22)
    rng = np.random.default_rng(0)
    scores = keras.ops.convert_to_numpy(model({
        "input_ids": rng.integers(0, 64, (3, 8)).astype("int32"),
        "attention_mask": np.ones((3, 8), dtype="int32"),
    }, training=False))
    assert scores.shape == (3,) and (scores >= 0).all() and (scores <= 1).all(), scores


#: The documented-probability exceptions: name -> (census keys, probe).
CLASS_SCORE_EXCEPTIONS: Dict[str, Tuple[Tuple[Key, ...], Callable[[], None]]] = {
    "CapsNet": ((), _probe_capsnet),
    "CapsNetV2": ((), _probe_capsnet_v2),
    "RBFProtoNet": ((), _probe_rbf_protonet),
    "CliffordRBFProtoNet": ((), _probe_clifford_rbf_protonet),
    "Qwen3RerankerModel": (
        (("language/qwen/qwen3_embeddings.py", "Qwen3RerankerLayer.call", "softmax()"),),
        _probe_qwen3_reranker),
}


def _images(shape) -> np.ndarray:
    return np.random.default_rng(0).random(shape).astype("float32")


def _guide_exceptions() -> Set[str]:
    """The names listed under the GUIDE logits bullet's "The exceptions are exactly these"."""
    with open(GUIDE, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    start = next(i for i, line in enumerate(lines) if "The exceptions are exactly" in line)
    names = set()
    for line in lines[start + 1:]:
        match = re.match(r"^  - `([A-Za-z0-9_]+)`: ", line)
        if match:
            names.add(match.group(1))
        elif names and line.strip() and not line.startswith("    "):
            break  # the first line after the sub-list ends it
    return names


def test_the_census_sees_the_known_population():
    """Anti-vacuity: an instrument that stopped matching would pass everything below."""
    found = census()
    assert len(found) >= 40, len(found)
    assert ("language/qwen/qwen3_embeddings.py", "Qwen3RerankerLayer.call", "softmax()") in found


def test_every_census_site_is_classified_exactly_once():
    exception_keys = {key for keys, _ in CLASS_SCORE_EXCEPTIONS.values() for key in keys}
    tables = [set(INTERNAL), set(RANGE_MAPS), exception_keys]
    for i, a in enumerate(tables):
        for b in tables[i + 1:]:
            assert not a & b, f"classified twice: {sorted(a & b)}"
    classified = set().union(*tables)
    found = census()
    unclassified = found - classified
    assert not unclassified, (
        "softmax/sigmoid sites in models/ that no table names. A classifier output "
        "returns logits (models/GUIDE.md); a genuine range map or internal weight "
        f"gets a row with its reason: {sorted(unclassified)}")
    stale = classified - found
    assert not stale, f"table rows the census no longer finds, remove them: {sorted(stale)}"


def test_the_exception_list_equals_the_guide_in_both_directions():
    guide = _guide_exceptions()
    assert guide, "no exception list parsed from models/GUIDE.md"
    assert guide == set(CLASS_SCORE_EXCEPTIONS), (
        f"GUIDE only: {sorted(guide - set(CLASS_SCORE_EXCEPTIONS))}; "
        f"census only: {sorted(set(CLASS_SCORE_EXCEPTIONS) - guide)}")


@pytest.mark.parametrize("name", sorted(CLASS_SCORE_EXCEPTIONS))
def test_each_named_exception_emits_the_probability_its_reason_states(name):
    keras.utils.set_random_seed(0)
    CLASS_SCORE_EXCEPTIONS[name][1]()


def test_a_planted_softmax_head_is_found(tmp_path):
    package = tmp_path / "planted"
    package.mkdir()
    (package / "model.py").write_text(
        "import keras\n"
        "class Planted(keras.Model):\n"
        "    \"\"\"A docstring naming 'softmax' is prose and is not counted.\"\"\"\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.head = keras.layers.Dense(3, activation='softmax')\n"
        "        kind = 'sigmoid' if True else None\n")
    assert census(str(tmp_path)) == {
        ("planted/model.py", "Planted.__init__", "activation=softmax"),
        ("planted/model.py", "Planted.__init__", "'sigmoid'"),
    }


def test_every_shared_output_layer_is_scanned_and_still_returned():
    """D-159: each shared layer row is scanned and its model still imports it."""
    found = census()
    for rel, (model_rel, class_name) in SHARED_OUTPUT_LAYERS.items():
        assert os.path.isfile(os.path.join(SRC_ROOT, rel)), rel
        with open(os.path.join(MODELS_ROOT, model_rel), encoding="utf-8") as handle:
            source = handle.read()
        assert re.search(rf"^from dl_techniques\.{rel[:-3].replace('/', '.')} import .*\b{class_name}\b",
                         source, re.MULTILINE), (
            f"{model_rel} no longer imports {class_name} from {rel}; remove or update the row")
    shared = {key for key in found if key[0].startswith("src:")}
    assert all(key[0][4:] in SHARED_OUTPUT_LAYERS for key in shared), shared
