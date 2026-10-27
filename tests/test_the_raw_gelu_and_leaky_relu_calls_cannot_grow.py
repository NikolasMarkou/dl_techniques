"""Ratchet: the raw GELU / LeakyReLU spellings under ``src/`` cannot grow.

TensorFlow 2.18 builds GELU's ``sqrt(2)`` and LeakyReLU's slope in float32 even
for a float64 input (D-218): every route below lands on ``tf.nn.gelu`` or
``tf.nn.leaky_relu`` and is off by up to 5.0e-9 (GELU) or 2.2e-8 relative
(slope) at float64. The repo's exact forms
(``layers/activations/exact_activations.py``: ``exact_gelu``,
``exact_leaky_relu``, ``ExactLeakyReLU``) are float64-exact and bit-identical
to stock at float32, and the shared resolvers return them for ``"gelu"`` and
``"leaky_relu"`` (D-277).

Step 2 of plan-2026-10-14T062616-81938e81 iteration 3 migrated the enumerated
sites and stopped there: a tree-wide migration is the mass migration D-082
refused. The rest is pinned here, per file, by an AST walk (docstrings and
comments are never counted). Kinds counted:

* an attribute chain ending in ``ops.gelu``, ``activations.gelu``,
  ``layers.LeakyReLU``, ``activations.leaky_relu``, ``nn.gelu``,
  ``nn.leaky_relu`` or ``ops.leaky_relu`` (``keras.ops.gelu``,
  ``tf.nn.gelu``, ``ops.gelu`` after ``from keras import ops``, ...);
* ``from keras... / tensorflow... import gelu | leaky_relu | LeakyReLU``;
* ``Activation("gelu" | "leaky_relu")``, positional or keyword;
* ``activation="gelu" | "leaky_relu"`` on a stock ``keras.layers.X(...)`` /
  ``layers.X(...)`` call.

A string handed to a REPO layer is not counted: the repo layers resolve it
through the shared resolvers. A migration deletes a file's line here, in the
same commit; a new raw site anywhere fails until it is routed through the exact
forms.
"""

import ast
import pathlib
import warnings
from typing import Dict

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Attribute-chain suffixes that reach TensorFlow's float32-constant kernels.
RAW_SUFFIXES = ("ops.gelu", "activations.gelu", "layers.LeakyReLU",
                "activations.leaky_relu", "nn.gelu", "nn.leaky_relu", "ops.leaky_relu")
RAW_IMPORT_NAMES = frozenset({"gelu", "leaky_relu", "LeakyReLU"})
RAW_STRINGS = frozenset({"gelu", "leaky_relu"})

#: Exact census over ``src/``, ``{path relative to the repo root: raw sites}``,
#: measured on the migrated tree at Step 2.5 of plan-2026-10-14T062616-81938e81
#: iteration 3 (D-277): 22 sites in 18 files (49 before the migration).
RAW_GELU_LEAKY_SITES: Dict[str, int] = {
    # The one legitimate raw spelling: the exact layer's base class.
    "src/dl_techniques/layers/activations/exact_activations.py": 1,
    # Float32 or mixed-precision trainers: the rounding matters only at
    # float64, so these were left raw on purpose (plan Step 2, Files To Modify).
    "src/train/bfunet/train_bfcnn_denoiser.py": 1,
    "src/train/bfunet/train_convunext_denoiser.py": 1,
    "src/train/bfunet/train_unet_denoiser.py": 1,
    # Not enumerated (no registry subject reaches them, D-082); held here.
    "src/dl_techniques/layers/geometric/point_cloud_autoencoder.py": 2,
    "src/dl_techniques/layers/heads/vlm/vqa.py": 1,
    "src/dl_techniques/layers/regularization/router.py": 1,
    "src/dl_techniques/layers/transformers/vision_encoder.py": 1,
    "src/dl_techniques/models/graph/graph_energy_transformer/model.py": 1,
    "src/dl_techniques/models/time_series/adaptive_ema/model.py": 1,
    "src/dl_techniques/models/vision/fftnet/model.py": 1,
    "src/dl_techniques/models/vision/levjepa/blocks.py": 1,
    "src/dl_techniques/models/vision/levjepa/projector.py": 1,
    "src/dl_techniques/models/vision_language/sam/sam3/necks.py": 1,
    "src/train/rms_variants_train/experiments/e4_deep_residual_reg.py": 1,
    "src/train/rms_variants_train/experiments/e5_norm_layer_microbench.py": 1,
    "src/train/som_nd_soft/train_cifar.py": 3,
    "src/train/time_series/mdn/train_mdn.py": 2,
}

#: Files Step 2 migrated (plus ``ffn/swin_mlp.py``, enumerated by the plan and already
#: routed through ``common.resolve_activation``): each must stay at zero raw sites.
MIGRATED_CLEAN = (
    "src/dl_techniques/layers/acc_unet/hanc_block.py",
    "src/dl_techniques/layers/acc_unet/hanc_layer.py",
    "src/dl_techniques/layers/acc_unet/multi_level_feature_compilation.py",
    "src/dl_techniques/layers/conv_blocks/res_path.py",
    "src/dl_techniques/layers/conv_blocks/convnext_v1_block.py",
    "src/dl_techniques/layers/conv_blocks/convnext_v2_block.py",
    "src/dl_techniques/layers/conv_blocks/mobile_one_block.py",
    "src/dl_techniques/layers/fastvit/conv_mlp.py",
    "src/dl_techniques/layers/fastvit/reparam_large_kernel_conv.py",
    "src/dl_techniques/layers/ffn/gated_dconv_ffn.py",
    "src/dl_techniques/layers/ffn/gated_mlp.py",
    "src/dl_techniques/layers/ffn/geglu_ffn.py",
    "src/dl_techniques/layers/ffn/gelu_mlp_ffn.py",
    "src/dl_techniques/layers/ffn/swin_mlp.py",
    "src/dl_techniques/layers/hierarchical_mlp_stem.py",
    "src/dl_techniques/layers/modality_projection.py",
    "src/dl_techniques/layers/structured_linear/crystal_multihead_projection.py",
    "src/dl_techniques/models/common/masked_language_model/mlm.py",
    "src/dl_techniques/models/language/wave_field/model.py",
    "src/dl_techniques/models/vision/convunext/model.py",
    "src/dl_techniques/models/vision/fastvit/model.py",
    "src/dl_techniques/models/vision/image_restoration/darkir/components.py",
    "src/dl_techniques/models/vision/image_restoration/pw_fnet/model.py",
    "src/dl_techniques/models/vision/lewm/projector.py",
    "src/dl_techniques/models/vision/masked_autoencoder/conv_decoder.py",
    "src/dl_techniques/models/vision/super_resolution/pft_sr/blocks.py",
    "src/dl_techniques/models/vision/super_resolution/pft_sr/model.py",
    "src/dl_techniques/models/vision/thera/tails.py",
    "src/dl_techniques/models/vision/vae/model.py",
    "src/dl_techniques/models/vision/video_jepa/predictor.py",
    "src/dl_techniques/models/vision/vit_siglip/model.py",
    "src/dl_techniques/models/vision_language/bit_diffusion/token_decoder.py",
    "src/dl_techniques/models/vision_language/sam/sam1/mask_decoder.py",
    "src/dl_techniques/models/vision_language/sd3_mmdit/text_encoders.py",
)


def _dotted(node: ast.AST):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def raw_sites(tree: ast.AST) -> int:
    """Number of raw GELU / LeakyReLU spellings in a parsed module (kinds in the docstring)."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted and any(dotted == s or dotted.endswith("." + s) for s in RAW_SUFFIXES):
                count += 1
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (
                node.module.split(".")[0] in ("keras", "tensorflow")):
            count += sum(alias.name in RAW_IMPORT_NAMES for alias in node.names)
        elif isinstance(node, ast.Call):
            func = _dotted(node.func) or ""
            if func == "Activation" or func.endswith("layers.Activation"):
                values = list(node.args[:1]) + [k.value for k in node.keywords
                                                 if k.arg == "activation"]
                count += sum(isinstance(v, ast.Constant) and v.value in RAW_STRINGS
                             for v in values)
            elif func.startswith("keras.layers.") or func.startswith("layers."):
                count += sum(k.arg == "activation" and isinstance(k.value, ast.Constant)
                             and k.value.value in RAW_STRINGS for k in node.keywords)
    return count


def _census() -> Dict[str, int]:
    per_file = {}
    for path in (REPO_ROOT / "src").rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
            tree = ast.parse(path.read_text(encoding="utf-8"))
        n = raw_sites(tree)
        if n:
            per_file[path.relative_to(REPO_ROOT).as_posix()] = n
    return per_file


@pytest.fixture(scope="module")
def census():
    return _census()


def test_the_census_is_exact(census):
    if census == RAW_GELU_LEAKY_SITES:
        return
    added = sorted((p, census[p]) for p in set(census) - set(RAW_GELU_LEAKY_SITES))
    gone = sorted(set(RAW_GELU_LEAKY_SITES) - set(census))
    changed = sorted(f"{p}: {RAW_GELU_LEAKY_SITES[p]} -> {census[p]}"
                     for p in set(census) & set(RAW_GELU_LEAKY_SITES)
                     if census[p] != RAW_GELU_LEAKY_SITES[p])
    pytest.fail(
        f"{sum(census.values())} raw GELU / LeakyReLU sites in {len(census)} files; the "
        f"census pins {sum(RAW_GELU_LEAKY_SITES.values())} in {len(RAW_GELU_LEAKY_SITES)}. "
        f"New: {added}; migrated: {gone}; changed: {changed}. A new site: use "
        "exact_gelu / exact_leaky_relu / ExactLeakyReLU, or hand the stock layer "
        "substitute_exact(spec) (layers/activations/exact_activations.py, D-277). "
        "A migration: delete its line here in the same commit.")


def test_every_migrated_file_stays_clean(census):
    dirty = sorted(p for p in MIGRATED_CLEAN if p in census)
    assert dirty == [], f"a migrated file has raw GELU / LeakyReLU sites again: {dirty}"
    for path in MIGRATED_CLEAN + tuple(RAW_GELU_LEAKY_SITES):
        assert (REPO_ROOT / path).is_file(), f"census file moved or deleted: {path}"


@pytest.mark.parametrize("source, expected", [
    ("import keras\n", 0),
    ("y = keras.ops.gelu(x)\n", 1),
    ("y = keras.ops.gelu(x, approximate=False)\n", 1),
    ("y = keras.activations.gelu(x)\n", 1),
    ("f = keras.activations.gelu\n", 1),
    ("y = ops.gelu(x)\n", 1),
    ("y = tf.nn.gelu(x)\n", 1),
    ("y = tf.nn.leaky_relu(x)\n", 1),
    ("y = keras.ops.leaky_relu(x)\n", 1),
    ("y = keras.activations.leaky_relu(x, negative_slope=0.1)\n", 1),
    ("l = keras.layers.LeakyReLU(negative_slope=0.01)\n", 1),
    ("from keras.layers import LeakyReLU\n", 1),
    ("from keras.activations import gelu, relu\n", 1),
    ("l = keras.layers.Activation('gelu')\n", 1),
    ("l = keras.layers.Activation(activation='leaky_relu')\n", 1),
    ("l = keras.layers.Dense(4, activation='gelu')\n", 1),
    ("l = layers.Conv2D(4, 3, activation='leaky_relu')\n", 1),
    # not raw
    ("l = keras.layers.Activation(exact_gelu)\n", 0),
    ("l = keras.layers.Activation(substitute_exact(spec))\n", 0),
    ("l = keras.layers.Dense(4, activation='relu')\n", 0),
    ("l = MLPBlock(4, activation='gelu')\n", 0),
    ("y = exact_gelu(x)\n", 0),
    ("l = ExactLeakyReLU(0.01)\n", 0),
    ('"""keras.ops.gelu(x)"""\n', 0),
    ("# keras.ops.gelu(x)\n", 0),
    ("from .layers import LeakyReLU\n", 0),
    ("y = gelu_tanh(x)\n", 0),
])
def test_the_counter_on_planted_cases(source, expected):
    assert raw_sites(ast.parse(source)) == expected
