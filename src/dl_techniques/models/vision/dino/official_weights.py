"""Load the official DINOv2 weights into `DINOv2VisionTransformer`, strictly.

The official checkpoints are the Hugging Face conversions of Meta's DINOv2
ViT-S/14, ViT-B/14 and ViT-L/14 (``facebook/dinov2-small``, ``-base``,
``-large``, no register tokens), the encoders Depth Anything V1 starts from.
No weights ship with this repository: each ``model.safetensors`` is
downloaded once to a local directory, :data:`DINOV2_WEIGHTS_DIR` or the
directory named by the ``DL_TECHNIQUES_DINOV2_WEIGHTS_DIR`` environment
variable, as ``<dir>/dinov2-<variant>/model.safetensors``::

    curl -L -o <dir>/dinov2-small/model.safetensors \\
        https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors

The load is strict: :func:`official_weight_map` pairs every official tensor
with exactly one backbone variable (transposing torch's ``Linear`` and
``Conv2d`` layouts, fusing ``query``/``key``/``value`` into the backbone's one
``qkv`` kernel), and :func:`load_official_dinov2_weights` raises before
assigning anything if an official tensor is missing, left over or of another
shape, or if a backbone weight would stay unfilled. The backbone must have
the official structure (patch 14, ``pos_embed_grid_size=37``, no register
tokens, LayerScale on, MLP FFN, final norm); `DINOv2VisionTransformer.from_variant`
builds it that way when ``pretrained=True``.

References:
    - Oquab et al., 2023. DINOv2: Learning Robust Visual Features without
      Supervision. (https://arxiv.org/abs/2304.07193)
    - facebookresearch/dinov2, ``dinov2/models/vision_transformer.py``.
    - huggingface/transformers, ``models/dinov2/modeling_dinov2.py`` (the
      tensor names of the safetensors files).
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

__all__ = [
    "DINOV2_WEIGHTS_DIR",
    "DINOV2_WEIGHTS_DIR_ENV",
    "OFFICIAL_DINOV2_CHECKPOINTS",
    "OFFICIAL_DINOV2_POS_GRID",
    "official_dinov2_weights_path",
    "official_weight_map",
    "load_official_dinov2_weights",
    "load_official_dinov2_file",
]

#: Where the downloaded checkpoints live unless the environment variable says otherwise.
DINOV2_WEIGHTS_DIR = "/media/arxwn/data0_4tb/datasets/pretrained/dinov2"

#: The environment variable that overrides :data:`DINOV2_WEIGHTS_DIR`.
DINOV2_WEIGHTS_DIR_ENV = "DL_TECHNIQUES_DINOV2_WEIGHTS_DIR"

#: ``variant -> (Hugging Face repository, sha256 of its model.safetensors)``.
#: The hashes are the files' published ``X-Linked-ETag`` values (fetched
#: 2026-10-17; small at repository commit ed25f3a3, base f9e44c81, large 47b73eef).
OFFICIAL_DINOV2_CHECKPOINTS: Dict[str, Tuple[str, str]] = {
    "small": ("facebook/dinov2-small",
              "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"),
    "base": ("facebook/dinov2-base",
             "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"),
    "large": ("facebook/dinov2-large",
              "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"),
}

#: Side of the official square position grid (518 px at patch 14).
OFFICIAL_DINOV2_POS_GRID = 37

# One entry of the weight map: (official name, backbone variable, torch-to-Keras transform).
WeightMapEntry = Tuple[str, keras.Variable, Callable[[np.ndarray], np.ndarray]]

# ---------------------------------------------------------------------


def official_dinov2_weights_path(variant: str) -> Path:
    """The local ``model.safetensors`` of an official DINOv2 variant.

    Interface contract: returns the path for ``small``, ``base`` or
    ``large``; raises ``NotImplementedError`` for any other variant (no
    official weights are wired for it) and ``FileNotFoundError`` naming the
    path, the environment variable and the download URL when the file is
    absent. Touches nothing but the file system.

    :param variant: A `DINOv2VisionTransformer.MODEL_VARIANTS` key.
    :type variant: str
    :return: The checkpoint path.
    :rtype: Path
    """
    if variant not in OFFICIAL_DINOV2_CHECKPOINTS:
        raise NotImplementedError(
            f"No official DINOv2 weights are wired for variant '{variant}' "
            f"(wired: {sorted(OFFICIAL_DINOV2_CHECKPOINTS)}). Build it with "
            f"pretrained=False.")
    repo = OFFICIAL_DINOV2_CHECKPOINTS[variant][0]
    root = Path(os.environ.get(DINOV2_WEIGHTS_DIR_ENV) or DINOV2_WEIGHTS_DIR)
    path = root / repo.split("/")[1] / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(
            f"Official DINOv2-{variant} weights not found at {path}. Download "
            f"https://huggingface.co/{repo}/resolve/main/model.safetensors to "
            f"that path, or point {DINOV2_WEIGHTS_DIR_ENV} at the directory "
            f"holding '{repo.split('/')[1]}/model.safetensors'.")
    return path


def _transpose(t: np.ndarray) -> np.ndarray:
    return np.transpose(t)


def _same(t: np.ndarray) -> np.ndarray:
    return t


def official_weight_map(backbone: keras.Model) -> List[WeightMapEntry]:
    """Pair every official tensor name with the backbone variable it fills.

    Interface contract: ``backbone`` is a built `DINOv2VisionTransformer`.
    Returns one entry per official tensor, ``(name, variable, transform)``,
    where ``transform`` maps the official array (torch layout) to the
    variable's layout. The fused ``qkv`` kernel and bias appear once under
    the ``query`` name with a ``transform`` that reads the whole tensor dict
    (see :func:`load_official_dinov2_weights`); ``key`` and ``value`` map to
    the same variable with ``None`` as the transform, so every official name
    is listed. Raises ``ValueError`` when the backbone lacks a layer the
    official checkpoint has (no LayerScale, no final norm, no learned
    resampled position table).

    :param backbone: The backbone to fill.
    :type backbone: keras.Model
    :return: The weight map.
    :rtype: List[WeightMapEntry]
    """
    if backbone.norm is None or not backbone.transformer_blocks:
        raise ValueError("the official checkpoint has a final norm; build with include_top=True")
    if getattr(backbone, "pos_embed_grid_size", None) is None:
        raise ValueError(
            "the official position table is a square grid resampled to the input; "
            f"build with pos_embed_grid_size={OFFICIAL_DINOV2_POS_GRID}")
    entries: List[WeightMapEntry] = [
        ("embeddings.cls_token", backbone.cls_token_layer.weights[0], _same),
        ("embeddings.mask_token", backbone.mask_token_layer.weights[0],
         lambda t: t.reshape(1, 1, -1)),
        ("embeddings.position_embeddings", backbone.pos_embed.pos_embed, _same),
        ("embeddings.patch_embeddings.projection.weight", backbone.patch_embed.proj.kernel,
         lambda t: np.transpose(t, (2, 3, 1, 0))),
        ("embeddings.patch_embeddings.projection.bias", backbone.patch_embed.proj.bias, _same),
        ("layernorm.weight", backbone.norm.gamma, _same),
        ("layernorm.bias", backbone.norm.beta, _same),
    ]
    for i, block in enumerate(backbone.transformer_blocks):
        if block.ls1 is None or block.ffn_type != "mlp":
            raise ValueError(
                "the official blocks have LayerScale and an MLP FFN; build with "
                "init_values set and ffn_type='mlp'")
        p = f"encoder.layer.{i}."
        engine = block.attention.cross_attention
        entries += [
            (p + "norm1.weight", block.norm1.gamma, _same),
            (p + "norm1.bias", block.norm1.beta, _same),
            (p + "attention.attention.query.weight", engine.qkv_dense.kernel, _same),
            (p + "attention.attention.key.weight", engine.qkv_dense.kernel, None),
            (p + "attention.attention.value.weight", engine.qkv_dense.kernel, None),
            (p + "attention.attention.query.bias", engine.qkv_dense.bias, _same),
            (p + "attention.attention.key.bias", engine.qkv_dense.bias, None),
            (p + "attention.attention.value.bias", engine.qkv_dense.bias, None),
            (p + "attention.output.dense.weight", engine.proj_dense.kernel, _transpose),
            (p + "attention.output.dense.bias", engine.proj_dense.bias, _same),
            (p + "layer_scale1.lambda1", block.ls1.gamma, _same),
            (p + "norm2.weight", block.norm2.gamma, _same),
            (p + "norm2.bias", block.norm2.beta, _same),
            (p + "mlp.fc1.weight", block.ffn.fc1.kernel, _transpose),
            (p + "mlp.fc1.bias", block.ffn.fc1.bias, _same),
            (p + "mlp.fc2.weight", block.ffn.fc2.kernel, _transpose),
            (p + "mlp.fc2.bias", block.ffn.fc2.bias, _same),
            (p + "layer_scale2.lambda1", block.ls2.gamma, _same),
        ]
    return entries


def _official_value(name: str, transform, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
    """The array an entry assigns: fused q/k/v, or one tensor transformed."""
    if ".attention.attention.query." in name:
        parts = [tensors[name.replace(".query.", f".{x}.")] for x in ("query", "key", "value")]
        if name.endswith(".weight"):  # torch (out, in) -> Keras (in, 3 * out), [q | k | v]
            return np.concatenate([np.transpose(t) for t in parts], axis=1)
        return np.concatenate(parts, axis=0)
    return transform(tensors[name])


def load_official_dinov2_weights(
        backbone: keras.Model,
        tensors: Mapping[str, np.ndarray],
) -> None:
    """Assign an official DINOv2 tensor dict to a backbone, strictly.

    Interface contract: checks everything before assigning anything. Raises
    ``ValueError`` listing the offenders if an official tensor the map needs
    is missing from ``tensors``, if ``tensors`` holds a name the map does not
    use, if any mapped value's shape differs from its variable's, or if a
    backbone weight is filled by no entry or by two. On success every
    backbone weight holds its official value (cast to the variable's dtype).

    :param backbone: A built `DINOv2VisionTransformer` of the official structure.
    :type backbone: keras.Model
    :param tensors: Official name -> array (torch layout), e.g. from
        ``safetensors.numpy.load_file``.
    :type tensors: Mapping[str, np.ndarray]
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-137
    # Do NOT relax this to a partial load (skip_mismatch, by-name best effort,
    # warn and continue): a tensor that silently stays at its initialization
    # is a randomly initialized model reported as pretrained. Check every name,
    # shape and fill first; assign only when all of them hold.
    entries = official_weight_map(backbone)
    names = [name for name, _, _ in entries]
    missing = sorted(set(names) - set(tensors))
    unused = sorted(set(tensors) - set(names))
    if missing or unused:
        raise ValueError(
            f"official DINOv2 tensors do not match the backbone: missing {missing[:8]} "
            f"({len(missing)}), unused {unused[:8]} ({len(unused)})")
    fills: Dict[int, List[str]] = {}
    values = []
    mismatched = []
    for name, variable, transform in entries:
        if transform is None:  # key/value: fused into the query entry's value
            continue
        value = _official_value(name, transform, tensors)
        if tuple(value.shape) != tuple(variable.shape):
            mismatched.append(f"{name}: {tuple(value.shape)} -> {variable.path} {tuple(variable.shape)}")
        fills.setdefault(id(variable), []).append(name)
        values.append((variable, value))
    unfilled = [v.path for v in backbone.weights if id(v) not in fills]
    doubled = [f"{names}" for names in fills.values() if len(names) > 1]
    if mismatched or unfilled or doubled:
        raise ValueError(
            f"official DINOv2 tensors do not fit the backbone: shape mismatches "
            f"{mismatched[:8]}, unfilled backbone weights {unfilled[:8]}, "
            f"weights filled twice {doubled[:8]}")
    for variable, value in values:
        variable.assign(value.astype(variable.dtype))


def load_official_dinov2_file(backbone: keras.Model, variant: str) -> Path:
    """Read the variant's local checkpoint and load it strictly.

    Interface contract: :func:`official_dinov2_weights_path` then
    :func:`load_official_dinov2_weights`, raising whatever they raise
    (``NotImplementedError``, ``FileNotFoundError``, ``ValueError``), and
    ``ImportError`` without the ``safetensors`` package.

    :param backbone: A built `DINOv2VisionTransformer` of the official structure.
    :type backbone: keras.Model
    :param variant: ``small``, ``base`` or ``large``.
    :type variant: str
    :return: The path that was loaded.
    :rtype: Path
    """
    path = official_dinov2_weights_path(variant)
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "loading the official DINOv2 weights needs the 'safetensors' package, "
            "declared in the 'data' extra: pip install '.[data]'") from exc
    load_official_dinov2_weights(backbone, load_file(str(path)))
    logger.info(f"Loaded official DINOv2-{variant} weights from {path}")
    return path

# ---------------------------------------------------------------------
