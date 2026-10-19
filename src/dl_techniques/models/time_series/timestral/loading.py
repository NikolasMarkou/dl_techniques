"""Load and write checkpoints: a directory with config.json and model.safetensors.

Pure Keras 3 + NumPy; no PyTorch anywhere. The weight file uses the reference
PyTorch state-dict names and layouts (github.com/mistralai/timestral), the format
of the released Hugging Face checkpoints, so a released checkpoint and one written
by ``save_checkpoint`` load the same way. Names are derived from the model's layer
structure (``reference_weight_map``), never from ``Variable.path``, so they do not
depend on whether the model was built lazily, explicitly or by this loader.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import keras
import numpy as np

from dl_techniques.models.time_series.timestral.config import TimestralConfig
from dl_techniques.models.time_series.timestral.model import TimestralModel

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"

# Layouts of a reference tensor relative to the Keras variable holding it.
COPY = "copy"            # same array
LINEAR = "linear"        # torch Linear weight (out, in) is the Keras kernel (in, out) transposed
QKV = "qkv"              # LINEAR, and the fused output axis is reordered (see _qkv_order)
QKV_BIAS = "qkv_bias"    # the fused projection's bias, reordered the same way

_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_, "U16": np.uint16, "U32": np.uint32,
    "U64": np.uint64,
}
_CODES = {np.dtype(v): k for k, v in _DTYPES.items()}


def resolve_checkpoint(
    path_or_repo_id: Union[str, Path],
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    token: Optional[Union[str, bool]] = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local directory holding config.json and model.safetensors.

    An existing directory is used as is. Anything else is a Hugging Face Hub repo id
    (e.g. mistralai/Timestral-209M): its two files are downloaded to the Hub cache,
    or read from it when local_files_only (or HF_HUB_OFFLINE=1) is set.
    """
    path = Path(path_or_repo_id).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        raise NotADirectoryError(
            f"{str(path)!r} is a file; pass the export directory holding "
            f"{CONFIG_FILE} and {WEIGHTS_FILE}"
        )
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import (
            GatedRepoError, HFValidationError, RepositoryNotFoundError,
        )
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to load a Hub checkpoint; "
            "install with: pip install huggingface_hub"
        ) from e
    try:
        return Path(
            snapshot_download(
                str(path_or_repo_id),
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=[CONFIG_FILE, WEIGHTS_FILE],
            )
        )
    except GatedRepoError:
        raise  # The Hub's message says how to request access
    except (RepositoryNotFoundError, HFValidationError) as e:
        raise FileNotFoundError(
            f"{str(path_or_repo_id)!r} is neither a local directory nor a "
            f"Hugging Face Hub repo you can access"
        ) from e


def load_safetensors(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read a safetensors file without torch.

    Parses the 8-byte little-endian header (JSON with one entry per tensor:
    dtype, shape, byte offsets), then slices the raw buffer. Safe dtypes only.
    """
    path = Path(path)
    with path.open("rb") as f:
        header_len = int(np.frombuffer(f.read(8), dtype="<u8")[0])
        header = json.loads(f.read(header_len).decode("utf-8"))
        buf = f.read()

    out: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        arr = np.frombuffer(buf[start:end], dtype=_DTYPES[meta["dtype"]])
        if shape:
            arr = arr.reshape(shape)
        out[name] = arr.copy()  # detach from the big buffer
    return out


def save_safetensors(tensors: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
    """Write a safetensors file without torch; the inverse of ``load_safetensors``.

    Tensors are stored little-endian and C-contiguous, in sorted name order.
    """
    header: Dict[str, dict] = {}
    chunks = []
    offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
        data = arr.tobytes()
        header[name] = {
            "dtype": _CODES[np.dtype(arr.dtype.name)],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)  # the format pads the header to 8 bytes
    with Path(path).open("wb") as f:
        f.write(np.uint64(len(raw)).astype("<u8").tobytes())
        f.write(raw)
        for data in chunks:
            f.write(data)


def reference_weight_map(model: TimestralModel) -> Dict[str, Tuple[keras.Variable, str]]:
    """Map every reference state-dict name to the Keras variable holding it.

    Interface contract (shared by ``load_model``, ``save_checkpoint`` and the tests):

    Args:
        model: A built ``TimestralModel`` (``model.build(...)`` or one forward).

    Returns:
        ``{reference name: (variable, layout)}``, one entry per persisted reference
        tensor; ``layout`` is one of ``COPY``, ``LINEAR``, ``QKV``, ``QKV_BIAS``.
        The RoPE table is not listed: the reference does not persist it either
        (a non-persistent buffer), and ``SinCosRoPE.build`` recomputes it.

    Raises:
        ValueError: If the model is not built.
    """
    if not (model.head.built and model.representation.built and model.transformer.built):
        raise ValueError("Build the model (model.build(...) or one forward) before mapping it")
    out: Dict[str, Tuple[keras.Variable, str]] = {}

    def dense(name: str, layer: keras.layers.Dense, kernel_layout: str = LINEAR) -> None:
        out[f"{name}.weight"] = (layer.kernel, kernel_layout)
        if layer.use_bias:
            out[f"{name}.bias"] = (layer.bias, QKV_BIAS if kernel_layout == QKV else COPY)

    def residual_block(name: str, block) -> None:
        for part in ("hidden_layer", "output_layer", "residual_layer"):
            dense(f"{name}.{part}", getattr(block, part))

    residual_block("representation.patch_proj", model.representation.patch_proj)
    out["representation.forecast_query"] = (model.representation.forecast_query, COPY)
    for i, block in enumerate(model.transformer.layers):
        prefix = f"transformer.layers.{i}"
        for j, (norm, attn) in enumerate(zip(block.attn_norms, block.attns)):
            out[f"{prefix}.attn_norms.{j}.scale"] = (norm.scale, COPY)
            dense(f"{prefix}.attns.{j}.wQKV", attn.wqkv, QKV)
            dense(f"{prefix}.attns.{j}.wO", attn.wo)
            if attn.gate is not None:
                dense(f"{prefix}.attns.{j}._gate", attn.gate)
        out[f"{prefix}.mlp_norm.scale"] = (block.mlp_norm.scale, COPY)
        dense(f"{prefix}.mlp.w12", block.mlp.w12)
        dense(f"{prefix}.mlp.w3", block.mlp.w3)
    out["transformer.out_norm.scale"] = (model.transformer.out_norm.scale, COPY)
    residual_block("head.projection", model.head.projection)
    return out


def _qkv_order(rows: np.ndarray, num_heads: int, to_keras: bool) -> np.ndarray:
    """Reorder the fused projection's output axis (axis 0) between the two layouts.

    The reference splits the fused output as (3, head_dim, heads)
    (``reshape(batch, seq, 3, head_dim, num_heads)``); ``Attention._qkv`` here
    splits it as (3, heads, head_dim).
    """
    width = rows.shape[0] // 3
    head_dim = width // num_heads
    src = (3, head_dim, num_heads) if to_keras else (3, num_heads, head_dim)
    rest = rows.shape[1:]
    return rows.reshape(src + rest).transpose((0, 2, 1) + tuple(range(3, 3 + len(rest)))).reshape(rows.shape)


def to_keras_layout(array: np.ndarray, layout: str, num_heads: int) -> np.ndarray:
    """Convert one reference tensor to the layout of its Keras variable."""
    if layout == COPY:
        return array
    if layout == LINEAR:
        return array.T
    if layout == QKV:
        return _qkv_order(array, num_heads, to_keras=True).T
    if layout == QKV_BIAS:
        return _qkv_order(array, num_heads, to_keras=True)
    raise ValueError(f"Unknown layout {layout!r}")


def to_reference_layout(array: np.ndarray, layout: str, num_heads: int) -> np.ndarray:
    """Convert one Keras variable value to the reference layout; inverse of ``to_keras_layout``."""
    if layout == COPY:
        return array
    if layout == LINEAR:
        return array.T
    if layout == QKV:
        return _qkv_order(array.T, num_heads, to_keras=False)
    if layout == QKV_BIAS:
        return _qkv_order(array, num_heads, to_keras=False)
    raise ValueError(f"Unknown layout {layout!r}")


def load_model(
    path_or_repo_id: Union[str, Path],
    **hub_kwargs,
) -> TimestralModel:
    """Build the model from config.json and load model.safetensors.

    path_or_repo_id is a local directory or a Hugging Face Hub repo id; hub_kwargs
    go to resolve_checkpoint. The model is built explicitly (so it is marked built
    and ``model.save`` keeps every weight), then every reference tensor is assigned
    through ``reference_weight_map``; a missing, unused or misshapen tensor raises.
    The model is float32, the precision inference was validated in; there is no
    dtype argument (one was accepted and ignored until D-173).

    Raises:
        ValueError: If a tensor is missing, unused or has the wrong shape.
        TypeError: If a keyword other than resolve_checkpoint's is passed.
    """
    path = resolve_checkpoint(path_or_repo_id, **hub_kwargs)
    config = TimestralConfig.load(path / CONFIG_FILE)
    model = TimestralModel(config)
    model.build((None, 2 * config.patch_size))

    weights = load_safetensors(path / WEIGHTS_FILE)
    mapping = reference_weight_map(model)
    missing = sorted(set(mapping) - set(weights))
    if missing:
        raise ValueError(f"Checkpoint is missing variables: {missing}")
    unused = sorted(set(weights) - set(mapping))
    if unused:
        raise ValueError(f"Checkpoint has unused entries (name mismatch?): {unused}")
    for name, (variable, layout) in mapping.items():
        value = to_keras_layout(weights[name], layout, config.num_heads)
        if tuple(value.shape) != tuple(variable.shape):
            raise ValueError(
                f"{name}: checkpoint shape {weights[name].shape} does not fit "
                f"{variable.path} {tuple(variable.shape)}"
            )
        variable.assign(keras.ops.cast(value, variable.dtype))
    return model


def save_checkpoint(model: TimestralModel, directory: Union[str, Path]) -> Path:
    """Write config.json and model.safetensors in the reference format.

    The inverse of ``load_model``: ``load_model(save_checkpoint(m, d))`` restores
    every weight of ``m`` exactly. The directory is created if needed.

    Raises:
        ValueError: If the model is not built.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: to_reference_layout(
            keras.ops.convert_to_numpy(variable), layout, model.config.num_heads)
        for name, (variable, layout) in reference_weight_map(model).items()
    }
    model.config.save(directory / CONFIG_FILE)
    save_safetensors(tensors, directory / WEIGHTS_FILE)
    return directory
