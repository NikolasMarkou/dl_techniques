"""Load exported checkpoints: a directory with config.json and model.safetensors.

Pure Keras 3 + NumPy (+ safetensors for weight files); no PyTorch anywhere.
An export directory holds model.safetensors and config.json. Weights map 1:1 onto
the Keras model's variable names, so they are loaded by name and assigned.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import keras
from keras import ops

from dl_techniques.models.time_series.timestral.config import TimestralConfig
from dl_techniques.models.time_series.timestral.model import TimestralModel

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


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


def load_safetensors(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Read a safetensors file without torch.

    Parses the 8-byte little-endian header (JSON with one entry per tensor:
    dtype, shape, byte offsets), then slices the raw buffer. Safe dtypes only.
    """
    path = Path(path)
    with path.open("rb") as f:
        header_len = int(np.frombuffer(f.read(8), dtype="<u8")[0])
        header = json.loads(f.read(header_len).decode("utf-8"))
        buf = f.read()

    _DTYPES = {
        "F64": np.float64, "F32": np.float32, "F16": np.float16,
        "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
        "U8": np.uint8, "BOOL": np.bool_, "U16": np.uint16, "U32": np.uint32,
        "U64": np.uint64,
    }

    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta["dtype"]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        arr = np.frombuffer(buf[start:end], dtype=_DTYPES[dtype])
        if shape:
            arr = arr.reshape(shape)
        out[name] = arr.copy()  # detach from the big buffer
    return out


def load_model(
    path_or_repo_id: Union[str, Path],
    dtype: str = "float32",
    **hub_kwargs,
) -> TimestralModel:
    """Build the model from config.json and load model.safetensors.

    path_or_repo_id is a local directory or a Hugging Face Hub repo id; hub_kwargs
    go to resolve_checkpoint. Weights are matched to variables by name and
    assigned. Inference was validated in float32.
    """
    path = resolve_checkpoint(path_or_repo_id, **hub_kwargs)
    config = TimestralConfig.load(path / CONFIG_FILE)
    model = TimestralModel(config)

    weights = load_safetensors(path / WEIGHTS_FILE)

    # Build the model by running one dummy forward pass with random weights.
    rng = np.random.default_rng(0)
    context = rng.standard_normal((1, 2 * config.patch_size)).astype("float32")
    observed = np.ones((1, 2 * config.patch_size), dtype=bool)
    group_ids = np.zeros(1, dtype=np.int32)
    model.predict_block(
        keras.ops.convert_to_tensor(context),
        keras.ops.convert_to_tensor(observed),
        keras.ops.convert_to_tensor(group_ids),
        horizon_patches=1,
    )

    by_name = {v.path: v for v in model.weights}
    missing = []
    for var in model.weights:
        if var.path in weights:
            var.assign(ops.cast(weights[var.path], var.dtype))
            del weights[var.path]
        else:
            missing.append(var.path)
    if missing:
        raise ValueError(f"Checkpoint is missing variables: {missing}")
    if weights:
        raise ValueError(
            f"Checkpoint has unused entries (name mismatch?): {sorted(weights)}"
        )
    return model
