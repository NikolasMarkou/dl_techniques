"""Config serialization helpers shared across training scripts.

Promoted from ~12 inline ``json.dump(config.__dict__/asdict(...))`` copies plus
3 ``json_numpy_default`` re-implementations (plan_2026-06-02_30721a0f, F4) so
trainers persist their run config through one canonical helper instead of
re-inlining the dataclass-vs-``__dict__`` detection and the numpy-scalar
serialization shim.

D-003: the unified ``json_numpy_default`` emits *native* JSON numbers for numpy
scalars (``float`` / ``int``), not the quoted-string form mdn's local helper
used. Results JSON is a post-hoc artifact, not a training input, so the format
change is non-load-bearing and yields downstream-parseable numbers.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

import numpy as np

from dl_techniques.utils.logger import logger


def json_numpy_default(obj: Any) -> Any:
    """``json.dump(..., default=...)`` callable that serializes numpy scalars.

    Converts numpy numeric / array types to their native JSON-serializable
    equivalents so configs (or results) carrying numpy values do not crash
    ``json.dump``.

    Args:
        obj: The object ``json`` could not serialize natively.

    Returns:
        A native Python value: ``float`` for ``np.floating``, ``int`` for
        ``np.integer``, ``list`` for ``np.ndarray``.

    Raises:
        TypeError: If ``obj`` is none of the handled numpy types (the standard
            ``default=`` contract for an unserializable object).
    """
    # DECISION plan_2026-06-02_30721a0f/D-003: emit NATIVE JSON numbers for
    # numpy scalars. Do NOT restore mdn's str-coercion (np.floating/np.integer
    # -> str) or add a mode="str" flag (single-use param, earned-abstraction
    # violation). Native numbers are downstream-parseable; results JSON is a
    # post-hoc artifact, so the one-time format change is non-load-bearing.
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def save_config_json(
    config: Any,
    results_dir: str,
    filename: str = "config.json",
) -> str:
    """Serialize a run config to ``results_dir/filename`` as JSON.

    Resolves the payload by config shape: a dataclass instance is converted via
    ``dataclasses.asdict`` (recursing nested dataclasses correctly); an object
    with a ``__dict__`` falls back to ``vars(config)``; anything else is assumed
    already JSON-serializable (e.g. a plain dict). Numpy scalars are serialized
    via :func:`json_numpy_default`.

    Args:
        config: The config object to persist. A dataclass instance, an object
            with ``__dict__``, or an already-serializable mapping.
        results_dir: Directory to write into. Created with ``exist_ok=True``.
        filename: Output filename. Defaults to ``"config.json"``.

    Returns:
        The full path to the written JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)

    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        payload = dataclasses.asdict(config)
    elif hasattr(config, "__dict__"):
        payload = vars(config)
    else:
        payload = config

    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_numpy_default)

    logger.info(f"[config_io] saved config to {path}")
    return path


__all__ = ["json_numpy_default", "save_config_json"]
