"""Run-directory and run-artifact helpers shared by the training scripts.

Three pieces of boilerplate were copy-pasted across most trainers in
``src/train`` before this module existed:

- the ``training_history.json`` dump after ``model.fit`` (20 files)
- the ``output_dir / experiment_name`` + ``mkdir`` + ``save_config_json``
  preamble (16 files)
- the ``__post_init__`` "default the experiment name to a timestamped string"
  idiom (~20 dataclass configs, plus inline variants)

All three are here now. ``save_config_json`` / ``json_numpy_default`` remain in
:mod:`train.common.config_io` -- this module is about the run directory and the
artifacts written into it, not about config serialization itself.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dl_techniques.utils.logger import logger
from train.common.config_io import save_config_json

# ---------------------------------------------------------------------

#: The timestamp format every trainer uses for run directories and experiment
#: names. Sortable, filesystem-safe, second-resolution.
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


# ---------------------------------------------------------------------

def run_timestamp() -> str:
    """Return the current time in the repo-standard run-timestamp format.

    Returns:
        A ``"YYYYmmdd_HHMMSS"`` string, e.g. ``"20260812_073145"``.
    """
    return datetime.now().strftime(TIMESTAMP_FORMAT)


# ---------------------------------------------------------------------

def default_experiment_name(*parts: Any) -> str:
    """Build a timestamped experiment name from ``parts``.

    Joins the parts with underscores and appends the current timestamp, which
    is the naming convention every trainer already follows::

        >>> default_experiment_name("vit", "cifar10", "small")
        'vit_cifar10_small_20260812_073145'

    Intended for the ``__post_init__`` default::

        if self.experiment_name is None:
            self.experiment_name = default_experiment_name(
                "vit", self.dataset, self.model_variant)

    Conditional fragments stay at the call site -- pass an already-composed
    string (e.g. ``f"{variant}{'_ds' if deep_supervision else ''}"``) rather
    than teaching this helper about any trainer's flags.

    Args:
        *parts: Name fragments. Each is converted with ``str()``. Empty or
            ``None`` parts are dropped so a disabled optional fragment does not
            leave a doubled underscore.

    Returns:
        The underscore-joined parts followed by the run timestamp.
    """
    kept = [str(p) for p in parts if p is not None and str(p) != ""]
    kept.append(run_timestamp())
    return "_".join(kept)


# ---------------------------------------------------------------------

def prepare_run_dir(
        config: Any,
        output_dir: Optional[Union[str, Path]] = None,
        config_filename: str = "config.json",
) -> Path:
    """Create the run directory and write the config into it.

    Replaces the three-line preamble duplicated across the trainers::

        run_dir = Path(config.output_dir) / config.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config_json(config, str(run_dir), "config.json")

    Args:
        config: The training config. Must expose ``output_dir`` and
            ``experiment_name`` attributes unless ``output_dir`` is given
            explicitly. Passed through to
            :func:`train.common.config_io.save_config_json`, which is
            dataclass-aware and numpy-safe.
        output_dir: Optional fully-resolved run directory. Pass this when the
            trainer computes the path itself (e.g. the SAM trainers'
            ``resolved_output_dir(config)``); otherwise it is derived as
            ``Path(config.output_dir) / config.experiment_name``.
        config_filename: Name of the config file written into the run
            directory.

    Returns:
        The created run directory as a :class:`~pathlib.Path`.
    """
    run_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(config.output_dir) / config.experiment_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(run_dir), config_filename)
    return run_dir


# ---------------------------------------------------------------------

def save_training_history_json(
        history: Any,
        output_dir: Union[str, Path],
        filename: str = "training_history.json",
) -> Optional[Path]:
    """Write a Keras ``History`` to JSON as ``{metric: [per-epoch floats]}``.

    Every value is coerced with ``float()`` so numpy scalars and backend
    tensors serialize as plain JSON numbers.

    This is a BEST-EFFORT artifact, matching the behaviour of the 20 call sites
    it replaces: a failure here is logged as a warning and swallowed rather
    than raised, because losing the history dump must never destroy a run whose
    weights were already saved.

    Args:
        history: A ``keras.callbacks.History`` (anything with a ``.history``
            dict) or a raw ``{metric: values}`` mapping.
        output_dir: Directory to write into. Must already exist.
        filename: Name of the JSON file.

    Returns:
        The written path, or ``None`` if writing failed.
    """
    try:
        raw: Dict[str, Any] = (
            history if isinstance(history, dict) else history.history
        )
        history_dict = {
            key: [float(v) for v in values] for key, values in raw.items()
        }
        path = Path(output_dir) / filename
        with open(path, "w") as handle:
            json.dump(history_dict, handle, indent=2)
        return path
    except Exception as error:  # pragma: no cover - best-effort artifact
        logger.warning(f"Failed to save training history: {error}")
        return None
