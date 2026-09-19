"""Run-directory guards and artifacts shared by the classification trainers.

Three pieces that only ``train_power_mlp.py`` owned until a second trainer needed
them:

- :func:`refuse_existing_run` - a reused ``--experiment-name`` is refused before
  anything is written.
- :func:`attach_run_log` - tee the ``dl`` logger into ``<run_dir>/run.log`` for the
  duration of a ``with`` block.
- :func:`write_summary_json` - ``results_summary.json`` as STRICT JSON.

Only the standard library, numpy and the repo logger are imported (no TensorFlow /
Keras), so importing this module allocates nothing.
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence

import numpy as np

from dl_techniques.utils.logger import LOGGER_FORMAT, logger
from train.common.config_io import json_numpy_default

# ---------------------------------------------------------------------

#: The run's own narrative: the ``dl`` logger, tee'd into ``<run_dir>/run.log``.
RUN_LOG_NAME = "run.log"

#: The strict-JSON summary written at the end of (or on divergence in) a run.
RESULTS_SUMMARY_NAME = "results_summary.json"

#: Files whose presence means an experiment directory already holds a run.
RUN_ARTIFACT_NAMES: Sequence[str] = (
    RESULTS_SUMMARY_NAME, "config.json", "best_model.keras", RUN_LOG_NAME, "training_log.csv",
)


# ---------------------------------------------------------------------

# DECISION plan-2026-09-18T213948-68dcb72c/D-029: a reused --experiment-name is REFUSED,
# never merged, overwritten, deleted or auto-suffixed (``_r2``): results/ is gitignored,
# so an overwrite is unrecoverable and a silent rename changes the name the user asked
# for. Guard: test_a_reused_experiment_name_is_refused_and_the_first_run_is_byte_identical.
def refuse_existing_run(
        run_dir: Path, artifact_names: Sequence[str] = RUN_ARTIFACT_NAMES
) -> None:
    """Raise ``FileExistsError`` when ``run_dir`` already holds a run's files.

    Nothing is written, overwritten or deleted, ever: results are unrecoverable
    (gitignored). A missing or empty directory, or one holding only unrelated
    files, is fine. Call it BEFORE creating the directory or writing anything.

    Args:
        run_dir: The experiment directory about to be used (may not exist yet).
        artifact_names: File names whose presence marks a finished or started run;
            defaults to :data:`RUN_ARTIFACT_NAMES`.

    Raises:
        FileExistsError: Naming ``run_dir`` and the files found, and telling the
            caller to choose a new ``--experiment-name``.
    """
    found = [name for name in artifact_names if (run_dir / name).exists()]
    if found:
        raise FileExistsError(
            f"Experiment directory {run_dir} already holds a run ({', '.join(found)}). "
            "Nothing was written. Choose a new --experiment-name (or omit it for a "
            "timestamped name); existing results are never overwritten or deleted."
        )


@contextmanager
def attach_run_log(run_dir: Path, log_name: str = RUN_LOG_NAME) -> Iterator[logging.Handler]:
    """Tee the ``dl`` logger into ``<run_dir>/<log_name>`` inside a ``with`` block.

    The file is opened in write mode with the repo ``LOGGER_FORMAT``. The handler
    is detached and closed when the block exits, on return AND on an exception,
    so a later run in the same process never writes into this run's log.

    Args:
        run_dir: An EXISTING run directory (a missing one raises
            ``FileNotFoundError`` from the file handler).
        log_name: File name inside ``run_dir``; defaults to :data:`RUN_LOG_NAME`.

    Yields:
        The attached :class:`logging.FileHandler`.
    """
    handler = logging.FileHandler(run_dir / log_name, mode="w")
    handler.setFormatter(logging.Formatter(LOGGER_FORMAT))
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        handler.close()


def write_summary_json(run_dir: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``results_summary.json`` as STRICT JSON and return what was written.

    Every summary (normal or diverged) goes through here. The dict is
    round-tripped through ``json`` (numpy values via ``json_numpy_default``),
    every non-finite float becomes ``None`` (``null``), and the dump uses
    ``allow_nan=False`` so a ``NaN`` / ``Infinity`` token can never reach the
    file (jq and most non-Python readers reject them).

    Args:
        run_dir: Existing run directory.
        summary: The summary dict (may hold numpy scalars / arrays).

    Returns:
        The sanitized, pure-JSON dict that was written.

    Raises:
        TypeError: If ``summary`` holds a value that is neither JSON nor numpy.
        ValueError: If a non-finite float survived the sanitizing (nothing is
            written in that case).
    """
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    written = clean(json.loads(json.dumps(summary, default=json_numpy_default)))
    text = json.dumps(written, indent=2, allow_nan=False)  # raises BEFORE the file is opened
    path = run_dir / RESULTS_SUMMARY_NAME
    path.write_text(text)
    logger.info(f"Wrote {path}")
    return written
