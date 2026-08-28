"""Run-directory paths for the embeddings study.

Extracted from ``train_embeddings`` so the evaluator can import them without a
cycle: ``train_embeddings`` calls the evaluator at the end of a cell, and the
evaluator needs to know where the encoder was written.

Every filename here has exactly ONE producer, and both ends of each contract
call it. This repository has already paid for the alternative once: a checkpoint
name with several readers and zero writers failed every default run, after
training had finished.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------

__all__ = [
    "ENCODER_FILENAME",
    "EVAL_FILENAME",
    "REPO_ROOT",
    "RESULTS_FILENAME",
    "encoder_path",
    "eval_path",
    "resolve_output_dir",
    "results_path",
]

#: Filename of the encoder handed from stage 1 to stage 2 and to evaluation.
ENCODER_FILENAME = "encoder.keras"

#: Filename of the embedding-evaluation output.
EVAL_FILENAME = "eval.json"

#: Filename of the per-cell results the sweep collects.
RESULTS_FILENAME = "results.json"

#: Repository root: this file is <repo>/src/train/embeddings_experimental/…
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_output_dir(output_dir: str) -> str:
    """Resolve a relative output directory against the REPO ROOT.

    Training artifacts belong in the repository's own ``results/``, never in
    ``src/results/``. The documented way to run a trainer here is
    ``python -m train.<pkg>.<script>`` from ``src/``, so a bare relative
    ``results`` resolves against ``src/`` and quietly creates a second, wrong
    results tree -- which is exactly what this trainer's first smoke runs did.
    An absolute path is returned unchanged, so an explicit
    ``--output-dir /somewhere`` still wins.

    :param output_dir: The configured output directory.
    :type output_dir: str
    :return: An absolute path.
    :rtype: str
    """
    path = Path(output_dir)
    return str(path if path.is_absolute() else REPO_ROOT / path)


def encoder_path(run_dir: str) -> str:
    """Return the encoder checkpoint path for a run directory.

    :param run_dir: The run directory.
    :type run_dir: str
    :return: Path to the encoder checkpoint.
    :rtype: str
    """
    return os.path.join(run_dir, ENCODER_FILENAME)


def eval_path(run_dir: str) -> str:
    """Return the embedding-evaluation output path for a run directory.

    :param run_dir: The run directory.
    :type run_dir: str
    :return: Path to ``eval.json``.
    :rtype: str
    """
    return os.path.join(run_dir, EVAL_FILENAME)


def results_path(run_dir: str) -> str:
    """Return the per-cell results path for a run directory.

    :param run_dir: The run directory.
    :type run_dir: str
    :return: Path to ``results.json``.
    :rtype: str
    """
    return os.path.join(run_dir, RESULTS_FILENAME)
