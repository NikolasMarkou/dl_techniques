"""Where a run lives, and the rule that a run is never overwritten.

Nothing in this module deletes a file (invariant 5): a run directory only
grows, and a directory that already holds a run is refused. Every ``train.common``
import happens inside a function, because importing it loads TensorFlow and
the help path may not (D-012).

Interface contract
------------------
``resolve_out(out, task, family, variant)``
    :param out: the ``out`` config value. ``"auto"`` names a fresh
        ``<repo>/results/<task>_<family>_<variant>_<timestamp>``; a relative path
        is anchored at the repo root, not the working directory; an absolute
        path is kept.
    :return: the run directory as a ``Path``. Nothing is created.
    :raises ConfigError: for an empty ``out`` (or only blanks), ``.``, a path
        ending in ``..``, the filesystem root, or a path naming an existing file.

``run_artifact_names()``
    :return: the names whose presence marks a directory as holding a run: the
        shared ``RUN_ARTIFACT_NAMES`` plus this API's own files.

``open_run(run_dir, resume)``
    :param run_dir: from :func:`resolve_out`.
    :param resume: ``False`` for a new run, ``True`` to continue one.
    :return: ``run_dir``. A new run creates the directory (empty or missing is
        fine); a resumed run touches nothing.
    :raises ConfigError: a new run into a directory holding any artifact; a
        resume without ``checkpoints/backup`` or with ``results_summary.json``
        present (a finished run has nothing to resume, D-016).

``write_requested(run_dir, mapping)`` / ``write_resolved(run_dir, config)``
    Write ``config_requested.json`` (the mapping as the caller gave it, AUTO
    values included) and ``config.json`` (the resolved config). Both raise
    ``OSError`` or ``TypeError`` on a failed write; neither is wrapped.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Tuple

from .errors import ConfigError

# ---------------------------------------------------------------------

#: The ``out`` value that asks for a timestamped directory under ``results/``.
AUTO_OUT = "auto"

#: Where ``out="auto"`` runs go, relative to the repo root.
AUTO_PARENT = "results"

#: The file a request is written to before any data is loaded.
REQUESTED_CONFIG_NAME = "config_requested.json"

#: A run that can be resumed holds this directory (written by ``BackupAndRestore``).
BACKUP_DIR = "checkpoints/backup"

# ---------------------------------------------------------------------


def resolve_out(out: Any, task: str, family: str, variant: str) -> Path:
    """Turn the ``out`` config value into a run directory path.

    :param out: ``"auto"``, a relative path or an absolute path.
    :param task: the task name, e.g. ``"image_classification"``.
    :param family: the model family, e.g. ``"vit"``.
    :param variant: the model variant, e.g. ``"vit_pico"``.
    :return: the run directory. Nothing is created or checked for existing runs.
    :raises ConfigError: see the module contract.
    """
    if not isinstance(out, (str, Path)):
        raise ConfigError(
            f"Invalid value for 'out': expected a path, got {out!r} ({type(out).__name__})"
        )
    text = str(out)
    if text.strip() == "":
        raise ConfigError("Invalid value for 'out': it is empty; use a path or 'auto'")

    from train.common.args import resolved_run_dir
    from train.common.run_io import default_experiment_name

    # DECISION plan-2026-10-30T132432-a865fa08/D-030
    # A relative `out` is anchored at the REPO ROOT through the shared
    # `resolved_run_dir`, so `out=results/x` lands in the same tree wherever the
    # user stands. Do NOT use `Path(out).resolve()` (it anchors at the working
    # directory and strays a `results/` tree anywhere), and do NOT re-derive
    # `parents[3]` here: that index is a layout coincidence owned by the helper
    # (plan-2026-08-12T123743-e798a9e1/D-014).
    if text == AUTO_OUT:
        config = SimpleNamespace(
            output_dir=AUTO_PARENT,
            experiment_name=default_experiment_name(task, family, variant),
        )
    else:
        path = Path(text)
        if not path.parts or path.name == "..":
            raise ConfigError(
                f"Invalid value for 'out': {text!r} does not name a run directory"
            )
        config = SimpleNamespace(output_dir=path, experiment_name="")
    run_dir = resolved_run_dir(config)

    if run_dir.parent == run_dir:
        raise ConfigError(f"Invalid value for 'out': {text!r} is the filesystem root")
    if run_dir.exists() and not run_dir.is_dir():
        raise ConfigError(
            f"Invalid value for 'out': {run_dir} is an existing file, not a directory"
        )
    return run_dir


def run_artifact_names() -> Tuple[str, ...]:
    """Return the names whose presence marks a directory as holding a run.

    :return: the shared ``RUN_ARTIFACT_NAMES`` followed by this API's files.
    """
    from train.common.run_artifacts import RUN_ARTIFACT_NAMES

    # DECISION plan-2026-10-30T132432-a865fa08/D-030
    # The API's own files are added HERE, in a tuple built per call. Do NOT widen
    # `RUN_ARTIFACT_NAMES` itself: seven other callers and the power_mlp
    # parametrized refusal test read it, and a legacy trainer's refusal must not
    # change when this package grows a file.
    return (*RUN_ARTIFACT_NAMES, REQUESTED_CONFIG_NAME, "final_model.keras", "task.json", BACKUP_DIR)


def open_run(run_dir: Path, resume: bool) -> Path:
    """Refuse a run directory that cannot be used, then make it usable.

    :param run_dir: the directory from :func:`resolve_out`.
    :param resume: ``True`` to continue an interrupted run.
    :return: ``run_dir``.
    :raises ConfigError: see the module contract.
    """
    from train.common.run_artifacts import RESULTS_SUMMARY_NAME, refuse_existing_run

    if resume:
        if not (run_dir / BACKUP_DIR).is_dir():
            raise ConfigError(
                f"Cannot resume: {run_dir / BACKUP_DIR} does not exist, so there is "
                "no interrupted run to continue"
            )
        if (run_dir / RESULTS_SUMMARY_NAME).exists():
            raise ConfigError(
                f"Cannot resume: {run_dir} holds {RESULTS_SUMMARY_NAME}, so the run "
                "has finished; choose a new 'out' to train again"
            )
        return run_dir

    try:
        refuse_existing_run(run_dir, run_artifact_names())
    except FileExistsError as exc:
        raise ConfigError(
            f"{exc} In this API the run directory is set by 'out'; "
            "'out=auto' picks a new timestamped one."
        ) from exc
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_requested(run_dir: Path, mapping: Mapping[str, Any]) -> Path:
    """Write ``config_requested.json``: the request as the caller gave it.

    :param run_dir: an existing run directory.
    :param mapping: the requested configuration, AUTO values left as ``"auto"``.
    :return: ``run_dir``.
    """
    from train.common.run_io import prepare_run_dir

    return prepare_run_dir(mapping, output_dir=run_dir, config_filename=REQUESTED_CONFIG_NAME)


def write_resolved(run_dir: Path, config: Any) -> str:
    """Write ``config.json``: the configuration after AUTO resolution.

    :param run_dir: an existing run directory.
    :param config: a dataclass instance (or a JSON-serializable mapping).
    :return: the path written, as text.
    """
    from train.common.config_io import save_config_json

    return save_config_json(config, str(run_dir), "config.json")
