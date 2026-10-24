"""Regenerate ``figures/figure_facts.json`` from finished runs.

The README of this package quotes numbers about its example figures, and the test
of the README holds the captions to ``figures/figure_facts.json``. That file used
to be written by a script in a gitignored plan directory, so nothing in the
repository could regenerate the numbers the README rests on (decision D-029).
This module is that script, tracked and tested:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg python -m train.topolm.figure_facts \\
        --run 2d=<run dir> --run 3d=<run dir> \\
        --run 2d_alpha0=<run dir> --run 3d_alpha0=<run dir> \\
        --controls 2d --controls 3d --capture-dir <dir> \\
        --out src/train/topolm/figures/figure_facts.json

``python -m train.topolm.figure_facts --help`` lists every option.

What it reads, and what it recomputes
-------------------------------------
Everything is read from the run directories (read only, nothing under them is
written) except three kinds of input and one kind of recomputation:

* **Run files.** ``config.json``, ``training_history.json``,
  ``topography_report.json``, ``topography_live/metrics.jsonl`` and the PNG files
  the example figures are copies of. Every number a caption quotes is read from
  these, or measured from the PNG, and stored both as a value and as the string
  the caption prints (``{"value": ..., "fmt": ...}``).
* **The capture of the training command** (``--capture-dir``, optional). A run
  directory does not hold the trainer's console output, the wall time of the
  command or the GPU memory samples, so they are read from three files per run,
  ``log_<name>.txt`` (stdout and stderr of ``python -m train.topolm.pretrain``),
  ``result_<name>.txt`` (one line, ``exit=<code> wall_s=<seconds>``) and
  ``gpu_samples_<name>.csv`` (one line per second, ``<epoch seconds>,<MiB of the
  trainer's process or NA>,<MiB used on the card>,<utilisation>``). Without the
  directory those fields are left out and ``generated_from.capture`` says so.
  They are properties of the machine the run was made on and cannot be recomputed.
* **The synthetic illustration** is drawn by
  :func:`train.topolm.controls.synthetic_illustration` (no model, no run).
* **The controls** (``--controls NAME``) are computed by
  :func:`train.topolm.controls.facts`, which this module calls rather than copies.
* **The colour limit of a t-map figure** is not stored by the run. It is
  recomputed on the machine running this module, from the run's own weights
  (``checkpoints/final.keras``) or, for the untrained figure, from the seeded
  initial weights, with the trainer's own ``evaluate_topography``; the recomputed
  report is compared with the run's own, and the limit is cross-checked against
  the colour bar of the PNG.

Device dependence
-----------------
The recomputed colour limit and its checks come from running the model here, and a
different device rounds them differently in the last digits.
:data:`DEVICE_DEPENDENT_FIELDS` lists those fields, the output records the list and
the device it was computed on (``generated_from.recomputation_device``), and
:func:`compare_facts` reports differences in them separately from the rest. The
controls are NOT on that list: they are counts, statistics of seeded draws and labels
(none is a duration), they reproduce exactly on the CPU, and a comparison that exempted
them could never fail on the most claim-bearing numbers of the README (review pass 6,
concern 3: the framing count 65.0 changed to 0.0 and the label-shuffle null 2 changed to
120 passed it). Run with ``CUDA_VISIBLE_DEVICES=`` to compute on the CPU. The git
provenance (``head_sha`` and the two fields beside it) differs by construction.

The module imports nothing from a plan directory and carries no absolute path.
"""

import argparse
import fnmatch
import functools
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib.ticker import AutoLocator
from PIL import Image

from dl_techniques.utils.logger import logger
from train.topolm import common as C
from train.topolm import controls as K
from train.topolm.plotting import _shared_limit, _validation_points, last_row_per_step

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

#: The repository root, found from this file (``<root>/src/train/topolm/``).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

#: Where the shipped figures and their facts live.
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

#: The config keys copied into ``runs.<name>.config``.
CONFIG_KEYS: Tuple[str, ...] = (
    "model_variant", "seed", "batch_size", "train_steps", "eval_every_steps",
    "viz_every_steps", "spatial_radius", "spatial_alpha", "max_seq_length",
    "tap_sites", "min_cluster_size", "num_permutations", "permutation_p_value",
)

#: Files a run directory must hold for the facts to be written from it.
REQUIRED_RUN_FILES: Tuple[str, ...] = (
    "config.json",
    "training_history.json",
    "topography_report.json",
    os.path.join("topography_live", "metrics.jsonl"),
)

#: The illustration's name in ``figures/`` and in the facts.
ILLUSTRATION_NAME = "t_maps_2d_illustration.png"

#: Dotted paths (``fnmatch`` patterns) of the fields that depend on the device the
#: facts were computed on: they come from running the model here, not from reading
#: a file. ``*`` matches any run name, figure name or key, dots included.
DEVICE_DEPENDENT_FIELDS: Tuple[str, ...] = (
    "files.*.facts.colour_limit.value",
    "files.*.facts.colour_limit.fmt",
    "files.*.facts.colour_limit.how",
    "files.*.facts.colour_limit.recomputation_check.*",
    "files.*.facts.colour_limit.pixel_cross_check.relative_difference",
)

#: Fields that differ by construction between two generations: the git state at
#: the time, the form of the run paths, and the keys this module records about
#: itself.
PROVENANCE_FIELDS: Tuple[str, ...] = (
    "generated_from.head_sha",
    "generated_from.last_commit_touching_topolm_trainer_or_metrics",
    "generated_from.topolm_and_metrics_tracked_files_clean",
    "generated_from.run_dir_*",
    "generated_from.run_dirs.*",
    "generated_from.recomputation_device",
    "generated_from.device_dependent_fields",
    "generated_from.capture",
)


@dataclass(frozen=True)
class FigureSpec:
    """One example figure: which run it comes from and what its facts hold.

    :param run: Name of the run (``--run NAME=DIR``) the figure is a copy from.
    :param source: Path of the PNG relative to the run directory.
    :param step: The optimizer step the figure shows.
    :param colour: ``'trained'`` or ``'initial'`` when the figure is a t-map figure
        whose colour limit is recomputed (from ``checkpoints/final.keras`` or from
        the seeded initial weights), ``None`` otherwise.
    :param cluster: True for a cluster figure: it is omitted, with the reason,
        when the run wrote none.
    :param curve: True for the live curve, whose facts hold the series.
    """

    run: str
    source: str
    step: int
    colour: Optional[str] = None
    cluster: bool = False
    curve: bool = False


#: The example figures, in the order of ``files`` in the facts. The names are the
#: files in ``figures/``; the paths are the ones the README's "where the pictures
#: come from" paragraph lists.
DEFAULT_FIGURES: Dict[str, FigureSpec] = {
    "t_maps_2d_untrained.png": FigureSpec(
        "2d", "topography_live/step_0000000/t_maps_raw.png", 0, colour="initial"),
    "t_maps_2d_trained.png": FigureSpec(
        "2d", "topography_live/step_0000400/t_maps_raw.png", 400, colour="trained"),
    "t_maps_3d_trained.png": FigureSpec(
        "3d", "topography_live/step_0000400/t_maps_raw.png", 400, colour="trained"),
    "live_curve_2d.png": FigureSpec(
        "2d", "topography_live/topography_curve.png", 400, curve=True),
    "clusters_2d.png": FigureSpec(
        "2d", "topography_live/step_0000400/clusters_raw.png", 400, cluster=True),
    "clusters_3d.png": FigureSpec(
        "3d", "topography_live/step_0000400/clusters_raw.png", 400, cluster=True),
    "morans_i_by_depth_2d.png": FigureSpec(
        "2d", "topography_live/step_0000400/morans_i_raw.png", 400),
    "t_maps_2d_alpha0_control.png": FigureSpec(
        "2d_alpha0", "topography_live/step_0000400/t_maps_raw.png", 400, colour="trained"),
}

# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------


def number(value: Any, fmt: str) -> Dict[str, Any]:
    """A number as the facts store it: the value and the string a caption prints.

    :param value: An int or float; ``None`` or NaN give ``{"value": None, "fmt":
        "n/a"}``.
    :param fmt: A ``format`` spec such as ``".3f"`` or ``"d"``.
    :return: ``{"value": value, "fmt": format(value, fmt)}``.
    """
    if value is None or (isinstance(value, float) and value != value):
        return {"value": None, "fmt": "n/a"}
    return {"value": value, "fmt": format(value, fmt)}


def sha256_file(path: str) -> str:
    """Hex SHA-256 of the bytes of ``path``, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tap_label(name: str) -> str:
    """The last two path parts of a tap name: ``'m/block_0/mlp_tap'`` -> ``'block_0/mlp_tap'``."""
    return "/".join(name.split("/")[-2:])


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_rows(run_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(all rows, one row per step)`` of ``topography_live/metrics.jsonl``."""
    path = os.path.join(run_dir, "topography_live", "metrics.jsonl")
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows, last_row_per_step(rows)


def missing_run_files(run_dir: str) -> List[str]:
    """The entries of :data:`REQUIRED_RUN_FILES` that ``run_dir`` does not hold."""
    return [
        name for name in REQUIRED_RUN_FILES
        if not os.path.isfile(os.path.join(run_dir, name))
    ]


def _repo_relative(path: str, root: str) -> str:
    """``path`` relative to ``root`` when it lies inside it, the absolute path otherwise."""
    absolute = os.path.abspath(path)
    if os.path.commonpath([absolute, root]) == root:
        return os.path.relpath(absolute, root)
    return absolute


# ---------------------------------------------------------------------
# facts of one draw (one row of metrics.jsonl)
# ---------------------------------------------------------------------


def step_facts(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Every number a caption quotes about one draw, from its ``metrics.jsonl`` row.

    :param row: A row with ``step``, ``grid_shape``, ``mean_morans_i``,
        ``mean_num_significant_units``, ``total_clusters_a`` / ``_b`` and ``per_tap``
        (``morans_i``, ``morans_i_islands``, ``num_significant_units``,
        ``clusters_a`` / ``_b``, ``t_min``, ``t_max`` for each tap).
    :return: A dictionary of :func:`number` entries, plus ``grid_shape`` and the
        nested ``per_tap``. A tap whose ``morans_i_islands`` is null or NaN is left
        out of the islands statistics, and the count of taps that have one is kept.
    """
    taps = row["per_tap"]
    names = list(taps)
    moran = [taps[n]["morans_i"] for n in names]
    islands = [
        taps[n]["morans_i_islands"] for n in names
        if taps[n]["morans_i_islands"] is not None
        and taps[n]["morans_i_islands"] == taps[n]["morans_i_islands"]
    ]
    significant = [taps[n]["num_significant_units"] for n in names]
    grid = list(row["grid_shape"])
    return {
        "step": number(row["step"], "d"),
        "grid_shape": {"value": grid, "fmt": "x".join(str(g) for g in grid)},
        "units_per_tap": number(int(np.prod(grid)), "d"),
        "num_taps": number(len(names), "d"),
        "mean_morans_i": number(row["mean_morans_i"], ".3f"),
        "morans_i_min_over_taps": number(min(moran), ".3f"),
        "morans_i_max_over_taps": number(max(moran), ".3f"),
        "morans_i_islands_max_over_taps": number(max(islands) if islands else None, ".3f"),
        "morans_i_islands_min_over_taps": number(min(islands) if islands else None, ".3f"),
        "taps_with_islands_morans_i_defined": number(len(islands), "d"),
        "mean_significant_units_per_tap": number(row["mean_num_significant_units"], ".1f"),
        "significant_units_min_over_taps": number(min(significant), "d"),
        "significant_units_max_over_taps": number(max(significant), "d"),
        "total_clusters_a": number(row["total_clusters_a"], "d"),
        "total_clusters_b": number(row["total_clusters_b"], "d"),
        "t_min_over_taps": number(min(taps[n]["t_min"] for n in names), ".2f"),
        "t_max_over_taps": number(max(taps[n]["t_max"] for n in names), ".2f"),
        "per_tap": {
            tap_label(n): {
                "morans_i": number(taps[n]["morans_i"], ".3f"),
                "significant_units": number(taps[n]["num_significant_units"], "d"),
                "clusters_a": number(taps[n]["clusters_a"], "d"),
                "clusters_b": number(taps[n]["clusters_b"], "d"),
            }
            for n in names
        },
    }


def cluster_tap(row: Mapping[str, Any], report: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The tap a cluster figure shows: the deepest tap with a cluster of condition A.

    That is the tap :func:`train.topolm.common.plot_topography` draws.

    :param row: The ``metrics.jsonl`` row of the draw.
    :param report: The run's ``topography_report.json`` (cluster sizes come from
        its ``raw`` arm).
    :return: The tap's facts and the title as drawn, or ``None`` when no tap has a
        cluster of condition A.
    """
    taps = row["per_tap"]
    populated = [n for n in taps if taps[n]["clusters_a"] > 0]
    if not populated:
        return None
    name = populated[-1]
    detail = report["arms"]["raw"]["per_tap"][name]
    count = taps[name]["clusters_a"]
    return {
        "name": {"value": name, "fmt": tap_label(name)},
        "clusters_a": number(count, "d"),
        "clusters_b": number(taps[name]["clusters_b"], "d"),
        "cluster_sizes_a": {
            "value": detail["cluster_sizes_a"],
            "fmt": ", ".join(map(str, detail["cluster_sizes_a"])),
        },
        "cluster_sizes_b": {
            "value": detail["cluster_sizes_b"],
            "fmt": ", ".join(map(str, detail["cluster_sizes_b"])),
        },
        "morans_i": number(taps[name]["morans_i"], ".3f"),
        "significant_units": number(taps[name]["num_significant_units"], "d"),
        "title_as_drawn": (
            f"{tap_label(name)} (raw), condition a: {count} cluster"
            + ("" if count == 1 else "s")
        ),
    }


# ---------------------------------------------------------------------
# the capture of the training command
# ---------------------------------------------------------------------

_STAMPED_LINE = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} ")


def _stamp(line: str) -> datetime:
    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")


def parse_log(text: str) -> Dict[str, Any]:
    """Timings and warnings from the console output of one training command.

    :param text: The stdout and stderr of ``python -m train.topolm.pretrain``. The
        trainer's log lines start with ``YYYY-MM-DD HH:MM:SS,mmm``; the ones used
        are the first such line, the ``Cadence:`` line, the ``Best evaluation``
        line and the last ``pretrain.py:main`` line that says ``Wrote``.
    :return: ``device`` (the GPU name of TensorFlow's ``Created device`` line, or
        ``None`` for a run without one), the three durations in seconds
        (``first_stamped_line_to_cadence_s``, ``fit_phase_s_cadence_to_best_evaluation``,
        ``cadence_to_wrote_s``), the count of interpreter-shutdown warnings and the
        count of ``Traceback`` lines.
    :raises ValueError: when a line the durations need is absent (the message names
        it).
    """
    lines = text.splitlines()
    stamped = [line for line in lines if _STAMPED_LINE.match(line)]

    def need(description: str, found: List[str]) -> str:
        if not found:
            raise ValueError(f"the training log has no {description} line")
        return found[0]

    first = need("timestamped", stamped)
    cadence = need("'Cadence:'", [l for l in stamped if "Cadence:" in l])
    best = need("'Best evaluation'", [l for l in stamped if "Best evaluation" in l])
    wrote = need(
        "'Wrote' (pretrain.py:main)",
        [l for l in stamped if "pretrain.py:main" in l and "Wrote" in l][::-1],
    )
    device_line = next((l for l in lines if "Created device" in l), None)
    match = re.search(r"name: ([^,]+),", device_line) if device_line else None
    return {
        "device": match.group(1) if match else None,
        "first_stamped_line_to_cadence_s": (_stamp(cadence) - _stamp(first)).total_seconds(),
        "fit_phase_s_cadence_to_best_evaluation": (_stamp(best) - _stamp(cadence)).total_seconds(),
        "cadence_to_wrote_s": (_stamp(wrote) - _stamp(cadence)).total_seconds(),
        "tf_shutdown_warnings_python_interpreter_state": sum(
            "Python interpreter state is not initialized" in l for l in lines
        ),
        "traceback_lines": sum("Traceback" in l for l in lines),
    }


def parse_result(text: str) -> Tuple[float, int]:
    """``(wall seconds, exit code)`` from a line ``exit=<code> wall_s=<seconds>``.

    :raises ValueError: when either field is missing.
    """
    wall = re.search(r"wall_s=([0-9.]+)", text)
    code = re.search(r"exit=(\d+)", text)
    if wall is None or code is None:
        raise ValueError(f"expected 'exit=<code> wall_s=<seconds>', got {text.strip()!r}")
    return float(wall.group(1)), int(code.group(1))


def gpu_samples(text: str) -> List[int]:
    """The trainer process's GPU memory in MiB, one entry per sample that has one.

    :param text: Lines ``<time>,<MiB of the process or NA>,<MiB used>,<util>``. A
        line whose second field is ``NA`` or empty (the process held no memory yet)
        is skipped.
    """
    peaks = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[1] not in ("NA", ""):
            peaks.append(int(parts[1]))
    return peaks


def read_capture(capture_dir: str, name: str) -> Dict[str, str]:
    """The three capture files of run ``name``, as text.

    :raises FileNotFoundError: naming the first missing file.
    """
    texts = {}
    for key, filename in (
        ("log", f"log_{name}.txt"),
        ("result", f"result_{name}.txt"),
        ("gpu", f"gpu_samples_{name}.csv"),
    ):
        path = os.path.join(capture_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"run {name!r}: capture file {path} is missing")
        with open(path, encoding="utf-8", errors="replace") as handle:
            texts[key] = handle.read()
    return texts


# ---------------------------------------------------------------------
# facts of one run
# ---------------------------------------------------------------------


def run_summary(
    rows: Sequence[Mapping[str, Any]],
    merged: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    history: Mapping[str, Any],
    config: Mapping[str, Any],
    capture: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """The facts of one run: draws, series, training result and, when given, the capture.

    :param rows: Every row of ``metrics.jsonl``.
    :param merged: One row per step (:func:`train.topolm.plotting.last_row_per_step`).
    :param report: ``topography_report.json``.
    :param history: ``training_history.json``.
    :param config: ``config.json``.
    :param capture: Texts of the capture files (:func:`read_capture`), or ``None``.
        Without it ``exit_code``, ``total_wall_s``, ``peak_own_gpu_mib_nvidia_smi_1s_samples``,
        ``gpu_samples`` and the log fields are absent.
    :return: A JSON-ready dictionary; the key order is the one of the shipped file.
    """
    seconds = [r["seconds"] for r in rows if r.get("seconds") is not None and "per_tap" in r]
    last = merged[-1]
    raw = report["arms"]["raw"]
    out: Dict[str, Any] = {}
    peaks: List[int] = []
    if capture is not None:
        wall, exit_code = parse_result(capture["result"])
        peaks = gpu_samples(capture["gpu"])
        out["exit_code"] = exit_code
        out["total_wall_s"] = number(wall, ".1f")
    out.update({
        "number_of_draws": number(len(seconds), "d"),
        "seconds_per_draw_min": number(min(seconds), ".2f"),
        "seconds_per_draw_median": number(statistics.median(seconds), ".2f"),
        "seconds_per_draw_max": number(max(seconds), ".2f"),
        "seconds_per_draw_sum": number(sum(seconds), ".1f"),
    })
    if capture is not None:
        out["peak_own_gpu_mib_nvidia_smi_1s_samples"] = (
            number(max(peaks), "d") if peaks else None
        )
        out["gpu_samples"] = len(peaks)
    same = (
        last["mean_morans_i"] == raw["mean_morans_i"]
        and last["mean_num_significant_units"] == raw["mean_num_significant_units"]
        and last["total_clusters_a"] == raw["total_clusters_a"]
        and last["total_clusters_b"] == raw["total_clusters_b"]
    )
    out.update({
        "last_live_row_equals_report": same,
        "last_live_row_mean_morans_i": last["mean_morans_i"],
        "report_mean_morans_i": raw["mean_morans_i"],
        "rows_in_metrics_jsonl": len(rows),
        "rows_after_merge_per_step": len(merged),
        "steps_drawn": [r["step"] for r in merged],
        "restored_weights_rows": sum(bool(r.get("restored_weights")) for r in rows),
        "val_only_rows": sum(bool(r.get("val_only")) for r in rows),
        "stimuli_are_smoke_set": report["stimuli_are_smoke_set"],
        "readout_skipped": report.get("readout_skipped"),
        "arms": list(report["arms"]),
        "mean_morans_i_series": {str(r["step"]): r["mean_morans_i"] for r in merged},
        "mean_significant_units_series": {
            str(r["step"]): r["mean_num_significant_units"] for r in merged
        },
        "clusters_ab_series": {
            str(r["step"]): [r["total_clusters_a"], r["total_clusters_b"]] for r in merged
        },
    })
    significant = sum(v["num_significant_units"] for v in raw["per_tap"].values())
    in_clusters = sum(
        sum(v.get("cluster_sizes_a", [])) + sum(v.get("cluster_sizes_b", []))
        for v in raw["per_tap"].values()
    )
    out["significant_units_total"] = significant
    out["significant_units_in_clusters"] = in_clusters
    out["share_of_significant_units_in_clusters"] = (
        in_clusters / significant if significant else None
    )
    out["training"] = {
        "final_val_loss": history["val_loss"][-1],
        "final_val_perplexity": history["val_perplexity"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "val_loss_per_virtual_epoch": history["val_loss"],
        "final_train_loss": history["loss"][-1],
        "epochs_run": len(history["loss"]),
        "vocab_size": config["vocab_size"],
        "model_variant": config["model_variant"],
    }
    out["val_loss_points_in_metrics_jsonl"] = {
        str(r["val_loss_step"]): r["val_loss"] for r in rows if r.get("val_loss_step") is not None
    }
    if capture is not None:
        out.update(parse_log(capture["log"]))
    return out


def stimuli_framing(config: Mapping[str, Any]) -> Dict[str, Any]:
    """How the trainer frames the stimuli of ``config``: token counts and positions.

    Read through the trainer's own :func:`train.topolm.common._encode_stimuli` with
    the shipped context, for each condition of :data:`~train.topolm.common.SMOKE_STIMULI`.

    :param config: A run's ``config.json`` (encoding and the special-token ids).
    :return: Per condition the sentence token counts, the context length, whether
        the id before each sentence is the end-of-text id, the sentence start
        positions, the window length, whether CLS or SEP occurs; plus the encoder's
        end-of-text id and the cross-condition summaries.
    """
    preprocessor = K.preprocessor_from_config(config)
    eot = int(preprocessor.tokenizer.eot_token)
    out: Dict[str, Any] = {}
    for condition, prompts in C.SMOKE_STIMULI.items():
        tokens, text, non_text = C._encode_stimuli(preprocessor, list(prompts), None)
        starts = [int(np.argmax(row)) for row in text]
        out[condition] = {
            "text_token_counts": text.sum(1).tolist(),
            "context_tokens_before_eot": starts[0] - 1,
            "id_before_the_sentence_is_eot": bool(
                all(tokens[i, starts[i] - 1] == eot for i in range(len(starts)))
            ),
            "sentence_starts_at_position": starts,
            "window_length": int(tokens.shape[1]),
            "non_text_phrase": non_text,
            "cls_id_present": bool((tokens == config["cls_token_id"]).any()),
            "sep_id_present": bool((tokens == config["sep_token_id"]).any()),
        }
    out["eot_token_of_encoder"] = eot
    out["text_token_counts_equal_across_conditions"] = all(
        out[c]["text_token_counts"] == out["a"]["text_token_counts"] for c in C.SMOKE_STIMULI
    )
    out["mean_text_tokens_per_condition"] = {
        c: sum(out[c]["text_token_counts"]) / len(out[c]["text_token_counts"])
        for c in C.SMOKE_STIMULI
    }
    return out


# ---------------------------------------------------------------------
# the colour limit of a t-map figure
# ---------------------------------------------------------------------
#
# The exact limit L (a pooled 99th percentile of |t|) is not stored by the run.
# The colour bar prints ticks chosen by matplotlib's default locator for the range
# (-L, L), so L is also recoverable from pixels: the bar's top and bottom edge give
# the range in pixels, the tick label centres say where the ticks sit, and the L
# whose default-locator ticks best match the measured centres wins. The pixel
# measurement is accurate to about 0.3 percent of L, and geometry alone cannot tell
# "step 1, limit L" from "step 2, limit 2L", so the exact recomputed limit picks
# among the candidates.


def _longest_run(column: np.ndarray) -> Tuple[int, int]:
    """``(start, length)`` of the longest run of True in a boolean column."""
    best = best_start = length = start = 0
    for y, value in enumerate(column):
        if value:
            if length == 0:
                start = y
            length += 1
            if length > best:
                best, best_start = length, start
        else:
            length = 0
    return best_start, best


def find_bar(image: np.ndarray) -> Tuple[int, int, int, int]:
    """``(x_left, x_right, y_top, y_bottom)`` of the colour bar of an RGB image.

    The bar is the rightmost coloured strip at least 8 px wide whose columns hold a
    run of at least 150 non-white pixels.

    :raises ValueError: when the image has no such strip.
    """
    height, width, _ = image.shape
    nonwhite = image.min(2) < 252
    groups: List[Dict[str, int]] = []
    for x in range(width - 1, width // 2, -1):
        start, length = _longest_run(nonwhite[:, x])
        if length < 150:
            continue
        if (
            groups and groups[-1]["x0"] - x == 1
            and abs(groups[-1]["s"] - start) <= 1 and abs(groups[-1]["n"] - length) <= 1
        ):
            groups[-1]["x0"] = x
        else:
            groups.append({"x0": x, "x1": x, "s": start, "n": length})
    wide = [g for g in groups if g["x1"] - g["x0"] >= 8]
    if not wide:
        raise ValueError("no colour bar found in the image")
    g = max(wide, key=lambda g: g["x1"])
    return g["x0"], g["x1"], g["s"], g["s"] + g["n"] - 1


def _ink_width(image: np.ndarray, x1: int, y0: int, y1: int) -> Tuple[int, float]:
    """``(first column, width)`` of the first group of dark columns right of the bar."""
    sub = image[y0 - 6: y1 + 7, x1 + 3: x1 + 3 + 60]
    cols = np.where((sub.max(2) < 150).any(0))[0]
    if not len(cols):
        return 0, 0.0
    end = cols[0]
    for c in cols[1:]:
        if c - end > 4:
            break
        end = c
    return int(cols[0]), float(end - cols[0] + 1)


@functools.lru_cache(maxsize=None)
def _label_width_px(label: str, fontsize: float, dpi: float) -> float:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    extent = TextPath((0, 0), label, size=fontsize, prop=FontProperties()).get_extents()
    return extent.width * dpi / 72.0


def _label_centres(
    image: np.ndarray, x1: int, y0: int, y1: int, col0: int, width: float
) -> List[float]:
    """Vertical centres of the tick labels (their own columns, not the axis label)."""
    left = x1 + 3 + col0
    sub = image[y0 - 6: y1 + 7, left: left + int(width) + 1]
    rows = np.where((sub.max(2) < 150).any(1))[0]
    groups: List[List[int]] = []
    for r in rows:
        if groups and r - groups[-1][-1] <= 1:
            groups[-1].append(r)
        else:
            groups.append([r])
    return [y0 - 6 + (g[0] + g[-1]) / 2.0 for g in groups if len(g) >= 5]


def measure_colour_limit(
    path: str, max_abs_t: Optional[float] = None, fontsize: float = 7, dpi: float = 160
) -> Dict[str, Any]:
    """Candidate colour limits read from the colour bar of a t-map PNG.

    :param path: A t-map figure of :func:`train.topolm.plotting.plot_t_maps`.
    :param max_abs_t: The largest finite ``|t|`` over all taps; the limit is a 99th
        percentile and never above it, so candidates above it are dropped.
    :param fontsize: Tick label font size in points (the figure's own, 7).
    :param dpi: The figure's resolution (160).
    :return: ``{"candidates": [...up to 8, best first, each {"limit",
        "mismatch_px", "ticks"}], "ink_width_px", "bar", "label_centres"}``. Several
        candidates can fit equally (``[-8..8]`` step 2 and ``[-4..4]`` step 1 have
        the same number of ticks), so the caller picks the one nearest an
        independent value.
    :raises ValueError: when no colour bar is found.
    """
    with Image.open(path) as handle:
        image = np.array(handle.convert("RGB")).astype(int)
    x0, x1, y0, y1 = find_bar(image)
    col0, ink = _ink_width(image, x1, y0, y1)
    centres = _label_centres(image, x1, y0, y1, col0, ink)
    top, bottom = y0 + 1.0, float(y1)
    height = bottom - top
    best: Dict[Tuple[float, ...], Tuple[float, float, float]] = {}
    for limit in np.arange(0.5, 60.0, 0.01):
        ticks = [t for t in AutoLocator().tick_values(-limit, limit) if -limit <= t <= limit]
        pixels = sorted(bottom - (t + limit) / (2 * limit) * height for t in ticks)
        if len(pixels) != len(centres):
            continue
        error_position = float(np.mean(np.abs(np.array(pixels) - np.array(sorted(centres)))))
        labels = [("%g" % t).replace("-", "\u2212") for t in ticks]
        error_width = abs(
            0.8 * max((_label_width_px(l, fontsize, dpi) for l in labels), default=0.0) - ink
        )
        error = error_position + 0.5 * error_width
        key = tuple(ticks)
        if key not in best or error < best[key][0]:
            best[key] = (error, float(limit), error_position)
    fitting = sorted(
        ((error, limit, error_position, ticks)
         for ticks, (error, limit, error_position) in best.items() if error_position < 0.8),
        key=lambda c: c[0],
    )
    if max_abs_t is not None:
        fitting = [c for c in fitting if c[1] <= max_abs_t * 1.0001]
    return {
        "candidates": [
            {"limit": limit, "mismatch_px": mismatch, "ticks": [float(t) for t in ticks]}
            for _, limit, mismatch, ticks in fitting[:8]
        ],
        "ink_width_px": ink,
        "bar": [x0, x1, y0, y1],
        "label_centres": centres,
    }


def _evaluate_run_stimuli(backbone: Any, preprocessor: Any, config: Mapping[str, Any]):
    """The trainer's own evaluation of a run's stimuli: ``(report, arrays)``."""
    stimuli = config.get("stimuli") or C.SMOKE_STIMULI
    return C.evaluate_topography(
        backbone, stimuli, ("a", "b"), preprocessor, output_dir=None, readout_fwhm=None,
        min_cluster_size=config["min_cluster_size"],
        permutation_p_value=config["permutation_p_value"],
        num_permutations=config["num_permutations"], seed=config["seed"],
        is_smoke_set=config.get("stimuli") is None,
    )


def recompute_trained_limit(run_dir: str) -> Dict[str, Any]:
    """The shared colour limit of the final t-maps, recomputed from the run's weights.

    Loads ``checkpoints/final.keras``, runs :func:`train.topolm.common.evaluate_topography`
    on the run's stimuli with the run's seed, and takes the pooled 99th percentile of
    ``|t|`` (:func:`train.topolm.plotting._shared_limit`, the number the figure is
    scaled by). The recomputed report is compared with the run's own, so a mismatch is
    visible in the result.

    :return: ``limit_pooled_p99_abs_t``, ``max_abs_t``, the recomputed and the run's
        mean Moran's I and their absolute difference, whether the significant units
        per tap and the cluster totals are equal.
    """
    backbone, preprocessor, config = K.load_run(run_dir)
    report, arrays = _evaluate_run_stimuli(backbone, preprocessor, config)
    reference = _read_json(os.path.join(run_dir, "topography_report.json"))["arms"]["raw"]
    new = report["arms"]["raw"]
    t_maps = arrays["arms"]["raw"]["t_maps"]
    return {
        "limit_pooled_p99_abs_t": _shared_limit(t_maps, 99.0),
        "max_abs_t": float(max(np.abs(np.asarray(v)).max() for v in t_maps.values())),
        "recomputed_mean_morans_i": new["mean_morans_i"],
        "run_mean_morans_i": reference["mean_morans_i"],
        "mean_morans_abs_diff": abs(new["mean_morans_i"] - reference["mean_morans_i"]),
        "significant_units_equal": (
            [new["per_tap"][k]["num_significant_units"] for k in new["per_tap"]]
            == [reference["per_tap"][k]["num_significant_units"] for k in reference["per_tap"]]
        ),
        "clusters_equal": (
            (new["total_clusters_a"], new["total_clusters_b"])
            == (reference["total_clusters_a"], reference["total_clusters_b"])
        ),
    }


def recompute_initial_limit(run_dir: str) -> Dict[str, Any]:
    """The shared colour limit of the step-0 t-maps, from the seeded initial weights.

    The untrained model is rebuilt the way the trainer builds it (same seed, same
    config) and evaluated on the run's stimuli; the result is validated against the
    run's own step-0 row of ``metrics.jsonl`` (mean Moran's I, per-tap ``t_min`` and
    ``t_max``, significant units).

    :return: ``limit_pooled_p99_abs_t``, ``max_abs_t``, the recomputed and the run's
        step-0 mean Moran's I, the largest absolute differences of ``t_min`` and
        ``t_max`` against the row, and whether every significant-unit count is zero.
    :raises ValueError: when the run has no step-0 row.
    """
    config = _read_json(os.path.join(run_dir, "config.json"))
    preprocessor = K.preprocessor_from_config(config)
    known = {f.name for f in fields(C.TopoLMTrainingConfig)}
    training_config = C.TopoLMTrainingConfig(**{k: v for k, v in config.items() if k in known})
    C.set_seeds(training_config.seed)
    _, backbone = C.create_topolm_model(
        training_config, required_vocab_size=C._required_vocab_size(preprocessor)
    )
    report, arrays = _evaluate_run_stimuli(backbone, preprocessor, config)
    rows, _ = _read_rows(run_dir)
    row0 = next((r for r in rows if r["step"] == 0 and "per_tap" in r), None)
    if row0 is None:
        raise ValueError(f"{run_dir}: metrics.jsonl has no step-0 draw to validate against")
    t_maps = arrays["arms"]["raw"]["t_maps"]
    names_in_row = list(row0["per_tap"])
    t_min = [float(np.min(t_maps[k])) for k in t_maps]
    t_max = [float(np.max(t_maps[k])) for k in t_maps]
    return {
        "limit_pooled_p99_abs_t": _shared_limit(t_maps, 99.0),
        "max_abs_t": float(max(np.abs(v).max() for v in t_maps.values())),
        "recomputed_mean_morans_i": report["arms"]["raw"]["mean_morans_i"],
        "run_step0_mean_morans_i": row0["mean_morans_i"],
        "max_abs_diff_t_min": max(
            abs(a - row0["per_tap"][n]["t_min"]) for a, n in zip(t_min, names_in_row)
        ),
        "max_abs_diff_t_max": max(
            abs(a - row0["per_tap"][n]["t_max"]) for a, n in zip(t_max, names_in_row)
        ),
        "significant_units_all_zero": all(
            v["num_significant_units"] == 0 for v in report["arms"]["raw"]["per_tap"].values()
        ),
    }


def colour_limit_facts(
    png_path: str, recomputed: Mapping[str, Any], device: str = "cpu"
) -> Dict[str, Any]:
    """The ``colour_limit`` entry of a t-map figure.

    :param png_path: The figure.
    :param recomputed: :func:`recompute_trained_limit` or :func:`recompute_initial_limit`.
    :param device: What :func:`recomputation_device` returned; the ``how`` text names it.
    :return: The exact limit with its string, how it was obtained, the recomputation
        check, the largest tick the PNG prints and the pixel cross-check against the
        candidate nearest to the exact limit.
    :raises ValueError: when the colour bar of the PNG yields no candidate.
    """
    exact = recomputed["limit_pooled_p99_abs_t"]
    measured = measure_colour_limit(png_path, max_abs_t=recomputed["max_abs_t"])
    if not measured["candidates"]:
        raise ValueError(f"{png_path}: no colour limit candidate fits the colour bar")
    near = min(measured["candidates"], key=lambda c: abs(c["limit"] - exact))
    largest_tick = max(abs(t) for t in near["ticks"])
    return {
        "value": exact,
        "fmt": f"{exact:.1f}",
        "how": (
            "pooled 99th percentile of |t| over all taps (plotting._shared_limit), "
            f"recomputed on the {'CPU' if device == 'cpu' else device} from the run's own "
            "weights (checkpoints/final.keras, or the seed-42 initial weights for step 0) "
            "with the trainer's own "
            "evaluate_topography; the colour bar spans -limit..+limit"
        ),
        "recomputation_check": dict(recomputed),
        "printed_tick_range": {
            "value": largest_tick, "fmt": f"{largest_tick:g}", "ticks": near["ticks"],
        },
        "pixel_cross_check": {
            "nearest_candidate_limit": near["limit"],
            "relative_difference": abs(near["limit"] - exact) / exact,
            "mismatch_px": near["mismatch_px"],
            "all_candidates": measured["candidates"][:4],
        },
    }


# ---------------------------------------------------------------------
# the entries of ``files``
# ---------------------------------------------------------------------


def file_entry(
    name: str,
    spec: FigureSpec,
    run_dir: str,
    rows: Sequence[Mapping[str, Any]],
    merged: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    recomputed: Optional[Mapping[str, Any]] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """The ``files`` entry of one example figure, read from the PNG in the run directory.

    :param name: The figure's file name in ``figures/`` (an unchanged copy of the
        source).
    :param spec: Where the figure comes from and what its facts hold.
    :param run_dir: The run directory the figure was made in.
    :param rows: All rows of ``metrics.jsonl``.
    :param merged: One row per step.
    :param report: The run's ``topography_report.json``.
    :param recomputed: The colour-limit recomputation of the run, needed when
        ``spec.colour`` is set.
    :param device: What :func:`recomputation_device` returned.
    :return: ``sha256``, ``bytes``, pixel size, ``source`` (the last two directory
        names of the run and the path inside it), the step and run, and ``facts``.
        A cluster figure the run did not write is ``{"omitted": True, "reason": ...}``.
    :raises FileNotFoundError: for a missing figure that is not a cluster figure,
        and for a step the run did not draw.
    """
    path = os.path.join(run_dir, spec.source)
    if not os.path.isfile(path):
        if spec.cluster:
            raw = report["arms"]["raw"]
            return {
                "omitted": True,
                "reason": (
                    "no cluster figure was written by the run: the figure draws condition "
                    "A clusters only and this run found none of condition A (clusters a = "
                    f"{raw['total_clusters_a']}, b = {raw['total_clusters_b']} at step "
                    f"{spec.step})"
                ),
            }
        raise FileNotFoundError(f"figure {name}: {path} does not exist")
    by_step = {r["step"]: r for r in merged}
    if spec.step not in by_step:
        raise FileNotFoundError(
            f"figure {name}: run has no draw at step {spec.step} (drawn: {sorted(by_step)})"
        )
    with Image.open(path) as handle:
        width, height = handle.size
    facts = step_facts(by_step[spec.step])
    if spec.colour is not None:
        if recomputed is None:
            raise ValueError(f"figure {name}: a colour-limit recomputation is required")
        facts["colour_limit"] = colour_limit_facts(path, recomputed, device)
    if spec.cluster:
        facts["cluster_figure_tap"] = cluster_tap(by_step[spec.step], report)
    if spec.curve:
        steps, values = _validation_points(merged)
        peak = max(merged, key=lambda r: r["mean_morans_i"])
        facts["validation_loss_points_drawn"] = {
            "value": {str(int(s)): float(v) for s, v in zip(steps, values)},
            "fmt": ", ".join(f"{int(s)}: {v:.3f}" for s, v in zip(steps, values)),
        }
        facts["mean_morans_i_first_step"] = number(merged[0]["mean_morans_i"], ".3f")
        facts["mean_morans_i_last_step"] = number(merged[-1]["mean_morans_i"], ".3f")
        facts["first_step"] = number(merged[0]["step"], "d")
        facts["last_step"] = number(merged[-1]["step"], "d")
        facts["mean_morans_i_peak"] = number(peak["mean_morans_i"], ".3f")
        facts["mean_morans_i_peak_step"] = number(peak["step"], "d")
        facts["significant_units_mean_series"] = {
            str(r["step"]): r["mean_num_significant_units"] for r in merged
        }
        facts["clusters_a_series"] = {str(r["step"]): r["total_clusters_a"] for r in merged}
        facts["clusters_b_series"] = {str(r["step"]): r["total_clusters_b"] for r in merged}
    absolute = os.path.abspath(run_dir)
    return {
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "width": width,
        "height": height,
        "source": (
            f"{os.path.basename(os.path.dirname(absolute))}/{os.path.basename(absolute)}"
            f"/{spec.source}"
        ),
        "step": spec.step,
        "run": spec.run,
        "facts": facts,
    }


def illustration_entry(png_path: str) -> Dict[str, Any]:
    """Draw the synthetic illustration to ``png_path`` and describe it.

    The picture is drawn by :func:`train.topolm.controls.synthetic_illustration`
    (made-up responses through the real chain); the entry is marked
    ``"kind": "synthetic"``.

    :param png_path: Where the PNG is written.
    :return: The ``files`` entry: hash, size, how it was made, the panels' counts.
    """
    illustration = K.synthetic_illustration(png_path)["facts"]
    # The path of a scratch file is no fact about the figure, and it would make the
    # facts differ from one machine to the next.
    illustration["written"] = None
    with Image.open(png_path) as handle:
        width, height = handle.size
    panels = {
        key: {
            "significant_units": v["significant_units"],
            "clusters_a": v["clusters_a"],
            "clusters_b": v["clusters_b"],
            "cluster_sizes_a": v["cluster_sizes_a"],
            "cluster_sizes_b": v["cluster_sizes_b"],
            "t_min": number(v["t_min"], ".2f"),
            "t_max": number(v["t_max"], ".2f"),
        }
        for key, v in illustration["per_panel"].items()
    }
    return {
        "kind": "synthetic",
        "sha256": sha256_file(png_path),
        "bytes": os.path.getsize(png_path),
        "width": width,
        "height": height,
        "source": "python -m train.topolm.controls --illustration (no model, no run)",
        "step": None,
        "run": None,
        "how_made": (
            "four 16x16 made-up response matrices (10 stimuli by 256 units: smooth "
            "unit-variance noise plus a planted effect of two positive blobs and one "
            "negative blob, added to the 5 a stimuli and subtracted from the 5 b stimuli) "
            "go through the real chain train.topolm.common.score_contrast (Student t-test, "
            f"ONE BH-FDR {illustration['fdr_alpha']} across the four panels, queen cluster "
            f"growth, clusters of at least {illustration['min_cluster_size']} cells) and are "
            "drawn by train.topolm.plotting.plot_t_maps with a SYNTHETIC title and the "
            "figure's own note parameter"
        ),
        "caption_must_say": "synthetic illustration, made-up numbers, not a run",
        "facts": {
            "grid_shape": {
                "value": illustration["grid_shape"],
                "fmt": "x".join(map(str, illustration["grid_shape"])),
            },
            "panels": {"value": illustration["panels"], "fmt": str(illustration["panels"])},
            "fdr_alpha": number(illustration["fdr_alpha"], ".2f"),
            "min_cluster_size": number(illustration["min_cluster_size"], "d"),
            "n_per_condition": number(illustration["n_per_condition"], "d"),
            "per_panel": panels,
        },
        "params": illustration,
    }


# ---------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------


def _git(root: str, *args: str) -> Optional[str]:
    try:
        done = subprocess.run(
            ["git", "-C", root, *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip()


def git_provenance(root: str = REPO_ROOT) -> Dict[str, Any]:
    """The git state the facts were generated at.

    :param root: A git work tree.
    :return: ``head_sha``, ``last_commit_touching_topolm_trainer_or_metrics`` and
        ``topolm_and_metrics_tracked_files_clean`` (no tracked change under
        ``src/train/topolm`` and ``src/dl_techniques/metrics``); every value is
        ``None`` when ``git`` cannot answer.
    """
    paths = ("src/train/topolm", "src/dl_techniques/metrics")
    status = _git(root, "status", "--porcelain", "--untracked-files=no", "--", *paths)
    return {
        "head_sha": _git(root, "rev-parse", "HEAD"),
        "last_commit_touching_topolm_trainer_or_metrics": (
            _git(root, "log", "-1", "--format=%H", "--", *paths) or None
        ),
        "topolm_and_metrics_tracked_files_clean": None if status is None else status == "",
    }


def recomputation_device() -> str:
    """``'cpu'`` when TensorFlow sees no GPU, otherwise the visible GPU devices."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return "cpu" if not gpus else ", ".join(g.name for g in gpus)


# ---------------------------------------------------------------------
# the whole file
# ---------------------------------------------------------------------


def build_facts(
    runs: Mapping[str, str],
    illustration_png: str,
    figures: Mapping[str, FigureSpec] = DEFAULT_FIGURES,
    capture_dir: Optional[str] = None,
    root: str = REPO_ROOT,
) -> Dict[str, Any]:
    """The image and run facts (without the controls), read from the run directories.

    :param runs: ``{name: run directory}``; every ``spec.run`` of ``figures`` must be
        a key.
    :param illustration_png: Where the synthetic illustration is drawn.
    :param figures: The example figures (default :data:`DEFAULT_FIGURES`).
    :param capture_dir: Directory of the capture files, or ``None``.
    :param root: The git work tree for the provenance block.
    :return: ``{"generated_from", "files", "runs"}``, JSON-ready.
    :raises KeyError: naming a figure whose run is not in ``runs``.
    :raises FileNotFoundError: naming a run file or capture file that is missing.
    """
    # DECISION plan-2026-10-23T190557-972a9ac5/D-029: every number of
    # figure_facts.json is produced HERE, from the run directories, the capture
    # files and a recomputation on the machine that runs this. Do NOT type a number
    # into the facts, do NOT read a script, a result or a timing from a plan
    # directory (plans/ is gitignored: nothing there can be regenerated), and do NOT
    # hide a field that depends on the device: list it in DEVICE_DEPENDENT_FIELDS
    # and record the device. Step 12 once shipped facts made by an untracked script
    # (README.md pointed at make_figure_facts.py, which never existed). See
    # decisions.md D-029; guarded by TestTheModuleStandsOnTheRepoAlone and
    # TestTheRegenerationCommandInThisReadme.
    for name, spec in figures.items():
        if spec.run not in runs:
            raise KeyError(f"figure {name} comes from run {spec.run!r}, which was not given")
    for name, directory in runs.items():
        absent = missing_run_files(directory)
        if absent:
            raise FileNotFoundError(f"run {name!r} ({directory}) lacks {', '.join(absent)}")
    loaded: Dict[str, Dict[str, Any]] = {}
    for name, directory in runs.items():
        rows, merged = _read_rows(directory)
        loaded[name] = {
            "rows": rows,
            "merged": merged,
            "report": _read_json(os.path.join(directory, "topography_report.json")),
            "history": _read_json(os.path.join(directory, "training_history.json")),
            "config": _read_json(os.path.join(directory, "config.json")),
        }
    seeds = {loaded[n]["config"]["seed"] for n in runs}
    out: Dict[str, Any] = {"generated_from": git_provenance(root), "files": {}, "runs": {}}
    out["generated_from"]["seed"] = seeds.pop() if len(seeds) == 1 else {
        n: loaded[n]["config"]["seed"] for n in runs
    }
    for name, directory in runs.items():
        out["generated_from"][f"run_dir_{name}"] = _repo_relative(directory, root)
    device = recomputation_device()
    out["generated_from"]["recomputation_device"] = device
    out["generated_from"]["device_dependent_fields"] = list(DEVICE_DEPENDENT_FIELDS)
    out["generated_from"]["capture"] = (
        "none given: exit code, wall time, GPU memory and the training-log fields are absent"
        if capture_dir is None
        else "read from log_<name>.txt, result_<name>.txt and gpu_samples_<name>.csv"
    )
    for name, directory in runs.items():
        data = loaded[name]
        capture = read_capture(capture_dir, name) if capture_dir is not None else None
        entry = run_summary(
            data["rows"], data["merged"], data["report"], data["history"], data["config"],
            capture,
        )
        entry["config"] = {k: data["config"].get(k) for k in CONFIG_KEYS}
        out["runs"][name] = entry
    recomputed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for name, spec in figures.items():
        data = loaded[spec.run]
        directory = runs[spec.run]
        if spec.colour is not None and (spec.run, spec.colour) not in recomputed:
            logger.info(f"Recomputing the {spec.colour} colour limit of run {spec.run!r}")
            recompute = (
                recompute_trained_limit if spec.colour == "trained" else recompute_initial_limit
            )
            recomputed[(spec.run, spec.colour)] = recompute(directory)
        out["files"][name] = file_entry(
            name, spec, directory, data["rows"], data["merged"], data["report"],
            recomputed.get((spec.run, spec.colour)), device,
        )
    for name in runs:
        out["runs"][name]["stimuli_framing"] = stimuli_framing(loaded[name]["config"])
    out["files"][ILLUSTRATION_NAME] = illustration_entry(illustration_png)
    return json.loads(json.dumps(out, allow_nan=False))


def write_facts(
    runs: Mapping[str, str],
    out_path: str,
    controls: Sequence[str] = (),
    n_draws: int = 12,
    seed: int = 0,
    capture_dir: Optional[str] = None,
    illustration_png: Optional[str] = None,
    notes: Optional[Mapping[str, str]] = None,
    figures: Mapping[str, FigureSpec] = DEFAULT_FIGURES,
    root: str = REPO_ROOT,
) -> Dict[str, Any]:
    """Write ``figure_facts.json``: the image and run facts, then the controls.

    The file is assembled in a temporary directory and moved to ``out_path`` at the
    end, so a failure anywhere leaves an existing ``out_path`` as it was.

    :param runs: ``{name: run directory}``, read only.
    :param out_path: The facts file to write.
    :param controls: Names of the runs that get the controls block (computed by
        :func:`train.topolm.controls.facts`); none by default.
    :param n_draws: Draws per row of the length control.
    :param seed: Seed of the length control and of a split subsample.
    :param capture_dir: Directory of the capture files (see the module docstring).
    :param illustration_png: Where the synthetic illustration is drawn; default
        ``t_maps_2d_illustration.png`` beside ``out_path``.
    :param notes: Free-text annotations appended to ``generated_from`` (written by
        hand; this module cannot derive them).
    :param figures: The example figures.
    :param root: The git work tree for the provenance block.
    :return: The written dictionary.
    :raises KeyError: for a ``controls`` name that is not in ``runs``.
    """
    unknown = [n for n in controls if n not in runs]
    if unknown:
        raise KeyError(f"--controls names {unknown}, not among the runs {sorted(runs)}")
    if illustration_png is None:
        illustration_png = os.path.join(os.path.dirname(os.path.abspath(out_path)), ILLUSTRATION_NAME)
    facts = build_facts(runs, illustration_png, figures, capture_dir, root)
    with tempfile.TemporaryDirectory() as scratch:
        base = os.path.join(scratch, "base.json")
        final = os.path.join(scratch, "final.json")
        K.dump_facts(base, facts)
        if controls:
            facts = K.facts(
                {n: os.path.abspath(runs[n]) for n in controls}, final,
                n_draws=n_draws, seed=seed, existing_path=base,
            )
        facts["generated_from"].update(dict(notes or {}))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        K.dump_facts(final, facts)
        shutil.move(final, out_path)
    return facts


# ---------------------------------------------------------------------
# comparing two facts files
# ---------------------------------------------------------------------


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    """A nested dictionary as ``{dotted.path: leaf}``; lists are leaves."""
    if isinstance(value, dict):
        flat: Dict[str, Any] = {}
        for key, child in value.items():
            flat.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def compare_facts(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    device_dependent: Sequence[str] = DEVICE_DEPENDENT_FIELDS,
    provenance: Sequence[str] = PROVENANCE_FIELDS,
) -> Dict[str, List[Tuple[str, Any, Any]]]:
    """Differences between two facts files, split by kind.

    :param expected: The tracked facts.
    :param actual: Regenerated facts.
    :param device_dependent: ``fnmatch`` patterns of dotted paths that depend on the
        device the model was run on.
    :param provenance: Patterns of fields that differ by construction.
    :return: ``{"other", "device_dependent", "provenance"}``, each a list of
        ``(path, expected, actual)`` (``None`` for a side that lacks the path), in
        path order. Only ``other`` means the regeneration does not reproduce the
        tracked file.
    """
    old, new = _flatten(expected), _flatten(actual)
    result: Dict[str, List[Tuple[str, Any, Any]]] = {
        "other": [], "device_dependent": [], "provenance": [],
    }
    for path in sorted(set(old) | set(new)):
        before, after = old.get(path), new.get(path)
        if path in old and path in new and before == after:
            continue
        if _matches(path, provenance):
            kind = "provenance"
        elif _matches(path, device_dependent):
            kind = "device_dependent"
        else:
            kind = "other"
        result[kind].append((path, before, after))
    return result


# ---------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train.topolm.figure_facts",
        description=(
            "Write figures/figure_facts.json from finished TopoLM runs: the facts of "
            "every example figure (hash, size, the numbers its caption quotes, the "
            "recomputed colour limit), the facts of every run, the synthetic "
            "illustration and, with --controls, the framing, length and label-shuffle "
            "controls. Read only on the runs. Set CUDA_VISIBLE_DEVICES= to compute on "
            "the CPU."
        ),
    )
    parser.add_argument(
        "--run", action="append", default=[], metavar="NAME=DIR",
        help="a finished run, e.g. 2d=results/topolm_.../topolm_tiny_...; repeatable. "
             "The names the example figures come from are 2d, 3d and 2d_alpha0",
    )
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="the facts file to write, e.g. src/train/topolm/figures/figure_facts.json")
    parser.add_argument(
        "--controls", action="append", default=[], metavar="NAME",
        help="a run name that also gets the controls block; repeatable (slow: about a minute per run on the CPU)",
    )
    parser.add_argument(
        "--capture-dir", default=None, metavar="DIR",
        help="directory with log_<name>.txt, result_<name>.txt and gpu_samples_<name>.csv "
             "for each run (the training command's console output, 'exit=<code> wall_s=<s>' "
             "and per-second GPU samples); without it those fields are left out",
    )
    parser.add_argument(
        "--illustration-png", default=None, metavar="PNG",
        help=f"where the synthetic illustration is drawn (default: {ILLUSTRATION_NAME} beside --out)",
    )
    parser.add_argument(
        "--note", action="append", default=[], metavar="KEY=TEXT",
        help="a hand-written annotation stored in generated_from; repeatable",
    )
    parser.add_argument("--n-draws", type=int, default=12,
                        help="random draws per length-control row (default 12)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the length-control draws (default 0)")
    parser.add_argument(
        "--compare", default=None, metavar="PATH",
        help="after writing, compare --out with this facts file (the tracked one); "
             "differences are listed as provenance, device-dependent or other, and "
             "the exit code is 1 when there are 'other' ones",
    )
    return parser


def _split_pairs(parser: argparse.ArgumentParser, flag: str, specs: Sequence[str]) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    for spec in specs:
        key, separator, value = spec.partition("=")
        if not separator or not key or not value:
            parser.error(f"{flag} expects NAME=VALUE, got {spec!r}")
        pairs[key] = value
    return pairs


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse and validate every argument first, then write the facts file.

    A run directory that does not exist, or lacks a file the facts need, is refused
    by its name before anything is computed.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    runs = _split_pairs(parser, "--run", args.run)
    if not runs:
        parser.error("give at least one --run NAME=DIR")
    notes = _split_pairs(parser, "--note", args.note)
    for name, directory in runs.items():
        if not os.path.isdir(directory):
            parser.error(f"--run {name}: no such run directory {directory!r}")
        absent = missing_run_files(directory)
        if absent:
            parser.error(f"--run {name}: {directory!r} lacks {', '.join(absent)}")
    for name, spec in DEFAULT_FIGURES.items():
        if spec.run not in runs:
            parser.error(f"figure {name} comes from run {spec.run!r}: give --run {spec.run}=DIR")
    for name in args.controls:
        if name not in runs:
            parser.error(f"--controls {name}: not among the runs {sorted(runs)}")
    if args.capture_dir is not None and not os.path.isdir(args.capture_dir):
        parser.error(f"--capture-dir: no such directory {args.capture_dir!r}")
    if args.n_draws < 1:
        parser.error(f"--n-draws must be >= 1, got {args.n_draws}")
    if args.compare is not None and not os.path.isfile(args.compare):
        parser.error(f"--compare: no such file {args.compare!r}")
    facts = write_facts(
        runs, args.out, controls=args.controls, n_draws=args.n_draws, seed=args.seed,
        capture_dir=args.capture_dir, illustration_png=args.illustration_png, notes=notes,
    )
    logger.info(f"Wrote {args.out}")
    if args.compare is not None:
        differences = compare_facts(_read_json(args.compare), facts)
        for kind in ("provenance", "device_dependent", "other"):
            logger.info(f"{kind}: {len(differences[kind])} differing field(s)")
            for path, before, after in differences[kind]:
                logger.info(f"  {path}: {before!r} -> {after!r}")
        if differences["other"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
