"""``train.topolm.figure_facts`` writes ``figures/figure_facts.json`` (D-029).

Every number in the README's guide to the figures is held to that file by the README
test, and the file used to be written by a script nobody could run (it lived in a
gitignored plan directory). The module is tracked now, so these tests hold the module
itself to account.

Kinds of oracle:

* hand-computed small cases for the pure functions: a row of ``metrics.jsonl`` with two
  taps whose statistics are worked out on paper, a log whose three durations are
  differences of timestamps written in the test, a GPU sample file with a known peak;
* an independent route for the colour limit: figures drawn by ``plot_t_maps`` from
  t-maps whose pooled 99th percentile is known by construction (constant magnitude), and
  the ticks the colour bar must print, written out by hand;
* the run's own report: the recomputed colour limit of a real tiny run must agree with
  the report the run wrote (the module validates the recomputation against it, so a
  recomputation that drifted from the trainer's evaluation shows up here);
* the command line: a missing run directory is refused BY NAME before any work, the
  command in the README names a module that exists and flags that its ``--help`` lists.

The tracked file itself is not regenerated here (that needs the four run directories
under ``results/``, which a clone does not have); the regeneration and its comparison
with the tracked file are recorded in step 12.3 of the plan.
"""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from train.topolm import SMOKE_STIMULI
from train.topolm import common as C
from train.topolm import figure_facts as F
from train.topolm.plotting import plot_t_maps

from .test_topolm_trainer import (  # noqa: F401  (the autouse fixture must be in scope)
    _config,
    _never_write_to_repo_results,
    _patch_data,
)

#: The imported autouse fixture is what keeps run directories out of ``results/``.
_AUTOUSE_FIXTURES_IN_SCOPE = (_never_write_to_repo_results,)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
README = os.path.join(REPO, "src", "train", "topolm", "README.md")
TRACKED_FACTS = os.path.join(REPO, "src", "train", "topolm", "figures", "figure_facts.json")


def _tap(morans, islands, significant, clusters_a, clusters_b, t_min, t_max):
    return {
        "morans_i": morans,
        "morans_i_islands": islands,
        "num_significant_units": significant,
        "clusters_a": clusters_a,
        "clusters_b": clusters_b,
        "t_min": t_min,
        "t_max": t_max,
    }


def _row(step=100, grid=(4, 4), taps=None, **overrides):
    taps = taps if taps is not None else {
        "m/block_0/attention_tap": _tap(0.5, 0.2, 10, 1, 2, -3.0, 4.0),
        "m/block_0/mlp_tap": _tap(0.25, float("nan"), 20, 0, 1, -5.5, 2.0),
    }
    row = {
        "step": step,
        "grid_shape": list(grid),
        "mean_morans_i": 0.375,
        "mean_num_significant_units": 15.0,
        "total_clusters_a": 1,
        "total_clusters_b": 3,
        "per_tap": taps,
        "seconds": 1.0,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------
# the small pure functions
# ---------------------------------------------------------------------


class TestTheSmallFunctions:
    def test_a_number_is_its_value_and_the_string_a_caption_prints(self):
        assert F.number(65.0, ".1f") == {"value": 65.0, "fmt": "65.0"}
        assert F.number(9, "d") == {"value": 9, "fmt": "9"}
        assert F.number(0.7709789, ".3f") == {"value": 0.7709789, "fmt": "0.771"}

    def test_a_missing_or_nan_number_prints_n_a(self):
        assert F.number(None, ".3f") == {"value": None, "fmt": "n/a"}
        assert F.number(float("nan"), ".3f") == {"value": None, "fmt": "n/a"}

    def test_a_tap_label_keeps_the_last_two_path_parts(self):
        assert F.tap_label("topolm/block_2/mlp_tap") == "block_2/mlp_tap"
        assert F.tap_label("a/b/c/block_0/attention_tap") == "block_0/attention_tap"
        assert F.tap_label("mlp_tap") == "mlp_tap"

    def test_sha256_is_hashlibs_over_the_bytes(self, tmp_path):
        path = tmp_path / "x.bin"
        payload = bytes(range(256)) * 9000  # more than one 1 MiB block
        path.write_bytes(payload)
        assert F.sha256_file(str(path)) == hashlib.sha256(payload).hexdigest()

    def test_missing_run_files_are_named(self, tmp_path):
        assert F.missing_run_files(str(tmp_path)) == list(F.REQUIRED_RUN_FILES)
        (tmp_path / "config.json").write_text("{}")
        assert "config.json" not in F.missing_run_files(str(tmp_path))

    def test_a_path_inside_the_root_is_made_relative_and_one_outside_is_kept(self, tmp_path):
        root = str(tmp_path / "repo")
        assert F._repo_relative(os.path.join(root, "results", "a", "b"), root) == os.path.join(
            "results", "a", "b"
        )
        outside = str(tmp_path / "elsewhere" / "run")
        assert F._repo_relative(outside, root) == outside


class TestStepFacts:
    """One row, two taps, every statistic worked out by hand."""

    def test_the_statistics_of_two_taps(self):
        facts = F.step_facts(_row())
        assert facts["step"] == {"value": 100, "fmt": "100"}
        assert facts["grid_shape"] == {"value": [4, 4], "fmt": "4x4"}
        assert facts["units_per_tap"]["value"] == 16
        assert facts["num_taps"]["value"] == 2
        assert facts["mean_morans_i"]["fmt"] == "0.375"
        assert (facts["morans_i_min_over_taps"]["fmt"], facts["morans_i_max_over_taps"]["fmt"]) == (
            "0.250", "0.500")
        # the second tap has no islands Moran's I (NaN): only the first counts
        assert facts["morans_i_islands_max_over_taps"]["fmt"] == "0.200"
        assert facts["morans_i_islands_min_over_taps"]["fmt"] == "0.200"
        assert facts["taps_with_islands_morans_i_defined"]["value"] == 1
        assert facts["mean_significant_units_per_tap"]["fmt"] == "15.0"
        assert (facts["significant_units_min_over_taps"]["value"],
                facts["significant_units_max_over_taps"]["value"]) == (10, 20)
        assert (facts["total_clusters_a"]["value"], facts["total_clusters_b"]["value"]) == (1, 3)
        assert facts["t_min_over_taps"]["fmt"] == "-5.50"
        assert facts["t_max_over_taps"]["fmt"] == "4.00"

    def test_the_per_tap_block_is_keyed_by_the_short_label(self):
        facts = F.step_facts(_row())
        assert list(facts["per_tap"]) == ["block_0/attention_tap", "block_0/mlp_tap"]
        first = facts["per_tap"]["block_0/attention_tap"]
        assert first["morans_i"]["fmt"] == "0.500"
        assert (first["significant_units"]["value"], first["clusters_a"]["value"],
                first["clusters_b"]["value"]) == (10, 1, 2)

    def test_a_3d_grid_has_its_cube_of_units(self):
        facts = F.step_facts(_row(grid=(7, 7, 7)))
        assert facts["grid_shape"]["fmt"] == "7x7x7"
        assert facts["units_per_tap"]["value"] == 343

    def test_without_any_islands_value_the_statistics_say_n_a(self):
        taps = {"m/block_0/mlp_tap": _tap(0.5, None, 3, 0, 0, -1.0, 1.0)}
        facts = F.step_facts(_row(taps=taps))
        assert facts["morans_i_islands_max_over_taps"] == {"value": None, "fmt": "n/a"}
        assert facts["taps_with_islands_morans_i_defined"]["value"] == 0


class TestTheClusterFigureTap:
    TAPS = {
        "m/block_0/attention_tap": _tap(0.5, 0.1, 5, 2, 0, -1.0, 1.0),
        "m/block_0/mlp_tap": _tap(0.4, 0.1, 5, 0, 1, -1.0, 1.0),
        "m/block_1/attention_tap": _tap(0.3, 0.1, 7, 1, 2, -1.0, 1.0),
        "m/block_1/mlp_tap": _tap(0.2, 0.1, 2, 0, 0, -1.0, 1.0),
    }
    REPORT = {"arms": {"raw": {"per_tap": {
        name: {"cluster_sizes_a": [12, 10], "cluster_sizes_b": [11]} for name in TAPS
    }}}}

    def test_it_is_the_deepest_tap_with_a_cluster_of_condition_a(self):
        shown = F.cluster_tap(_row(taps=self.TAPS), self.REPORT)
        assert shown["name"] == {"value": "m/block_1/attention_tap", "fmt": "block_1/attention_tap"}
        assert shown["title_as_drawn"] == "block_1/attention_tap (raw), condition a: 1 cluster"
        assert shown["clusters_b"]["value"] == 2
        assert shown["cluster_sizes_a"] == {"value": [12, 10], "fmt": "12, 10"}
        assert shown["significant_units"]["value"] == 7

    def test_the_title_is_plural_from_two_clusters(self):
        taps = dict(self.TAPS)
        taps["m/block_1/mlp_tap"] = _tap(0.2, 0.1, 9, 3, 0, -1.0, 1.0)
        shown = F.cluster_tap(_row(taps=taps), self.REPORT)
        assert shown["title_as_drawn"] == "block_1/mlp_tap (raw), condition a: 3 clusters"

    def test_without_a_cluster_of_condition_a_there_is_no_tap(self):
        taps = {n: _tap(0.1, 0.1, 1, 0, 4, -1.0, 1.0) for n in self.TAPS}
        assert F.cluster_tap(_row(taps=taps), self.REPORT) is None


# ---------------------------------------------------------------------
# the capture of the training command
# ---------------------------------------------------------------------

LOG = """\
2026-10-24 20:03:03,367 INFO a.py:f:1] first line
I0000 00:00:1792861392.205854 1373669 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 21651 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4090, pci bus id: 0000:05:00.0, compute capability: 8.9
2026-10-24 20:03:13,867 INFO common.py:resolve_cadence:1052] Cadence: train_steps=400 at a 100-step cadence
a line without a timestamp
2026-10-24 20:05:03,367 INFO common.py:train_topolm:2065] Best evaluation 3: val_loss=7.1624 of 4 run
2026-10-24 20:05:05,000 INFO pretrain.py:main:577] Wrote an earlier thing
2026-10-24 20:05:08,117 INFO pretrain.py:main:578] Wrote /some/run
Exception ignored in: <function X>  Python interpreter state is not initialized
Exception ignored in: <function Y>  Python interpreter state is not initialized
Traceback (most recent call last):
"""


class TestTheCapture:
    def test_the_three_durations_are_differences_of_the_timestamps(self):
        facts = F.parse_log(LOG)
        # 20:03:03.367 -> 20:03:13.867 is 10.5 s; the cadence line to 20:05:03.367 is
        # 109.5 s; to the LAST 'Wrote' line (20:05:08.117) it is 114.25 s
        assert facts["first_stamped_line_to_cadence_s"] == 10.5
        assert facts["fit_phase_s_cadence_to_best_evaluation"] == 109.5
        assert facts["cadence_to_wrote_s"] == 114.25

    def test_the_device_the_shutdown_warnings_and_the_tracebacks_are_counted(self):
        facts = F.parse_log(LOG)
        assert facts["device"] == "NVIDIA GeForce RTX 4090"
        assert facts["tf_shutdown_warnings_python_interpreter_state"] == 2
        assert facts["traceback_lines"] == 1

    def test_a_cpu_run_has_no_device_name(self):
        text = "\n".join(l for l in LOG.splitlines() if "Created device" not in l)
        assert F.parse_log(text)["device"] is None

    @pytest.mark.parametrize("missing", ["Cadence:", "Best evaluation", "Wrote"])
    def test_a_log_without_a_needed_line_is_refused_by_what_is_missing(self, missing):
        text = "\n".join(l for l in LOG.splitlines() if missing not in l)
        with pytest.raises(ValueError, match=re.escape(missing)):
            F.parse_log(text)

    def test_the_result_line_gives_wall_seconds_and_exit_code(self):
        assert F.parse_result("exit=0 wall_s=142.845647461\n") == (142.845647461, 0)
        assert F.parse_result("exit=3 wall_s=7.5") == (7.5, 3)

    def test_a_malformed_result_line_is_refused(self):
        with pytest.raises(ValueError, match="wall_s"):
            F.parse_result("exit=0")

    def test_the_gpu_samples_skip_lines_without_a_process_value(self):
        text = "1.0,NA,498,0\n2.0,,498,0\n3.0,4770,5300,90\n4.0,4100,5000,80\n5.0,NA,498,0\n"
        assert F.gpu_samples(text) == [4770, 4100]

    def test_a_missing_capture_file_is_named(self, tmp_path):
        (tmp_path / "log_2d.txt").write_text(LOG)
        with pytest.raises(FileNotFoundError, match="result_2d.txt"):
            F.read_capture(str(tmp_path), "2d")


# ---------------------------------------------------------------------
# the facts of one run
# ---------------------------------------------------------------------


def _run_inputs():
    rows = [
        _row(step=0, seconds=1.0, taps={}),
        _row(step=50, seconds=2.0, taps={}, val_loss=7.0, val_loss_step=50),
        _row(step=100, seconds=4.0, taps={}, val_loss=6.5, val_loss_step=100),
        {"step": 100, "val_only": True, "val_loss": 6.5, "val_loss_step": 100, "seconds": 9.0},
    ]
    merged = rows[:3]
    for row in merged:
        row["mean_morans_i"] = 0.1 * (1 + row["step"] // 50)
    merged[-1].update(mean_morans_i=0.5, mean_num_significant_units=3.0,
                      total_clusters_a=2, total_clusters_b=1)
    report = {
        "stimuli_are_smoke_set": True,
        "readout_skipped": None,
        "arms": {"raw": {
            "mean_morans_i": 0.5, "mean_num_significant_units": 3.0,
            "total_clusters_a": 2, "total_clusters_b": 1,
            "per_tap": {
                "t1": {"num_significant_units": 10, "cluster_sizes_a": [4, 3], "cluster_sizes_b": [2]},
                "t2": {"num_significant_units": 6, "cluster_sizes_a": [], "cluster_sizes_b": []},
            },
        }},
    }
    history = {"val_loss": [8.0, 7.5], "val_perplexity": [2981.0, 1808.0],
               "val_accuracy": [0.1, 0.2], "loss": [9.0, 8.0]}
    config = {"vocab_size": 50304, "model_variant": "tiny"}
    return rows, merged, report, history, config


class TestRunSummary:
    def test_the_per_draw_seconds_are_those_of_the_draws_only(self):
        rows, merged, report, history, config = _run_inputs()
        summary = F.run_summary(rows, merged, report, history, config)
        # draws take 1, 2 and 4 s; the val-only row (9 s) is not a draw
        assert summary["number_of_draws"]["value"] == 3
        assert summary["seconds_per_draw_min"]["fmt"] == "1.00"
        assert summary["seconds_per_draw_median"]["fmt"] == "2.00"
        assert summary["seconds_per_draw_max"]["fmt"] == "4.00"
        assert summary["seconds_per_draw_sum"]["fmt"] == "7.0"

    def test_clusters_hold_nine_of_sixteen_significant_units(self):
        rows, merged, report, history, config = _run_inputs()
        summary = F.run_summary(rows, merged, report, history, config)
        assert summary["significant_units_total"] == 16
        assert summary["significant_units_in_clusters"] == 9  # 4 + 3 + 2
        assert summary["share_of_significant_units_in_clusters"] == 9 / 16

    def test_no_significant_unit_gives_no_share(self):
        rows, merged, report, history, config = _run_inputs()
        for tap in report["arms"]["raw"]["per_tap"].values():
            tap.update(num_significant_units=0, cluster_sizes_a=[], cluster_sizes_b=[])
        summary = F.run_summary(rows, merged, report, history, config)
        assert summary["share_of_significant_units_in_clusters"] is None

    def test_rows_and_series_count_what_the_file_holds(self):
        rows, merged, report, history, config = _run_inputs()
        summary = F.run_summary(rows, merged, report, history, config)
        assert summary["rows_in_metrics_jsonl"] == 4
        assert summary["rows_after_merge_per_step"] == 3
        assert summary["steps_drawn"] == [0, 50, 100]
        assert summary["val_only_rows"] == 1
        assert summary["restored_weights_rows"] == 0
        assert summary["mean_morans_i_series"] == {"0": 0.1, "50": 0.2, "100": 0.5}
        assert summary["clusters_ab_series"]["100"] == [2, 1]
        assert summary["val_loss_points_in_metrics_jsonl"] == {"50": 7.0, "100": 6.5}

    def test_the_last_live_row_equals_the_report_only_when_all_four_numbers_do(self):
        rows, merged, report, history, config = _run_inputs()
        assert F.run_summary(rows, merged, report, history, config)["last_live_row_equals_report"]
        report["arms"]["raw"]["total_clusters_b"] = 2
        assert not F.run_summary(rows, merged, report, history, config)["last_live_row_equals_report"]

    def test_the_training_block_reads_the_last_epoch(self):
        rows, merged, report, history, config = _run_inputs()
        training = F.run_summary(rows, merged, report, history, config)["training"]
        assert training["final_val_loss"] == 7.5
        assert training["final_train_loss"] == 8.0
        assert training["epochs_run"] == 2
        assert training["val_loss_per_virtual_epoch"] == [8.0, 7.5]
        assert (training["vocab_size"], training["model_variant"]) == (50304, "tiny")

    def test_the_capture_fields_are_there_only_with_a_capture(self):
        rows, merged, report, history, config = _run_inputs()
        bare = F.run_summary(rows, merged, report, history, config)
        for key in ("exit_code", "total_wall_s", "gpu_samples", "device",
                    "peak_own_gpu_mib_nvidia_smi_1s_samples", "cadence_to_wrote_s"):
            assert key not in bare, key
        capture = {"result": "exit=0 wall_s=12.34\n",
                   "gpu": "1.0,NA,1,0\n2.0,100,1,0\n3.0,300,1,0\n", "log": LOG}
        full = F.run_summary(rows, merged, report, history, config, capture)
        assert full["exit_code"] == 0
        assert full["total_wall_s"] == {"value": 12.34, "fmt": "12.3"}
        assert full["peak_own_gpu_mib_nvidia_smi_1s_samples"] == {"value": 300, "fmt": "300"}
        assert full["gpu_samples"] == 2
        assert full["device"] == "NVIDIA GeForce RTX 4090"
        # the capture keys come where the shipped file has them: wall time first
        keys = list(full)
        assert keys[:2] == ["exit_code", "total_wall_s"]
        assert keys.index("seconds_per_draw_sum") < keys.index("gpu_samples") < keys.index(
            "last_live_row_equals_report")


# ---------------------------------------------------------------------
# a file entry
# ---------------------------------------------------------------------


def _png(path, size=(7, 5), color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


class TestAFileEntry:
    def _run(self, tmp_path):
        run = tmp_path / "topolm_stamp" / "topolm_tiny_topo"
        _png(run / "topography_live" / "step_0000100" / "morans_i_raw.png")
        return run

    def test_hash_size_source_and_facts_of_the_figure_the_run_wrote(self, tmp_path):
        run = self._run(tmp_path)
        merged = [_row(step=100)]
        spec = F.FigureSpec("2d", "topography_live/step_0000100/morans_i_raw.png", 100)
        entry = F.file_entry("morans_i_by_depth_2d.png", spec, str(run), merged, {})
        png = run / "topography_live" / "step_0000100" / "morans_i_raw.png"
        assert entry["sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()
        assert entry["bytes"] == png.stat().st_size
        assert (entry["width"], entry["height"]) == (7, 5)
        assert entry["source"] == "topolm_stamp/topolm_tiny_topo/topography_live/step_0000100/morans_i_raw.png"
        assert (entry["step"], entry["run"]) == (100, "2d")
        assert entry["facts"]["mean_morans_i"]["fmt"] == "0.375"
        assert "colour_limit" not in entry["facts"]

    def test_a_cluster_figure_the_run_did_not_write_is_omitted_with_the_reason(self, tmp_path):
        run = self._run(tmp_path)
        report = {"arms": {"raw": {"total_clusters_a": 0, "total_clusters_b": 4}}}
        spec = F.FigureSpec("3d", "topography_live/step_0000100/clusters_raw.png", 100, cluster=True)
        entry = F.file_entry("clusters_3d.png", spec, str(run), [_row(step=100)], report)
        assert entry["omitted"] is True
        assert "clusters a = 0, b = 4 at step 100" in entry["reason"]

    def test_another_missing_figure_is_an_error_naming_it(self, tmp_path):
        run = self._run(tmp_path)
        spec = F.FigureSpec("2d", "topography_live/step_0000100/t_maps_raw.png", 100)
        with pytest.raises(FileNotFoundError, match="t_maps_raw.png"):
            F.file_entry("t_maps_2d_trained.png", spec, str(run), [_row(step=100)], {})

    def test_a_step_the_run_did_not_draw_is_an_error_naming_the_drawn_steps(self, tmp_path):
        run = self._run(tmp_path)
        spec = F.FigureSpec("2d", "topography_live/step_0000100/morans_i_raw.png", 400)
        with pytest.raises(FileNotFoundError, match=r"step 400.*\[100\]"):
            F.file_entry("x.png", spec, str(run), [_row(step=100)], {})

    def test_the_curve_carries_the_series_and_the_peak(self, tmp_path):
        run = tmp_path / "stamp" / "run"
        _png(run / "topography_live" / "topography_curve.png")
        merged = [
            _row(step=0, mean_morans_i=0.0, mean_num_significant_units=0.0,
                 total_clusters_a=0, total_clusters_b=0, val_loss=None, val_loss_step=None),
            _row(step=50, mean_morans_i=0.6, mean_num_significant_units=5.0,
                 total_clusters_a=1, total_clusters_b=0, val_loss=7.25, val_loss_step=50),
            _row(step=100, mean_morans_i=0.4, mean_num_significant_units=9.0,
                 total_clusters_a=2, total_clusters_b=1, val_loss=6.5, val_loss_step=100),
        ]
        spec = F.FigureSpec("2d", "topography_live/topography_curve.png", 100, curve=True)
        facts = F.file_entry("live_curve_2d.png", spec, str(run), merged, {})["facts"]
        assert facts["mean_morans_i_peak"]["fmt"] == "0.600"
        assert facts["mean_morans_i_peak_step"]["value"] == 50
        assert (facts["first_step"]["value"], facts["last_step"]["value"]) == (0, 100)
        assert facts["mean_morans_i_first_step"]["fmt"] == "0.000"
        assert facts["mean_morans_i_last_step"]["fmt"] == "0.400"
        # each validation is drawn once, at the step it was measured
        assert facts["validation_loss_points_drawn"]["fmt"] == "50: 7.250, 100: 6.500"
        assert facts["significant_units_mean_series"] == {"0": 0.0, "50": 5.0, "100": 9.0}
        assert facts["clusters_a_series"] == {"0": 0, "50": 1, "100": 2}
        assert facts["clusters_b_series"] == {"0": 0, "50": 0, "100": 1}


# ---------------------------------------------------------------------
# the colour limit
# ---------------------------------------------------------------------


def _t_map_figure(path, limit):
    """A figure whose shared colour limit is ``limit`` by construction.

    Every t-map cell is +limit or -limit, so the pooled 99th percentile of |t| is
    exactly ``limit`` whatever the signs.
    """
    rng = np.random.default_rng(0)
    t_maps = {f"m/block_{i}/mlp_tap": limit * np.sign(rng.standard_normal((16, 16)))
              for i in range(4)}
    plot_t_maps(t_maps, arm="raw", condition_a="a", condition_b="b", output_path=str(path))


class TestTheColourLimit:
    """The limit is read back from the colour bar of a PNG, and chosen by the exact
    value among the candidates the geometry allows."""

    # limit -> the ticks matplotlib's default locator puts on a bar of that range,
    # written out by hand
    CASES = [
        (7.3, [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]),
        (2.6, [-2.0, -1.0, 0.0, 1.0, 2.0]),
        (11.4, [-10.0, -5.0, 0.0, 5.0, 10.0]),
    ]

    @pytest.mark.parametrize("limit,ticks", CASES)
    def test_the_candidate_nearest_the_exact_limit_prints_the_expected_ticks(
        self, tmp_path, limit, ticks
    ):
        png = tmp_path / "t.png"
        _t_map_figure(png, limit)
        measured = F.measure_colour_limit(str(png), max_abs_t=1.5 * limit)
        near = min(measured["candidates"], key=lambda c: abs(c["limit"] - limit))
        assert near["ticks"] == ticks
        assert abs(near["limit"] - limit) / limit < 0.01, "about 0.3 percent of the limit"

    def test_the_facts_entry_names_the_largest_tick_and_the_pixel_cross_check(self, tmp_path):
        png = tmp_path / "t.png"
        _t_map_figure(png, 7.3)
        entry = F.colour_limit_facts(
            str(png), {"limit_pooled_p99_abs_t": 7.3, "max_abs_t": 10.0}
        )
        assert entry["value"] == 7.3 and entry["fmt"] == "7.3"
        assert entry["printed_tick_range"] == {
            "value": 6.0, "fmt": "6", "ticks": [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]}
        assert entry["pixel_cross_check"]["relative_difference"] < 0.01
        assert len(entry["pixel_cross_check"]["all_candidates"]) <= 4
        assert "CPU" in entry["how"]

    def test_the_how_text_names_the_device_it_was_recomputed_on(self, tmp_path):
        png = tmp_path / "t.png"
        _t_map_figure(png, 7.3)
        entry = F.colour_limit_facts(
            str(png), {"limit_pooled_p99_abs_t": 7.3, "max_abs_t": 10.0}, device="NVIDIA X"
        )
        assert "on the NVIDIA X" in entry["how"] and "CPU" not in entry["how"]

    def test_a_candidate_above_the_largest_t_is_not_offered(self, tmp_path):
        png = tmp_path / "t.png"
        _t_map_figure(png, 7.3)
        measured = F.measure_colour_limit(str(png), max_abs_t=5.0)
        assert all(c["limit"] <= 5.0 * 1.0001 for c in measured["candidates"])

    def test_an_image_without_a_colour_bar_is_refused(self, tmp_path):
        png = _png(tmp_path / "blank.png", size=(400, 300), color=(255, 255, 255))
        with pytest.raises(ValueError, match="colour bar"):
            F.measure_colour_limit(str(png))

    def test_longest_run_finds_the_start_and_length_by_hand(self):
        column = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
        assert F._longest_run(column) == (4, 3)


# ---------------------------------------------------------------------
# the stimulus framing, through the real tokenizer
# ---------------------------------------------------------------------


class TestTheStimulusFraming:
    CONFIG = dict(encoding_name="gpt2", max_seq_length=512, cls_token_id=50257,
                  sep_token_id=50258, pad_token_id=50259, mask_token_id=50260)

    def test_token_counts_equal_gpt2s_own_encoding_of_each_sentence(self):
        import tiktoken

        encoder = tiktoken.get_encoding("gpt2")
        facts = F.stimuli_framing(self.CONFIG)
        for condition, prompts in SMOKE_STIMULI.items():
            assert facts[condition]["text_token_counts"] == [len(encoder.encode(p)) for p in prompts]

    def test_the_context_is_followed_by_the_end_of_text_id_and_the_sentence_starts_after_it(self):
        import tiktoken

        encoder = tiktoken.get_encoding("gpt2")
        facts = F.stimuli_framing(self.CONFIG)
        context = len(encoder.encode(C.STIMULUS_CONTEXT))
        assert facts["eot_token_of_encoder"] == encoder.eot_token == 50256
        for condition in SMOKE_STIMULI:
            entry = facts[condition]
            assert entry["context_tokens_before_eot"] == context
            assert entry["sentence_starts_at_position"] == [context + 1] * 5
            assert entry["id_before_the_sentence_is_eot"] is True
            assert entry["cls_id_present"] is False and entry["sep_id_present"] is False
            # the window is the prefix plus the longest sentence, not the trainer's 512
            assert entry["window_length"] == context + 1 + max(entry["text_token_counts"])

    def test_the_summaries_are_the_means_over_the_sentences(self):
        facts = F.stimuli_framing(self.CONFIG)
        for condition in SMOKE_STIMULI:
            counts = facts[condition]["text_token_counts"]
            assert facts["mean_text_tokens_per_condition"][condition] == sum(counts) / 5
        assert facts["text_token_counts_equal_across_conditions"] is True


# ---------------------------------------------------------------------
# a real tiny run: the recomputation is validated against the run's own report
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """One real tiny run with the live view, shared by the tests that need a model."""
    root = tmp_path_factory.mktemp("figure_facts_run")
    mp = pytest.MonkeyPatch()
    try:
        _patch_data(mp)
        real = C._run_dir
        mp.setattr(C, "_run_dir", lambda config: str(root / os.path.basename(real(config))))
        config = _config(
            vocab_size=50261, max_seq_length=160, save_dir="figure_facts_run",
            readout_fwhm=None, min_cluster_size=2, viz_every_steps=2, eval_every_steps=2,
        )
        result = C.train_topolm(config)
    finally:
        mp.undo()
    return result["results_dir"]


class TestTheRecomputationAgreesWithTheRunsOwnReport:
    def test_the_trained_limit_is_checked_against_the_report_the_run_wrote(self, tiny_run):
        recomputed = F.recompute_trained_limit(tiny_run)
        assert recomputed["significant_units_equal"] is True
        assert recomputed["clusters_equal"] is True
        assert recomputed["mean_morans_abs_diff"] < 1e-3
        assert 0.0 < recomputed["limit_pooled_p99_abs_t"] <= recomputed["max_abs_t"]
        # an independent route to the same limit: the percentile of the run's own
        # evaluation, taken here with numpy and no helper of the module
        backbone, preprocessor, config = F.K.load_run(tiny_run)
        _, arrays = C.evaluate_topography(
            backbone, SMOKE_STIMULI, ("a", "b"), preprocessor, output_dir=None,
            readout_fwhm=None, min_cluster_size=config["min_cluster_size"],
            permutation_p_value=config["permutation_p_value"],
            num_permutations=config["num_permutations"], seed=config["seed"],
            is_smoke_set=True,
        )
        pooled = np.concatenate(
            [np.abs(v).ravel() for v in arrays["arms"]["raw"]["t_maps"].values()])
        assert recomputed["limit_pooled_p99_abs_t"] == pytest.approx(
            float(np.percentile(pooled[np.isfinite(pooled)], 99.0)), rel=1e-6)

    def test_the_initial_limit_reproduces_the_step_0_row_of_the_run(self, tiny_run):
        recomputed = F.recompute_initial_limit(tiny_run)
        assert recomputed["max_abs_diff_t_min"] < 1e-3
        assert recomputed["max_abs_diff_t_max"] < 1e-3
        assert abs(recomputed["recomputed_mean_morans_i"]
                   - recomputed["run_step0_mean_morans_i"]) < 1e-3

    def test_a_run_without_a_step_0_draw_is_refused_by_name(self, tiny_run, tmp_path):
        copy = tmp_path / "run"
        copy.mkdir()
        (copy / "config.json").write_text(open(os.path.join(tiny_run, "config.json")).read())
        (copy / "topography_live").mkdir()
        (copy / "topography_live" / "metrics.jsonl").write_text(json.dumps(_row(step=2)) + "\n")
        with pytest.raises(ValueError, match="step-0"):
            F.recompute_initial_limit(str(copy))


class TestTheFactsFileOfARealRun:
    def _figures(self, tiny_run):
        steps = sorted(
            n for n in os.listdir(os.path.join(tiny_run, "topography_live")) if n.startswith("step_"))
        last = steps[-1]
        return {
            "t_maps_untrained.png": F.FigureSpec("t", "topography_live/step_0000000/t_maps_raw.png", 0),
            "t_maps_trained.png": F.FigureSpec(
                "t", f"topography_live/{last}/t_maps_raw.png", int(last.split("_")[1])),
            "live_curve.png": F.FigureSpec(
                "t", "topography_live/topography_curve.png", int(last.split("_")[1]), curve=True),
        }

    def test_the_file_is_written_with_the_controls_and_the_illustration(self, tiny_run, tmp_path):
        out = tmp_path / "out" / "facts.json"
        before = sorted(os.listdir(tiny_run))
        facts = F.write_facts(
            {"t": tiny_run}, str(out), controls=["t"], n_draws=1, figures=self._figures(tiny_run),
            notes={"runs": "a note"}, root=REPO,
        )
        assert json.loads(out.read_text()) == facts, "what was written is what is returned"
        assert sorted(os.listdir(tiny_run)) == before, "the run directory is read only"
        assert list(facts) == ["generated_from", "files", "runs"]
        provenance = facts["generated_from"]
        assert isinstance(provenance["recomputation_device"], str) and provenance["recomputation_device"]
        assert provenance["device_dependent_fields"] == list(F.DEVICE_DEPENDENT_FIELDS)
        assert provenance["run_dirs"] == {"t": os.path.abspath(tiny_run)}
        assert provenance["runs"] == "a note"
        assert provenance["capture"].startswith("none given")
        # the illustration is drawn beside the output and described
        illustration = facts["files"][F.ILLUSTRATION_NAME]
        png = out.parent / F.ILLUSTRATION_NAME
        assert illustration["kind"] == "synthetic"
        assert illustration["sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()
        assert illustration["params"]["written"] is None
        run = facts["runs"]["t"]
        assert run["controls"]["label_shuffle_null"]["n_splits"] == 252
        assert run["stimuli_framing"]["a"]["context_tokens_before_eot"] == 60
        assert "exit_code" not in run, "no capture, no capture fields"
        assert [name for name in facts["files"]][:3] == [
            "t_maps_untrained.png", "t_maps_trained.png", "live_curve.png"]

    def test_the_same_inputs_give_the_same_file(self, tiny_run, tmp_path):
        figures = self._figures(tiny_run)
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        F.write_facts({"t": tiny_run}, str(first), figures=figures, root=REPO)
        F.write_facts({"t": tiny_run}, str(second), figures=figures, root=REPO)
        assert first.read_bytes() == second.read_bytes()

    def test_an_error_leaves_an_existing_file_as_it_was(self, tiny_run, tmp_path):
        out = tmp_path / "facts.json"
        out.write_text("precious")
        figures = {"x.png": F.FigureSpec("t", "topography_live/step_0009999/t_maps_raw.png", 9999)}
        with pytest.raises(FileNotFoundError, match="x.png"):
            F.write_facts({"t": tiny_run}, str(out), figures=figures, root=REPO)
        assert out.read_text() == "precious"

    def test_a_figure_whose_run_was_not_given_is_named(self, tiny_run):
        with pytest.raises(KeyError, match="t_maps_2d_untrained.png.*'2d'"):
            F.build_facts({"t": tiny_run}, "unused.png", root=REPO)

    def test_a_controls_name_that_is_not_a_run_is_refused(self, tiny_run, tmp_path):
        with pytest.raises(KeyError, match="'nope'"):
            F.write_facts({"t": tiny_run}, str(tmp_path / "f.json"), controls=["nope"],
                          figures={}, root=REPO)

    def test_the_capture_fields_come_from_the_capture_directory(self, tiny_run, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        (capture / "log_t.txt").write_text(LOG)
        (capture / "result_t.txt").write_text("exit=0 wall_s=12.34\n")
        (capture / "gpu_samples_t.csv").write_text("1.0,NA,1,0\n2.0,250,1,0\n")
        facts = F.build_facts({"t": tiny_run}, str(tmp_path / "i.png"), figures={},
                              capture_dir=str(capture), root=REPO)
        run = facts["runs"]["t"]
        assert run["exit_code"] == 0 and run["total_wall_s"]["fmt"] == "12.3"
        assert run["peak_own_gpu_mib_nvidia_smi_1s_samples"]["value"] == 250
        assert facts["generated_from"]["capture"].startswith("read from")

    def test_a_missing_capture_file_is_named(self, tiny_run, tmp_path):
        capture = tmp_path / "capture"
        capture.mkdir()
        with pytest.raises(FileNotFoundError, match="log_t.txt"):
            F.build_facts({"t": tiny_run}, str(tmp_path / "i.png"), figures={},
                          capture_dir=str(capture), root=REPO)


# ---------------------------------------------------------------------
# comparing two facts files
# ---------------------------------------------------------------------


class TestComparingFactsFiles:
    EXPECTED = {
        "generated_from": {"head_sha": "aaa", "seed": 42, "run_dir_2d": "results/x"},
        "files": {"f.png": {"facts": {"colour_limit": {"value": 8.0, "fmt": "8.0"}}, "sha256": "h"}},
        "runs": {"2d": {"controls": {"y": 2.0}, "exit_code": 0}},
    }

    def _actual(self, **changes):
        actual = json.loads(json.dumps(self.EXPECTED))
        for path, value in changes.items():
            node = actual
            *parents, last = path.split("/")
            for key in parents:
                node = node[key]
            node[last] = value
        return actual

    def test_identical_files_have_no_difference(self):
        assert F.compare_facts(self.EXPECTED, self._actual()) == {
            "other": [], "device_dependent": [], "provenance": []}

    def test_each_difference_lands_in_its_kind(self):
        actual = self._actual(**{
            "generated_from/head_sha": "bbb",
            "files/f.png/facts/colour_limit/value": 8.5,
            "generated_from/seed": 43,
        })
        diff = F.compare_facts(self.EXPECTED, actual)
        assert diff["provenance"] == [("generated_from.head_sha", "aaa", "bbb")]
        assert diff["device_dependent"] == [("files.f.png.facts.colour_limit.value", 8.0, 8.5)]
        assert diff["other"] == [("generated_from.seed", 42, 43)]

    def test_a_changed_control_number_is_an_other_difference(self):
        """Review pass 6, concern 3: the controls were classed as device dependent, so a
        changed control count could never fail the comparison."""
        diff = F.compare_facts(self.EXPECTED, self._actual(**{"runs/2d/controls": {"y": 2.5}}))
        assert diff["other"] == [("runs.2d.controls.y", 2.0, 2.5)]
        assert not diff["device_dependent"] and not diff["provenance"]

    def test_a_changed_hash_is_an_other_difference(self):
        diff = F.compare_facts(self.EXPECTED, self._actual(**{"files/f.png/sha256": "g"}))
        assert [d[0] for d in diff["other"]] == ["files.f.png.sha256"]

    def test_a_key_only_one_side_has_is_a_difference_with_none_for_the_other(self):
        actual = self._actual()
        actual["runs"]["2d"]["extra"] = 1
        del actual["runs"]["2d"]["exit_code"]
        diff = F.compare_facts(self.EXPECTED, actual)
        assert ("runs.2d.exit_code", 0, None) in diff["other"]
        assert ("runs.2d.extra", None, 1) in diff["other"]

    def test_the_colour_limit_is_device_dependent_and_the_hash_beside_it_is_not(self):
        actual = self._actual(**{"files/f.png/facts/colour_limit/value": 8.0000001})
        diff = F.compare_facts(self.EXPECTED, actual)
        assert diff["device_dependent"] and not diff["other"]

    def test_the_tracked_file_lists_no_field_this_module_would_call_device_dependent_by_accident(self):
        """Every pattern matches something in the tracked file, so none is dead text."""
        with open(TRACKED_FACTS, encoding="utf-8") as handle:
            flat = F._flatten(json.load(handle))
        for pattern in F.DEVICE_DEPENDENT_FIELDS:
            assert any(F._matches(path, [pattern]) for path in flat), pattern


# ---------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------


def _fake_run_dir(path):
    path.mkdir(parents=True)
    for name in F.REQUIRED_RUN_FILES:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    return str(path)


def _all_runs(tmp_path):
    return [
        arg for name in ("2d", "3d", "2d_alpha0")
        for arg in ("--run", f"{name}={_fake_run_dir(tmp_path / ('run_' + name))}")
    ]


class TestTheFigureFactsCommandLine:
    def test_help_exits_zero_with_a_usage_line(self):
        env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src"), "CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run(
            [sys.executable, "-m", "train.topolm.figure_facts", "--help"],
            capture_output=True, text=True, env=env, cwd=REPO,
        )
        assert done.returncode == 0, done.stderr[-500:]
        assert "usage: python -m train.topolm.figure_facts" in done.stdout
        for flag in ("--run", "--out", "--controls", "--capture-dir", "--compare", "--note"):
            assert flag in done.stdout, flag

    def test_a_missing_run_directory_is_refused_by_its_name_before_any_work(
        self, monkeypatch, capsys, tmp_path
    ):
        def boom(*args, **kwargs):
            raise AssertionError("work started before the arguments were validated")

        monkeypatch.setattr(F, "write_facts", boom)
        missing = str(tmp_path / "no" / "such" / "run")
        with pytest.raises(SystemExit) as excinfo:
            F.main(["--run", f"2d={missing}", "--out", str(tmp_path / "f.json")])
        assert excinfo.value.code == 2
        error = capsys.readouterr().err
        assert "--run 2d" in error and missing in error and "no such run directory" in error

    def test_a_run_directory_without_its_files_names_them(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(F, "write_facts", lambda *a, **k: pytest.fail("work started"))
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(SystemExit) as excinfo:
            F.main(["--run", f"3d={empty}", "--out", str(tmp_path / "f.json")])
        assert excinfo.value.code == 2
        error = capsys.readouterr().err
        assert "--run 3d" in error and "config.json" in error and "metrics.jsonl" in error

    def test_the_second_run_of_two_is_the_one_named_when_it_is_the_missing_one(
        self, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(F, "write_facts", lambda *a, **k: pytest.fail("work started"))
        good = _fake_run_dir(tmp_path / "good")
        with pytest.raises(SystemExit):
            F.main(["--run", f"2d={good}", "--run", f"3d={tmp_path / 'gone'}",
                    "--out", str(tmp_path / "f.json")])
        error = capsys.readouterr().err
        assert "--run 3d" in error and "--run 2d" not in error

    def test_a_figure_whose_run_is_not_given_is_named(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(F, "write_facts", lambda *a, **k: pytest.fail("work started"))
        with pytest.raises(SystemExit):
            F.main(["--run", f"2d={_fake_run_dir(tmp_path / 'only')}", "--out", str(tmp_path / "f.json")])
        error = capsys.readouterr().err
        assert "3d" in error and "give --run 3d=DIR" in error

    @pytest.mark.parametrize("extra,needle", [
        (["--controls", "nope"], "--controls nope"),
        (["--capture-dir", "/no/such/capture"], "--capture-dir"),
        (["--n-draws", "0"], "--n-draws"),
        (["--compare", "/no/such/file.json"], "--compare"),
        (["--note", "no-equals-sign"], "--note"),
    ])
    def test_each_bad_option_is_refused_by_name(self, monkeypatch, capsys, tmp_path, extra, needle):
        monkeypatch.setattr(F, "write_facts", lambda *a, **k: pytest.fail("work started"))
        with pytest.raises(SystemExit) as excinfo:
            F.main(_all_runs(tmp_path) + ["--out", str(tmp_path / "f.json")] + extra)
        assert excinfo.value.code == 2
        assert needle in capsys.readouterr().err

    def test_no_run_at_all_and_a_malformed_run_are_usage_errors(self, capsys, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            F.main(["--out", str(tmp_path / "f.json")])
        assert excinfo.value.code == 2
        with pytest.raises(SystemExit) as excinfo:
            F.main(["--run", "no-equals-sign", "--out", str(tmp_path / "f.json")])
        assert excinfo.value.code == 2
        assert "usage:" in capsys.readouterr().err

    def test_main_passes_its_arguments_to_write_facts(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(F, "write_facts", lambda *a, **k: calls.append((a, k)) or {})
        out = str(tmp_path / "f.json")
        F.main(_all_runs(tmp_path) + [
            "--out", out, "--controls", "2d", "--controls", "3d", "--n-draws", "5",
            "--seed", "9", "--note", "runs=a note",
        ])
        (args, kwargs), = calls
        assert sorted(args[0]) == ["2d", "2d_alpha0", "3d"] and args[1] == out
        assert kwargs["controls"] == ["2d", "3d"]
        assert kwargs["n_draws"] == 5 and kwargs["seed"] == 9
        assert kwargs["notes"] == {"runs": "a note"}
        assert kwargs["capture_dir"] is None

    def test_a_comparison_with_an_other_difference_exits_1(self, monkeypatch, tmp_path):
        facts = {"generated_from": {"seed": 1}}
        monkeypatch.setattr(F, "write_facts", lambda *a, **k: facts)
        tracked = tmp_path / "tracked.json"
        tracked.write_text(json.dumps({"generated_from": {"seed": 2}}))
        argv = _all_runs(tmp_path) + ["--out", str(tmp_path / "f.json"), "--compare", str(tracked)]
        with pytest.raises(SystemExit) as excinfo:
            F.main(argv)
        assert excinfo.value.code == 1
        tracked.write_text(json.dumps({"generated_from": {"seed": 1}}))
        F.main(argv)  # identical: returns


# ---------------------------------------------------------------------
# the module stands on the repository alone
# ---------------------------------------------------------------------


class TestTheModuleStandsOnTheRepoAlone:
    def _source(self):
        with open(os.path.join(REPO, "src", "train", "topolm", "figure_facts.py"),
                  encoding="utf-8") as handle:
            return handle.read()

    def test_it_imports_nothing_from_a_plan_directory_and_hacks_no_path(self):
        source = self._source()
        modules = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                modules += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        assert modules, "no imports found: the check is vacuous"
        assert not [m for m in modules if m.split(".")[0] in ("plans", "scratch", "colour_limit")]
        assert "sys.path" not in source

    def test_it_carries_no_absolute_path_and_no_em_dash(self):
        source = self._source()
        for fragment in ("/media/", "/home/", "/tmp/", "plans/plan"):
            assert fragment not in source, fragment
        assert chr(0x2014) not in source

    def test_it_logs_through_the_logger_and_never_prints(self):
        calls = [
            node for node in ast.walk(ast.parse(self._source()))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
        ]
        assert not calls


# ---------------------------------------------------------------------
# the regeneration command in the README
# ---------------------------------------------------------------------


def _readme():
    with open(README, encoding="utf-8") as handle:
        return handle.read()


def _regeneration_block():
    blocks = re.findall(r"```bash\n(.*?)```", _readme(), re.S)
    found = [b for b in blocks if "train.topolm.figure_facts" in b]
    assert len(found) == 1, "the README must show the regeneration command exactly once"
    return found[0].replace("\\\n", " ")


class TestTheRegenerationCommandInThisReadme:
    def test_it_names_a_module_that_exists_and_whose_help_works(self):
        import importlib.util

        block = _regeneration_block()
        module = re.search(r"-m (train\.topolm\.figure_facts)\b", block).group(1)
        assert importlib.util.find_spec(module) is not None
        env = {**os.environ, "PYTHONPATH": os.path.join(REPO, "src"), "CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True,
                              text=True, env=env, cwd=REPO)
        assert done.returncode == 0, done.stderr[-500:]
        # the flags are the command's own arguments, not words inside a --note text
        import shlex

        flags = {a for a in shlex.split(block) if re.fullmatch(r"--[a-z][a-z-]+", a)}
        assert flags >= {"--run", "--out", "--controls", "--capture-dir", "--note"}
        for flag in flags:
            assert flag in done.stdout, f"{flag} is in the README command but not in --help"

    def test_it_writes_the_tracked_file_and_names_every_run_the_figures_come_from(self):
        block = _regeneration_block()
        assert "--out src/train/topolm/figures/figure_facts.json" in block
        assert os.path.isfile(TRACKED_FACTS)
        named = set(re.findall(r"--run (\w+)=", block))
        assert named >= {spec.run for spec in F.DEFAULT_FIGURES.values()}
        controls = set(re.findall(r"--controls (\w+)", block))
        assert controls <= named

    def test_no_readme_points_at_a_script_that_does_not_exist(self):
        for path in (README, os.path.join(REPO, "src", "dl_techniques", "models", "language",
                                          "topolm", "README.md")):
            with open(path, encoding="utf-8") as handle:
                assert "make_figure_facts" not in handle.read(), path
        assert not os.path.exists(
            os.path.join(REPO, "src", "train", "topolm", "figures", "make_figure_facts.py"))

    def test_every_test_class_the_readme_cites_for_the_module_exists_here(self):
        with open(__file__, encoding="utf-8") as handle:
            mine = handle.read()
        readme = _readme()
        cited = re.search(r"`test_topolm_figure_facts\.py`: (.*?)\)\.", readme, re.S)
        assert cited, "the README names the test file"
        names = re.findall(r"`(Test\w+)`", cited.group(1))
        assert names
        for name in names:
            assert re.search(rf"^class {name}\b", mine, re.M), name

    def test_the_default_figure_sources_are_the_paths_the_readme_lists(self):
        readme = _readme()
        for name, spec in F.DEFAULT_FIGURES.items():
            assert spec.source.split("/")[-1] in readme, name
            assert name in readme or name == "t_maps_2d_illustration.png", name


# ---------------------------------------------------------------------
# guards against reading the module's numbers wrongly
# ---------------------------------------------------------------------


class TestTheTrackedFactsHaveTheShapeTheModuleWrites:
    """The module's output is what the README test consumes: the tracked file has every
    top-level key and every per-run key this module writes (it was regenerated from the
    module's predecessor, and step 12.3 recorded the comparison)."""

    def test_the_tracked_file_has_the_keys_of_a_run_summary(self):
        with open(TRACKED_FACTS, encoding="utf-8") as handle:
            tracked = json.load(handle)
        rows, merged, report, history, config = _run_inputs()
        capture = {"result": "exit=0 wall_s=1\n", "gpu": "1,2,3,4\n", "log": LOG}
        written = F.run_summary(rows, merged, report, history, config, capture)
        written["config"] = {}
        written["stimuli_framing"] = {}
        assert set(written) <= set(tracked["runs"]["2d"]), set(written) - set(tracked["runs"]["2d"])
        assert list(tracked["runs"]["2d"])[: len(written) - 2] == list(written)[:-2]
        assert set(tracked["files"]) >= set(F.DEFAULT_FIGURES) | {F.ILLUSTRATION_NAME}
        assert list(tracked["files"]) == list(F.DEFAULT_FIGURES) + [F.ILLUSTRATION_NAME]

    def test_the_figure_manifest_matches_the_tracked_sources(self):
        with open(TRACKED_FACTS, encoding="utf-8") as handle:
            tracked = json.load(handle)
        for name, spec in F.DEFAULT_FIGURES.items():
            entry = tracked["files"][name]
            assert entry["source"].endswith(spec.source), name
            assert entry["step"] == spec.step and entry["run"] == spec.run, name
            has_colour = "colour_limit" in entry["facts"]
            assert has_colour == (spec.colour is not None), name
            assert ("cluster_figure_tap" in entry["facts"]) == spec.cluster, name
            assert ("validation_loss_points_drawn" in entry["facts"]) == spec.curve, name



# ---------------------------------------------------------------------
# the controls are compared exactly (review pass 6, concern 3)
# ---------------------------------------------------------------------


def _tracked():
    with open(TRACKED_FACTS, encoding="utf-8") as handle:
        return json.load(handle)


def _with(facts, dotted, value):
    """A deep copy of ``facts`` with the value at ``dotted`` replaced."""
    changed = json.loads(json.dumps(facts))
    *parents, last = dotted.split(".")
    node = changed
    for key in parents:
        node = node[key]
    assert last in node, dotted
    node[last] = value
    return changed


#: The three control numbers the review changed (the README's four-controls numbers):
#: the shipped framing's count, the label-shuffle null's count at or above the real
#: split, and a median of the length control.
CONTROL_FIELDS = {
    "runs.2d.controls.framings.context.mean_significant_units_per_tap": 0.0,
    "runs.2d.controls.label_shuffle_null.n_at_or_above": 120,
    "runs.2d.controls.length_control.b_vocab_3v7.median": 64.0,
    "runs.3d.controls.observed_a_vs_b.clusters_a": 0,
}


class TestAChangedControlNumberFailsTheComparison:
    def test_the_controls_are_not_listed_as_device_dependent(self):
        assert not [p for p in F.DEVICE_DEPENDENT_FIELDS if "controls" in p]

    @pytest.mark.parametrize("dotted, value", sorted(CONTROL_FIELDS.items()))
    def test_each_changed_control_number_is_an_other_difference(self, dotted, value):
        tracked = _tracked()
        diff = F.compare_facts(tracked, _with(tracked, dotted, value))
        assert [d[0] for d in diff["other"]] == [dotted]
        assert not diff["device_dependent"] and not diff["provenance"]

    @pytest.mark.parametrize("dotted, value", sorted(CONTROL_FIELDS.items()))
    def test_the_command_exits_1_for_it(self, monkeypatch, tmp_path, dotted, value):
        tracked = _tracked()
        monkeypatch.setattr(F, "write_facts", lambda *a, **k: _with(tracked, dotted, value))
        path = tmp_path / "tracked.json"
        path.write_text(json.dumps(tracked))
        argv = _all_runs(tmp_path) + ["--out", str(tmp_path / "f.json"), "--compare", str(path)]
        with pytest.raises(SystemExit) as excinfo:
            F.main(argv)
        assert excinfo.value.code == 1

    def test_the_unchanged_file_compares_clean(self):
        tracked = _tracked()
        assert F.compare_facts(tracked, json.loads(json.dumps(tracked))) == {
            "other": [], "device_dependent": [], "provenance": []}

    def test_no_control_field_is_a_wall_clock_measurement(self):
        """The reason nothing in the controls block needs an exemption: every field is a
        count, a statistic of a seeded draw or a label, and none a duration."""
        flat = F._flatten(_tracked())
        names = [p for p in flat if ".controls." in p]
        assert names
        clock = {"second", "seconds", "wall", "elapsed", "duration", "time", "ms", "s"}
        found = [p for p in names if clock & set(re.split(r"[._]", p))]
        assert not found, found


# ---------------------------------------------------------------------
# the README command reproduces the tracked file (review pass 6, concern 6)
# ---------------------------------------------------------------------


def _command_arguments():
    """The arguments of the README's regeneration command, split the way a shell does."""
    import shlex

    return shlex.split(_regeneration_block())


def _notes_of_the_command():
    arguments = _command_arguments()
    return dict(
        argument.split("=", 1)
        for flag, argument in zip(arguments, arguments[1:]) if flag == "--note"
    )


class TestTheTrackedFileIsWhatTheModuleWrites:
    """The tracked file used to be the scratch output of step 12, never rewritten by the
    module: it lacked the provenance keys the README described, and the README command
    (no ``--note``) did not reproduce its two hand-written notes, so a regeneration
    exited 1. Now the tracked file is the module's own output, and the command carries
    the notes."""

    def test_it_has_the_provenance_keys_the_readme_describes(self):
        provenance = _tracked()["generated_from"]
        assert provenance["device_dependent_fields"] == list(F.DEVICE_DEPENDENT_FIELDS)
        assert isinstance(provenance["recomputation_device"], str) and provenance["recomputation_device"]
        assert "read from log_<name>.txt" in provenance["capture"]
        assert provenance["recomputation_device"].lower() == "cpu"

    def test_the_readme_says_what_the_tracked_file_has(self):
        text = " ".join(_readme().split())
        for key in ("generated_from.device_dependent_fields", "generated_from.recomputation_device"):
            assert key in text
            assert key.split(".")[1] in _tracked()["generated_from"]

    def test_the_commands_notes_are_the_tracked_notes(self):
        notes = _notes_of_the_command()
        assert sorted(notes) == ["note_tracked_clean", "runs"]
        for key, text in notes.items():
            assert _tracked()["generated_from"][key] == text

    def test_every_run_and_every_controls_name_of_the_command_is_in_the_tracked_file(self):
        arguments = _command_arguments()
        named = [a.split("=", 1)[0] for flag, a in zip(arguments, arguments[1:]) if flag == "--run"]
        controls = [a for flag, a in zip(arguments, arguments[1:]) if flag == "--controls"]
        tracked = _tracked()
        assert sorted(named) == sorted(tracked["runs"])
        assert controls and all("controls" in tracked["runs"][name] for name in controls)
        assert sorted(controls) == sorted(n for n, r in tracked["runs"].items() if "controls" in r)

    def test_the_commands_placeholders_are_the_ones_the_readme_explains(self):
        block = _regeneration_block()
        assert set(re.findall(r"<([^>]+)>", block)) == {"run-dir", "capture-dir"}
        text = _readme()
        assert "`<run-dir>`" in text and "`<capture-dir>`" in text

    def test_the_command_writes_the_tracked_file_and_compare_is_how_it_is_checked(self):
        assert "--out src/train/topolm/figures/figure_facts.json" in _regeneration_block()
        assert "--compare" in _readme()
