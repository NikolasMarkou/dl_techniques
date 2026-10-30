"""The Phase B aggregator: rows keyed by the full tuple, paired differences, nothing dropped.

Plan plan-2026-10-30T025213-f75ab4e9 step 5. The rows below are synthetic but have the shape
``common.run_sweep`` writes (a real smoke row was read to pin it). Every rule is pinned by a
test that a plausible mutant turns red: a reader keyed on the seed alone, pairing by position
instead of by seed, a std with ddof=0, ranking by test accuracy, dropping failed cells.
Everything is written under ``tmp_path``; nothing reads ``results/``.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from train.mean_shift import corruption, report

REPO = Path(__file__).resolve().parents[3]
CONDITIONS = list(corruption.CONDITIONS)


def accuracy_row(arm, seed, clean, *, status="ok", attempt=1, dataset="cifar10", params=1000,
                 seconds_per_epoch=2.0, select=None, error_tail=None):
    """An exp1 row; every condition is ``clean`` minus a fixed amount so diffs are easy."""
    accuracy = {name: round(clean - 0.1 * index, 6) for index, name in enumerate(CONDITIONS)}
    accuracy["clean"] = clean
    metrics = None
    if status == "ok":
        metrics = {"params": params, "seconds_per_epoch": seconds_per_epoch, "finite": True,
                   "metrics": {"accuracy": accuracy, "dataset": dataset, "epochs": 2, "seed": seed,
                               "select_accuracy": clean if select is None else select}}
    return {"experiment": "exp1", "arm": arm, "config": dataset, "seed": seed, "status": status,
            "attempt": attempt, "cell_dir": f"cells/exp1__{arm}__{dataset}__s{seed}__try{attempt}",
            "returncode": 0 if status == "ok" else 1, "metrics": metrics, "error_tail": error_tail}


def bandwidth_block(layer_means, *, inside=True):
    return {"status": "extracted", "mode": "per_head", "probe_images": 512, "layers": [
        {"layer": i, "mean": m, "min": m - 0.1, "max": m + 0.1,
         "per_head_mean": [m - 0.05, m + 0.05], "inside_clip_range": inside}
        for i, m in enumerate(layer_means)]}


def exp2_row(config, seed, select, test, *, steps=3, attention="mean_shift", bandwidths=None,
             params=900, status="ok", attempt=1):
    if bandwidths is None:
        bandwidths = {"status": "not applicable: the model has no MeanShiftAttention layer",
                      "layers": []}
    args = {"num_steps": steps} if attention == "mean_shift" else {}
    metrics = None
    if status == "ok":
        metrics = {"params": params, "seconds_per_epoch": 3.0, "finite": True, "metrics": {
            "dataset": "cifar100", "epochs": 2, "seed": seed, "config": config,
            "config_args": args, "select_accuracy": select, "test_accuracy": test,
            "throughput_images_per_s": 1000.0, "bandwidths": bandwidths}}
    return {"experiment": "exp2", "arm": attention, "config": config, "seed": seed,
            "status": status, "attempt": attempt, "returncode": 0 if status == "ok" else 1,
            "cell_dir": f"cells/exp2__{attention}__{config}__s{seed}__try{attempt}",
            "metrics": metrics, "error_tail": None if status == "ok" else "boom"}


def make_root(tmp_path, name, rows, manifest_cells=None):
    root = tmp_path / name
    root.mkdir()
    (root / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if manifest_cells is not None:
        cells = [{"experiment": e, "arm": a, "config": c, "seed": s, "extra_args": []}
                 for e, a, c, s in manifest_cells]
        (root / "manifest.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return root


def variant(summary, name, experiment="exp1"):
    group = next(g for g in summary["groups"] if g["experiment"] == experiment)
    return next(v for v in group["variants"] if v["variant"] == name)


# the exp1 fixture: two arms x two seeds, deliberately not seed-aligned in value
MEAN_S0, MEAN_S1 = 0.50, 0.60
CLS_S0, CLS_S1 = 0.55, 0.70


@pytest.fixture
def two_by_two(tmp_path):
    rows = [accuracy_row("mean", 1, MEAN_S1), accuracy_row("mean", 0, MEAN_S0),
            accuracy_row("cls", 0, CLS_S0), accuracy_row("cls", 1, CLS_S1)]
    return make_root(tmp_path, "exp1", rows)


# ---------------------------------------------------------------------------
# the key is the full tuple
# ---------------------------------------------------------------------------

def test_two_arms_with_the_same_seeds_stay_two_cells(two_by_two):
    """A reader keyed on the seed alone keeps one row per seed and loses an arm."""
    summary = report.build_summary([two_by_two])
    assert summary["counts"] == {"ok": 4, "failed": 0, "missing": 0}
    mean, cls = variant(summary, "mean"), variant(summary, "cls")
    assert mean["n_seeds"] == cls["n_seeds"] == 2
    assert mean["metrics"]["clean"]["values"] == [MEAN_S0, MEAN_S1]
    assert cls["metrics"]["clean"]["values"] == [CLS_S0, CLS_S1]
    assert mean["metrics"]["clean"]["mean"] == pytest.approx(0.55)
    assert cls["metrics"]["clean"]["mean"] == pytest.approx(0.625)


def test_the_same_seed_in_two_experiments_is_two_cells(tmp_path):
    rows = [accuracy_row("mean", 0, 0.5), exp2_row("mha", 0, 0.30, 0.31, attention="multi_head")]
    summary = report.build_summary([make_root(tmp_path, "both", rows)])
    assert {g["experiment"] for g in summary["groups"]} == {"exp1", "exp2"}


def test_the_same_ok_cell_in_two_roots_is_refused(tmp_path):
    first = make_root(tmp_path, "a", [accuracy_row("mean", 0, 0.5)])
    second = make_root(tmp_path, "b", [accuracy_row("mean", 0, 0.6)])
    with pytest.raises(ValueError, match="appears twice"):
        report.build_summary([first, second])


# ---------------------------------------------------------------------------
# mean, std and the paired difference
# ---------------------------------------------------------------------------

def test_the_std_is_the_sample_std(two_by_two):
    """Two seeds 0.50 and 0.60: ddof=1 gives 0.0707107, ddof=0 would give 0.05."""
    mean = variant(report.build_summary([two_by_two]), "mean")
    assert mean["metrics"]["clean"]["std"] == pytest.approx(0.0707106781)


def test_the_paired_difference_is_formed_seed_by_seed(two_by_two):
    cls = variant(report.build_summary([two_by_two]), "cls")
    paired = cls["paired"]["clean"]
    assert paired["reference"] == "mean"
    assert paired["seeds"] == [0, 1] and paired["n_pairs"] == 2
    assert paired["diffs"] == pytest.approx([CLS_S0 - MEAN_S0, CLS_S1 - MEAN_S1])
    assert paired["mean"] == pytest.approx(0.075)
    assert paired["std"] == pytest.approx(0.0353553391)


def test_pairing_uses_the_seeds_both_arms_finished(tmp_path):
    """cls lost seed 0: positional pairing would give cls[1] - mean[0] and cls[2] - mean[1]."""
    rows = [accuracy_row("mean", 0, 0.50), accuracy_row("mean", 1, 0.60),
            accuracy_row("mean", 2, 0.70),
            accuracy_row("cls", 0, 0.0, status="failed", error_tail="oom"),
            accuracy_row("cls", 1, 0.65), accuracy_row("cls", 2, 0.80)]
    summary = report.build_summary([make_root(tmp_path, "r", rows)])
    paired = variant(summary, "cls")["paired"]["clean"]
    assert paired["seeds"] == [1, 2]
    assert paired["diffs"] == pytest.approx([0.05, 0.10])
    assert paired["unpaired_seeds"] == [0]
    assert variant(summary, "cls")["n_seeds"] == 2
    assert variant(summary, "mean")["n_seeds"] == 3


def test_the_reference_has_no_paired_block_and_every_metric_is_paired(two_by_two):
    summary = report.build_summary([two_by_two])
    assert variant(summary, "mean")["paired"] is None
    assert set(variant(summary, "cls")["paired"]) == {*CONDITIONS, "select_accuracy"}


def test_a_missing_reference_is_noted_not_silent(tmp_path):
    summary = report.build_summary([make_root(tmp_path, "r", [accuracy_row("cls", 0, 0.5)])])
    group = summary["groups"][0]
    assert any("reference 'mean'" in note for note in group["notes"])
    assert group["variants"][0]["paired"] is None


def test_a_single_seed_has_no_std(tmp_path):
    summary = report.build_summary([make_root(tmp_path, "r", [accuracy_row("mean", 0, 0.5)])])
    entry = variant(summary, "mean")["metrics"]["clean"]
    assert entry["n"] == 1 and entry["mean"] == 0.5 and entry["std"] is None


def test_cost_columns_are_aggregated(tmp_path):
    rows = [accuracy_row("mean", 0, 0.5, params=2000, seconds_per_epoch=4.0),
            accuracy_row("mean", 1, 0.5, params=2000, seconds_per_epoch=6.0)]
    cost = variant(report.build_summary([make_root(tmp_path, "r", rows)]), "mean")["cost"]
    assert cost["params"]["mean"] == 2000.0
    assert cost["seconds_per_epoch"]["mean"] == pytest.approx(5.0)
    assert "throughput_images_per_s" not in cost


# ---------------------------------------------------------------------------
# failed and missing cells are listed
# ---------------------------------------------------------------------------

def test_a_failed_cell_is_listed_with_its_tail(tmp_path):
    rows = [accuracy_row("mean", 0, 0.5), accuracy_row("cls", 0, 0.0, status="failed",
                                                      error_tail="x" * 400 + "MemoryError")]
    summary = report.build_summary([make_root(tmp_path, "r", rows)])
    assert summary["counts"] == {"ok": 1, "failed": 1, "missing": 0}
    (item,) = summary["failed"]
    assert item["key"] == ["exp1", "cls", "cifar10", 0]
    assert item["attempts"] == 1 and item["returncode"] == 1
    assert item["error_tail"].endswith("MemoryError") and len(item["error_tail"]) == 300
    assert "failed exp1/cls/cifar10/0" in report.render_markdown(summary)


def test_a_cell_that_failed_then_succeeded_counts_as_ok(tmp_path):
    rows = [accuracy_row("mean", 0, 0.0, status="failed", error_tail="oom"),
            accuracy_row("mean", 0, 0.5, attempt=2)]
    summary = report.build_summary([make_root(tmp_path, "r", rows)])
    assert summary["counts"] == {"ok": 1, "failed": 0, "missing": 0}
    assert summary["failed"] == []


def test_a_manifest_cell_without_a_row_is_missing(tmp_path):
    cells = [("exp1", "mean", "cifar10", 0), ("exp1", "cls", "cifar10", 0),
             ("exp1", "mean", "cifar10", 1)]
    root = make_root(tmp_path, "r", [accuracy_row("mean", 0, 0.5)], manifest_cells=cells)
    summary = report.build_summary([root])
    assert summary["missing"] == [["exp1", "cls", "cifar10", 0], ["exp1", "mean", "cifar10", 1]]
    text = report.render_markdown(summary)
    assert "missing exp1/cls/cifar10/0" in text and "missing exp1/mean/cifar10/1" in text


def test_a_root_with_a_manifest_and_no_rows_reports_everything_missing(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"cells": [
        {"experiment": "exp1", "arm": "mean", "config": "cifar10", "seed": 0, "extra_args": []}]}),
        encoding="utf-8")
    summary = report.build_summary([root])
    assert summary["counts"]["missing"] == 1 and summary["groups"] == []


def test_a_root_with_nothing_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        report.build_summary([tmp_path / "empty"])


def test_a_malformed_ok_row_is_refused_not_skipped(tmp_path):
    row = accuracy_row("mean", 0, 0.5)
    del row["metrics"]["metrics"]["select_accuracy"]
    with pytest.raises(ValueError, match="exp1/mean/cifar10/0"):
        report.build_summary([make_root(tmp_path, "r", [row])])


# ---------------------------------------------------------------------------
# exp2: ranking and bandwidths
# ---------------------------------------------------------------------------

def exp2_rows():
    """Selection order a > b > c; test order c > b > a; d ties a on selection with more steps."""
    rows = []
    for seed in (0, 1):
        rows += [
            exp2_row("mha", seed, 0.30, 0.30, attention="multi_head", params=1000),
            exp2_row("cfg_a", seed, 0.40, 0.10, steps=1),
            exp2_row("cfg_b", seed, 0.35, 0.20, steps=1),
            exp2_row("cfg_c", seed, 0.30, 0.45, steps=1),
            exp2_row("cfg_d", seed, 0.40, 0.05, steps=3),
        ]
    return rows


def test_the_ranking_uses_the_selection_split(tmp_path):
    summary = report.build_summary([make_root(tmp_path, "r", exp2_rows())])
    ranking = summary["groups"][0]["ranking"]
    assert ranking["metric"] == "select_accuracy"
    order = [r["config"] for r in ranking["rows"]]
    assert order == ["cfg_a", "cfg_d", "cfg_b", "cfg_c"]
    assert order != sorted(order, key=lambda c: -variant(summary, c, "exp2")["metrics"]["test_accuracy"]["mean"])
    assert [r["rank"] for r in ranking["rows"]] == [1, 2, 3, 4]


def test_ranking_ties_go_to_fewer_steps_then_the_config_id(tmp_path):
    rows = []
    for config, steps in (("zz", 1), ("aa", 3), ("bb", 3)):
        rows.append(exp2_row(config, 0, 0.40, 0.40, steps=steps))
    rows.append(exp2_row("mha", 0, 0.1, 0.1, attention="multi_head"))
    order = [r["config"] for r in report.build_summary(
        [make_root(tmp_path, "r", rows)])["groups"][0]["ranking"]["rows"]]
    assert order == ["zz", "aa", "bb"]


def test_the_reference_is_not_ranked_and_exp1_has_no_ranking(tmp_path):
    summary = report.build_summary([make_root(tmp_path, "r", exp2_rows())])
    assert "mha" not in [r["config"] for r in summary["groups"][0]["ranking"]["rows"]]
    only_exp1 = report.build_summary([make_root(tmp_path, "s", [accuracy_row("mean", 0, 0.5)])])
    assert "ranking" not in only_exp1["groups"][0]


def test_exp2_paired_differences_use_the_mha_reference(tmp_path):
    summary = report.build_summary([make_root(tmp_path, "r", exp2_rows())])
    paired = variant(summary, "cfg_a", "exp2")["paired"]["select_accuracy"]
    assert paired["reference"] == "mha"
    assert paired["diffs"] == pytest.approx([0.10, 0.10])
    assert variant(summary, "mha", "exp2")["arm"] == "multi_head"


def test_bandwidths_are_summarized_per_config(tmp_path):
    rows = [exp2_row("mha", 0, 0.3, 0.3, attention="multi_head"),
            exp2_row("cfg_a", 0, 0.4, 0.4, bandwidths=bandwidth_block([0.2, 0.4])),
            exp2_row("cfg_a", 1, 0.4, 0.4, bandwidths=bandwidth_block([0.4, 0.6], inside=False))]
    summary = report.build_summary([make_root(tmp_path, "r", rows)])
    band = variant(summary, "cfg_a", "exp2")["bandwidths"]
    assert band["n_seeds"] == 2 and band["mode"] == "per_head"
    first, second = band["layers"]
    assert first["mean"] == pytest.approx(0.3) and second["mean"] == pytest.approx(0.5)
    assert first["min"] == pytest.approx(0.1) and first["max"] == pytest.approx(0.5)
    assert first["per_head_mean"] == pytest.approx([0.25, 0.35])
    assert first["inside_clip_range"] is False
    assert variant(summary, "mha", "exp2")["bandwidths"]["status"].startswith("not applicable")


# ---------------------------------------------------------------------------
# strict JSON, files and the command line
# ---------------------------------------------------------------------------

def test_strict_json_has_no_nan_tokens(tmp_path, two_by_two):
    summary = report.build_summary([two_by_two])
    text = report.dumps_strict(summary)
    for token in ("NaN", "Infinity"):
        assert token not in text
    json.loads(text, parse_constant=lambda name: pytest.fail(f"non-strict token {name}"))
    with pytest.raises(ValueError):
        report.dumps_strict({"x": float("nan")})


def test_a_missing_seconds_per_epoch_becomes_null_not_nan(tmp_path):
    row = accuracy_row("mean", 0, 0.5)
    row["metrics"]["seconds_per_epoch"] = None
    summary = report.build_summary([make_root(tmp_path, "r", [row])])
    assert "seconds_per_epoch" not in variant(summary, "mean")["cost"]
    json.loads(report.dumps_strict(summary))


def test_the_cli_writes_both_files_and_refuses_to_overwrite(tmp_path, two_by_two):
    exp2_root = make_root(tmp_path, "exp2", exp2_rows())
    out = tmp_path / "out"
    code = report.main(["--root", str(two_by_two), "--root", str(exp2_root), "--out", str(out)])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["ok"] == 14
    assert (out / "summary.md").read_text(encoding="utf-8").startswith("# Mean shift report")
    before = (out / "summary.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        report.main(["--root", str(two_by_two), "--out", str(out)])
    assert (out / "summary.json").read_text(encoding="utf-8") == before


def test_help_prints_usage():
    done = subprocess.run([sys.executable, "-m", "train.mean_shift.report", "--help"],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0 and "usage:" in done.stdout


def test_the_report_imports_and_runs_without_tensorflow(tmp_path, two_by_two):
    code = ("import sys, train.mean_shift.report as r; "
            f"s = r.build_summary([{str(two_by_two)!r}]); r.render_markdown(s); "
            "raise SystemExit(int('tensorflow' in sys.modules or 'keras' in sys.modules))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr[-800:]


def test_the_markdown_is_ascii(two_by_two, tmp_path):
    text = report.render_markdown(report.build_summary([two_by_two]))
    text += report.render_markdown(
        report.build_summary([make_root(tmp_path, "e2", exp2_rows())]))
    assert text.isascii()


# ---------------------------------------------------------------------------
# the markdown, golden
# ---------------------------------------------------------------------------

GOLDEN = """\
# Mean shift report

Roots:
- {root}

Cells: 4 ok, 0 failed, 0 missing.

Std convention: sample std, ddof=1 (train.common.stats.mean_std); null below two seeds.

Claims: {claims}

## exp1 on cifar10

Reference variant: `mean`. Epochs: [2].

Accuracy, mean +- std over seeds:

| variant | n_seeds | params | s/epoch | clean | occ25 | occ50 | noise0.05 | noise0.1 | noise0.2 | select_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 2 | 1000 | 2.00 +- 0.00 | 0.5500 +- 0.0707 | 0.4500 +- 0.0707 | 0.3500 +- 0.0707 | 0.2500 +- 0.0707 | 0.1500 +- 0.0707 | 0.0500 +- 0.0707 | 0.5500 +- 0.0707 |
| cls | 2 | 1000 | 2.00 +- 0.00 | 0.6250 +- 0.1061 | 0.5250 +- 0.1061 | 0.4250 +- 0.1061 | 0.3250 +- 0.1061 | 0.2250 +- 0.1061 | 0.1250 +- 0.1061 | 0.6250 +- 0.1061 |

Paired difference against `mean` (variant minus reference, seed by seed):

| variant | metric | n_pairs | seeds | mean diff | per-seed diffs |
|---|---|---|---|---|---|
{paired}

## Failed and missing cells

None.
"""


def test_the_markdown_matches_the_golden_text(two_by_two):
    summary = report.build_summary([two_by_two])
    paired = "\n".join(
        f"| cls | {metric} | 2 | 0,1 | +0.0750 | +0.0500, +0.1000 |"
        for metric in (*CONDITIONS, "select_accuracy"))
    expected = GOLDEN.format(root=two_by_two, claims=report.CLAIMS_NOTE, paired=paired)
    assert report.render_markdown(summary) == expected


def test_a_group_with_only_the_reference_says_so(tmp_path):
    rows = [exp2_row("mha", 0, 0.3, 0.3, attention="multi_head")]
    text = report.render_markdown(report.build_summary([make_root(tmp_path, "r", rows)]))
    assert "no variant besides the reference has an ok cell" in text
    assert "No variant besides the reference to rank." in text
