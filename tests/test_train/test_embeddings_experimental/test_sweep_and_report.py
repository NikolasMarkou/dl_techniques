"""Guards on the sweep grid and the report's statistics.

The sweep is cheap to get subtly wrong in ways that only show up after hours of
GPU time, so the grid, the budget refusal and the environment hardening are
asserted directly rather than by running cells.
"""

import json
import os

import numpy as np
import pytest

from train.embeddings_experimental.config import BASELINE_MODEL
from train.embeddings_experimental.metric_directions import direction_of
from train.embeddings_experimental.report import (
    HEADLINE_METRICS,
    RNG_SEED,
    build_report,
    write_report,
)
from train.embeddings_experimental.sweep import (
    DEFAULT_MAX_CELLS,
    RunSpec,
    build_run_specs,
    collect_results,
    parse_args,
)


class TestGrid:
    """The Cartesian product and its budget."""

    def test_cell_count_is_the_product_of_the_axes(self, tmp_path):
        specs = build_run_specs(
            models=["ascii_bert", "ascii_clifford_bert"],
            variants=["tiny", "small"],
            poolings=["cls", "mean"],
            seeds=[0, 1, 2],
            sweep_root=str(tmp_path),
        )
        assert len(specs) == 2 * 2 * 2 * 3

    def test_every_cell_id_is_unique(self, tmp_path):
        specs = build_run_specs(
            models=["ascii_bert", "ascii_clifford_bert"],
            variants=["tiny", "small"],
            poolings=["cls", "mean"],
            seeds=[0, 1],
            sweep_root=str(tmp_path),
        )
        assert len({s.cell_id for s in specs}) == len(specs)

    def test_the_budget_is_checked_before_anything_launches(self, tmp_path):
        """Discovering an oversized grid at cell 300 wastes hours."""
        with pytest.raises(ValueError, match="cell budget"):
            build_run_specs(
                models=["ascii_bert", "ascii_clifford_bert"],
                variants=["tiny", "small", "base"],
                poolings=["cls", "mean", "attention"],
                seeds=list(range(50)),
                sweep_root=str(tmp_path),
            )

    def test_an_unknown_model_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown models"):
            build_run_specs(
                models=["ascii_bert", "telepathy_bert"],
                variants=["tiny"], poolings=["mean"], seeds=[0],
                sweep_root=str(tmp_path),
            )

    @pytest.mark.parametrize(
        "axis", ["models", "variants", "poolings", "seeds"]
    )
    def test_an_empty_axis_is_refused(self, axis, tmp_path):
        kwargs = dict(
            models=["ascii_bert"], variants=["tiny"],
            poolings=["mean"], seeds=[0], sweep_root=str(tmp_path),
        )
        kwargs[axis] = []
        with pytest.raises(ValueError, match="is empty"):
            build_run_specs(**kwargs)

    def test_the_spec_is_frozen(self, tmp_path):
        spec = build_run_specs(
            models=["ascii_bert"], variants=["tiny"], poolings=["mean"],
            seeds=[0], sweep_root=str(tmp_path),
        )[0]
        with pytest.raises(Exception):
            spec.model = "something_else"


class TestCommand:
    """The argv each cell runs."""

    def test_the_command_carries_every_axis(self, tmp_path):
        spec = RunSpec(
            model="ascii_clifford_bert", variant="small", pooling="cls",
            seed=4, sweep_root=str(tmp_path),
        )
        command = spec.command("python")
        joined = " ".join(command)
        assert "--model ascii_clifford_bert" in joined
        assert "--variant small" in joined
        assert "--pooling-strategy cls" in joined
        assert "--seed 4" in joined
        assert "--experiment-name ascii_clifford_bert/small/cls/seed_4" in joined

    def test_extra_trainer_args_are_appended(self, tmp_path):
        spec = RunSpec(
            model="ascii_bert", variant="tiny", pooling="mean", seed=0,
            sweep_root=str(tmp_path), extra_args=("--mlm-epochs", "3"),
        )
        assert spec.command("python")[-2:] == ["--mlm-epochs", "3"]

    def test_the_cell_dir_is_under_the_sweep_root(self, tmp_path):
        spec = RunSpec(
            model="ascii_bert", variant="tiny", pooling="mean", seed=0,
            sweep_root=str(tmp_path),
        )
        assert spec.cell_dir.startswith(str(tmp_path))

    def test_the_gpu_and_backend_are_hard_set_not_defaulted(self):
        """`setdefault` would inherit the parent shell and land on a stray GPU."""
        import inspect

        from train.embeddings_experimental import sweep

        source = inspect.getsource(sweep.run_one)
        assert 'env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)' in source
        assert 'env["MPLBACKEND"] = "Agg"' in source
        # Match the CALL, not the word: the function's own comment explains
        # why setdefault is wrong, and a bare substring test matches that.
        assert "env.setdefault(" not in source


class TestCollectResults:
    """Reading cells back off disk."""

    def _write_cell(self, root, model, variant, pooling, seed, val_loss):
        cell = os.path.join(root, model, variant, pooling, f"seed_{seed}")
        os.makedirs(cell, exist_ok=True)
        payload = {
            "config": {
                "model": model, "variant": variant,
                "pooling_strategy": pooling, "seed": seed,
            },
            "parameters": 1234,
            "mlm": {"val_loss": [5.0, val_loss], "val_accuracy": [0.1, 0.2]},
            "contrastive": {"val_loss": [1.0, val_loss / 2]},
        }
        with open(os.path.join(cell, "results.json"), "w") as handle:
            json.dump(payload, handle)

    def test_records_carry_the_axes_and_final_metrics(self, tmp_path):
        self._write_cell(str(tmp_path), "ascii_bert", "tiny", "mean", 0, 4.0)
        records = collect_results(str(tmp_path))
        assert len(records) == 1
        record = records[0]
        assert record["model"] == "ascii_bert"
        assert record["seed"] == 0
        assert record["mlm_val_loss_final"] == 4.0
        assert record["mlm_val_loss_best"] == 4.0
        assert record["contrastive_val_loss_final"] == 2.0
        assert record["contrastive_val_loss_best"] == 1.0

    def test_an_unreadable_cell_is_skipped_not_fatal(self, tmp_path):
        self._write_cell(str(tmp_path), "ascii_bert", "tiny", "mean", 0, 4.0)
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "results.json").write_text("{not json")
        assert len(collect_results(str(tmp_path))) == 1


class TestReport:
    """The statistics, on fabricated records with a known answer."""

    #: The PRIMARY endpoint; only this role yields a BETTER/WORSE verdict.
    PRIMARY = "eval_squad_mrr_at_10"

    def _records(self, baseline_values, arm_values, metric=None):
        """Fabricate paired cells. Higher is better for the primary metric."""
        metric = metric or self.PRIMARY
        records = []
        for seed, value in enumerate(baseline_values):
            records.append({
                "model": BASELINE_MODEL, "variant": "tiny", "pooling": "mean",
                "seed": seed, "parameters": 100, metric: value,
            })
        for seed, value in enumerate(arm_values):
            records.append({
                "model": "ascii_clifford_bert", "variant": "tiny",
                "pooling": "mean", "seed": seed, "parameters": 90,
                metric: value,
            })
        return records

    def test_a_clearly_better_arm_is_reported_better(self):
        # MRR@10 is a maximize metric, so the better arm scores HIGHER.
        report = build_report(
            self._records([0.10, 0.11, 0.12, 0.105, 0.115, 0.108],
                          [0.30, 0.31, 0.32, 0.305, 0.315, 0.308])
        )
        rows = [r for r in report["paired"] if r["metric"] == self.PRIMARY]
        assert len(rows) == 1
        assert rows[0]["verdict"] == "BETTER"
        assert rows[0]["diff_vs_baseline"] > 0
        assert rows[0]["correction"] == "holm"

    def test_a_clearly_worse_arm_is_reported_worse(self):
        report = build_report(
            self._records([0.30, 0.31, 0.32, 0.305, 0.315, 0.308],
                          [0.10, 0.11, 0.12, 0.105, 0.115, 0.108])
        )
        rows = [r for r in report["paired"] if r["metric"] == self.PRIMARY]
        assert rows[0]["verdict"] == "WORSE"

    def test_identical_arms_are_indistinguishable(self):
        values = [0.10, 0.11, 0.12, 0.105, 0.115, 0.108]
        report = build_report(self._records(values, list(values)))
        rows = [r for r in report["paired"] if r["metric"] == self.PRIMARY]
        assert rows[0]["verdict"] == "INDISTINGUISHABLE"

    def test_a_secondary_metric_never_gets_a_better_or_worse_verdict(self):
        """Only the pre-registered primary endpoint yields a verdict."""
        report = build_report(
            self._records([4.0, 4.1, 4.2, 4.05, 4.15, 4.08],
                          [1.0, 1.1, 1.2, 1.05, 1.15, 1.08],
                          metric="mlm_val_loss_best")
        )
        rows = [r for r in report["paired"]
                if r["metric"] == "mlm_val_loss_best"]
        assert rows and all(r["role"] == "secondary" for r in rows)
        assert all(r["verdict"] not in {"BETTER", "WORSE"} for r in rows)
        assert all(r["correction"] == "bh" for r in rows)

    def test_a_diagnostic_metric_gets_no_paired_row_at_all(self):
        """A random projection wins on anisotropy while retrieving nothing."""
        report = build_report(
            self._records([0.1] * 8, [0.9] * 8,
                          metric="eval_squad_ctx_anisotropy")
        )
        assert not [r for r in report["paired"]
                    if r["metric"] == "eval_squad_ctx_anisotropy"]

    def test_the_adjusted_p_is_never_smaller_than_the_raw_p(self):
        report = build_report(
            self._records([0.10, 0.11, 0.12, 0.105, 0.115, 0.108],
                          [0.30, 0.31, 0.32, 0.305, 0.315, 0.308])
        )
        for row in report["paired"]:
            assert row["p_adjusted"] >= row["p_value"] - 1e-12

    def test_direction_is_honoured_for_a_maximize_metric(self):
        """Higher accuracy is better; the sign convention must flip."""
        assert HEADLINE_METRICS["mlm_val_accuracy_best"][1] == "max"
        assert HEADLINE_METRICS["mlm_val_loss_best"][1] == "min"
        # direction stays at index 1 so positional access keeps working
        assert HEADLINE_METRICS["mlm_val_loss_best"].direction == "min"

    def test_arms_are_only_compared_within_a_matched_group(self):
        """Comparing across pooling strategies would confound the axes."""
        records = self._records([4.0, 4.1, 4.2], [3.0, 3.1, 3.2])
        for record in records:
            if record["model"] != BASELINE_MODEL:
                record["pooling"] = "cls"
        report = build_report(records)
        assert report["paired"] == []

    def test_high_variance_is_flagged(self):
        report = build_report(
            self._records([0.001, 10.0, -9.0, 0.5, 0.2, 0.3],
                          [0.002, 11.0, -9.5, 0.4, 0.1, 0.2])
        )
        assert report["flags"]

    def test_duplicate_cells_are_deduped(self):
        records = self._records([4.0, 4.1, 4.2], [3.0, 3.1, 3.2])
        report_once = build_report(records)
        report_twice = build_report(records + records)
        assert (
            report_once["headline"][0]["n"] == report_twice["headline"][0]["n"]
        )

    def test_the_report_is_reproducible(self):
        records = self._records([4.0, 4.1, 4.2, 4.05], [3.9, 4.2, 4.1, 4.0])
        first = build_report(records)["paired"]
        second = build_report(records)["paired"]
        assert [r["p_value"] for r in first] == [r["p_value"] for r in second]
        assert isinstance(RNG_SEED, int)

    def test_write_report_emits_the_expected_artifacts(self, tmp_path):
        report = build_report(
            self._records([4.0, 4.1, 4.2], [3.0, 3.1, 3.2])
        )
        write_report(report, str(tmp_path))
        for name in ("summary.md", "headline_summary.csv", "paired_summary.csv"):
            assert (tmp_path / name).exists(), name
        text = (tmp_path / "summary.md").read_text()
        assert BASELINE_MODEL in text
        assert "parameters" in text


class TestTheStudyIsPowered:
    """Five seeds or fewer cannot reach significance, for ANY effect size.

    ``paired_permutation_test`` is a two-sided sign-flip test, so with ``n``
    pairs the smallest reachable p-value is about ``2/2**n``. MEASURED against
    maximally separated arms: n=3 -> 0.248, n=4 -> 0.125, n=5 -> 0.063,
    n=6 -> 0.031. A sweep run with three seeds would report every comparison
    as "no significant difference", which reads like a finding and is not one.
    """

    def _extreme_records(self, n_seeds):
        records = []
        for seed in range(n_seeds):
            records.append({
                "model": BASELINE_MODEL, "variant": "tiny", "pooling": "mean",
                "seed": seed, "parameters": 100,
                "eval_squad_mrr_at_10": 0.10 + seed * 0.001,
            })
            records.append({
                "model": "ascii_clifford_bert", "variant": "tiny",
                "pooling": "mean", "seed": seed, "parameters": 90,
                "eval_squad_mrr_at_10": 0.90 + seed * 0.001,
            })
        return records

    @pytest.mark.parametrize("n_seeds", [2, 3, 4, 5])
    def test_too_few_seeds_is_reported_as_underpowered_not_as_no_difference(
        self, n_seeds
    ):
        report = build_report(self._extreme_records(n_seeds))
        rows = [r for r in report["paired"]
                if r["metric"] == "eval_squad_mrr_at_10"]
        assert rows, n_seeds
        assert rows[0]["verdict"] == "UNDERPOWERED", (
            f"at {n_seeds} seeds the test cannot reach p<0.05, so a verdict of "
            f"{rows[0]['verdict']!r} would overstate what the study knows"
        )

    def test_six_seeds_can_reach_significance(self):
        report = build_report(self._extreme_records(6))
        rows = [r for r in report["paired"]
                if r["metric"] == "eval_squad_mrr_at_10"]
        assert rows[0]["verdict"] == "BETTER"
        assert rows[0]["p_value"] < 0.05

    def test_the_sweep_default_seed_count_clears_the_CORRECTED_floor(self):
        """The floor that matters is the corrected one, not the m=1 one.

        The primary endpoint is Holm-corrected across the non-baseline arms,
        so the default must clear `min_pairs_for_significance(n_arms - 1)`,
        which is 7 at four arms -- not the uncorrected 6.
        """
        from train.common.stats import min_pairs_for_significance
        from train.embeddings_experimental.config import available_models

        n_non_baseline = len(available_models()) - 1
        floor = min_pairs_for_significance(n_non_baseline)
        assert len(parse_args([]).seeds) >= floor, (
            f"a {n_non_baseline}-arm primary family needs {floor} seeds; the "
            "default sweep would be incapable of reporting any difference"
        )


class TestSweepCli:
    def test_defaults_cover_both_arms(self):
        args = parse_args([])
        assert set(args.models) >= {"ascii_bert", "ascii_clifford_bert"}
        assert args.max_cells == DEFAULT_MAX_CELLS

    def test_dry_run_is_a_flag(self):
        assert parse_args(["--dry-run"]).dry_run is True


class TestTheBestReductionHonoursDirection:
    """`collect_results` reduced EVERY metric with `min`.

    That made `mlm_val_accuracy_best` the WORST accuracy, while
    `HEADLINE_METRICS` described the same key as a maximize metric. Silent, and
    it inverted one of the three metrics the study reported.
    """

    def _cell(self, root, history):
        cell = os.path.join(root, "ascii_bert", "tiny", "mean", "seed_0")
        os.makedirs(cell, exist_ok=True)
        payload = {
            "config": {
                "model": "ascii_bert", "variant": "tiny",
                "pooling_strategy": "mean", "seed": 0,
            },
            "parameters": 1,
            "mlm": history,
        }
        with open(os.path.join(cell, "results.json"), "w") as handle:
            json.dump(payload, handle)
        return collect_results(root)[0]

    def test_a_maximize_metric_takes_the_max(self, tmp_path):
        record = self._cell(str(tmp_path), {"val_accuracy": [0.1, 0.9, 0.4]})
        assert record["mlm_val_accuracy_best"] == pytest.approx(0.9)
        assert record["mlm_val_accuracy_final"] == pytest.approx(0.4)

    def test_a_minimize_metric_takes_the_min(self, tmp_path):
        record = self._cell(str(tmp_path), {"val_loss": [5.0, 1.0, 3.0]})
        assert record["mlm_val_loss_best"] == pytest.approx(1.0)
        assert record["mlm_val_loss_final"] == pytest.approx(3.0)

    def test_an_unregistered_metric_gets_no_best_rather_than_a_guess(
        self, tmp_path
    ):
        """Silence is how the original bug survived; refuse to guess."""
        record = self._cell(str(tmp_path), {"val_mystery": [5.0, 1.0]})
        assert "mlm_val_mystery_final" in record
        assert "mlm_val_mystery_best" not in record

    def test_direction_has_exactly_one_producer(self):
        """`report` and `sweep` must not be able to disagree."""
        for key, spec in HEADLINE_METRICS.items():
            for stage in ("mlm_", "contrastive_"):
                if key.startswith(stage) and key.endswith("_best"):
                    bare = key[len(stage):-len("_best")]
                    assert direction_of(bare) == spec.direction, key


class TestCorrectionRaisesTheSeedFloor:
    """A correction is not free; it is paid for in seeds.

    A sign-flip test over n pairs cannot produce a p below 2/2**n, so a family
    of size m needs n >= 1 + log2(m/alpha). With three non-baseline arms in the
    primary family that is 7 seeds, not 6.
    """

    def _records(self, n_seeds, n_arms):
        records = []
        for seed in range(n_seeds):
            records.append({
                "model": BASELINE_MODEL, "variant": "tiny", "pooling": "mean",
                "seed": seed, "parameters": 100,
                "eval_squad_mrr_at_10": 0.10 + seed * 0.001,
            })
            for arm in range(n_arms):
                records.append({
                    "model": f"arm_{arm}", "variant": "tiny", "pooling": "mean",
                    "seed": seed, "parameters": 90,
                    "eval_squad_mrr_at_10": 0.90 + seed * 0.001,
                })
        return records

    def test_family_size_matches_the_number_of_non_baseline_arms(self):
        report = build_report(self._records(8, 3))
        rows = [r for r in report["paired"] if r["role"] == "primary"]
        assert rows and all(r["family_size"] == 3 for r in rows)
        assert all(r["min_seeds_for_family"] == 7 for r in rows)

    def test_six_seeds_is_underpowered_for_a_three_arm_family(self):
        """Uncorrected 6 seeds would reject; corrected it cannot."""
        report = build_report(self._records(6, 3))
        rows = [r for r in report["paired"] if r["role"] == "primary"]
        assert rows
        assert all(r["verdict"] == "UNDERPOWERED" for r in rows), (
            "at 6 seeds a Holm-corrected 3-arm family cannot reject for any "
            "effect size; anything else overstates the study"
        )

    def test_seven_seeds_clears_the_floor(self):
        report = build_report(self._records(7, 3))
        rows = [r for r in report["paired"] if r["role"] == "primary"]
        assert rows and all(r["verdict"] == "BETTER" for r in rows)
