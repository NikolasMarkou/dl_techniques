"""Unit tests for the VARIANT_HYPOTHESES registry and ``evaluate_hypothesis``.

Covers:
- Registry completeness: every variant in ``NORM_VARIANTS`` has a spec.
- All four verdicts (CONFIRMED, REJECTED, INCONCLUSIVE, N/A) reachable on
  synthetic dataframes.
- Per-variant evaluation: each of the 8 variants gets a smoke verdict on a
  synthetic single-cell frame containing its referenced ``metric_column``.
- Edge cases: missing column → INCONCLUSIVE; insufficient samples →
  INCONCLUSIVE; applicability gates (experiment, mode) → N/A.
- ``evaluate_all``: groupwise enumeration emits one row per cell with the
  expected columns.

Plan: ``plans/plan_2026-05-18_e1f12eab/plan.md`` Step 1 verification (SC1/SC2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train.rms_variants_train.config import NORM_VARIANTS
from train.rms_variants_train.hypotheses import (
    HypothesisSpec,
    VARIANT_HYPOTHESES,
    Verdict,
    evaluate_all,
    evaluate_hypothesis,
)


# ---------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------

class TestRegistryCompleteness:
    def test_registry_covers_all_norm_variants(self):
        missing = [v for v in NORM_VARIANTS if v not in VARIANT_HYPOTHESES]
        assert missing == [], (
            f"VARIANT_HYPOTHESES missing entries for NORM_VARIANTS: {missing}. "
            f"Every variant must have a hypothesis spec (or N/A via empty "
            f"applicable_experiments)."
        )

    def test_registry_has_exactly_eight_entries(self):
        assert len(VARIANT_HYPOTHESES) == 8

    @pytest.mark.parametrize("variant", list(NORM_VARIANTS))
    def test_each_variant_has_well_formed_spec(self, variant):
        spec = VARIANT_HYPOTHESES[variant]
        assert isinstance(spec, HypothesisSpec)
        assert spec.claim and isinstance(spec.claim, str)
        assert spec.metric_column and isinstance(spec.metric_column, str)
        assert spec.comparator in ("<=", "<", ">=", ">")
        assert np.isfinite(spec.threshold)
        assert spec.reduction in ("mean", "median", "max", "min")
        assert isinstance(spec.applicable_experiments, tuple)
        assert isinstance(spec.applicable_modes, tuple)
        assert spec.min_samples >= 1


# ---------------------------------------------------------------------
# Verdict paths — all four reachable
# ---------------------------------------------------------------------

class TestVerdictPaths:
    def test_confirmed_path_rms_norm_grad_norm_within_bound(self):
        # rms_norm hypothesis: grad_norm_global <= 1e2
        df = pd.DataFrame({"grad_norm_global": [0.5, 0.6, 0.7, 0.8]})
        v = evaluate_hypothesis("rms_norm", df, experiment="e1")
        assert v == "CONFIRMED"

    def test_rejected_path_rms_norm_grad_norm_explodes(self):
        df = pd.DataFrame({"grad_norm_global": [500.0, 600.0, 700.0]})
        v = evaluate_hypothesis("rms_norm", df, experiment="e1")
        assert v == "REJECTED"

    def test_inconclusive_path_missing_column(self):
        df = pd.DataFrame({"some_other_metric": [1.0, 2.0, 3.0]})
        v = evaluate_hypothesis("rms_norm", df, experiment="e1")
        assert v == "INCONCLUSIVE"

    def test_inconclusive_path_insufficient_samples(self):
        df = pd.DataFrame({"grad_norm_global": [0.5]})  # only 1 sample, min=2
        v = evaluate_hypothesis("rms_norm", df, experiment="e1")
        assert v == "INCONCLUSIVE"

    def test_inconclusive_path_all_nan_values(self):
        df = pd.DataFrame({"grad_norm_global": [np.nan, np.nan, np.nan]})
        v = evaluate_hypothesis("rms_norm", df, experiment="e1")
        assert v == "INCONCLUSIVE"

    def test_na_path_off_label_experiment(self):
        # band_logit_norm hypothesis is applicable only to (e1, e3).
        df = pd.DataFrame({"generalization_gap": [0.05, 0.06, 0.07]})
        v = evaluate_hypothesis("band_logit_norm", df, experiment="e2")
        assert v == "N/A"

    def test_na_path_unknown_variant(self):
        df = pd.DataFrame({"grad_norm_global": [0.5, 0.6]})
        v = evaluate_hypothesis("nonexistent_variant", df, experiment="e1")
        assert v == "N/A"


# ---------------------------------------------------------------------
# Per-variant smoke evaluation
# ---------------------------------------------------------------------

class TestPerVariantSmoke:
    """For each of the 8 variants, evaluate the hypothesis on a synthetic
    cell where the registered metric column is set to a value that satisfies
    the claim — expected verdict CONFIRMED (or N/A if the variant has empty
    applicable_experiments)."""

    @pytest.mark.parametrize("variant", list(NORM_VARIANTS))
    def test_variant_confirmed_on_satisfying_data(self, variant):
        spec = VARIANT_HYPOTHESES[variant]
        if not spec.applicable_experiments:
            pytest.skip(f"variant {variant} has no applicable experiments")
        # Pick a value that satisfies the comparator.
        if spec.comparator in ("<=", "<"):
            satisfying_value = spec.threshold * 0.5  # well below
        else:
            satisfying_value = spec.threshold * 2.0  # well above
        df = pd.DataFrame({
            spec.metric_column: [satisfying_value] * max(spec.min_samples, 3),
        })
        experiment = spec.applicable_experiments[0]
        v = evaluate_hypothesis(variant, df, experiment=experiment)
        assert v == "CONFIRMED", (
            f"Expected CONFIRMED for {variant} on satisfying data; got {v}. "
            f"Spec: column={spec.metric_column}, comparator={spec.comparator}, "
            f"threshold={spec.threshold}, value={satisfying_value}"
        )

    @pytest.mark.parametrize("variant", list(NORM_VARIANTS))
    def test_variant_rejected_on_violating_data(self, variant):
        spec = VARIANT_HYPOTHESES[variant]
        if not spec.applicable_experiments:
            pytest.skip(f"variant {variant} has no applicable experiments")
        # Pick a value that violates the comparator.
        if spec.comparator in ("<=", "<"):
            violating_value = spec.threshold * 10.0  # well above
        else:
            violating_value = spec.threshold * 0.1  # well below
        df = pd.DataFrame({
            spec.metric_column: [violating_value] * max(spec.min_samples, 3),
        })
        experiment = spec.applicable_experiments[0]
        v = evaluate_hypothesis(variant, df, experiment=experiment)
        assert v == "REJECTED", (
            f"Expected REJECTED for {variant} on violating data; got {v}."
        )


# ---------------------------------------------------------------------
# Comparator semantics
# ---------------------------------------------------------------------

class TestComparatorSemantics:
    def test_le_boundary_is_confirmed(self):
        # rms_norm threshold = 1e2; value exactly 1e2 should be CONFIRMED ("<=")
        df = pd.DataFrame({"grad_norm_global": [1.0e2, 1.0e2]})
        v = evaluate_hypothesis("rms_norm", df, experiment="e1")
        assert v == "CONFIRMED"

    def test_le_boundary_plus_epsilon_is_rejected(self):
        df = pd.DataFrame({"grad_norm_global": [1.0e2 + 1.0, 1.0e2 + 1.0]})
        v = evaluate_hypothesis("rms_norm", df, experiment="e1")
        assert v == "REJECTED"


# ---------------------------------------------------------------------
# Applicability gates
# ---------------------------------------------------------------------

class TestApplicabilityGates:
    def test_experiment_gate_blocks_unapplicable(self):
        # band_rms applies to (e1, e2, e3, e5), not e4.
        df = pd.DataFrame({
            "act_per_sample_rms_max_max": [0.99, 1.00, 1.01],
        })
        v = evaluate_hypothesis("band_rms", df, experiment="e4")
        assert v == "N/A"

    def test_experiment_gate_allows_applicable(self):
        df = pd.DataFrame({
            "act_per_sample_rms_max_max": [0.99, 1.00, 1.01],
        })
        v = evaluate_hypothesis("band_rms", df, experiment="e1")
        assert v == "CONFIRMED"

    def test_no_experiment_filter_when_unspecified(self):
        # If experiment is None, applicable_experiments gate is skipped.
        df = pd.DataFrame({
            "act_per_sample_rms_max_max": [0.99, 1.00, 1.01],
        })
        v = evaluate_hypothesis("band_rms", df, experiment=None)
        assert v == "CONFIRMED"


# ---------------------------------------------------------------------
# evaluate_all — groupwise enumeration
# ---------------------------------------------------------------------

class TestEvaluateAll:
    def test_empty_df_returns_empty_frame_with_expected_columns(self):
        out = evaluate_all(pd.DataFrame())
        assert out.empty
        for col in (
            "experiment", "norm_type", "mode",
            "hypothesis_verdict", "hypothesis_metric",
            "hypothesis_threshold", "hypothesis_observed",
        ):
            assert col in out.columns

    def test_groupwise_emits_one_row_per_cell(self):
        # 2 experiments × 2 variants × 1 mode = 4 cells.
        rows = []
        for exp in ("e1", "e3"):
            for variant in ("rms_norm", "band_rms"):
                for seed in range(3):
                    rows.append({
                        "experiment": exp,
                        "norm_type": variant,
                        "mode": "oob",
                        "seed": seed,
                        "grad_norm_global": 0.5 + seed * 0.1,
                        "act_per_sample_rms_max_max": 0.95 + seed * 0.02,
                    })
        df = pd.DataFrame(rows)
        out = evaluate_all(df)
        assert len(out) == 4
        # All cells should evaluate to CONFIRMED on satisfying data.
        assert (out["hypothesis_verdict"] == "CONFIRMED").all(), (
            out[["experiment", "norm_type", "hypothesis_verdict"]].to_string()
        )

    def test_evaluate_all_mixed_verdicts(self):
        rows = []
        # rms_norm: grad bounded (CONFIRMED)
        for seed in range(3):
            rows.append({
                "experiment": "e1", "norm_type": "rms_norm", "mode": "oob",
                "seed": seed, "grad_norm_global": 0.5,
                "act_per_sample_rms_max_max": np.nan,
            })
        # band_rms: per-sample RMS blown out (REJECTED)
        for seed in range(3):
            rows.append({
                "experiment": "e1", "norm_type": "band_rms", "mode": "oob",
                "seed": seed, "grad_norm_global": np.nan,
                "act_per_sample_rms_max_max": 2.0,
            })
        out = evaluate_all(pd.DataFrame(rows))
        verdicts = dict(zip(out["norm_type"], out["hypothesis_verdict"]))
        assert verdicts["rms_norm"] == "CONFIRMED"
        assert verdicts["band_rms"] == "REJECTED"


# ---------------------------------------------------------------------
# Threshold provenance — claims grounded in design, not data
# ---------------------------------------------------------------------

class TestThresholdProvenance:
    """Smoke checks to keep hypothesis thresholds honest. If these need
    relaxing to match observed data, that's a Scenario-A pre-mortem
    violation — revisit the claim, don't move the threshold."""

    def test_zero_centered_threshold_is_tight(self):
        # Zero-centering's whole point is mean ~= 0; threshold > 0.5 would be
        # vacuous.
        assert VARIANT_HYPOTHESES["zero_centered_rms_norm"].threshold <= 0.5

    def test_band_threshold_near_unity(self):
        # Band's upper bound is 1.0 exactly; threshold > 1.5 would be vacuous.
        for variant in (
            "band_rms",
            "zero_centered_band_rms_norm",
            "adaptive_band_rms",
            "zero_centered_adaptive_band_rms_norm",
        ):
            spec = VARIANT_HYPOTHESES[variant]
            assert spec.threshold <= 1.5, (
                f"{variant} band threshold {spec.threshold} is too loose; "
                f"the layer's design claim is upper bound = 1.0."
            )

    def test_grad_norm_threshold_is_finite_and_positive(self):
        for variant in ("rms_norm", "dynamic_tanh"):
            spec = VARIANT_HYPOTHESES[variant]
            assert spec.threshold > 0 and np.isfinite(spec.threshold)


# ---------------------------------------------------------------------
# Step 2 wiring — report.py integration (synthetic fixture)
# ---------------------------------------------------------------------

class TestReportWiring:
    """End-to-end test: build a synthetic all_runs.csv frame, call
    ``report.write_report``, and assert the hypothesis_verdict column lands
    on ``headline_summary.csv`` and the hypothesis block lands in
    ``summary.md``.

    Note: write_report walks per-cell directories for probe CSVs (which
    don't exist in this synthetic test); the probe-snapshot frame will be
    empty and hypotheses that depend on probe columns will yield
    INCONCLUSIVE. The headline-driven hypotheses (band_logit_norm uses
    generalization_gap which IS on the synthetic frame) yield real verdicts.
    """

    def test_write_report_emits_hypothesis_verdict_column(self, tmp_path):
        from train.rms_variants_train.report import write_report

        # Build a minimal synthetic frame mimicking the merged all_runs.csv.
        rows = []
        for exp in ("e1", "e3"):
            for variant in ("rms_norm", "band_logit_norm"):
                for seed in range(3):
                    rows.append({
                        "experiment": exp,
                        "norm_type": variant,
                        "mode": "oob",
                        "seed": seed,
                        "best_val_acc": 0.85 + seed * 0.01,
                        "final_val_loss": 0.5 - seed * 0.01,
                        "generalization_gap": 0.05 + seed * 0.01,
                    })
        df = pd.DataFrame(rows)

        write_report(df, out_dir=str(tmp_path))

        # 1. headline_summary.csv has the hypothesis_verdict column.
        headline_path = tmp_path / "headline_summary.csv"
        assert headline_path.exists(), "headline_summary.csv not written"
        headline = pd.read_csv(headline_path)
        assert "hypothesis_verdict" in headline.columns, (
            f"hypothesis_verdict missing from headline_summary.csv "
            f"(columns={list(headline.columns)})"
        )

        # 2. hypothesis_verdicts.csv is emitted.
        hyp_path = tmp_path / "hypothesis_verdicts.csv"
        assert hyp_path.exists(), "hypothesis_verdicts.csv not written"
        hyp = pd.read_csv(hyp_path)
        assert "hypothesis_verdict" in hyp.columns
        # band_logit_norm hypothesis uses generalization_gap (which IS in the
        # synthetic frame) — so it should give a real verdict (not
        # INCONCLUSIVE). Gap values are small (0.05-0.07) < threshold 0.20
        # so CONFIRMED.
        blog = hyp[hyp["norm_type"] == "band_logit_norm"]
        assert not blog.empty
        assert (blog["hypothesis_verdict"] == "CONFIRMED").all(), (
            f"Expected band_logit_norm CONFIRMED on small gap; got "
            f"{blog['hypothesis_verdict'].tolist()}"
        )

        # 3. summary.md contains the hypothesis verdict section header.
        summary_path = tmp_path / "summary.md"
        assert summary_path.exists()
        content = summary_path.read_text()
        assert "Hypothesis verdicts" in content, (
            "summary.md missing 'Hypothesis verdicts' section header"
        )
        assert "VARIANT_HYPOTHESES" in content, (
            "summary.md missing reference to VARIANT_HYPOTHESES registry"
        )

    def test_write_report_preserves_existing_verdict_column(self, tmp_path):
        """I4 invariant: the existing PASS/FAIL/INDISTINGUISHABLE verdict
        column remains intact (we're additive, not replacing)."""
        from train.rms_variants_train.report import write_report

        rows = []
        for variant in ("rms_norm", "band_logit_norm"):
            for seed in range(3):
                rows.append({
                    "experiment": "e1",
                    "norm_type": variant,
                    "mode": "oob",
                    "seed": seed,
                    "best_val_acc": 0.85,
                    "generalization_gap": 0.05,
                })
        df = pd.DataFrame(rows)
        write_report(df, out_dir=str(tmp_path))

        headline = pd.read_csv(tmp_path / "headline_summary.csv")
        assert "verdict" in headline.columns, (
            "Existing 'verdict' column dropped — violates I4 invariant"
        )
        assert "hypothesis_verdict" in headline.columns, (
            "New 'hypothesis_verdict' column not added"
        )
