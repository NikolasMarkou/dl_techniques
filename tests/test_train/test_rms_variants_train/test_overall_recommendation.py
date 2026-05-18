"""Tests for ``report.compute_overall_recommendation`` (Refinement B, step 3).

Synthetic-CSV exercises must cover all 4 verdict slots (RECOMMENDED_DEFAULT,
RECOMMENDED_NICHE, NULL, AVOID), tie/off-label edge rows, and AVOID triggers
from each of the three independent paths (headline FAIL, hypothesis REJECTED,
overhead-ceiling without redeeming PASS).

Falsification trigger per plan.md Pre-Mortem scenario 3: if ALL recommendations
collapse to the same slot regardless of input, OR rules NEVER recommend AVOID
even for adversarial input, the rule-set is unfalsifiable. The
``test_all_four_slots_reachable`` and ``test_avoid_reachable_*`` tests below
pin this.

Plan: ``plans/plan_2026-05-18_6776f8ba`` (step 3 gate; D-002 anchor).
"""
from __future__ import annotations

import pandas as pd
import pytest

from train.rms_variants_train.report import (
    BASELINE_NORM,
    OVERALL_RULES,
    compute_overall_recommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _hl(norm: str, verdict: str, *, exp: str = "e1", mode: str = "oob") -> dict:
    return {"experiment": exp, "norm_type": norm, "mode": mode, "verdict": verdict}


def _hyp(norm: str, v: str, *, exp: str = "e1", mode: str = "oob") -> dict:
    return {
        "experiment": exp, "norm_type": norm, "mode": mode,
        "hypothesis_verdict": v,
    }


# ---------------------------------------------------------------------------
# Anchor + frozen-rules surface
# ---------------------------------------------------------------------------

class TestFrozenRules:
    def test_overall_rules_dict_present(self) -> None:
        assert isinstance(OVERALL_RULES, dict)
        for k in (
            "headline_pass_required_for_recommend",
            "hypothesis_confirm_required_for_default",
            "overhead_ceiling_step_time_ratio",
            "calibration_ece_delta_max",
            "robustness_shift_acc_delta_min",
            "avoid_on_headline_fail",
            "avoid_on_hypothesis_rejected",
        ):
            assert k in OVERALL_RULES

    def test_decision_anchor_present_in_source(self) -> None:
        # SC15 spot-check for D-002.
        from train.rms_variants_train import report
        with open(report.__file__, "r") as fh:
            text = fh.read()
        assert "DECISION plan_2026-05-18_6776f8ba/D-002" in text


# ---------------------------------------------------------------------------
# RECOMMENDED_DEFAULT
# ---------------------------------------------------------------------------

class TestRecommendedDefault:
    def test_two_passes_plus_confirmed(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("zero_centered_rms_norm", "pass", exp="e1"),
            _hl("zero_centered_rms_norm", "pass", exp="e2"),
        ])
        hyp = pd.DataFrame([
            _hyp("zero_centered_rms_norm", "CONFIRMED", exp="e1"),
        ])
        out = compute_overall_recommendation(headline, hyp)
        zc = out[out["norm_type"] == "zero_centered_rms_norm"].iloc[0]
        assert zc["recommendation"] == "RECOMMENDED_DEFAULT"


# ---------------------------------------------------------------------------
# RECOMMENDED_NICHE
# ---------------------------------------------------------------------------

class TestRecommendedNiche:
    def test_one_pass_no_confirmation(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("band_rms", "pass", exp="e1"),
            _hl("band_rms", "indistinguishable", exp="e2"),
        ])
        hyp = pd.DataFrame([
            _hyp("band_rms", "INCONCLUSIVE", exp="e1"),
        ])
        out = compute_overall_recommendation(headline, hyp)
        br = out[out["norm_type"] == "band_rms"].iloc[0]
        assert br["recommendation"] == "RECOMMENDED_NICHE"

    def test_two_passes_but_no_confirmation_still_niche(self) -> None:
        # Two PASSes but no CONFIRMED hypothesis -> NICHE, not DEFAULT.
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("dynamic_tanh", "pass", exp="e1"),
            _hl("dynamic_tanh", "pass", exp="e5"),
        ])
        hyp = pd.DataFrame([
            _hyp("dynamic_tanh", "INCONCLUSIVE", exp="e1"),
            _hyp("dynamic_tanh", "INCONCLUSIVE", exp="e5"),
        ])
        out = compute_overall_recommendation(headline, hyp)
        dt = out[out["norm_type"] == "dynamic_tanh"].iloc[0]
        assert dt["recommendation"] == "RECOMMENDED_NICHE"


# ---------------------------------------------------------------------------
# NULL
# ---------------------------------------------------------------------------

class TestNull:
    def test_all_indistinguishable(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("band_rms", "indistinguishable", exp="e1"),
            _hl("band_rms", "indistinguishable", exp="e2"),
        ])
        hyp = pd.DataFrame([_hyp("band_rms", "INCONCLUSIVE")])
        out = compute_overall_recommendation(headline, hyp)
        br = out[out["norm_type"] == "band_rms"].iloc[0]
        assert br["recommendation"] == "NULL"

    def test_baseline_is_null(self) -> None:
        # Baseline must self-classify as NULL (not RECOMMENDED).
        headline = pd.DataFrame([_hl("rms_norm", "baseline")])
        hyp = pd.DataFrame()
        out = compute_overall_recommendation(headline, hyp)
        base = out[out["norm_type"] == BASELINE_NORM].iloc[0]
        assert base["recommendation"] == "NULL"

    def test_off_label_only_is_null(self) -> None:
        # A variant whose ONLY cells are off-label should be NULL (not AVOID).
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("band_logit_norm", "n/a (off-label)", exp="e2"),
            _hl("band_logit_norm", "n/a (off-label)", exp="e3"),
        ])
        hyp = pd.DataFrame()
        out = compute_overall_recommendation(headline, hyp)
        bl = out[out["norm_type"] == "band_logit_norm"].iloc[0]
        assert bl["recommendation"] == "NULL"


# ---------------------------------------------------------------------------
# AVOID — 3 independent triggers
# ---------------------------------------------------------------------------

class TestAvoid:
    def test_avoid_on_headline_fail(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("band_rms", "pass", exp="e1"),
            _hl("band_rms", "fail", exp="e2"),  # fail blocks recommendation
        ])
        hyp = pd.DataFrame([_hyp("band_rms", "CONFIRMED")])
        out = compute_overall_recommendation(headline, hyp)
        br = out[out["norm_type"] == "band_rms"].iloc[0]
        assert br["recommendation"] == "AVOID"
        assert "FAIL" in br["reason"]

    def test_avoid_on_hypothesis_rejected(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("band_rms", "pass", exp="e1"),
            _hl("band_rms", "pass", exp="e2"),
        ])
        hyp = pd.DataFrame([
            _hyp("band_rms", "CONFIRMED", exp="e1"),
            _hyp("band_rms", "REJECTED", exp="e2"),
        ])
        out = compute_overall_recommendation(headline, hyp)
        br = out[out["norm_type"] == "band_rms"].iloc[0]
        assert br["recommendation"] == "AVOID"
        assert "REJECTED" in br["reason"]

    def test_avoid_on_high_overhead_with_no_pass(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("adaptive_band_rms", "indistinguishable", exp="e1"),
        ])
        hyp = pd.DataFrame()
        overhead = pd.DataFrame([
            {"norm": "rms_norm", "mean_step_ms_fp32": 1.0},
            {"norm": "adaptive_band_rms", "mean_step_ms_fp32": 5.0},  # 5x baseline
        ])
        out = compute_overall_recommendation(headline, hyp, overhead_df=overhead)
        ab = out[out["norm_type"] == "adaptive_band_rms"].iloc[0]
        assert ab["recommendation"] == "AVOID"
        assert "overhead" in ab["reason"].lower()


# ---------------------------------------------------------------------------
# Overhead with redeeming PASS -> NULL not AVOID
# ---------------------------------------------------------------------------

class TestOverheadWithPass:
    def test_high_overhead_with_pass_demotes_to_null(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("adaptive_band_rms", "pass", exp="e1"),
            _hl("adaptive_band_rms", "pass", exp="e2"),
        ])
        hyp = pd.DataFrame([_hyp("adaptive_band_rms", "CONFIRMED")])
        overhead = pd.DataFrame([
            {"norm": "rms_norm", "mean_step_ms_fp32": 1.0},
            {"norm": "adaptive_band_rms", "mean_step_ms_fp32": 3.0},  # 3x > 1.5 ceiling
        ])
        out = compute_overall_recommendation(headline, hyp, overhead_df=overhead)
        ab = out[out["norm_type"] == "adaptive_band_rms"].iloc[0]
        # Gain exists but cost too high — demoted to NULL, not RECOMMENDED.
        assert ab["recommendation"] == "NULL"


# ---------------------------------------------------------------------------
# Calibration / robustness gating
# ---------------------------------------------------------------------------

class TestCalibrationRobustnessGating:
    def test_calibration_regression_demotes_to_null(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("dynamic_tanh", "pass", exp="e1"),
            _hl("dynamic_tanh", "pass", exp="e2"),
        ])
        hyp = pd.DataFrame([_hyp("dynamic_tanh", "CONFIRMED")])
        calibration = pd.DataFrame([
            {"norm_type": "dynamic_tanh", "ece_delta_vs_baseline": 0.10},  # >> 0.02
        ])
        out = compute_overall_recommendation(headline, hyp, calibration_df=calibration)
        dt = out[out["norm_type"] == "dynamic_tanh"].iloc[0]
        assert dt["recommendation"] == "NULL"
        assert "calibration" in dt["reason"].lower()

    def test_robustness_regression_demotes_to_null(self) -> None:
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("band_rms", "pass", exp="e1"),
            _hl("band_rms", "pass", exp="e2"),
        ])
        hyp = pd.DataFrame([_hyp("band_rms", "CONFIRMED")])
        robustness = pd.DataFrame([
            {"norm_type": "band_rms", "shift_acc_delta_vs_baseline": -0.20},  # < -0.05 floor
        ])
        out = compute_overall_recommendation(headline, hyp, robustness_df=robustness)
        br = out[out["norm_type"] == "band_rms"].iloc[0]
        assert br["recommendation"] == "NULL"
        assert "robustness" in br["reason"].lower()


# ---------------------------------------------------------------------------
# Falsification triggers (Pre-Mortem scenario 3)
# ---------------------------------------------------------------------------

class TestFalsificationTriggers:
    def test_all_four_slots_reachable(self) -> None:
        """Rules must NOT collapse — all 4 slots produced for a constructed input."""
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            # DEFAULT
            _hl("zero_centered_rms_norm", "pass", exp="e1"),
            _hl("zero_centered_rms_norm", "pass", exp="e2"),
            # NICHE
            _hl("dynamic_tanh", "pass", exp="e1"),
            _hl("dynamic_tanh", "indistinguishable", exp="e2"),
            # NULL
            _hl("band_rms", "indistinguishable", exp="e1"),
            _hl("band_rms", "indistinguishable", exp="e2"),
            # AVOID
            _hl("adaptive_band_rms", "fail", exp="e1"),
        ])
        hyp = pd.DataFrame([
            _hyp("zero_centered_rms_norm", "CONFIRMED", exp="e1"),
            _hyp("dynamic_tanh", "INCONCLUSIVE", exp="e1"),
        ])
        out = compute_overall_recommendation(headline, hyp)
        recs = set(out["recommendation"].unique())
        assert {"RECOMMENDED_DEFAULT", "RECOMMENDED_NICHE", "NULL", "AVOID"}.issubset(recs)

    def test_avoid_reachable_with_adversarial_input(self) -> None:
        # The corollary check: build the worst possible variant — every cell FAILs.
        headline = pd.DataFrame([
            _hl("rms_norm", "baseline"),
            _hl("bad_norm", "fail", exp="e1"),
            _hl("bad_norm", "fail", exp="e2"),
        ])
        hyp = pd.DataFrame([_hyp("bad_norm", "REJECTED")])
        out = compute_overall_recommendation(headline, hyp)
        bad = out[out["norm_type"] == "bad_norm"].iloc[0]
        assert bad["recommendation"] == "AVOID"


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_all_empty_returns_empty_frame(self) -> None:
        out = compute_overall_recommendation(pd.DataFrame(), pd.DataFrame())
        assert out.empty
        assert list(out.columns) == ["norm_type", "recommendation", "reason"]

    def test_only_baseline_present(self) -> None:
        headline = pd.DataFrame([_hl("rms_norm", "baseline")])
        out = compute_overall_recommendation(headline, pd.DataFrame())
        assert len(out) == 1
        assert out.iloc[0]["recommendation"] == "NULL"


if __name__ == "__main__":
    pytest.main([__file__, "-vvv"])
