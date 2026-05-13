# E3 Faithfulness Report

*Plan: plan_2026-05-13_798d3a60. Rows: 3.*

## Headline (hard-extraction Δ + per-method faithfulness)

| Task | Model | Params | Band | Soft | Hard | Δ | RT | Circuit suff/comp | LIME suff/comp | SHAP suff/comp |
|---|---|---|---|---|---|---|---|---|---|---|
| mux_11bit | circuit | 673 | [0.70,0.95] (0.703) | 0.703 | 0.688 | -0.016 | 0.0e+00 | 0.589/0.098 | 0.562/0.107 | 0.577/0.100 |
| parity_k8 | circuit | 577 | [0.70,0.95] (0.725) | 0.725 | 0.729 | +0.004 | 0.0e+00 | 0.528/0.069 | 0.538/0.075 | 0.528/0.073 |
| random_dnf_8input_4term | circuit | 577 | [0.70,0.95] (0.715) | 0.715 | 0.709 | -0.006 | 0.0e+00 | 0.558/0.050 | 0.556/0.066 | 0.577/0.068 |

## Per-task notes
