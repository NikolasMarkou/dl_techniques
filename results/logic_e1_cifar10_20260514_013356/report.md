# E1 Hard-Extraction Δ Report — CIFAR10

*Plan: plan_2026-05-13_798d3a60. Rows: 2.*

## Headline (hard-extraction Δ at non-saturation)

| Model | Params | Band [low,high] | Band Acc | Soft Acc | Hard Acc | Δ (hard-soft) | Roundtrip Δ | Verdict |
|---|---|---|---|---|---|---|---|---|
| circuit | 98538 | [0.50, 0.80] | 0.5552 | 0.5551 | 0.5542 | -0.0009 | 0.00e+00 | FAITHFUL |
| cnn_matched | 96845 | [0.50, 0.80] | 0.5208 | 0.5208 | — | — | 0.00e+00 | no-circuit |

## Notes

- Headline metric is hard-extraction Δ on the **circuit** model at the **band-entry checkpoint** (val_accuracy ∈ [band_low, band_high]).
- CNN baseline has no inner ops to hard-extract; its row reports soft accuracy only for matched-param comparison.
- If `band_entered=False`, the model either saturated past the band in one epoch or never reached `band_low` within `max_epochs` — recorded as honest negative per the plan's Pre-Mortem (LESSONS L51 precedent).