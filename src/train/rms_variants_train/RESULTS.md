# RMSNorm Variants Study — Results

*This file is populated by `train.rms_variants_train.report` after the full
multi-seed sweep. Until then, see `README.md` for the planned matrix.*

## Status

- Harness: built (plan_2026-05-14_3764496e).
- Sweep: pending — run via `python -m train.rms_variants_train.sweep` then
  `python -m train.rms_variants_train.report`.

## Verdict template

For each variant (`band_rms`, `zero_centered_rms_norm`,
`zero_centered_band_rms_norm`):

- **PASS** / **FAIL** / **INDISTINGUISHABLE**: <reason>.
- Experiments where this verdict holds: E#, E#, ...
- Counter-evidence (if any): ...
