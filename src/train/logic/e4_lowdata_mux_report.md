# E4 low-data — 11-bit MUX learning curves

Plan: plan_2026-05-14_e26eede2.

Test set: fixed 2048-example 11-bit MUX (3 address + 8 data).
Models: circuit (depth=2, channels=32), mlp_matched (param-matched), xgboost.
Train sizes: N ∈ {32, 64, 128, 256, 512, 1024}.
3 seeds per cell.

| N | circuit (mean±std) | mlp_matched (mean±std) | xgboost (mean±std) |
|---|---|---|---|
| 32| 0.591±0.048| 0.599±0.043| 0.591±0.024 |
| 64| 0.630±0.041| 0.653±0.010| 0.641±0.037 |
| 128| 0.723±0.010| 0.744±0.006| 0.735±0.024 |
| 256| 0.847±0.009| 0.842±0.040| 0.889±0.010 |
| 512| 0.965±0.007| 0.969±0.007| 0.971±0.010 |
| 1024| 0.998±0.001| 0.999±0.002| 0.997±0.004 |

## Headline criterion: circuit beats both baselines by >5pt at N <= 128

- N=32: circuit=0.591, mlp=0.599, xgb=0.591 -> wins by >5pt: **False**
- N=64: circuit=0.630, mlp=0.653, xgb=0.641 -> wins by >5pt: **False**
- N=128: circuit=0.723, mlp=0.744, xgb=0.735 -> wins by >5pt: **False**

### Verdict: **FAIL — circuit does not consistently win at the low-data regime.**
