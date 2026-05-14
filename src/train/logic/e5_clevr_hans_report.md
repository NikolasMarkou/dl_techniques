# E5 Report — CLEVR-Hans3 Visual Reasoning

*Plan: plan_2026-05-14_c95e848c. Decision D-001 (3-way: circuit / mlp / oracle).*

## Headline (shortcut-gap = val_acc - test_acc)

| Model | Status | Params | Epochs | Wall (s) | Val Acc (confounded) | Test Acc (clean) | Shortcut Gap |
|---|---|---|---|---|---|---|---|
| resnet50_circuit | complete | 23719363 | 17 | 132.7 | 0.8547 | 0.6742 | +0.1804 |
| resnet50_mlp | complete | 23723203 | 18 | 112.1 | 0.8533 | 0.6858 | +0.1676 |
| symbolic_circuit_oracle | complete | 12099 | 6 | 20.2 | 1.0000 | 0.9978 | +0.0022 |

## Notes
- val split keeps the train confounders (per CLEVR-Hans3 README); test split breaks them.
- Smaller `shortcut_gap` = more shortcut-resistant.
- `oracle` row uses scene-graph JSON directly (perfect perception); isolates reasoning-head bias.
- `mlp` is the param-matched baseline with identical ResNet50(frozen) backbone.