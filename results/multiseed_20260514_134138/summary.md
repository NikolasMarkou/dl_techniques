# Multi-seed Sweep Summary

Seeds requested: [0, 1, 2, 3, 4]

Sweep root: `results/multiseed_20260514_134138`

RNG seed for bootstrap/permutation: 20260514


## Run status

| experiment | sub | seed | status | wall_s |
|---|---|---|---|---|
| e1 | mnist | 0 | ok | 340.9 |
| e1 | cifar10 | 0 | ok | 94.1 |
| e3 | - | 0 | ok | 369.7 |
| e5 | - | 0 | ok | 254.2 |
| e1 | mnist | 1 | ok | 332.8 |
| e1 | cifar10 | 1 | ok | 81.9 |
| e3 | - | 1 | ok | 375.3 |
| e5 | - | 1 | ok | 227.6 |
| e1 | mnist | 2 | ok | 351.2 |
| e1 | cifar10 | 2 | ok | 93.6 |
| e3 | - | 2 | ok | 375.8 |
| e5 | - | 2 | ok | 285.3 |
| e1 | mnist | 3 | ok | 366.9 |
| e1 | cifar10 | 3 | ok | 85.8 |
| e3 | - | 3 | ok | 376.0 |
| e5 | - | 3 | ok | 256.5 |
| e1 | mnist | 4 | ok | 360.6 |
| e1 | cifar10 | 4 | ok | 81.7 |
| e3 | - | 4 | ok | 372.4 |
| e5 | - | 4 | ok | 262.5 |

## E1 (MNIST / CIFAR-10) — mean ± std per (dataset, model)

| dataset | model | n | final_val_acc | band_acc | delta_hard | roundtrip_diff | hard_acc | soft_acc |
|---|---|---|---|---|---|---|---|---|
| cifar10 | circuit | 5 | 0.5500 ± 0.0117 | 0.5500 ± 0.0117 | -0.0007 ± 0.0037 | 0.0000 ± 0.0000 | 0.5493 ± 0.0115 | 0.5500 ± 0.0117 |
| cifar10 | cnn_matched | 5 | 0.5333 ± 0.0264 | 0.5333 ± 0.0264 | nan ± nan | 0.0000 ± 0.0000 | nan ± nan | 0.5333 ± 0.0264 |
| mnist | circuit | 5 | 0.7007 ± 0.0004 | 0.7007 ± 0.0004 | -0.0337 ± 0.0628 | 0.0000 ± 0.0000 | 0.6670 ± 0.0631 | 0.7007 ± 0.0004 |
| mnist | cnn_matched | 5 | 0.6153 ± 0.0171 | nan ± nan | nan ± nan | 0.0000 ± 0.0000 | nan ± nan | 0.6153 ± 0.0171 |


## E3 (boolean tasks) — mean ± std per (task, model)

| task | model | n | band_acc | delta_hard | circuit_suff_auc | lime_suff_auc | shap_suff_auc | circuit_sparsity | lime_sparsity | shap_sparsity | circuit_stability | lime_stability | shap_stability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mux_11bit | circuit | 5 | 0.7508 ± 0.0275 | -0.0000 ± 0.0045 | 0.6547 ± 0.0407 | 0.6098 ± 0.0433 | 0.6463 ± 0.0375 | 0.9080 ± 0.0217 | 0.9318 ± 0.0239 | 0.8580 ± 0.0171 | 0.4701 ± 0.1029 | 0.5558 ± 0.1272 | 0.5198 ± 0.1063 |
| parity_k8 | circuit | 5 | 0.7545 ± 0.0434 | 0.0146 ± 0.0143 | 0.5527 ± 0.0188 | 0.5361 ± 0.0155 | 0.5510 ± 0.0161 | 0.8972 ± 0.0103 | 0.8744 ± 0.0192 | 0.8783 ± 0.0126 | 0.0656 ± 0.1595 | -0.0322 ± 0.2966 | 0.1312 ± 0.0834 |
| random_dnf_8input_4term | circuit | 5 | 0.7670 ± 0.0374 | -0.0004 ± 0.0060 | 0.6696 ± 0.0636 | 0.6649 ± 0.0758 | 0.6381 ± 0.0792 | 0.8755 ± 0.0180 | 0.8449 ± 0.0231 | 0.8253 ± 0.0425 | 0.4507 ± 0.1307 | 0.4633 ± 0.1465 | 0.4457 ± 0.1057 |


## E5 (CLEVR-Hans3) — mean ± std per model

| model | n | val_acc | test_acc | shortcut_gap |
|---|---|---|---|---|
| resnet50_circuit | 5 | 0.8503 ± 0.0034 | 0.6772 ± 0.0220 | 0.1731 ± 0.0199 |
| resnet50_mlp | 5 | 0.8502 ± 0.0032 | 0.6628 ± 0.0128 | 0.1874 ± 0.0109 |
| symbolic_circuit_oracle | 5 | 1.0000 ± 0.0000 | 0.9988 ± 0.0014 | 0.0012 ± 0.0014 |


## E5 inference

### E5 paired permutation test (circuit_shortcut_gap - mlp_shortcut_gap)

- n_pairs: 5
- mean(circuit - mlp): -0.0143 ± 0.0305
- 95% bootstrap CI of mean diff: [-0.0381, 0.0095]
- Permutation test (B=10000, two-sided, Phipson-Smyth corrected): observed_diff=-0.0143, **p=0.3753**
- Interpretation: no significant difference


## Regression vs prior n=1 (seed=42)

  - REGRESSION: e3//circuit:mux_11bit / delta_hard: prior_n1=-0.0156 outside new [mean ± 2*std]=[-0.0090, 0.0090]


## High-variance flags (std > |mean|)

  - HIGH-VARIANCE: cifar10/circuit / delta_hard: std=0.0037 > |mean|=0.0007 → retract or qualify
  - HIGH-VARIANCE: mnist/circuit / delta_hard: std=0.0628 > |mean|=0.0337 → retract or qualify
  - HIGH-VARIANCE: mux_11bit/circuit / delta_hard: std=0.0045 > |mean|=0.0000 → retract or qualify
  - HIGH-VARIANCE: parity_k8/circuit / circuit_stability: std=0.1595 > |mean|=0.0656 → retract or qualify
  - HIGH-VARIANCE: parity_k8/circuit / lime_stability: std=0.2966 > |mean|=0.0322 → retract or qualify
  - HIGH-VARIANCE: random_dnf_8input_4term/circuit / delta_hard: std=0.0060 > |mean|=0.0004 → retract or qualify
  - HIGH-VARIANCE: symbolic_circuit_oracle / shortcut_gap: std=0.0014 > |mean|=0.0012 → retract or qualify


### Claim survival audit

Prior n=1 claims (seed=42) with new n>=2 error bars:

| Experiment | Key | Metric | Prior n=1 | New mean ± std | Status |
|---|---|---|---|---|---|
| e1 | mnist/circuit | final_val_acc | 0.7006 | 0.7007 ± 0.0004 | WITHIN ± 2*std |
| e1 | mnist/circuit | delta_hard | 0.0008 | -0.0337 ± 0.0628 | WITHIN ± 2*std |
| e1 | mnist/circuit | band_acc | 0.7006 | 0.7007 ± 0.0004 | WITHIN ± 2*std |
| e1 | mnist/circuit | roundtrip_diff | 0.0000 | 0.0000 ± 0.0000 | saturated (std=0) |
| e1 | mnist/cnn_matched | final_val_acc | 0.6057 | 0.6153 ± 0.0171 | WITHIN ± 2*std |
| e1 | cifar10/circuit | final_val_acc | 0.5552 | 0.5500 ± 0.0117 | WITHIN ± 2*std |
| e1 | cifar10/circuit | delta_hard | -0.0009 | -0.0007 ± 0.0037 | WITHIN ± 2*std |
| e1 | cifar10/circuit | band_acc | 0.5552 | 0.5500 ± 0.0117 | WITHIN ± 2*std |
| e1 | cifar10/circuit | roundtrip_diff | 0.0000 | 0.0000 ± 0.0000 | saturated (std=0) |
| e1 | cifar10/cnn_matched | final_val_acc | 0.5208 | 0.5333 ± 0.0264 | WITHIN ± 2*std |
| e3 | -/circuit:mux_11bit | band_acc | 0.7031 | 0.7508 ± 0.0275 | WITHIN ± 2*std |
| e3 | -/circuit:mux_11bit | delta_hard | -0.0156 | -0.0000 ± 0.0045 | OUTLIER |
| e3 | -/circuit:parity_k8 | band_acc | 0.7246 | 0.7545 ± 0.0434 | WITHIN ± 2*std |
| e3 | -/circuit:parity_k8 | delta_hard | 0.0039 | 0.0146 ± 0.0143 | WITHIN ± 2*std |
| e3 | -/circuit:random_dnf_8input_4term | band_acc | 0.7100 | 0.7670 ± 0.0374 | WITHIN ± 2*std |
| e3 | -/circuit:random_dnf_8input_4term | delta_hard | 0.0000 | -0.0004 ± 0.0060 | WITHIN ± 2*std |
| e5 | -/resnet50_circuit | val_acc | 0.8547 | 0.8503 ± 0.0034 | WITHIN ± 2*std |
| e5 | -/resnet50_circuit | test_acc | 0.6742 | 0.6772 ± 0.0220 | WITHIN ± 2*std |
| e5 | -/resnet50_circuit | shortcut_gap | 0.1804 | 0.1731 ± 0.0199 | WITHIN ± 2*std |
| e5 | -/resnet50_mlp | val_acc | 0.8533 | 0.8502 ± 0.0032 | WITHIN ± 2*std |
| e5 | -/resnet50_mlp | test_acc | 0.6858 | 0.6628 ± 0.0128 | WITHIN ± 2*std |
| e5 | -/resnet50_mlp | shortcut_gap | 0.1676 | 0.1874 ± 0.0109 | WITHIN ± 2*std |
| e5 | -/symbolic_circuit_oracle | val_acc | 1.0000 | 1.0000 ± 0.0000 | saturated (std=0) |
| e5 | -/symbolic_circuit_oracle | test_acc | 0.9978 | 0.9988 ± 0.0014 | WITHIN ± 2*std |
| e5 | -/symbolic_circuit_oracle | shortcut_gap | 0.0022 | 0.0012 ± 0.0014 | WITHIN ± 2*std |
