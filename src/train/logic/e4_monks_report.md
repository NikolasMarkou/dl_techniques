# E4 — UCI Monks rule-recovery benchmark

Plan: plan_2026-05-14_e26eede2.

Models per problem: `circuit` (LearnableNeuralCircuit, depth=2, channels=32),
`mlp_matched` (param-budget match), `xgboost` (200 trees, max_depth=4).
Seeds: 3 per cell. Test set is the canonical 432-config Monks test split.
Rule-recovery score is accuracy of model predictions on the 432-config
categorical enumeration vs the published Monks rule.

## Monks-1

| Model | test_acc (mean±std) | rule_recovery_acc (mean±std) | exact_match (count) | hard_soft_delta (mean) |
|---|---|---|---|---|
| circuit | 0.995±0.002 | 0.995±0.002 | 0/3 | -0.001 |
| mlp_matched | 0.999±0.001 | 0.999±0.001 | 2/3 | n/a |
| xgboost | 0.949±0.000 | 0.949±0.000 | 0/3 | n/a |

Circuit `to_symbolic` (seed 0):
`depth 0: |   logic_op_0: xor(0.194) |   logic_op_1: xor(0.194) |   arithmetic_op_0: max(0.375) |   arithmetic_op_1: max(0.375) |   combination: logic_op_0(0.303) | depth 1: |   logic_op_0: xor(0.196) |   logic_op_1: xor(0.196) |   arithmetic_op_0: max(0.392) |   arithmetic_op_1: max(0.392) |   combination: logic_op_0(0.315)`

## Monks-2

| Model | test_acc (mean±std) | rule_recovery_acc (mean±std) | exact_match (count) | hard_soft_delta (mean) |
|---|---|---|---|---|
| circuit | 1.000±0.000 | 1.000±0.000 | 3/3 | -0.002 |
| mlp_matched | 0.998±0.003 | 0.998±0.003 | 2/3 | n/a |
| xgboost | 0.785±0.000 | 0.785±0.000 | 0/3 | n/a |

Circuit `to_symbolic` (seed 0):
`depth 0: |   logic_op_0: xor(0.270) |   logic_op_1: xor(0.270) |   arithmetic_op_0: max(0.412) |   arithmetic_op_1: max(0.412) |   combination: logic_op_0(0.342) | depth 1: |   logic_op_0: nand(0.377) |   logic_op_1: nand(0.377) |   arithmetic_op_0: max(0.455) |   arithmetic_op_1: max(0.455) |   combination: logic_op_0(0.382)`

## Monks-3

| Model | test_acc (mean±std) | rule_recovery_acc (mean±std) | exact_match (count) | hard_soft_delta (mean) |
|---|---|---|---|---|
| circuit | 0.965±0.007 | 0.965±0.007 | 0/3 | +0.000 |
| mlp_matched | 0.966±0.003 | 0.966±0.003 | 0/3 | n/a |
| xgboost | 0.956±0.000 | 0.956±0.000 | 0/3 | n/a |

Circuit `to_symbolic` (seed 0):
`depth 0: |   logic_op_0: not(0.181) |   logic_op_1: not(0.181) |   arithmetic_op_0: add(0.343) |   arithmetic_op_1: add(0.343) |   combination: arithmetic_op_0(0.257) | depth 1: |   logic_op_0: nor(0.179) |   logic_op_1: nor(0.179) |   arithmetic_op_0: add(0.351) |   arithmetic_op_1: add(0.351) |   combination: arithmetic_op_0(0.259)`

## Headline criterion (analyses/analysis_2026-05-13_9c535f78 §E4)

Circuit beats BOTH mlp_matched AND xgboost by >5 points on test_acc:
- Monks-1: circuit=0.995, mlp=0.999, xgb=0.949 -> beats both by >5pt: **False**
- Monks-2: circuit=1.000, mlp=0.998, xgb=0.785 -> beats both by >5pt: **False**
- Monks-3: circuit=0.965, mlp=0.966, xgb=0.956 -> beats both by >5pt: **False**

### Verdict (>5pt-on->=2-of-3): **FAIL — circuit does not beat both baselines by >5pt on >=2 of 3 Monks tasks. Inductive-bias claim NOT supported by this experiment.**

## Rule-recovery verdict

- Monks-1: circuit rule_recovery_acc = 0.995±0.002
- Monks-2: circuit rule_recovery_acc = 1.000±0.000
- Monks-3: circuit rule_recovery_acc = 0.965±0.007

### Verdict (rule-recovery >=0.85 on >=1 task): **Circuit recovers a published Monks rule to >=0.85 enumeration accuracy on at least one of the three tasks — differentiable rule-extraction *capability* DEMONSTRATED.**
