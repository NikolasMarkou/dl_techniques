# NAM — Neural Arithmetic Module: Training Plan

## Problem Statement

The NAM must learn **5 sub-skills** to evaluate arithmetic expressions. Our current approach trains end-to-end on the final result only, forcing the model to discover all sub-skills simultaneously through a deep indirection chain. This fails because gradient signal from the final result is too weak to teach number extraction and operator classification through 10+ layers.

## Solution: Multi-Task Single-Flow Training

Train all sub-modules with **direct supervision** in a **single forward pass and single loss**. The data generator produces rich ground-truth labels for every intermediate step, and the loss is a weighted sum of per-module losses. No separate training scripts — one flow, one optimizer, one backward pass.

## Sub-Skills & Their Supervision

### S1: Number Extraction
**Module**: `number_head` (Dense → tanh × scale)
**Task**: Given token embeddings for a number span like `[1, 2, 3]`, output `123.0`
**Ground truth**: Parse the expression string to extract each number's value and its token positions
**Loss**: MSE between `number_head(left_focused)` and true left operand value; same for right
**Why it's hard now**: The model must learn digit→value mapping AND positional composition (hundreds/tens/ones) through deep transformer layers with only final-result supervision

### S2: Operator Classification
**Module**: `op_classifier` (Dense → softmax over 4)
**Task**: Given the expression, classify the operator as +/−/×/÷
**Ground truth**: The operator token in the expression string → index 0-3
**Loss**: Cross-entropy between `op_logits` and true operator index
**Why it's hard now**: The op_classifier receives its signal only through the soft-gated arithmetic output — if number extraction is wrong, the op_classifier gets corrupted gradients

### S3: Tree Structure / Sub-Expression Identification
**Module**: `group_attention` + `reduction_scorer`
**Task**: Identify which sub-expression to reduce (innermost / highest precedence)
**Ground truth**: For single-op expressions (phase 1), the reduction target is the operator position
**Loss**: Cross-entropy between `reduction_weights` and true operator position (one-hot)
**Why it's hard now**: No direct supervision — the model must discover expression structure from result supervision alone

### S4: Arithmetic Execution (Verification)
**Module**: Fixed arithmetic units (`_fixed_add/sub/mul/div`)
**Task**: Given correct (left_val, right_val, op_class), verify output = expected
**Note**: These are NOT learned — they are `keras.ops`. No loss needed. But we verify the pipeline: if S1 extracts correct operands and S2 classifies correctly, then the fixed units MUST produce the correct result. This is a diagnostic, not a training signal.

### S5: Final Result
**Module**: `result_head` (Dense)
**Task**: Predict the expression's numerical result
**Ground truth**: Python eval of the expression
**Loss**: Huber loss between predicted and true result

## Combined Loss (Single Flow)

```
L_total = w1 * L_number     # S1: number extraction MSE
        + w2 * L_operator   # S2: operator classification CE
        + w3 * L_reduction  # S3: reduction target CE
        + w4 * L_result     # S5: final result Huber
        + w5 * L_valid      # validity BCE
        + w6 * L_ponder     # ACT step penalty

All computed in ONE forward pass. ONE backward pass. ONE optimizer step.
```

### Empirically Validated Weights (Phase 1)

| Loss | Weight | Rationale |
|------|--------|-----------|
| `L_number` | **0.5** | Keep LOW — high values (5.0) create enormous gradient norms that suppress operator/reduction learning via global clip. Number extraction still converges with 0.5. |
| `L_operator` | 3.0 | Operator classification — needs meaningful gradient share |
| `L_reduction` | **5.0** | MOST critical — reduction must converge first (points at operator token). All downstream sub-skills depend on correct reduction focus. Without high weight, reduction stays at random (~10% accuracy). |
| `L_result` | 1.0 | End-to-end signal. Result head uses `stop_gradient` on inputs so this only trains the result head Dense layer, preventing gradient interference with sub-skills. |
| `L_valid` | 0.5 | Auxiliary |
| `L_ponder` | 0.01 | Regularizer |

**Key findings from training experiments:**
- `global_clipnorm=10.0` (not 1.0) — with 1.0, total gradient norms of 4000+ reduce effective LR to ~2e-8, killing all learning.
- Reduction convergence order: reduction (by ~2K steps) → operator (by ~5K) → numbers (by ~10K) → step accuracy climbs steadily after.
- At 20K steps: op=100%, red=100%, num_mse=0.17, 80% of step results within 10% tolerance, 55% within 5%.
- Sub-skills converge WITHOUT end-to-end result supervision. The `stop_gradient` on the result head is critical — without it, L_result gradients propagate backward and degrade sub-skill accuracy.

## Data Generator Changes

The data generator must produce **rich labels** alongside each expression:

```python
# Current: (expression, result, valid)
# New:     (expression, labels_dict)

labels = {
    "result": 14.0,                    # S5: final result
    "valid": True,                     # validity flag
    "left_operand": 6.0,              # S1: left number value
    "right_operand": 8.0,             # S1: right number value
    "operator_index": 0,              # S2: operator class (0=+, 1=-, 2=*, 3=/)
    "operator_position": 5,           # S3: token position of the operator
    "left_span": (1, 2),              # token positions of left number
    "right_span": (7, 8),             # token positions of right number
}
```

For single-op expressions (`phase_1`), this is straightforward to extract by parsing the expression string. For multi-op expressions (later phases), we generate a sequence of reduction steps with labels for each step.

## NAMCell Output Changes

The cell must expose its intermediate predictions so the loss can supervise them:

```python
outputs = {
    # Existing
    "result": result,
    "valid": valid,
    "op_logits": op_logits,          # S2: supervise with L_operator
    "q_halt": q_halt,
    "q_continue": q_continue,
    "hidden": hidden,
    "break_prob": break_prob,
    "group_prob": group_prob,
    
    # New — expose for intermediate supervision
    "left_val": left_val,            # S1: supervise with L_number
    "right_val": right_val,          # S1: supervise with L_number
    "reduction_weights": reduction_weights,  # S3: supervise with L_reduction
}
```

## Implementation Steps

### Step 1: Enrich Data Generator
**File**: `src/train/nam/data_generator.py`

Add a `_parse_expression()` function that extracts structured labels from the expression string. For phase_1 (single-op), this is simple string parsing:

```python
def _parse_single_op(expression: str, token_ids: list[int]) -> dict:
    """Parse '6 + 8' → {left: 6.0, right: 8.0, op_idx: 0, op_pos: 3}"""
```

Update `generate_batch()` to return the labels dict alongside the existing outputs.

### Step 2: Expose Cell Intermediates
**File**: `src/dl_techniques/models/nam/cell.py`

Add `left_val`, `right_val`, and `reduction_weights` to the cell's output dict. These already exist as local variables — just include them in the return.

### Step 3: Multi-Task Compiled Training Function
**File**: `src/train/nam/train_nam.py`

Rewrite `_make_compiled_train_fn` to compute the combined loss:

```python
# In the compiled step:
for i in range(max_act_steps):
    carry, outputs = model(carry, batch_data, training=True)
    
    # S1: Number extraction loss
    L_number += MSE(outputs["step_left_val"], true_left) + MSE(outputs["step_right_val"], true_right)
    
    # S2: Operator classification loss  
    L_operator += CE(outputs["op_logits"], true_op_index)
    
    # S3: Reduction target loss
    L_reduction += CE(outputs["reduction_weights"], true_op_position)
    
    # S5: Result loss
    L_result += Huber(outputs["result"], true_result)

L_total = w1*L_number + w2*L_operator + w3*L_reduction + w4*L_result + w5*L_valid + w6*L_ponder
```

### Step 4: Update Metrics
**File**: `src/train/nam/train_nam.py`

Track per-module metrics:
- `number_mse`: How well are operands extracted?
- `operator_acc`: How often is the correct operator selected?
- `reduction_acc`: Does the model focus on the right token position?
- `result_err`: Final result relative error (as before)

### Step 5: Update Tests
**File**: `tests/test_models/test_nam/test_nam.py`

Add tests for the new cell outputs and the enriched data generator.

## Training Flow (Single Script)

```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m train.nam.train_nam \
    --variant small --phase phase_1 --steps 20000 --act-steps 2 \
    --w-number 0.5 --w-operator 3.0 --w-reduction 5.0 \
    --result-loss-weight 1.0 --clip-norm 10.0
```

Expected log output:
```
Step 1000/10000 | loss=2.34 | num_mse=0.82 | op_acc=0.89 | red_acc=0.75 | result_err=0.15 | exact_acc=0.42
  per-op: '+'=0.92 | '-'=0.88 | '*'=0.85 | '/'=0.91
  numbers: left_mse=0.45 right_mse=0.39
```

## Expected Improvement

With direct supervision on number extraction and operator classification:

1. **Number extraction** should converge in ~1-2K steps (it's a simple regression from numeric features we already inject)
2. **Operator classification** should converge in ~500 steps (it's a 4-way classification of a single token type)
3. **Once S1+S2 are correct**, the fixed arithmetic units produce exact results automatically
4. **Final result accuracy** should jump because correct_operands + correct_operator + fixed_arithmetic = correct_result

The key insight: **we don't need the model to learn arithmetic** (it's fixed). We need it to learn to **read numbers and classify operators**. These are simple tasks that are currently buried under 10 layers of indirection.

## Multi-Op Expressions (Phase 2+)

For multi-op expressions, the labels become a **sequence of reduction steps**:

```
Expression: "3 + 5 * 2"
Step 0: reduce "5 * 2" → left=5, right=2, op=*, result=10, pos=6
Step 1: reduce "3 + 10" → left=3, right=10, op=+, result=13, pos=2
```

The data generator produces these by parsing with Python's `ast` module, which respects operator precedence and parentheses. Each ACT step gets supervised with the corresponding reduction step's labels.

This is a natural extension — the single-flow training works identically, just with per-step labels instead of a single label repeated across steps.

## File Changes Summary

| File | Change |
|------|--------|
| `src/train/nam/data_generator.py` | Add `_parse_single_op()`, enrich `generate_batch()` return |
| `src/dl_techniques/models/nam/cell.py` | Add `left_val`, `right_val`, `reduction_weights` to outputs |
| `src/train/nam/train_nam.py` | Multi-task loss in compiled fn, new metrics, loss weight args |
| `tests/test_models/test_nam/test_nam.py` | Test new outputs and enriched data generator |
