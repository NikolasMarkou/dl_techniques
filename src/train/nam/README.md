# NAM Training — Implementation Log

## Overview

This document records the implementation of the NAM multi-task training plan (see `NAM.md` in repo root) and every fix discovered through iterative training experiments.

The NAM (Neural Arithmetic Module) evaluates arithmetic expressions by parsing them into tree structures and reducing sub-expressions using fixed arithmetic operations. The model must learn 5 sub-skills: number extraction, operator classification, reduction targeting, arithmetic execution (fixed), and final result prediction.

The original approach trained end-to-end on the final result only. This failed because the gradient signal from the result is too weak to teach number extraction and operator classification through 10+ layers of indirection.

## Architecture

### NAM Forward Pass (Single ACT Step)

```
  Expression tokens: "15 + 3"
          |
          v
  +-----------------+
  | Token Embedding  |  + numeric injection [digit_val, is_digit, op_type]
  | + Positional Enc |
  +-----------------+
          |
          v
  +-----------------+
  | Tree Encoder     |  (4-6 TreeTransformerBlocks)
  | GroupAttention    |  parse expression structure
  +-----------------+
          |
          v  hidden (B, L, D)
  +-------+-------+-------------------+--------------------+
  |               |                   |                    |
  v               v                   v                    v
+----------+  +----------+   +--------------+   +----------------+
| Reduction |  | left_proj|   | NTM Memory   |   | Halt Head      |
| Scorer    |  | right_proj|  | Read/Write   |   | (q_halt/cont)  |
| Dense(1)  |  | Dense(D) |  | Heads        |   +----------------+
+----------+  +----------+   +--------------+
  |               |                   |
  v               v                   v
reduction_wts   left_focused       controller_out
  (B, L)       right_focused        (B, D)
  |               |                   |
  |               v                   v
  |         +----------+       +------------+
  |         | number_   |       | op_        |
  |         | head      |       | classifier |
  |         | Dense(1)  |       | Dense(4)   |
  |         | + clip    |       +------------+
  |         +----------+             |
  |           |     |                v
  |           v     v           op_probs (B, 4)
  |       left_val right_val        |
  |        (B,1)   (B,1)            |
  |           |     |                |
  |           v     v                v
  |     +-----------------------------------+
  |     | Fixed Arithmetic (NOT learned)    |
  |     | add, sub, mul, div                |
  |     | soft-select by op_probs (train)   |
  |     | hard-select argmax    (inference) |
  |     +-----------------------------------+
  |                    |
  |                    v
  |              step_result (B, 1)  <-- direct arithmetic output
  |                    |
  +--------------------+
                       |
                       v
              +------------------+
              | result_head      |  <-- stop_gradient on input!
              | Dense(D+2 -> 1)  |
              +------------------+
                       |
                       v
                final_result (B, 1)
```

### Multi-Task Loss Flow

```
                        L_total
                          |
          +-------+-------+-------+-------+-------+
          |       |       |       |       |       |
          v       v       v       v       v       v
      w=0.5   w=3.0   w=5.0   w=1.0   w=0.5  w=0.01
          |       |       |       |       |       |
     L_number  L_op   L_red  L_result L_valid L_ponder
          |       |       |       |       |       |
   log-MSE on  CE on   CE on   Huber   BCE    step
   left_val   clamped  clamped  log-    clamped count
   right_val  op_logits red_wts space   valid
     vs        vs       vs      vs       vs
   true_left  true_op  true_pos true_res true_valid
   true_right

  Gradients flow:
  L_number ----> number_head, left/right_proj, reduction_scorer, tree_encoder
  L_op     ----> op_classifier, controller, NTM heads, tree_encoder
  L_red    ----> reduction_scorer, tree_encoder
  L_result ----> result_head ONLY (stop_gradient blocks upstream)
  L_valid  ----> validity_head ONLY (stop_gradient blocks upstream)
```

### Gradient Isolation (Critical)

```
  WITHOUT stop_gradient:                  WITH stop_gradient:

  L_result                                L_result
     |                                       |
     v                                       v
  result_head                             result_head
     |                                       |
     v                                       X  <-- gradient blocked
  pooled + cell_result                    pooled + cell_result
     |         |                          (frozen for result_head,
     v         v                           but trained by L_number,
  tree_enc  number_head                    L_op, L_red separately)
     |         |
  EVERYTHING gets noisy                   Sub-skills train cleanly
  result gradients that                   without interference
  DEGRADE sub-skill learning
```

### Smooth Curriculum Distribution

```
  Probability
  mass
   |
  50%|*
     | *        progress = 0.0 (start)
     |  *       81% on 1-digit
     |   *
  25%|    *
     |     *  .
     |      *.  .                        progress = 0.5 (mid)
     |   .  . *  .                       roughly uniform
  12%| .       .  *  .
     |.    .    .   *  .   .
     +--+--+--+--+--+--+--+--
      1d 1d 1- 2- 3- 4- 6- 8-
      a/ al 2d 3d 4d 6d 8d 10d     Difficulty level
      s  l

  At progress = 1.0 (end):
  - 67% on 6-10 digit numbers
  - Still 10% on 1-2 digit (anti-forgetting)
  - Every level >= 2% probability
```

### Convergence Order (Empirical)

```
  Accuracy
  100% |          xxxxxxxxxxxxxxxxxxxxxxxxxx  reduction
       |         x
       |        x     ooooooooooooooooooooo  operator
   75% |       x    oo
       |      x   o
       |     x  o
   50% |    x o
       |   xo        ......................  step_10%
   25% |  xo       ..
       | xo     ...        ,,,,,,,,,,,,,,,,  step_5%
       |xo   ..        ,,,,
    0% +--+--+--+--+--+--+--+--+--+--+-->
       0  2K 4K 6K 8K 10K 12K 14K 16K 18K 20K  steps
```

## What Was Implemented

### 1. Cell Intermediates Exposed (`cell.py`)

Added `left_val`, `right_val`, and `reduction_weights` to `NAMCell` output dict. These already existed as local variables — just included them in the return.

```python
outputs = {
    ...
    "left_val": left_val,              # (B, 1) extracted left operand
    "right_val": right_val,            # (B, 1) extracted right operand
    "reduction_weights": reduction_weights,  # (B, L) sub-expression focus
}
```

### 2. Model Forwards Cell Intermediates (`model.py`)

NAM model now passes cell intermediates through as `step_left_val`, `step_right_val`, `reduction_weights`.

**Critical change — `stop_gradient` on result_head inputs:**

```python
result_input = ops.stop_gradient(
    ops.concatenate([pooled, cell_outputs["result"], cell_outputs["valid"]], axis=-1)
)
final_result = self.result_head(result_input)
final_valid = self.validity_head(ops.stop_gradient(pooled))
```

Without `stop_gradient`, the result loss gradients propagate backward through the entire model and **degrade sub-skill learning**. This was the single most important architectural fix. See Run 5 and Run 6 in the experiments section below.

### 3. Number Head — Removed `tanh`, Added Clamp (`cell.py`)

Original: `left_val = ops.tanh(self.number_head(left_focused)) * 100.0`

Problems:
- `tanh * 100` caps at [-100, 100] — can't handle 10-digit numbers
- For values 1-100, uses only the 1-100% range of tanh — gradient vanishing
- Unnecessary constraint

Changed to hard clamp:
```python
left_val = ops.clip(self.number_head(left_focused), -1e10, 1e10)
right_val = ops.clip(self.number_head(right_focused), -1e10, 1e10)
```

The clamp allows 10-digit numbers (up to 10^10) while preventing float32 overflow when two operands are multiplied (10^10 * 10^10 = 10^20, within float32 range of ~3.4e38). Unlike tanh, the clamp has unit gradient inside the range.

### 3b. Log-Compressed Result Encoding (`cell.py`)

The raw arithmetic result (up to 10^20 for multiplications) fed directly into `result_encoder` → NTM memory → `state_update` blows up the internal Dense pipeline. Even with clamped operands, the multiplication result is too large to flow through the model safely.

Fix — log-compress before encoding:

```python
result_compressed = ops.sign(result) * ops.log1p(ops.abs(result))
result_embedding = self.result_encoder(
    ops.concatenate([result_compressed, valid], axis=-1)
)
```

The raw `result` is still returned as output for loss computation. Only the internal pipeline sees the log-compressed version, which maps any scale to a bounded range (~0-46 for 10^20).

**Without this fix**, training NaN'd at step 15K when the curriculum introduced 2-3 digit numbers. With it, 100K steps complete cleanly.

### 4. Enriched Data Generator (`data_generator.py`)

Added `_parse_single_op()` that extracts structured labels from single-operator expressions:

```python
def _parse_single_op(expression, token_ids) -> dict:
    # Returns: left_operand, right_operand, operator_index, operator_position
```

`generate_batch()` now returns a 5-tuple: `(input_ids, targets, validity, expressions, labels)`.

**Smooth curriculum system** added via `generate_curriculum_batch()`:

- 8 difficulty levels spanning 1-digit add/sub to 10-digit all-ops
- `_curriculum_probs(progress)` computes a shifted Gaussian over levels
- At progress=0: 81% on 1-digit expressions
- At progress=1: concentrated on 6-10 digits but still 10% on 1-2 digits (anti-forgetting)
- Every level always retains >= 2% probability

### 5. Multi-Task Training Function (`train_nam.py`)

Rewrote `_make_compiled_train_fn` with 6 loss terms, all with numerical stability clamps discovered through iterative debugging:

| Loss | Description | Stability fix |
|------|-------------|---------------|
| `L_number` | Log-compressed MSE on `left_val`/`right_val` | `sign(x) * log1p(|x|)` for scale invariance across 1-10 digit numbers |
| `L_operator` | CE on `op_logits` vs true operator | Logits clamped to [-30, 30] to prevent CE explosion on confident-but-wrong predictions |
| `L_reduction` | CE on `reduction_weights` vs true operator position | Softmax probabilities clamped to [1e-7, 1.0] to prevent log(0) |
| `L_result` | Huber in log-space on result_head output | Only trains result_head (stop_gradient on inputs). Log-space Huber with delta=2 |
| `L_valid` | BCE on validity prediction | Sigmoid output clamped to [1e-7, 1-1e-7] to prevent log(0) |
| `L_ponder` | Step count penalty | Regularizer |

**Warmup + cosine decay LR** via `WarmupSchedule` from `dl_techniques.optimization`.

**Step-result metrics**: `step_exact_acc`, `step_acc_5pct`, `step_acc_10pct` — accuracy of the cell's direct arithmetic output (not the result_head), which is the true end-to-end pipeline metric.

**Digit accuracy matrix**: `_eval_digit_matrix()` evaluates a [1..10] x [1..10] x 4-ops grid at configurable intervals, showing exactly where the model succeeds and fails.

### 6. Tests (`test_nam.py`)

Added 10 new tests (49 total):
- Cell forward pass: `left_val`, `right_val`, `reduction_weights` shapes
- Model single step: `step_left_val`, `step_right_val`, `reduction_weights` shapes
- `_parse_single_op` for all 4 operators
- Enriched labels: correct shapes, operator indices, arithmetic consistency

## Training Experiments

### Run 1: NAM.md Original Weights (5K steps)
**Settings**: `w_number=5.0, w_operator=3.0, w_reduction=1.0, result=1.0, clip=1.0`
**Result**: All sub-skills stuck at random. L_result dominated 64% of gradient budget. Global clip_norm=1.0 with gradient norms of 30K+ reduced effective LR to ~2e-8.

### Run 2: Lower Result Weight (5K steps)
**Settings**: `w_number=5.0, w_operator=3.0, w_reduction=5.0, result=0.05, clip=1.0`
**Result**: Same failure. w_number=5.0 still created huge gradient norms.

### Run 3: Low w_number + Higher Clip (5K steps)
**Settings**: `w_number=0.5, w_operator=3.0, w_reduction=5.0, result=0.05, clip=10.0`
**Result**: **Breakthrough.** Reduction solved (100%), operator reached 72%, number MSE 5.6.

**Key insight**: w_number must be LOW (0.5 not 5.0). High values create gradient norms that suppress all other learning via global clip.

### Run 4: Result Weight Back to 1.0 (10K steps)
**Settings**: Same as run 3 but `result=1.0`
**Result**: L_result gradients **regressed** sub-skills. Op_acc only 50% vs run 3's 72%.

### Run 5: Resume from Run 3 + Result Weight (5K steps)
**Settings**: Resume checkpoint, `result=0.3, lr=3e-5`
**Result**: Number MSE regressed from 5.6 to 16+. Reduction dropped from 100% to 75%. Result loss gradients destroy sub-skill learning even at moderate weight.

### Run 6: stop_gradient on Result Head (10K steps)
**Settings**: Added `ops.stop_gradient` on result_head inputs. `w_number=0.5, w_operator=3.0, w_reduction=5.0, result=1.0, clip=10.0`
**Result**: **All sub-skills solved.** op=98%, red=100%, num_mse=1.7, op_entropy=3%. But exact_acc=0% because result_head can't learn (frozen inputs shift over training).

**Key insight**: `stop_gradient` on result_head is essential. Without it, any L_result weight > 0 degrades sub-skills.

### Run 9: 20K Steps — Full Pipeline (best single-op run)
**Settings**: Same as run 6, 20K steps
**Result**:
- op=100%, red=100%, num_mse=0.17 (RMSE 0.41)
- step_1%=12-19%, step_5%=52-61%, step_10%=67-80%
- Gradient norms stable at 130-250

**Convergence order**: reduction (2K) → operator (5K) → numbers (10K) → step accuracy climbs after

### Run 10: Values 1-100 (20K steps, curriculum not yet implemented)
**Settings**: `--min-val 1 --max-val 100`
**Result**: Same convergence pattern, just slower. op=100%, red=100% by step 9K. step_10%=50% by step 12K.

### Curriculum Runs: 1-digit to 10-digit

After implementing the smooth curriculum, several stability issues surfaced:

1. **L_valid BCE explosion** (run with curriculum, step ~6K): validity_head sigmoid output drifted to exactly 0.0 → BCE = -log(0) = inf → loss = 12.8 billion. **Fix**: clamp validity predictions to [1e-7, 1-1e-7].

2. **L_operator CE explosion** (100K run, step ~15K): op_logits diverged to extreme values when curriculum introduced unfamiliar number sizes. Confident-but-wrong prediction → CE of 10,884. **Fix**: clamp op_logits to [-30, 30] before CE.

3. **Eval NaN** (base variant): eval loop used `model.config.halt_max_steps=32` while training used `act_steps=2`. Over 32 steps, hidden states diverged. **Fix**: use `act_steps` for eval too.

4. **Number head NaN** (after removing tanh): unbounded Dense output → float32 overflow in `_fixed_multiply`. **Fix**: `ops.clip(..., -1e10, 1e10)`.

5. **Internal pipeline NaN** (100K curriculum run, step 15K): the raw arithmetic result (up to 10^20 for multiplications) flowed into `result_encoder` → memory → `state_update`. Dense layers amplified it until values went to inf → NaN. Clamping losses didn't help because the NaN originated in the forward pass, not the loss. **Fix**: log-compress the result before encoding into the internal pipeline (`result_compressed = sign(result) * log1p(|result|)`). The raw result is still returned for loss computation.

### Final Run: 100K Steps with All Fixes

**Settings**: `--variant base --curriculum --steps 100000 --batch-size 64 --lr 1e-4 --clip-norm 10.0 --act-steps 2 --w-number 0.5 --w-operator 3.0 --w-reduction 5.0 --result-loss-weight 1.0`

**Zero NaN.** All 100K steps completed cleanly.

**Training trajectory:**

| Step | loss | num_mse | op | red | step_10% |
|------|------|---------|-----|-----|----------|
| 5K | 3.6 | 0.71 | 100% | 100% | 19% |
| 20K | 1.0 | 0.22 | 97% | 100% | 22% |
| 35K | 1.6 | 0.32 | 98% | 100% | 28% |
| 50K | 7.4 | 0.46 | 91% | 100% | 23% |
| 65K | 16.4 | 0.28 | 83% | 94% | 31% |
| 80K | 17.4 | 0.84 | 83% | 95% | 23% |
| 100K | 16.5 | 0.67 | 75% | 98% | 14% |

**Final digit accuracy matrix — addition (10% tolerance):**

```
       1d    2d    3d    4d    5d    6d    7d    8d    9d   10d
 1d   50%  100%    0%    0%    0%    0%    0%    0%    0%    0%
 2d  100%  100%   25%    0%    0%    0%    0%    0%    0%    0%
 3d    0%   50%  100%  100%    0%    0%    0%    0%    0%    0%
 4d    0%    0%  100%  100%   75%   25%    0%    0%    0%    0%
 5d    0%    0%    0%   25%   75%    0%   25%    0%    0%    0%
 6d    0%    0%    0%   50%    0%    0%   50%    0%    0%    0%
 7d    0%    0%    0%    0%    0%   25%    0%    0%    0%    0%
 8d    0%    0%    0%    0%    0%    0%    0%    0%   25%    0%
 9d    0%    0%    0%    0%    0%    0%    0%   75%   50%    0%
10d    0%    0%    0%    0%    0%    0%    0%    0%    0%    0%
```

**Final digit accuracy matrix — multiplication (10% tolerance):**

```
       1d    2d    3d    4d    5d    6d    7d    8d    9d   10d
 1d    0%   50%    0%    0%    0%    0%    0%    0%    0%    0%
 2d   50%  100%  100%    0%    0%    0%    0%    0%    0%    0%
 3d    0%   75%  100%   75%    0%    0%    0%    0%    0%    0%
 4d    0%   25%  100%  100%   50%   25%    0%    0%    0%    0%
 5d    0%    0%   25%    0%   25%   25%    0%    0%    0%    0%
 6d    0%   25%    0%   50%   25%   50%   25%    0%    0%    0%
 7d    0%    0%    0%   75%   50%    0%   25%    0%    0%    0%
 8d    0%    0%    0%    0%    0%    0%    0%    0%    0%    0%
 9d    0%    0%   25%    0%    0%    0%    0%    0%    0%    0%
10d    0%    0%    0%    0%    0%    0%    0%    0%    0%    0%
```

**Observations:**

1. **Strong diagonal band 2-4 digits** — the model solves same-size or adjacent-size problems reliably at 75-100%.
2. **Mismatched sizes fail** — e.g., `1d × 3d` is 0% while `3d × 3d` is 100%. The number_head can't simultaneously read small and large numbers from the same focused representation well.
3. **10-digit cliff** — the model never gets a 10-digit problem right in either operand position. The curriculum probability at end is only 25% on 8-10d — not enough exposure for the hardest cases.
4. **Operator accuracy drops under hard data** — op went from 100% at step 5K down to 75% at step 100K. The harder the problem, the more the operator classifier gets confused.
5. **Reduction stays stable** — 94-100% throughout.
6. **step_1% vs step_10% gap** — the model is doing approximate computation, not exact. RMSE of operands is small but amplified through multiplication.

**What works**: the multi-task training strategy with all the fixes does train. Sub-skills converge and the model handles 1-7 digit single-op arithmetic with meaningful accuracy.

**Remaining limitations**: cross-scale operations, 9-10 digit extremes, and precise (1%) accuracy. These likely need more curriculum weight on hard examples, a larger model, or a different number representation (e.g., digit-by-digit instead of scalar).

## Architecture Decisions

### Why stop_gradient on result_head?

The result_head takes `[pooled_hidden, cell_result, cell_valid]` and outputs a scalar prediction. Without stop_gradient, L_result gradients flow backward through:

```
result_head → pooled → hidden → tree_encoder → EVERYTHING
result_head → cell_result → fixed_arithmetic → number_head → hidden → EVERYTHING
```

This creates a competing gradient signal that drowns out the surgical sub-skill supervision. Empirically, even `result_loss_weight=0.3` degrades sub-skills when stop_gradient is absent.

With stop_gradient, L_result only trains the 130-parameter result_head Dense layer. The sub-skills train independently via their own losses.

### Why log-compressed number loss?

Raw MSE blows up for large numbers: a 10-digit prediction error of 10^9 produces MSE of 10^18. Log-compression (`sign(x) * log1p(|x|)`) makes the loss scale-invariant:

- log1p(5) ≈ 1.8 (1-digit)
- log1p(500) ≈ 6.2 (3-digit)
- log1p(5e9) ≈ 22.3 (10-digit)

All produce comparable loss magnitudes.

### Why smooth curriculum over discrete phases?

Discrete phases cause catastrophic forgetting — the model loses earlier skills when switched to harder data. The smooth curriculum:

- Always mixes in easy examples (>= 2% probability per level)
- Shifts gradually: at progress=0.5 all levels get meaningful probability
- At progress=1.0, hard levels dominate but easy levels retain 5-10%

## CLI Reference

```bash
# Smooth curriculum (recommended)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m train.nam.train_nam \
    --variant base --curriculum \
    --steps 100000 --batch-size 64 \
    --lr 1e-4 --clip-norm 10.0 --act-steps 2 \
    --w-number 0.5 --w-operator 3.0 --w-reduction 5.0 \
    --result-loss-weight 1.0 \
    --log-interval 5000 --eval-interval 25000 --save-interval 25000

# Fixed phase (legacy)
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m train.nam.train_nam \
    --variant small --phase phase_1 \
    --steps 20000 --batch-size 64 \
    --lr 1e-4 --clip-norm 10.0 --act-steps 2 \
    --w-number 0.5 --w-operator 3.0 --w-reduction 5.0 \
    --result-loss-weight 1.0

# Custom value range
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m train.nam.train_nam \
    --variant small --phase phase_1 --min-val 1 --max-val 1000 \
    --steps 20000 --act-steps 2
```

### Key Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--curriculum` | off | Smooth curriculum: 1-digit → 10-digit over training |
| `--variant` | small | Model size: tiny/small/base |
| `--act-steps` | from config | ACT depth override (use 2 for single-op) |
| `--w-number` | 0.5 | Number extraction loss weight (keep low) |
| `--w-operator` | 3.0 | Operator classification loss weight |
| `--w-reduction` | 5.0 | Reduction target loss weight (keep high) |
| `--result-loss-weight` | 1.0 | Result head loss weight |
| `--clip-norm` | 10.0 | Global gradient clip norm |
| `--eval-interval` | 1000 | Digit accuracy matrix eval frequency |
| `--min-val` / `--max-val` | from phase | Override operand value range |

## File Changes Summary

| File | Changes |
|------|---------|
| `src/dl_techniques/models/nam/cell.py` | Expose `left_val`, `right_val`, `reduction_weights`; replace `tanh*100` with `clip(-1e10, 1e10)`; log-compress arithmetic result before internal pipeline (`result_encoder`) |
| `src/dl_techniques/models/nam/model.py` | Forward cell intermediates; `stop_gradient` on result_head and validity_head inputs |
| `src/train/nam/data_generator.py` | `_parse_single_op()`; enriched `generate_batch()` returning labels; `generate_curriculum_batch()` with 8 difficulty levels and smooth probability shifting |
| `src/train/nam/train_nam.py` | Multi-task compiled loss (6 terms); log-compressed number/result losses; clamped op_logits/reduction_weights/validity losses; warmup+cosine LR; step_result metrics; digit accuracy matrix eval; `--curriculum` mode; `--min-val`/`--max-val` overrides |
| `tests/test_models/test_nam/test_nam.py` | 10 new tests for cell outputs, model outputs, parser, enriched labels |
