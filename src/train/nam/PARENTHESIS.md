# Parenthesis Support — Investigation Log

## Summary

Three attempts to add parenthesis support to the DFSA recursive architecture. All failed. Flat multi-op expressions remain 100% accurate. Parenthesized expressions reach ~61% at best. The root cause is architectural: the operator classifier produces wrong outputs when one operand is a `RESULT_PLACEHOLDER` token instead of digit tokens.

## Background

The DFSA recursive architecture (iteration 4) handles flat multi-op expressions like `3 + 5 * 2` perfectly via:
1. PEMDAS scoring selects the highest-precedence operator
2. `op_cumsum` segmentation extracts adjacent operands exactly
3. Fixed arithmetic computes the result
4. Re-tokenization replaces `<left> <op> <right>` with a `RESULT_PLACEHOLDER` token
5. Value buffer carries the result float with gradients into the next step

For parenthesized expressions like `(3 + 5) * 2`, the reduction order must change: `+` inside parens must be reduced before `*` outside, even though `*` has higher PEMDAS precedence.

## What Was Tried

### Design Verification (Pre-Implementation)

Before writing any code, verified that the existing architecture can theoretically handle parens:

**Adjacent masking**: Parentheses are tokens 18 `(` and 19 `)`. They are neither operators nor digits, so `op_cumsum` doesn't count them and `is_value` doesn't include them. The segment-based masking naturally "sees through" parens to find the correct adjacent operands.

Tested 8 cases including `(3 + 5) * 2`, `2 * (3 + 5)`, `(10 - 3) * (2 + 4)`, `((2 + 3)) * 4`, `(1 + 2) * (3 - 4) / 5`. All gave exact operand extraction.

**PEMDAS with paren depth**: `paren_depth = cumsum(is_open_paren) - cumsum(is_close_paren)` at each position. Adding `paren_depth * 100` to PEMDAS scores ensures innermost operators are always selected first. Score for `+` inside one level of parens: `1*100 + 15 = 115` > score for `*` outside: `0*100 + 20 = 20`.

**Gradient flow**: Verified with TF GradientTape that the value buffer bypass works across paren reductions. Step 1's `op_logits` receive meaningful gradients from step 2's result loss.

### Implementation Changes

Three code changes were needed:

1. **PEMDAS scoring** (`train_dfsa.py:reduce_step`):
```python
is_open_paren = ops.cast(ops.equal(token_ids, 18), "float32")
is_close_paren = ops.cast(ops.equal(token_ids, 19), "float32")
paren_depth = ops.cumsum(is_open_paren, axis=1) - ops.cumsum(is_close_paren, axis=1)
pemdas_scores = paren_depth * 100.0 + high_prec * 20.0 + low_prec * 15.0
```

2. **Re-tokenization paren clearing** (`train_dfsa.py:_retokenize`): After clearing `<left> <op> <right>`, expand the clear mask to also remove adjacent `(` and `)` tokens (2 expansion rounds for up to 2 nesting levels):
```python
is_clearable = ops.clip(is_space + is_paren, 0.0, 1.0)
for _ in range(2):
    shifted_r = concatenate([zeros, clear_mask[:-1]])
    shifted_l = concatenate([clear_mask[1:], zeros])
    adjacent_clear = is_clearable * clip(shifted_r + shifted_l, 0.0, 1.0)
    clear_mask = clip(clear_mask + adjacent_clear, 0.0, 1.0)
```

3. **Data generator** (`data_generator.py`):
   - Added difficulty levels 11-12 for parenthesized expressions (2-3 ops, 1-3 digit operands)
   - Added `_generate_paren_expr()` that wraps one random adjacent operand pair in parens
   - Rewrote `_parse_multi_op()` to respect paren depth when determining reduction order
   - Rewrote `prepare_per_step_labels()` to simulate the model's actual re-tokenization (instead of teacher-forced intermediate tokens) so per-step operator positions match what the model sees

### Attempt 1: Basic Implementation (20k steps, curriculum-cap=0.8)

**Command:**
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m train.nam.train_dfsa \
    --hidden-size 256 --num-tree-layers 3 --num-heads 8 \
    --max-len 128 --act-steps 3 --steps 20000 --batch-size 64 --lr 1e-4 \
    --clip-norm 10.0 --warmup-steps 1000 --w-operator 3.0 --w-reduction 5.0 \
    --result-loss-weight 1.0 --curriculum-cap 0.8 --multiop-start-step 0
```

**Training trajectory:**

| Step | op | red | step_1% |
|------|-----|-----|---------|
| 2K | 0.984 | 0.969 | 0.891 |
| 6K | 1.000 | 0.984 | 0.953 |
| 10K | 1.000 | 1.000 | 0.969 |
| 14K | 0.922 | 0.875 | 0.750 |
| 20K | 0.922 | 0.875 | 0.781 |

**Problem identified**: Accuracy declined after step 10K. The per-step labels from `prepare_per_step_labels` assumed teacher-forced intermediate tokens, but the model uses recursive re-tokenization. For `(3 + 5) * 2`:
- Step 0: reduces `+`. Both label and model agree (same original tokens).
- Step 1: label says `*` at position 3 (from teacher-forced "8 * 2"). Model sees `*` at position 9 (original position, surrounded by PADs and RESULT_PLACEHOLDER).
- **Position mismatch** → L_reduction trains the model to look at the wrong position.

**Eval:** Not run for this attempt (killed early after identifying the label bug).

### Attempt 2: Fixed Labels (20k steps, curriculum-cap=0.8)

Rewrote `prepare_per_step_labels` to simulate the model's re-tokenization: start from original tokens, apply the same PEMDAS + clearing logic, find operator positions in the re-tokenized sequences.

**Training trajectory:**

| Step | op | red | step_1% |
|------|-----|-----|---------|
| 2K | 0.984 | 0.953 | 0.875 |
| 8K | 1.000 | 1.000 | 0.969 |
| 10K | 1.000 | 1.000 | 0.953 |
| 16K | 0.938 | 0.828 | 0.594 |
| 20K | 0.922 | 0.875 | 0.781 |

Label fix helped convergence (step 8K: perfect) but accuracy still declined at high curriculum levels. At `curriculum-cap=0.8`, paren levels (11-12) get only 24% weight at the end of training — most of the curriculum is large-number flat expressions that overwhelm the paren signal.

**Eval results:**

| Test | Accuracy |
|------|----------|
| single-op | 300/300 (100%) |
| 2-op-flat | 320/320 (100%) |
| 3-op-flat | 200/200 (100%) |
| 2-op-paren | **183/320 (57.2%)** |
| 3-op-paren | **89/200 (44.5%)** |
| PEMDAS-vs-paren | **6/12** |
| edge-cases | **3/8** |

Sample failures:
```
(3 + 5) * 2          = 16.00   pred=30.00   err=87.5%
(10 - 2) * 3         = 24.00   pred=2.00    err=91.7%
8 / (4 + 1)          = 1.60    pred=4.00    err=150.0%
((3 + 5)) * 2        = 16.00   pred=0.43    err=97.3%
(7 - 3) * (8 - 6)    = 8.00    pred=6.64    err=17.0%
```

### Attempt 3: More Training + Higher Curriculum Cap (30k steps, curriculum-cap=1.0)

Increased `curriculum-cap` from 0.8 to 1.0 so the curriculum reaches the full distribution where paren levels get 43% weight. Extended to 30K steps for 50% more training.

**Training trajectory:**

| Step | op | red | step_1% |
|------|-----|-----|---------|
| 3K | 1.000 | 0.984 | 0.922 |
| 9K | 1.000 | 1.000 | 0.969 |
| 15K | 1.000 | 1.000 | 0.922 |
| 24K | 0.969 | 0.875 | 0.750 |
| 30K | 0.906 | 0.656 | 0.500 |

Same pattern: converges by step 9K, then collapses as harder curriculum data arrives. The final batch metrics are worse than attempt 2, though batch metrics on hard curriculum don't predict eval performance.

**Eval results:**

| Test | Accuracy |
|------|----------|
| single-op | 300/300 (100%) |
| 2-op-flat | 320/320 (100%) |
| 3-op-flat | 200/200 (100%) |
| 2-op-paren | **197/320 (61.6%)** |
| 3-op-paren | not recorded |
| PEMDAS-vs-paren | **6/12** |
| edge-cases | **3/8** |

Slightly better on 2-op-paren (61.6% vs 57.2%) but still far from usable.

## Root Cause Analysis

### Why flat multi-op works but parens don't

For flat expressions `3 + 5 * 2`:
- Step 0: all tokens are original digits and operators. The tree encoder sees familiar embeddings. Op classifier works perfectly.
- Step 1: the re-tokenized sequence has `RESULT_PLACEHOLDER` (token 21) at one position. But the operator and its adjacent digit operand are STILL original tokens. The op classifier pools features at the operator position, which has normal tree-encoded features from real digit neighbors.
- Result: operator classification works because the operator's local context (its neighbors) is still mostly original tokens.

For parenthesized expressions `(3 + 5) * 2`:
- Step 0: reduces `+` inside parens. Fine — all original tokens.
- Step 1: `*` is selected. Its left neighbor is `RESULT_PLACEHOLDER` (token 21) embedded in a sea of PAD tokens (the cleared `( 3 + 5 )` region). The tree encoder processes this unfamiliar token surrounded by PADs. The pooled features at `*`'s position are out-of-distribution compared to training.
- Result: op classifier sees features it wasn't trained on → wrong predictions.

### The specific failure mode

Looking at the failures:

```
(3 + 5) * 2 = 16    pred=30    → model computes 15*2=30 (treats result as 15, not 8)
(10 - 2) * 3 = 24   pred=2     → model divides instead of multiplying
8 / (4 + 1) = 1.6   pred=4     → model ignores paren result, does 8/4+1=3 or 8/(something wrong)
```

The errors are not random — the model is either:
1. Misreading the value buffer (getting wrong operand values)
2. Misclassifying the operator (the main failure)
3. Both

The value buffer read path is deterministic (`_assemble_with_bypass` checks `buffer_active > 0.5` then reads `value_buffer` directly), so #1 shouldn't fail. The issue is #2: the op_classifier gets wrong features when tree-encoded representations include RESULT_PLACEHOLDER positions with unusual context.

### Why more training doesn't help

The op_classifier is a single `Dense(D, 4)` layer that reads pooled tree features weighted by `reduction_weights`. It learns to map these features to operator types during training on primarily flat expressions. Paren expressions produce different feature distributions at the pooled position because:

1. Token embedding for ID=21 (RESULT_PLACEHOLDER) is randomly initialized and rarely trained
2. PAD tokens surrounding the placeholder create dead zones in the tree attention
3. The positional encoding at the operator position is correct, but the attending tokens are PAD/placeholder instead of digits

Even with 43% paren weight in the curriculum, the op_classifier doesn't receive a strong enough learning signal to generalize from "operator between digits" to "operator between digit and placeholder-in-PAD-sea."

## What Would Fix It

### Approach A: Dedicated result embedding (most promising)

Instead of using token_embedding(21) for the RESULT_PLACEHOLDER, inject the result value directly into the embedding space:

```python
# In reduce_step, after re-tokenization:
# At RESULT_PLACEHOLDER positions, replace the token embedding with a
# learned projection of the result value
result_emb = self.result_value_proj(value_buffer)  # (B, L) → (B, L, D)
x = x * (1 - buffer_active_expanded) + result_emb * buffer_active_expanded
```

This gives the tree encoder meaningful features at placeholder positions — features that represent the ACTUAL numerical value, not just "this is a placeholder." The op_classifier would then see "this operator is between a 8.0-valued embedding and a digit embedding" instead of "this operator is between a random embedding and a digit."

### Approach B: Two-phase training

1. Phase 1: Train on flat expressions until convergence (current 20K run)
2. Phase 2: Freeze number assembly and reduction scoring. Fine-tune ONLY the tree encoder and op_classifier on parenthesized expressions with a low learning rate.

This lets the op_classifier adapt to the new feature distribution at placeholder positions without forgetting flat expression handling.

### Approach C: Compact re-tokenization

Instead of leaving the cleared region as PADs with a single RESULT_PLACEHOLDER, LEFT-SHIFT all tokens after the cleared region to close the gap. The result becomes a compact sequence like `BOS RES SPACE * SPACE 2 EOS` instead of `BOS PAD PAD PAD RES PAD PAD PAD SPACE * SPACE 2 EOS`.

This eliminates the PAD desert around the placeholder, giving the tree encoder a more normal-looking sequence. However, this is complex to implement differentiably (variable-length shift within tf.function).

### Approach D: Skip the tree encoder for step 1+

For steps after the first, bypass the tree encoder entirely. Since:
- Reduction position is deterministic (PEMDAS, hardcoded)
- Number values are deterministic (digit assembly or buffer read)
- The only learned component is op_classifier

...we could classify the operator directly from its token ID (which is deterministic) without needing tree features at all. The op_classifier becomes trivial: `op_type = token_id - 14`. This would make parens work perfectly but removes the learning from the op_classifier entirely.

## Files Changed (Reverted)

All paren-specific changes were reverted after the 3 failed attempts. The files that were modified:

- `train_dfsa.py`: paren_depth computation in PEMDAS scoring, paren clearing in `_retokenize`
- `data_generator.py`: `_generate_paren_expr()`, paren-aware `_parse_multi_op()`, model-aligned `prepare_per_step_labels()`

The `eval_dfsa.py` script was kept as it's useful for testing all expression types.

## Conclusion

Parenthesis support requires more than training data and scoring changes. The fundamental issue is that the tree encoder + op_classifier pipeline produces unreliable features when operands are `RESULT_PLACEHOLDER` tokens in PAD-heavy contexts. Approach A (dedicated result embedding) is the most promising fix as it addresses the root cause without architectural surgery.
