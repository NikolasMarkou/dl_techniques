# NAM — Neural Arithmetic Module

This package had **no README** until 2026-08-18.

## What it is

A model for *evaluating* arithmetic expressions, assembled from three architectures
already in this tree:

1.  **Tree Transformer** (`models/tree_transformer/`, `GroupAttention`) — induces the
    expression's parse structure.
2.  **Neural Turing Machine** (`layers/memory/baseline_ntm.py`) — a recurrent context
    vector for the controller. It does **not** supply the operands.
3.  **Tiny Recursive Model**-style **ACT loop** — re-scores the expression and learns
    when to stop.

## Scope: single-operator, integer-only

MEASURED 2026-08-19 and pinned by
`tests/test_models/test_nam/test_operand_derivation_through_call.py`. The two operands
are assembled **exclusively** from the raw `input_ids`, split at
`argmax(reduction_weights)`, and `NAM.call` re-reads `batch["input_ids"]` unchanged on
every ACT step. Nothing written to NTM memory re-enters `left_val` / `right_val`, so:

*   **Multi-operator expressions do not chain, at any weights.** `"1 + 2 * 3"` splits
    at the `+` into `1` and the *concatenated* digits `23`; at the `*` into `12` and
    `3`. The answer 7 is not a candidate at any position. Parentheses (ids 18/19) are
    not digits, so they neither delimit nor group an operand — `"( 1 + 2 ) * 3"` gives
    `(12, 3)` at the `*`.
*   **Decimals are dropped.** `is_digit` is `4 <= id <= 13`; `DOT_ID = 20` is tokenized
    and round-trips, but number assembly has no fractional branch, so `"1.5 + 2"`
    assembles `15` and `2` and computes **17** — with `valid = 1.0`. There is no error
    and no validity flag for an out-of-scope expression.

The example below uses `"1 + 2 * 3"` to exercise the call contract, and not because the
model can evaluate it. Extra ACT steps re-score the same tokens; they do not reduce
them. Supporting either case would be a new mechanism, not a bug fix.

**The arithmetic itself is FIXED**, not learned. The model learns parsing, operand
routing, operator classification and halting; each operation emits a result *and* a
validity flag, so e.g. division by zero is representable as invalid rather than as a
number.

## The call contract is a loop, not a forward pass

`NAM.call` executes **one ACT step** and has the signature
`call(carry, batch, training=None) -> (new_carry, outputs)`. There is no
`model(x)` shortcut; a plain `model(ids)` raises `TypeError: missing a required
argument: 'batch'`. Drive it yourself:

```python
import numpy as np
from dl_techniques.models.nam import create_nam, ArithmeticTokenizer

tok = ArithmeticTokenizer(max_len=32)
ids = np.array([tok.encode("1 + 2 * 3")], dtype="int32")

model = create_nam("tiny")
batch = {"input_ids": ids}
carry = model.initial_carry(batch)

carry, out = model(carry, batch, training=False)   # one reduction step
# repeat until np.all(carry["halted"]) or carry["steps"] hits halt_max_steps
```

`carry` keys: `cell_carry`, `halted`, `steps`.
`outputs` keys, measured 2026-08-18 at `variant="tiny"`, batch 1
(184,334 parameters):

| key | shape | meaning |
|:---|:---|:---|
| `result`, `valid` | `(B, 1)` | running answer and its validity flag |
| `step_result`, `step_valid` | `(B, 1)` | this step's reduction |
| `step_left_val`, `step_right_val` | `(B, 1)` | the two operands this step consumed |
| `op_logits` | `(B, 4)` | operator classification |
| `q_halt_logits`, `q_continue_logits` | `(B,)` | ACT Q-values |
| `reduction_weights` | `(B, L)` | which sub-expression was reduced |

## Tokenizer

`ArithmeticTokenizer` has a **fixed 21-token vocabulary** and round-trips:

```python
tok = ArithmeticTokenizer(max_len=32)
tok.encode("1 + 2 * 3")   # [1, 5, 3, 14, 3, 6, 3, 16, 3, 7, 2, 0, ...]
tok.decode(_)             # '1 + 2 * 3'
```

## Variants

`NAM_VARIANTS` (module-level, the package's original spelling) and
`NAM.MODEL_VARIANTS` (class-level alias, added for the house shape) are the **same
dict**: `tiny`, `small`, `base`. They scale `hidden_size`, `num_heads`,
`num_tree_layers`, `intermediate_size`, `memory_size`, `max_expression_len` and
`halt_max_steps` together. They do **not** scale the head counts: all three
variants pin `num_read_heads=2` (left/right operand), which the previous
wording claimed otherwise, and `num_write_heads` no longer exists at all (see
below).

`create_nam(variant="base", **overrides)` applies individual `NAMConfig` field
overrides on top of a variant and raises `ValueError` for an unknown variant or a
config that fails `NAMConfig` validation.

## Two knobs that used to do nothing, and are now gone

`NAMConfig.shift_range` was documented, validated and serialized while configuring
nothing: `cell.py` builds every NTM read and write head with
`AddressingMode.CONTENT`, under which `layers/memory/baseline_ntm.py::NTMReadHead`
creates no circular-shift projection at all. The field was **removed on 2026-08-18**;
`NAMConfig.from_dict` ignores it, so a config dict serialized before then still
loads. Do not restore it — `cell.py:254` says so at the site. See `config.py`'s
class docstring.

`NAMConfig.num_write_heads` was the same defect one field away, and the audit that
removed `shift_range` missed it. `cell.py` constructs exactly one `NTMWriteHead` as
a single attribute — not a comprehension over a count, unlike `num_read_heads`
directly above it — so the field was a default, three identical variant entries
(all `1`) and a `to_dict()` key that no code read. Removed on **2026-08-19**;
`NAMConfig.from_dict` ignores it, so a config dict serialized before then still
loads with byte-identical behaviour, because the value never reached anything.
`num_read_heads` is live and unaffected.
