# Sequence Pooling Review — `src/dl_techniques/layers/sequence_pooling/` (2026-10-15)

Method: static re-read of all 5 source files + README/GUIDE, test file (2209 lines), and consumer call sites. CPU-only; no GPU used. No source code changed during review. Findings double-checked line-by-line in a second pass; verdicts below mark confirmations, corrections, and downgrades.

Package: `AttentionPooling` (`attention_pooling.py`), `WeightedPooling` (`weighted_pooling.py`), `SequencePooling` facade (`sequence_pooling.py`, 18 strategies × 4 aggregations), registry factory (`factory.py`), curated `__init__.py`. Test suite is unusually strong (isolation probes F-24/F-25/H-02, golden numerics I1, second-length predict guard).

## Executive summary

No forward-pass math bug found in the shipped masked paths — the hard masking work (sentinels, ranking + validity, positional index, XLA-safe gather) is correct and well-pinned. All high-value findings are contract/validation/shape-inference gaps plus two consumer bugs: dead `bias_*` knobs, factory silent-drop violating the repo contract, `compute_output_shape` wrong for multi-`flatten`, late validation, `TextEncoder` strategy mutation that can `KeyError`, and mask-dropping similarity-head paths.

## 1. Confirmed bugs / dead knobs

- **SP-1 Dead `bias_initializer` / `bias_regularizer`** (`sequence_pooling.py:217-220, 248-266`): stored + serialized, never forwarded. `AttentionPooling` accepts neither (`attention_pooling.py:75-85`); `WeightedPooling` is forced to `initializer='ones', regularizer=kernel_regularizer`; `aggregation_weights` uses `kernel_regularizer` (`:280-285`). Setting either is a silent no-op.
- **SP-2 Factory silently drops unknown kwargs** (`factory.py:280-282`): filter-to-`valid_param_names` discards misspellings (e.g. `dropout=` for `dropout_rate`) with no error. `validate_sequence_pooling_config:171-234` never rejects unknown keys. Violates the repo factory contract (raise on undeclared keyword).
- **SP-3 `compute_output_shape` wrong for multi-`flatten`** (`sequence_pooling.py:845-854`): every non-composite strategy counts as `hidden_dim`. `['mean','flatten']` calls to `(B, D+S*D)` but reports `(B, 2D)`. Correction to first pass: `none`+`concat` IS guarded (`:788-789`); the live holes are `flatten` width, and `none`/dim-mismatched mixes under `add`/`multiply`/`weighted_sum` failing late with cryptic broadcast errors.
- **SP-4 Single-strategy skips aggregation validation** (`:783-784` returns before `:787-812`): `SequencePooling(strategy='mean', aggregation_method='invalid')` never raises.
- **SP-5 Late strategy validation**: no `__init__`/`build` check; `ValueError` only at `call` (`:757-758`). Factory validates numerics only, not strategy/aggregation names.
- **SP-6 No `seq_len > max_seq_len` guard** (`weighted_pooling.py:139`): `position_weights[:seq_len]` with over-long input has no explicit `ValueError`; failure mode is backend-dependent/cryptic.

## 2. Validation / robustness gaps (all verified)

- Zero-division `temperature` (`attention_pooling.py:169`, `weighted_pooling.py:140`): factory checks `>0`, direct construction doesn't.
- Same split for `hidden_dim`/`num_heads`/`top_k`/`max_seq_len` (factory-only checks); `top_k=0` flows into `top_k(k=0)` obscurely.
- `exclude_positions=[-1]` is a silent no-op (`sequence_pooling.py:321-324`: `arange != -1` always true). Out-of-range documented no-op; negatives/duplicates/non-ints unchecked.
- `get_config` drift: `'mean'` in → `['mean']` out (`:227` vs `:867`). Behavior-preserving, config-unequal.
- `get_sequence_pooling_info` overclaim (`factory.py:165-168`): per-entry `.copy()` still shares nested `optional_params`; mutating it mutates the registry.
- `mean_std`: hardcoded `1e-6` (`:689`, subnormal in fp16) + unmasked `ops.std` (`:691`) whose `ddof` vs masked `÷N` equivalence is unpinned.
- Sentinel split: `max`/`min` via `mask_sentinel(compute_dtype)` (`:638-660`) vs hardcoded `-1e4` in attention (`:177`), weighted (`:163`), `top_k_max` (`:746`). Both mask; uniformity nit only.
- `flatten`/`none` ignore mask and exclusions (`:604-605`, `:751-755`). Deliberate, but `flatten` bakes pad content into the vector.
- `mean` fully-masked → zeros (lengths floored `:688`), `max`/`min` → sentinels; only `WeightedPooling.call` docstring states its degeneration — `AttentionPooling.call:145-153` omits it.

## 3. Consumers

- **Vision excludes CLS for `mean`/`max` only** (`transformers/vision_encoder.py:334`, `models/vision/vit/model.py:674-677`); `mean_max`/`mean_std`/`sum`/`min`/learnable/top-k still average CLS. **Text never excludes** (`transformers/text_encoder.py:400-403`). Cross-encoder inconsistency, confirmed.
- **`TextEncoder` strategy mutation is a live bug pattern** (`text_encoder.py:720-721,759-760`): switching `pooling_layer.strategy` post-build leaves `learnable_components` stale — `get_pooled_features('attention')` on an encoder built with `output_mode='none'` hits `KeyError` instead of pooling. Also diverges config from runtime.
- **NLP heads**: main `_pool_sequence` passes mask (`heads/nlp/factory.py:453`); the delegated four (`cls`/`mean`/`max`/`last`) are sound. But the pairwise similarity path strips dicts to tensors and pools maskless (`:1513-1521`), with further maskless call sites (`:1571`, `:1999`). `last`/`mean`/`max` there read pad. Head `'attention'` is inline `Dense(1,tanh)` (`:363-372`, deliberate D-002) — different mechanism/weights from `SequencePooling('attention')`; README option table doesn't disambiguate.
- **Clifford-CLIP** (`models/vision_language/clifford_clip/model.py:898-935`): single-strategy `mean`/`max`/`attention` only — no multi-concat width trap reachable today (first-pass caution refuted).
- **`heads/vlm` `attention_pooling`**: `create_attention_layer` output, unrelated — grep hazard only, no coupling.

## 4. Docs drift (verified)

- `GUIDE.md:198-199` Pattern-B example uses additive `+(1-m)*-1e9` — the exact fp16-NaN form the package removed. Copy-paste reintroduces the bug.
- `GUIDE.md:101-102` "additive `-1e9` for `max`" contradicts shipped `where`+`mask_sentinel` selection.
- Shared-knob constraints undocumented in one place: one `temperature` drives both leaf poolers, `attention_dropout_rate` drives `WeightedPooling`, facade forces `WeightedPooling(initializer='ones')` (`sequence_pooling.py:248-266`).

## 5. Test gaps (observed absences via grep)

- `bias_*` only presence-checked in config (test `:876-878`), no effect test (nothing to test — dead).
- No factory unknown-kwarg rejection, single-strategy bad-aggregation, multi-`flatten` shape, `seq_len>max_seq_len`, `temperature=0`, or `exclude=[-1]` cases.

## 6. Non-issues (checked, cleared)

- `_take_positions` cumsum gather (`:545-553`), `_positional_index` middle/last ordinals, `_select_top_k` ranking+validity two-half fix, unknown-length pad/exclude path, `Dropout.build` with `None` batch (stateless no-op), `context_vector` glorot shape, dropout-after-softmax rescaling (standard inverted-dropout skew): read and behaving as documented/tested.
- Registry defaults match `__init__` signatures on cross-read; `list_sequence_pooling_types` sorted; `none`+`concat` guard present.

## Suggested order

SP-2 (factory contract) → SP-1 (dead knobs: forward or remove + test) → SP-3/SP-4/SP-5 (fail-fast + shape fn) → GUIDE fixes → consumer items (`exclude_positions` widening decision, `TextEncoder` mutation removal, similarity-head mask plumbing).
