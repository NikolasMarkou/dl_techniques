## High value

| # | Duplicated code | Where | Proposed `common.py` function | Risk |
|---|---|---|---|---|
| 1 | Reshape `(B, N, H*d)` to `(B, H, N, d)` and back | anchor (6 times across two paths), beit `_split_heads`, capsule_routing, differential, kimi_delta | `split_heads(x, num_heads, head_dim)`, `merge_heads(x)` | Low. Pure reshape and transpose. Use `-1` for batch so static/dynamic shapes both work. |
| 2 | Scores, then probability, then dropout, then `@ v` | anchor (twice), area `_compute_attention`, beit, differential `_stream` | `dot_product_scores(q, k, scale)` and `attend(scores, v, prob_layer, dropout_layer=None, training=None)` | Medium. Keep these as two functions so beit (relative bias added between them) and capsule_routing (routing added between them) can still insert their own step. |
| 3 | Rank 2/3/4 mask reshaped to rank 4 | beit `_broadcast_mask`, capsule_routing `_apply_attention_mask`, area (inline reshape) | `broadcast_keep_mask(mask, seq_len=None) -> rank-4 keep` | Low. This only reshapes. It feeds `apply_attention_mask` and is not a second masking helper, so it respects the AGENTS.md rule. |
| 4 | `getattr(x.dtype, "name", None) or str(x.dtype)` | common, area, capsule_routing, beit, kimi_delta | `dtype_name(x) -> str` | None. It is the D-007 idiom, written once. |
| 5 | Rejecting `routing`, `deterministic_routing`, `hierarchical`, `hierarchical_routing` | capsule_routing `_DISALLOWED_PROB_TYPES`, differential (inline tuple), window (per AGENTS.md) | `NON_LOGIT_PROBABILITY_TYPES` constant and `validate_logit_probability_type(probability_type, layer_name)` | Low. The error messages differ today, so keep a `layer_name` in the message and check what the tests match on. |
| 6 | `(self.probability_config or {}).get("axis", -1)` for `rescue_axis` | area, and per the common.py docstring 9 of the 13 adopters | `probability_axis(probability_config, default=-1)` | None. |
| 7 | Q/K norm pair from `create_normalization_layer(type, name="q_norm", **(kwargs or {}))`, or `(None, None)` | area, capsule_routing, differential (two pairs), gated/hopfield/MLA per the README | `create_qk_norms(qk_norm_type, qk_norm_kwargs, suffix="") -> (q_norm, k_norm)` | Low. Pass the names through unchanged (`q_norm_1`, ...), because weight paths depend on them. |
| 8 | A Dense projection with a cloned initializer, shared regularizers and the D-056 "only the first site keeps the instance" rule | anchor (`common_kwargs`), beit (`dense_kwargs()` closure), differential (`dense_kwargs` dict), capsule_routing (4 blocks written out), kimi_delta (`dense()` closure) | `make_projection(units, name, *, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, keep_instance=False)` | Medium. The draw order has to stay the same or seeded weights move. `test_the_attention_projections_draw_independently.py` guards this. beit clones the bias initializer too and the others do not, so add a flag for that. |

## Bug-adjacent duplicate (fix first)

`area_attention.py` defines its own `_fresh_initializer` (serialize, then `get`). It reimplements `dl_techniques.initializers.clone_initializer`, which every other file imports. Replace it with that import. It belongs in `initializers`, not in `common.py`.

## Medium value

| # | Duplicated code | Where | Proposed function | Risk |
|---|---|---|---|---|
| 9 | `if x <= 0: raise ValueError(f"{name} must be positive, got {x}")` | every file | `validate_positive(**named_values)` | Medium. Message wording varies (`"dropout must be between 0 and 1"` in differential, `"must be in [0, 1]"` in area), and tests `match=` on it. Standardize the wording only alongside the tests. |
| 10 | Dropout rate range check | every file. The ranges differ: `[0, 1]` almost everywhere, `[0, 1)` in CapsuleBlockV2 | `validate_rate(name, value, *, upper_inclusive=True)` | Same as #9. |
| 11 | A static sequence length is required, else `ValueError` | anchor hierarchical, capsule_routing positional, ring, lighthouse (`RuntimeError`) | `require_static_length(x, axis, *, layer_name, hint)` | Low for the `ValueError` sites. Leave lighthouse alone: its `RuntimeError` is in the public contract. |
| 12 | `build()` checks the rank and that the last dim equals `dim` | anchor, beit, differential, channel, kimi_delta | `validate_input_shape(input_shape, *, rank, last_dim=None, layer_name)` | Medium. Tests match messages such as `"must match dim"`. |
| 13 | `Dropout(rate) if rate > 0 else None` | anchor, differential (`attn_dropout_*`), kimi_delta | `maybe_dropout(rate, name)` | Low. Dropout has no weights, so checkpoints are unaffected. Do not convert area and beit, which always create the layer: that changes their sub-layer tracking. |

## Low value or wrong home

- **Safe vector norm, `sqrt(sum(x^2) + eps)`**: capsule_routing `_squash`, AttentionRoutingCapsule direction, CapsuleBlockV2 (twice). This is capsule-specific, so a shared capsule utility fits better than `common.py`.
- **Serializing initializers and regularizers in `get_config()`**: repeated everywhere, but each call is one line. A helper saves little and hides the config keys.
- **`_scan_with_outputs`** (kimi_delta): it has one user. Move it only when ring or a second recurrent layer needs it.

## Do not extract

- **Relative position bias** (beit, single_window): it is a real-valued offset and must stay out of the mask path.
- **The `1/sqrt(d)` divide in AttentionRoutingCapsule and wave_field**: these divide on purpose. AGENTS.md says not to adopt `compute_attention_scale` there.
- **capsule_routing's per-site probability axes (`-1`, `-2`, `-1`)**: they are fixed by the routing math, so the `probability_axis` helper (#6) does not apply.
- **The fp16 `where(keep, score, -1e9)` in AttentionRoutingCapsule top-k**: it is safe because top-k guarantees kept entries. It is not a padding mask, so leave it out of `apply_attention_mask`.

## Suggested order

1. #4, #6, #3 and the `_fresh_initializer` fix: no behaviour change.
2. #1 and #7: mechanical, guarded by the existing shape and weight-path tests.
3. #5 and #2.
4. #8, then #9 to #13: these touch seeded draws or error messages, so update the tests in the same change.

Each adoption changes the importer counts in the `common.py` docstring, which are derived mechanically. Re-run its AST walk after every change.