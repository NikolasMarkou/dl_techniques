# clifford-primitives — exact shapes & contracts

Source: `src/dl_techniques/layers/geometric/clifford_block.py` (1063 lines, fully read).

## Primitives (all `@keras.saving.register_keras_serializable()`)

### `SparseRollingGeometricProduct(channels, shifts, cli_mode="full", ...)`
- Inputs: `z_det, z_ctx` — **both identical shape** `(B, H, W, D)` (actually any
  rank, `roll` is along axis -1 and `proj` is a Dense on last dim — so 5D works
  element-wise on non-channel axes).
- Filters `shifts >= channels` automatically (warns).
- `cli_mode="full"` concatenates 2 tensors per shift → `Dense(2*len(shifts)*D → D)`.
- Output: `(B, H, W, channels)`.
- **Build takes a single input_shape** (not dual) — the layer build assumes both
  streams share the shape.

### `GatedGeometricResidual(channels, layer_scale_init=1e-5, drop_path_rate=0.0)`
- Inputs: `h_norm, g_feat` both `(B, H, W, D)`.
- LayerScale gamma `(channels,)` init Constant(1e-5).
- Output: residual term (not residual-added — caller adds to `x_prev`).

### `CliffordNetBlock(channels, shifts, cli_mode, ctx_mode, use_global_context, ...)`
- Input: `(B, H, W, D)` — 4D.
- Forward: `LayerNorm → (Detail: Dense) ∥ (Ctx: DWConv→DWConv→BN→SiLU) → [diff]
  → local geo product → [+ optional global GAP branch] → GGR → residual`.
- Output: `(B, H, W, D)` same shape (residual; `compute_output_shape` returns input).
- Global branch hardcodes `shifts=[1,2], cli_mode="full", ctx_mode="diff"`.
- BatchNorm inside context stream — ⚠ batch-of-1 is unsafe; relevant for smoke.

### `CausalCliffordNetBlock(channels, shifts, ...)` — identical call signature
- Input: `(B, 1, seq_len, D)` — **H must be 1**.
- DWConv kernel `(1, 3)` with `padding="valid"`, manual **left-only** pad of 2
  zeros on W axis (`_causal_pad`).
- Global branch uses **`_causal_cumulative_mean`** along axis=2 (division by
  [1..seq_len]) — prefix mean, preserves causality.
- Output: same shape `(B, 1, seq_len, D)`.
- **Causality invariant**: pos i depends only on pos ≤ i. This is the hardest
  case to verify per user instruction. A regression test must perturb pos k and
  assert output at pos < k is unchanged.

## Implications for Video-JEPA predictor

- **Factorized 5D (D-002 option a)**: treat `(B, T, H_p, W_p, D)`.
  - Spatial pass: reshape to `(B*T, H_p, W_p, D)`, apply `CliffordNetBlock` → reshape back.
  - Temporal pass: transpose to `(B*H_p*W_p, T, D)`, reshape to `(B*H_p*W_p, 1, T, D)`,
    apply `CausalCliffordNetBlock` → reshape back. Cost per frame = O(T) in T (cumsum),
    O(1) amortized with rolling buffer.
- **BN-batch-of-1 risk**: at smoke time with B=2, T=4 the flattened batch is 8
  (spatial) and H_p·W_p·B (temporal). Both > 1. OK.
- **No BatchNorm-free variant exists** — if BN shows up unstable with small batches,
  this is a known trap (user's depth-estimation lesson). Fallback: LayerNorm swap.
- **Serialization**: the blocks are already serializable — if we just use them
  by composition in our predictor, no extra get_config work required for them.

## Shape-contract pitfalls to test
1. `CausalCliffordNetBlock` expects H=1. If the outer reshape gets `(B, H_p, W_p, D)`
   we MUST collapse to `(B*H_p*W_p, 1, T, D)`, not pass `(B, H_p, T*W_p, D)`.
2. `SparseRollingGeometricProduct.build(input_shape)` uses only the last dim —
   rank-agnostic, but the Dense projection input shape must match. Safe.
3. `GGR` sums `SiLU(h_norm) + alpha * g_feat` then `* gamma` then optional DropPath.
   With `layer_scale_init=1e-5`, residual term ≈ 0 at init. Useful for
   "identity-at-init" invariants alongside AdaLN-zero.
