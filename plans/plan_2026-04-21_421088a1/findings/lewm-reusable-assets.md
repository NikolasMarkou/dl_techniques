# lewm-reusable-assets — what to lift from the just-closed LeWM port

## AdaLN-zero (`src/dl_techniques/layers/adaln_zero.py`)

- Class: `AdaLNZeroConditionalBlock(dim, num_heads, dim_head, mlp_dim, dropout=0.0,
  use_causal_mask=True, eps=1e-6)`.
- **Takes `inputs=[x, c]`** where `x` is `(B, T, D)` and `c` is `(B, T, D)` (or
  broadcastable to it).
- Zero-init `adaLN_linear` (Dense 6*dim) → at init: `shift=scale=gate=0` →
  **block is identity in x**. Test gate: `max|block([x,c]) - x| < 1e-6` pre-training.
- Uses Keras `MultiHeadAttention` with `use_causal_mask=True`. Good for
  AR predictor on T axis. But for the *patch-level* predictor over `(B, T, N, D)`
  it does NOT natively do spatial attention — we'd use Clifford blocks there, and
  keep AdaLN-zero only on the **temporal** axis (or skip it in favor of AdaLN-zero
  *conditioning* on Clifford blocks).

**Decision implication for D-004**: AdaLN-zero is cheapest to inject where there's
already a `(B, T, D)` per-frame context sequence. For 5D patch grids, we'd either:
  (i) expand `c` to `(B, T, N, D)` and apply modulation before each Clifford block, or
  (ii) keep conditioning only in the temporal pass.

## SIGReg (`src/dl_techniques/regularizers/sigreg.py`)

- **`SIGRegLayer(knots=17, num_proj=1024, seed=None)`** — `keras.layers.Layer`, not
  `keras.regularizers.Regularizer`.
- Input convention `(..., N, D)` — averages cos/sin over axis=-3 (the N axis).
- Upstream LeWM used `(T, B, D)` by transposing the batch axis to position -3.
- For our patch case: per-patch latent collapse could feed `(T*B*N_patches, D)`
  reshaped to e.g. `(T*B, N_patches, D)` — averaging over patches per frame.
- **D-005 interpretation**:
  - Per-frame pooled `(B*T, D)`: need to reshape to `(1, B*T, D)` and SIGReg
    averages over B*T samples. Cheap.
  - Per-patch flattened `(B*T*N, D)`: reshape to `(1, B*T*N, D)`. More samples
    → tighter statistic; may over-regularize dense features (user's concern).
  - Middle ground: `(B*T, N, D)` — SIGReg averages over N-patches per frame,
    then the outer mean over (B, T). Most principled IMO.
- Sampling `num_proj` random projections every call — cost is `D * num_proj`
  memory + one matmul. At smoke-scale `num_proj=64` is fine.

## LeWM model.py — reusable patterns

- `encode_pixels(pixels)`: `(B, T, H, W, C)` → reshape to `(B*T, H, W, C)`,
  single forward through ViT, reshape to `(B, T, D)`. **This pattern lifts
  directly to patch-level** — ViT returns `(B*T, N, D)` with `include_top=False,
  pooling=None`, reshape to `(B, T, N, D)`.
- `rollout(...)`: autoregressive loop — keeps `emb[:, -HS:]` as a rolling window.
  Exactly the streaming-inference pattern we need for D-007.
- `call()`: uses `self.add_loss(pred_loss)` + `self.add_loss(sigreg_weight*sigreg)`,
  compile with `loss=None`, `jit_compile=False`. **Reuse this pattern.**
- Config serialization via dataclass `to_dict()` / `from_dict()` in `get_config` /
  `from_config`.

## Encoder choice (D-001)

- ViT-tiny at patch=14, img=224 → `(B, 196, 192)` with `include_top=False, pooling=None`.
  Reasonable for drone footage if we downscale. BUT: ViT has no causal / multi-scale
  inductive bias, and drone small-object sensitivity argues for CNN-like locality.
- Clifford encoder stack on `PatchEmbedding2D → (B, H_p, W_p, D)` is 4D-friendly
  for `CliffordNetBlock` — direct fit. No dropout / no FFN but residual+GGR.
- **Recommendation**: hybrid (c) — `PatchEmbedding2D` then 2–4 `CliffordNetBlock`
  layers. Parameter-cheap, local-aware, matches the "dual-stream detail+context"
  story that fits small objects. Keep ViT as fallback comparison.
