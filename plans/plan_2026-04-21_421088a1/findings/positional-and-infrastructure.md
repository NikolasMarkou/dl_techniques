# positional-and-infrastructure

## Positional primitives available (`src/dl_techniques/layers/embedding/`)

- `PatchEmbedding2D(patch_size, embed_dim, ...)` — Conv2D patcher, input
  `(B, H, W, C)`, output `(B, N, embed_dim)`. **Flattens spatial dims.**
  For a 5D patch grid we need spatial awareness; we can either:
  (a) use PatchEmbedding2D then reshape `(B, N, D) → (B, H_p, W_p, D)`, or
  (b) build our own inline `Conv2D(patch, patch, stride=patch)` returning the
      4D grid directly (simpler for Clifford downstream).
- `PositionEmbeddingSine2D(num_pos_feats, temperature=10000, normalize=True)` —
  **output shape `(B, 2*num_pos_feats, H, W)`** i.e. channels-first. Needs a
  transpose to `(B, H, W, 2*num_pos_feats)` before broadcasting. Produces a
  fixed, non-learnable encoding suitable for patch grids. Add directly to the
  patch embedding tensor (after dim-match — `2*num_pos_feats` should equal D).
- `ContinuousSinCosEmbed(dim, ndim, max_wavelength=10000)` — embed continuous
  scalar/vector coords. Useful for **temporal axis** (embed frame index as float)
  and for **telemetry scalars** (altitude, velocity). 1D-temporal-learned is the
  user's default (D-006) but ContinuousSinCos is a reusable shoulder for it.
- `factory.py` + `patch_2d` / `positional_learned` exist — not needed to extend.

## Recommendation for D-006
- Patch PE: `PositionEmbeddingSine2D` on the `(B, H_p, W_p, D)` grid once, added
  per-frame (broadcasts over T).
- Temporal PE: new `add_weight(shape=(1, T_max, D))` inside the predictor
  (learned) — matches LeWM `pos_embedding` pattern already in `ARPredictor`.
- Telemetry encoding: `ContinuousSinCosEmbed(dim=cond_dim, ndim=k)` where
  `k = len(telemetry_channels)` (e.g. 7 = 3 IMU Δ + 3 velocity + 1 altitude).

## LeWM model — streaming precedent

From `models/lewm/model.py:rollout`: maintains `emb` buffer, truncates to last
`HS` steps before each predictor call. This is the O(1) amortized streaming
pattern for D-007. Direct analog for our case:
- Buffer: `emb_buf: (B, K, H_p, W_p, D)` (last K frame-patch-grids).
- Encode new frame → `(B, 1, H_p, W_p, D)` → append → truncate to K.
- Predictor call on the K-window returns `(B, 1, H_p, W_p, D)` predicted next.

## Training script template (`src/train/lewm/train_lewm.py`)

- Uses `train.common.setup_gpu`, `add_loss` inside call, compiles `loss=None,
  jit_compile=False`. Clone this pattern verbatim for `src/train/video_jepa/`.

## Datasets ecosystem
- `src/dl_techniques/datasets/pusht_hdf5.py` has `synthetic_lewm_dataset(...)`
  generator precedent for synthetic tf.data smoke fixtures. We mirror it for
  `synthetic_drone_dataset(...)` — moving shapes + fake IMU/GPS telemetry.
- `src/dl_techniques/datasets/simple_2d.py` — trivial synthetic 2D data (ref).
- No real drone dataset present. Scope out of plan.

## Keras 3.8 serialization gotchas (from LESSONS)
- Every custom layer/model: `@keras.saving.register_keras_serializable()` +
  `get_config()` + **all sublayers stored via explicit attrs, not dicts**.
- `load_weights(path.keras, by_name=True)` is broken — irrelevant here (fresh
  model). But for streaming, if we checkpoint mid-training, use
  `model.save(...)` / `keras.models.load_model(...)` only.
- BatchNorm inside `CliffordNetBlock` — risky at batch-of-1; smoke defaults
  should keep B ≥ 2.

## V-JEPA masking precedent (`src/dl_techniques/models/jepa/`)
- Existing `JEPAMaskingStrategy` implements block-based semantic masking for
  images. **For tube masking over `(T, H_p, W_p)` we need new code** — the
  existing utility doesn't cover temporal tubes. Small extension (~50 lines).
- If D-003 picks "next-frame only" this is skipped entirely.
