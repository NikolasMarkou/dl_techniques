# Plan v1

## Goal
Port LeWM (JEPA-based action-conditioned world model) from PyTorch to dl_techniques Keras 3 framework. Deliver a runnable smoke-test training script on synthetic data, a round-trip-serializable model, and a thin HDF5 loader skeleton for the PushT schema.

## Problem Statement

**Expected behavior.** Given a sequence of image observations `pixels: (B, T, H, W, C)` and the actions taken between them `action: (B, T-1, A)`, the model:
1. Encodes each frame to a latent via ViT-tiny (patch=14, img=224) → CLS → projector → `(B, T, D=192)`.
2. Embeds actions via Conv1d(k=1) + 2-layer MLP → `(B, T, D)`. Last action is padded with zeros so action time axis matches state time axis.
3. Autoregressively predicts next-frame latents via a 6-deep causal Transformer with AdaLN-zero conditioning on the action embedding → `(B, T, D)`.
4. Training target = the encoder's own output on the next frame (JEPA-style self-supervised). Loss = MSE(pred, target) + 0.09 · SIGReg(emb).

**Invariants.**
- Serialization round-trip: `model.save(path.keras); keras.models.load_model(path)` yields identical outputs given identical inputs (within `atol=1e-5`).
- `compute_output_shape` is defined on every custom layer.
- Causal mask: predictor must not attend to future tokens (same decision surface as upstream `is_causal=True`).
- AdaLN-zero: at init, `adaLN_modulation` final linear has zero weight+bias, so the block is an identity map. Verify by forward pass at init step.
- Shape contract: `rollout(context_pixels, action_sequence)` returns `predicted_emb: (B, S, T, D)` where T = history + n_future_steps.

**Edge cases.**
- `action_dim != embed_dim` (Embedder must handle arbitrary input dim). PushT action_dim = 2; embed_dim = 192.
- `T=1` history: pos-embedding slice must be valid for any `T <= num_frames`.
- Boundary NaNs in actions (episode breaks) → replaced with 0 in preprocessing.
- `proprio` optional — loader should gracefully skip if absent.

## Context (from EXPLORE)

Findings 1–4 in `findings.md`. Key grounding:
- Existing ViT in `src/dl_techniques/models/vit/model.py` supports `include_top=False, pooling='cls'` with patch=14, img=224, scale='tiny' (192d, 3h, 12L). Returns `(B, 192)` directly — perfect for `encode()` body.
- `src/dl_techniques/layers/transformers/transformer.py` has `TransformerLayer` with factory-driven attn+FFN, but **no AdaLN-zero support** — we must write `AdaLNZeroConditionalBlock` as a new layer.
- `src/dl_techniques/layers/film.py` does scale+shift but not the 6-way AdaLN-zero modulation (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
- Existing `models/jepa/` is masked-JEPA (I/V-JEPA) — different paper, different architecture. We must use a new name `models/lewm/` to avoid collision.
- SIGReg has no analogue — write as a new `keras.Layer` (stateless, uses `keras.random.normal` per forward pass for projection matrix).

## Files To Modify / Create

**New — library code:**
1. `src/dl_techniques/layers/adaln_zero.py` — `AdaLNZeroConditionalBlock` layer.
2. `src/dl_techniques/regularizers/sigreg.py` — `SIGRegLayer` (inherits `keras.layers.Layer`, returns scalar loss tensor; we treat as a layer rather than a Keras regularizer because it depends on forward activations, not weights).
3. `src/dl_techniques/models/lewm/__init__.py` — empty per convention.
4. `src/dl_techniques/models/lewm/config.py` — `LeWMConfig` dataclass + defaults matching `config_lewm.yaml`.
5. `src/dl_techniques/models/lewm/embedder.py` — `ActionEmbedder` (Conv1D k=1 + 2-layer MLP with SiLU).
6. `src/dl_techniques/models/lewm/predictor.py` — `ARPredictor` (learned pos embedding + stack of AdaLN-zero blocks + input/cond/output projections + final LayerNorm).
7. `src/dl_techniques/models/lewm/projector.py` — `MLPProjector` (Linear → BatchNorm → GELU → Linear). Used for both `projector` and `pred_proj`.
8. `src/dl_techniques/models/lewm/model.py` — `LeWM` keras.Model with `encode`, `predict`, `rollout`, `call` (training forward = encode + predict + loss assembly).

**New — training & data:**
9. `src/dl_techniques/datasets/pusht_hdf5.py` — thin `PushTHDF5Dataset` class (h5py + tf.data wrapper) + `SyntheticLeWMDataset` generator for smoke test. ImageNet normalization + resize to 224.
10. `src/train/lewm/__init__.py`
11. `src/train/lewm/train_lewm.py` — production-grade training script following `src/train/vit/` / `cliffordnet/` pattern: argparse, config dict, synthetic or HDF5 path, callbacks (TerminateOnNaN, CSVLogger, ModelCheckpoint), `.venv/bin/python` + `MPLBACKEND=Agg` + `CUDA_VISIBLE_DEVICES=1` compatible.

**New — tests:**
12. `tests/test_layers/test_adaln_zero.py` — AdaLN-zero init identity + serialization round-trip.
13. `tests/test_models/test_lewm.py` — forward-pass + serialization round-trip + shape assertions.

**Total: 13 new files, 0 modifications to existing files.** (Existing ViT + Transformer primitives are reused unchanged.)

## Steps

**Step 1 — checkpoint + scaffolding** `[RISK: low] [deps: —]` `[x] commit 866eb82`
- Create `cp-000-iter1.md` with current commit hash (8493641).
- Create empty `models/lewm/__init__.py`, `train/lewm/__init__.py`.
- Commit: `[iter-1/step-1] Scaffold lewm model + training package directories`.

**Step 2 — `AdaLNZeroConditionalBlock` layer** `[RISK: medium] [deps: 1]` `[x] commit 0db2b16`
- Write `src/dl_techniques/layers/adaln_zero.py`: Dense(6·dim, zero-init) → chunk(6) → pre-LN (no affine) → modulate → MHA (causal) → gated residual → pre-LN → modulate → MLP → gated residual.
- Reuse `keras.layers.MultiHeadAttention` with `use_causal_mask=True` in call.
- Full `get_config`, `compute_output_shape`.
- Test: `tests/test_layers/test_adaln_zero.py` — build, forward with random x and c, verify shapes; verify at init (after build) that forward(x, c) ≈ x within `atol=1e-6` (identity property of zero-init AdaLN-zero); save/load round-trip.

**Step 3 — `SIGRegLayer`** `[RISK: medium] [deps: 1]` `[x] commit 9f15dfc`
- Write `src/dl_techniques/regularizers/sigreg.py`: store `t`, `phi`, `weights` as non-trainable buffers via `self.add_weight(trainable=False)`; on call `(proj: (T, B, D))` sample random projection matrix via `keras.random.normal`, normalize columns, compute characteristic-function residual against Gaussian target, return scalar.
- Test: forward pass on random (3, 8, 192) tensor returns a scalar; output is finite.

**Step 4 — `ActionEmbedder` + `MLPProjector`** `[RISK: low] [deps: 1]` `[x] commit 8812989`
- `src/dl_techniques/models/lewm/embedder.py`: Conv1D(filters=smoothed_dim, k=1) → (time axis handled via transpose) → Dense(mlp_scale·emb) → SiLU → Dense(emb). Input `(B, T, A)`.
- `src/dl_techniques/models/lewm/projector.py`: Dense → BatchNormalization → GELU → Dense.
- Both with `get_config` + `compute_output_shape`.
- Quick sanity forward pass inline in training script test; unit tests deferred (covered by model test).

**Step 5 — `ARPredictor`** `[RISK: medium] [deps: 2, 4]` `[x] commit 7c1cfeb`
- `src/dl_techniques/models/lewm/predictor.py`: learned pos embedding `add_weight(shape=(1, num_frames, input_dim))`, input_proj, cond_proj, stack of `AdaLNZeroConditionalBlock` × depth, final LayerNorm, output_proj.
- Forward `(x, c)` with causal attention; slice pos embedding to `x.shape[1]`.
- Test: within model test — forward `(B=2, T=3, D=192)` → output `(B=2, T=3, D=192)`.

**Step 6 — `LeWM` keras.Model** `[RISK: high] [deps: 2, 3, 4, 5]` `[x] commit 67d4c61`
- `src/dl_techniques/models/lewm/config.py`: dataclass matching upstream config (img_size=224, patch_size=14, encoder_scale='tiny', embed_dim=192, history_size=3, num_preds=1, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, sigreg_weight=0.09, sigreg_knots=17, sigreg_num_proj=1024, action_dim=2, smoothed_dim=10, mlp_scale=4).
- `src/dl_techniques/models/lewm/model.py`:
  - `__init__`: build ViT encoder via existing `create_vit` (or direct `ViT(include_top=False, pooling='cls', scale='tiny', patch_size=14, input_shape=(224,224,3))`). Build `MLPProjector`, `ActionEmbedder`, `ARPredictor`, `MLPProjector` (pred_proj), `SIGRegLayer`.
  - `encode(pixels)`: reshape `(B, T, H, W, C)` → `(B·T, H, W, C)`, forward through encoder → `(B·T, 192)`, projector → reshape to `(B, T, D)`.
  - `predict(emb, act_emb)`: predictor(emb, act_emb) → `(B, T, D)` → flatten time → pred_proj → reshape.
  - `call(inputs, training=None)`: inputs = dict `{pixels, action}`. Compute context emb, target emb (via `stop_gradient` or NOT — upstream uses live encoder for both, target is NOT stop-grad — a key LeWM choice vs EMA-based JEPAs). Produce `pred_emb`, compute MSE against target[:, 1:], add SIGReg loss via `self.add_loss(...)`, return pred_emb.
  - `rollout(pixels_history, action_sequence)`: replicate upstream AR rollout logic with `keras.ops.concatenate` instead of `torch.cat`. History_size truncation.
  - `get_config`: emit `config.to_dict()`. `from_config`: reconstruct via dataclass.
- **DECISION D-001 candidate**: target uses live encoder (no EMA, no stop-gradient in upstream). Anchor inline comment.

**Step 7 — synthetic + HDF5 dataset module** `[RISK: low] [deps: —]` `[x] commit 02887bf`
- `src/dl_techniques/datasets/pusht_hdf5.py`:
  - `SyntheticLeWMDataset(num_episodes, num_steps, img_size, action_dim)` — `tf.data.Dataset` yielding `({pixels, action}, pred_target_placeholder)` batches of shape `((B, T, 224, 224, 3), (B, T-1, action_dim))`. Pixels = random uint8 → normalized. Useful for smoke test; num_episodes small (e.g. 64).
  - `PushTHDF5Dataset(h5_path, keys=['pixels', 'action', 'proprio'], history_size, num_preds, frameskip, normalizer_stats)` — skeleton with `h5py` load + per-episode slicing to (history + num_preds) windows; ImageNet normalization; NaN→0 replacement for actions. Fully functional but **not tested on real data** (noted).
  - Both produce the same batch schema so the training script is dataset-agnostic.

**Step 8 — `train_lewm.py` smoke-test training script** `[RISK: medium] [deps: 6, 7]` `[x] commit 754e0dc`
- Follow `src/train/vit/train_vit.py` / `cliffordnet/` pattern: argparse (`--synthetic | --hdf5-path`, `--batch-size=4`, `--epochs=2`, `--steps-per-epoch=3`, `--seed`), `utils.logger`, config dict, build model from `LeWMConfig`, `model.compile(optimizer=AdamW(lr=5e-5, wd=1e-3), jit_compile=False)` (jit off to avoid tracing issues with add_loss), `CSVLogger` + `TerminateOnNaN` + `ModelCheckpoint`, results directory under `src/results/lewm/<timestamp>_<tag>/`.
- No custom training loop — standard `model.fit(dataset, ...)`. Loss comes from `add_loss` calls inside `call`.

**Step 9 — unit tests for model** `[RISK: low] [deps: 6]` `[x] commit 36fca55`
- `tests/test_models/test_lewm.py`:
  - `test_forward_pass_shapes` — `(B=2, T=3, 224, 224, 3)` + `(B=2, T-1, 2)` → predicts, loss is finite.
  - `test_serialization_round_trip` — save/load, assert identical outputs on same input (`atol=1e-5`).
  - `test_rollout_shape` — `(B=2, S=2, T=5, 224, 224, 3)` + `(B=2, S=2, T=5, 2)` → `predicted_emb: (B=2, S=2, T=5, 192)`.
  - `test_adaln_identity_at_init` — at init, LeWM predictor should be identity-ish (inherited property).

**Step 10 — smoke-test execution + REFLECT** `[RISK: low] [deps: 8, 9]`
- Run `.venv/bin/python -m pytest tests/test_layers/test_adaln_zero.py tests/test_models/test_lewm.py -vvv` on CPU.
- Run `MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lewm.train_lewm --synthetic --batch-size=2 --epochs=1 --steps-per-epoch=2`.
- Record outputs in `verification.md`.

## Assumptions

| # | Assumption | Grounding finding | Steps affected |
|---|------------|-------------------|----------------|
| A1 | ViT model at `models/vit/model.py` with `include_top=False, pooling='cls', scale='tiny', patch_size=14` returns `(B, 192)` per frame and is serializable | Finding 4 | 6 |
| A2 | `keras.layers.MultiHeadAttention(use_causal_mask=True)` matches `F.scaled_dot_product_attention(is_causal=True)` semantics for decoder self-attention | Keras docs; Finding 3 | 2, 5, 6 |
| A3 | `add_loss` inside `call` works with `model.fit` and is included in reported loss | Keras 3 docs | 6, 8 |
| A4 | LeWM does NOT use EMA target encoder (target = live encoder output); gradient flows through both paths | `/tmp/lewm_source/jepa.py` lines 29–45 — no EMA, no `.detach()` on emb | 6 |
| A5 | Action time axis can be padded to match pixels time axis by appending a zero action (standard JEPA loader practice) | Upstream train.py preprocessing (not shown in detail; inferred from shapes) | 7, 8 |
| A6 | BatchNormalization inside projector is safe under `model.fit` (updates moving stats correctly in training mode) | Keras docs | 4, 6 |
| A7 | Smoke test with B=2, T=3 on RTX 4070 (12 GB) fits comfortably; jit_compile=False avoids XLA issues with dynamic shapes | Our convention | 8, 10 |

## Failure Modes

| Dependency | Slow | Garbage | Down | Blast radius |
|------------|------|---------|------|--------------|
| Existing `ViT` model | ViT forward on 224×224 slow on CPU (expected, ~1s/batch) — acceptable for smoke | ViT outputs NaN if patch_size mismatch not caught → loss NaN | ViT import fails → crash at import; fallback: write thin ViT-tiny inline | M6, M8 |
| `keras.layers.MultiHeadAttention` | Fast | Wrong causal mask → future leak → too-easy loss | Unlikely | S2, S5 |
| `h5py` for HDF5 loader | Slow on cold cache | — | Not installed → smoke path unaffected (synthetic only); import inside class to defer | S7 |
| `keras.random.normal` in SIGReg | Non-deterministic across forward passes (expected — upstream does the same) | Tiny projection → noisy loss; default 1024 projections is upstream default | — | S3 |

## Pre-Mortem & Falsification Signals

**Scenario 1 — AdaLN-zero is not identity at init.** If `test_adaln_identity_at_init` fails with max abs diff > 1e-4, our zero-init is wrong (e.g., bias not zeroed, or `modulate` formula off). **STOP IF** — pause after Step 2, re-read `/tmp/lewm_source/module.py:85-91`, fix before proceeding.

**Scenario 2 — Loss is NaN on first step.** Likely culprits: SIGReg numerical issue (log of 0?), BatchNorm with batch-of-1 in projector, dtype mismatch in modulation. **STOP IF** — `train_lewm.py` NaN on first batch with `TerminateOnNaN`. Revert to checkpoint, bisect: run with sigreg_weight=0 first; if pass, SIGReg numeric; else arch.

**Scenario 3 — Serialization round-trip fails.** Almost always a missing `get_config` key or a layer tracked via plain dict. **STOP IF** — `test_serialization_round_trip` diff > 1e-5. Check every custom layer's `get_config` / `from_config`; check sublayer registration uses `setattr`.

## Success Criteria

| # | Criterion | Measurable outcome |
|---|-----------|---------------------|
| C1 | `AdaLNZeroConditionalBlock` is identity at init | `max(abs(block(x, c) - x))) < 1e-5` on random inputs |
| C2 | `LeWM` forward pass produces finite loss on synthetic batch | `loss.numpy()` is finite; `pred_emb.shape == (B, T, 192)` |
| C3 | Serialization round-trip is lossless | `model.save/load`; max abs diff on same-input predict `< 1e-5` |
| C4 | Rollout shape is correct | `model.rollout(pixels=(2,1,3,224,224,3), actions=(2,1,5,2))['predicted_emb'].shape == (2, 1, 5, 192)` |
| C5 | `train_lewm.py --synthetic` runs 2 epochs × 3 steps without NaN / error | CSV log shows 2 rows; exit code 0 |
| C6 | All unit tests pass | `pytest tests/test_layers/test_adaln_zero.py tests/test_models/test_lewm.py` → all green |

## Verification Strategy

| Criterion | Method | Command |
|-----------|--------|---------|
| C1 | Unit test | `.venv/bin/python -m pytest tests/test_layers/test_adaln_zero.py::TestAdaLNZero::test_identity_at_init -vvv` |
| C2, C3, C4 | Unit tests | `.venv/bin/python -m pytest tests/test_models/test_lewm.py -vvv` |
| C5 | Smoke run | `MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lewm.train_lewm --synthetic --batch-size=2 --epochs=1 --steps-per-epoch=2` (exit 0) |
| C6 | Aggregate | `.venv/bin/python -m pytest tests/test_layers/test_adaln_zero.py tests/test_models/test_lewm.py -vvv` |

## Complexity Budget
- Files added: 13/15 max (slightly over default 3 — this IS a model port; 13 files is minimal for a new model package + tests + trainer + dataset).
- New abstractions: 6 / 6 max (AdaLNZeroBlock, SIGRegLayer, ActionEmbedder, MLPProjector, ARPredictor, LeWM — each maps 1:1 to an upstream class, no incidental abstractions).
- Lines added vs removed: net +~1200 lines added (large but unavoidable for a full model port). **No modifications to existing files.**

## Out of Scope (follow-up plans)
- `eval.py` / MPC / planning (depends on `stable_worldmodel` — separate integration).
- Real PushT HDF5 data (no local data to validate loader against).
- Multi-GPU / bf16 training.
- LR schedule (upstream uses default AdamW; we match).
- EMA target encoder variant (upstream does not use; noted for future ablation).
