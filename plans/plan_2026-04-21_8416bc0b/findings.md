# Findings
*Summary and index of all findings. Detailed files go in findings/ directory.*

*Cross-plan context: see plans/FINDINGS.md, plans/DECISIONS.md, and plans/LESSONS.md*

## Index
1. **lewm-source-schema.md** — (seed, user-provided) PyTorch LeWM architecture at `/tmp/lewm_source/`. JEPA top-level, ARPredictor (6-deep ConditionalBlock stack), Embedder (Conv1d + 2-layer MLP), MLP projector/pred_proj, SIGReg regularizer, ViT-tiny encoder (patch=14, img=224). Config: embed_dim=192, history_size=3, num_preds=1, sigreg.weight=0.09, knots=17, num_proj=1024. Loss: `pred_loss + 0.09 * sigreg_loss`.
2. **framework-conventions.md** (inline, below) — dl_techniques conventions: `@keras.saving.register_keras_serializable()`, `keras.ops`, `utils.logger`, factory-driven construction, explicit-create-vs-build rule, `compute_output_shape()` required, Dict config serialization. Models live under `src/dl_techniques/models/<name>/` with one model subdirectory per architecture. Training scripts under `src/train/<name>/train_<name>.py` using `train.common` utilities. Key guide: `research/2026_keras_custom_models_instructions.md`.
3. **reusable-layers.md** (inline, below) — Primitives available; what's missing. `TransformerLayer` (pre/post LN, factory-driven attention+FFN) at `layers/transformers/transformer.py`. Attention factory + `multi_head_attention`. `MLP` FFN at `layers/ffn/mlp.py`. Norms factory (`layer_norm`, `rms_norm`, …) at `layers/norms/factory.py`. Embeddings factory incl. `patch_2d` + `positional_learned` at `layers/embedding/factory.py`. **FiLM** at `layers/film.py` — closest to AdaLN but scale/shift only (6-way modulation would need extension). **No AdaLN-zero layer exists** → write new `AdaLNZeroConditionalBlock` layer. **No Conv1d-based action embedder primitive** → write as small custom layer or inline in model. **SIGReg** has no analogue → write new `layers/regularizers/...` or inline.
4. **existing-vit-and-training.md** (inline, below) — `src/dl_techniques/models/vit/model.py` — fully featured ViT (`include_top`, `pooling`, patch+pos embedding, CLS token, factory). `SCALE_CONFIGS["tiny"] = (192, 3, 12, 4.0)` — but LeWM uses `patch_size=14`, `img_size=224` which gives `num_patches = 256`, `max_seq_len = 257`. ViT encoder with `include_top=False, pooling='cls'` returns `(B, 192)` — perfect for our `encode()` body. **No `interpolate_pos_encoding` needed** because we control the input size. Training pattern: Pattern 2 (synthetic data) with `create_callbacks(monitor='val_loss', include_terminate_on_nan=True, use_lr_schedule=False)`. Existing `jepa/` model directory implements I-JEPA/V-JEPA (masked prediction), not our LeWM (action-conditioned). Must use distinct name: **`lewm/`** (matches upstream name, distinguishes from masked-JEPA).

## Key Constraints

### Hard (from user message + LESSONS)
- **Smoke test only.** Synthetic dataset generator + a few batches. No long training runs on real data.
- **GPU policy.** `CUDA_VISIBLE_DEVICES=1` (RTX 4070, 12 GB). Never run GPU jobs in parallel. Use `MPLBACKEND=Agg`.
- **Keras 3.8 serialization.** `@keras.saving.register_keras_serializable()` + complete `get_config()` on every custom layer/model. Sublayers tracked via `setattr`, never inside a dict.
- **Keras 3.8 weight-load trap.** `load_weights(path.keras, by_name=True)` fails — noted for completeness (no transfer needed here).
- **No `stable_worldmodel` / `stable_pretraining` dependency** — must be reimplemented with tf.data + keras + numpy.
- **No HDF5 dataset available.** Design loader against the PyTorch schema; provide synthetic generator for smoke test.

### Soft
- Python 3.11+, TF 2.18, Keras >= 3.8.
- Single-file models preferred per dl_techniques convention (config-driven factory functions).
- `dl_techniques.utils.logger` — no print statements.
- Commit cadence: `[iter-N/step-M] desc`. User pushes.

### Out of scope (follow-up)
- `eval.py` / MPC / planning (depends on stable_worldmodel).
- Real PushT HDF5 data loading.
- Multi-GPU / bf16 training infra.

## User-provided architecture summary (from /tmp/lewm_source/)

- **JEPA** top-level: `encode` / `predict` / `rollout` / `criterion` / `get_cost`. Encode: pixels (B,T,C,H,W) → flatten time, run ViT encoder, take CLS token, project to embed_dim. Training loss: `(pred_emb - tgt_emb).pow(2).mean() + λ·sigreg(emb.transpose(0,1))`.
- **ARPredictor**: learned pos embedding (1, num_frames, input_dim) + Transformer with `ConditionalBlock` (AdaLN-zero). Forward (x, c): x=context emb (B, T=3, D=192), c=action emb (B, T=3, D=192). Causal self-attn.
- **ConditionalBlock**: LayerNorm (no affine) + modulate(shift,scale) + self-attn + gated residual. AdaLN module = `SiLU → Linear(dim → 6*dim)`, zero-init.
- **Embedder**: `Conv1d(action_dim, smoothed_dim, k=1) → permute → 2-layer MLP (SiLU)`. Outputs (B, T, emb_dim).
- **MLP projector / pred_proj**: `Linear → BatchNorm1d → GELU → Linear`. BN1d over embeddings.
- **SIGReg**: sketch-based isotropic-Gaussian regularizer. Random projections (D, num_proj) each forward; computes characteristic-function residual against Gaussian target.
- **Encoder**: ViT-tiny patch=14, img=224, CLS token, `interpolate_pos_encoding=True` behavior.
- **Config**: embed_dim=192, history_size=3, num_preds=1, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1, sigreg.weight=0.09.
