# Cosmos 3 Omni-Model — Integration Research

**Papers:** NVIDIA Cosmos Team, *Cosmos 3: Omnimodal World Models for Physical AI*, arXiv:2606.02800v4 [cs.CV] 23 Jun 2026 (CC BY 4.0). Glossary: `nvidia.com/en-us/glossary/omni-model/`; Cookbook: `nvidia-cosmos.github.io/cosmos-cookbook/`.
**Code:** `github.com/NVIDIA/cosmos` (inference/eval/cookbooks) + `github.com/NVIDIA/cosmos-framework` (SFT/LoRA/RL/distill training) + `huggingface/diffusers:.../transformer_cosmos3.py` + `transformers:.../cosmos3_omni/modeling_cosmos3_omni.py` + `vllm-project/vllm-omni:.../cosmos3/transformer_cosmos3.py`.
**Weights:** `huggingface.co/collections/nvidia/cosmos3` — Cosmos3-Edge 4B, Nano 16B, Super 64B (+ post-trained `Super-Text2Image[-4Step]`, `Super-Image2Video[-4Step]`, `Nano-Policy-DROID`, `Edge-Policy-DROID`). License **OpenMDW-1.1** (not Apache-2.0).
**Status:** research-only, no code written. Target: new model family properly integrated into `dl_techniques`.
**Date:** 2026-10-15. **Author:** integration investigation (no implementation).

> House rules that govern everything below: `research/2026_keras_custom_models_instructions_v2.md` (mandatory authoring guide),
> `src/dl_techniques/models/AGENTS.md` (authoring contract), `src/dl_techniques/models/GUIDE.md` (measured defects),
> `REPO_MAP.md` (model/trainer/test triangle), per-package `AGENTS.md` files.

---

## 1. What Cosmos 3 is in one page

Cosmos 3 is the first NVIDIA **omni-model** for physical AI: one **Mixture-of-Transformers (MoT)** backbone that jointly consumes and produces **language + image + video + audio + action**. It subsumes four previously separate lines:

* VLM reasoner (cf. Cosmos-Reason 7B) — spatial/temporal grounding, CoT, planning;
* video generator / world simulator (cf. Cosmos-Predict diffusion, Cosmos-Transfer ControlNet);
* world-action / policy model (forward dynamics, inverse dynamics, visuomotor policy);
* audio-visual generator (video + synchronized sound).

Three claims structure the report:

1. **Dual-tower MoT, not MoE.** Every decoder layer carries *two full parameter sets* (reasoner tower for the AR prefix, generator tower for the diffusion suffix), both init from a pretrained VLM (Qwen3-VL 8B for Nano, 32B for Super; Edge trains its 2B LLM from scratch). Towers meet only at a shared attention operator with asymmetric masking (§4).
2. **AR + diffusion in one forward pass.** AR subsequence (text + optional ViT vision tokens) is causal next-token; diffusion subsequence (VAE video/image + audio + action continuous tokens) is flow-matching denoised conditioned on AR. Reasoner runs alone; generator always runs both towers.
3. **Absolute-time 3D MRoPE.** Temporal/height/width rotary split extended with physical-time modulation so video (latent fps), audio (25 tok/s), action (per-step) align on one temporal axis despite different rates.

Scope for us: **phase 1 = reasoner tower + text/image/video understanding + tiny MoT block + 3D MRoPE**. Defer audio VAE, action tokenizer/policy heads, full diffusion generator, distillation/RL post-training to phase 2/3. A 16–64B faithful port is not trainable/testable here; ship a `tiny` smoke variant plus `nano/super` config rows.

---

## 2. Encoders and tokenization (what to implement verbatim)

All modalities project into one hidden space (`hidden_size` 2048/4096/5120). Each **non-language** token gets a learnable modality-embedding vector added before the backbone (§2, Fig. 5).

### 2.1 Image and video — two encoders, different roles

```
Understand path (jointly trained with backbone):
  ViT, patch 16x16 -> 2-layer MLP merging 2x2 tokens -> hidden dim
  + DeepStack aggregation + text timestamp tokens interleaved with frames

Generate path (frozen during training):
  Wan2.2-TI2V-5B video VAE encoder
  temporal 4x, spatial 32x32 (= 16x16 VAE + 2x2 patch merge)
  linear proj per VAE token -> hidden dim
```

Shapes: video tokens vary `(t,h,w)`; image = single-frame video, varies `(h,w)` only. Spatial/temporal indices reset to zero at each vision segment start (§2.4).

### 2.2 Audio

```
Lee et al. 2025b audio VAE, stereo 48 kHz, hop 1920 samples
  -> 25 tokens/s, continuous latents, linear proj -> hidden dim
  position: t-only, h = w = 0; temporal index advances per audio hop
```

### 2.3 Action

Dedicated action-token class bridging language reasoning and control signals:

```
Action representations: per-embodiment continuous control chunks
  (robot joints / hand pose / camera motion / AV trajectory)
Action tokenization: linear proj of action chunk -> hidden dim
  + embodiment domain id (num_embodiment_domains = 32 in diffusers config)
  position: t-only, h = w = 0; temporal index advances per sampling step
```

Forward: `(obs, action) -> future video`. Inverse: `video -> action`. Policy: `(obs, task text) -> action sequence`. All three share the generator tower; only conditioning/target split in DM changes.

---

## 3. Token arrangement and generation modes (§2.2)

One joint sequence `z = [AR ; DM]`:

```
AR (reasoner, discrete): [text tokens, optional ViT vision tokens, ..., <BoG>]
DM (generator, continuous): [optional clean conditioning | noisy targets]
  vision (VAE) + audio + action tokens, noise-perturbed during training
```

Undifferentiated packed layout is the point: the *same* backbone supports the I/O table by changing which DM slots are clean vs noisy:

| Input | Output | Application |
|---|---|---|
| Text | Image | T2I generation |
| Text | Video (+Audio) | world / edge-case video synthesis |
| Text+Image | Video | prediction / I2V |
| Text+Image+Video | Text | VLM reasoning / grounding / planning |
| Action+Video | Text+Video | action-conditioned world model (forward dynamics) |
| Video | Text+Video+Action | world-action / policy model |

`und_len` (diffusers `forward(..., und_len, ...)`) marks the AR/DM boundary. `vision_mse_loss_indexes / sound_mse_loss_indexes / action_mse_loss_indexes` select which DM positions contribute to the velocity loss; `*_timesteps` carry per-patch diffusion times; `*_noisy_frame_indexes` distinguish clean conditioning from noisy targets.

---

## 4. MoT architecture (§2.3) — the core new block

Per decoder layer: **two independent transformer layers** (own LayerNorms, QKV/O projections, FFNs), all co-initialized from the pretrained VLM. AR tokens route to reasoner tower, DM tokens to generator tower. They interact only in attention.

### 4.1 Dual-stream joint attention (Eq. 7–8)

```
Q_AR, K_AR, V_AR  = reasoner-tower projections of AR tokens
Q_DM, K_DM, V_DM  = generator-tower projections of DM tokens

O_AR = Attn_causal(Q_AR, K_AR, V_AR)                          # Eq. 7
O_DM = Attn_full(Q_DM, [K_AR; K_DM], [V_AR; V_DM])            # Eq. 8
       [.;.] = concat along sequence dim
```

AR is causally self-contained (preserves VLM text ability); DM cross-attends to everything (AR context + intra-DM consistency). AR is never updated from DM.

Reference implementations (read before designing ours):

* `huggingface/diffusers:.../transformer_cosmos3.py::Cosmos3AttnProcessor` — per-pathway QKV, per-head QK RMSNorm, separate RoPE cos/sin halves, causal vs full dispatch, separate out-projections.
* `vllm-project/vllm-omni:.../cosmos3/transformer_cosmos3.py::Cosmos3CrossAttention` — non-SP explicit `cat([k_und,k_gen])` vs SP Ulysses joint_key path.
* `huggingface/transformers:.../cosmos3_omni/modeling_cosmos3_omni.py` — HF reasoner-side RoPE index builder.

QK normalization: per-head RMSNorm on Q and K in both pathways (`qk_norm_for_text=True`, `rms_norm_eps=1e-6`, Nemotron variant in some configs). GQA with 8 KV heads (Nano 32Q/8KV, Super 64Q/8KV, head_dim 128).

Outputs: AR head = next-token logits (vocab 151936); DM heads = denoised latents (`proj_out` vision, `audio_proj_out` sound, action proj), trained with flow-matching velocity, shown as clean target in Fig. 5 for clarity.

---

## 5. Multimodal position embedding (§2.4)

3D MRoPE (Qwen3-VL lineage, Bai et al. 2025a) with **absolute temporal modulation**.

Base: each attention head dim splits into `(t, h, w)` rotary bands. Video tokens advance `t` per latent frame and tile `(h,w)` over `(0..H-1, 0..W-1)` per frame. Image = single frame. Audio/action carry `t` only (`h=w=0`). Indices reset per vision segment; transfer-task control + generation tokens both start from last-AR offset (Fig. 6).

Extension: discrete token index is insufficient when modalities generate simultaneously at different rates. Associate unit temporal length with `TPS_base`; for a DM stream with its own tokens-per-second `TPS`:

```
dt = TPS_base / TPS                                            # Eq. 9
```

Example: `base_fps=24` video vs `sound_latent_fps=25.0` audio vs per-step action — each advances its rotary `t` by its own `dt`, aligning all streams to absolute physical time. AR/DM margin `unified_3d_mrope_temporal_modality_margin=15000` separates the subsequences; `rope_theta=5000000.0`; `rope_axes_dim` / `mrope_section` (e.g. Ideogram4 `(24,20,20)`) splits head dim across axes.

---

## 6. Model variants (§2.5 + diffusers config)

| Variant | Backbone | Layers | Hidden | Heads (Q/KV) | Head dim | FFN | Total | Runs on |
|---|---|---|---|---|---|---|---|---|
| Edge | dense 2B from scratch (Qwen3-1.7B-like, no QK-norm, ReLU², 9216 FFN) | 28 | 2048 | 16 / 8 | 128 | 9216 | ~4B | Jetson Orin/Thor, RTX Pro 6000 |
| Nano | Qwen3-VL 8B | 36 | 4096 | 32 / 8 | 128 | 12288 | ~16B | RTX Pro 6000 / H100 / B200 |
| Super | Qwen3-VL 32B | 64 | 5120 | 64 / 8 | 128 | 25600 | ~64B | H200 / B200 / GB200 |

Diffusers `Cosmos3OmniTransformer` extras: `vocab_size=151936`, `hidden_act=silu`, `latent_channel=48`, `latent_patch_size=2`, `patch_latent_dim=192`, `timestep_scale=0.001`, `enable_fps_modulation=True`, `sound_dim/action_dim + sound_gen/action_gen` flags, `num_embodiment_domains=32`, `intermediate_size=12288` (Nano).

Post-trained checkpoints (same arch, different heads/schedules): `Super-Text2Image[-4Step]`, `Super-Image2Video[-4Step]` (distilled 17–25x faster), `Nano-Policy-DROID`, `Edge-Policy-DROID`.

---

## 7. Training contracts (§4)

### 7.1 Reasoner (§4.1)

Standard VLM pipeline on image/video+text: pretrain (semantic dedup + AI-judge filtering + thresholded mix) then SFT (spatial, temporal, AV, robotics, smart-infra splits). Loss: causal cross-entropy on text tokens. ViT encoder trained jointly.

### 7.2 Generator (§4.2) — flow matching

```
x0 = clean latent, x1 ~ N(0,I) (t=0 data, t=1 noise per Ideogram4 convention)
x_t = (1-t)*x0 + t*x1,  v_target = x1 - x0
t ~ logit-normal (P_MEAN/P_STD per bridge; scheduler logit-normal warp + clamp)
L = mean((v_pred(x_t, t, AR_ctx) - v_target)^2) over loss-indexed DM tokens
```

Multi-resolution training, FPS modulation, per-patch timesteps (`vision_timesteps`, `t*1000` scale in SD3-style embeds, `timestep_scale=0.001` in Cosmos3). Mid-training: domain specialization (drive/robot/warehouse/human) + multimodal integration (video+audio+action joint). Post-training: T2I, I2V, robot policy (DROID) recipes in `cosmos-framework`.

Transfer generation uses **two-weight CFG** (separate guidance for text vs control). DMD2-style 4-step distillation for the `-4Step` checkpoints.

---

## 8. Inference contracts (§5–§6.3)

Two surfaces, one checkpoint:

* **Reasoner:** text out. Backends: Transformers, vLLM, TensorRT-LLM Reasoner, NIM `cosmos3-reasoner`. Needs `mm_token_type_ids` (0=text,1=image,2=video) + `image/video_grid_thw` for MRoPE.
* **Generator:** vision/sound/action out. Backends: Diffusers `Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano")`, vLLM-Omni (`--omni --model-class-name Cosmos3OmniDiffusers...`), TensorRT-LLM Generator, NIM Generator. Guardrail `nvidia/Cosmos-1.0-Guardrail` gated (HF access required, loads at server start). `Cosmos3OmniPipeline(prompt=...).video` then `export_to_video(..., fps=24)`.

Sampling hyperparams (§6.3): media specs (resolution/fps/duration grids), negative prompt (App. B.6), Euler steps, CFG scales, reasoner-as-prompt-upsampler templates (App. B). Reasoner tower KV-cacheable; generator caches AR prefix across denoise steps.

---

## 9. Empirical anchors (what "working" means)

Per report at write time: #1 open T2I + I2V (Artificial Analysis), #1 open robot policy (RoboArena), #1 open reasoner on robotics/smart-space/driving averages. Post-trained demos only — base checkpoints are the product; `-4Step`/DROID are capability demos. Our port claims nothing until tiny-variant smoke + value goldens pass (docs guard `test_docs_make_no_unearned_performance_claims.py` forbids crediting this code with paper numbers).

---

## 10. Codebase reuse inventory (verified paths)

### 10.1 Model templates — copy these, do not re-derive

| Asset | Path | Reuse for Cosmos-Omni |
|---|---|---|
| Dual-stream block shape | `models/vision_language/sd3_mmdit/transformer.py:SD3MMDiT`, `blocks.py:MMDiTBlock/MMDiTFinalLayer`, `config.py:SD3MMDiTConfig`, `scheduler.py:FlowMatchEulerScheduler`, `pipeline.py:SD3Pipeline` | Dual-stream AdaLN modulation + joint-attention wiring; `full/tiny` variant pattern (`depth24/dim1536` vs `depth4/dim192`); Euler sampler + `x_t=(1-t)x0+t*noise`, `v=noise-x0` |
| Packed multimodal sequence | `models/vision_language/ideogram4/transformer.py:Ideogram4Transformer`, `config.py:Ideogram4Config`, `vae.py:AutoEncoder`, `scheduler.py:LogitNormalSchedule`, `pipeline.py:Ideogram4Pipeline` | Packed `dict(llm_features,x,t,position_ids(B,L,3),segment_ids,indicator)` + block-diagonal mask + indicator gating = closest to MoT `[AR;DM]` packing; `mrope_section`, `rope_theta=5M`; asymmetric CFG blend; VAE reference |
| Bridge/diffusion math | `models/vision_language/bit_diffusion/{model.py:DiTXA,blocks.py:DiTXABlock,sde.py:BridgeSDE,bridge_process.py,token_bridge.py,config.py:BridgeConfig}` | Bidirectional conditioning, token<->latent packing, `TIME_EPS` clamping, never-narrow dtype, SDE abstraction if audio/action bridges added later |
| Canonical DiT + sampler | `models/vision/dit/{model.py:DiT,blocks.py:DiTBlock/DiTFinalLayer,config.py:DIT_VARIANTS,diffusion.py:GaussianDiffusion}` | adaLN-Zero init, CFG batch-duplication, DDPM/DDIM loops with explicit seed, `utils/ddpm_schedule.py:DDPMSchedule` precedent for a `CosmosSchedule` value-object in `utils/` |
| Contrastive alignment | `models/vision_language/clip/model.py:CLIP`, `mobile_clip/*`, `losses/{clip_contrastive_loss.py,siglip_contrastive_loss.py,infonce_loss.py}` | Text-video-audio alignment if staged pretraining is added |
| Prefix VLM fusion | `models/vision_language/nano_vlm/model.py:NanoVLM`, `losses/nano_vlm_loss.py`, `layers/modality_projection.py`, `layers/transformers/{vision_encoder.py,text_encoder.py,text_decoder.py}` | Vision-prefix + single causal decoder = closest to AR reasoner tower; modality projection + tied vocab head |
| Video/latent | `models/vision/{video_jepa/{model.py:VideoJEPA,encoder.py,predictor.py,masking.py:TubeMaskGenerator},levjepa/,vae/model.py:VAE,vq_vae/model.py:VQVAEModel}`, `models/vision_language/sd3_mmdit/vae.py` | Tube masking, EMA target, predictor for video SSL; VAE/VQ-VAE + sampling/quantizer layers for tokenizer design |

### 10.2 Backbone pieces — factory-first (`layers/AGENTS.md` § Layer Reuse Policy)

| Domain | Entry point | Keys / classes to use |
|---|---|---|
| Attention | `layers/attention/factory.py::create_attention_layer` (`ATTENTION_REGISTRY`, 35 keys) | `group_query` (has `rope_percentage/rope_theta/max_seq_len`), `multi_head_latent` (MLA + decoupled RoPE), `multi_head`, `multi_head_cross`; joint attention via `layers/attention/mmdit_joint_attention.py::MMDiTJointAttention` (direct import, no key) |
| Norms | `layers/norms/factory.py::create_normalization_layer` (19 types) | `rms_norm`, `layer_norm`; pass `epsilon=` explicitly (factory forces `1e-6` vs Keras `1e-3` — 1000x hazard); per-head QK RMSNorm mirrors `Cosmos3NemotronRMSNorm` |
| FFN | `layers/ffn/factory.py::create_ffn_layer` (`FFN_REGISTRY`, 22 keys) | `swiglu` (LLaMA-style, Nano/Super FFN), `gelu_tanh` (SD3-style), `geglu`; MoE via `layers/moe/*::MixtureOfExperts` + `TransformerLayer(moe_config=...)` |
| Embeddings | `layers/embedding/factory.py::create_embedding_layer` (`EMBEDDING_REGISTRY`, 15 keys) | `timestep` (DiT cos-first Dense-SiLU-Dense), `scalar_sinusoidal`, `class_label` (CFG null row), `patch_2d`/`patch_1d`, `rope`/`dual_rope`/`continuous_rope`, `mrope_ideogram4::Ideogram4MRoPE` (3D t/h/w — starting point for Cosmos MRoPE) |
| Transformer | `layers/transformers/transformer.py::TransformerLayer` (direct import, no factory) + `vision_encoder.py::VisionEncoder`, `text_encoder.py::TextEncoder`, `patch_causal_transformer.py`, `causal_stack.py` | Composes attention/ffn/norm factories; `hidden_size/num_heads/n_kv_head/ffn_type/normalization_position/pre-post/stochastic-depth/layer-scale` cover AR tower; generator tower needs new MoT wrapper around two such stacks |
| AdaLN/modulation | `layers/transformers/{adaln_zero.py::AdaLNZeroConditionalBlock,sd3_adaln.py::AdaLayerNormZero/ZeroX/Continuous + modulate,ideogram4_block.py::Ideogram4TransformerBlock/FinalLayer}` | 6-way (shift/scale/gate x attn/mlp) and 4-way scale+gate variants; `t*1000` scale convention; identity-at-init zeros |
| Positional extras | `layers/embedding/{video_rope.py::VideoRoPE3D,axial_rope_2d.py::AxialRoPE2D,patch_embed_3d.py::PatchEmbed3D,sincos_pos_embed_2d/3d.py,mask_token.py,register_tokens.py,class_token.py}` (direct import) | 3D rope + tubelet patchify + frozen sincos tables for vision path |
| VAE/quantize | `layers/generative/{sampling.py::Sampling,vector_quantizer.py::VectorQuantizer,vector_quantizer_rotation_trick.py}` + `layers/tokenizers/{bpe.py,ascii_char.py}` | Reparam samplers, codebooks; text via existing tokenizers, action via `patch_1d`/codebook (new) |
| Fusion | `layers/fusion/multimodal_fusion.py::MultiModalFusion` | Generic per-modality tensor fuser (docstring cites audio) — candidate joint for MoT expert fusion |
| Masking | `utils/masking/*` + `layers/attention/common.py::apply_attention_mask` | Canonical masks; never hand-roll `+(1-mask)*(-1e9)` (fp16 `0*-inf=NaN`) — use `ops.where` |
| Heads | `layers/heads/factory.py::create_head(domain,...)` -> `nlp/vision/vlm` | `VQAHead/ImageCaptioningHead/VisualGroundingHead/DetectionHead` patterns for grounding/box/policy heads; no audio/action head exists — new |
| Losses | `losses/{flow_matching_velocity_loss.py::FlowMatchingVelocityLoss,ddpm_hybrid_loss.py::DDPMHybridLoss,clip_contrastive_loss.py,nano_vlm_loss.py,masked_causal_lm_loss.py,multi_task_loss.py}` | Velocity MSE + DDPM hybrid + contrastive + AR CE all present; time-weighting stays in sampler per `FlowMatchingVelocityLoss` docstring; `call->(B,)` shape rule |
| Utils | `utils/{logger.py,keras_registration.py,weight_transfer.py,sequence_length_guard.py,tensors.py,dtype_policy.py,ddpm_schedule.py,activation_serialization.py}` | Central logger, stripped registration, backbone weight transfer, int64 XLA tables, schedule precedent |
| Trainers | `src/train/{dit/train_dit.py,AGENTS.md Pattern 3,common/{callbacks.py,args.py,run_io.py,optimizer.py},common/nlp.py}` | Stock-`fit` discipline, `CLI_TO_CONFIG`, explicit `build()` before fit for EMA, `WeightEMACallback(0.9999)`, timestamped `results/<run>/` |

### 10.3 Gaps (must be new, in `layers/` + registry, not buried in model dir)

Audio encoder, discrete/continuous action tokenizer + embodiment head, unified `(t,h,w,audio,action)` position-id MRoPE with `dt=TPS_base/TPS` modulation, MoT dual-tower routing block with expert-partitioned FFN wiring, VAE encoder/decoder pair at Cosmos compression (closest existing: ideogram4/sd3 VAEs at different ratios).

### 10.4 Tests to mirror

`tests/test_models/test_dit/` (forward/values/config/variants/CFG/build-parity `test_build_materialization_parity.py`/gradient/`mixed_float16`+`float64`/frozen-table/emas) + `test_sd3_mmdit/` + `test_ideogram4/` + `test_losses/test_ddpm_hybrid_loss.py` + `test_package_api_contract.py` + `test_the_legacy_alias_namespace_has_no_collisions.py` + tree-wide `pytest tests/test_models/ -q --collect-only` after public-surface changes. Scoped runs only (`CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg`, no parallel pytest, full suite ~1.5h).

---

## 11. Proposed integration (no code yet)

**Location:** `src/dl_techniques/models/vision_language/cosmos_omni/` (file by input modality; omni consumes image+text streams. Family `__init__` stays docstring-only; leaf binds factory + `__all__` + `README.md`). Registration strips family:

```text
@register_dl_technique("dl_techniques.models.cosmos_omni.mot")
class CosmosOmniMoTBlock / CosmosOmniReasoner / CosmosOmniGenerator ...
```

**File list (phase 1 + stubs):**

```
models/vision_language/cosmos_omni/
  __init__.py        # create_cosmos_omni, CosmosOmniReasoner, __all__
  config.py          # COSMOS_OMNI_VARIANTS (tiny/nano/super/edge rows) + CosmosOmniConfig
  mot_block.py       # dual-tower block: 2x TransformerLayer stacks + joint attention (Eq. 7-8)
  mrope.py           # CosmosMRoPE: 3D split + dt=TPS_base/TPS + margin (new layer, register in layers/embedding/factory.py)
  encoders.py        # ViT-understand wrapper (VisionEncoder) + frozen VAE-generate adapter; audio/action stubs raising NotImplementedError
  reasoner.py        # AR tower: prefix fusion (NanoVLM pattern) + causal stack + vocab head
  generator.py       # DM tower: adaLN + flow-matching head (Ideogram4/SD3 pattern); phase-1 stub
  sampler.py         # Euler sampler + logit-normal t + two-weight CFG stub (pipeline plain-Python object like SD3Pipeline)
  README.md          # usage + variant table + limits (no paper-number claims)
```

Shared schedule/value-object → `utils/cosmos_schedule.py` (DDPMSchedule precedent); velocity loss reuses `losses/flow_matching_velocity_loss.py` (no new loss unless weighting contract differs); trainer `src/train/cosmos_omni/train_cosmos_omni.py` (Pattern 3 + DiT stock-fit discipline).

**Variant tables:** `tiny` (depth 2–4, dim 192–256, heads 4–6, for CPU smoke + `.keras` round-trip) + `nano`/`super`/`edge` rows transcribed from §6 with citations, never from memory. `pretrained=True` → `NotImplementedError` (no weights redistributed; OpenMDW-1.1 + gated guardrail noted in README).

---

## 12. Keras hazards to design around (from GUIDE + v2 guide)

1. Create in `__init__` (named sublayers `block_{i}`), build exactly what `call()` runs; `compute_output_shape` from stored config; `get_config` round-trips all args; constant tables via initializer (never `add_weight(zeros)+.assign()` — StatelessScope discards); never `convert_to_tensor` in `build()`.
2. No `ops.tril/triu`; no `+(1-mask)*(-1e9)`; use `ops.where` + `apply_attention_mask`. No NumPy fancy indexing (gather/flat-row for XLA). No `if` on tensor values; forward `training=` explicitly; no logging in `call()`; `keras.ops` over raw TF; `import keras` qualified.
3. Dual-tower weight paths: flat `List[Layer]` with per-index names, never `List[List[Layer]]` (98% weight-loss precedent). Build-parity test mandatory (explicit `build(shape)` vs lazy `call()`).
4. Dtypes: never-narrow (`max(input,float32)`, velocity head float32); int tables int64; `mixed_float16` + `float64` probes; `isfinite` asserts, non-square fixtures.
5. Registration: never bare `register_keras_serializable`; literal key check; `custom_objects` by `get_registered_name`; new validation must not break old archives (`from_config` substitute+warn); re-derive counts with commands beside them; match Sphinx style of `models/language/bert/model.py`, never wholesale convert.

---

## 13. Phased build plan

**Phase 0 (this doc):** agree leaf `vision_language/cosmos_omni`, phase-1 scope (reasoner + tiny MoT + MRoPE), deferred (audio/action/generator/distill/RL).
**Phase 1:** `CosmosMRoPE` + `dt` modulation (float64 goldens vs Eq. 9 + HF `get_rope_index` vectors); tiny reasoner (prefix fusion + causal stack + CE) over synthetic image+text; joint-attention mask oracles (AR-causal / DM-full, AR-never-reads-DM); build-parity + value round-trip + knob-sensitivity guards, each proven RED first.
**Phase 2:** DM tower (adaLN + velocity head + Euler sampler + logit-normal + CFG) on latent video stubs; T2I/I2V smoke; `FlowMatchingVelocityLoss` wiring.
**Phase 3:** audio VAE adapter + action tokenizer/embodiment heads + forward/inverse/policy modes; DROID-style post-train recipe; distillation/RL via `cosmos-framework` patterns.

## 14. Open decisions for reviewer

1. Leaf `vision_language/cosmos_omni/` vs `general_purpose/` (modality-agnostic claim)?
2. Reuse `Ideogram4MRoPE`/`VideoRoPE3D` by extension vs new `CosmosMRoPE` layer in `layers/embedding/`?
3. Share `TimestepEmbedding`/`AdaLayerNormZero` as-is vs per-patch `(B,L)` time variant?
4. New `CosmosSchedule` in `utils/` vs reuse `DDPMSchedule` + `LogitNormalSchedule`?
5. Reasoner-first (fastest signal, NanoVLM template) vs MoT-block-first (core risk)?

---

## References

* NVIDIA Cosmos Team 2026 (report §§2–6, App. B–E; Eq. 7–9; Fig. 5–6; Table 2 variants).
* `docs.nvidia.com/cosmos/latest/cosmos3/`, `research.nvidia.com/labs/cosmos-lab/cosmos3/`, `developer.nvidia.com/blog/...cosmos-3/`.
* `github.com/NVIDIA/cosmos` (`cookbooks/cosmos3/{generator/{audiovisual,action,transfer},reasoner}`, `evaluation/`, `docs/reference/models.md`), `github.com/NVIDIA/cosmos-framework`, `github.com/nvidia-cosmos/cosmos-cookbook`.
* `huggingface/diffusers/.../transformer_cosmos3.py`, `transformers/.../cosmos3_omni/modeling_cosmos3_omni.py`, `vllm-omni/.../cosmos3/transformer_cosmos3.py`; HF collections `nvidia/cosmos3`, `nvidia/cosmos3-examples`.
* In-tree: `models/{vision_language/{sd3_mmdit,ideogram4,bit_diffusion,clip,nano_vlm},vision/{dit,video_jepa,vae,vq_vae},language/bert/model.py,AGENTS.md,GUIDE.md,README.md}`, `layers/{attention/factory.py,norms/factory.py,ffn/factory.py,embedding/{factory.py,multi_axis_rope.py},transformers/{transformer.py,sd3_adaln.py,adaln_zero.py,ideogram4_block.py},attention/mmdit_joint_attention.py,generative/,heads/factory.py}`, `losses/{flow_matching_velocity_loss,ddpm_hybrid_loss,clip_contrastive_loss}.py`, `utils/{keras_registration.py,ddpm_schedule.py}`, `src/train/{dit,AGENTS.md}`, `REPO_MAP.md`.
